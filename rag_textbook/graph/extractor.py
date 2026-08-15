"""Извлечение сущностей и связей из чанка.

Проблемы прежней реализации и что с ними сделано.

* **13% ответов были ``invalid`` или ``invalid_structure``.** Разбор строился на трёх
  эвристиках подряд: сырой ``json.loads``, вырезание блока в тройных кавычках, поиск
  первого валидного JSON в тексте. Здесь используется строгая схема на стороне сервера
  (``response_format=json_schema``), а разбор — один и с явным статусом.
* **Нулевого кэша.** Повторный прогон заново вызывал модель на тех же чанках.
  Ключ кэша включает хеш текста, модель и версию промпта, поэтому переиндексация
  бесплатна, а смена промпта корректно инвалидирует кэш.
* **Сущности были сырыми токенами.** Теперь имя канонизируется леммами,
  благодаря чему «сингулярных разложений» и «сингулярное разложение» — один узел.
* **Правиловый экстрактор строил клики.** Он остаётся как запасной вариант,
  но выдаёт только сущности; рёбра из него не порождаются.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.config import GraphSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, Entity, ExtractionResult, Relation, content_hash
from rag_textbook.utils.cache import ArtifactCache
from rag_textbook.utils.text import canonicalize_entity, content_terms, truncate

logger = get_logger("graph.extractor")

EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "entities": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"name": {"type": "string"}},
                "required": ["name"],
            },
        },
        "relations": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "source": {"type": "string"},
                    "relation": {"type": "string"},
                    "target": {"type": "string"},
                },
                "required": ["source", "relation", "target"],
            },
        },
    },
    "required": ["entities", "relations"],
}

# Закрытый список меток. Свободный текст в поле relation давал сотни синонимов
# одного отношения, по которым потом невозможно осмысленно обходить граф.
RELATION_LABELS: tuple[str, ...] = (
    "определяется_через",
    "является_частным_случаем",
    "обобщает",
    "используется_в",
    "вычисляется_по",
    "эквивалентно",
    "противопоставляется",
    "требует",
    "свойство",
)

PROMPT_TEMPLATE = """Ты извлекаешь граф знаний из фрагмента учебника по математике.

Извлеки:
1. entities — ключевые математические понятия, методы, объекты. Только термины,
   не общие слова. Не более {max_entities}.
2. relations — связи между извлечёнными сущностями. Не более {max_relations}.

Поле relation выбирай СТРОГО из списка:
{labels}

Правила:
- source и target обязаны присутствовать в списке entities;
- не выдумывай связи, которых нет в тексте;
- если связей нет, верни пустой список relations.

Фрагмент:
{text}"""


def _build_prompt(text: str, settings: GraphSettings) -> str:
    return PROMPT_TEMPLATE.format(
        max_entities=settings.max_entities_per_chunk,
        max_relations=settings.max_relations_per_chunk,
        labels="\n".join(f"- {label}" for label in RELATION_LABELS),
        text=truncate(text, settings.extraction_max_chars),
    )


def _normalize_label(raw: str) -> str:
    value = str(raw or "").strip().lower().replace(" ", "_")
    if value in RELATION_LABELS:
        return value
    # Мягкое приведение синонимов к ближайшей метке из списка.
    for label in RELATION_LABELS:
        if value and (value in label or label in value):
            return label
    return "используется_в"


def _strip_code_fence(raw: str) -> str:
    """Снимает обёртку ```json ... ``` вокруг ответа.

    Строгий структурированный вывод её не создаёт, но движки поддерживают
    ``response_format`` по-разному, и на части из них модель возвращает JSON
    внутри блока кода. Это единственная допущенная здесь эвристика: снимается
    ровно обёртка целиком, содержимое не трогается. Разбирать ответ несколькими
    догадками подряд, как делала прежняя версия, мы не возвращаемся.
    """
    text = str(raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


class EntityExtractor:
    def __init__(
        self,
        settings: GraphSettings,
        llm: LLMClient | None = None,
        cache: ArtifactCache | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.cache = cache

    # ------------------------------------------------------------------- ключи

    def _llm_variant(self) -> str:
        """Параметры вызова модели, влияющие на результат.

        Глубина размышления меняет ответ радикально — вплоть до пустого, — и
        обязана входить в ключ кэша. Иначе после смены параметра мы вычитываем
        из кэша результаты, полученные при старом, и правка выглядит
        не подействовавшей.
        """
        llm_settings = getattr(self.llm, "settings", None)
        if llm_settings is None:
            return ""
        return str(getattr(llm_settings, "reasoning_effort", ""))

    def _cache_key(self, chunk: Chunk, model: str) -> str:
        return content_hash(
            chunk.text_hash or content_hash(chunk.text),
            model,
            self._llm_variant(),
            self.settings.extraction_prompt_version,
            str(self.settings.max_entities_per_chunk),
            str(self.settings.max_relations_per_chunk),
        )

    # ------------------------------------------------------------- нормализация

    def _make_entity(self, raw_name: str) -> Entity | None:
        name = str(raw_name or "").strip()
        if not name:
            return None
        canonical = canonicalize_entity(name, lemmatize=self.settings.lemmatize_entities)
        if not canonical or len(canonical) < self.settings.min_entity_length:
            return None
        return Entity(
            id=Entity.make_id(canonical),
            name=name[:96],
            canonical=canonical,
            aliases=[name.lower()] if name.lower() != canonical else [],
            count=1,
        )

    def _parse_payload(self, payload: dict[str, Any], chunk: Chunk) -> ExtractionResult:
        entities_raw = payload.get("entities")
        relations_raw = payload.get("relations")
        if not isinstance(entities_raw, list):
            entities_raw = []
        if not isinstance(relations_raw, list):
            relations_raw = []

        by_canonical: dict[str, Entity] = {}
        alias_to_canonical: dict[str, str] = {}
        for item in entities_raw[: self.settings.max_entities_per_chunk * 2]:
            name = item.get("name") if isinstance(item, dict) else item
            entity = self._make_entity(str(name or ""))
            if entity is None:
                continue
            if entity.canonical in by_canonical:
                by_canonical[entity.canonical].count += 1
            else:
                by_canonical[entity.canonical] = entity
            alias_to_canonical[str(name).strip().lower()] = entity.canonical
            if len(by_canonical) >= self.settings.max_entities_per_chunk:
                break

        relations: list[Relation] = []
        seen: set[tuple[str, str, str]] = set()
        for item in relations_raw:
            if not isinstance(item, dict):
                continue
            source_raw = str(item.get("source") or "").strip()
            target_raw = str(item.get("target") or "").strip()
            if not source_raw or not target_raw:
                continue
            source_canonical = alias_to_canonical.get(
                source_raw.lower(),
                canonicalize_entity(source_raw, lemmatize=self.settings.lemmatize_entities),
            )
            target_canonical = alias_to_canonical.get(
                target_raw.lower(),
                canonicalize_entity(target_raw, lemmatize=self.settings.lemmatize_entities),
            )
            # Связь имеет смысл только между известными узлами: иначе граф
            # заполняется висячими сущностями, которых нет ни в одном пассаже.
            if source_canonical not in by_canonical or target_canonical not in by_canonical:
                continue
            if source_canonical == target_canonical:
                continue
            label = _normalize_label(str(item.get("relation") or ""))
            key = (source_canonical, target_canonical, label)
            if key in seen:
                continue
            seen.add(key)
            relations.append(
                Relation(
                    source_id=Entity.make_id(source_canonical),
                    target_id=Entity.make_id(target_canonical),
                    label=label,
                    chunk_id=chunk.id,
                    doc_id=chunk.doc_id,
                    weight=1.0,
                )
            )
            if len(relations) >= self.settings.max_relations_per_chunk:
                break

        status = "ok" if (by_canonical or relations) else "empty"
        return ExtractionResult(
            entities=list(by_canonical.values()), relations=relations, status=status
        )

    # ---------------------------------------------------------------- стратегии

    def extract_rule_based(self, chunk: Chunk) -> ExtractionResult:
        """Запасной путь без LLM.

        Отдаёт только сущности: строить рёбра по совместной встречаемости здесь
        нельзя — именно так и получался граф из клик частотных слов.
        """
        terms = content_terms(
            chunk.text,
            min_length=self.settings.min_entity_length,
            lemmatize=self.settings.lemmatize_entities,
            limit=self.settings.max_entities_per_chunk,
        )
        entities = [
            Entity(id=Entity.make_id(term), name=term, canonical=term, count=1) for term in terms
        ]
        return ExtractionResult(entities=entities, relations=[], status="rule")

    def extract_llm(self, chunk: Chunk) -> ExtractionResult:
        if self.llm is None:
            return ExtractionResult(status="no_llm")
        prompt = _build_prompt(chunk.text, self.settings)
        try:
            raw = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="extraction",
                json_schema=EXTRACTION_SCHEMA,
                temperature=0.0,
                max_tokens=768,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Экстракция не удалась для %s: %s", chunk.id, exc)
            return ExtractionResult(status="error", raw_preview=str(exc)[:200])

        if not str(raw or "").strip():
            # Пустой ответ — это, как правило, исчерпанный лимит токенов:
            # рассуждающая модель потратила его на размышление. Отдельный
            # статус нужен, чтобы такое не путалось с невалидным JSON.
            return ExtractionResult(status="empty_response")

        try:
            payload = json.loads(_strip_code_fence(raw))
        except json.JSONDecodeError:
            return ExtractionResult(status="invalid_json", raw_preview=str(raw)[:200])
        if not isinstance(payload, dict):
            return ExtractionResult(status="invalid_structure", raw_preview=str(raw)[:200])
        return self._parse_payload(payload, chunk)

    # ---------------------------------------------------------------- публично

    def extract(self, chunk: Chunk, model_name: str = "") -> ExtractionResult:
        key = self._cache_key(chunk, model_name)
        if self.cache is not None and self.settings.extraction_cache_enabled:
            cached = self.cache.get(key)
            if isinstance(cached, dict):
                try:
                    return ExtractionResult.model_validate(cached)
                except Exception:  # noqa: BLE001
                    pass

        if self.settings.extractor == "rule" or self.llm is None:
            result = self.extract_rule_based(chunk)
        else:
            result = self.extract_llm(chunk)
            if result.status in {
                "error",
                "invalid_json",
                "invalid_structure",
                "empty_response",
                "no_llm",
            }:
                # Падение экстрактора не должно оставлять пассаж вне графа.
                result = self.extract_rule_based(chunk)
                result.status = "rule_fallback"

        # Результат отката НЕ кэшируется. Кэш существует, чтобы не повторять
        # дорогую удачную работу; запоминание неудачи превращает разовый сбой
        # модели в постоянный: следующий прогон возьмёт из кэша пустой граф и
        # даже не попробует ещё раз. Именно так пустое извлечение пережило бы
        # исправление, если бы мы не чистили кэш вручную.
        cacheable = result.status != "rule_fallback"
        if self.cache is not None and self.settings.extraction_cache_enabled and cacheable:
            self.cache.set(key, result.model_dump())
        return result

    def extract_many(self, chunks: Sequence[Chunk], model_name: str = "") -> list[ExtractionResult]:
        return [self.extract(chunk, model_name) for chunk in chunks]
