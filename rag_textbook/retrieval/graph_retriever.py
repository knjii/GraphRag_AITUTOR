"""Графовый канал поиска.

Устройство канала: термины вопроса → канонические формы → full-text поиск
стартовых сущностей → расширение на один хоп по типизированным ``RELATES`` →
взвешенное ранжирование пассажей.

Отличия от прежней версии, каждое соответствует конкретной поломке:

* термины запроса лемматизируются, поэтому «сингулярных разложений» находит узел
  «сингулярное разложение» — раньше точное совпадение почти всегда промахивалось,
  а fallback через ``CONTAINS`` тащил случайные подстроки;
* поиск идёт по full-text индексу, а не полным сканом по метке ``Entity``;
* расширение не ходит по ``CO_OCCURS``: на плотном графе это давало почти весь граф;
* вклад сущности взвешен близостью к запросу, а ранжирование нормировано на
  насыщенность пассажа терминами, вместо ``count(*)``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_textbook.config import GraphSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.stores.graph_store import GraphStore
from rag_textbook.utils.text import canonicalize_entity, content_terms

logger = get_logger("retrieval.graph")


class GraphRetriever:
    def __init__(self, settings: GraphSettings, store: GraphStore) -> None:
        self.settings = settings
        self.store = store

    def _query_terms(self, question: str) -> list[str]:
        """Кандидаты для поиска сущностей: леммы и биграммы лемм."""
        unigrams = content_terms(
            question,
            min_length=self.settings.min_entity_length,
            lemmatize=self.settings.lemmatize_entities,
            limit=16,
        )
        bigrams: list[str] = []
        for left, right in zip(unigrams, unigrams[1:], strict=False):
            phrase = canonicalize_entity(f"{left} {right}", lemmatize=False)
            if phrase and phrase not in bigrams:
                bigrams.append(phrase)
        # Биграммы идут первыми: составной термин точнее одиночного слова.
        return (bigrams + unigrams)[:24]

    def retrieve(
        self,
        question: str,
        limit: int | None = None,
        seed_chunk_ids: Sequence[str] | None = None,
    ) -> list[ScoredChunk]:
        """Возвращает фрагменты графового канала.

        ``seed_chunk_ids`` — опорные фрагменты для режимов ``passages``
        и ``both``: обход начинается от их сущностей, а не от терминов вопроса.
        """
        if not self.settings.retrieval_enabled:
            return []

        mode = self.settings.seed_mode
        weights: dict[str, float] = {}
        exclude: set[str] = set()

        if mode in ("query", "both"):
            weights.update(self._weights_from_query(question))
        if mode in ("passages", "both"):
            seeds = list(seed_chunk_ids or [])[: self.settings.seed_passages]
            exclude = set(seeds)
            weights.update(self._weights_from_passages(seeds))

        if not weights:
            logger.debug("Графовый канал: стартовые сущности не найдены (режим %s)", mode)
            return []

        try:
            weights = self._expand(weights)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Расширение графа не удалось: %s", exc)

        try:
            # Запрашиваем с запасом: опорные фрагменты из выдачи исключаются,
            # иначе канал вернёт то, что уже найдено векторным поиском.
            requested = (limit or self.settings.passage_limit) + len(exclude)
            rows = self.store.find_passages(
                weights, requested, use_idf=self.settings.passage_idf_enabled
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Поиск пассажей в графе не удался: %s", exc)
            return []

        results: list[ScoredChunk] = []
        for row in rows:
            if str(row.get("chunk_id") or "") in exclude:
                continue
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            chunk = Chunk(
                id=str(row.get("chunk_id") or ""),
                doc_id=str(row.get("doc_id") or ""),
                doc_name=str(row.get("doc_name") or ""),
                source_path="",
                ordinal=int(row.get("ordinal") or 0),
                text=text,
                pages=[int(page) for page in (row.get("pages") or [])],
            )
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(row.get("score") or 0.0),
                    channels=["graph_entity"],
                    channel_scores={"graph_entity": float(row.get("score") or 0.0)},
                    matched_entities=[str(name) for name in (row.get("matched_entities") or [])],
                )
            )

        logger.debug(
            "Графовый канал (%s): сущностей %s, пассажей %s",
            mode,
            len(weights),
            len(results),
        )
        return results

    # ------------------------------------------------------------- источники

    def _weights_from_query(self, question: str) -> dict[str, float]:
        """Стартовые сущности по терминам вопроса."""
        terms = self._query_terms(question)
        if not terms:
            return {}
        try:
            seeds = self.store.find_seed_entities(terms, self.settings.seed_entity_limit)
        except Exception as exc:  # noqa: BLE001
            # Графовый канал — дополнение. Его недоступность не должна ронять запрос.
            logger.warning("Поиск стартовых сущностей не удался: %s", exc)
            return {}
        return _normalized(seeds, "score")

    def _weights_from_passages(self, chunk_ids: Sequence[str]) -> dict[str, float]:
        """Стартовые сущности по уже найденным фрагментам."""
        if not chunk_ids:
            return {}
        try:
            seeds = self.store.entities_of_passages(chunk_ids, self.settings.seed_entity_limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Сущности опорных фрагментов не получены: %s", exc)
            return {}
        return _normalized(seeds, "weight")

    def _expand(self, weights: dict[str, float]) -> dict[str, float]:
        """Расширяет множество сущностей, сохраняя веса стартовых."""
        if self.settings.hop_decay <= 0:
            # Расширение выключено намеренно: проверено, что так хуже,
            # но настройка нужна для A/B без правки кода.
            return weights
        expanded = self.store.expand_entities(
            list(weights),
            hops=self.settings.expansion_hops,
            rel_types=list(self.settings.expansion_rel_types),
            limit=self.settings.seed_entity_limit * 4,
            decay=self.settings.hop_decay,
        )
        # Стартовые сущности не должны терять свой вес: расширение
        # проставляет им единицу независимо от исходной релевантности.
        expanded.update(weights)
        return expanded


def _normalized(rows: Sequence[dict], key: str) -> dict[str, float]:
    """Приводит веса стартовых сущностей к отрезку от нуля до единицы.

    Нормировка обязательна: полнотекстовый поиск и подсчёт по упоминаниям
    живут в разных шкалах, и в режиме ``both`` без неё один источник
    полностью подавил бы другой.
    """
    values = {
        str(row.get("id") or ""): float(row.get(key) or 0.0) for row in rows if row.get("id")
    }
    top = max(values.values(), default=0.0)
    if top <= 0:
        return dict.fromkeys(values, 1.0)
    return {entity_id: value / top for entity_id, value in values.items()}
