"""Сборка эталонного набора вопросов.

Ключевая идея: генерировать **от чанка к вопросу**, а не наоборот. Тогда
идентификатор эталонного фрагмента известен по построению, и Recall@k считается
точно. В прежнем наборе эталонных фрагментов не было вовсе — указывалось лишь
имя файла-источника, поэтому измерить качество поиска было нельзя в принципе.

Дополнительно набор строится стратифицированно, с сохранением типов вопросов
из прежней разметки: ``single_chunk``, ``multi_hop``, ``relation``, ``formula_table``.
Многохоповые вопросы собираются из пар чанков, связанных общими сущностями, —
именно на них должен проявляться выигрыш графа.
"""

from __future__ import annotations

import json
import random
import re
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, GoldQuestion, content_hash
from rag_textbook.utils.text import content_terms, truncate

logger = get_logger("evaluation.goldset")

SINGLE_PROMPT = """На основе фрагмента учебника составь ОДИН вопрос, ответ на который
содержится только в этом фрагменте.

Требования:
- вопрос должен быть конкретным и проверяемым;
- нельзя использовать формулировки «в тексте», «в данном фрагменте», «согласно отрывку»;
- вопрос должен быть понятен человеку, который не видел фрагмент;
- если во фрагменте есть формула или таблица, спроси о её смысле или содержании.

Верни строго JSON: {{"question": "...", "answer": "..."}}

Фрагмент:
{text}"""

MULTIHOP_PROMPT = """Ниже два фрагмента одного учебника, связанные общими понятиями.
Составь ОДИН вопрос, для ответа на который нужны ОБА фрагмента.

Требования:
- ответ не должен полностью содержаться ни в одном фрагменте по отдельности;
- вопрос должен звучать естественно, без ссылок на «фрагменты»;
- используй связь между понятиями, а не простое перечисление.

Верни строго JSON: {{"question": "...", "answer": "..."}}

Фрагмент А:
{text_a}

Фрагмент Б:
{text_b}"""

QUESTION_SCHEMA = {
    "type": "object",
    "properties": {"question": {"type": "string"}, "answer": {"type": "string"}},
    "required": ["question", "answer"],
}

# Формулировки, выдающие вопрос, привязанный к тексту, а не к предмету.
_LEAKY_PATTERNS = (
    "в тексте",
    "в фрагмент",
    "в данном отрывке",
    "согласно отрывку",
    "в приведённом",
    "в приведенном",
    "автор пишет",
    "上文",
    # Ссылки на конкретную иллюстрацию или таблицу без её номера. Такой вопрос
    # осмыслен только рядом с исходным фрагментом: «какие значения x
    # соответствуют минимуму функции, изображённой на графике» невозможно
    # адресовать поиску, потому что графиков в учебнике сотни.
    "на графике",
    "на рисунке",
    "на изображении",
    "на диаграмме",
    "на иллюстрации",
    "в таблице выше",
    "на схеме",
    "изображённой",
    "изображенной",
    "показанной на",
    "представленной на",
)


def looks_leaky(question: str) -> bool:
    lowered = (question or "").lower()
    return any(pattern in lowered for pattern in _LEAKY_PATTERNS)


class GoldsetBuilder:
    def __init__(
        self,
        llm: LLMClient,
        seed: int = 20260814,
        graph_store: Any = None,
        max_lexical_overlap: float = 0.12,
        min_ordinal_distance: int = 10,
    ) -> None:
        self.llm = llm
        self.random = random.Random(seed)
        # Хранилище графа нужно только для отбора пар, связанных структурно,
        # а не лексически. Без него сборка работает как раньше.
        self.graph_store = graph_store
        self.max_lexical_overlap = max_lexical_overlap
        self.min_ordinal_distance = min_ordinal_distance
        # Счётчик исходов каждого обращения к модели: без него отказ выглядит
        # как «просто ничего не сгенерировалось».
        self.failures: Counter[str] = Counter()

    # ------------------------------------------------------------ примитивы

    def _ask(self, prompt: str) -> tuple[str, str] | None:
        """Возвращает пару «вопрос, ответ» либо None, зафиксировав причину отказа.

        Причина обязательна: сборка из 140 фрагментов, вернувшая ноль вопросов
        без единой записи в журнале, стоила прогона впустую и не дала понять,
        сломан ли промпт, разбор ответа или сама модель.
        """
        try:
            raw = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                # Служебный вызов, а не ответ пользователю: с назначением `chat`
                # рассуждающая модель тратит все 400 токенов на размышление
                # и возвращает пустой content.
                purpose="utility",
                json_schema=QUESTION_SCHEMA,
                temperature=0.3,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            self.failures["llm_error"] += 1
            logger.warning("Генерация вопроса не удалась: %s", exc)
            return None
        if not str(raw or "").strip():
            self.failures["empty_response"] += 1
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if not match:
                self.failures["not_json"] += 1
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                self.failures["not_json"] += 1
                return None
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if not question:
            self.failures["no_question_field"] += 1
            return None
        self.failures["ok"] += 1
        return question, answer

    @staticmethod
    def _classify(chunk: Chunk) -> str:
        if chunk.has_formula or chunk.has_table:
            return "formula_table"
        return "single_chunk"

    # -------------------------------------------------------------- отборы

    def _select_single(self, chunks: Sequence[Chunk], count: int) -> list[Chunk]:
        """Отбирает содержательные чанки.

        Слишком короткие фрагменты и оглавления дают бессмысленные вопросы,
        поэтому фильтруем по длине и по числу содержательных терминов.
        """
        candidates = [
            chunk
            for chunk in chunks
            if len(chunk.text) >= 400 and len(content_terms(chunk.text, limit=40)) >= 12
        ]
        if not candidates:
            candidates = list(chunks)
        self.random.shuffle(candidates)

        # Стратификация: половину набора отдаём фрагментам с формулами и таблицами,
        # раз уж работа с ними заявлена как ключевая функция продукта.
        special = [chunk for chunk in candidates if chunk.has_formula or chunk.has_table]
        plain = [chunk for chunk in candidates if not (chunk.has_formula or chunk.has_table)]
        target_special = min(len(special), count // 2)
        selected = special[:target_special] + plain[: count - target_special]
        return selected[:count]

    def _select_graph_linked_pairs(
        self, chunks: Sequence[Chunk], count: int
    ) -> list[tuple[Chunk, Chunk]]:
        """Ищет пары фрагментов, связанные **в графе** и непохожие по словам.

        Зачем понадобился отдельный отбор. Обычный отбор берёт пары с общими
        содержательными терминами, то есть лексически похожие. На таких парах
        и векторный, и лексический канал справляются сами, и вопрос о пользе
        графа на них принципиально неразрешим: измеритель по построению слеп
        к тому, что должен измерять.

        Здесь наоборот: пара берётся, если фрагменты соединены типизированной
        связью графа, но разделяют мало слов. Ровно на таких вопросах граф
        обязан выигрывать у лексического поиска, если он вообще нужен.

        Возвращает пустой список, если граф недоступен или подходящих пар нет —
        тогда работает обычный отбор.
        """
        if self.graph_store is None:
            return []
        by_id = {chunk.id: chunk for chunk in chunks}
        try:
            rows = self.graph_store.linked_passage_pairs(
                limit=max(count * 20, count),
                min_distance=self.min_ordinal_distance,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Не удалось получить связанные пары из графа: %s", exc)
            return []

        terms_by_chunk: dict[str, set[str]] = {}

        def terms(chunk_id: str) -> set[str]:
            if chunk_id not in terms_by_chunk:
                chunk = by_id.get(chunk_id)
                terms_by_chunk[chunk_id] = (
                    set(content_terms(chunk.text, limit=60)) if chunk else set()
                )
            return terms_by_chunk[chunk_id]

        scored: list[tuple[float, Chunk, Chunk]] = []
        for row in rows:
            left_id, right_id = str(row.get("left") or ""), str(row.get("right") or "")
            left, right = by_id.get(left_id), by_id.get(right_id)
            if left is None or right is None:
                continue
            left_terms, right_terms = terms(left_id), terms(right_id)
            union = left_terms | right_terms
            if not union:
                continue
            overlap = len(left_terms & right_terms) / len(union)
            # Чем меньше словесного пересечения, тем ценнее пара для проверки:
            # такой связи неоткуда взяться, кроме как из графа.
            if overlap > self.max_lexical_overlap:
                continue
            scored.append((overlap, left, right))

        scored.sort(key=lambda item: item[0])
        selected = [(left, right) for _, left, right in scored[: max(count * 4, count)]]
        self.random.shuffle(selected)
        logger.info(
            "Пары, связанные графом: найдено %s, отобрано %s (порог пересечения %.2f)",
            len(rows),
            min(len(selected), count),
            self.max_lexical_overlap,
        )
        return selected[:count]

    def _select_multihop_pairs(
        self, chunks: Sequence[Chunk], count: int
    ) -> list[tuple[Chunk, Chunk]]:
        """Ищет пары чанков с общими понятиями, но из разных мест документа.

        Требование «разные места» существенно: соседние чанки перекрываются,
        и вопрос по ним не будет многохоповым по-настоящему.
        """
        terms_by_chunk = {chunk.id: set(content_terms(chunk.text, limit=60)) for chunk in chunks}
        by_id = {chunk.id: chunk for chunk in chunks}
        pairs: list[tuple[Chunk, Chunk, int]] = []

        chunk_list = list(chunks)
        for index, left in enumerate(chunk_list):
            for right in chunk_list[index + 1 :]:
                if left.doc_id != right.doc_id:
                    continue
                # Расстояние в чанках: соседи не подходят.
                if abs(left.ordinal - right.ordinal) < 10:
                    continue
                shared = terms_by_chunk[left.id] & terms_by_chunk[right.id]
                if len(shared) < 3:
                    continue
                pairs.append((by_id[left.id], by_id[right.id], len(shared)))

        pairs.sort(key=lambda item: item[2], reverse=True)
        # Из верхушки берём случайную выборку, иначе все пары будут про одну тему.
        top = pairs[: max(count * 8, count)]
        self.random.shuffle(top)
        return [(left, right) for left, right, _ in top[:count]]

    # ---------------------------------------------------------------- сборка

    def build(
        self,
        chunks: Sequence[Chunk],
        *,
        single_count: int = 100,
        multihop_count: int = 50,
        max_chars: int = 2500,
    ) -> list[GoldQuestion]:
        questions: list[GoldQuestion] = []

        selected = self._select_single(chunks, single_count)
        logger.info("Генерация одношаговых вопросов: %s фрагментов", len(selected))
        for chunk in selected:
            produced = self._ask(SINGLE_PROMPT.format(text=truncate(chunk.text, max_chars)))
            if produced is None:
                continue
            question, answer = produced
            if looks_leaky(question):
                self.failures["leaky"] += 1
                logger.debug("Вопрос отброшен как привязанный к тексту: %s", question[:60])
                continue
            questions.append(
                GoldQuestion(
                    id=content_hash("single", chunk.id, question)[:16],
                    question=question,
                    answer=answer,
                    gold_chunk_ids=[chunk.id],
                    gold_doc_ids=[chunk.doc_id],
                    question_type=self._classify(chunk),
                    expected_hops=1,
                )
            )

        # Сначала пары, связанные структурно: только на них видно, даёт ли граф
        # что-то сверх лексического поиска. Лексически связанные пары добирают
        # остаток, чтобы набор не выродился в одну узкую категорию.
        pairs = self._select_graph_linked_pairs(chunks, multihop_count)
        graph_linked = {(left.id, right.id) for left, right in pairs}
        if len(pairs) < multihop_count:
            for pair in self._select_multihop_pairs(chunks, multihop_count - len(pairs)):
                if (pair[0].id, pair[1].id) not in graph_linked:
                    pairs.append(pair)
        logger.info(
            "Генерация многошаговых вопросов: %s пар, из них связанных графом %s",
            len(pairs),
            len(graph_linked),
        )
        for left, right in pairs:
            produced = self._ask(
                MULTIHOP_PROMPT.format(
                    text_a=truncate(left.text, max_chars // 2),
                    text_b=truncate(right.text, max_chars // 2),
                )
            )
            if produced is None:
                continue
            question, answer = produced
            if looks_leaky(question):
                self.failures["leaky"] += 1
                continue
            questions.append(
                GoldQuestion(
                    id=content_hash("multi", left.id, right.id, question)[:16],
                    question=question,
                    answer=answer,
                    gold_chunk_ids=[left.id, right.id],
                    gold_doc_ids=sorted({left.doc_id, right.doc_id}),
                    # Отдельный тип, чтобы вклад графа считался именно там,
                    # где лексический поиск заведомо не справляется.
                    question_type=(
                        "graph_linked" if (left.id, right.id) in graph_linked else "multi_hop"
                    ),
                    expected_hops=2,
                )
            )

        logger.info(
            "Эталонный набор собран: всего=%s (одношаговых=%s, многошаговых=%s), исходы=%s",
            len(questions),
            sum(1 for q in questions if q.expected_hops == 1),
            sum(1 for q in questions if q.expected_hops > 1),
            dict(self.failures),
        )
        if not questions:
            logger.error(
                "Не сгенерировано ни одного вопроса. Разбор исходов: %s. "
                "Пустые ответы модели означают, что размышление съедает лимит "
                "токенов — проверьте LLM_REASONING_EFFORT.",
                dict(self.failures),
            )
        return questions


def merge_goldsets(
    existing: Sequence[GoldQuestion], added: Sequence[GoldQuestion]
) -> tuple[list[GoldQuestion], int]:
    """Дописывает новые вопросы к набору, не трогая прежние.

    Расширять набор пересборкой нельзя: сборка перезаписывает файл целиком,
    и все прежние прогоны становятся несравнимыми — не с чем сопоставлять
    даже точку отсчёта. Здесь прежние вопросы сохраняются как есть, включая
    отметку ``verified``, поставленную вручную.

    Совпадения ищутся и по идентификатору, и по тексту вопроса: идентификатор
    считается от текста и фрагмента, поэтому один и тот же вопрос, полученный
    от другого фрагмента, дал бы новый идентификатор и попал в набор дважды.
    """
    merged = list(existing)
    seen_ids = {question.id for question in merged}
    seen_text = {" ".join(question.question.lower().split()) for question in merged}

    appended = 0
    for question in added:
        text = " ".join(question.question.lower().split())
        if question.id in seen_ids or text in seen_text:
            continue
        merged.append(question)
        seen_ids.add(question.id)
        seen_text.add(text)
        appended += 1

    logger.info(
        "Набор дополнен: было %s, добавлено %s, повторов отброшено %s",
        len(existing),
        appended,
        len(added) - appended,
    )
    return merged, appended


def save_goldset(questions: Sequence[GoldQuestion], path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "count": len(questions),
        "questions": [question.model_dump() for question in questions],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Эталонный набор сохранён: %s (%s вопросов)", path, len(questions))
    return path


def load_goldset(path: Path) -> list[GoldQuestion]:
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Эталонный набор не найден: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    items = payload.get("questions") if isinstance(payload, dict) else payload
    return [GoldQuestion.model_validate(item) for item in (items or [])]
