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
from collections.abc import Sequence
from pathlib import Path

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
)


def looks_leaky(question: str) -> bool:
    lowered = (question or "").lower()
    return any(pattern in lowered for pattern in _LEAKY_PATTERNS)


class GoldsetBuilder:
    def __init__(self, llm: LLMClient, seed: int = 20260814) -> None:
        self.llm = llm
        self.random = random.Random(seed)

    # ------------------------------------------------------------ примитивы

    def _ask(self, prompt: str) -> tuple[str, str] | None:
        try:
            raw = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="chat",
                json_schema=QUESTION_SCHEMA,
                temperature=0.3,
                max_tokens=400,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Генерация вопроса не удалась: %s", exc)
            return None
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", raw or "", re.DOTALL)
            if not match:
                return None
            try:
                payload = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        question = str(payload.get("question") or "").strip()
        answer = str(payload.get("answer") or "").strip()
        if not question:
            return None
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

        pairs = self._select_multihop_pairs(chunks, multihop_count)
        logger.info("Генерация многошаговых вопросов: %s пар", len(pairs))
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
                continue
            questions.append(
                GoldQuestion(
                    id=content_hash("multi", left.id, right.id, question)[:16],
                    question=question,
                    answer=answer,
                    gold_chunk_ids=[left.id, right.id],
                    gold_doc_ids=sorted({left.doc_id, right.doc_id}),
                    question_type="multi_hop",
                    expected_hops=2,
                )
            )

        logger.info(
            "Эталонный набор собран: всего=%s (одношаговых=%s, многошаговых=%s)",
            len(questions),
            sum(1 for q in questions if q.expected_hops == 1),
            sum(1 for q in questions if q.expected_hops > 1),
        )
        return questions


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
