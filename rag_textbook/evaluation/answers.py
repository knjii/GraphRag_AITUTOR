"""Оценка ответов, а не только поиска.

Все прежние метрики — про поиск: дошёл ли нужный фрагмент до контекста.
Для продукта это промежуточный показатель. Учащемуся достаётся ответ, и
ответ может быть неверным при идеальном поиске: модель способна переврать
формулу, дописать несуществующее условие или уверенно ответить по контексту,
в котором ответа нет.

Здесь считаются четыре величины, и они намеренно разной природы.

**Две объективные, без участия модели-судьи.** Их можно предъявлять как есть.

``latex_recall``     доля формул эталонного фрагмента, дословно дошедших
                     до ответа. Для учебника математики это главный признак
                     сохранности: формула либо совпадает, либо нет.
``unsupported``      доля содержательных четвёрок слов ответа, которых нет
                     в поданном контексте. Грубый, но независимый признак
                     выдумки: если модель сочиняет, доля растёт.

**Две судейские, с оговоркой.** Судьёй работает та же модель, что и отвечает,
потому что другой на арендованной карте нет. Модель склонна одобрять
собственные ответы, поэтому судейские оценки годятся для **сравнения
конфигураций между собой** и не годятся как абсолютная оценка качества.
Ровно та же оговорка, что и у эталонного набора, сгенерированного моделью.

``correctness``      отвечает ли ответ на вопрос по существу;
``groundedness``     следует ли ответ из поданного контекста.

Отдельно считается доля отказов: ответ «в контексте нет данных» — это удача,
когда контекст и правда пуст, и провал, когда материал был подан.
"""

from __future__ import annotations

import json
import re
import statistics
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Answer, GoldQuestion
from rag_textbook.utils.text import content_terms, extract_latex_fragments, truncate

logger = get_logger("evaluation.answers")

# Судейская шкала намеренно короткая. Просить у модели 4B оценку по десяти
# делениям — значит получить шум с видом точности: воспроизводимость такой
# оценки ниже, чем расстояние между её делениями.
JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "correctness": {"type": "integer", "minimum": 0, "maximum": 2},
        "groundedness": {"type": "integer", "minimum": 0, "maximum": 2},
        "reason": {"type": "string"},
    },
    "required": ["correctness", "groundedness"],
}

JUDGE_PROMPT = """Ты проверяешь ответ учебного помощника по математике.

ВОПРОС:
{question}

ОТВЕТ ПОМОЩНИКА:
{answer}

ФРАГМЕНТЫ, ПОДАННЫЕ ПОМОЩНИКУ:
{context}

ЭТАЛОННЫЙ ОТВЕТ (может быть неточным, это лишь ориентир):
{reference}

Оцени двумя числами.

correctness — отвечает ли ответ на заданный вопрос по существу:
  0 — не отвечает или отвечает неверно;
  1 — отвечает частично либо с погрешностью;
  2 — отвечает верно.

groundedness — следует ли ответ из поданных фрагментов:
  0 — содержит утверждения, которых во фрагментах нет;
  1 — в основном следует, но есть добавленное от себя;
  2 — целиком следует из фрагментов.

Отказ «в контексте нет данных» при непустых фрагментах — это correctness 0.

Верни строго JSON: {{"correctness": 0, "groundedness": 0, "reason": "коротко"}}
"""

# Признаки отказа отвечать. Список короткий намеренно: расширять его —
# значит подгонять метрику под формулировки конкретной модели.
REFUSAL_MARKERS = (
    "не содержится",
    "нет данных",
    "недостаточно информации",
    "не удалось найти",
)


@dataclass
class AnswerOutcome:
    """Результат по одному вопросу."""

    question_id: str
    question_type: str
    answer: str = ""
    refused: bool = False
    context_size: int = 0
    # Объективные признаки
    latex_expected: int = 0
    latex_found: int = 0
    unsupported: float = 0.0
    # Судейские
    correctness: int | None = None
    groundedness: int | None = None
    judge_reason: str = ""
    latency_ms: float = 0.0

    @property
    def latex_recall(self) -> float | None:
        if not self.latex_expected:
            return None
        return self.latex_found / self.latex_expected

    def as_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["latex_recall"] = self.latex_recall
        return payload


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _normalize_latex(fragment: str) -> str:
    """Убирает то, что не меняет смысла формулы.

    Пробелы внутри LaTeX расставляются как придётся — и парсером, и моделью, —
    поэтому дословное сравнение без нормализации занижало бы совпадение
    до нуля почти всегда.
    """
    return re.sub(r"\s+", "", fragment or "")


def latex_overlap(reference_text: str, answer: str) -> tuple[int, int]:
    """Сколько формул эталона дошло до ответа.

    Возвращает пару «сколько ожидалось, сколько нашлось».
    """
    expected = {
        _normalize_latex(item)
        for item in extract_latex_fragments(reference_text)
        # Однобуквенные обозначения не считаем: их совпадение случайно.
        if len(_normalize_latex(item)) >= 12
    }
    if not expected:
        return 0, 0
    answer_normalized = _normalize_latex(answer)
    found = sum(1 for item in expected if item and item in answer_normalized)
    return len(expected), found


def unsupported_share(answer: str, context: str, window: int = 4) -> float:
    """Доля содержательных четвёрок слов ответа, которых нет в контексте.

    Признак грубый: перефразирование он засчитает как выдумку. Зато он
    не зависит ни от модели-судьи, ни от языка, и потому пригоден для
    сравнения прогонов между собой — а именно этого от него и нужно.
    """
    answer_terms = content_terms(answer)
    if len(answer_terms) < window:
        return 0.0
    context_terms = content_terms(context)
    context_grams = {
        tuple(context_terms[index : index + window])
        for index in range(max(len(context_terms) - window + 1, 0))
    }
    if not context_grams:
        return 1.0
    total = 0
    missing = 0
    for index in range(len(answer_terms) - window + 1):
        total += 1
        if tuple(answer_terms[index : index + window]) not in context_grams:
            missing += 1
    return missing / total if total else 0.0


def judge_answer(
    llm: LLMClient,
    *,
    question: str,
    answer: str,
    context: str,
    reference: str,
    max_context_chars: int = 6000,
) -> dict[str, Any]:
    """Судейская оценка одного ответа."""
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer=truncate(answer, 2000),
        context=truncate(context, max_context_chars),
        reference=truncate(reference or "—", 1000),
    )
    try:
        raw = llm.chat(
            [ChatMessage(role="user", content=prompt)],
            purpose="judge",
            json_schema=JUDGE_SCHEMA,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Судья не ответил: %s", error)
        return {}
    try:
        parsed = json.loads(str(raw).strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError:
        logger.warning("Судья вернул невалидный JSON: %.120s", raw)
        return {}
    return parsed if isinstance(parsed, dict) else {}


def evaluate_answer(
    question: GoldQuestion,
    produced: Answer,
    *,
    reference_text: str,
    llm: LLMClient | None = None,
) -> AnswerOutcome:
    """Считает все четыре величины по одному вопросу."""
    context = "\n\n".join(item.chunk.text for item in produced.contexts)
    expected, found = latex_overlap(reference_text, produced.answer)

    outcome = AnswerOutcome(
        question_id=question.id,
        question_type=question.question_type,
        answer=produced.answer,
        refused=is_refusal(produced.answer),
        context_size=len(produced.contexts),
        latex_expected=expected,
        latex_found=found,
        unsupported=round(unsupported_share(produced.answer, context), 4),
        latency_ms=produced.timings_ms.get("total", 0.0),
    )

    if llm is not None:
        verdict = judge_answer(
            llm,
            question=question.question,
            answer=produced.answer,
            context=context,
            reference=question.answer,
        )
        if verdict:
            outcome.correctness = int(verdict.get("correctness", 0))
            outcome.groundedness = int(verdict.get("groundedness", 0))
            outcome.judge_reason = str(verdict.get("reason", ""))[:300]
    return outcome


def summarize_answers(outcomes: Sequence[AnswerOutcome]) -> dict[str, Any]:
    """Сводка с разбивкой по типам вопросов.

    Разбивка обязательна, а не желательна: среднее по набору уже однажды
    скрыло, что реранкер помогает одному типу вопросов и вредит другому.
    """

    def block(items: Sequence[AnswerOutcome]) -> dict[str, Any]:
        if not items:
            return {}
        judged = [item for item in items if item.correctness is not None]
        with_latex = [item for item in items if item.latex_recall is not None]
        result: dict[str, Any] = {
            "вопросов": len(items),
            "отказов": round(sum(1 for item in items if item.refused) / len(items), 4),
            "выдумка": round(statistics.fmean(item.unsupported for item in items), 4),
        }
        if with_latex:
            result["формулы дошли"] = round(
                statistics.fmean(item.latex_recall or 0.0 for item in with_latex), 4
            )
            result["вопросов с формулами"] = len(with_latex)
        if judged:
            result["верность"] = round(
                statistics.fmean(item.correctness or 0 for item in judged), 4
            )
            result["обоснованность"] = round(
                statistics.fmean(item.groundedness or 0 for item in judged), 4
            )
            result["оценено судьёй"] = len(judged)
        return result

    by_type: dict[str, Any] = {}
    grouped: dict[str, list[AnswerOutcome]] = {}
    for item in outcomes:
        grouped.setdefault(item.question_type, []).append(item)
    for name, items in sorted(grouped.items()):
        by_type[name] = block(items)

    return {"всего": block(outcomes), "по типам": by_type}


def run_answer_evaluation(
    context: Any,
    questions: Sequence[GoldQuestion],
    *,
    chunks: dict[str, Any] | None = None,
    judge: bool = True,
    max_workers: int = 2,
) -> tuple[dict[str, Any], list[AnswerOutcome]]:
    """Прогоняет вопросы через генерацию и оценивает ответы.

    ``chunks`` нужны, чтобы взять текст эталонного фрагмента: по нему
    считается сохранность формул. Без них эта величина не считается,
    а остальные — считаются.
    """
    judge_llm = context.llm if judge else None

    def evaluate_one(question: GoldQuestion) -> AnswerOutcome:
        produced = context.generator.answer(question.question, history=[])
        reference_text = ""
        if chunks:
            reference_text = "\n".join(
                getattr(chunks.get(chunk_id), "text", "")
                for chunk_id in question.gold_chunk_ids
            )
        return evaluate_answer(
            question, produced, reference_text=reference_text, llm=judge_llm
        )

    logger.info(
        "Оценка ответов: вопросов=%s, судья=%s", len(questions), "да" if judge else "нет"
    )
    if max_workers <= 1:
        outcomes = [evaluate_one(item) for item in questions]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(evaluate_one, questions))

    return summarize_answers(outcomes), outcomes


def save_answer_evaluation(
    summary: dict[str, Any],
    outcomes: Sequence[AnswerOutcome],
    directory: Path,
    *,
    label: str,
) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"answers_{label}.json"
    path.write_text(
        json.dumps(
            {
                "label": label,
                "summary": summary,
                "outcomes": [item.as_dict() for item in outcomes],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return path
