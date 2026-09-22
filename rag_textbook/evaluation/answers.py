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
``опора на контекст`` доля предложений ответа, для которых в контексте
                     нашлось предложение, разделяющее с ними больше половины
                     содержательных слов. Прежняя мера того же назначения,
                     ``unsupported``, оказалась негодной: она давала
                     0.982-0.988 во всех прогонах и различать конфигурации
                     ею было нельзя. Оставлена только для чтения старых
                     файлов метрик.

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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Answer, GoldQuestion, ScoredChunk
from rag_textbook.utils.text import (
    content_terms,
    extract_latex_fragments,
    split_sentences,
    strip_latex,
    truncate,
)

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
# Типичные зачины рассуждения. Список короткий намеренно: он опознаёт
# не «плохой стиль», а англоязычный поток мыслей, который модель выдаёт
# вместо ответа.
_REASONING_OPENERS = (
    "the user",
    "i need",
    "okay",
    "let me",
    "we need",
    "first,",
    "looking at",
)

REFUSAL_MARKERS = (
    "не содержится",
    "нет данных",
    "недостаточно информации",
    "не удалось найти",
    # Задача 019: «В контексте нет информации для ответа» не считался
    # отказом и при невидимом эталоне получал 0 вместо +0.5.
    "нет информации",
    "недостаточно данных",
    "невозможно ответить",
    "не могу ответить",
)


@dataclass
class AnswerOutcome:
    """Результат по одному вопросу."""

    question_id: str
    question_type: str
    answer: str = ""
    refused: bool = False
    reasoning_leak: bool = False
    latin_share: float = 0.0
    context_size: int = 0
    # Объективные признаки
    latex_expected: int = 0
    latex_found: int = 0
    unsupported: float = 0.0  # негодная мера, оставлена для чтения старых файлов
    sentences_judged: int = 0
    sentences_supported: int = 0
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


def looks_like_reasoning(answer: str) -> bool:
    """Ответ содержит размышление модели, а не ответ.

    Признак понадобился задним числом: в прогоне 2026-08-19 46% ответов
    содержали тег think, 60% начинались с английского «The user is asking».
    Метрики считались по потоку мыслей и выглядели правдоподобно — верность
    1.116, сохранность формул 0.05, — а причина была не в поиске и не
    в модели, а в том, что ответа в тексте не было вовсе.
    """
    text = (answer or "").strip()
    if not text:
        return False
    if "<think>" in text.lower():
        return True
    opening = text.lower()[:60]
    return opening.startswith(_REASONING_OPENERS)


def latin_share(answer: str) -> float:
    """Доля латиницы среди букв ответа.

    Корпус русский, вопросы русские. Ответ на английском — это не стиль,
    а отказ работать: студент его не ждёт. Мера грубая, зато не зависит
    ни от судьи, ни от разметки.

    Формулы из подсчёта исключаются, и это исправление, а не придирка.
    В замере 2026-09-03 мера дала 0.330 на формульных вопросах против
    0.054 на связывающих — но правильный ответ про матрицы состоит
    из LaTeX почти целиком, и латиницы в нём законно больше половины.
    Мера ловила бы ровно те ответы, ради которых всё делается.
    """
    letters = [c for c in strip_latex(answer) if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if "a" <= c.lower() <= "z") / len(letters)


def is_refusal(answer: str) -> bool:
    lowered = (answer or "").lower()
    return any(marker in lowered for marker in REFUSAL_MARKERS)


def _normalize_latex(fragment: str) -> str:
    """Убирает то, что не меняет смысла формулы.

    Пробелы внутри LaTeX расставляются как придётся — и парсером, и моделью, —
    поэтому дословное сравнение без нормализации занижало бы совпадение
    до нуля почти всегда.

    Номер формулы убирается отдельно, и это не мелочь. В учебнике формулы
    пронумерованы, и парсер оставляет номер внутри самой формулы:
    ``$$p(x|t),\tag{8.24}$$``. Отвечая, модель переносит формулу без номера —
    так и надо, — и прежнее сравнение считало это промахом. Замер 2026-09-03:
    ответы содержали формулу дословно, а метрика показывала 0.065.
    """
    cleaned = re.sub(r"\\(?:tag|label|nonumber)\s*\{[^}]*\}", "", fragment or "")
    cleaned = re.sub(r"\\(?:begin|end)\s*\{[a-z*]+\}", "", cleaned)
    return re.sub(r"[\s,.]+", "", cleaned)


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
    """Негодная мера. Оставлена, чтобы старые файлы метрик читались.

    Давала 0.982-0.988 во всех прогонах при разбросе по типам вопросов
    в сотые доли: различать конфигурации ею нельзя. Причин две. Внешняя:
    четвёрка слов подряд совпадает только при дословном переписывании,
    а пересказ она засчитывает как выдумку. Внутренняя, найденная позже:
    ``content_terms`` выбрасывает повторы, поэтому «четвёрка слов подряд»
    считалась не по тексту, а по списку уникальных терминов в порядке
    первого появления — совпадений там не бывает почти никогда.

    Заменена на :func:`sentence_support`.
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


# Ниже порога предложение считается не опирающимся на контекст. Значение
# выбрано по разметке: пересказ фрагмента удерживает 0.6-0.9 содержательных
# слов исходного предложения, а привнесённое извне утверждение делит
# с контекстом единичные термины из самого вопроса.
SUPPORT_THRESHOLD = 0.5

# Предложения короче этого числа содержательных слов не оцениваются:
# «Итого:», «Следовательно, да.» и заголовки списков не несут утверждения,
# которое можно было бы проверить по контексту.
_MIN_TERMS = 3


def sentence_support(
    answer: str, context: str, *, threshold: float = SUPPORT_THRESHOLD
) -> tuple[int, int]:
    """Сколько предложений ответа опирается на контекст.

    Для каждого предложения ответа берётся лучшее предложение контекста
    и считается доля содержательных слов ответа, которые в нём есть.
    Мера лексическая, как и прежняя, но узел зернистости другой:
    пересказ сохраняет содержательные слова исходного предложения,
    даже когда переставляет их и меняет связки, — а именно на перестановке
    ломалась прежняя мера.

    Возвращает пару «сколько предложений оценено, сколько опирается
    на контекст». Пара, а не доля: при коротком ответе знаменатель
    важен не меньше значения.
    """
    context_sets = [
        set(content_terms(sentence)) for sentence in split_sentences(context)
    ]
    context_sets = [item for item in context_sets if item]
    judged = 0
    supported = 0
    for sentence in split_sentences(answer):
        terms = set(content_terms(sentence))
        if len(terms) < _MIN_TERMS:
            continue
        judged += 1
        if not context_sets:
            continue
        best = max(len(terms & item) / len(terms) for item in context_sets)
        if best >= threshold:
            supported += 1
    return judged, supported


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
    judged, supported = sentence_support(produced.answer, context)

    outcome = AnswerOutcome(
        question_id=question.id,
        question_type=question.question_type,
        answer=produced.answer,
        refused=is_refusal(produced.answer),
        reasoning_leak=looks_like_reasoning(produced.answer),
        latin_share=round(latin_share(produced.answer), 4),
        context_size=len(produced.contexts),
        latex_expected=expected,
        latex_found=found,
        unsupported=round(unsupported_share(produced.answer, context), 4),
        sentences_judged=judged,
        sentences_supported=supported,
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
            # Две величины ниже — про то, ответ ли перед нами вообще.
            # Без них качество считается по потоку мыслей модели.
            "размышление вместо ответа": round(
                sum(1 for item in items if item.reasoning_leak) / len(items), 4
            ),
            "ответ не по-русски": round(
                sum(1 for item in items if item.latin_share > 0.5) / len(items), 4
            ),
        }
        # Доля предложений ответа, опирающихся на контекст. Знаменатель
        # выводится рядом: при коротких ответах он важен не меньше.
        judged_total = sum(item.sentences_judged for item in items)
        if judged_total:
            result["опора на контекст"] = round(
                sum(item.sentences_supported for item in items) / judged_total, 4
            )
            result["предложений оценено"] = judged_total
        if with_latex:
            result["формулы дошли"] = round(
                statistics.fmean(item.latex_recall or 0.0 for item in with_latex), 4
            )
            # Главная величина из двух. Доля выше требует, чтобы в ответ попали
            # ВСЕ формулы эталонного фрагмента, а их там медианно четыре: на
            # странице учебника формул несколько, и хороший ответ приводит
            # ту, о которой спрашивали. Эта же считает вопросы, где до ответа
            # дошла хотя бы одна, и именно её надо читать как «дошло или нет».
            result["хотя бы одна формула"] = round(
                sum(1 for item in with_latex if item.latex_found) / len(with_latex), 4
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
    frozen_contexts: dict[str, Sequence[str]] | None = None,
) -> tuple[dict[str, Any], list[AnswerOutcome]]:
    """Прогоняет вопросы через генерацию и оценивает ответы.

    ``chunks`` нужны, чтобы взять текст эталонного фрагмента: по нему
    считается сохранность формул. Без них эта величина не считается,
    а остальные — считаются.
    """
    judge_llm = context.llm if judge else None

    def answer_one(question: GoldQuestion) -> Answer:
        """Ответ на вопрос: обычным путём либо по замороженному контексту.

        Замороженный контекст берётся из слепка и нужен для сравнения
        генераторов: все модели отвечают по одному и тому же материалу,
        поэтому разница относится к генератору, а не к тому, что кому
        досталось при поиске.
        """
        if frozen_contexts is None:
            return context.generator.answer(question.question, history=[])

        chunk_ids = frozen_contexts.get(question.id, ())
        picked = [
            ScoredChunk(chunk=chunks[chunk_id], score=1.0)
            for chunk_id in chunk_ids
            if chunks and chunk_id in chunks
        ]
        if not picked:
            logger.warning("Вопрос %s: замороженный контекст пуст", question.id)
        ordered, text, generation_ms = context.generator.answer_from_context(
            question.question, picked
        )
        return Answer(
            question=question.question,
            answer=text,
            contexts=ordered,
            timings_ms={"generation": round(generation_ms, 1), "total": round(generation_ms, 1)},
        )

    def evaluate_one(question: GoldQuestion) -> AnswerOutcome:
        produced = answer_one(question)
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
        "Оценка ответов: вопросов=%s, судья=%s, контекст=%s",
        len(questions),
        "да" if judge else "нет",
        "из слепка" if frozen_contexts is not None else "из поиска",
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
