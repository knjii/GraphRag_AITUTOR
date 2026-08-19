"""Абляционная проверка: действительно ли вопросу нужны оба фрагмента.

Главный изъян набора измерен вручную и оказался крупным: половине
«связывающих» вопросов хватает одного фрагмента из двух. Значит, отставание
связывающих вопросов по recall наполовину создано разметкой, а не системой.
Проверить это на всём наборе руками нельзя — вопросов сотни, а вычитывать их
дорого. Спросить модель «нужны ли оба фрагмента?» тоже нельзя: она отвечает
про собственную же работу и склонна соглашаться.

Здесь применён приём, который не спрашивает мнения, а **ставит опыт**.
Вопрос задаётся трижды: по фрагменту А, по фрагменту Б и по обоим. Затем
сравнивается, что получилось. Если ответ по одному фрагменту уже совпадает
с эталонным, значит второй фрагмент вопросу не нужен, и никакое суждение
модели этого не отменит.

    ответ по А верен          -> single_hop_enough (второй фрагмент лишний)
    ответ по Б верен          -> single_hop_enough
    верен только по обоим     -> ok (вопрос действительно двухшаговый)
    неверен даже по обоим     -> unanswerable (вопрос или эталон испорчены)

Судья здесь всё-таки нужен — сравнить два свободных текста иначе нельзя, —
но задача у него куда более узкая, чем «оцени качество»: он сравнивает
ответ с эталоном, а не судит систему. Ошибётся он одинаково во всех трёх
прогонах, поэтому **разность** между ними устойчивее, чем каждая оценка
по отдельности. Именно на разности и строится вердикт.

Отдельно: отказ модели («в контексте нет данных») распознаётся до обращения
к судье. Это и экономит вызовы, и убирает случай, в котором судья мог бы
счесть отказ верным ответом.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.evaluation.answers import is_refusal
from rag_textbook.evaluation.verdicts import QuestionVerdict
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, GoldQuestion
from rag_textbook.utils.text import truncate

logger = get_logger("evaluation.ablation")

ANSWER_PROMPT = """Ответь на вопрос, опираясь только на приведённые фрагменты учебника.
Если ответа в них нет, ответь ровно: «нет данных».

ФРАГМЕНТЫ:
{context}

ВОПРОС: {question}

Ответ (не более трёх предложений):"""

MATCH_SCHEMA = {
    "type": "object",
    "properties": {
        "match": {"type": "integer", "minimum": 0, "maximum": 1},
        "reason": {"type": "string"},
    },
    "required": ["match"],
}

MATCH_PROMPT = """Сравни два ответа на один вопрос по существу.

ВОПРОС: {question}

ЭТАЛОННЫЙ ОТВЕТ: {reference}

ПРОВЕРЯЕМЫЙ ОТВЕТ: {candidate}

Совпадают ли они по существу? Различия в формулировках, порядке слов
и полноте пояснений значения не имеют — важно только, назван ли тот же
предмет и то же утверждение.

Верни строго JSON: {{"match": 1, "reason": "коротко"}}
Где match = 1, если ответы совпадают по существу, и 0, если нет."""

# Ответ длиннее этого в промпт не помещаем: сравнение по существу от хвоста
# не выигрывает, а лимит контекста расходуется.
_MAX_ANSWER_CHARS = 1200
_MAX_CHUNK_CHARS = 3000


@dataclass
class AblationResult:
    """Исход опыта по одному вопросу."""

    question_id: str
    question_type: str
    verdict: str
    # Что получилось в каждом прогоне: индекс -> совпал ли с эталоном.
    single_matches: list[bool]
    joint_match: bool
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_verdict(self) -> QuestionVerdict:
        return QuestionVerdict(
            question_id=self.question_id, verdict=self.verdict, note=self.note
        )


def _answer_with(llm: LLMClient, question: str, texts: Sequence[str]) -> str:
    context = "\n\n---\n\n".join(truncate(text, _MAX_CHUNK_CHARS) for text in texts)
    try:
        raw = llm.chat(
            [ChatMessage(role="user", content=ANSWER_PROMPT.format(context=context, question=question))],
            # Служебное назначение: с назначением `chat` рассуждающая модель
            # тратит лимит токенов на размышление и возвращает пустой ответ.
            purpose="utility",
            temperature=0.0,
            max_tokens=400,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Абляция: модель не ответила (%s)", error)
        return ""
    return str(raw or "").strip()


def answers_match(llm: LLMClient, *, question: str, reference: str, candidate: str) -> bool:
    """Совпадают ли ответы по существу.

    Отказ отвечать распознаётся до обращения к судье: он заведомо не совпадает
    с эталоном, а судья на нём мог бы ошибиться в любую сторону.
    """
    if not candidate.strip() or is_refusal(candidate):
        return False
    if not reference.strip():
        return False
    prompt = MATCH_PROMPT.format(
        question=question,
        reference=truncate(reference, _MAX_ANSWER_CHARS),
        candidate=truncate(candidate, _MAX_ANSWER_CHARS),
    )
    try:
        raw = llm.chat(
            [ChatMessage(role="user", content=prompt)],
            purpose="utility",
            json_schema=MATCH_SCHEMA,
            temperature=0.0,
            max_tokens=256,
        )
    except Exception as error:  # noqa: BLE001
        logger.warning("Абляция: судья не ответил (%s)", error)
        return False
    try:
        payload = json.loads(str(raw).strip().removeprefix("```json").removesuffix("```"))
    except json.JSONDecodeError:
        logger.warning("Абляция: судья вернул не JSON: %.120s", raw)
        return False
    return bool(isinstance(payload, dict) and int(payload.get("match", 0)) == 1)


def ablate_question(
    llm: LLMClient, question: GoldQuestion, chunks: dict[str, Chunk]
) -> AblationResult:
    """Ставит опыт по одному вопросу и выносит вердикт."""
    texts = [
        chunks[chunk_id].text for chunk_id in question.gold_chunk_ids if chunk_id in chunks
    ]
    if not texts:
        return AblationResult(
            question_id=question.id,
            question_type=question.question_type,
            verdict="unanswerable",
            single_matches=[],
            joint_match=False,
            note="эталонных фрагментов нет в корпусе",
        )

    reference = question.answer or ""
    joint = _answer_with(llm, question.question, texts)
    joint_match = answers_match(
        llm, question=question.question, reference=reference, candidate=joint
    )

    single_matches: list[bool] = []
    if len(texts) > 1:
        for text in texts:
            produced = _answer_with(llm, question.question, [text])
            single_matches.append(
                answers_match(
                    llm, question=question.question, reference=reference, candidate=produced
                )
            )

    if len(texts) > 1 and any(single_matches):
        which = "первого" if single_matches[0] else "второго"
        verdict, note = "single_hop_enough", f"ответ получен по одному только {which} фрагменту"
    elif joint_match:
        verdict = "ok"
        note = "оба фрагмента нужны" if len(texts) > 1 else "ответ получен по эталонному фрагменту"
    else:
        verdict, note = "unanswerable", "ответ не получен даже по всем эталонным фрагментам"

    return AblationResult(
        question_id=question.id,
        question_type=question.question_type,
        verdict=verdict,
        single_matches=single_matches,
        joint_match=joint_match,
        note=note,
    )


def run_ablation(
    llm: LLMClient,
    questions: Sequence[GoldQuestion],
    chunks: dict[str, Chunk],
    *,
    max_workers: int = 4,
) -> list[AblationResult]:
    logger.info("Абляционная проверка: вопросов %s", len(questions))

    def one(question: GoldQuestion) -> AblationResult:
        return ablate_question(llm, question, chunks)

    if max_workers <= 1:
        return [one(item) for item in questions]
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(one, questions))


def summarize_ablation(results: Sequence[AblationResult]) -> dict[str, Any]:
    by_type: dict[str, dict[str, int]] = {}
    totals: dict[str, int] = {}
    for item in results:
        bucket = by_type.setdefault(item.question_type, {})
        bucket[item.verdict] = bucket.get(item.verdict, 0) + 1
        totals[item.verdict] = totals.get(item.verdict, 0) + 1
    summary: dict[str, Any] = {"всего": totals, "по типам": by_type}
    linked = by_type.get("graph_linked", {})
    checked = sum(linked.values())
    if checked:
        summary["доля одношаговых среди связывающих"] = round(
            linked.get("single_hop_enough", 0) / checked, 4
        )
    return summary
