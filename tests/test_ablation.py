"""Абляционная проверка «нужны ли оба фрагмента».

Смысл проверки в том, что она не спрашивает у модели мнения о разметке,
а ставит опыт: задаёт вопрос по каждому фрагменту отдельно и по обоим сразу.
Поэтому тесты устроены так же — подставная модель отвечает по-разному
в зависимости от того, что ей подали, и вердикт обязан следовать из
разницы между прогонами, а не из чьей-либо оценки.
"""

from __future__ import annotations

from rag_textbook.evaluation.ablation import (
    AblationResult,
    ablate_question,
    answers_match,
    summarize_ablation,
)
from rag_textbook.models import Chunk, GoldQuestion


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=0,
        text=text,
    )


def _question(chunk_ids: list[str]) -> GoldQuestion:
    return GoldQuestion(
        id="q1",
        question="Чему равно произведение собственных значений матрицы?",
        answer="Определителю матрицы.",
        gold_chunk_ids=chunk_ids,
        gold_doc_ids=["doc"],
        question_type="graph_linked",
        expected_hops=2,
    )


CHUNKS = {
    "A": _chunk("A", "Определитель равен произведению собственных значений."),
    "B": _chunk("B", "Собственные значения — корни характеристического многочлена."),
}


class _StubLLM:
    """Отвечает по содержимому поданного контекста.

    ``answer_when`` — какие фрагменты должны присутствовать, чтобы модель
    дала верный ответ. Всё остальное — отказ.
    """

    def __init__(self, answer_when: set[str]) -> None:
        self.answer_when = answer_when
        self.judged = 0
        self.contexts: list[str] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        content = messages[0].content
        if kwargs.get("json_schema"):
            self.judged += 1
            verdict = "Определителю" in content.split("ПРОВЕРЯЕМЫЙ ОТВЕТ:")[-1]
            return f'{{"match": {int(verdict)}}}'
        self.contexts.append(content)
        present = {name for name in ("A", "B") if CHUNKS[name].text in content}
        if self.answer_when <= present:
            return "Определителю матрицы."
        return "В предоставленном контексте нет данных."


def test_one_chunk_is_enough_is_detected():
    """Если ответ получается по одному фрагменту, второй вопросу не нужен —
    и никакое суждение модели этого не отменит."""
    llm = _StubLLM(answer_when={"A"})

    result = ablate_question(llm, _question(["A", "B"]), CHUNKS)

    assert result.verdict == "single_hop_enough"
    assert result.single_matches == [True, False]
    assert "первого" in result.note


def test_genuinely_two_hop_question_is_accepted():
    llm = _StubLLM(answer_when={"A", "B"})

    result = ablate_question(llm, _question(["A", "B"]), CHUNKS)

    assert result.verdict == "ok"
    assert result.single_matches == [False, False]
    assert result.joint_match


def test_question_unanswerable_from_its_own_gold_chunks():
    """Ответ не получается даже по всем эталонным фрагментам — испорчен
    либо вопрос, либо эталонный ответ. Такой вопрос меряет не систему."""
    llm = _StubLLM(answer_when={"A", "B", "нет такого"})

    result = ablate_question(llm, _question(["A", "B"]), CHUNKS)

    assert result.verdict == "unanswerable"


def test_missing_chunks_do_not_reach_the_model():
    llm = _StubLLM(answer_when=set())

    result = ablate_question(llm, _question(["нет-в-корпусе"]), CHUNKS)

    assert result.verdict == "unanswerable"
    assert llm.contexts == [], "обращаться к модели не за чем"


def test_refusal_never_counts_as_a_match():
    """Отказ распознаётся до судьи: это и экономит вызов, и убирает случай,
    в котором судья мог бы счесть отказ верным ответом."""
    llm = _StubLLM(answer_when={"A"})

    assert not answers_match(
        llm,
        question="вопрос",
        reference="Определителю матрицы.",
        candidate="В предоставленном контексте нет данных.",
    )
    assert llm.judged == 0


def test_single_chunk_question_is_checked_for_answerability():
    llm = _StubLLM(answer_when={"A"})
    question = GoldQuestion(
        id="q2",
        question="Чему равно произведение собственных значений?",
        answer="Определителю матрицы.",
        gold_chunk_ids=["A"],
        gold_doc_ids=["doc"],
        question_type="single_chunk",
    )

    result = ablate_question(llm, question, CHUNKS)

    assert result.verdict == "ok"
    assert result.single_matches == [], "у одного фрагмента отдельных прогонов нет"


def test_summary_reports_share_of_single_hop_among_linked():
    results = [
        AblationResult("q1", "graph_linked", "single_hop_enough", [True, False], False),
        AblationResult("q2", "graph_linked", "ok", [False, False], True),
        AblationResult("q3", "single_chunk", "ok", [], True),
    ]

    summary = summarize_ablation(results)

    assert summary["доля одношаговых среди связывающих"] == 0.5
    assert summary["всего"]["ok"] == 2


def test_result_converts_to_verdict_record():
    result = AblationResult("q1", "graph_linked", "single_hop_enough", [True, False], False, "почему")

    verdict = result.to_verdict()

    assert (verdict.question_id, verdict.verdict, verdict.note) == (
        "q1",
        "single_hop_enough",
        "почему",
    )
