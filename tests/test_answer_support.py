"""Замена негодной мере «выдумки».

Прежняя мера ``unsupported_share`` считала долю четвёрок слов ответа,
которых нет в контексте, и давала 0.982-0.988 во всех прогонах при разбросе
по типам вопросов в сотые доли. Различать конфигурации ею было нельзя,
но в сводке она выглядела измерением.

Причин у этого две, и обе проверяются ниже. Внешняя: пересказ она
засчитывает как выдумку, а генератор пересказывает почти всегда.
Внутренняя: ``content_terms`` выбрасывает повторы, поэтому «четвёрка слов
подряд» бралась не из текста, а из списка уникальных терминов.

Новая мера ``sentence_support`` работает на предложениях: пересказ
сохраняет содержательные слова исходного предложения, даже когда меняет
порядок и связки.
"""

from __future__ import annotations

from rag_textbook.evaluation.answers import (
    AnswerOutcome,
    sentence_support,
    summarize_answers,
    unsupported_share,
)

CONTEXT = (
    "Определитель матрицы равен произведению всех её собственных значений. "
    "Собственные значения являются корнями характеристического многочлена."
)


def test_paraphrase_counts_as_supported():
    """Ровно то, на чём ломалась прежняя мера."""
    answer = "Определитель равен произведению собственных значений матрицы."

    judged, supported = sentence_support(answer, CONTEXT)

    assert (judged, supported) == (1, 1)


def test_invention_counts_as_unsupported():
    answer = "Теорема Пифагора связывает катеты прямоугольного треугольника."

    assert sentence_support(answer, CONTEXT) == (1, 0)


def test_measure_separates_the_two_cases():
    """Главное требование: между пересказом и выдумкой должно быть
    расстояние. У прежней меры его не было."""
    good = "Определитель равен произведению собственных значений матрицы."
    bad = "Теорема Пифагора связывает катеты прямоугольного треугольника."

    assert sentence_support(good, CONTEXT)[1] > sentence_support(bad, CONTEXT)[1]
    # А прежняя мера обе называет выдумкой почти в равной степени.
    assert unsupported_share(good, CONTEXT) > 0.9
    assert unsupported_share(bad, CONTEXT) > 0.9


def test_short_statements_are_not_judged():
    """«Да.» и «Итого:» не несут проверяемого утверждения."""
    judged, _ = sentence_support("Да. Итого:", CONTEXT)

    assert judged == 0


def test_mixed_answer_is_counted_by_sentences():
    answer = (
        "Определитель равен произведению собственных значений матрицы. "
        "Теорема Пифагора связывает катеты прямоугольного треугольника."
    )

    assert sentence_support(answer, CONTEXT) == (2, 1)


def test_empty_context_supports_nothing():
    answer = "Определитель равен произведению собственных значений матрицы."

    assert sentence_support(answer, "") == (1, 0)


def test_summary_reports_support_with_its_denominator():
    outcomes = [
        AnswerOutcome(
            question_id="q1",
            question_type="formula_table",
            sentences_judged=4,
            sentences_supported=3,
        ),
        AnswerOutcome(
            question_id="q2",
            question_type="formula_table",
            sentences_judged=6,
            sentences_supported=3,
        ),
    ]

    summary = summarize_answers(outcomes)

    assert summary["всего"]["опора на контекст"] == 0.6
    assert summary["всего"]["предложений оценено"] == 10


def test_summary_omits_support_when_nothing_was_judged():
    """Отсутствие величины честнее нуля: ноль читался бы как «ни одно
    предложение не опирается на контекст»."""
    summary = summarize_answers([AnswerOutcome(question_id="q1", question_type="single_chunk")])

    assert "опора на контекст" not in summary["всего"]
