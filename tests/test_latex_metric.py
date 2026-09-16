"""Сохранность формул: что именно мы считаем.

Замер 2026-09-03 показал 0.065 при том, что ответы содержали формулы
дословно — глазами это было видно на первых же пяти вопросах. Разбор
нашёл три причины, и все три проверяются здесь.

1. Выделение формул захватывало прозу. Выражение «доллар, что угодно
   кроме доллара, доллар» при нечётном числе долларов в абзаце склеивало
   закрывающий знак одной формулы с открывающим следующей, и в «формулу»
   попадал абзац целиком.
2. Номер формулы. Парсер оставляет «\tag{8.24}» внутри самой формулы,
   а модель переносит формулу без номера — и это правильно. Сравнение
   считало такой перенос промахом.
3. Знаменатель. Доля требовала, чтобы в ответ попали ВСЕ формулы
   эталонного фрагмента, а их там медианно четыре.
"""

from __future__ import annotations

from rag_textbook.evaluation.answers import AnswerOutcome, latex_overlap, summarize_answers
from rag_textbook.utils.text import extract_latex_fragments

FORMULA = r"$$ p ( { \pmb x } | { \boldsymbol \theta } , z ) $$"
NUMBERED = r"$$ p ( { \pmb x } | { \boldsymbol \theta } , z ) ,\tag{8.24} $$"


def test_prose_between_formulas_is_not_a_formula():
    text = (
        r"Формула $$a = b$$ и дальше 348 Глава 8. О сочетании модели "
        r"и данных при помощи которого сможем сгенерировать данные $x$"
    )

    fragments = extract_latex_fragments(text)

    assert fragments == ["$$a = b$$", "$x$"]


def test_paragraph_swallowed_by_a_lone_dollar_is_dropped():
    """Ровно тот случай, что раздувал знаменатель до семи «формул»."""
    text = r"$$a=b$$ при помощи которого сможем сгенерировать данные для любых $"

    assert extract_latex_fragments(text) == ["$$a=b$$"]


def test_formula_carried_without_its_number_counts_as_carried():
    reference = f"Это распределение {NUMBERED} далее по тексту."
    answer = f"Это условное распределение называется правдоподобием. {FORMULA} [1]"

    assert latex_overlap(reference, answer) == (1, 1)


def test_different_formula_still_counts_as_missing():
    """Послабление не должно засчитывать любую формулу за любую."""
    reference = f"Это распределение {NUMBERED} далее."
    answer = r"Ответ: $$ \det(A) = \prod_i \lambda_i $$"

    assert latex_overlap(reference, answer) == (1, 0)


def test_single_letters_are_still_ignored():
    reference = r"Пусть $x$ и $y$ — переменные."

    assert latex_overlap(reference, "Ответ про $x$.") == (0, 0)


def test_summary_reports_hit_rate_next_to_the_share():
    """Доля и «хотя бы одна» отвечают на разные вопросы, и в сводке
    нужны обе: доля занижена знаменателем, а «хотя бы одна» говорит,
    дошла формула до ответа или нет."""
    outcomes = [
        AnswerOutcome(question_id="q1", question_type="formula_table",
                      latex_expected=4, latex_found=1),
        AnswerOutcome(question_id="q2", question_type="formula_table",
                      latex_expected=4, latex_found=0),
    ]

    summary = summarize_answers(outcomes)["всего"]

    assert summary["формулы дошли"] == 0.125
    assert summary["хотя бы одна формула"] == 0.5
