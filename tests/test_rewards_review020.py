"""Эксплойты из повторного ревью награды (задача 020).

Ревью искало обходы уже исправленной после 019 награды. Каждая правка
проверена на ручной сверке (``scripts/reward_recheck.py``): две
предложенные правки её роняли и отвергнуты — тесты ниже фиксируют
и их, чтобы решение не пересматривалось вслепую.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from rag_textbook.evaluation.goldset import looks_leaky, references_numbering
from rag_textbook.rewards import compute_reward
from rag_textbook.rewards.composite import copied_words
from rag_textbook.rewards.formula import canonical_tokens, extract_math, score_formulas

Q = "Как определяется скалярное произведение векторов?"
G = "Скалярное произведение векторов определяется формулой $$s=x_1y_1+x_2y_2$$."
ROOT = Path(__file__).resolve().parents[1]


def reward(answer, *, context=G, reference=G, question=Q, gold_in_context=True) -> float:
    return compute_reward(
        answer,
        context=context,
        reference=reference,
        question=question,
        gold_in_context=gold_in_context,
    ).total


# --- 1: вопрос вместо ответа ---------------------------------------------------


def test_question_with_a_formula_is_not_an_answer():
    honest = reward(G)
    assert reward("Как определяется скалярное произведение векторов $$s=x_1y_1+x_2y_2$$?") < honest
    assert reward("Как определяется скалярное произведение векторов $z$?") <= 0.0


# --- 2, 6: аргументы команд ------------------------------------------------------


def test_unbraced_arguments_keep_their_structure():
    assert canonical_tokens(r"y=\sqrt[3]{ab}") != canonical_tokens(r"y=\sqrt[3]a b")
    assert canonical_tokens(r"y=\frac1{ab}") != canonical_tokens(r"y=\frac1a b")


def test_equivalent_tex_spellings_are_equal():
    for left, right in (
        (r"y=\frac12", r"y=\frac{1}{2}"),
        (r"y=\frac1{ab}", r"y=\frac{1}{ab}"),
        (r"y=\sqrt[3]x", r"y=\sqrt[3]{x}"),
        (r"y=x+\frac12", r"y=x+\frac{1}{2}"),
    ):
        assert canonical_tokens(left) == canonical_tokens(right), (left, right)


def test_equation_with_swapped_sides_in_context_is_visible():
    gold = "Результат задаётся формулой $$s=x_1y_1+x_2y_2$$."
    context = "Результат задаётся формулой $$x_1y_1+x_2y_2=s$$."
    assert score_formulas(gold, gold, context).expected == 1


# --- 3, 5: выдуманное рядом с известным -----------------------------------------


def test_invented_tail_after_one_known_formula_is_foreign():
    honest = reward(G)
    for tail in (r"z=\omega+\psi", r"\omega=\psi"):
        answer = (
            f"Скалярное произведение векторов определяется формулой $$s=x_1y_1+x_2y_2; {tail}$$."
        )
        assert reward(answer) < honest, tail


def test_continuation_in_context_symbols_is_not_split():
    """Ручная сверка, ответ 10A: продолжение известной формулы — не «лишняя»."""
    context = reference = "Сумма $$a+b=c$$ и далее $$d=e+f$$."
    score = score_formulas(reference, "$$a+b=c=d+a$$", context)
    assert score.foreign == 0


def test_russian_words_do_not_hide_a_formula():
    assert extract_math(r"$z=\omega+\psi неверная формула$") == [r"z=\omega+\psi"]
    assert reward(G + r" $z=\omega+\psi неверная формула$") < reward(G)
    # Проза между сбитыми долларами формулой не становится.
    assert extract_math("цена $5, а при x = 3 получаем $") == []


# --- 7: отказы --------------------------------------------------------------------

REFUSAL = "Недостаточно информации для ответа."


def test_trivial_formula_does_not_cancel_a_short_refusal():
    assert reward(REFUSAL + " $z$") == reward(REFUSAL) == -0.5


def test_refusal_with_invented_formula_is_worse_than_refusal():
    assert reward(REFUSAL + r" $$z=\omega+\psi$$") < reward(REFUSAL)
    hidden = dict(context="Векторы описывают направление.", gold_in_context=False)
    assert reward(REFUSAL + r" $$z=\omega+\psi$$", **hidden) < reward(REFUSAL, **hidden)


def test_long_explanation_with_inline_symbols_is_not_a_refusal():
    """Ручная сверка, ответ 14B (оценка 3): объяснение, почему связи нет."""
    answer = (
        "В предоставленном контексте нет информации о связи градиентов и поворотов. "
        "Градиенты используются для поиска минимума функции $f(x)$ движением против "
        "градиента, а поворот — линейное отображение, поворачивающее плоскость на угол "
        "$\\theta$ относительно начала координат. В тексте нет формул, связывающих "
        "оптимизацию параметров модели с поворотами манипулятора робота, поэтому "
        "прямое соотношение из контекста вывести нельзя."
    )
    breakdown = compute_reward(answer, context=G, reference=G, question=Q)
    assert breakdown.gate != "отказ"


# --- дамп контекста -----------------------------------------------------------------


def test_copied_words_count_eight_word_runs():
    context = "один два три четыре пять шесть семь восемь девять десять"
    assert copied_words("один два три четыре пять шесть семь восемь", context) == 8
    # Семь слов подряд — ещё не копирование.
    assert copied_words("один два три четыре пять шесть семь", context) == 0
    # Замена одного слова не снимает совпадение с остальным отрезком.
    assert copied_words("один два три четыре пять шесть семь восемь СЛОВО десять", context) == 8


def test_dumping_a_long_fragment_loses_to_an_honest_answer():
    filler = " ".join(
        f"Пункт {i} описывает свойство номер {i} векторного пространства подробно и явно."
        for i in range(20)
    )
    context = G + " " + filler
    dump = G + " " + filler[:1500]
    assert reward(dump, context=context) < reward(G, context=context)


# --- отвергнутое: штраф «вне эталона» при частично видимом эталоне ------------------


def test_context_formulas_with_partly_visible_gold_are_not_offtarget():
    """Ручная сверка, вопросы 6 и 9: такой штраф ронял согласие до 0.663."""
    extra = " ".join(f"$$u_{i}=v_{i}+w_{i}$$" for i in range(8))
    context = G + " Дополнительные формулы " + extra
    reference = G + r" Требуется также $$q=\gamma+\theta$$."
    result = compute_reward(
        context, context=context, reference=reference, question=Q, gold_in_context=False
    )
    assert "offtarget" not in result.parts


# --- 8, 9: среда ----------------------------------------------------------------------


def test_leak_and_numbering_filters():
    assert looks_leaky("Что утверждается в данном фрагменте?")
    assert references_numbering("Что следует из формулы № 2.3?")
    assert references_numbering("Что следует из уравнения (2.3)?")
    assert not references_numbering("Как решить уравнение 2x + 3 = 0?")


def test_missing_verdicts_file_is_an_error(tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "rl_dataset.py"),
            "--trace",
            str(tmp_path / "t.jsonl"),
            "--goldset",
            str(tmp_path / "g.json"),
            "--prompt",
            str(tmp_path / "p.txt"),
            "--out",
            str(tmp_path / "out"),
            "--verdicts",
            str(tmp_path / "no-such.json"),
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        # Без этого дочерний Python на Windows пишет stderr в cp1251,
        # и проверка падает на разборе вывода, а не на сути.
        env={**os.environ, "PYTHONIOENCODING": "utf-8"},
    )
    assert result.returncode != 0
    assert "вердикт" in result.stderr


def test_copied_word_count_is_reported():
    breakdown = compute_reward(G, context=G, reference=G, question=Q)
    assert breakdown.diagnostics["copied_words"] == copied_words(G, G)
