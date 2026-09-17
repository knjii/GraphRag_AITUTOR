"""Награда R6: каноническая форма формул, ворота и защита от взлома."""

from __future__ import annotations

from rag_textbook.rewards import (
    canonical_tokens,
    compute_reward,
    extract_math,
    format_reward,
    random_reward,
    score_formulas,
)

GOLD = (
    "Скалярное произведение векторов задаётся формулой "
    "$$\\langle x, y \\rangle = x^{T} A y ,\\tag{3.5}$$ "
    "где матрица $A$ симметрична и положительно определена."
)
CONTEXT = GOLD + " Норма вводится как $$\\|x\\| = \\sqrt{\\langle x, x \\rangle}.$$"


def test_spacing_and_decoration_do_not_matter():
    assert canonical_tokens(r"\left( \frac{a}{b} \right)") == canonical_tokens(r"(\dfrac{a}{b})")
    assert canonical_tokens(r"x^{2} + y^{2}") == canonical_tokens(r"x^2+y^2")


def test_transpose_spellings_are_equal():
    forms = [r"x^T A y", r"x^{T} A y", r"x^\top A y", r"x^{\top} A y", r"x^{\intercal}Ay"]
    assert len({canonical_tokens(item) for item in forms}) == 1


def test_equation_number_is_ignored():
    assert canonical_tokens(r"a = b + c \tag{8.24}") == canonical_tokens("a=b+c")


def test_different_formula_stays_different():
    assert canonical_tokens(r"x^T A y") != canonical_tokens(r"y^T A x")


def test_all_delimiters_are_extracted():
    text = r"Итак, $$a+b$$, затем \[c+d\], потом \(e+f\) и $g+h$."
    assert extract_math(text) == ["a+b", "c+d", "e+f", "g+h"]


def test_prose_between_dollars_is_not_a_formula():
    text = "Цена $5, а скидка составляет примерно половину от цены $"
    assert extract_math(text) == []


def test_formula_in_other_notation_counts_as_carried():
    answer = r"Скалярное произведение: \[ \langle x,y\rangle = x^\top A\, y \] [1]."
    score = score_formulas(GOLD, answer, CONTEXT)
    assert score.expected == 1
    assert score.carried == 1
    assert score.foreign == 0


def test_formula_inside_longer_expression_counts():
    answer = r"Получаем $$f(x) = \langle x, y \rangle = x^{T} A y + b$$."
    assert score_formulas(GOLD, answer, CONTEXT).carried == 1


def test_near_miss_gets_partial_not_full_credit():
    answer = r"Скалярное произведение равно $$\langle x, y \rangle = x^{T} B y$$."
    score = score_formulas(GOLD, answer, CONTEXT)
    assert score.carried == 0
    assert 0.5 < score.partial < 1.0


def test_invented_formula_is_foreign():
    answer = r"По определению $$\langle x, y \rangle = \sum_i x_i^3 y_i^3$$."
    score = score_formulas(GOLD, answer, CONTEXT)
    assert score.foreign == 1
    assert score.foreign_share == 1.0


def test_short_formulas_are_not_judged():
    assert score_formulas("Пусть $x$ и $y=1$.", "Возьмём $x$.", "").expected == 0


def test_good_answer_beats_answer_without_formula():
    good = (
        "Скалярное произведение задаётся как $$\\langle x, y \\rangle = x^{T} A y$$, "
        "где матрица A симметрична и положительно определена [1]."
    )
    plain = "Скалярное произведение задаётся через симметричную положительно определённую матрицу [1]."
    assert compute_reward(good, context=CONTEXT, reference=GOLD).total > compute_reward(
        plain, context=CONTEXT, reference=GOLD
    ).total


def test_invented_formula_is_penalized():
    honest = "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$, матрица симметрична [1]."
    liar = (
        "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$ "
        "и ещё $$\\langle x, y \\rangle = \\sum_i x_i^3 y_i^3 + 7$$, матрица симметрична [1]."
    )
    assert compute_reward(liar, context=CONTEXT, reference=GOLD).total < compute_reward(
        honest, context=CONTEXT, reference=GOLD
    ).total


def test_dumping_formulas_without_prose_is_penalized():
    dump = " ".join(f"$$a_{{{i}}} + b_{{{i}}} = c_{{{i}}} \\cdot d$$" for i in range(8))
    context = CONTEXT + " " + dump
    explained = "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$, матрица симметрична [1]."
    dumped = "$$\\langle x, y \\rangle = x^{T} A y$$ " + dump
    assert compute_reward(dumped, context=context, reference=GOLD).total < compute_reward(
        explained, context=context, reference=GOLD
    ).total


def test_copying_whole_context_is_penalized_by_length():
    long_context = CONTEXT + " " + "Определение нормы опирается на скалярное произведение. " * 80
    copy = long_context[:3900]
    concise = "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$, матрица симметрична [1]."
    reward = compute_reward(copy, context=long_context, reference=GOLD)
    assert reward.parts.get("length", 0) < 0
    assert reward.total < compute_reward(concise, context=long_context, reference=GOLD).total


def test_gates_short_circuit():
    for answer, reason in (
        ("", "пустой ответ"),
        ("<think>let me see</think> Ответ", "размышление вместо ответа"),
        ("The inner product is defined via a symmetric matrix.", "ответ не по-русски"),
    ):
        reward = compute_reward(answer, context=CONTEXT, reference=GOLD)
        assert reward.gate == reason
        assert reward.total == -1.0
    truncated = compute_reward("Скалярное произведение", context=CONTEXT, reference=GOLD, truncated=True)
    assert truncated.gate == "оборван пределом токенов"


def test_formula_heavy_russian_answer_passes_language_gate():
    answer = (
        "Ответ: $$\\langle x, y \\rangle = x^{T} A y$$ и $$\\|x\\| = \\sqrt{\\langle x, x \\rangle}$$, "
        "где A симметрична."
    )
    assert compute_reward(answer, context=CONTEXT, reference=GOLD).gate == ""


def test_refusal_depends_on_whether_gold_was_in_context():
    refusal = "В доступных материалах нет данных, чтобы ответить на этот вопрос."
    wrong = compute_reward(refusal, context=CONTEXT, reference=GOLD, gold_in_context=True)
    right = compute_reward(refusal, context=CONTEXT, reference=GOLD, gold_in_context=False)
    assert wrong.total < 0 < right.total


def test_reward_is_deterministic():
    answer = "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$ [1]."
    first = compute_reward(answer, context=CONTEXT, reference=GOLD).as_dict()
    second = compute_reward(answer, context=CONTEXT, reference=GOLD).as_dict()
    assert first == second


def test_random_control_is_reproducible_and_varies_within_group():
    answers = [f"ответ {i}" for i in range(16)]
    first = [random_reward(a, seed=7, key="q1") for a in answers]
    assert first == [random_reward(a, seed=7, key="q1") for a in answers]
    assert len(set(first)) == 2


def test_format_control_ignores_content():
    assert format_reward("Совершенно неверный, но русский ответ.") == 1.0
    assert format_reward("") == 0.0


def test_grouping_braces_do_not_matter():
    assert canonical_tokens(r"f\colon {\mathbb R}^n \to {\mathbb R}^m") == canonical_tokens(
        r"f: \mathbb{R}^{n} \rightarrow \mathbb{R}^{m}"
    )


def test_superscript_grouping_is_kept():
    assert canonical_tokens(r"x^{ab}") != canonical_tokens(r"x^a b")


def test_restatement_in_own_notation_is_not_foreign():
    context = r"Отображение $$f : \mathbb{R}^n \rightarrow \mathbb{R}^m$$ задаётся функциями $$f_i : \mathbb{R}^n \rightarrow \mathbb{R}$$."
    answer = r"Здесь \(f_1, \dots, f_m\) — компоненты отображения \(f\colon \mathbb R^n\to\mathbb R^m\)."
    assert score_formulas("", answer, context).foreign == 0


def test_math_in_parentheses_does_not_count_as_latin():
    answer = r"Матрица \(\mathbf{A}^\top \mathbf{A} + \lambda \mathbf{I}\) обратима при \(\lambda > 0\)."
    assert compute_reward(answer, context=CONTEXT, reference=GOLD).gate == ""


def test_repetition_loop_is_penalized():
    loop = "Ответ: " + " ".join(["$$A A^{T}$$ $$A^{T} A$$"] * 10)
    normal = "Ответ: матрицы $$A A^{T}$$ и $$A^{T} A$$ симметричны, их собственные значения неотрицательны."
    looped = compute_reward(loop, context=CONTEXT, reference=GOLD)
    assert looped.parts.get("repetition", 0) < 0
    assert looped.total < compute_reward(normal, context=CONTEXT, reference=GOLD).total
