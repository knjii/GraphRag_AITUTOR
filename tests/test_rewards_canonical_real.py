"""Каноническая форма на парах из реальных данных.

Пары взяты из эталонных фрагментов учебника (после разбора MinerU) и ответов
моделей в замере 2026-09-03 (``capture/session-0903/answers_model-*``):
сравнивались формулы ответа с формулами эталона того же вопроса.
Первая группа — та же формула в другой записи (должна сводиться),
вторая — разные формулы с высоким сходством (не должна).
"""

from __future__ import annotations

import pytest

from rag_textbook.rewards import canonical_tokens, extract_math

SAME = [
    pytest.param(
        r"z _ { n } \sim \mathcal { N } ( z \mid \mathbf { 0 } , I ) ;\tag{10.65}",
        r"z _ { n } \sim \mathcal {N } ( z \mid \mathbf { 0 } , I )",
        id="номер-формулы-и-пунктуация",
    ),
    pytest.param(
        r"\pmb { x } \in \mathbb { R } ^ { D }",
        r"\boldsymbol { x } \in \mathbb { R } ^ { D }",
        id="pmb-и-boldsymbol",
    ),
    pytest.param(
        r"A = { \left[ \begin{array} { l l l } { 3 } & { 2 } & { 2 } \\ { 2 } & { 3 } & { - 2 } \end{array} \right] } .",
        r"A = \begin{bmatrix} 3 & 2 & 2 \\ 2 & 3 & -2 \end{bmatrix}",
        id="array-и-bmatrix",
    ),
    pytest.param(
        r"A = { \left[ \begin{array} { l l l l } { 0 } & { 1 } \end{array} \right] }",
        r"A = \left[ \begin{array} { l l } 0 & 1 \end{array} \right]",
        id="спецификация-столбцов",
    ),
    pytest.param(
        r"\operatorname* { m i n } _ { A x = y } f ( { \pmb y } ) + g ( { \pmb x } )",
        r"\min_{Ax=y} f(\mathbf{y}) + g(\mathbf{x})",
        id="min-буквами-и-командой",
    ),
    pytest.param(
        r"\mathbf { \pmb { a } } \pmb { b } ^ { \mathrm { T } } \in \mathbb { R } ^ { n \times n }",
        r"\mathbf{a} \mathbf{b}^{\mathrm{T}} \in \mathbb{R}^{n \times n}",
        id="двойной-жирный",
    ),
    pytest.param(
        r"\pmb { A } ^ { \mathrm { T } } \pmb { A } = \pmb { A } \pmb { A } ^ { \mathrm { T } } = \pmb { I } ,",
        r"A ^ { \mathrm { T } } \pmb { A } = \pmb { A } \pmb { A } ^ { \mathrm { T } } = \pmb { I }",
        id="жирный-потерян-моделью",
    ),
    pytest.param(
        r"\langle \pmb { x } , \pmb { y } \rangle = \hat { \pmb { x } } ^ { \mathrm { T } } \pmb { A } \hat { \pmb { y } } .\tag{3.15}",
        r"\langle \pmb{x}, \pmb{y} \rangle = \hat{\pmb{x}}^T \pmb{A} \hat{\pmb{y}}",
        id="транспонирование-T-и-mathrm-T",
    ),
    pytest.param(
        r"{ \bf R } ( \mathbb { V } | M )",
        r"\mathbf { R } ( \mathbb { V } | M )",
        id="bf-и-mathbf",
    ),
    pytest.param(
        r"\Phi _ { k } \colon \mathbb { R } ^ { D } \to \mathbb { R } ^ { K } -",
        r"\Phi _ { k } \colon \mathbb { R } ^ { D } \to \mathbb { R } ^ { K }",
        id="висящий-минус-разбора",
    ),
    pytest.param(
        r"\theta _ { \mathrm { M L } } = ( \Phi ^ { \mathrm { T } } \Phi ) ^ { - 1 } \Phi ^ { \mathrm { T } } \pmb { y }\tag{9.19}",
        r"\theta_{ML} = (\Phi^\top \Phi)^{-1} \Phi^\top y",
        id="мнк-оценка",
    ),
    pytest.param(
        r"( \pmb { b } _ { 1 } , . . . , \pmb { b } _ { n } )",
        r"(\mathbf{b}_1, \ldots, \mathbf{b}_n)",
        id="многоточие-точками-и-командой",
    ),
    pytest.param(
        r"p ( { \boldsymbol { x } } | { \boldsymbol { \theta } } )",
        r"p(\mathbf{x} \mid \boldsymbol{\theta})",
        id="mid-и-черта",
    ),
    pytest.param(
        r"< \pi _ { \mathit { U } } ( { \pmb x } ) - { \pmb x } , { \pmb b } > = 0",
        r"\langle \pi _ { \mathit { U } } ( { \pmb x } ) - { \pmb x } , { \pmb b } \rangle = 0",
        id="угловые-скобки-знаками-сравнения",
        marks=pytest.mark.xfail(
            strict=True,
            reason="«<» разбора и \\langle не сводятся: без контекста «<» — знак сравнения",
        ),
    ),
    pytest.param(
        r"\underset { w , b , \xi } { \mathrm { m i n } } \frac 1 2 \| w \| ^ { 2 }",
        r"\min_{w, b, \xi} \frac{1}{2} \|w\|^2",
        id="underset-и-нижний-индекс",
        marks=pytest.mark.xfail(
            strict=True,
            reason="\\underset{под}{над} меняет порядок аргументов относительно _{под}",
        ),
    ),
]

DIFFERENT = [
    pytest.param(r"{ \pmb A } = { \pmb L } { \pmb U } ^ { \mathrm { T } }",
                 r"{ \pmb A } = { \pmb L } { \pmb L } ^ { \mathrm { T } }", id="LU-и-LL"),
    pytest.param(r"x _ { 0 } = 0", r"x_0 = 1", id="разные-числа"),
    pytest.param(r"f : \mathbb { R } ^ { n } \to \mathbb { R }",
                 r"f : \mathbb { R } ^ { n } \to \mathbb { R } ^ { m }", id="разная-размерность"),
    pytest.param(r"( \pmb { b } _ { 1 } , \pmb { b } _ { 2 } )",
                 r"( \pmb { c } _ { 1 } , \pmb { c } _ { 2 } )", id="разные-базисы"),
    pytest.param(r"\Omega ( x , \lambda y + \Psi z ) = \lambda \Omega ( x , y ) + \Psi \Omega ( x , z )",
                 r"\Omega ( \lambda x + \Psi y , z ) = \lambda \Omega ( x ,z ) + \Psi \Omega ( y ,z )",
                 id="линейность-по-разным-аргументам"),
    pytest.param(r"p ( { \boldsymbol { x } } )", r"p(\boldsymbol{y})", id="разные-переменные"),
    pytest.param(r"\mathcal { N } ( 0 , \pmb { I } )", r"\mathcal { N } ( 0 , \Sigma )",
                 id="разная-ковариация"),
    pytest.param(r"\frac { a b } { c }", r"\frac { a } { b c }", id="дробь-разные-группы",
                 marks=pytest.mark.xfail(
                     strict=True,
                     reason="группирующие скобки сняты — цена за устойчивость к разбору",
                 )),
    pytest.param(r"\sqrt { x + 1 }", r"\sqrt x + 1", id="корень-разные-группы",
                 marks=pytest.mark.xfail(strict=True, reason="та же причина, что у дроби")),
    pytest.param(r"x ^ { T } A y", r"x ^ { t } A y", id="T-и-t"),
]


@pytest.mark.parametrize(("left", "right"), SAME)
def test_same_formula_different_spelling(left, right):
    assert canonical_tokens(left) == canonical_tokens(right)


@pytest.mark.parametrize(("left", "right"), DIFFERENT)
def test_different_formulas_stay_different(left, right):
    assert canonical_tokens(left) != canonical_tokens(right)


def test_quadruple_dollars_from_chunker():
    """Фрагменты учебника несут «$$$$ формула $$$$» (разметка обёрнута дважды)."""
    text = r"получим распределение $$$$ p ( { \pmb x } | { \boldsymbol \theta } , z ) ,\tag{8.24} $$$$ 348 Глава"
    [formula] = extract_math(text)
    assert "$" not in formula
    assert canonical_tokens(formula) == canonical_tokens(r"p(\mathbf{x} \mid \boldsymbol\theta, z)")


def test_chunk_starting_inside_a_formula():
    """Фрагмент начинается с хвоста формулы: первый «$$» — закрывающий."""
    text = (
        r"\| y \| } = - \frac { 1 } { 3 } \tag{3.28} $$ и x и y не ортогональны. "
        r"Матрица ортогональна, так что $$ A A ^ { T } = I \tag{3.29} $$ что подразумевает "
        r"$$ A ^ { - 1 } = A ^ { T } \tag{3.30} $$ то есть обратная получается транспонированием."
    )
    formulas = extract_math(text)
    assert [f for f in formulas if "tag" in f] == [
        r" A A ^ { T } = I \tag{3.29} ", r" A ^ { - 1 } = A ^ { T } \tag{3.30} ",
    ]


def test_adjacent_formulas_after_a_cut():
    """Соседние формулы без прозы между ними: сдвиг виден только по всему фрагменту."""
    text = (
        r"a _ { 1 } = 2 \tag{1} $$ $$ x _ { 1 } - x _ { 2 } = 2 \tag{2} $$ (2.6) "
        r"$$ 2 x _ { 1 } + 3 x _ { 3 } = 5 \tag{3} $$ Поскольку сумма первых двух равна третьему"
    )
    tags = [f.strip()[-7:] for f in extract_math(text)]
    assert tags == [r"\tag{2}", r"\tag{3}"]


def test_russian_words_inside_text_command_are_a_formula():
    text = r"Итак, $$ f ( x ) = 0 \quad \text{для всех} \ x $$ и далее."
    assert extract_math(text) == [r" f ( x ) = 0 \quad \text{для всех} \ x "]


def test_equation_sides_swapped_still_carried():
    """Ответ 11B ручной сверки: «Σ = AAᵀ» при эталоне «AAᵀ = Σ»."""
    from rag_textbook.rewards import score_formulas

    gold = r"где $$ \pmb { A } \pmb { A } ^ { \mathrm { T } } = \Sigma $$ и далее"
    answer = r"Используем разложение $\Sigma = AA^{\mathrm{T}}$, матрица треугольна."
    assert score_formulas(gold, answer, gold).carried == 1


def test_chain_is_not_flipped():
    from rag_textbook.rewards import score_formulas

    gold = r"$$ a + b = c + d = e + f $$"
    answer = r"$$ e + f = c + d = a + b $$"
    assert score_formulas(gold, answer, gold).carried == 0
