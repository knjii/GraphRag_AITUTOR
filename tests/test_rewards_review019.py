"""Эксплойты из независимого ревью обучения (задача 019).

Каждый был воспроизведён запуском на прежней награде; тест фиксирует,
что он больше не окупается. Сравнение, где можно, — с честным ответом
на тот же вопрос: GRPO видит порядок внутри группы, а не абсолют.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from rag_textbook.rewards import compute_reward, score_formulas
from rag_textbook.rewards.composite import RewardConfig, format_reward

Q = "Как определяется скалярное произведение векторов?"
G = "Скалярное произведение векторов определяется формулой $$s=x_1y_1+x_2y_2$$."

_spec = importlib.util.spec_from_file_location(
    "train_grpo", Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
)
train_grpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_grpo)


def reward(
    answer: str,
    *,
    context: str = G,
    reference: str = G,
    question: str = Q,
    gold_in_context: bool = True,
) -> float:
    return compute_reward(
        answer,
        context=context,
        reference=reference,
        question=question,
        gold_in_context=gold_in_context,
    ).total


# --- P1: неверная математика с полным баллом ---------------------------------


def test_wrong_fraction_is_not_carried():
    """Эксплойт C: \\frac{a}{bc} вместо \\frac{ab}{c} получал 1.0."""
    gold = r"$$y=\frac{ab}{c}$$"
    assert score_formulas(gold, r"$$y=\frac{a}{bc}$$", gold).carried == 0
    assert score_formulas(gold, r"$$y=\dfrac{ab}{c}$$", gold).carried == 1
    assert reward(r"$$y=\frac{a}{bc}$$", context=gold, reference=gold, question="") < reward(
        r"$$y=\frac{ab}{c}$$", context=gold, reference=gold, question=""
    )


def test_reshaped_matrix_is_not_carried():
    gold = r"$$A=\begin{bmatrix}a&b\\c&d\end{bmatrix}$$"
    assert score_formulas(gold, r"$$A=\begin{bmatrix}a&b&c&d\end{bmatrix}$$", gold).carried == 0
    assert (
        score_formulas(gold, r"$$A=\begin{bmatrix} a & b \\ c & d \end{bmatrix}$$", gold).carried
        == 1
    )


# --- P1: невидимый эталон ------------------------------------------------------


def test_formula_from_memory_does_not_beat_refusal():
    """Эксплойт D: невидимая формула +1.0 против отказа +0.5."""
    context = "Векторы описывают направление движения."
    guessed = reward("$$s=x_1y_1+x_2y_2$$", context=context, gold_in_context=False)
    refused = reward("Недостаточно информации для ответа.", context=context, gold_in_context=False)
    assert refused == 0.5
    assert guessed < refused


def test_truncated_context_does_not_expect_hidden_formula():
    """Эталон целиком, контекст усечён: формула за границей не ожидается."""
    reference = G + " Далее $$u=v+w_1+w_2$$."
    score = score_formulas(reference, "$$s=x_1y_1+x_2y_2$$", G)
    assert score.expected == 1 and score.carried == 1


def test_other_refusal_wording_is_recognized():
    context = "Векторы описывают направление движения."
    assert (
        reward("В контексте нет информации для ответа.", context=context, gold_in_context=False)
        == 0.5
    )


# --- P1: вопрос вместо ответа, склейка с выдумкой ------------------------------


def test_repeating_the_question_earns_nothing():
    """Эксплойт B: повтор вопроса получал 0.3 за опору."""
    assert reward(Q) <= 0.0
    assert reward(Q) < reward(G)


def test_glued_formulas_do_not_hide_an_invented_tail():
    """Склейка двух эталонных формул с выдуманным хвостом получала 1.0."""
    reference = context = "Первое: $$a+b=c$$ и второе: $$d+e=f$$."
    glued = r"$$a+b=c\quad d+e=f\quad \zeta\omega\psi\theta=\alpha\beta\gamma$$"
    score = score_formulas(reference, glued, context)
    assert score.foreign == 1
    honest = "Это $$a+b=c$$ и $$d+e=f$$."
    assert reward(glued, context=context, reference=reference, question="") < reward(
        honest, context=context, reference=reference, question=""
    )


# --- P2: отказы ----------------------------------------------------------------

LONG_REFUSAL = (
    "Недостаточно информации для ответа. Для точного ответа мне потребуется "
    "дополнительный материал из учебника. Пожалуйста, уточните тему или предоставьте "
    "соответствующий раздел с определениями и пояснениями. Без исходных предпосылок я "
    "не могу обоснованно выбрать нужное выражение и объяснить его смысл. Возможно, "
    "полезно проверить обозначения переменных и условия применимости результата. После "
    "получения этих сведений можно будет составить подробное решение."
)


def test_long_refusal_keeps_the_refusal_penalty():
    """Эксплойт E: вежливое продолжение снимало штраф −0.5."""
    assert len(LONG_REFUSAL) > RewardConfig().max_refusal_chars
    assert reward(LONG_REFUSAL) == reward("Недостаточно информации для ответа.") == -0.5


def test_caveat_inside_an_answer_with_formulas_is_not_a_refusal():
    answer = "Нет данных о знаке, но по определению $$s=x_1y_1+x_2y_2$$."
    assert reward(answer) > 0.5


# --- шкала ---------------------------------------------------------------------


def test_passing_the_gates_is_never_worse_than_failing_them():
    """Плохой ответ (до −2.2) был хуже пустого (−1): ворота выгодно провалить."""
    context = reference = "$$a+b=c$$"
    awful = r"$$\zeta\omega=\psi\theta$$" + "яa" * 1750
    assert reward(awful, context=context, reference=reference, question="") >= -1.0
    assert reward("", context=context, reference=reference) == -1.0


def test_copied_share_is_reported_but_not_scored():
    context = G + " Норма вектора вычисляется по формуле $$n=x_1^2+x_2^2$$."
    breakdown = compute_reward(context, context=context, reference=G, question=Q)
    assert breakdown.diagnostics["copied"] == 1.0
    assert "copied" not in breakdown.parts


# --- P2: обрыв и контроль формата ----------------------------------------------

EOS = 151645


def test_answer_ending_with_eos_at_the_limit_is_not_truncated():
    assert not train_grpo.is_truncated([100, 101, EOS], 3, {EOS})
    assert train_grpo.is_truncated([100, 101, 102], 3, {EOS})
    assert not train_grpo.is_truncated([100, EOS], 3, {EOS})
    # Без известных концов прежнее правило: достиг предела — оборван.
    assert train_grpo.is_truncated([100, 101, EOS], 3, None)


def test_main_and_format_share_the_truncation_gate(tmp_path):
    for kind, clean in (("main", None), ("format", 1.0)):
        fn = train_grpo.build_reward(
            kind, seed=1, max_completion_tokens=3, samples_path=None, sample_every=1, stop_ids={EOS}
        )
        kwargs = dict(context=["$$a+b=c$$"], reference=["$$a+b=c$$"], gold_in_context=[True])
        [ended] = fn(completions=["Ответ: $$a+b=c$$."], completion_ids=[[1, 2, EOS]], **kwargs)
        [cut] = fn(completions=["Ответ: $$a+b=c$$."], completion_ids=[[1, 2, 3]], **kwargs)
        assert cut < ended, kind
        if clean is not None:
            assert ended == clean
    assert format_reward("Ответ.", truncated=True) == 0.0


# --- P2: критические поля TRL ---------------------------------------------------


def test_missing_critical_trl_field_is_refused():
    import dataclasses

    @dataclasses.dataclass
    class NoLossType:
        max_steps: int = 0

    with pytest.raises(SystemExit, match="loss_type"):
        train_grpo._supported(NoLossType, {"max_steps": 1, "loss_type": "dr_grpo"})

    @dataclasses.dataclass
    class NoVllmMode:
        use_vllm: bool = False

    with pytest.raises(SystemExit, match="vllm_mode"):
        train_grpo._supported(NoVllmMode, {"use_vllm": True, "vllm_mode": "server"})
    # Без vLLM режим движка не нужен — не отказ.
    assert train_grpo._supported(NoVllmMode, {"use_vllm": False, "vllm_mode": "colocate"}) == {
        "use_vllm": False
    }


# --- поправки, которых потребовала ручная сверка --------------------------------


def test_short_answer_that_restates_the_question_keeps_its_support():
    """Ручная сверка, ответ 8A (оценка 3): вопрос плюс слово ответа — не повтор."""
    question = "Как называется первое ненулевое значение слева в строке ступенчатой матрицы?"
    context = (
        "Первое ненулевое значение слева в строке ступенчатой матрицы называется ведущим элементом."
    )
    answer = "Первое ненулевое значение слева в строке ступенчатой матрицы называется ведущим."
    assert reward(answer, context=context, reference=context, question=question) > 0


def test_context_formulas_are_not_offtarget_when_gold_is_invisible():
    """Ручная сверка, вопрос 6: честный ответ по контексту штрафовался как «лишний»."""
    formulas = [
        r"p_{ij}=\frac{n_{ij}}{N}",
        r"p(x_i)=\sum_{j} p_{ij}",
        r"p(y_j)=\sum_{i} p_{ij}",
        r"\sum_{i}\sum_{j} p_{ij}=1",
    ]
    context = "В таблице " + " ".join(f"$${f}$$." for f in formulas)
    answer = "По контексту: " + " ".join(f"$${f}$$." for f in formulas)
    result = compute_reward(
        answer, context=context, reference="$$q=r+s+t$$", question="", gold_in_context=False
    )
    assert "offtarget" not in result.parts
    refused = reward("Недостаточно информации для ответа.", context=context, gold_in_context=False)
    assert result.total < refused
