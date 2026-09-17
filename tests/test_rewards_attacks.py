"""Атаки на награду R6: плохой ответ не должен получать больше честного.

Каждый тест — один приём, которым политика могла бы набрать награду,
не отвечая лучше. Сравнение всегда с честным ответом на тот же вопрос,
а не с абсолютным порогом: GRPO учится на разнице внутри группы.
"""

from __future__ import annotations

from rag_textbook.rewards import compute_reward

QUESTION = "Как определяется ответственность компонента смеси за точку данных?"

GOLD = (
    "Ответственность $k$-го компонента за точку $x_n$ определяется как "
    "$$r_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n \\mid \\mu_k, \\Sigma_k)}"
    "{\\sum_{j=1}^{K} \\pi_j \\mathcal{N}(x_n \\mid \\mu_j, \\Sigma_j)}.$$ "
    "Сумма ответственностей по компонентам равна единице: "
    "$$\\sum_{k=1}^{K} r_{nk} = 1.$$"
)

OTHER_FORMULAS = [
    r"\mu_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} x_n",
    r"N_k = \sum_{n=1}^{N} r_{nk}",
    r"\pi_k = \frac{N_k}{N}",
    r"\Sigma_k = \frac{1}{N_k} \sum_{n=1}^{N} r_{nk} (x_n - \mu_k)(x_n - \mu_k)^\top",
    r"\log p(X \mid \theta) = \sum_{n=1}^{N} \log \sum_{k=1}^{K} \pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)",
    r"p(x) = \sum_{k=1}^{K} \pi_k \mathcal{N}(x \mid \mu_k, \Sigma_k)",
    r"Q(\theta, \theta^{old}) = \sum_{n=1}^{N} \sum_{k=1}^{K} r_{nk} \log \pi_k",
    r"\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(X, Z \mid \theta)] + H(q)",
]

NEIGHBOURS = (
    "На шаге максимизации параметры пересчитываются по ответственностям. "
    "Среднее компонента равно $$" + OTHER_FORMULAS[0] + "$$, "
    "где эффективное число точек $$" + OTHER_FORMULAS[1] + "$$. "
    "Веса смеси обновляются как $$" + OTHER_FORMULAS[2] + "$$. "
    "Ковариация компонента вычисляется как $$" + OTHER_FORMULAS[3] + "$$. "
    "Логарифм правдоподобия модели равен $$" + OTHER_FORMULAS[4] + "$$. "
    "Плотность смеси гауссиан записывается как $$" + OTHER_FORMULAS[5] + "$$. "
    "Ожидаемое полное правдоподобие имеет вид $$" + OTHER_FORMULAS[6] + "$$. "
    "Нижняя граница правдоподобия задаётся выражением $$" + OTHER_FORMULAS[7] + "$$. "
    "Алгоритм повторяет шаги ожидания и максимизации до сходимости."
)
CONTEXT = GOLD + " " + NEIGHBOURS

HONEST = (
    "Ответственность компонента смеси за точку данных определяется "
    "формулой ниже. Это апостериорная "
    "вероятность того, что точка $x_n$ порождена $k$-м компонентом:\n"
    "$$r_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n \\mid \\mu_k, \\Sigma_k)}"
    "{\\sum_{j=1}^{K} \\pi_j \\mathcal{N}(x_n \\mid \\mu_j, \\Sigma_j)}$$\n"
    "Числитель — взвешенная плотность компонента, знаменатель нормирует "
    "ответственности, поэтому их сумма по компонентам равна единице: "
    "$$\\sum_{k=1}^{K} r_{nk} = 1.$$"
)


def reward(answer: str, *, gold_in_context: bool = True, **kwargs) -> float:
    return compute_reward(
        answer, context=CONTEXT, reference=GOLD, question=QUESTION,
        gold_in_context=gold_in_context, **kwargs,
    ).total


def test_honest_answer_is_rewarded():
    assert reward(HONEST) > 1.0


def test_dump_all_context_formulas_separately():
    """Приём 1: перечислить все формулы контекста — эталон среди них найдётся."""
    all_formulas = [
        r"r_{nk} = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}"
        r"{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}",
        r"\sum_{k=1}^{K} r_{nk} = 1",
        *OTHER_FORMULAS,
    ]
    dump = "Ответственность компонента смеси описывается формулами: " + " ".join(
        f"$${item}$$" for item in all_formulas
    )
    assert reward(dump) < reward(HONEST) - 0.3


def test_dump_all_context_formulas_as_one_block():
    """Приём 2: склеить формулы в одну — вхождение эталона засчитывается."""
    block = " ".join(
        [r"r_{nk} = \frac{\pi_k \mathcal{N}(x_n \mid \mu_k, \Sigma_k)}"
         r"{\sum_{j=1}^{K} \pi_j \mathcal{N}(x_n \mid \mu_j, \Sigma_j)}",
         r"\sum_{k=1}^{K} r_{nk} = 1", *OTHER_FORMULAS]
    )
    # Проза та же, что у честного ответа: разница — только склейка.
    dump = HONEST + f" $${block}$$"
    assert reward(dump) < reward(HONEST) - 0.3


def test_spray_a_few_formulas_is_not_better_than_precise():
    """Приём 3: добавить к верной формуле пять соседних «на всякий случай»."""
    spray = HONEST + " Также полезны формулы " + " ".join(
        f"$${item}$$" for item in OTHER_FORMULAS[:5]
    )
    assert reward(spray) < reward(HONEST)


def test_copy_unrelated_context_sentences():
    """Приём 4: переписать предложения контекста, не отвечая на вопрос."""
    copy = (
        "На шаге максимизации параметры пересчитываются по ответственностям. "
        "Алгоритм повторяет шаги ожидания и максимизации до сходимости. "
        "Веса смеси обновляются на каждом шаге максимизации параметров."
    )
    assert reward(copy) < 0.5 * reward(HONEST)


def test_repeat_formula_with_cosmetic_variation():
    """Приём 5: повтор одной формулы с разными пробелами и скобками."""
    variants = [
        r"\sum_{k=1}^{K} r_{nk} = 1",
        r"\sum_{k=1}^K r_{nk}=1",
        r"\sum_{k = 1}^{K} r_{n k} = 1",
        r"\sum\limits_{k=1}^{K} r_{nk} = 1",
        r"\displaystyle\sum_{k=1}^{K}r_{nk}=1",
    ]
    loop = HONEST + " " + " ".join(f"Итак, $${item}$$" for item in variants * 2)
    result = compute_reward(loop, context=CONTEXT, reference=GOLD, question=QUESTION)
    assert "repetition" in result.parts
    assert result.total < reward(HONEST)


def test_repeat_sentence_with_cosmetic_variation():
    """Приём 6: одно предложение с разной пунктуацией и регистром."""
    sentence = "сумма ответственностей по компонентам равна единице"
    loop = HONEST + " " + " ".join(
        f"{sentence.capitalize()}{end}" for end in [".", "!", " .", "…", ";."] * 2
    )
    assert reward(loop) < reward(HONEST)


def test_english_prose_below_gate_is_penalized():
    """Приём 7: писать наполовину по-английски, оставаясь под порогом ворот."""
    mixed = (
        "Ответственность компонента смеси за точку данных определяется "
        "формулой ниже, it is the posterior probability of component k: "
        "$$r_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n \\mid \\mu_k, \\Sigma_k)}"
        "{\\sum_{j=1}^{K} \\pi_j \\mathcal{N}(x_n \\mid \\mu_j, \\Sigma_j)}$$ "
        "Числитель — взвешенная плотность, знаменатель normalizes the sum: "
        "$$\\sum_{k=1}^{K} r_{nk} = 1.$$"
    )
    result = compute_reward(mixed, context=CONTEXT, reference=GOLD, question=QUESTION)
    assert not result.gate
    assert "latin" in result.parts
    assert result.total < reward(HONEST)


def test_refusal_phrase_inside_full_answer_is_not_a_refusal():
    """Приём 8: оговорка «нет данных» внутри полного ответа.

    Без исправления такой ответ получал фиксированные −0.5 при эталоне
    в контексте (хороший ответ наказан) и +0.5 без эталона.
    """
    hedged = HONEST + " Про выбор числа компонентов в контексте нет данных."
    result = compute_reward(hedged, context=CONTEXT, reference=GOLD, question=QUESTION)
    assert result.gate != "отказ"
    assert result.total > 1.0


def test_short_refusal_is_still_a_refusal():
    refusal = "В предоставленном контексте нет данных для ответа на этот вопрос."
    assert reward(refusal, gold_in_context=False) > 0
    assert reward(refusal, gold_in_context=True) < 0


def test_leaked_reasoning_tail_hits_the_gate():
    """Приём 9: размышление без открывающего тега, только с закрывающим."""
    leaked = "Нужно найти формулу ответственности в контексте.</think>\n" + HONEST
    assert compute_reward(leaked, context=CONTEXT, reference=GOLD).gate


def test_wrong_formula_close_to_gold_earns_little():
    """Приём 10: формула, похожая на эталон, но неверная (π_j вместо π_k)."""
    wrong = HONEST.replace(r"\frac{\pi_k", r"\frac{\pi_j").replace(r"= 1.$$", r"= K.$$")
    assert reward(wrong) < reward(HONEST) - 0.5


def test_fabricated_formula_outside_math_markup_is_not_free():
    """Приём 11: выдумка в блоке кода — вне разметки формул, вне штрафа.

    Защиты нет и она не нужна для награды: формула вне разметки не
    приносит и формульной части. Тест фиксирует, что приём не выгоднее
    честного ответа.
    """
    fabricated = HONEST + "\n```latex\nr_{nk} = \\sigma(w_k^\\top x_n + b_k)\n```"
    assert reward(fabricated) <= reward(HONEST)


def test_formulas_without_prose():
    """Приём 12: только формулы, почти без слов."""
    bare = (
        "Ответ: $$r_{nk} = \\frac{\\pi_k \\mathcal{N}(x_n \\mid \\mu_k, \\Sigma_k)}"
        "{\\sum_{j=1}^{K} \\pi_j \\mathcal{N}(x_n \\mid \\mu_j, \\Sigma_j)}$$ "
        "$$\\sum_{k=1}^{K} r_{nk} = 1$$"
    )
    assert reward(bare) < reward(HONEST)
