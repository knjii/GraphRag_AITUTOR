"""Парное сравнение конфигураций.

Прежний критерий значимости считал выборки независимыми, хотя обе конфигурации
оцениваются на одном и том же наборе вопросов. Вопросы различаются по трудности
сильнее, чем конфигурации между собой, поэтому независимый критерий списывал
почти весь реальный эффект в шум: на 140 вопросах он не увидел бы прироста
меньше 8 процентных пунктов.
"""

from __future__ import annotations

from rag_textbook.evaluation.metrics import (
    QueryOutcome,
    compare,
    compare_paired,
    evaluate_retrieval,
)


def _outcomes(hits: list[bool], prefix: str = "q") -> list[QueryOutcome]:
    """Строит исходы, где попадание означает эталонный фрагмент на первом месте."""
    result = []
    for index, hit in enumerate(hits):
        gold = f"gold-{index}"
        retrieved = [gold, "other-1", "other-2"] if hit else ["other-1", "other-2", "other-3"]
        result.append(
            QueryOutcome(
                question_id=f"{prefix}-{index}",
                question_type="single_chunk",
                retrieved=retrieved,
                relevant=[gold],
            )
        )
    return result


def test_paired_detects_consistent_small_improvement() -> None:
    """Небольшой, но устойчивый прирост должен признаваться значимым.

    Сценарий: 120 вопросов, кандидат чинит 12 из них и не ломает ни одного.
    Прирост 10 процентных пунктов при нулевом разбросе знака различий —
    это ровно тот случай, ради которого нужен парный критерий.
    """
    base_hits = [True] * 60 + [False] * 60
    cand_hits = [True] * 72 + [False] * 48

    paired = compare_paired(_outcomes(base_hits), _outcomes(cand_hits), k=3)
    recall = paired["metrics"]["recall"]

    assert paired["questions"] == 120
    assert recall["improved"] == 12
    assert recall["worsened"] == 0
    assert recall["delta"] > 0
    assert recall["significant"], "устойчивый прирост без единой потери обязан быть значимым"
    assert recall["ci_low"] > 0


def test_paired_is_more_sensitive_than_independent() -> None:
    """Прямое сравнение двух критериев на одних данных.

    Эффект намеренно взят маленьким: кандидат чинит 6 вопросов из 120, то есть
    5 процентных пунктов. Независимый критерий на такой выборке не видит
    ничего меньше 8.9 пункта — а именно в этот диапазон попадает типичный
    выигрыш от графового канала, ради измерения которого всё и затевалось.
    """
    base_hits = [True] * 60 + [False] * 60
    cand_hits = [True] * 66 + [False] * 54
    base_outcomes = _outcomes(base_hits)
    cand_outcomes = _outcomes(cand_hits)

    independent = compare(
        evaluate_retrieval(base_outcomes, k_values=(3,)),
        evaluate_retrieval(cand_outcomes, k_values=(3,)),
        k=3,
    )
    paired = compare_paired(base_outcomes, cand_outcomes, k=3)

    assert not independent["likely_significant"]["recall"], (
        "независимый критерий этот эффект не видит — в этом и была проблема"
    )
    assert paired["metrics"]["recall"]["significant"]


def test_paired_reports_no_effect_when_there_is_none() -> None:
    """Критерий не должен находить эффект там, где его нет."""
    hits = [True] * 60 + [False] * 60
    paired = compare_paired(_outcomes(hits), _outcomes(hits), k=3)
    recall = paired["metrics"]["recall"]

    assert recall["delta"] == 0.0
    assert recall["improved"] == 0
    assert recall["worsened"] == 0
    assert not recall["significant"]


def test_paired_counts_wins_and_losses_separately() -> None:
    """Прирост из «+12/−9» и из «+3/−0» — разные ситуации, и это должно быть видно."""
    base_hits = [True] * 30 + [False] * 30
    # Кандидат чинит первые 10 из провальных и ломает 7 из успешных.
    cand_hits = [False] * 7 + [True] * 23 + [True] * 10 + [False] * 20

    paired = compare_paired(_outcomes(base_hits), _outcomes(cand_hits), k=3)
    recall = paired["metrics"]["recall"]

    assert recall["improved"] == 10
    assert recall["worsened"] == 7
    assert recall["unchanged"] == 43


def test_paired_ignores_questions_missing_on_one_side() -> None:
    """Сравнивать можно только вопросы, пройденные обеими конфигурациями."""
    base_outcomes = _outcomes([True] * 10)
    cand_outcomes = _outcomes([True] * 6)

    paired = compare_paired(base_outcomes, cand_outcomes, k=3)
    assert paired["questions"] == 6
