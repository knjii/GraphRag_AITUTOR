"""Сравнение двух сохранённых прогонов.

Нужно там, где `eval ab` бессилен: настройки, влияющие на индекс, нельзя
переключить на лету, две конфигурации не существуют одновременно. Порог
отсечения хабов — ровно такой случай.

Проверяется главное свойство: сравнение идёт **по одним и тем же вопросам**.
Прогоны легко снять на разных наборах, и тогда молчаливое сравнение чего попало
дало бы правдоподобное, но бессмысленное число.
"""

from __future__ import annotations

import pytest

from rag_textbook.evaluation.metrics import QueryOutcome, compare_paired
from rag_textbook.evaluation.runner import load_outcomes, save_evaluation
from rag_textbook.evaluation.metrics import RetrievalMetrics


def _outcome(index: int, *, found: bool, question_type: str = "graph_linked") -> QueryOutcome:
    gold = f"gold-{index}"
    return QueryOutcome(
        question_id=f"q-{index}",
        question_type=question_type,
        retrieved=[gold, "other"] if found else ["other", "noise"],
        relevant=[gold],
        used_graph=found,
        graph_share=0.5 if found else 0.0,
        graph_only_share=0.25 if found else 0.0,
        latency_ms=100.0,
    )


def _save(settings, label: str, outcomes):
    metrics = RetrievalMetrics(k_values=(8,), questions=len(outcomes))
    return save_evaluation(metrics, outcomes, settings, label=label)


def test_saved_run_reads_back_unchanged(settings):
    outcomes = [_outcome(i, found=i % 2 == 0) for i in range(6)]
    path = _save(settings, "точка-отсчёта", outcomes)

    label, restored = load_outcomes(path)

    assert label == "точка-отсчёта"
    assert len(restored) == len(outcomes)
    assert restored[0].question_id == outcomes[0].question_id
    assert restored[0].retrieved == outcomes[0].retrieved
    assert restored[0].question_type == outcomes[0].question_type
    assert restored[1].used_graph == outcomes[1].used_graph


def test_comparison_finds_improvement(settings):
    before = [_outcome(i, found=False) for i in range(10)]
    after = [_outcome(i, found=i < 6) for i in range(10)]
    base_path = _save(settings, "порог-64", before)
    cand_path = _save(settings, "порог-40", after)

    _, base = load_outcomes(base_path)
    _, cand = load_outcomes(cand_path)
    paired = compare_paired(base, cand, 8)

    assert paired["questions"] == 10
    assert paired["metrics"]["recall"]["delta"] == pytest.approx(0.6)
    assert paired["metrics"]["recall"]["improved"] == 6
    assert paired["metrics"]["recall"]["worsened"] == 0
    assert paired["metrics"]["recall"]["significant"] is True


def test_runs_on_different_goldsets_share_nothing(settings):
    """Разные наборы вопросов сравнивать нельзя, и это должно быть видно."""
    left = [_outcome(i, found=True) for i in range(5)]
    right = [_outcome(i, found=True) for i in range(100, 105)]

    _, base = load_outcomes(_save(settings, "набор-А", left))
    _, cand = load_outcomes(_save(settings, "набор-Б", right))
    paired = compare_paired(base, cand, 8)

    assert paired["questions"] == 0


def test_breakdown_by_type_is_present(settings):
    """Разбор по типам — то, чего не хватало при решении по реранкеру."""
    before = [
        _outcome(i, found=True, question_type="formula_table") for i in range(4)
    ] + [_outcome(10 + i, found=True, question_type="graph_linked") for i in range(4)]
    after = [
        _outcome(i, found=True, question_type="formula_table") for i in range(4)
    ] + [_outcome(10 + i, found=False, question_type="graph_linked") for i in range(4)]

    _, base = load_outcomes(_save(settings, "до", before))
    _, cand = load_outcomes(_save(settings, "после", after))
    paired = compare_paired(base, cand, 8)

    by_type = paired["by_type"]
    assert set(by_type) == {"formula_table", "graph_linked"}
    # Среднее показало бы падение вдвое, а разбор — что пострадал один тип.
    assert by_type["formula_table"]["metrics"]["recall"]["delta"] == pytest.approx(0.0)
    assert by_type["graph_linked"]["metrics"]["recall"]["delta"] == pytest.approx(-1.0)
