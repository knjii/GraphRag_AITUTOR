"""Проверки массы и переходов на графах с известным решением."""

import math

import pytest

from rag_textbook.evaluation.graph_offline import OfflineGraph, PPRGraph, rank_ppr
from scripts.ppr_offline import compare, normalized, question_metrics, question_seeds


@pytest.fixture
def graph() -> OfflineGraph:
    return OfflineGraph(
        mentions={"a": {"x": 1}, "b": {"x": 1}, "c": {"y": 1}, "d": {"z": 1}},
        neighbours={"x": {"y"}, "y": {"x"}},
        idf={"x": 0.0, "y": 1.0, "z": 1.0},
    )


def test_mass_and_known_solution() -> None:
    ppr = PPRGraph(OfflineGraph(mentions={"a": {"x": 1}}))
    result = ppr.probabilities({("passage", "a"): 1})
    assert sum(result.values()) == pytest.approx(1, abs=1e-10)
    assert result[("passage", "a")] == pytest.approx(2 / 3)
    assert result[("entity", "x")] == pytest.approx(1 / 3)


def test_reachability_and_exclusion(graph: OfflineGraph) -> None:
    seeds = {("passage", "a"): 1.0}
    result = dict(rank_ppr(graph, seeds))
    assert set(result) == {"b", "c"}
    assert result["b"] > result["c"] > 0
    assert set(dict(rank_ppr(graph, seeds, entity_weight=0))) == {"b"}
    assert sum(result.values()) < 1


def test_alpha_and_convergence(graph: OfflineGraph) -> None:
    ppr = PPRGraph(graph)
    seeds = {("passage", "a"): 1.0}
    low = ppr.probabilities(seeds, alpha=0.3)
    high = ppr.probabilities(seeds, alpha=0.85)
    assert high[("passage", "a")] > low[("passage", "a")]
    assert high[("passage", "c")] < low[("passage", "c")]
    strict = ppr.probabilities(seeds, alpha=0.3, tolerance=1e-13)
    assert sum(abs(low[n] - strict[n]) for n in low) < 1e-9
    with pytest.raises(RuntimeError, match="не сошёлся"):
        ppr.probabilities(seeds, max_iterations=1)
    assert ppr.rank(seeds, alpha=1) == []


def test_dangling_and_typed_ids() -> None:
    ppr = PPRGraph(OfflineGraph(mentions={"same": {"same": 1}, "alone": {}}))
    seeds = {("passage", "same"): 2, ("passage", "alone"): 1, ("entity", "missing"): 1}
    result = ppr.probabilities(seeds)
    assert sum(result.values()) == pytest.approx(1)
    assert result[("passage", "alone")] > 0
    assert result[("entity", "missing")] > 0
    assert result[("passage", "same")] > result[("entity", "same")] > 0
    scaled = ppr.probabilities({n: w * 5 for n, w in seeds.items()})
    assert scaled == pytest.approx(result)


def test_idf_removes_hub(graph: OfflineGraph) -> None:
    assert rank_ppr(graph, {("passage", "a"): 1}, use_idf=True) == []


@pytest.mark.parametrize("alpha", [0, -1, 1.1, math.nan])
def test_invalid_alpha(graph: OfflineGraph, alpha: float) -> None:
    with pytest.raises(ValueError):
        rank_ppr(graph, {("passage", "a"): 1}, alpha=alpha)


@pytest.mark.parametrize("seeds", [{}, {("passage", "a"): 0}, {("passage", "a"): -1}, {("passage", "a"): math.inf}])
def test_invalid_seeds(graph: OfflineGraph, seeds: dict[tuple[str, str], float]) -> None:
    with pytest.raises(ValueError):
        rank_ppr(graph, seeds)


def test_mention_counts() -> None:
    graph = OfflineGraph(mentions={"a": {"x": 1}, "b": {"x": 3}})
    probabilities = PPRGraph(graph).probabilities({("entity", "x"): 1})
    assert probabilities[("passage", "b")] / probabilities[("passage", "a")] == pytest.approx(2)


def test_question_protocol() -> None:
    row = {
        "question_id": "q", "question_type": "multi_hop", "question": "Методы матриц",
        "channels": {"base": [
            {"rank": 1, "chunk_id": "b", "score": 0.1},
            {"rank": 0, "chunk_id": "a", "score": 0.5},
        ]},
    }
    seeds = question_seeds(row, {"x": normalized("матриц"), "y": normalized("мет")}, 1)
    assert seeds == {("passage", "a"): 0.5, ("entity", "x"): 1.0}
    outcome = question_metrics([row], {"q": ["b", "c"]}, {"q": {"a", "b", "c"}})[0]
    assert outcome["recall@30"] == pytest.approx(2 / 3)
    assert outcome["graph_only_found"] == 1
    assert outcome["graph_only_total"] == 1


def test_channel_cutoff_and_pair_alignment() -> None:
    row = {"question_id": "q", "question_type": "single_chunk", "channels": {"base": []}}
    outcome = question_metrics([row], {"q": [str(i) for i in range(31)]}, {"q": {"30"}})[0]
    assert outcome["recall@30"] == 0
    assert outcome["graph_only_found"] == 0
    assert outcome["graph_only_total"] == 1
    with pytest.raises(ValueError, match="одинакового порядка"):
        compare([outcome], [{**outcome, "question_id": "other"}])


def test_relation_weight_not_doubled() -> None:
    one = OfflineGraph(mentions={"a": {"x": 1}, "b": {"y": 1}}, neighbours={"x": {"y"}})
    both = OfflineGraph(mentions=one.mentions, neighbours={"x": {"y"}, "y": {"x"}})
    seeds = {("passage", "a"): 1}
    assert PPRGraph(one).probabilities(seeds) == PPRGraph(both).probabilities(seeds)


def test_complete_variant(graph: OfflineGraph) -> None:
    from scripts.ppr_offline import evaluate_variant

    rows = [{"question_id": "q", "question_type": "multi_hop", "channels": {"base": []}}]
    gold = {"q": {"b"}}
    baseline = question_metrics(rows, {"q": []}, gold)
    result = evaluate_variant(
        graph, [("a", "b")], rows, gold, {"q": {("passage", "a"): 1.0}},
        baseline, 1.0, 0.5, False,
    )
    assert result["hops"]["measurements"] == 2
    assert result["questions"]["all"]["graph_only_found"] == 1
    assert result["paired_difference"]["recall@30"]["mean"] == 1.0
