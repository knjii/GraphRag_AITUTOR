"""Полнота пары (full@k) и бюджет контекста — протокол серии К."""

from __future__ import annotations

from rag_textbook.evaluation.metrics import (
    QueryOutcome,
    compare_paired,
    evaluate_retrieval,
    full_at_k,
)


def test_full_counts_only_complete_pairs() -> None:
    assert full_at_k(["a", "b", "x"], ["a", "b"], 3) == 1.0
    assert full_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.0
    assert full_at_k(["a"], [], 1) == 0.0


def test_full_and_chars_in_metrics_and_paired_comparison() -> None:
    half = QueryOutcome("q", "graph_linked", ["a", "x"], ["a", "b"], context_chars=[100, 900])
    both = QueryOutcome("q", "graph_linked", ["a", "b"], ["a", "b"], context_chars=[100, 300])
    metrics = evaluate_retrieval([half], (2,))
    assert metrics.per_k[2]["recall"] == 0.5
    assert metrics.per_k[2]["full"] == 0.0
    assert metrics.per_k[2]["chars"] == 1000
    assert metrics.by_type["graph_linked"]["full"] == 0.0

    report = compare_paired([half], [both], 2)
    assert report["metrics"]["full"]["delta"] == 1.0
    assert report["metrics"]["chars"] == {"baseline": 1000.0, "candidate": 400.0}


def test_chars_absent_for_old_runs() -> None:
    old = QueryOutcome("q", "single_chunk", ["a"], ["a"])
    assert "chars" not in evaluate_retrieval([old], (1,)).per_k[1]
    assert "chars" not in compare_paired([old], [old], 1)["metrics"]
