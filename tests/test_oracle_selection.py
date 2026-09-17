"""Оракул отбора: пул, выдача и находки только графа."""

from __future__ import annotations

import pytest

from rag_textbook.evaluation.oracle import evaluate, graph_only_of, pool_of, summarize


def _row(final, base, graph, rerank=(), qid="q", kind="graph_linked"):
    return {
        "question_id": qid, "question_type": kind, "final": list(final),
        "channels": {
            "base": [{"chunk_id": c, "rank": i, "score": 1.0} for i, c in enumerate(base)],
            "graph": [{"chunk_id": c, "rank": i, "score": 1.0} for i, c in enumerate(graph)],
        },
        "rerank_scores": {c: 0.5 for c in rerank},
    }


def test_pool_is_union_of_everything_seen():
    row = _row(final=["a"], base=["a", "b"], graph=["c"], rerank=["d"])
    assert pool_of(row) == {"a", "b", "c", "d"}


def test_graph_only_excludes_chunks_found_elsewhere():
    assert graph_only_of(_row(final=[], base=["a", "b"], graph=["b", "c"])) == {"c"}


def test_headroom_when_gold_is_in_pool_but_not_selected():
    rows = [_row(final=["a", "x"], base=["a"], graph=["b"], qid="q1")]
    [outcome] = evaluate(rows, {"q1": {"a", "b"}})
    assert outcome.recall == 0.5
    assert outcome.oracle_recall == 1.0
    assert outcome.graph_only_gold == 1
    assert outcome.graph_only_in_final == 0


def test_no_headroom_when_gold_missing_from_pool():
    [outcome] = evaluate([_row(final=["a"], base=["a"], graph=[], qid="q1")], {"q1": {"a", "z"}})
    assert outcome.recall == outcome.oracle_recall == 0.5


def test_summary_groups_by_type():
    rows = [
        _row(final=["a"], base=["a"], graph=["b"], qid="q1", kind="graph_linked"),
        _row(final=["c"], base=["c"], graph=[], qid="q2", kind="single_chunk"),
    ]
    summary = summarize(evaluate(rows, {"q1": {"a", "b"}, "q2": {"c"}}))
    assert summary["all"]["questions_with_headroom"] == 1
    assert summary["by_type"]["single_chunk"]["headroom"] == 0.0


def test_missing_gold_is_an_error():
    with pytest.raises(ValueError):
        evaluate([_row(final=[], base=[], graph=[], qid="q1")], {})
