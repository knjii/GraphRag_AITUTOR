"""Рёбра-синонимы (К11): порог, k соседей, обозначения и хабы не связываются."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from rag_textbook.stores.graph_file import GraphFile, MemoryGraphStore

_SPEC = importlib.util.spec_from_file_location(
    "graph_synonyms", Path(__file__).resolve().parents[1] / "scripts" / "graph_synonyms.py"
)
gs = importlib.util.module_from_spec(_SPEC)
sys.modules["graph_synonyms"] = gs
_SPEC.loader.exec_module(gs)


def _graph() -> GraphFile:
    graph = GraphFile()
    graph.add_passage("A:00001", doc_id="A")
    graph.add_passage("B:00001", doc_id="B")
    graph.add_entity("inner", canonical="внутренний произведение")
    graph.add_entity("dot", canonical="скалярный произведение вектор")
    graph.add_entity("sym", canonical="B", kind="notation")
    graph.add_entity("sym2", canonical="B", kind="notation")
    graph.add_mention("A:00001", "inner")
    graph.add_mention("B:00001", "dot")
    graph.add_mention("A:00001", "sym")
    graph.add_mention("B:00001", "sym2")
    return graph


def test_threshold_and_symmetry():
    ids = ["a", "b", "c"]
    vectors = np.array([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0]])
    edges = gs.synonym_edges(ids, vectors, threshold=0.9, k=2)
    assert [(a, b) for a, b, _ in edges] == [("a", "b")]
    assert edges[0][2] > 0.99
    assert gs.synonym_edges(ids, vectors, threshold=0.999, k=2) == []


def test_k_limits_neighbours():
    ids = [f"e{i}" for i in range(5)]
    vectors = np.ones((5, 3)) + np.arange(5)[:, None] * 1e-3
    assert len(gs.synonym_edges(ids, vectors, threshold=0.5, k=1)) < 10


def test_notation_and_hubs_are_not_candidates():
    graph = _graph()
    assert gs.candidates(graph, max_degree=64) == ["dot", "inner"]
    assert gs.candidates(graph, max_degree=0) == []


def test_walk_crosses_books_only_with_synonym(tmp_path):
    graph = _graph()
    graph.add_relation("inner", "dot", gs.REL, "синоним", 0.9)
    store = MemoryGraphStore(graph)
    plain = store.expand_entities(["inner"], hops=1, rel_types=["RELATES"], limit=10)
    both = store.expand_entities(["inner"], hops=1, rel_types=["RELATES", gs.REL], limit=10)
    assert "dot" not in plain and "dot" in both


def test_main_writes_graph_and_refuses_twice(tmp_path, monkeypatch):
    src, out = tmp_path / "g.json.gz", tmp_path / "syn.json.gz"
    _graph().save(src)
    monkeypatch.setattr(
        gs,
        "embed_service",
        lambda names: np.array([[1.0, 0.05] if "внутр" in n else [1.0, 0.0] for n in names]),
    )
    assert gs.main(["--graph", str(src), "--out", str(out), "--threshold", "0.8"]) == 0
    rels = [r for r in GraphFile.load(out).relations if r[2] == gs.REL]
    assert len(rels) == 1 and {rels[0][0], rels[0][1]} == {"inner", "dot"}
    try:
        gs.main(["--graph", str(out), "--out", str(tmp_path / "x.json.gz")])
    except SystemExit as stop:
        assert "уже содержит" in str(stop)
    else:
        raise AssertionError("повторное наложение синонимов должно отвергаться")
