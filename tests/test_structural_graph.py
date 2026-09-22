"""Структурный граф К1: узлы и рёбра по правилам, без модели."""

from __future__ import annotations

import json

import pytest

from rag_textbook.config import GraphSettings
from rag_textbook.graph.structural import STRUCT_REL, build_structural, merge_graphs
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.stores.graph_file import GraphFile, MemoryGraphStore


def _chunks(doc: str = "book") -> list[dict]:
    return [
        {
            "id": f"{doc}:1", "doc_id": doc, "doc_name": "Книга", "ordinal": 1,
            "headers": ["2. МАТРИЦЫ"], "text": "Вводная часть главы о матрицах.",
        },
        {
            "id": f"{doc}:2", "doc_id": doc, "doc_name": "Книга", "ordinal": 2,
            "headers": ["2.1. Определитель"],
            "text": "Определение 2.1 (определитель). Число $\\det A = \\sum x \\tag{2.3}$.",
        },
        {
            "id": f"{doc}:3", "doc_id": doc, "doc_name": "Книга", "ordinal": 3,
            "headers": ["2.1.1. Свойства"], "text": "Прочий текст без ссылок и номеров.",
        },
        {
            "id": f"{doc}:4", "doc_id": doc, "doc_name": "Книга", "ordinal": 4,
            "headers": ["3.2. Собственные значения"],
            "text": "Подставив (2.3) и пользуясь определением 2.1, см. раздел 2.1, а также (9.9).",
        },
    ]


def test_references_connect_passages_through_shared_node() -> None:
    graph, report = build_structural(_chunks())
    roles = {
        (pid, graph.entities[eid]["kind"]): role
        for pid, mentions in graph.mentions.items()
        for eid, (_, role) in mentions.items()
    }
    assert roles[("book:2", "formula")] == "defines"
    assert roles[("book:2", "statement")] == "defines"
    assert roles[("book:4", "formula")] == "refers"
    assert roles[("book:4", "statement")] == "refers"
    assert roles[("book:4", "section")] == "refers"
    book = report["books"]["book"]
    # (9.9) нигде не введена — ссылка не разрешена и учтена.
    assert book["references"]["equation"] == 2 and book["resolved"]["equation"] == 1
    assert 0 < book["unresolved_share"] < 1


def test_statement_name_is_searchable() -> None:
    graph, _ = build_structural(_chunks())
    names = [row["name"] for row in graph.entities.values() if row["kind"] == "statement"]
    assert names == ["2.1 определитель"]
    store = MemoryGraphStore(graph)
    assert store.find_seed_entities(["определитель"], 5)


def test_nesting_edges_are_separate_type() -> None:
    graph, _ = build_structural(_chunks())
    by_id = graph.entities
    edges = {(by_id[s]["name"], by_id[t]["name"]) for s, t, kind, _, _ in graph.relations if kind == STRUCT_REL}
    # Заголовок главы «2.» правилом не ловится (как в crossref), подраздел — да.
    assert ("раздел 2.1.1", "раздел 2.1") in edges
    assert ("(2.3)", "раздел 2.1") in edges
    assert all(kind == STRUCT_REL for _, _, kind, _, _ in graph.relations)


def test_numbers_do_not_collide_between_books() -> None:
    graph, _ = build_structural(_chunks("a") + _chunks("b"))
    formulas = [row for row in graph.entities.values() if row["kind"] == "formula"]
    assert len(formulas) == 2


def test_graph_channel_walks_reference_to_definition() -> None:
    graph, _ = build_structural(_chunks())
    retriever = GraphRetriever(
        GraphSettings().model_copy(update={"seed_mode": "passages"}), MemoryGraphStore(graph)
    )
    found = [item.chunk.id for item in retriever.retrieve("вопрос", seed_chunk_ids=["book:4"])]
    assert "book:2" in found


def test_merge_keeps_strongest_role_and_all_edges(tmp_path) -> None:
    structural, _ = build_structural(_chunks())
    model = GraphFile(variant="model")
    for pid, row in structural.passages.items():
        model.add_passage(pid, doc_id=row["doc_id"], doc_name=row["doc_name"], ordinal=row["ordinal"], text=row["text"])
    model.add_entity("m", canonical="матрица")
    model.add_mention("book:1", "m", 2, role="mentions")
    model.add_mention("book:2", "m", 1, role="defines")
    union = merge_graphs(model, structural)
    assert union.summary()["entities"] == structural.summary()["entities"] + 1
    assert union.mentions["book:2"]["m"] == (1, "defines")
    assert len(union.relations) == len(structural.relations)
    path = union.save(tmp_path / "u.json.gz")
    assert GraphFile.load(path).summary() == union.summary()


def test_merge_rejects_broken_input() -> None:
    graph = GraphFile(variant="x")
    graph.add_mention("p", "ghost", 1)
    with pytest.raises(ValueError, match="испорчено"):
        merge_graphs(graph)


def test_script_writes_file_and_report(tmp_path) -> None:
    import importlib.util

    parsed = tmp_path / "parsed"
    parsed.mkdir()
    (parsed / "book_chunks.json").write_text(json.dumps(_chunks(), ensure_ascii=False), encoding="utf-8")
    spec = importlib.util.spec_from_file_location("graph_structural", "scripts/graph_structural.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(
        ["--parsed", str(parsed), "--out", str(tmp_path / "s.json.gz"), "--report", str(tmp_path / "r.json")]
    )
    assert code == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["total"]["entities"] > 0 and report["sha256"]
