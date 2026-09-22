"""Граф как сменный файл: формат и поведение хранилища в памяти.

Хранилище в памяти годится для замеров, только если повторяет запросы
Neo4j. Здесь формулы баллов проверяются вручную посчитанными числами,
а совпадение с живым Neo4j проверяет ``scripts/graph_fidelity.py`` на
выгрузке настоящего графа.
"""

from __future__ import annotations

import json
import math

import pytest

from rag_textbook.config import GraphSettings
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.stores.graph_file import GraphFile, MemoryGraphStore, analyze, file_hash


def _graph() -> GraphFile:
    graph = GraphFile(variant="test")
    texts = {
        "p1": "Сингулярное разложение матрицы.",
        "p2": "Метод главных компонент через сингулярное разложение.",
        "p3": "Ковариационная матрица признаков.",
        "p4": "Собственные векторы ковариационной матрицы.",
    }
    for ordinal, (pid, text) in enumerate(texts.items()):
        graph.add_passage(pid, doc_id="d", doc_name="Книга", ordinal=ordinal, text=text, pages=[1])
    graph.add_entity("svd", canonical="сингулярный разложение", name="сингулярное разложение")
    graph.add_entity("pca", canonical="метод главный компонента", name="метод главных компонент")
    graph.add_entity("cov", canonical="ковариационный матрица", name="ковариационная матрица")
    graph.add_entity("eig", canonical="собственный вектор", name="собственные векторы")
    graph.add_mention("p1", "svd", 2)
    graph.add_mention("p2", "svd", 1)
    graph.add_mention("p2", "pca", 1)
    graph.add_mention("p3", "cov", 1)
    graph.add_mention("p4", "cov", 1)
    graph.add_mention("p4", "eig", 3)
    graph.add_relation("pca", "svd", "RELATES", "вычисляется_по", 1.0)
    graph.add_relation("pca", "cov", "RELATES", "использует", 1.0)
    graph.add_relation("svd", "eig", "CO_OCCURS", "", 2.5)
    return graph


def test_round_trip_is_stable(tmp_path) -> None:
    graph = _graph()
    first = graph.save(tmp_path / "a.json.gz")
    loaded = GraphFile.load(first)
    second = loaded.save(tmp_path / "b.json.gz")
    assert loaded.summary() == graph.summary()
    # Один и тот же граф — один и тот же хэш: он пишется в результаты замера.
    assert file_hash(first) == file_hash(second)


def test_validate_catches_dangling_references() -> None:
    graph = _graph()
    graph.add_mention("p1", "ghost", 1)
    graph.add_relation("svd", "nowhere")
    problems = graph.validate()
    assert any("ghost" in item for item in problems)
    assert any("nowhere" in item for item in problems)


def test_unknown_format_is_rejected() -> None:
    with pytest.raises(ValueError, match="формате"):
        GraphFile.from_json({"format": "other/0"})


def test_analyzer_splits_like_standard_tokenizer() -> None:
    assert analyze("Метод Главных-Компонент, x_1 и 3.14") == [
        "метод", "главных", "компонент", "x_1", "и", "3.14",
    ]


def test_seed_search_is_phrase_bm25_over_both_fields() -> None:
    store = MemoryGraphStore(_graph())
    rows = store.find_seed_entities(["сингулярный разложение"], 10)
    assert [row["id"] for row in rows] == ["svd"]

    # Фраза требует соседства слов: «разложение сингулярный» не совпадает.
    assert store.find_seed_entities(["разложение сингулярный"], 10) == []

    # Слово «матрица» есть у одного узла в обоих полях — ручной расчёт BM25
    # Lucene 9; баллы полей складываются, как в дизъюнкции Lucene.
    rows = store.find_seed_entities(["матрица"], 10)
    assert [row["id"] for row in rows] == ["cov"]
    docs, df, length = 4, 1, 2
    avg = (2 + 3 + 2 + 2) / 4
    idf = math.log(1 + (docs - df + 0.5) / (df + 0.5))
    per_field = idf * 1 / (1 + 1.2 * (1 - 0.75 + 0.75 * length / avg))
    assert rows[0]["score"] == pytest.approx(2 * per_field)


def test_entities_of_passages_weights_by_rarity() -> None:
    store = MemoryGraphStore(_graph())
    rows = {row["id"]: row for row in store.entities_of_passages(["p2"], 10)}
    # svd встречается в двух фрагментах из четырёх, pca — в одном.
    assert rows["svd"]["weight"] == pytest.approx(math.log(2) * math.log(4 / 2))
    assert rows["pca"]["weight"] == pytest.approx(math.log(2) * math.log(4 / 1))
    assert list(rows) == ["pca", "svd"]


def test_expansion_uses_only_allowed_types_and_decays() -> None:
    store = MemoryGraphStore(_graph())
    weights = store.expand_entities(["svd"], hops=1, rel_types=["RELATES"], limit=10, decay=0.8)
    assert weights == {"svd": 1.0, "pca": pytest.approx(0.8)}
    weights = store.expand_entities(["svd"], hops=2, rel_types=["RELATES"], limit=10, decay=0.8)
    assert weights["cov"] == pytest.approx(0.64)
    # CO_OCCURS не участвует, если не разрешён явно.
    assert "eig" not in weights


def test_expansion_limit_counts_reachable_seeds() -> None:
    # Cypher возвращает и затравку, достижимую от другой затравки: она
    # занимает место в LIMIT, хотя в веса не попадает.
    store = MemoryGraphStore(_graph())
    weights = store.expand_entities(["svd", "pca"], hops=1, rel_types=["RELATES"], limit=2, decay=0.5)
    assert weights == {"svd": 1.0, "pca": 1.0}


def test_find_passages_matches_cypher_formula() -> None:
    store = MemoryGraphStore(_graph())
    rows = store.find_passages({"svd": 1.0, "pca": 0.5}, 10)
    by_id = {row["chunk_id"]: row["score"] for row in rows}
    idf_svd = math.log(4 / 2)
    idf_pca = math.log(4 / 1)
    assert by_id["p1"] == pytest.approx(idf_svd * math.log(3) / math.sqrt(1))
    assert by_id["p2"] == pytest.approx(
        (idf_svd * math.log(2) + 0.5 * idf_pca * math.log(2)) / math.sqrt(2)
    )
    assert set(by_id) == {"p1", "p2"}


def test_passage_links_shared_and_dependency() -> None:
    graph = _graph()
    graph.add_mention("p1", "svd", 2, role="defines")
    graph.add_mention("p2", "svd", 1, role="uses")
    store = MemoryGraphStore(graph)
    shared = store.passage_links(["p1", "p2", "p3"])
    assert shared[("p1", "p2")] == pytest.approx(math.log(4 / 2))
    assert ("p1", "p3") not in shared
    assert store.passage_links(["p1", "p2"], mode="dependency") == {("p2", "p1"): 1.0}
    assert store.definitions_for(["p2"]) == ["p1"]


def test_defines_role_survives_later_mention() -> None:
    graph = _graph()
    graph.add_mention("p1", "svd", 2, role="defines")
    graph.add_mention("p1", "svd", 2, role="mentions")
    assert graph.mentions["p1"]["svd"] == (2, "defines")


def test_ppr_reaches_passage_two_steps_away() -> None:
    store = MemoryGraphStore(_graph())
    rows = store.ppr_passages({"pca": 1.0}, 10, alpha=0.5, rel_types=["RELATES"], use_idf=False)
    ranked = [row["chunk_id"] for row in rows]
    # p2 упоминает pca напрямую; p1 и p3/p4 достижимы через связи pca.
    assert ranked[0] == "p2"
    assert {"p1", "p3"} <= set(ranked)
    assert sum(row["score"] for row in rows) < 1.0


def test_retriever_runs_on_memory_store(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_SEED_MODE", "both")
    store = MemoryGraphStore(_graph())
    retriever = GraphRetriever(GraphSettings(), store)
    results = retriever.retrieve("Как метод главных компонент связан с разложением?", seed_chunk_ids=["p2"])
    ids = [item.chunk.id for item in results]
    assert "p2" not in ids  # опорный фрагмент исключён
    assert ids, "канал по файлу должен что-то находить"


def test_ppr_ranker_requires_memory_backend(monkeypatch) -> None:
    monkeypatch.setenv("GRAPH_RANKER", "ppr")
    with pytest.raises(ValueError, match="memory"):
        GraphSettings()


def test_ppr_ranker_on_memory_store(monkeypatch, tmp_path) -> None:
    path = _graph().save(tmp_path / "g.json")
    monkeypatch.setenv("GRAPH_BACKEND", "memory")
    monkeypatch.setenv("GRAPH_FILE", str(path))
    monkeypatch.setenv("GRAPH_RANKER", "ppr")
    monkeypatch.setenv("GRAPH_SEED_MODE", "passages")
    retriever = GraphRetriever(GraphSettings(), MemoryGraphStore.from_file(path))
    results = retriever.retrieve("вопрос", seed_chunk_ids=["p2"])
    assert results and all(item.chunk.id != "p2" for item in results)


def test_fidelity_script_accepts_identical_channel(tmp_path, monkeypatch) -> None:
    """Сквозная проверка скрипта допуска на слепке, снятом с того же графа."""
    import importlib.util

    from rag_textbook.evaluation.trace import QueryTrace, TracedCandidate, TraceSet

    monkeypatch.setenv("GRAPH_SEED_MODE", "both")
    path = _graph().save(tmp_path / "g.json.gz")
    retriever = GraphRetriever(GraphSettings(), MemoryGraphStore.from_file(path))
    question = "метод главных компонент"
    produced = retriever.retrieve(question, seed_chunk_ids=["p2"])
    traces = TraceSet(settings_snapshot={"graph.seed_mode": "both"})
    traces.traces.append(
        QueryTrace(
            question_id="q1",
            question=question,
            rewritten_question=question,
            used_graph=True,
            channels={
                "base": [TracedCandidate("p2", 0, 1.0)],
                "graph": [TracedCandidate(item.chunk.id, i, item.score) for i, item in enumerate(produced)],
            },
        )
    )
    traces.save(tmp_path / "t.jsonl")

    spec = importlib.util.spec_from_file_location("graph_fidelity", "scripts/graph_fidelity.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    code = module.main(
        ["--graph-file", str(path), "--trace", str(tmp_path / "t.jsonl"), "--json", str(tmp_path / "r.json")]
    )
    assert code == 0
    report = json.loads((tmp_path / "r.json").read_text(encoding="utf-8"))
    assert report["against_trace"]["same_set_share"] == 1.0
    assert report["verdict"] == "годен"
