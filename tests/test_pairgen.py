"""Независимые источники пар, двухшаговость и устойчивость эталона."""

from __future__ import annotations

import ast
import json
import random
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from rag_textbook.evaluation import pairgen as pg
from rag_textbook.evaluation.ablation import AblationResult
from rag_textbook.evaluation.goldset import GoldsetBuilder, load_goldset
from rag_textbook.models import Chunk, GoldQuestion


def chunk(doc: str, ordinal: int, text: str = "", headers: tuple[str, ...] = ("Матрицы",),
          formula: bool = False) -> Chunk:
    return Chunk(id=f"{doc}:{ordinal}", doc_id=doc, doc_name=doc, source_path=f"{doc}.pdf",
                 ordinal=ordinal, headers=list(headers), has_formula=formula,
                 text=text + " Линейная алгебра изучает пространство векторов и матрицы." * 9)


@pytest.fixture
def corpus() -> list[Chunk]:
    return [chunk("a", 0, r"Определение 3.8 (спектр). $$x=1\tag{9.19}$$", formula=True),
            chunk("a", 1, "В определении 3.8 и (9.19)."),
            chunk("a", 4, "Применим определение 3.8 и (9.19)."),
            chunk("b", 0, headers=("Теория матриц",)),
            chunk("c", 0, headers=("Интегралы функций",))]


@pytest.mark.parametrize("source", pg.SOURCES)
def test_sources_and_common_filters(corpus, source):
    dirty = [chunk("a", 8, headers=("УПРАЖНЕНИЯ",)),
             chunk("b", 8, ". . " + "..... 12 " * 10),
             chunk("c", 8).model_copy(update={"text": "Короткий текст"})]
    fn = getattr(pg, source)
    kwargs = {"vectors": {c.id: [1, 0] for c in corpus + dirty}} if source == "dense" else {}
    pairs = fn(corpus + dirty, random.Random(7), **kwargs)
    assert pairs
    assert all(p.source == source for p in pairs)
    assert all(p.ordinal_distance is None or p.ordinal_distance >= 3 for p in pairs)
    assert all(p.left not in {c.id for c in dirty} and p.right not in {c.id for c in dirty}
               for p in pairs)
    assert len({(p.left, p.right) for p in pairs}) == len(pairs)
    if source in ("chapter_random", "explicit_ref"):
        assert any({p.left, p.right} == {"a:0", "a:4"} for p in pairs)
    if source == "cross_book":
        assert all(not p.same_doc for p in pairs)
        assert any("b:0" in (p.left, p.right) for p in pairs)
        assert all("c:0" not in (p.left, p.right) for p in pairs)


def test_chapter_fallback_window():
    chunks = [chunk("a", i, headers=()) for i in (0, 3, 40)]
    pairs = pg.chapter_random(chunks, random.Random(1))
    assert [(p.left, p.right) for p in pairs] == [("a:0", "a:3")]


def test_bm25_rare_terms_and_frequency():
    rare = "квазигруппа гомоморфизм эндоморфизм "
    chunks = [chunk("a", 0, rare * 2), chunk("a", 1, rare * 60),
              chunk("b", 0, rare * 30), chunk("c", 0, rare),
              chunk("d", 0), chunk("e", 0)]
    pairs = pg.bm25(chunks, random.Random(1))
    assert any(p.left == "a:0" and p.right == "b:0" for p in pairs)
    assert any("квазигруппа" in p.evidence for p in pairs)


def test_dense_cosine_and_missing_vectors(corpus):
    vectors = {"a:0": [10, 0], "a:1": [10, 0], "a:4": [1, 0.1],
               "b:0": [1, 1], "c:0": [0, 0]}
    pairs = pg.dense(corpus, random.Random(0), vectors)
    assert any(p.left == "a:0" and p.right == "a:4" for p in pairs)
    assert all("c:0" not in (p.left, p.right) for p in pairs)
    assert pg.dense(corpus, random.Random(0)) == []
    assert pg.summarize_pairs([], 5, vectors_available=False)["dense"]["warning"]
    with pytest.raises(ValueError, match="Размерности"):
        pg.dense(corpus, random.Random(0), {"a:0": [1], "a:4": [1, 2]})


def test_references_are_document_local():
    chunks = [chunk("a", 0, r"$$x=1\tag{9.19}$$"),
              chunk("b", 4, "По (9.19) получаем результат.")]
    assert pg.explicit_ref(chunks, random.Random(0)) == []


def test_dedup_priority_quota_and_reproducibility(corpus):
    vectors = {c.id: [1, 0] for c in corpus}
    pairs = pg.sample_pairs(corpus, 50, 42, vectors)
    assert pairs == pg.sample_pairs(list(reversed(corpus)), 50, 42, vectors)
    assert len({(p.left, p.right) for p in pairs}) == len(pairs)
    pair = next(p for p in pairs if (p.left, p.right) == ("a:0", "a:4"))
    assert pair.source == "chapter_random"
    assert "explicit_ref" in pair.evidence
    small = pg.sample_pairs(corpus, 1, 42, vectors)
    summary = pg.summarize_pairs(small, 1)
    assert all(row["count"] <= 1 for row in summary.values())
    assert sum(row["shortfall"] for row in summary.values()) == 5 - len(small)
    assert pg.sample_pairs(corpus, 0, 42) == []


class Model:
    def __init__(self):
        self.prompts = []
        self.closed = False

    def chat(self, messages, **kwargs):
        self.prompts.append(messages[0].content)
        return json.dumps({"question": f"Как вычислить результат метода {len(self.prompts)}?",
                           "answer": "С помощью спектра."})

    def close(self):
        self.closed = True


def test_build_v2(corpus, monkeypatch):
    model = Model()
    builder = GoldsetBuilder(model)

    def forbidden(*args, **kwargs):
        pytest.fail("Графовый отбор вызываться не должен")

    monkeypatch.setattr(builder, "_select_graph_linked_pairs", forbidden)
    pairs = pg.explicit_ref(corpus, random.Random(1)) + pg.cross_book(corpus, random.Random(1))
    questions = pg.build_v2(builder, corpus, pairs, 1, 1, 7)
    assert {q.slice for q in questions} == {"single", "formula", "linking", "cross_book"}
    assert sum(q.slice == "formula" for q in questions) == 1
    for q in questions:
        assert q.split in ("dev", "test")
        assert q.expected_hops == (2 if q.pair_source else 1)
        if q.pair_source:
            assert q.question_type == "multi_hop"
    assert any("Фрагмент А:" in p for p in model.prompts)
    old = GoldQuestion(id="old", question="?", gold_chunk_ids=[])
    assert (old.pair_source, old.slice, old.split) == ("", "", "")


def test_keep_two_hop():
    questions = [GoldQuestion(id=str(i), question="?", gold_chunk_ids=[], expected_hops=1 if i == 0 else 2)
                 for i in range(5)]
    results = [AblationResult(str(i), "multi_hop", verdict, [], False)
               for i, verdict in ((1, "ok"), (2, "single_hop_enough"), (3, "unanswerable"))]
    kept = pg.keep_two_hop(questions, results)
    assert kept == questions[:2]
    assert kept[0] is questions[0]


def test_split_stable_with_additions_and_within_strata():
    questions = [GoldQuestion(id=str(i), question="?", gold_chunk_ids=[], slice=s, pair_source=p)
                 for s, p in (("single", ""), ("linking", "bm25"), ("cross_book", "cross_book"))
                 for i in range(1000)]
    result = pg.assign_split(questions, 7)
    assert result == pg.assign_split(questions, 7)
    assert result != pg.assign_split(questions, 8)
    assert result[:900] == pg.assign_split(questions[:900], 7)
    for s in ("single", "linking", "cross_book"):
        count = sum(q.slice == s and q.split == "dev" for q in result)
        assert 330 < count < 470
    assert all(q.split == "" for q in questions)
    assert all(q.split == "test" for q in pg.assign_split(questions, 7, 0))
    assert all(q.split == "dev" for q in pg.assign_split(questions, 7, 1))


def test_no_graph_imports():
    tree = ast.parse(Path(pg.__file__).read_text(encoding="utf-8-sig"))
    forbidden = {"graph_store", "graph_offline", "extractor"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [alias.name for alias in node.names]
            if isinstance(node, ast.ImportFrom):
                names.append(node.module or "")
            assert all(not forbidden.intersection(name.split(".")) for name in names)


def test_cli_pairs_and_build_with_ablation(corpus, monkeypatch, tmp_path):
    from rag_textbook.cli import main as cli
    from rag_textbook.clients import llm
    from rag_textbook.stores import vector_store

    monkeypatch.setattr(cli, "_settings", lambda: SimpleNamespace(vector_store=None, llm=None))
    monkeypatch.setattr(cli, "build_context", lambda *_: pytest.fail("Контекст не нужен"))
    monkeypatch.setattr(vector_store, "build_vector_store",
                        lambda _: SimpleNamespace(iter_chunks=lambda: iter(corpus)))
    model = Model()
    monkeypatch.setattr(llm, "build_llm_client", lambda _: model)
    pairs_path = tmp_path / "pairs.jsonl"
    runner = CliRunner()
    result = runner.invoke(cli.app, ["goldset", "pairs", "--out", str(pairs_path), "--per-source", "50"])
    assert result.exit_code == 0, result.output
    assert "векторы не переданы" in result.output
    assert not model.prompts
    candidates = pg.explicit_ref(corpus, random.Random(0))
    pairs_path.write_text("\n".join(json.dumps(asdict(p)) for p in candidates), encoding="utf-8")

    def ablate(model, questions, chunks, **kwargs):
        return [AblationResult(q.id, q.question_type, "single_hop_enough", [True], True)
                for q in questions]

    monkeypatch.setattr(cli, "run_ablation", ablate)
    output = tmp_path / "gold.json"
    result = runner.invoke(cli.app, ["goldset", "build-v2", "--pairs", str(pairs_path),
                                    "--out", str(output), "--single", "1", "--formula", "1", "--ablate"])
    assert result.exit_code == 0, result.output
    questions = load_goldset(output)
    assert len(questions) == 2
    assert all(q.expected_hops == 1 for q in questions)
    verdicts = Path(str(output) + ".ablation.jsonl").read_text(encoding="utf-8")
    assert "single_hop_enough" in verdicts
    assert model.closed
