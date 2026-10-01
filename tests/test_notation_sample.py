"""Проверки несмещённого отбора и ручного критерия К3 без модели."""

import json
import math
from collections import Counter
from pathlib import Path

import pytest

from scripts import notation_sample as ns


def write_chunks(path: Path, chunks: list[dict]) -> None:
    (path / "toy_chunks.json").write_text(json.dumps(chunks), encoding="utf-8")


def test_candidates_independent_of_rules(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    texts = [
        "Пусть $x$ произволен",
        "Назовём $y$ меткой",
        "We denote $z$",
        "где нет формулы",
        "$x$ без маркера",
        "ОБОЗНАЧИМ $t$",
        "через $v$",
    ]
    write_chunks(tmp_path, [{"id": str(i), "doc_id": "a", "text": t} for i, t in enumerate(texts)])

    def forbidden(text: str) -> list:
        pytest.fail("Отбор вызвал правила")

    monkeypatch.setattr(ns, "find_notations", forbidden)
    pool = ns.candidates(tmp_path)
    assert [c["id"] for c in pool] == ["0", "1", "2", "5", "6"]
    assert len(ns.stratified_sample(pool, 3, 42)) == 3


@pytest.mark.parametrize(
    "sizes,sample", [([100, 100, 100], 50), ([100, 1, 1], 50), ([1, 1, 1], 50), ([5] * 20, 3)]
)
def test_stratification(sizes: list[int], sample: int) -> None:
    chunks = [
        {"id": f"{b}:{i}", "doc_id": str(b)} for b, size in enumerate(sizes) for i in range(size)
    ]
    chosen = ns.stratified_sample(chunks, sample, 17)
    assert chosen == ns.stratified_sample(chunks, sample, 17)
    counts = Counter(c["doc_id"] for c in chosen)
    cap = math.ceil(sample / len(sizes)) + 1
    assert max(counts.values()) <= cap
    assert len(chosen) == min(sample, sum(min(size, cap) for size in sizes))
    assert len({c["id"] for c in chosen}) == len(chosen)
    if sample >= len(sizes):
        assert len(counts) == len(sizes)


def test_normalization() -> None:
    assert ns.normalize_symbol(" $$ x _ { i } $$ \n") == "x_{i}"
    assert ns.normalize_symbol(r"$\mathbf{x}$") == r"\mathbf{x}"
    assert (
        len(
            ns.symbols(
                [
                    r"\mathbf{x} := вектор",
                    r"\boldsymbol{x} := вектор",
                    "x := скаляр",
                    "$ x $ := другой смысл",
                ]
            )
        )
        == 3
    )


def key_with(rows: list[dict], graph: bool = True) -> dict:
    return {"items": rows, "graph_available": graph}


def test_metrics_and_graph_separate() -> None:
    key = key_with(
        [
            {
                "human": ["x := икс", "y := игрек"],
                "rules": ["$ x $ := другое", "z := лишнее", "x := дубль"],
                "graph": ["y := игрек"],
            },
            {"human": ["x := икс"], "rules": ["x := икс"], "graph": []},
        ]
    )
    result = ns.score(key, 300, 1)
    assert result["rules"]["recall"] == pytest.approx(2 / 3)
    assert result["rules"]["precision"] == pytest.approx(2 / 3)
    assert result["graph"]["recall"] == pytest.approx(1 / 3)
    assert result["graph"]["precision"] == 1
    assert result["rules"]["recall_ci"] == [0.5, 1.0]
    assert result["decision"] == "не различимо"
    assert result == ns.score(key, 300, 1)


@pytest.mark.parametrize(
    "interval,decision",
    [
        ([0.81, 1.0], "правила достаточны"),
        ([0.2, 0.79], "нужен запасной путь v4"),
        ([0.8, 0.9], "не различимо"),
        ([0.7, 0.8], "не различимо"),
        (None, "не различимо"),
    ],
)
def test_threshold(interval: list[float] | None, decision: str) -> None:
    assert ns.threshold_decision(interval) == decision


@pytest.mark.parametrize(
    "found,decision", [(True, "правила достаточны"), (False, "нужен запасной путь v4")]
)
def test_score_decision(found: bool, decision: str) -> None:
    result = ns.score(
        key_with([{"human": ["x := смысл"], "rules": ["x := смысл"] if found else []}], False), 50
    )
    assert result["decision"] == decision
    assert result["graph"] is None


def test_missing_and_empty_annotations() -> None:
    with pytest.raises(ValueError, match="null"):
        ns.score(key_with([{"human": None}], False))
    with pytest.raises(ValueError, match="пуста"):
        ns.score(key_with([], False))
    result = ns.score(key_with([{"human": [], "rules": ["x := лишнее"]}], False), 50)
    assert result["rules"]["precision"] == 0
    assert result["rules"]["recall"] is None
    assert result["decision"] == "не различимо"
    with pytest.raises(ValueError):
        ns.symbols(["просто символ"])


def test_graph_file_sheet_and_cli(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    graph = ns.GraphFile()
    for pid in ("a:0", "a:1"):
        graph.add_passage(pid, doc_id="a")
    graph.add_entity("x", canonical="x := вектор", name="x", kind="notation")
    graph.add_entity("y", canonical="y", name="y")
    graph.add_entity("m", canonical="метка")
    graph.add_mention("a:0", "x", role="defines")
    graph.add_mention("a:0", "y")
    graph.add_mention("a:1", "x", role="uses")
    graph.add_mention("a:1", "m")
    graph.relations.append(("y", "m", "RELATES", ns.NOTATION_LABEL, 1.0))
    graph.save(tmp_path / "graph.json")
    write_chunks(tmp_path, [{"id": "a:0", "doc_id": "a", "text": "где $x$ — вектор. " * 120}])
    output = tmp_path / "sheet"
    assert (
        ns.main(
            [
                "sheet",
                "--parsed",
                str(tmp_path),
                "--graph",
                str(tmp_path / "graph.json"),
                "--output",
                str(output),
            ]
        )
        == 0
    )
    key = json.loads(output.with_suffix(".json").read_text(encoding="utf-8"))
    row = key["items"][0]
    assert row["graph"] == ["x := вектор", "y := метка"]
    assert ns.graph_notations(graph, "a:1") == []
    assert len(row["text"]) == 1500 < len(row["full_text"])
    assert row["human"] is None
    assert "Человек" in output.with_suffix(".md").read_text(encoding="utf-8")
    row["human"] = ["x := вектор"]
    output.with_suffix(".json").write_text(json.dumps(key), encoding="utf-8")
    capsys.readouterr()
    assert ns.main(["score", "--key", str(output.with_suffix(".json")), "--bootstrap", "50"]) == 0
    assert json.loads(capsys.readouterr().out)["decision"] == "правила достаточны"


def test_sheet_without_graph_and_empty_pool(tmp_path: Path) -> None:
    assert ns.make_sheet(tmp_path, 50, 0, None)["items"] == []
    write_chunks(tmp_path, [{"id": "a", "doc_id": "a", "text": "Пусть $x$"}])
    key = ns.make_sheet(tmp_path, 50, 0, None)
    assert key["items"][0]["graph"] is None
    assert key["items"][0]["rules"] == []
    with pytest.raises(ValueError):
        ns.stratified_sample([], 0)
