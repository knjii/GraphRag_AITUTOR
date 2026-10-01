"""Приёмка эталона v2: выборка, решение по порогу, запись отметки."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

from rag_textbook.evaluation.goldset import load_goldset, save_goldset
from rag_textbook.models import GoldQuestion

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("goldset_review", ROOT / "scripts/goldset_review.py")
review = importlib.util.module_from_spec(spec)
sys.modules["goldset_review"] = review
spec.loader.exec_module(review)


def make_set(tmp_path: Path) -> tuple[Path, Path]:
    questions = []
    for i in range(10):
        questions.append(
            GoldQuestion(
                id=f"l{i}",
                question=f"q{i}",
                gold_chunk_ids=[f"d:{i}", f"d:{i + 1}"],
                question_type="graph_linked",
                expected_hops=2,
                slice="linking",
                split="test",
            )
        )
    for i in range(4):
        questions.append(
            GoldQuestion(
                id=f"c{i}",
                question=f"c{i}",
                gold_chunk_ids=[f"d:{i}", f"e:{i}"],
                question_type="multi_hop",
                expected_hops=2,
                slice="cross_book",
                split="test",
            )
        )
    for i in range(6):
        questions.append(
            GoldQuestion(
                id=f"s{i}",
                question=f"s{i}",
                gold_chunk_ids=[f"d:{i}"],
                question_type="single_chunk",
                split="test" if i < 4 else "dev",
            )
        )
    path = tmp_path / "goldset-v2.json"
    save_goldset(questions, path)
    parsed = tmp_path / "parsed"
    parsed.mkdir()
    rows = [{"id": f"{doc}:{i}", "text": f"текст {doc} {i}"} for doc in "de" for i in range(12)]
    (parsed / "d_chunks.json").write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path, parsed


def test_sheet_samples_only_test_split_and_writes_template(tmp_path: Path) -> None:
    goldset, parsed = make_set(tmp_path)
    out = tmp_path / "review"
    code = review.main(
        [
            "sheet",
            "--goldset",
            str(goldset),
            "--parsed",
            str(parsed),
            "--out",
            str(out),
            "--linking",
            "3",
            "--cross",
            "2",
            "--single",
            "10",
        ]
    )
    assert code == 0
    template = json.loads((out / "verdicts.json").read_text(encoding="utf-8"))["verdicts"]
    ids = [row["question_id"] for row in template]
    assert len([i for i in ids if i.startswith("l")]) == 3
    assert len([i for i in ids if i.startswith("c")]) == 2
    assert {i for i in ids if i.startswith("s")} == {"s0", "s1", "s2", "s3"}  # dev не берётся
    assert "текст d" in (out / "sheet.md").read_text(encoding="utf-8")


def fill(path: Path, verdicts: dict[str, str]) -> Path:
    target = path.parent / "verdicts.json"
    target.write_text(
        json.dumps({"verdicts": [{"question_id": k, "verdict": v} for k, v in verdicts.items()]}),
        encoding="utf-8",
    )
    return target


def test_accept_rejects_too_many_single_hop(tmp_path: Path) -> None:
    goldset, _ = make_set(tmp_path)
    verdicts = {f"l{i}": ("single_hop_enough" if i < 4 else "ok") for i in range(10)}
    code = review.main(
        ["accept", "--goldset", str(goldset), "--verdicts", str(fill(goldset, verdicts))]
    )
    assert code == 1
    assert not goldset.with_suffix(".accepted").exists()


def test_accept_drops_unusable_demotes_single_hop_and_writes_marker(tmp_path: Path) -> None:
    goldset, _ = make_set(tmp_path)
    verdicts = {f"l{i}": "ok" for i in range(10)}
    verdicts["l0"] = "single_hop_enough"
    verdicts["s0"] = "unanswerable"
    code = review.main(
        ["accept", "--goldset", str(goldset), "--verdicts", str(fill(goldset, verdicts))]
    )
    assert code == 0
    kept = {q.id: q for q in load_goldset(goldset)}
    assert "s0" not in kept
    assert kept["l0"].expected_hops == 1 and kept["l0"].slice == "single"
    assert kept["l0"].question_type == "single_chunk"  # иначе K4 снова сочтёт его связывающим
    assert kept["l1"].verified
    marker = goldset.with_suffix(".accepted").read_text(encoding="utf-8")
    assert marker.split()[0] == hashlib.sha256(goldset.read_bytes()).hexdigest()


def test_accept_filtered_skips_thresholds_and_drops_unreviewed_two_hop(tmp_path: Path) -> None:
    goldset, _ = make_set(tmp_path)
    # 3 из 5 проверенных двухшаговых — single_hop_enough: по порогам выборки это провал.
    verdicts = {
        "l0": "single_hop_enough",
        "l1": "single_hop_enough",
        "l2": "single_hop_enough",
        "l3": "ok",
        "c0": "ok",
        "s0": "unanswerable",
    }
    code = review.main(
        [
            "accept",
            "--goldset",
            str(goldset),
            "--verdicts",
            str(fill(goldset, verdicts)),
            "--filtered",
        ]
    )
    assert code == 0
    kept = {q.id: q for q in load_goldset(goldset)}
    assert {"l4", "c1", "s0"}.isdisjoint(kept)  # непроверенные двухшаговые и негодный удалены
    assert {"s1", "s5"} <= set(kept)  # непроверенные одношаговые остаются
    assert kept["l0"].slice == "single" and kept["l3"].slice == "linking"
    assert goldset.with_suffix(".accepted").exists()


def test_accept_refuses_empty_and_unknown_verdicts(tmp_path: Path) -> None:
    goldset, _ = make_set(tmp_path)
    assert (
        review.main(
            ["accept", "--goldset", str(goldset), "--verdicts", str(fill(goldset, {"l0": ""}))]
        )
        == 1
    )
    assert (
        review.main(
            ["accept", "--goldset", str(goldset), "--verdicts", str(fill(goldset, {"l0": "good"}))]
        )
        == 1
    )
    assert (
        review.main(
            ["accept", "--goldset", str(goldset), "--verdicts", str(fill(goldset, {"zz": "ok"}))]
        )
        == 1
    )


def test_fragment_puts_display_math_on_its_own_lines() -> None:
    text = "Итак, $$x = 1\tag{2.1}$$ Мы знаем, что $y$ мало."
    out = review.fragment_markdown(text)
    assert "\n$$\nx = 1\tag{2.1}\n$$\n" in out
    assert "Мы знаем, что $y$ мало." in out
    assert "⚠" not in out


def test_fragment_cut_inside_inline_math_is_escaped_and_flagged() -> None:
    # Нарезка начала фрагмент с хвоста формулы и оборвала последнюю.
    text = "} ( s ) > 0$ строго, где $a$ и $b _ {"
    out = review.fragment_markdown(text)
    assert out.startswith("*⚠ Нарезка: начинается посреди формулы, обрывается посреди формулы.*")
    assert "} ( s ) > 0\$ строго, где $a$ и \$b _ {" in out
