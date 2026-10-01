import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("bench_answers", ROOT / "scripts" / "bench_answers.py")
bench_answers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_answers)


def test_squad_normalization_em_and_f1():
    assert bench_answers.exact_match("The Beatles.", ["beatles"]) == 1.0
    assert bench_answers.exact_match("Beatles band", ["beatles"]) == 0.0
    assert bench_answers.f1_score("the Beatles band", ["Beatles"]) == pytest.approx(2 / 3)
    # Максимум по синонимам, как в оценке MuSiQue.
    assert bench_answers.f1_score("USA", ["United States", "USA"]) == 1.0
    assert bench_answers.f1_score("", ["x"]) == 0.0


def test_answer_is_taken_from_last_marker():
    raw = "Passage 1 says Answer: wrong.\nSo the answer is clear.\nAnswer: **Leo Tolstoy**."
    assert bench_answers.extract_answer(raw) == "Leo Tolstoy"
    assert bench_answers.extract_answer("just text\n\nParis") == "Paris"
    assert bench_answers.extract_answer("") == ""


def test_mcnemar_exact_counts_only_discordant_pairs():
    base = [1, 1, 0, 0, 0, 0, 0, 0, 1]
    cand = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    result = bench_answers.mcnemar_exact(base, cand)
    assert (result["better"], result["worse"]) == (6, 0)
    assert result["p"] == pytest.approx(2 / 2**6)
    assert bench_answers.mcnemar_exact([1, 0], [1, 0])["p"] == 1.0


def test_paired_t_matches_reference_values():
    # t = 2.0 при df = 9: двусторонний p = 0.07652 (таблица Стьюдента).
    diffs = [1.0, -1.0] * 5
    shift = 2.0 * math.sqrt(sum(d * d for d in diffs) / 9) / math.sqrt(10)
    base = [0.0] * 10
    cand = [d + shift for d in diffs]
    result = bench_answers.paired_t(base, cand)
    assert result["t"] == pytest.approx(2.0)
    assert result["p"] == pytest.approx(0.07652, abs=2e-4)
    # df = 29, t = 2.756 — p = 0.0100.
    diffs = [1.0, -1.0] * 15
    shift = 2.756 * math.sqrt(sum(d * d for d in diffs) / 29) / math.sqrt(30)
    result = bench_answers.paired_t([0.0] * 30, [d + shift for d in diffs])
    assert result["p"] == pytest.approx(0.0100, abs=2e-4)
    assert bench_answers.paired_t([0.5, 0.5], [0.5, 0.5])["p"] == 1.0


def test_holm_is_step_down_and_monotone():
    out = bench_answers.holm({"a": 0.01, "b": 0.04, "c": 0.03})
    assert out["a"]["p_holm"] == pytest.approx(0.03)
    assert out["c"]["p_holm"] == pytest.approx(0.06)
    assert out["b"]["p_holm"] == pytest.approx(0.06)  # не меньше предыдущего
    assert out["a"]["significant"] and not out["b"]["significant"]


def test_retrieval_row_metrics():
    row = bench_answers.retrieval_row({"g1", "g2"}, ["x", "g1", "y", "g2", "z"], ["g1", "x"])
    assert row["P@1"] == 0.0 and row["R@3"] == 0.5 and row["R@5"] == 1.0
    assert row["MRR@10"] == 0.5
    ideal = 1 + 1 / math.log2(3)
    assert row["NDCG@10"] == pytest.approx((1 / math.log2(3) + 1 / math.log2(5)) / ideal)
    assert row["ctx_precision"] == 0.5 and row["ctx_recall"] == 0.5 and row["ctx_all"] == 0.0


def test_context_uses_selected_set_or_records_fallback():
    row = {"ranked": ["a", "b", "c", "d"], "selected": ["c"]}
    assert bench_answers.context_ids(row, 2, True) == (["c"], "selected")
    assert bench_answers.context_ids({**row, "selected": []}, 2, True) == (["a", "b"], "fallback")
    assert bench_answers.context_ids(row, 2, False) == (["a", "b"], "top_k")


def test_end_to_end_with_fake_llm(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    chunks = [{"chunk_id": f"c{i}", "doc_id": f"c{i}", "title": "t", "text": f"text {i}"} for i in range(6)]
    questions = [{"qid": f"q{i}", "question": f"question {i}?", "answer": "x", "type": "2hop",
                  "gold_chunk_ids": ["c0", "c1"]} for i in range(3)]
    for name, rows in (("chunks.jsonl", chunks), ("questions.jsonl", questions)):
        (bundle / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ranked = [{"qid": q["qid"], "ranked": [c["chunk_id"] for c in chunks]} for q in questions]
    selected = [{**r, "selected": ["c1", "c0"]} for r in ranked]
    for name, rows in (("a.jsonl", ranked), ("b.jsonl", selected)):
        (tmp_path / name).write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    out = tmp_path / "out"
    env = {"LLM_PROVIDER": "fake", "PYTHONPATH": str(ROOT), "PYTHONIOENCODING": "utf-8",
           "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")}
    proc = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "bench_answers.py"), "--bundle", str(bundle),
         "--run", f"a={tmp_path / 'a.jsonl'}", "--run", f"b={tmp_path / 'b.jsonl'}:selected",
         "--baseline", "a", "--k", "3", "--workers", "2", "--out", str(out)],
        capture_output=True, text=True, encoding="utf-8", env=env, cwd=tmp_path, check=False)
    assert proc.returncode == 0, proc.stderr[-2000:]
    report = json.loads((out / "answers-report.json").read_text(encoding="utf-8"))
    assert report["summary"]["a"]["context_size"] == 3
    assert report["summary"]["b"]["context_size"] == 2
    assert report["summary"]["b"]["sources"] == {"selected": 1.0}
    assert report["summary"]["b"]["retrieval"]["ctx_all"] == 1.0
    assert "b" in report["vs_baseline"] and "p_holm" in report["vs_baseline"]["b"]["em"]
