import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "bench_metrics", Path(__file__).resolve().parents[1] / "scripts" / "bench_metrics.py"
)
bench_metrics = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(bench_metrics)


def test_all_at_k_requires_every_gold_chunk():
    row = bench_metrics.per_question(["a", "b"], ["a", "x", "y", "b"], [2, 4])
    assert row["recall@2"] == 0.5
    assert row["all@2"] == 0.0 and row["any@2"] == 1.0
    assert row["all@4"] == 1.0
    assert row["last_rank"] == 4


def test_last_rank_is_none_when_gold_missing():
    row = bench_metrics.per_question(["a", "b"], ["a", "x"], [5])
    assert row["last_rank"] is None
    assert row["recall@pool"] == 0.5 and row["all@pool"] == 0.0


def test_pool_field_overrides_ranking(tmp_path):
    bundle = tmp_path / "b"
    bundle.mkdir()
    (bundle / "questions.jsonl").write_text(
        '{"qid": "q1", "question": "?", "type": "2hop", "gold_chunk_ids": ["a", "b"]}\n',
        encoding="utf-8",
    )
    rankings = tmp_path / "r.jsonl"
    rankings.write_text(
        '{"qid": "q1", "ranked": ["a"], "pool": ["a", "b", "c"]}\n', encoding="utf-8"
    )
    summary, _ = bench_metrics.evaluate(bundle, rankings, [5])
    assert summary["overall"]["all@5"] == 0.0
    assert summary["overall"]["all@pool"] == 1.0
    assert summary["by_type"]["2hop"]["n"] == 1


def test_paired_bootstrap_counts_direction():
    a = {"q1": {"k": 0.0}, "q2": {"k": 1.0}, "q3": {"k": 0.0}}
    b = {"q1": {"k": 1.0}, "q2": {"k": 1.0}, "q3": {"k": 0.0}}
    result = bench_metrics.paired_bootstrap(a, b, "k", n=200)
    assert result["better"] == 1 and result["worse"] == 0
    assert result["delta"] == round(1 / 3, 4)
