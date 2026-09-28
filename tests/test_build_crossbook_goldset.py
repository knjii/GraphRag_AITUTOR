"""Сборка межкнижного эталона: только ok, test ровно N, стратификация, отпечатки."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

from rag_textbook.evaluation.goldset import load_goldset

_SPEC = importlib.util.spec_from_file_location(
    "build_crossbook_goldset", Path(__file__).resolve().parents[1] / "scripts" / "build_crossbook_goldset.py"
)
bx = importlib.util.module_from_spec(_SPEC)
sys.modules["build_crossbook_goldset"] = bx
_SPEC.loader.exec_module(bx)

MML, CH, SOK = "0690bb81b7e3c831", "567b09c6f1443478", "4911cf37c5d9ce23"


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    chunks = tmp_path / "chunks"
    out = tmp_path / "out"
    chunks.mkdir()
    out.mkdir()
    for doc in (MML, CH, SOK):
        rows = [{"id": f"{doc}:{i:05d}", "text_hash": f"h-{doc[:4]}-{i}"} for i in range(200)]
        (chunks / f"{doc}_chunks.json").write_text(json.dumps(rows), encoding="utf-8")
    candidates = []
    # 60 пар MML × Чернова и 30 пар Чернова × Соколов: доли 2:1.
    for i in range(90):
        other = CH if i < 60 else SOK
        first = MML if i < 60 else CH
        candidates.append({"id": f"x-t-{i:02d}", "question": f"вопрос {i}?",
                           "gold_chunk_ids": [f"{first}:{i:05d}", f"{other}:{i + 100:05d}"],
                           "gold_doc_ids": [first, other], "answer": "ответ",
                           "roles": {}, "topic": "т"})
    (out / "t.jsonl").write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates),
                                 encoding="utf-8-sig")
    monkeypatch.setattr(bx, "CANDIDATES", out)
    monkeypatch.setattr(bx, "CHUNKS", chunks)
    monkeypatch.setattr(bx, "REVIEW", tmp_path / "review.json")  # настоящий файл проверки не подхватывать
    return tmp_path, candidates


def _verdicts(tmp_path, candidates, ok):
    rows = [{"question_id": c["id"], "verdict": "ok" if i < ok else "one_enough", "votes": {"ok": 2}}
            for i, c in enumerate(candidates)]
    path = tmp_path / "verdicts.json"
    path.write_text(json.dumps({"verdicts": rows}), encoding="utf-8")
    return path


def test_only_ok_and_exact_test_size(corpus):
    tmp_path, candidates = corpus
    # Принятые — первые 80: 60 MML×Чернова и 20 Чернова×Соколов.
    data = bx.build(60, 7, _verdicts(tmp_path, candidates, 80))
    assert data["count"] == 80
    split = Counter(q["split"] for q in data["questions"])
    assert split == {"test": 60, "dev": 20}
    by_pair = Counter((bx.pair_of(q), q["split"]) for q in data["questions"])
    assert by_pair[("MML × Чернова", "test")] == 45  # 60 · 60/80
    assert by_pair[("Соколов × Чернова", "test")] == 15
    assert all(q["slice"] == "cross_book" and q["expected_hops"] == 2 for q in data["questions"])


def test_split_is_deterministic_and_seeded(corpus):
    tmp_path, candidates = corpus
    verdicts = _verdicts(tmp_path, candidates, 90)
    first = {q["id"]: q["split"] for q in bx.build(60, 1, verdicts)["questions"]}
    again = {q["id"]: q["split"] for q in bx.build(60, 1, verdicts)["questions"]}
    other = {q["id"]: q["split"] for q in bx.build(60, 2, verdicts)["questions"]}
    assert first == again
    assert first != other


def test_too_few_accepted_refuses(corpus):
    tmp_path, candidates = corpus
    with pytest.raises(SystemExit, match="набор не собирается"):
        bx.build(60, 7, _verdicts(tmp_path, candidates, 59))


def test_chunk_hashes_cover_gold(corpus):
    tmp_path, candidates = corpus
    data = bx.build(60, 7, _verdicts(tmp_path, candidates, 70))
    gold = {cid for q in data["questions"] for cid in q["gold_chunk_ids"]}
    assert set(data["chunk_hashes"]) == gold
    assert data["chunk_hashes"][f"{MML}:00003"] == f"h-{MML[:4]}-3"


def test_unknown_chunk_refuses(corpus):
    tmp_path, candidates = corpus
    candidates[0]["gold_chunk_ids"][1] = f"{CH}:09999"
    (bx.CANDIDATES / "t.jsonl").write_text(
        "".join(json.dumps(c, ensure_ascii=False) + "\n" for c in candidates), encoding="utf-8")
    with pytest.raises(SystemExit, match="нет фрагментов"):
        bx.build(60, 7, _verdicts(tmp_path, candidates, 70))


def test_write_freezes_and_loads(corpus):
    tmp_path, candidates = corpus
    target = tmp_path / "goldset-x.json"
    bx.main(["--verdicts", str(_verdicts(tmp_path, candidates, 70)), "--output", str(target), "--write"])
    digest, name = (tmp_path / "goldset-x.accepted").read_text(encoding="utf-8").split()
    assert digest == hashlib.sha256(target.read_bytes()).hexdigest()
    assert name == "goldset-x.json"
    assert len(load_goldset(target)) == 70


def test_review_excludes_accepted(corpus):
    tmp_path, candidates = corpus
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"exclude": {"x-t-00": "причина", "x-t-61": "причина"}}), encoding="utf-8")
    data = bx.build(60, 7, _verdicts(tmp_path, candidates, 80), review)
    ids = {q["id"] for q in data["questions"]}
    assert data["count"] == 78 and not ids & {"x-t-00", "x-t-61"}
    assert Counter(q["split"] for q in data["questions"])["test"] == 60
    assert data["excluded_by_review"] == ["x-t-00", "x-t-61"]


def test_review_cannot_touch_rejected(corpus):
    tmp_path, candidates = corpus
    review = tmp_path / "review.json"
    review.write_text(json.dumps({"exclude": {"x-t-85": "причина"}}), encoding="utf-8")
    with pytest.raises(SystemExit, match="не приняло"):
        bx.build(60, 7, _verdicts(tmp_path, candidates, 80), review)


def test_test_size_below_owner_threshold_is_rejected(corpus):
    with pytest.raises(SystemExit):
        bx.main(["--test-size", "40"])
