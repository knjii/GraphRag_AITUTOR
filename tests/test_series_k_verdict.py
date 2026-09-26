"""Вердикт серии К: пороги, срезы, Холм, бюджет контекста."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from rag_textbook.evaluation.metrics import QueryOutcome
from rag_textbook.models import GoldQuestion

_SPEC = importlib.util.spec_from_file_location(
    "series_k_verdict", Path(__file__).resolve().parents[1] / "scripts" / "series_k_verdict.py"
)
sk = importlib.util.module_from_spec(_SPEC)
sys.modules["series_k_verdict"] = sk
_SPEC.loader.exec_module(sk)


def _questions(n: int) -> list[GoldQuestion]:
    items = []
    for i in range(n):
        items.append(GoldQuestion(id=f"l{i}", question="?", gold_chunk_ids=["a", "b"],
                                  question_type="multi_hop", expected_hops=2,
                                  slice="linking", split="test"))
        items.append(GoldQuestion(id=f"s{i}", question="?", gold_chunk_ids=["a"], split="test"))
    return items


def _run(questions, found_second: set[str], chars: int = 100) -> dict[str, QueryOutcome]:
    runs = {}
    for q in questions:
        retrieved = ["a", "b"] if q.id in found_second or len(q.gold_chunk_ids) == 1 else ["a", "x"]
        runs[q.id] = QueryOutcome(q.id, q.question_type, retrieved, list(q.gold_chunk_ids),
                                  context_chars=[chars, chars])
    return runs


def test_k2_accepted_needs_both_slices_and_missing_run_is_not_failure() -> None:
    questions = _questions(60)
    base = _run(questions, set())
    k2 = _run(questions, {f"l{i}" for i in range(30)})
    result = sk.verdict({"base": base, "K2": k2}, questions, "test")
    # Межкнижного среза в наборе нет — К2 не может быть принята без него.
    assert result["decisions"]["К2"] == "отвергнута"
    rows = [r for r in result["tests"] if r["hypothesis"] == "К2"]
    assert rows[0]["status"] == "выполнено" and rows[1]["status"] == "нет вопросов среза"
    assert result["decisions"]["К4"] == "не ставилась"


def test_longer_context_blocks_acceptance_and_dev_is_marked() -> None:
    questions = _questions(60)
    base = _run(questions, set())
    k8 = _run(questions, {f"l{i}" for i in range(30)}, chars=150)
    result = sk.verdict({"base": base, "K8": k8}, questions, "test")
    assert result["tests"][6]["status"] == "не принято: бюджет контекста не уравнен"
    assert "только выбор" in sk.verdict({"base": base}, questions, "dev")["decisions"]["К1"]


def test_holm_is_monotone() -> None:
    adjusted = sk.holm({0: 0.01, 1: 0.04, 2: 0.03})
    assert adjusted[0] == 0.03 and adjusted[2] == 0.06 and adjusted[1] == 0.06
