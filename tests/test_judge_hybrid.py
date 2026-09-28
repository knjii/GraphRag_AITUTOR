"""Гибридный судья r1: v1 поправляется проверками v2 только в трёх ветках."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_hybrid", Path(__file__).resolve().parents[1] / "scripts" / "judge_hybrid.py"
)
jh = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_hybrid"] = jh
_SPEC.loader.exec_module(jh)


def checks(refusal=False, contradicts=False, ctx=(True,), ans=(True,)):
    return {"refusal": refusal, "contradicts": contradicts,
            "facts_in_context": list(ctx), "facts_in_answer": list(ans)}


def test_branches():
    assert jh.hybrid(1, None) == 1
    assert jh.hybrid(None, checks()) is None
    assert jh.hybrid(0, checks(refusal=True, ctx=(False,))) == 2
    assert jh.hybrid(2, checks(refusal=True, ctx=(True, False))) == 0
    assert jh.hybrid(2, checks(contradicts=True)) == 0
    assert jh.hybrid(2, checks(ans=(False, False))) == 1
    assert jh.hybrid(1, checks(ans=(False,))) == 1
    assert jh.hybrid(2, checks()) == 2


def test_combine_pairs_by_key_and_refuses_mismatched_grades():
    v1 = [{"question_id": "q", "model": "m", "group": "within", "human_grade": 3, "correctness": 2}]
    v2 = [{"question_id": "q", "model": "m", "group": "within", "human_grade": 3,
           "judge_checks": checks(ans=(False,))}]
    rows, stats = jh.combine(v1, v2, "within")
    assert rows[0]["correctness"] == 1 and stats["изменено v2"] == 1
    with pytest.raises(SystemExit):
        jh.combine(v1, [{**v2[0], "human_grade": 0}], "within")
