"""Разбор across судьи v2: отвергает within, находит заниженные и завышенные случаи."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "judge_v2_review", Path(__file__).resolve().parents[1] / "scripts" / "judge_v2_review.py"
)
jr = importlib.util.module_from_spec(_SPEC)
sys.modules["judge_v2_review"] = jr
_SPEC.loader.exec_module(jr)


def _checks(answer, context=None, refusal=False, contradicts=False):
    return {"facts_in_answer": answer, "facts_in_context": context or [True] * len(answer),
            "refusal": refusal, "contradicts": contradicts, "unsupported": False}


def _data(group="across", row_group="across"):
    rows = [
        {"question_id": "q1", "model": "a", "group": row_group, "human_grade": 3, "correctness": 2,
         "judge_checks": _checks([True, True]), "answer": "полный"},
        {"question_id": "q2", "model": "a", "group": row_group, "human_grade": 2, "correctness": 0,
         "judge_checks": _checks([True, False], contradicts=True), "answer": "занижен"},
        {"question_id": "q3", "model": "b", "group": row_group, "human_grade": 0, "correctness": 2,
         "judge_checks": _checks([True]), "answer": "завышен"},
        {"question_id": "q4", "model": "b", "group": row_group, "human_grade": 1, "correctness": None,
         "judge_checks": None, "answer": ""},
    ]
    return {"judge_provenance": {"calibration_group": group},
            "calibration": {"spearman_across": 0.5}, "outcomes": rows}


def test_review_counts_and_branches():
    result = jr.review(_data(), {"q2": ["факт"]}, show=5)
    assert result["rows"] == 4 and result["invalid"] == 1
    assert result["invalid_share"] == 0.25
    assert result["under"] == 1 and result["under_branches"] == {"противоречие": 1}
    assert result["over"] == 1 and result["over_branches"] == {"факты 1/1": 1}
    assert result["table"] == {"0->2": 1, "2->0": 1, "3->2": 1}
    kinds = [(c["kind"], c["question_id"]) for c in result["cases"]]
    assert kinds == [("занижен", "q2"), ("завышен", "q3")]
    assert result["cases"][0]["facts"] == ["факт"]


@pytest.mark.parametrize("group,row_group", [("within", "within"), ("across", "within")])
def test_within_is_refused(group, row_group):
    with pytest.raises(SystemExit, match="только для группы across"):
        jr.review(_data(group, row_group), {}, show=5)


def test_branch_refusal_kinds():
    assert jr.branch(_checks([False], context=[True], refusal=True)) == "отказ при факте во фрагментах"
    assert jr.branch(_checks([False], context=[False], refusal=True)) == "верный отказ"
    assert jr.branch(None) == "невалидно"


def test_main_writes_json(tmp_path, capsys):
    cal = tmp_path / "calibration.json"
    cal.write_text(json.dumps(_data(), ensure_ascii=False), encoding="utf-8")
    out = tmp_path / "r.json"
    assert jr.main([str(cal), "--facts", str(tmp_path / "нет.json"), "--json", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["under"] == 1
    assert "занижен 1" in capsys.readouterr().out
