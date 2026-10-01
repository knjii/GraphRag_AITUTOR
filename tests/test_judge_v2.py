"""Атомарный судья и разделение калибровки на синтетических данных."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rag_textbook.evaluation import answers as a
from rag_textbook.evaluation.trace import QueryTrace, TraceSet
from scripts import judge_saved as js


def verdict(count: int = 2, **changes: Any) -> dict[str, Any]:
    return {"facts_in_answer": [True] * count, "facts_in_context": [True] * count,
            "refusal": False, "contradicts": False, "unsupported": False, **changes}


class Judge:
    def __init__(self, raw: str | None = None) -> None:
        self.raw = raw if raw is not None else json.dumps(verdict())
        self.calls: list[str] = []

    def chat(self, messages: Any, **kwargs: Any) -> str:
        assert kwargs["purpose"] == "judge"
        self.calls.append(messages[0].content)
        return self.raw


@pytest.mark.parametrize(("changes", "correctness", "groundedness"), [
    ({"refusal": True, "facts_in_context": [False, False]}, 2, 2),
    ({"refusal": True, "facts_in_context": [True, False]}, 0, 2),
    ({"refusal": True, "unsupported": True, "contradicts": True,
      "facts_in_context": [False, False]}, 2, 2),
    ({"contradicts": True}, 0, 2),
    ({"facts_in_answer": [False, False]}, 0, 2),
    ({"facts_in_answer": [True, False]}, 1, 2),
    ({}, 2, 2),
    ({"unsupported": True}, 2, 1),
    ({"contradicts": True, "unsupported": True}, 0, 1),
])
def test_score_branches(changes: dict[str, Any], correctness: int, groundedness: int) -> None:
    result = a.score_judge_v2(verdict(**changes), fact_count=2)
    assert (result["correctness"], result["groundedness"]) == (correctness, groundedness)


@pytest.mark.parametrize(("count", "found", "expected"), [
    (1, 0, 0), (1, 1, 2), (3, 1, 0), (3, 2, 1), (3, 3, 2),
    (4, 1, 0), (4, 2, 1), (4, 3, 1), (4, 4, 2),
])
def test_fraction(count: int, found: int, expected: int) -> None:
    checks = verdict(count, facts_in_answer=[True] * found + [False] * (count - found))
    assert a.score_judge_v2(checks, fact_count=count)["correctness"] == expected


@pytest.mark.parametrize("key", ["facts_in_answer", "facts_in_context"])
@pytest.mark.parametrize("values", [[], [True], [True] * 3, [1, 0], ["да", "нет"], None])
def test_bad_arrays(key: str, values: Any) -> None:
    assert a.score_judge_v2(verdict(**{key: values}), fact_count=2) == {}


@pytest.mark.parametrize("key", ["refusal", "contradicts", "unsupported"])
@pytest.mark.parametrize("value", [0, 1, "false", None])
def test_strict_booleans(key: str, value: Any) -> None:
    assert a.score_judge_v2(verdict(**{key: value}), fact_count=2) == {}


@pytest.mark.parametrize("raw", ["", "не JSON", "[]", "{}", "null"])
def test_invalid_json(raw: str) -> None:
    assert a.judge_answer_v2(Judge(raw), question="?", answer="!", context="", facts=["Факт"]) == {}


def test_latex_and_full_texts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Дополнительное поле имитирует модель, повторившую формулу с одной косой чертой.
    raw = r'{"facts_in_answer":[true,true],"facts_in_context":[true,true],"refusal":false,"contradicts":false,"unsupported":false,"fact":"\frac{1}{2}\alpha"}'
    original = a.loads_llm_json
    parsed = []

    def capture(text: str) -> Any:
        result = original(text)
        parsed.append(result)
        return result

    monkeypatch.setattr(a, "loads_llm_json", capture)
    judge = Judge("```json\n" + raw + "\n```")
    facts = [r"$\frac{1}{2}$", r"$\alpha$"]
    result = a.judge_answer_v2(judge, question="?", answer="а" * 2100 + "КОНЕЦ ОТВЕТА",
                             context="б" * 6100 + "КОНЕЦ КОНТЕКСТА", facts=facts)
    assert result["correctness"] == 2
    assert parsed[0]["fact"] == r"\frac{1}{2}\alpha"
    assert all(fact in judge.calls[0] for fact in facts)
    assert "КОНЕЦ ОТВЕТА" in judge.calls[0] and "КОНЕЦ КОНТЕКСТА" in judge.calls[0]


def dump(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def inputs(tmp_path: Path) -> list[str]:
    gold = dump(tmp_path / "gold.json", [{"id": "q", "question": "Вопрос?", "answer": "Эталон",
                                        "gold_chunk_ids": ["c"], "gold_doc_ids": ["doc"]}])
    chunks = dump(tmp_path / "chunks.json", [{"id": "c", "text": "Контекст"}])
    trace = tmp_path / "trace.jsonl"
    TraceSet(traces=[QueryTrace(question_id="q", question="Вопрос?", final=["c"])]).save(trace)
    source = dump(tmp_path / "answers.json", {"outcomes": [
        a.AnswerOutcome(question_id="q", question_type="single_chunk", answer="Ответ").as_dict()]})
    return [str(source), "--goldset", str(gold), "--trace", str(trace), "--chunks", str(chunks),
            "--output-dir", str(tmp_path / "out")]


@pytest.mark.parametrize("explicit", [False, True])
def test_v1_prompt_unchanged(inputs: list[str], explicit: bool) -> None:
    judge = Judge('{"correctness":2,"groundedness":2}')
    assert js.main(inputs + (["--judge-version", "v1"] if explicit else []), llm=judge) == 0
    expected = a.JUDGE_PROMPT.format(question="Вопрос?", answer="Ответ", context="Контекст",
                                     reference="Эталон")
    assert judge.calls[0].encode("utf-8") == expected.encode("utf-8")


@pytest.mark.parametrize("valid", [False, True])
def test_cli_v2(inputs: list[str], tmp_path: Path, valid: bool) -> None:
    facts = dump(tmp_path / "facts.json", {"q": ["Факт 1", "Факт 2"]})
    judge = Judge(json.dumps(verdict(2 if valid else 1)))
    assert js.main(inputs + ["--judge-version", "v2", "--facts", str(facts)], llm=judge) == int(not valid)
    result = js.read_json(tmp_path / "out/answers_judged.json")
    row = result["outcomes"][0]
    assert row["correctness"] == (2 if valid else None)
    assert row["judge_checks"] == (verdict() if valid else None)
    meta = result["judge_provenance"]
    assert meta["judge_version"] == "v2"
    assert meta["facts_sha256"] == hashlib.sha256(facts.read_bytes()).hexdigest()
    assert meta["judge_prompt_sha256"] == hashlib.sha256(a.JUDGE_PROMPT_V2.encode()).hexdigest()


@pytest.mark.parametrize("facts", [{}, {"q": []}, {"q": [""]}, {"q": "Факт"},
                                   {"q": [True]}, {"q": ["Факт"] * 5}, []])
def test_bad_facts_before_model(inputs: list[str], tmp_path: Path, facts: Any) -> None:
    path = dump(tmp_path / "facts.json", facts)
    judge = Judge()
    assert js.main(inputs + ["--judge-version", "v2", "--facts", str(path)], llm=judge) == 1
    assert judge.calls == []


@pytest.mark.parametrize("group", ["across", "within", "all"])
def test_group_isolation(tmp_path: Path, group: str, monkeypatch: pytest.MonkeyPatch) -> None:
    selected = ("within", "across") if group == "all" else (group,)
    for name in selected:
        dump(tmp_path / f"{name}-key.json", [{"id": 1, "qid": name, "model": name}])
        dump(tmp_path / f"{name}-grades.json", {"1": 2})
        dump(tmp_path / f"answers_model-{name}-w16384.json",
             {"outcomes": [{"question_id": name, "answer": "Синтетический ответ"}]})
    original = js.read_json
    reads: list[Path] = []

    def read(path: Path) -> Any:
        reads.append(path)
        return original(path)

    monkeypatch.setattr(js, "read_json", read)
    rows, sources = js.restore_calibration(tmp_path, tmp_path, group=group)
    assert {row["group"] for row in rows} == set(selected)
    assert set(reads) == set(sources)
    assert len(reads) == 3 * len(selected)


@pytest.mark.parametrize("dry", [False, True])
def test_across_cli(inputs: list[str], tmp_path: Path, dry: bool) -> None:
    dump(tmp_path / "across-key.json", [{"id": 1, "qid": "q", "model": "fake"}])
    dump(tmp_path / "across-grades.json", {"1": 3})
    dump(tmp_path / "answers_model-fake-w16384.json",
         {"outcomes": [{"question_id": "q", "answer": "Ответ"}]})
    facts = dump(tmp_path / "facts.json", {"q": ["Факт 1", "Факт 2"]})
    args = [*inputs[1:], "--calibrate", "--group", "across", "--checks", str(tmp_path),
            "--answers-dir", str(tmp_path), "--judge-version", "v2", "--facts", str(facts)]
    assert js.main(args + (["--dry-run"] if dry else []), llm=None if dry else Judge()) == 0
    result = js.read_json(tmp_path / "out/calibration.json")
    assert result["calibration"]["valid_within"] == 0
    assert result["calibration"]["valid_across"] == 1
    assert not result["calibration"]["accepted"]
    assert result["judge_provenance"]["calibration_group"] == "across"


@pytest.mark.skipif(not (Path(__file__).resolve().parents[1] / "evaluation/reward_checks").exists(),
                    reason="нет локальных калибровочных данных")
def test_fact_coverage() -> None:
    root = Path(__file__).resolve().parents[1]
    # Только ключи: ни оценки, ни ответы калибровки не читаются.
    qids = {row["qid"] for group in ("within", "across") for row in js.read_json(
        root / f"evaluation/reward_checks/2026-09-17/{group}-key.json")}
    facts = js.read_json(root / "evaluation/judge_facts/calibration.json")
    assert len(qids) == 65 and set(facts) == qids
    assert all(1 <= len(values) <= 4 for values in facts.values())
    assert all(isinstance(fact, str) and fact.strip() and "???" not in fact
               and not any(ord(char) < 32 for char in fact)
               for values in facts.values() for fact in values)
