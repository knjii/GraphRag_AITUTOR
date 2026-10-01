"""Проверки пакетного судьи без сети и моделей."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from rag_textbook.evaluation.answers import JUDGE_PROMPT, AnswerOutcome
from rag_textbook.evaluation.trace import QueryTrace, TraceSet
from scripts import judge_saved as js


class Judge:
    def __init__(self, verdict: str = '{"correctness": 2, "groundedness": 1}') -> None:
        self.verdict = verdict
        self.calls: list[str] = []

    def describe_model(self, *, purpose: str) -> dict[str, str]:
        assert purpose == "judge"
        return {"model": "server-model"}

    def chat(self, messages: list[Any], **kwargs: Any) -> str:
        assert kwargs["purpose"] == "judge"
        self.calls.append(messages[0].content)
        return self.verdict


def dump(path: Path, value: Any) -> Path:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture
def sample(tmp_path: Path) -> tuple[list[str], Path]:
    gold = dump(
        tmp_path / "gold.json",
        [
            {
                "id": "q",
                "question": "Вопрос?",
                "answer": "Эталон",
                "gold_chunk_ids": ["c"],
                "gold_doc_ids": ["doc"],
            }
        ],
    )
    trace = tmp_path / "trace.jsonl"
    TraceSet(traces=[QueryTrace(question_id="q", question="Вопрос?", final=["c2", "c"])]).save(
        trace
    )
    chunks = tmp_path / "parsed"
    chunks.mkdir()
    dump(
        chunks / "doc_chunks.json", [{"id": "c", "text": "Первый"}, {"id": "c2", "text": "Второй"}]
    )
    # Чужой корпус даже не должен читаться.
    (chunks / "unrelated_chunks.json").write_text("не JSON", encoding="utf-8")
    source = dump(
        tmp_path / "answers_a.json",
        {
            "label": "a",
            "summary": {"верность": 0, "чем сделано": {"source": "old"}},
            "outcomes": [
                AnswerOutcome(
                    question_id="q", question_type="single_chunk", answer="Ответ", correctness=0
                ).as_dict()
            ],
        },
    )
    return [
        str(source),
        "--goldset",
        str(gold),
        "--trace",
        str(trace),
        "--chunks",
        str(chunks),
        "--output-dir",
        str(tmp_path / "out"),
    ], source


def test_saved_copy_and_provenance(sample: tuple[list[str], Path]) -> None:
    args, source = sample
    before = source.read_bytes()
    judge = Judge()
    assert js.main(args, llm=judge) == 0
    result = js.read_json(source.parent / "out/answers_a_judged.json")
    assert source.read_bytes() == before
    assert result["outcomes"][0]["correctness"] == 2
    assert result["summary"]["всего"]["верность"] == 2
    assert result["summary"]["чем сделано"] == {"source": "old"}
    meta = result["judge_provenance"]
    assert meta["judge_model"] == "server-model"
    assert meta["input_sha256"][str(source)] == hashlib.sha256(before).hexdigest()
    assert meta["judge_prompt_sha256"] == hashlib.sha256(JUDGE_PROMPT.encode()).hexdigest()
    assert "Второй\n\nПервый" in judge.calls[0]
    assert "Вопрос?" in judge.calls[0] and "Эталон" in judge.calls[0]
    # Повторный запуск не затирает результат.
    assert js.main(args, llm=judge) == 1
    assert len(judge.calls) == 1


@pytest.mark.parametrize(
    "verdict",
    [
        "",
        "не JSON",
        "[]",
        "{}",
        '{"correctness": 3, "groundedness": 1}',
        '{"correctness": true, "groundedness": 1}',
        '{"correctness": "2", "groundedness": 1}',
        '{"correctness": 2, "groundedness": -1}',
        '{"correctness": 2, "groundedness": 1, "reason": null}',
    ],
)
def test_invalid_clears_old_grade(sample: tuple[list[str], Path], verdict: str) -> None:
    args, source = sample
    assert js.main(args, llm=Judge(verdict)) == 1
    result = js.read_json(source.parent / "out/answers_a_judged.json")
    assert result["invalid_judge_fraction"] == 1
    assert result["outcomes"][0]["correctness"] is None
    assert "верность" not in result["summary"]["всего"]


@pytest.mark.parametrize(("count", "status"), [(20, 0), (19, 1)])
def test_five_percent_boundary(sample: tuple[list[str], Path], count: int, status: int) -> None:
    args, source = sample
    data = js.read_json(source)
    data["outcomes"] *= count
    dump(source, data)

    class OnceBad(Judge):
        def chat(self, messages: list[Any], **kwargs: Any) -> str:
            result = super().chat(messages, **kwargs)
            return "" if len(self.calls) == 1 else result

    assert js.main([*args, "--workers", "1"], llm=OnceBad()) == status


def test_missing_chunk_fails_before_judge(sample: tuple[list[str], Path]) -> None:
    args, source = sample
    dump(source.parent / "parsed/doc_chunks.json", [{"id": "c", "text": "Первый"}])
    judge = Judge()
    assert js.main(args, llm=judge) == 1
    assert not judge.calls


def test_multiple_files(sample: tuple[list[str], Path]) -> None:
    args, source = sample
    second = dump(source.parent / "answers_b.json", js.read_json(source))
    originals = [p.read_bytes() for p in (source, second)]
    judge = Judge()
    assert js.main([str(second), *args], llm=judge) == 0
    assert len(judge.calls) == 2
    assert [p.read_bytes() for p in (source, second)] == originals
    assert (
        js.read_json(source.parent / "out/answers_b_judged.json")["outcomes"][0]["groundedness"]
        == 1
    )


def test_server_error_is_invalid(sample: tuple[list[str], Path]) -> None:
    class BrokenJudge(Judge):
        def chat(self, messages: list[Any], **kwargs: Any) -> str:
            raise RuntimeError("Сервер недоступен")

    args, source = sample
    assert js.main(args, llm=BrokenJudge()) == 1
    assert js.read_json(source.parent / "out/answers_a_judged.json")["invalid_judge_fraction"] == 1


def test_calibration_metrics() -> None:
    rows = [
        dict(group="within", question_id="q", human_grade=h, correctness=j)
        for h, j in [(0, 0), (1, 1), (2, 1)]
    ]
    rows.extend(
        dict(group="across", question_id=str(h), human_grade=h, correctness=2 - h) for h in range(3)
    )
    metrics = js.calibration_metrics(rows)
    assert metrics["within_concordance"] == pytest.approx(5 / 6)
    assert metrics["within_pairs"] == 3
    assert metrics["within_ci95"] == pytest.approx([5 / 6, 5 / 6])
    assert metrics["accepted"]
    assert metrics["spearman_across"] == pytest.approx(-1)
    assert js.spearman([(0, 0), (1, 1), (2, 1)]) == pytest.approx(0.8660254)
    assert js.spearman([(0, 1), (1, 1)]) is None
    assert not js.calibration_metrics([])["accepted"]


def test_lower_bound_must_exceed_random() -> None:
    rows = [dict(group="within", question_id="q", human_grade=h, correctness=1) for h in range(4)]
    result = js.calibration_metrics(rows)
    assert result["within_ci95"] == [0.5, 0.5]
    assert not result["accepted"]


# Калибровочные данные (ручные оценки и слепок сессии) в репозиторий не входят.
@pytest.mark.skipif(
    not (js.ROOT / "evaluation/reward_checks/2026-09-17").exists()
    or not (js.ROOT / "capture/session-0903").exists(),
    reason="нет локальных калибровочных данных",
)
def test_real_calibration_dry_run(tmp_path: Path) -> None:
    root = js.ROOT
    rows, _ = js.restore_calibration(
        root / "evaluation/reward_checks/2026-09-17", root / "capture/session-0903"
    )
    assert len(rows) == 110
    assert len({(r["question_id"], r["model"]) for r in rows}) == 110
    judge = Judge()
    assert (
        js.main(
            [
                "--calibrate",
                "--dry-run",
                "--goldset",
                str(root / "capture/goldset.json"),
                "--trace",
                str(root / "capture/session-0819/trace-always.jsonl"),
                "--chunks",
                str(root / "capture"),
                "--output-dir",
                str(tmp_path),
            ],
            llm=judge,
        )
        == 0
    )
    result = js.read_json(tmp_path / "calibration.json")
    assert result["restored_answers"] == len(judge.calls) == 110
    assert result["restored_questions"] == 65
    assert result["calibration"]["valid_within"] == 60
    assert result["calibration"]["valid_across"] == 50
    assert not result["calibration"]["accepted"]
    assert all(row["answer"].strip() for row in result["outcomes"])
