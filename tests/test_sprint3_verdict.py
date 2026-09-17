"""Вердикт спринта 3 применяет критерии, а не пересказывает их.

Смысл тестов — закрепить, что порог нельзя пройти «почти»: прирост ниже
записанного, интервал через ноль, контроль, догнавший основной прогон,
и выписка из формул вместо ответа должны давать «не принято».
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "sprint3_verdict", ROOT / "scripts" / "sprint3_verdict.py"
)
verdict = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(verdict)

FORMULA = "Ответ: $$ A x = b $$ — как в источнике [1], где матрица невырождена."
PROSE = "Ответ дан словами, без переноса выражения из источника, но по существу."


def run_file(
    path: Path, label: str, found: list[int], *, support: float = 0.66, answer: str = PROSE
) -> None:
    outcomes = [
        {
            "question_id": f"q{i}",
            "question_type": "formula_table",
            "answer": answer,
            "latex_expected": 2,
            "latex_found": value,
        }
        for i, value in enumerate(found)
    ]
    (path / f"answers_{label}.json").write_text(
        json.dumps(
            {
                "label": label,
                "summary": {"всего": {"вопросов": len(found), "опора на контекст": support}},
                "outcomes": outcomes,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


def scenario(
    tmp_path: Path,
    main: list[int],
    *,
    random_: list[int] | None = None,
    control: list[int] | None = None,
    base: list[int] | None = None,
    **kwargs,
) -> Path:
    base = base if base is not None else [1] * 30 + [0] * 70
    run_file(tmp_path, "base", base)
    run_file(tmp_path, "main", main, **kwargs)
    run_file(tmp_path, "random", random_ if random_ is not None else [1] * 31 + [0] * 69)
    run_file(tmp_path, "control", control if control is not None else [1] * 30 + [0] * 70)
    return tmp_path


def call(path: Path, out: Path | None = None) -> int:
    argv = [
        "--metrics",
        str(path),
        "--base",
        "base",
        "--main",
        "main",
        "--controls",
        "random",
        "control",
    ]
    if out:
        argv += ["--out", str(out)]
    return verdict.main(argv)


def test_clear_gain_is_accepted(tmp_path: Path):
    assert call(scenario(tmp_path, [1] * 45 + [0] * 55)) == 0


def test_gain_below_the_threshold_is_refused(tmp_path: Path, capsys):
    # +0.04 — меньше записанных +0.055, даже если интервал не через ноль.
    assert call(scenario(tmp_path, [1] * 34 + [0] * 66)) == 2
    assert "НЕТ прирост" in capsys.readouterr().out


def test_control_catching_up_refuses(tmp_path: Path):
    # Случайная награда дала почти то же — значит, учит не награда.
    assert call(scenario(tmp_path, [1] * 45 + [0] * 55, random_=[1] * 43 + [0] * 57)) == 2


def test_answers_made_of_formulas_are_counted_as_hacking(tmp_path: Path, capsys):
    path = scenario(tmp_path, [1] * 45 + [0] * 55, answer="$$ A x = b $$")
    assert call(path) == 2
    assert "НЕТ взлом" in capsys.readouterr().out


def test_support_below_base_refuses(tmp_path: Path):
    path = scenario(tmp_path, [1] * 45 + [0] * 55, support=0.40)
    assert call(path) == 2


def test_missing_controls_refuse_even_a_good_run(tmp_path: Path):
    run_file(tmp_path, "base", [1] * 30 + [0] * 70)
    run_file(tmp_path, "main", [1] * 45 + [0] * 55)
    assert verdict.main(["--metrics", str(tmp_path), "--base", "base", "--main", "main"]) == 2


def test_different_question_sets_are_refused(tmp_path: Path):
    run_file(tmp_path, "base", [1] * 30 + [0] * 70)
    run_file(tmp_path, "main", [1] * 45 + [0] * 5)
    with pytest.raises(ValueError):
        verdict.main(["--metrics", str(tmp_path), "--base", "base", "--main", "main"])


def test_missing_run_is_named(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        verdict.load(tmp_path, "нет-такого")


def test_report_is_written(tmp_path: Path):
    out = tmp_path / "verdict.json"
    call(scenario(tmp_path, [1] * 45 + [0] * 55), out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["вердикт"] == "принято"
    assert report["прогоны"]["main"]["прирост"] == pytest.approx(0.15, abs=0.001)


def test_hack_share_ignores_prose_with_one_formula():
    run = {"outcomes": [{"answer": FORMULA}], "summary": {"всего": {}}}
    assert verdict.hack_share(run) < verdict.MAX_HACK_SHARE
