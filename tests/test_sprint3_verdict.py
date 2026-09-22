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
    path: Path,
    label: str,
    found: list[int],
    *,
    support: float = 0.66,
    answer: str = PROSE,
    model: str | None = None,
    prompt: str = "qa-v4:abc",
    ids: list[str] | None = None,
) -> None:
    ids = ids if ids is not None else [f"q{i}" for i in range(len(found))]
    outcomes = [
        {
            "question_id": ids[i],
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
                "summary": {
                    "всего": {"вопросов": len(found), "опора на контекст": support},
                    "чем сделано": {
                        "модель ответа": model or f"/runs/{label}",
                        "модель по настройке": model or f"/runs/{label}",
                        "промпт": prompt,
                        "окно контекста": 16384,
                        "контекст": "из слепка",
                    },
                },
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


def test_missing_controls_give_no_verdict(tmp_path: Path):
    run_file(tmp_path, "base", [1] * 30 + [0] * 70)
    run_file(tmp_path, "main", [1] * 45 + [0] * 55)
    assert verdict.main(["--metrics", str(tmp_path), "--base", "base", "--main", "main"]) == 1
    # Один контроль из двух — тоже не вердикт.
    run_file(tmp_path, "random", [1] * 30 + [0] * 70)
    argv = ["--metrics", str(tmp_path), "--base", "base", "--main", "main", "--controls", "random"]
    assert verdict.main(argv) == 1


def test_different_question_sets_are_refused(tmp_path: Path, capsys):
    scenario(tmp_path, [1] * 45 + [0] * 55)
    run_file(tmp_path, "main", [1] * 45 + [0] * 5)
    assert call(tmp_path) == 1
    assert "другой набор вопросов" in capsys.readouterr().err


def test_missing_run_is_named(tmp_path: Path):
    with pytest.raises(verdict.BadInputs, match="нет-такого"):
        verdict.load(tmp_path, "нет-такого")


# --- находки задачи 021 ---------------------------------------------------------


def test_stale_file_with_suffix_is_not_used(tmp_path: Path):
    """Свежий main равен базе, рядом старый main_old с приростом."""
    scenario(tmp_path, [1] * 30 + [0] * 70)
    run_file(tmp_path, "main_old", [1] * 45 + [0] * 55, model="/runs/old")
    assert call(tmp_path) == 2


def test_threshold_is_checked_before_rounding(tmp_path: Path):
    """21/382 = 0.05497 округлялось до 0.055 и проходило."""
    base = [1] * 76 + [0] * 306
    main = [1] * 97 + [0] * 285
    scenario(tmp_path, main, base=base, random_=base, control=base)
    assert call(tmp_path) == 2


def test_duplicate_questions_are_refused(tmp_path: Path):
    scenario(tmp_path, [1] * 45 + [0] * 55)
    ids = [f"q{i}" for i in range(99)] + ["q0"]
    run_file(tmp_path, "main", [1] * 45 + [0] * 55, ids=ids)
    assert call(tmp_path) == 1


def test_same_served_model_for_two_runs_is_refused(tmp_path: Path, capsys):
    """Переключение модели не сработало — кандидат отвечал базой."""
    scenario(tmp_path, [1] * 45 + [0] * 55)
    run_file(tmp_path, "main", [1] * 45 + [0] * 55, model="/runs/base")
    assert call(tmp_path) == 1
    assert "одна модель" in capsys.readouterr().err


def test_different_prompts_are_refused(tmp_path: Path):
    scenario(tmp_path, [1] * 45 + [0] * 55)
    run_file(tmp_path, "main", [1] * 45 + [0] * 55, prompt="qa-v3:zzz")
    assert call(tmp_path) == 1


def test_failed_run_removes_the_old_verdict(tmp_path: Path):
    out = tmp_path / "verdict.json"
    out.write_text('{"вердикт": "принято"}', encoding="utf-8")
    run_file(tmp_path, "base", [1] * 30 + [0] * 70)
    assert call(tmp_path, out) == 1
    assert not out.exists()


def test_report_is_written(tmp_path: Path):
    out = tmp_path / "verdict.json"
    call(scenario(tmp_path, [1] * 45 + [0] * 55), out)
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["вердикт"] == "принято"
    assert report["прогоны"]["main"]["прирост"] == pytest.approx(0.15, abs=0.001)


def test_hack_share_ignores_prose_with_one_formula():
    run = {"outcomes": [{"answer": FORMULA}], "summary": {"всего": {}}}
    assert verdict.hack_share(run) < verdict.MAX_HACK_SHARE


def _edit(path: Path, label: str, change) -> None:
    file = path / f"answers_{label}.json"
    data = json.loads(file.read_text(encoding="utf-8"))
    change(data["summary"])
    file.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_missing_provenance_fields_everywhere_are_refused(tmp_path: Path, capsys):
    # Задача 022: без поля во всех четырёх замерах сравнение шло None == None.
    path = scenario(tmp_path, [1] * 45 + [0] * 55)
    for label in ("base", "main", "random", "control"):
        _edit(path, label, lambda summary: summary["чем сделано"].pop("промпт"))
    assert call(path) == 1
    assert "промпт" in capsys.readouterr().err


@pytest.mark.parametrize("value", ["inf", float("nan"), 1.5, True])
def test_support_must_be_a_share(tmp_path: Path, value):
    path = scenario(tmp_path, [1] * 45 + [0] * 55)
    _edit(path, "main", lambda summary: summary["всего"].update({"опора на контекст": value}))
    assert call(path) == 1


@pytest.mark.parametrize("window", ["16384", 16384.0, None])
def test_window_must_be_an_integer(tmp_path: Path, window):
    path = scenario(tmp_path, [1] * 45 + [0] * 55)
    _edit(path, "main", lambda summary: summary["чем сделано"].update({"окно контекста": window}))
    assert call(path) == 1


def test_control_equal_to_base_is_refused_without_traceback(tmp_path: Path, capsys):
    path = scenario(tmp_path, [1] * 45 + [0] * 55)
    argv = ["--metrics", str(path), "--base", "base", "--main", "main",
            "--controls", "base", "random"]
    assert verdict.main(argv) == 1
    assert "различаться" in capsys.readouterr().err
