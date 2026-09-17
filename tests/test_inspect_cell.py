"""Проверка ячеек замера без внешних сервисов и готовых артефактов."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

_spec = importlib.util.spec_from_file_location(
    "inspect_cell", Path(__file__).resolve().parents[1] / "scripts" / "inspect_cell.py"
)
assert _spec is not None and _spec.loader is not None
inspect_cell = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(inspect_cell)


def _cell(tmp_path: Path, **overrides: object) -> Path:
    data = {
        "outcomes": [{"answer": "Ответ"}],
        "summary": {"чем сделано": {
            "модель ответа": "test-model",
            "модель по настройке": "configured-model",
            "промпт": "test-prompt",
            "окно контекста": 4096,
        }},
    }
    data.update(overrides)
    path = tmp_path / "cell.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def test_valid_cell_preserves_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _cell(tmp_path)
    assert inspect_cell.inspect(path, expected=1, max_empty_share=0.05) == []
    expected_output = (
        "cell.json\n"
        "   ответов 1, пустых 0\n"
        "   модель ответа: test-model\n"
        "   по настройке:  configured-model\n"
        "   промпт: test-prompt, окно: 4096\n"
    )
    assert capsys.readouterr().out == expected_output
    assert inspect_cell.main([str(path)]) == 0
    assert capsys.readouterr().out == expected_output


@pytest.mark.parametrize("kind", ["missing", "directory", "json", "encoding"])
def test_unreadable_cell(tmp_path: Path, capsys: pytest.CaptureFixture[str], kind: str) -> None:
    path = tmp_path / "broken.json"
    if kind == "directory":
        path.mkdir()
    elif kind == "json":
        path.write_text("{", encoding="utf-8")
    elif kind == "encoding":
        path.write_bytes(b"\xff")
    problems = inspect_cell.inspect(path, expected=None, max_empty_share=0.05)
    assert len(problems) == 1
    assert "не удалось прочитать JSON" in problems[0]
    assert inspect_cell.main([str(path)]) == 1
    assert "   НЕГОДЕН: " in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "args", "message"),
    [
        ({"outcomes": []}, [], "нет ни одного ответа"),
        ({}, ["--expected", "2"], "число ответов 1 не равно ожидаемому 2"),
        ({"outcomes": [{"answer": " \n\t"}]}, [], "доля пустых ответов"),
        ({"outcomes": [{}]}, [], "доля пустых ответов"),
        ({"summary": {}}, [], "в метаданных нет модели ответа"),
        ({"summary": {"чем сделано": {}}}, [], "в метаданных нет модели ответа"),
        ({"summary": {"чем сделано": {"модель ответа": " "}}}, [], "в метаданных нет модели ответа"),
        ({"summary": {"чем сделано": {"модель ответа": None}}}, [], "в метаданных нет модели ответа"),
    ],
)
def test_rejected_cell(
    tmp_path: Path, capsys: pytest.CaptureFixture[str],
    overrides: dict[str, object], args: list[str], message: str,
) -> None:
    path = _cell(tmp_path, **overrides)
    expected = int(args[1]) if args else None
    problems = inspect_cell.inspect(path, expected=expected, max_empty_share=0.05)
    assert any(message in problem for problem in problems)
    assert inspect_cell.main([str(path), *args]) == 1
    assert f"   НЕГОДЕН: {'; '.join(problems)}\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ([1, 2], "корень должен быть объектом"),
        (None, "корень должен быть объектом"),
        ("text", "корень должен быть объектом"),
        ({"outcomes": {}}, "outcomes должен быть списком"),
        ({"outcomes": None}, "outcomes должен быть списком"),
        ({"outcomes": "text"}, "outcomes должен быть списком"),
        ({"outcomes": [1]}, "outcomes[0] должен быть объектом"),
        ({"outcomes": [None]}, "outcomes[0] должен быть объектом"),
        ({"outcomes": [{"answer": "Ответ"}, []]}, "outcomes[1] должен быть объектом"),
        ({"summary": []}, "summary должен быть объектом"),
        ({"summary": None}, "summary должен быть объектом"),
        ({"summary": {"чем сделано": []}}, "summary / чем сделано должен быть объектом"),
        ({"summary": {"чем сделано": None}}, "summary / чем сделано должен быть объектом"),
    ],
)
def test_invalid_structure_does_not_stop_next_file(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], data: object, message: str,
) -> None:
    bad = tmp_path / "broken.json"
    bad.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    good = _cell(tmp_path)
    problem = f"неверная структура файла: {message}"
    assert inspect_cell.inspect(bad, expected=None, max_empty_share=0.05) == [problem]
    capsys.readouterr()

    assert inspect_cell.main([str(bad), str(good)]) == 1
    output = capsys.readouterr().out
    assert output.startswith(f"broken.json\n   НЕГОДЕН: {problem}\ncell.json\n")
    assert output.count("   НЕГОДЕН: ") == 1
    assert "   ответов 1, пустых 0\n" in output
    assert "   модель ответа: test-model\n" in output


def test_empty_share_boundary_and_override(tmp_path: Path) -> None:
    path = _cell(tmp_path, outcomes=[{"answer": ""}] + [{"answer": "Ответ"}] * 19)
    assert inspect_cell.main([str(path), "--expected", "20"]) == 0
    assert inspect_cell.main([str(path), "--max-empty-share", "0.049"]) == 1
    path = _cell(tmp_path, outcomes=[{"answer": ""}, {"answer": "Ответ"}])
    assert inspect_cell.main([str(path)]) == 1
    assert inspect_cell.main([str(path), "--max-empty-share", "0.5"]) == 0


def test_all_problems_are_printed_together(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = _cell(tmp_path, outcomes=[{}], summary={})
    problems = inspect_cell.inspect(path, expected=2, max_empty_share=0.05)
    assert len(problems) == 3
    assert inspect_cell.main([str(path), "--expected", "2"]) == 1
    assert f"   НЕГОДЕН: {'; '.join(problems)}\n" in capsys.readouterr().out


def test_all_files_are_checked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    good = _cell(tmp_path)
    bad = tmp_path / "missing.json"
    assert inspect_cell.main([str(bad), str(good), str(bad)]) == 1
    output = capsys.readouterr().out
    assert output.count("   НЕГОДЕН: ") == 2
    assert "   модель ответа: test-model\n" in output
    assert inspect_cell.main([str(good), str(good)]) == 0


def test_main_uses_sys_argv(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = _cell(tmp_path)
    monkeypatch.setattr("sys.argv", ["inspect_cell.py", str(path), "--expected", "1"])
    assert inspect_cell.main() == 0


def test_at_least_one_file_is_required() -> None:
    with pytest.raises(SystemExit) as exc:
        inspect_cell.main([])
    assert exc.value.code == 2
