"""Ошибки сценария должны обнаруживаться без запуска команд проекта."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_day1", ROOT / "scripts/check_day1.py")
checker = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = checker
SPEC.loader.exec_module(checker)


def scenario(tmp_path: Path, text: str) -> Path:
    path = tmp_path / "day1.sh"
    path.write_text(text, encoding="utf-8")
    return path


def test_real_scenario() -> None:
    findings = checker.check(ROOT / "deploy/day1.sh")
    assert not [str(f) for f in findings if not f.unknown]
    # Пробы передаются массивом "${FRESH[@]}", который статически
    # не разворачивается: проверяющий обязан сказать «не знаю», а не
    # считать обязательный аргумент по-настоящему отсутствующим.
    assert any(f.unknown and "probe_compare.py" in f.message for f in findings)


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("uv run rag-textbook ingest --does-not-exist", "неизвестная опция --does-not-exist"),
        ("uv run rag-textbook eval nonexistent", "неизвестная подкоманда"),
        ("uv run rag-textbook ask", "обязательный аргумент question"),
        (
            "env CHUNKER_RESPECT_FORMULSA=true uv run rag-textbook ingest",
            "CHUNKER_RESPECT_FORMULSA",
        ),
        ("export GRAPH_ENABELD=false\nuv run rag-textbook ingest", "GRAPH_ENABELD"),
        (
            'OPTS=(GRAPH_ENABELD=false)\nenv "${OPTS[@]}" uv run rag-textbook ingest',
            "GRAPH_ENABELD",
        ),
        ("python scripts/sample_groups.py --out /tmp/out018", "обязательный аргумент --dataset"),
        ("python scripts/check_parsed_text.py --typo", "неизвестная опция --typo"),
        ("uv run rag-textbook ingest --stages", "нет значения --stages"),
        ("python scripts/reward_fingerprint.py", "обязателен один из аргументов --write, --check"),
    ],
)
def test_contract_errors(tmp_path: Path, text: str, message: str) -> None:
    findings = checker.check(scenario(tmp_path, "step_X1() {\n" + text + "\n}"))
    errors = [f for f in findings if not f.unknown]
    assert any(message in f.message for f in errors)
    assert all(f.step == "X1" and f.line >= 2 for f in errors)
    assert all(str(f.source) in str(f) for f in errors)


def test_missing_input_and_late_producer(tmp_path: Path) -> None:
    missing = (tmp_path / "missing.json").as_posix()
    text = f'uv run rag-textbook goldset audit --path "{missing}"\necho data > "{missing}"'
    findings = checker.check(scenario(tmp_path, text))
    assert any(not f.unknown and "нет файла" in f.message for f in findings)


@pytest.mark.parametrize(
    "producer",
    [
        'echo data > "{path}"',
        'uv run rag-textbook goldset build --output "{path}"',
        'python scripts/sample_groups.py --dataset "{existing}" --out "{path}"',
    ],
)
def test_earlier_producer(tmp_path: Path, producer: str) -> None:
    path = (tmp_path / "made.json").as_posix()
    existing = tmp_path / "existing.jsonl"
    existing.write_text("", encoding="utf-8")
    text = producer.format(path=path, existing=existing.as_posix())
    text += f'\nuv run rag-textbook goldset audit --path "{path}"'
    assert checker.check(scenario(tmp_path, text)) == []


def test_unrelated_output_does_not_hide_missing_input(tmp_path: Path) -> None:
    text = (
        f'echo data > "{tmp_path.as_posix()}/other.json"\n'
        f'uv run rag-textbook goldset audit --path "{tmp_path.as_posix()}/missing.json"'
    )
    assert any(
        not f.unknown and "нет файла" in f.message for f in checker.check(scenario(tmp_path, text))
    )


def test_dynamic_input_is_unknown(tmp_path: Path) -> None:
    findings = checker.check(
        scenario(tmp_path, "python scripts/probe_compare.py runs/probe-*/probe.json")
    )
    assert findings and all(f.unknown for f in findings)


def test_heredoc_and_comments_are_not_commands(tmp_path: Path) -> None:
    text = """# rag-textbook ingest --bad
echo "rag-textbook ingest --bad"
python - <<'PY'
rag-textbook ingest --bad
PY
uv run rag-textbook ingest --stages parse --no-monitor
"""
    findings = checker.check(scenario(tmp_path, text))
    assert findings and all(f.unknown for f in findings)


def test_no_import_or_execution_of_script(tmp_path: Path) -> None:
    # Даже исполняемый код перед ArgumentParser не должен запускаться.
    (tmp_path / "scripts").mkdir()
    for relative in ("rag_textbook/cli/main.py", "rag_textbook/config.py"):
        dest = tmp_path / relative
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text((ROOT / relative).read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "scripts/trap.py").write_text(
        """raise RuntimeError("не импортировать")
import unavailable_heavy_package
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--required", required=True)
""",
        encoding="utf-8",
    )
    findings = checker.check(scenario(tmp_path, "python scripts/trap.py"), root=tmp_path)
    assert any("обязательный аргумент --required" in f.message for f in findings)


def test_exit_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    path = scenario(tmp_path, "rag-textbook ingest --bad")
    monkeypatch.setattr(sys, "argv", ["check_day1.py", str(path)])
    assert checker.main() == 1
    assert "Расхождений: 1" in capsys.readouterr().out
    path.write_text("rag-textbook ingest", encoding="utf-8")
    assert checker.main() == 0
