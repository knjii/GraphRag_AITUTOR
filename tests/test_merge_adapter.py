"""Вплавление адаптера: проверки, которые можно сделать без весов.

Полный прогон требует torch и базовой модели, поэтому здесь закреплено
то, что решается до загрузки: откуда берётся базовая модель, что
считается «адаптер ничего не изменил» и почему нельзя писать в непустой
каталог.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "merge_adapter", ROOT / "scripts" / "merge_adapter.py"
)
merge_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_adapter)


def adapter_dir(tmp_path: Path, **config) -> Path:
    path = tmp_path / "adapter"
    path.mkdir()
    (path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    return path


def test_base_model_comes_from_the_adapter(tmp_path: Path):
    path = adapter_dir(tmp_path, base_model_name_or_path="Qwen/Qwen3.5-4B")
    assert merge_adapter.base_model_of(path) == "Qwen/Qwen3.5-4B"


def test_directory_without_adapter_config_is_refused(tmp_path: Path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileNotFoundError):
        merge_adapter.base_model_of(tmp_path / "runs")


def test_adapter_without_base_model_is_refused(tmp_path: Path):
    with pytest.raises(ValueError):
        merge_adapter.base_model_of(adapter_dir(tmp_path, r=16))


def test_non_empty_output_is_refused(tmp_path: Path, capsys):
    path = adapter_dir(tmp_path, base_model_name_or_path="Qwen/Qwen3.5-4B")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    # Молчаливая дозапись в чужой каталог дала бы смесь двух моделей.
    assert merge_adapter.main(["--adapter", str(path), "--out", str(out)]) == 1
    assert "не пуст" in capsys.readouterr().err


def test_changed_parameters_counts_differences():
    torch = pytest.importorskip("torch")
    before = [("q", torch.zeros(2, 2)), ("v", torch.zeros(2, 2))]
    same = [("q", torch.zeros(2, 2)), ("v", torch.zeros(2, 2))]
    other = [("q", torch.ones(2, 2)), ("v", torch.zeros(2, 2))]
    assert merge_adapter.changed_parameters(before, same) == 0
    assert merge_adapter.changed_parameters(before, other) == 1
