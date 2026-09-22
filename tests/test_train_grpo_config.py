"""Настройки GRPO подставляются по установленной версии TRL.

Версии расходятся: в TRL 1.13 удалён `max_prompt_length` (задачи 014, 017).
Лишнее поле — падение конструктора на оплаченной карте. Модели ниже —
синтетические, а не копии версий: `vllm_mode` есть уже в 0.24.0
(задача 019), здесь проверяется только сам механизм отбора.
"""

from __future__ import annotations

import dataclasses
import importlib.util
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "train_grpo", Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
)
assert _spec is not None and _spec.loader is not None
train_grpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_grpo)

SETTINGS = {
    "max_steps": 20,
    "loss_type": "dr_grpo",
    "max_prompt_length": 11520,
    "vllm_mode": "colocate",
    "vllm_enable_sleep_mode": True,
}


@dataclasses.dataclass
class _OldConfig:
    max_steps: int = 0
    loss_type: str = "grpo"
    max_prompt_length: int = 512


@dataclasses.dataclass
class _NewConfig:
    max_steps: int = 0
    loss_type: str = "dapo"
    vllm_mode: str = "colocate"
    vllm_enable_sleep_mode: bool = False


def test_old_trl_keeps_prompt_limit_and_drops_vllm_options():
    kept = train_grpo._supported(_OldConfig, SETTINGS)
    assert kept == {"max_steps": 20, "loss_type": "dr_grpo", "max_prompt_length": 11520}
    _OldConfig(**kept)


def test_new_trl_drops_removed_prompt_limit():
    kept = train_grpo._supported(_NewConfig, SETTINGS)
    assert "max_prompt_length" not in kept
    assert kept["vllm_mode"] == "colocate" and kept["vllm_enable_sleep_mode"] is True
    _NewConfig(**kept)


def test_dropped_options_are_reported(capsys):
    train_grpo._supported(_NewConfig, SETTINGS)
    assert "max_prompt_length" in capsys.readouterr().out


def _run(argv, monkeypatch):
    import sys

    monkeypatch.setattr(sys, "argv", ["train_grpo.py", "--dataset", "x.jsonl", *argv])
    return train_grpo.main()


def test_vllm_with_unsloth_is_refused(monkeypatch, capsys):
    import pytest

    with pytest.raises(SystemExit):
        _run(["--vllm"], monkeypatch)
    assert "014" in capsys.readouterr().err


def test_server_mode_without_vllm_is_refused(monkeypatch, capsys):
    import pytest

    with pytest.raises(SystemExit):
        _run(["--backend", "hf", "--vllm-mode", "server"], monkeypatch)
    assert "--vllm-mode server" in capsys.readouterr().err
