"""Правило выбора связки обучения проверяется до аренды, а не после."""

from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "probe_compare", ROOT / "scripts" / "probe_compare.py"
)
probe_compare = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(probe_compare)


def probe(
    stack: str,
    seconds: float,
    *,
    gens: int = 4,
    completion: int = 768,
    reserved: float = 18.0,
    total: float = 24.0,
) -> dict:
    return {
        "stack": stack,
        "seconds_per_step": seconds,
        "num_generations": gens,
        "max_completion_length": completion,
        "peak_memory_gib": reserved - 1,
        "reserved_memory_gib": reserved,
        "device_memory_gib": total,
    }


def stacks(*probes: dict) -> dict:
    return probe_compare.split_by_stack(list(probes))


def test_faster_stack_wins_when_gain_is_above_noise():
    result = probe_compare.verdict(stacks(probe("vllm", 52.0), probe("unsloth", 71.0)))
    assert any("выбор: vllm:g4" in line and "27%" in line for line in result), result


def test_close_times_are_decided_by_headroom():
    # 52 против 56 — 7%, ниже порога: решает запас памяти.
    result = probe_compare.verdict(
        stacks(probe("vllm", 52.0, reserved=22.5), probe("unsloth", 56.0, reserved=15.0))
    )
    assert any("решает запас памяти" in line for line in result), result
    assert any("выбор: unsloth:g4" in line for line in result), result


def test_stack_that_needs_fewer_generations_is_dropped():
    # Две генерации дают беднее сигнал, чем четыре, — скорость это не окупает.
    result = probe_compare.verdict(
        stacks(probe("vllm", 30.0, gens=2), probe("unsloth", 71.0, gens=4))
    )
    assert any("выбор: unsloth:g4" in line for line in result), result


def test_no_stack_survives():
    result = probe_compare.verdict(stacks(probe("vllm", 30.0, gens=1)))
    assert "ни одна связка не прошла пробу" in result[0]


def test_generation_share_needs_a_short_probe():
    only_full = stacks(probe("vllm", 52.0))
    assert probe_compare.generation_share(only_full["vllm:g4"]) is None
    assert "не измерена" in probe_compare.second_card(only_full)[0]

    both = stacks(probe("vllm", 52.0), probe("vllm", 39.0, completion=96))
    share = probe_compare.generation_share(both["vllm:g4"])
    assert abs(share - 0.25) < 0.01


def test_second_card_verdict_follows_the_share():
    cheap = stacks(probe("vllm", 52.0, reserved=18.0), probe("vllm", 46.0, completion=96))
    assert any("не окупается" in line for line in probe_compare.second_card(cheap))

    heavy = stacks(probe("vllm", 52.0, reserved=18.0), probe("vllm", 30.0, completion=96))
    assert any("вторая карта окупается" in line for line in probe_compare.second_card(heavy))


def test_tight_memory_is_reported_even_when_generation_is_cheap():
    tight = stacks(probe("vllm", 52.0, reserved=23.0), probe("vllm", 48.0, completion=96))
    assert any("память впритык" in line for line in probe_compare.second_card(tight))


def test_headroom_winner_prints_its_own_free_memory():
    """Находка 018-8: печаталось 0.0 ГиБ — free_gib получал обёртку связки."""
    result = probe_compare.verdict(
        stacks(probe("vllm", 52.0, reserved=22.5), probe("unsloth", 56.0, reserved=15.0))
    )
    assert any("свободно 9.0 ГиБ против 1.5 ГиБ" in line for line in result), result


def test_two_probes_of_the_same_length_are_not_a_pair():
    """Находка 018-7: разность двух одинаковых прогонов — шум, не доля генерации."""
    entry = stacks(probe("vllm", 52.0), probe("vllm", 47.0))["vllm:g4"]
    assert probe_compare.generation_share(entry) is None


def test_short_probe_with_other_settings_is_not_a_pair():
    entry = stacks(probe("vllm", 52.0, gens=4), probe("vllm", 30.0, gens=4, completion=96))[
        "vllm:g4"
    ]
    assert probe_compare.generation_share(entry) is not None
    mixed = probe_compare.split_by_stack(
        [
            probe("vllm", 52.0, gens=4),
            {**probe("vllm", 30.0, gens=4, completion=96), "max_seq_length": 4096},
        ]
    )
    assert probe_compare.generation_share(mixed["vllm:g4"]) is None
