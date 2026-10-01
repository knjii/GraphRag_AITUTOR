"""Награда Context-Picker: порядок ответов, который обязан сохраняться."""

from __future__ import annotations

import pytest

from rag_textbook.rewards.picker import STAGE_ONE, STAGE_TWO, picker_reward


def _line(*numbers: int) -> str:
    return "Нужно два факта. ### Final Selection: " + " ".join(f"[{n}]" for n in numbers)


@pytest.mark.parametrize("config", [STAGE_ONE, STAGE_TWO])
def test_order_exact_over_partial_over_nothing_over_invalid(config) -> None:
    gold, window = [2, 7], 20
    exact = picker_reward(_line(3, 8), gold, window, config=config).total
    half = picker_reward(_line(3), gold, window, config=config).total
    empty = picker_reward("### Final Selection:", gold, window, config=config).total
    everything = picker_reward(_line(*range(1, 21)), gold, window, config=config).total
    invalid = picker_reward("второй абзац", gold, window, config=config).total
    assert exact == 1.0
    assert exact > half > empty >= everything > invalid == -1.0


def test_empty_choice_gets_no_bonus() -> None:
    # Буквальная формула статьи давала пустому выбору +0.25 на стадии II.
    assert picker_reward("### Final Selection:", [0, 1], 20, config=STAGE_TWO).total == 0.0


def test_stage_two_penalizes_excess_stage_one_does_not() -> None:
    text = _line(1, 2, 3)
    assert picker_reward(text, [0, 1], 20, config=STAGE_ONE).total == 1.0
    assert picker_reward(text, [0, 1], 20, config=STAGE_TWO).total == pytest.approx(1 - 0.5 / 4)


def test_over_margin_is_zero_and_truncation_is_invalid() -> None:
    assert picker_reward(_line(1, 2, 3, 4, 5), [0, 1], 20, config=STAGE_TWO).total == 0.0
    score = picker_reward(_line(1, 2), [0, 1], 20, truncated=True)
    assert score.total == -1.0 and not score.valid
