"""Сетка офлайн-перебора.

Сетка — это список того, что мы собираемся проверить на слепке. Ошибка в ней
дорога вдвойне: настройка, меняющая состав кандидатов, не просто не сработает,
а даст правдоподобное число, полученное не о том. Поэтому каждая строка сетки
проверяется на воспроизводимость заранее, здесь, а не при разборе результатов.
"""

from __future__ import annotations

import pytest

from rag_textbook.cli.main import REPLAY_GRID, _apply_overrides, _describe
from rag_textbook.config import Settings
from rag_textbook.evaluation.trace import assert_replayable, snapshot_settings


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_every_grid_entry_is_replayable():
    """Ни одна строка сетки не должна упираться в границу слепка."""
    settings = _settings()
    snapshot = snapshot_settings(settings)

    for group, variants in REPLAY_GRID.items():
        for overrides in variants:
            candidate = _apply_overrides(settings, overrides)
            assert_replayable(snapshot, candidate)  # не должно бросить


def test_overrides_actually_change_settings():
    """Строка сетки, ничего не меняющая, — это молчаливо потерянная проверка."""
    settings = _settings()

    for group, variants in REPLAY_GRID.items():
        for overrides in variants:
            candidate = _apply_overrides(settings, overrides)
            differences = [
                (section, key)
                for section, values in overrides.items()
                for key in values
                if getattr(getattr(candidate, section), key)
                != getattr(getattr(settings, section), key)
            ]
            assert differences, f"{group}: {_describe(overrides)} ничего не меняет"


def test_overrides_do_not_leak_into_original():
    """Перебор не должен портить исходные настройки: варианты сравниваются
    с одной и той же точкой отсчёта."""
    settings = _settings()
    before = settings.retrieval.diversity_mode

    _apply_overrides(settings, {"retrieval": {"diversity_mode": "mmr"}})

    assert settings.retrieval.diversity_mode == before


def test_grid_covers_the_stated_hypotheses():
    """Сетка обязана покрывать гипотезы плана, иначе план и код разошлись."""
    assert set(REPLAY_GRID) == {
        "П1-разнообразие",
        "П2-реранкер",
        "П3-окно",
        "П4-слияние",
    }


def test_window_hypothesis_requires_wide_trace():
    """П3 расширяет окно кандидатов — это законно только при широком слепке.

    Проверка нужна, чтобы расширение окна не прошло молча по узкому слепку:
    баллов реранкера для добавленных кандидатов там нет, и результат был бы
    получен не о том.
    """
    from rag_textbook.evaluation.replay import replay
    from rag_textbook.evaluation.trace import TraceSet

    narrow = TraceSet(settings_snapshot=snapshot_settings(_settings()), rerank_window=30)
    widest = max(
        item["reranker"]["candidates"] for item in REPLAY_GRID["П3-окно"]
    )
    candidate = _apply_overrides(_settings(), {"reranker": {"candidates": widest}})

    with pytest.raises(ValueError, match="шире снятого"):
        replay(narrow, candidate, {})
