"""Реестр A/B-экспериментов должен ссылаться на существующие настройки.

Опечатка в пути вида ``retrieval.min_graph_doc`` не упала бы заметно: сравнение
прошло бы, отчёт напечатался бы, и обе стороны оказались бы одной и той же
конфигурацией. Прирост при этом честно вышел бы нулевым — и был бы записан
как «проверено, эффекта нет». Такой результат хуже ошибки.
"""

from __future__ import annotations

import pytest

from rag_textbook.cli.main import AB_EXPERIMENTS
from rag_textbook.config import Settings


@pytest.mark.parametrize("name", sorted(AB_EXPERIMENTS))
def test_experiment_paths_exist_and_differ(name: str) -> None:
    baseline, candidate, labels = AB_EXPERIMENTS[name]
    settings = Settings()

    for overrides in (baseline, candidate):
        for path in overrides:
            section_name, _, field_name = path.partition(".")
            section = getattr(settings, section_name, None)
            assert section is not None, f"{name}: нет раздела «{section_name}» в настройках"
            assert field_name, f"{name}: путь «{path}» без имени поля"
            assert hasattr(section, field_name), (
                f"{name}: в разделе «{section_name}» нет параметра «{field_name}»"
            )

    assert baseline != candidate, f"{name}: обе стороны сравнения одинаковы"
    assert len(labels) == 2 and labels[0] != labels[1], f"{name}: подписи не различаются"


@pytest.mark.parametrize("name", sorted(AB_EXPERIMENTS))
def test_override_values_pass_validation(name: str) -> None:
    """Значение должно приниматься настройкой, а не только именем совпадать."""
    baseline, candidate, _labels = AB_EXPERIMENTS[name]
    for overrides in (baseline, candidate):
        settings = Settings()
        for path, value in overrides.items():
            section_name, _, field_name = path.partition(".")
            setattr(getattr(settings, section_name), field_name, value)
            assert getattr(getattr(settings, section_name), field_name) == value
