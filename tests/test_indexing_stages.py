"""Выбор стадий индексации.

Существует ради конкретного физического ограничения: видеокарта одна, сервер
инференса держит свою долю памяти всё время, пока запущен контейнер, и вместе
с MinerU они на 24 ГБ не помещаются. Разнести их по времени можно только так —
проход по стадиям выполняется одним процессом.
"""

from __future__ import annotations

import pytest

from rag_textbook.indexing.pipeline import ALL_STAGES, IndexingPipeline


def test_empty_selection_means_all_stages() -> None:
    assert IndexingPipeline._resolve_stages(None) == ALL_STAGES
    assert IndexingPipeline._resolve_stages([]) == ALL_STAGES
    assert IndexingPipeline._resolve_stages([""]) == ALL_STAGES


def test_selection_is_normalized() -> None:
    assert IndexingPipeline._resolve_stages([" Parse ", "GRAPH"]) == ("parse", "graph")


def test_duplicates_collapse() -> None:
    assert IndexingPipeline._resolve_stages(["embed", "embed"]) == ("embed",)


def test_unknown_stage_is_rejected() -> None:
    """Опечатка не должна молча превращаться в «ничего не делать»."""
    with pytest.raises(ValueError, match="rerank"):
        IndexingPipeline._resolve_stages(["parse", "rerank"])
