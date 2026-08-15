"""Резерв мест в пуле кандидатов за находками графового канала.

Основание — замер на наборе, где пары фрагментов связаны в графе и почти
не пересекаются по словам. Графовый канал находит там 10-12 процентных
пунктов эталонного материала, отсутствующего в векторной выдаче, но доля
таких фрагментов в итоговом контексте равна нулю: ранговое слияние ставит
их ниже плотной векторной выдачи, а до реранкера доезжают только первые
30 кандидатов.
"""

from __future__ import annotations

from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.pipeline import _reserve_graph_candidates


def _item(index: int, channels: list[str]) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=f"c{index}",
            doc_id="d",
            doc_name="Учебник",
            source_path="/book.pdf",
            ordinal=index,
            text=f"текст {index}",
        ),
        score=1.0 / (index + 1),
        channels=channels,
    )


def _merged(vector_count: int, graph_count: int) -> list[ScoredChunk]:
    """Векторные кандидаты сверху, находки одного графа — за границей отсечения."""
    items = [_item(index, ["dense"]) for index in range(vector_count)]
    items += [
        _item(vector_count + index, ["graph_entity"]) for index in range(graph_count)
    ]
    return items


def test_without_quota_graph_findings_are_cut_off() -> None:
    """Поведение по умолчанию не меняется — и в нём теряется вклад графа."""
    candidates = _reserve_graph_candidates(_merged(30, 5), limit=30, quota=0)

    assert len(candidates) == 30
    assert not any(item.only_from_graph for item in candidates)


def test_quota_brings_graph_findings_into_the_pool() -> None:
    candidates = _reserve_graph_candidates(_merged(30, 5), limit=30, quota=3)

    assert len(candidates) == 30, "размер пула не должен расти"
    assert sum(1 for item in candidates if item.only_from_graph) == 3


def test_vector_head_is_preserved() -> None:
    """Вытесняется хвост векторной выдачи, а не её начало."""
    candidates = _reserve_graph_candidates(_merged(30, 5), limit=30, quota=3)
    ids = [item.chunk.id for item in candidates]

    assert ids[:27] == [f"c{index}" for index in range(27)]


def test_quota_is_not_topped_up_when_already_met() -> None:
    """Если графовые находки и так в пуле, добирать нечего."""
    items = [_item(0, ["graph_entity"]), _item(1, ["graph_entity"])]
    items += [_item(index, ["dense"]) for index in range(2, 30)]
    items += [_item(index, ["graph_entity"]) for index in range(30, 35)]

    candidates = _reserve_graph_candidates(items, limit=30, quota=2)
    assert [item.chunk.id for item in candidates] == [item.chunk.id for item in items[:30]]


def test_overlap_does_not_count_towards_the_quota() -> None:
    """Фрагмент, найденный обоими каналами, — не находка графа.

    Именно смешение этих двух понятий и скрывало проблему: метрика
    присутствия графа показывала 18% при нулевом вкладе.
    """
    items = [_item(index, ["dense", "graph_entity"]) for index in range(30)]
    items += [_item(30, ["graph_entity"])]

    candidates = _reserve_graph_candidates(items, limit=30, quota=1)
    assert sum(1 for item in candidates if item.only_from_graph) == 1


def test_nothing_to_add_leaves_the_pool_untouched() -> None:
    candidates = _reserve_graph_candidates(_merged(30, 0), limit=30, quota=5)
    assert len(candidates) == 30


def test_short_pool_is_returned_as_is() -> None:
    items = _merged(5, 0)
    assert _reserve_graph_candidates(items, limit=30, quota=3) == items
