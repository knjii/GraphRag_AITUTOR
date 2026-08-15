"""Вклад графового канала против его присутствия.

Метрика «доля графа в контексте» считала графовым любой фрагмент, который
графовый канал нашёл, — включая те, что векторный канал находит сам.
На полном корпусе она показывала 16.3% вклада при **нулевой** разнице
по всем метрикам в парном A/B. Величина измеряла пересечение каналов,
а не вклад одного из них.
"""

from __future__ import annotations

from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.pipeline import RetrievalResult


def _scored(chunk_id: str, channels: list[str]) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            doc_id="d",
            doc_name="Учебник",
            source_path="/book.pdf",
            ordinal=0,
            text="текст",
        ),
        score=1.0,
        channels=channels,
    )


def test_overlap_counts_as_found_but_not_as_contribution() -> None:
    """Фрагмент, найденный обоими каналами, — не вклад графа."""
    item = _scored("a", ["dense", "graph_entity"])
    assert item.from_graph
    assert not item.only_from_graph


def test_graph_exclusive_chunk_counts_as_contribution() -> None:
    item = _scored("a", ["graph_entity"])
    assert item.from_graph
    assert item.only_from_graph


def test_vector_only_chunk_is_neither() -> None:
    item = _scored("a", ["dense", "sparse"])
    assert not item.from_graph
    assert not item.only_from_graph


def test_two_shares_diverge_on_full_overlap() -> None:
    """Ровно тот случай, что наблюдался на полном корпусе.

    Граф «нашёл» половину контекста, но не добавил ни одного фрагмента —
    старая метрика показала бы заметный вклад, новая показывает ноль.
    """
    result = RetrievalResult(
        question="вопрос",
        rewritten_question="вопрос",
        chunks=[
            _scored("a", ["dense", "graph_entity"]),
            _scored("b", ["dense", "graph_entity"]),
            _scored("c", ["dense"]),
            _scored("d", ["sparse"]),
        ],
    )
    assert result.graph_share == 0.5
    assert result.graph_only_share == 0.0


def test_contribution_is_counted_when_it_is_real() -> None:
    result = RetrievalResult(
        question="вопрос",
        rewritten_question="вопрос",
        chunks=[
            _scored("a", ["dense", "graph_entity"]),
            _scored("b", ["graph_entity"]),
            _scored("c", ["dense"]),
            _scored("d", ["dense"]),
        ],
    )
    assert result.graph_share == 0.5
    assert result.graph_only_share == 0.25


def test_empty_context_gives_zero() -> None:
    result = RetrievalResult(question="в", rewritten_question="в", chunks=[])
    assert result.graph_share == 0.0
    assert result.graph_only_share == 0.0
