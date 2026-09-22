"""Отбор после реранкера, гипотезы К6–К8.

Числа посчитаны вручную: режим, который молча выродился в обычный порядок,
здесь обязан упасть, а не пройти.
"""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from rag_textbook.config import RetrievalSettings
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval import selection
from rag_textbook.stores.graph_file import GraphFile, MemoryGraphStore


class WordOverlapReranker:
    """Балл — число общих слов запроса и документа; считает вызовы."""

    def __init__(self) -> None:
        self.calls = 0

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        self.calls += 1
        words = set(query.split())
        scored = [(index, float(len(words & set(doc.split())))) for index, doc in enumerate(documents)]
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[:top_n]


def _item(chunk_id: str, text: str, score: float | None, ordinal: int = 0) -> ScoredChunk:
    chunk = Chunk(
        id=chunk_id, doc_id="d", doc_name="Книга", source_path="", ordinal=ordinal, text=text, pages=[1]
    )
    return ScoredChunk(chunk=chunk, rerank_score=score, channels=["dense"])


def _settings(**values) -> RetrievalSettings:
    return RetrievalSettings().model_copy(update=values)


def _store(roles: bool = False) -> MemoryGraphStore:
    graph = GraphFile(variant="test")
    for ordinal, pid in enumerate(["a", "b", "c", "d", "e"]):
        graph.add_passage(pid, doc_id="d", doc_name="Книга", ordinal=ordinal, text=f"текст {pid}", pages=[1])
    for eid in ["x", "y", "z"]:
        graph.add_entity(eid, canonical=eid, name=eid)
    # a—c делят редкий узел x; b—c делят y; d нигде не связан.
    graph.add_mention("a", "x", 1, role="uses" if roles else "")
    graph.add_mention("c", "x", 1, role="defines" if roles else "")
    graph.add_mention("b", "y", 1)
    graph.add_mention("c", "y", 1)
    graph.add_mention("e", "z", 1, role="defines" if roles else "")
    graph.add_mention("a", "z", 1, role="uses" if roles else "")
    return MemoryGraphStore(graph)


def test_conditional_picks_complement_of_selected() -> None:
    items = [
        _item("a", "альфа бета", 1.0),
        _item("b", "гамма", 0.6),
        _item("c", "бета дельта", 0.5),
    ]
    reranker = WordOverlapReranker()
    scorer = selection.PairScorer(reranker)
    result = selection.conditional(items, "вопрос", scorer, _settings(selection_lambda=0.5), top_k=3)
    # b: 0.5·0.6 + 0.5·0 = 0.30; c: 0.5·0.5 + 0.5·1 = 0.75 — c дополняет a.
    assert [item.chunk.id for item in result] == ["a", "c", "b"]


def test_conditional_with_lambda_one_keeps_reranker_order() -> None:
    items = [_item("a", "альфа бета", 1.0), _item("b", "гамма", 0.6), _item("c", "бета", 0.5)]
    scorer = selection.PairScorer(WordOverlapReranker())
    result = selection.conditional(items, "вопрос", scorer, _settings(selection_lambda=1.0), top_k=3)
    assert [item.chunk.id for item in result] == ["a", "b", "c"]


def test_pair_scorer_caches_and_persists(tmp_path) -> None:
    reranker = WordOverlapReranker()
    cache = tmp_path / "pairs.jsonl"
    scorer = selection.PairScorer(reranker, cache_path=cache)
    first = scorer.score("альфа бета", ["альфа", "гамма"])
    again = scorer.score("альфа бета", ["гамма", "альфа"])
    assert first == [1.0, 0.0] and again == [0.0, 1.0]
    assert reranker.calls == 1

    fresh = WordOverlapReranker()
    reloaded = selection.PairScorer(fresh, cache_path=cache)
    assert reloaded.score("альфа бета", ["альфа"]) == [1.0]
    assert fresh.calls == 0


def test_pairs_lift_partner_of_strong_fragment() -> None:
    items = [
        _item("a", "альфа", 1.0),
        _item("b", "бета", 0.9),
        _item("d", "дельта", 0.8),
        _item("c", "вопрос вопрос", 0.1),
    ]
    scorer = selection.PairScorer(WordOverlapReranker())
    result = selection.pairs(items, "вопрос", scorer, _store(), _settings(selection_lambda=0.5), top_k=2)
    # Пара a+c: 0.5·1.0 + 0.5·1 = 1.0 ≥ одиночного a (1.0) и идёт после него
    # по порядку; единица a берётся первой, затем пара a+c добавляет c.
    assert [item.chunk.id for item in result[:2]] == ["a", "c"]
    assert {item.chunk.id for item in result} == {"a", "b", "c", "d"}


def test_diffusion_raises_neighbour_of_strong_fragment() -> None:
    items = [
        _item("a", "", 1.0),
        _item("b", "", 0.55),
        _item("d", "", 0.5),
        _item("c", "", 0.0),
    ]
    store = _store()
    result = selection.diffusion(items, store, _settings(selection_alpha=1.0))
    # Нормировка баллов: a 1, b 0.55, d 0.5, c 0. Рёбра a—c и b—c имеют
    # одинаковый вес IDF, после нормировки 1. c получает 0 + 1·max(1, 0.55) = 1;
    # a получает 1 + 0 (сосед c слаб), b получает 0.55 + 0.
    assert [item.chunk.id for item in result] == ["a", "c", "b", "d"]


def test_closure_brings_definition_into_tail() -> None:
    final = [_item("a", "текст a", 1.0), _item("b", "текст b", 0.9), _item("d", "текст d", 0.8)]
    store = _store(roles=True)
    result = selection.closure(final, store, _settings(selection_max_replacements=1), top_k=3)
    ids = [item.chunk.id for item in result]
    # a использует x (определён в c) и z (в e); бюджет одна замена,
    # берётся первое по порядку изложения определение — c.
    assert ids == ["a", "b", "c"]
    assert result[-1].channels == ["graph_closure"]

    wider = selection.closure(final, store, _settings(selection_max_replacements=2), top_k=5)
    assert [item.chunk.id for item in wider] == ["a", "b", "d", "c", "e"]


def test_modes_refuse_to_run_without_their_tools() -> None:
    items = [_item("a", "", 1.0), _item("b", "", 0.5)]
    with pytest.raises(ValueError, match="реранкер"):
        selection.reorder(items, "q", _settings(selection_mode="conditional"), 2)
    with pytest.raises(ValueError, match="GRAPH_BACKEND=memory"):
        selection.reorder(items, "q", _settings(selection_mode="diffusion"), 2, store=object())
    assert selection.reorder(items, "q", _settings(), 2) == items


def test_replay_refuses_closure_and_runs_diffusion() -> None:
    from rag_textbook.config import Settings
    from rag_textbook.evaluation.replay import replay
    from rag_textbook.evaluation.trace import NotReplayable, QueryTrace, TracedCandidate, TraceSet

    store = _store()
    # Тексты разные: иначе дедупликация склеит фрагменты до отбора.
    texts = {"a": "матрица ранг", "b": "интеграл предел", "c": "вероятность событие", "d": "граф вершина"}
    chunks = {
        pid: Chunk(id=pid, doc_id="d", doc_name="Книга", source_path="", ordinal=i, text=texts[pid], pages=[1])
        for i, pid in enumerate(texts)
    }
    trace = QueryTrace(
        question_id="q1",
        question="вопрос",
        channels={"base": [TracedCandidate(pid, i, 1.0 - i / 10) for i, pid in enumerate(["a", "b", "d", "c"])]},
        rerank_scores={"a": 1.0, "b": 0.55, "d": 0.5, "c": 0.0},
    )
    traces = TraceSet()
    traces.traces.append(trace)

    base = Settings()
    closure = base.model_copy(deep=True)
    closure.retrieval = closure.retrieval.model_copy(update={"selection_mode": "closure"})
    with pytest.raises(NotReplayable):
        replay(traces, closure, chunks, link_store=store)

    diffused = base.model_copy(deep=True)
    diffused.retrieval = diffused.retrieval.model_copy(
        update={"selection_mode": "diffusion", "selection_alpha": 1.0, "top_k": 2}
    )
    diffused.reranker = diffused.reranker.model_copy(update={"enabled": True, "mode": "always"})
    outcomes = replay(traces, diffused, chunks, link_store=store)
    assert outcomes[0].retrieved == ["a", "c"]
