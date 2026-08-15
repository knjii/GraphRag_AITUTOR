"""Слияние результатов каналов и дедупликация.

Дедупликация вынесена отдельно осознанно: при перекрытии чанков соседние
фрагменты имеют общий хвост, а прежний ключ сравнивал тексты точно, поэтому
почти-дубликаты занимали места в контексте, вытесняя разное содержимое.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_textbook.models import ScoredChunk
from rag_textbook.utils.text import content_terms, jaccard


def reciprocal_rank_fusion(
    channels: dict[str, Sequence[ScoredChunk]],
    weights: dict[str, float] | None = None,
    rrf_k: int = 60,
) -> list[ScoredChunk]:
    """Слияние взаимных рангов с весами каналов."""
    weights = weights or {}
    fused: dict[str, float] = {}
    best: dict[str, ScoredChunk] = {}
    channel_map: dict[str, list[str]] = {}
    channel_scores: dict[str, dict[str, float]] = {}

    for channel, items in channels.items():
        weight = float(weights.get(channel, 1.0))
        if weight <= 0:
            continue
        for rank, item in enumerate(items):
            chunk_id = item.chunk.id
            fused[chunk_id] = fused.get(chunk_id, 0.0) + weight / (rrf_k + rank + 1)
            if chunk_id not in best:
                best[chunk_id] = item
            channel_map.setdefault(chunk_id, [])
            for existing in item.channels or [channel]:
                if existing not in channel_map[chunk_id]:
                    channel_map[chunk_id].append(existing)
            channel_scores.setdefault(chunk_id, {})[channel] = item.score

    merged: list[ScoredChunk] = []
    for chunk_id, score in sorted(fused.items(), key=lambda pair: pair[1], reverse=True):
        item = best[chunk_id]
        merged.append(
            ScoredChunk(
                chunk=item.chunk,
                score=score,
                channels=channel_map.get(chunk_id, item.channels),
                channel_scores=channel_scores.get(chunk_id, {}),
                matched_entities=item.matched_entities,
            )
        )
    return merged


def deduplicate(
    items: Sequence[ScoredChunk], similarity_threshold: float = 0.92
) -> list[ScoredChunk]:
    """Убирает почти-дубликаты, сохраняя первый (наиболее релевантный).

    Сравнение идёт по множеству лемм: точное сравнение строк не ловит
    перекрывающиеся чанки, а полноценное вычисление расстояния избыточно.
    """
    kept: list[ScoredChunk] = []
    kept_terms: list[set[str]] = []
    seen_ids: set[str] = set()

    for item in items:
        if item.chunk.id in seen_ids:
            continue
        terms = set(content_terms(item.chunk.text, lemmatize=False, limit=200))
        duplicate = False
        for existing in kept_terms:
            if jaccard(terms, existing) >= similarity_threshold:
                duplicate = True
                break
        if duplicate:
            continue
        seen_ids.add(item.chunk.id)
        kept.append(item)
        kept_terms.append(terms)
    return kept


def enforce_minimum_graph_documents(
    items: Sequence[ScoredChunk], minimum: int, top_k: int
) -> list[ScoredChunk]:
    """Гарантирует присутствие графовых кандидатов в финальном контексте.

    Нужна как инструмент диагностики: без неё графовый канал может стабильно
    вытесняться базовым, и тогда невозможно понять, плох ли граф сам по себе
    или он просто не доходит до контекста.
    """
    top_k = max(1, int(top_k))
    minimum = max(0, min(int(minimum), top_k))
    selected = list(items[:top_k])
    if minimum == 0:
        return selected

    graph_count = sum(1 for item in selected if item.from_graph)
    if graph_count >= minimum:
        return selected

    reserve = [
        item
        for item in items[top_k:]
        if item.from_graph and all(item.chunk.id != kept.chunk.id for kept in selected)
    ]
    for candidate in reserve:
        if graph_count >= minimum:
            break
        # Выбрасываем худший неграфовый элемент.
        for index in range(len(selected) - 1, -1, -1):
            if not selected[index].from_graph:
                selected.pop(index)
                break
        else:
            break
        selected.append(candidate)
        graph_count += 1

    return selected[:top_k]
