"""Воспроизведение отбора по слепку.

Повторяет ту часть конвейера, что идёт после получения кандидатов: слияние,
дедупликация, резерв под графовые находки, реранкинг, разнообразие, отсечка.
Функции слияния берутся **те же самые**, что и в рабочем конвейере: своя копия
разошлась бы с продуктом незаметно, и офлайн-выводы снова оказались бы
про другую систему.

Стадии, которые здесь не воспроизводятся, потому что определяют состав
кандидатов, а не порядок: переписывание вопроса, маршрутизация, разложение,
запрос к хранилищам, обход графа. Попытка изменить их настройки прерывается
``assert_replayable``.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_textbook.config import Settings
from rag_textbook.evaluation.metrics import QueryOutcome
from rag_textbook.evaluation.trace import QueryTrace, TraceSet, assert_replayable
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.diversity import apply_diversity
from rag_textbook.retrieval.fusion import (
    deduplicate,
    enforce_minimum_graph_documents,
    reciprocal_rank_fusion,
)

logger = get_logger("evaluation.replay")


def _as_scored(
    trace: QueryTrace, channel: str, chunks: dict[str, Chunk]
) -> list[ScoredChunk]:
    """Собирает кандидатов канала обратно в объекты конвейера."""
    items: list[ScoredChunk] = []
    for candidate in sorted(trace.channels.get(channel, []), key=lambda item: item.rank):
        chunk = chunks.get(candidate.chunk_id)
        if chunk is None:
            # Фрагмент исчез из корпуса: пропускаем, но громко.
            logger.warning("Нет текста для фрагмента %s", candidate.chunk_id)
            continue
        items.append(
            ScoredChunk(
                chunk=chunk,
                score=candidate.score,
                channels=[channel],
                channel_scores={channel: candidate.score},
            )
        )
    return items


def _rerank_from_trace(
    trace: QueryTrace, items: list[ScoredChunk], settings: Settings, width: int
) -> list[ScoredChunk]:
    """Ранжирует по сохранённым баллам реранкера.

    Смесь с рангом слияния — проверяемая гипотеза: измерено, что реранкер
    помогает формульным вопросам и вредит связывающим, и вопрос в том, можно ли
    сохранить первое, убрав второе.
    """
    mode = settings.reranker.mode
    if not settings.reranker.enabled or mode == "off":
        return items[:width]

    if mode == "by_route" and trace.used_graph:
        # На связывающем маршруте порядок слияния сохраняется как есть.
        return items[:width]

    missing = [item for item in items if item.chunk.id not in trace.rerank_scores]
    if missing:
        logger.debug(
            "Для %s фрагментов нет балла реранкера: они уйдут в конец", len(missing)
        )

    fusion_rank = {item.chunk.id: index for index, item in enumerate(items)}
    alpha = settings.reranker.blend_alpha
    scale = max(len(items), 1)

    def key(item: ScoredChunk) -> float:
        score = trace.rerank_scores.get(item.chunk.id)
        if score is None:
            return float("-inf")
        if mode != "blend":
            return score
        # Ранг слияния переводится в убывающую величину того же знака, что балл.
        fusion_component = 1.0 - fusion_rank[item.chunk.id] / scale
        return alpha * score + (1.0 - alpha) * fusion_component

    ordered = sorted(items, key=key, reverse=True)
    for item in ordered:
        stored = trace.rerank_scores.get(item.chunk.id)
        if stored is not None:
            item.rerank_score = float(stored)
    return ordered[:width]


def replay_one(
    trace: QueryTrace, settings: Settings, chunks: dict[str, Chunk]
) -> list[ScoredChunk]:
    """Пересчитывает выдачу по одному вопросу."""
    top_k = settings.retrieval.top_k_for(trace.used_graph)

    base_items = _as_scored(trace, "base", chunks)
    graph_items = _as_scored(trace, "graph", chunks) if trace.used_graph else []

    graph_weight = settings.graph.weight if graph_items else 0.0
    merged = reciprocal_rank_fusion(
        {"base": base_items, "graph": graph_items},
        weights={"base": 1.0 - graph_weight, "graph": graph_weight},
        rrf_k=settings.retrieval.rrf_k,
    )
    if settings.retrieval.dedup_enabled:
        merged = deduplicate(merged, settings.retrieval.dedup_similarity)

    from rag_textbook.retrieval.pipeline import _reserve_graph_candidates

    candidates = _reserve_graph_candidates(
        merged,
        limit=settings.reranker.candidates,
        quota=settings.retrieval.graph_candidate_quota,
    )

    width = max(settings.reranker.top_n, top_k) + settings.retrieval.min_graph_docs
    if settings.retrieval.diversity_mode != "off":
        # См. комментарий в конвейере: разнообразию нужен запас сверх top_k,
        # иначе оно не может ничего переставить.
        width = max(width, len(candidates))
    reranked = _rerank_from_trace(trace, candidates, settings, width)

    diversified = apply_diversity(reranked, settings, top_k=top_k)

    return enforce_minimum_graph_documents(
        diversified, minimum=settings.retrieval.min_graph_docs, top_k=top_k
    )


def replay(
    traces: TraceSet,
    settings: Settings,
    chunks: dict[str, Chunk],
    gold: dict[str, Sequence[str]] | None = None,
) -> list[QueryOutcome]:
    """Пересчитывает выдачу по всему слепку.

    ``gold`` — эталонные фрагменты по идентификатору вопроса. Без них метрики
    не посчитать, но выдачу посмотреть можно.
    """
    assert_replayable(traces.settings_snapshot, settings)
    if settings.reranker.candidates > traces.rerank_window > 0:
        raise ValueError(
            f"Окно кандидатов {settings.reranker.candidates} шире снятого "
            f"({traces.rerank_window}): баллов реранкера для остальных нет. "
            "Снимите слепок с более широким окном."
        )

    outcomes: list[QueryOutcome] = []
    for trace in traces.traces:
        final = replay_one(trace, settings, chunks)
        retrieved = [item.chunk.id for item in final]
        outcomes.append(
            QueryOutcome(
                question_id=trace.question_id,
                question_type=trace.question_type,
                retrieved=retrieved,
                relevant=list((gold or {}).get(trace.question_id, [])),
                used_graph=trace.used_graph,
                graph_share=(
                    sum(1 for item in final if item.from_graph) / len(final) if final else 0.0
                ),
                graph_only_share=(
                    sum(1 for item in final if item.only_from_graph) / len(final)
                    if final
                    else 0.0
                ),
            )
        )
    return outcomes


def fidelity_report(traces: TraceSet, replayed: Sequence[QueryOutcome]) -> dict[str, float]:
    """Насколько воспроизведение совпало с тем, что выдал сервер.

    Первая проверка после возвращения с сервера, до любых экспериментов:
    воспроизведение с рабочими настройками обязано повторить серверную выдачу.
    Если не повторяет — слепок неполон, и все выводы по нему недействительны.
    """
    by_id = {outcome.question_id: outcome.retrieved for outcome in replayed}
    exact = 0
    overlap_total = 0.0
    compared = 0
    for trace in traces.traces:
        replayed_ids = by_id.get(trace.question_id)
        if replayed_ids is None or not trace.final:
            continue
        compared += 1
        if replayed_ids == trace.final:
            exact += 1
        shared = len(set(replayed_ids) & set(trace.final))
        overlap_total += shared / max(len(trace.final), 1)
    return {
        "вопросов сверено": float(compared),
        "совпало точно": float(exact),
        "доля точных совпадений": exact / compared if compared else 0.0,
        "среднее пересечение": overlap_total / compared if compared else 0.0,
    }
