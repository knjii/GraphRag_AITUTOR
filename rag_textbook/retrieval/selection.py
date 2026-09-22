"""Отбор после реранкера: гипотезы К6–К8 (docs/HYPOTHESES.md, серия К).

Реранкер оценивает каждый фрагмент против вопроса поодиночке. Второй
фрагмент связывающей пары на вопрос сам не отвечает и получает низкий
балл. Слепок показал, что пул накрывает эталон на 0.87–0.95, а выдача
на 0.75: потеря — после поиска. Здесь четыре способа это исправить.

``conditional`` (К6а)
    Жадный отбор. Следующий фрагмент оценивается реранкером против
    «вопрос + начало уже выбранного»: фрагмент, дополняющий выбранное,
    получает высокий условный балл, хотя сам на вопрос не отвечает.
``pairs`` (К6б)
    Пары фрагментов, связанные в графе, оцениваются как одна единица.
``diffusion`` (К7)
    Нулевая модель для К6, без новых вызовов реранкера: балл = свой +
    α · max(балл соседа × вес ребра). Если она забирает столько же,
    сколько К6, условная оценка не нужна.
``closure`` (К8)
    После отсечки к выбранному добавляются места определений понятий,
    которые оно использует. Добавленное вытесняет последние места.
    Приносит фрагменты не из пула, поэтому проверяется только онлайн.

Рёбра между фрагментами берутся у хранилища графа методом
``passage_links``; он есть у хранилища в памяти (``GRAPH_BACKEND=memory``).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

from rag_textbook.clients.reranker import RerankerClient
from rag_textbook.config import RetrievalSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, ScoredChunk

logger = get_logger("retrieval.selection")

# Сколько пар К6б оценивать на вопрос. Пул в 30 фрагментов даёт до 435
# пар по общим узлам; берутся самые сильные по весу ребра. Порог в коде,
# а не в настройках: он ограничивает цену, а не меняет гипотезу.
MAX_PAIRS = 64

MODES_NEEDING_RERANKER = ("conditional", "pairs")
MODES_NEEDING_GRAPH = ("pairs", "diffusion", "closure")
# Моды, которые меняют состав выдачи, а не только порядок внутри пула.
MODES_LEAVING_POOL = ("closure",)


class LinkStore(Protocol):
    def passage_links(
        self, chunk_ids: Sequence[str], *, mode: str = "shared", max_entity_passages: int = 64
    ) -> dict[tuple[str, str], float]: ...

    def definitions_for(self, chunk_ids: Sequence[str], limit_per_entity: int = 2) -> list[str]: ...

    def passage_row(self, chunk_id: str) -> dict[str, Any] | None: ...


def supports_links(store: object | None) -> bool:
    return store is not None and all(
        hasattr(store, name) for name in ("passage_links", "definitions_for", "passage_row")
    )


class PairScorer:
    """Реранкер с кэшем по хэшу пары «запрос — документ».

    Условие К6а меняется на каждом шаге, но одни и те же пары повторяются
    между вопросами и между перебором параметров. На CPU каждый вызов
    стоит секунды, поэтому кэш можно сохранить в файл и переиспользовать
    между прогонами (``cache_path``).
    """

    def __init__(self, reranker: RerankerClient, cache_path: Path | None = None) -> None:
        self.reranker = reranker
        self.cache_path = cache_path
        self.cache: dict[str, float] = {}
        self.calls = 0
        self.hits = 0
        if cache_path is not None and cache_path.exists():
            for line in cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = float(row["score"])

    @staticmethod
    def key(query: str, document: str) -> str:
        return hashlib.sha1(f"{query}\x00{document}".encode()).hexdigest()

    def score(self, query: str, documents: Sequence[str]) -> list[float]:
        keys = [self.key(query, document) for document in documents]
        missing = [index for index, key in enumerate(keys) if key not in self.cache]
        self.hits += len(documents) - len(missing)
        if missing:
            self.calls += 1
            ranked = self.reranker.rerank(query, [documents[i] for i in missing], len(missing))
            scores = {missing[position]: float(value) for position, value in ranked}
            fresh: list[dict[str, Any]] = []
            for index in missing:
                # Реранкер вправе вернуть не всех; пропущенный — худший.
                value = scores.get(index, float("-inf"))
                self.cache[keys[index]] = value
                fresh.append({"key": keys[index], "score": value})
            if self.cache_path is not None:
                self.cache_path.parent.mkdir(parents=True, exist_ok=True)
                with self.cache_path.open("a", encoding="utf-8") as handle:
                    for row in fresh:
                        handle.write(json.dumps(row) + "\n")
        return [self.cache[key] for key in keys]


def _excerpt(item: ScoredChunk, chars: int) -> str:
    return item.chunk.text[:chars]


def _own_scores(items: Sequence[ScoredChunk]) -> list[float]:
    """Собственный балл: балл реранкера, а без него — место в списке."""
    if items and all(item.rerank_score is not None for item in items):
        return [float(item.rerank_score) for item in items]  # type: ignore[arg-type]
    size = max(len(items), 1)
    return [1.0 - index / size for index in range(len(items))]


def _normalized(values: Sequence[float]) -> list[float]:
    """Приводит баллы к [0, 1]: у реранкеров бывают и логиты со знаком."""
    if not values:
        return []
    low, high = min(values), max(values)
    if high - low <= 0:
        return [1.0 for _ in values]
    return [(value - low) / (high - low) for value in values]


def conditional(
    items: Sequence[ScoredChunk],
    question: str,
    scorer: PairScorer,
    settings: RetrievalSettings,
    top_k: int,
) -> list[ScoredChunk]:
    """К6а: жадный отбор по баллу при уже выбранном."""
    if len(items) <= 1:
        return list(items)
    own = _own_scores(items)
    weight = settings.selection_lambda
    chosen = [0]
    remaining = list(range(1, len(items)))
    while remaining and len(chosen) < top_k:
        context = "\n\n".join(
            _excerpt(items[index], settings.selection_excerpt_chars)
            for index in chosen[-settings.selection_condition_items :]
        )
        condition = f"{question}\n\n{context}"
        scores = scorer.score(condition, [items[index].chunk.text for index in remaining])
        values = [
            weight * own[index] + (1.0 - weight) * score
            for index, score in zip(remaining, scores, strict=True)
        ]
        # При равенстве побеждает стоявший выше у реранкера.
        best = max(range(len(remaining)), key=lambda position: (values[position], -remaining[position]))
        chosen.append(remaining.pop(best))
    return [items[index] for index in chosen] + [items[index] for index in remaining]


def _unordered_links(links: dict[tuple[str, str], float]) -> dict[tuple[str, str], float]:
    merged: dict[tuple[str, str], float] = {}
    for (left, right), weight in links.items():
        if left == right or weight <= 0:
            continue
        key = (left, right) if left < right else (right, left)
        merged[key] = max(merged.get(key, 0.0), weight)
    return merged


def pairs(
    items: Sequence[ScoredChunk],
    question: str,
    scorer: PairScorer,
    store: LinkStore,
    settings: RetrievalSettings,
    top_k: int,
) -> list[ScoredChunk]:
    """К6б: пары вдоль рёбер графа оцениваются как единица."""
    if len(items) <= 1:
        return list(items)
    position = {item.chunk.id: index for index, item in enumerate(items)}
    links = _unordered_links(
        store.passage_links(list(position), mode=settings.selection_links)
    )
    edges = sorted(links.items(), key=lambda pair: (-pair[1], pair[0]))[:MAX_PAIRS]
    if not edges:
        return list(items)
    chars = settings.selection_excerpt_chars
    documents = [
        f"{_excerpt(items[position[left]], chars)}\n\n{_excerpt(items[position[right]], chars)}"
        for (left, right), _ in edges
    ]
    pair_scores = scorer.score(question, documents)
    own = _own_scores(items)
    weight = settings.selection_lambda

    # Единица — одиночный фрагмент со своим баллом или пара со смесью
    # лучшего своего балла и балла пары. Первой берётся сильнейшая единица.
    units: list[tuple[float, int, tuple[int, ...]]] = [
        (own[index], index, (index,)) for index in range(len(items))
    ]
    for ((left, right), _), score in zip(edges, pair_scores, strict=True):
        members = tuple(sorted((position[left], position[right])))
        value = weight * max(own[members[0]], own[members[1]]) + (1.0 - weight) * score
        units.append((value, members[0], members))
    units.sort(key=lambda unit: (-unit[0], unit[1], len(unit[2])))

    chosen: list[int] = []
    taken: set[int] = set()
    for _, _, members in units:
        if len(chosen) >= top_k:
            break
        for index in sorted(members, key=lambda member: (-own[member], member)):
            if index not in taken and len(chosen) < top_k:
                taken.add(index)
                chosen.append(index)
    rest = [index for index in range(len(items)) if index not in taken]
    return [items[index] for index in chosen] + [items[index] for index in rest]


def diffusion(
    items: Sequence[ScoredChunk], store: LinkStore, settings: RetrievalSettings
) -> list[ScoredChunk]:
    """К7: балл = свой + α · max(балл соседа × вес ребра) внутри пула.

    Баллы и веса рёбер нормированы на [0, 1] по пулу вопроса: вес ребра
    по общим узлам — сумма IDF и шкалы не имеет, а у реранкера бывают
    логиты со знаком.
    """
    if len(items) <= 1:
        return list(items)
    own = _normalized(_own_scores(items))
    position = {item.chunk.id: index for index, item in enumerate(items)}
    links = _unordered_links(store.passage_links(list(position), mode=settings.selection_links))
    if not links:
        return list(items)
    top = max(links.values())
    boost = [0.0] * len(items)
    for (left, right), weight in links.items():
        a, b = position[left], position[right]
        share = weight / top
        boost[a] = max(boost[a], own[b] * share)
        boost[b] = max(boost[b], own[a] * share)
    values = [own[index] + settings.selection_alpha * boost[index] for index in range(len(items))]
    order = sorted(range(len(items)), key=lambda index: (-values[index], index))
    return [items[index] for index in order]


def closure(
    final: Sequence[ScoredChunk],
    store: LinkStore,
    settings: RetrievalSettings,
    top_k: int,
) -> list[ScoredChunk]:
    """К8: места определений вытесняют последние места выдачи."""
    budget = settings.selection_max_replacements
    if budget <= 0 or not final:
        return list(final)
    present = {item.chunk.id for item in final}
    additions: list[ScoredChunk] = []
    for chunk_id in store.definitions_for([item.chunk.id for item in final]):
        if len(additions) >= budget:
            break
        if chunk_id in present:
            continue
        row = store.passage_row(chunk_id)
        if not row or not str(row.get("text") or "").strip():
            continue
        chunk = Chunk(
            id=chunk_id,
            doc_id=str(row.get("doc_id") or ""),
            doc_name=str(row.get("doc_name") or ""),
            source_path="",
            ordinal=int(row.get("ordinal") or 0),
            text=str(row["text"]),
            pages=[int(page) for page in (row.get("pages") or [])],
        )
        additions.append(ScoredChunk(chunk=chunk, channels=["graph_closure"]))
        present.add(chunk_id)
    if not additions:
        return list(final)
    keep = max(0, min(len(final), top_k - len(additions)))
    return list(final[:keep]) + additions


def check_ready(settings: RetrievalSettings, scorer: PairScorer | None, store: object | None) -> None:
    """Режим отбора без нужного ему инструмента молча выродился бы в обычный порядок."""
    mode = settings.selection_mode
    if mode in MODES_NEEDING_RERANKER and scorer is None:
        raise ValueError(f"Отбор {mode} требует реранкер для оценки пар")
    if mode in MODES_NEEDING_GRAPH and not supports_links(store):
        raise ValueError(
            f"Отбор {mode} требует рёбра между фрагментами: нужен граф в памяти "
            "(GRAPH_BACKEND=memory и GRAPH_FILE)"
        )


def reorder(
    items: Sequence[ScoredChunk],
    question: str,
    settings: RetrievalSettings,
    top_k: int,
    *,
    scorer: PairScorer | None = None,
    store: object | None = None,
) -> list[ScoredChunk]:
    """Переупорядочивает пул после реранкера. Состав не меняется."""
    mode = settings.selection_mode
    if mode in ("off", "closure"):
        return list(items)
    check_ready(settings, scorer, store)
    if mode == "conditional":
        return conditional(items, question, scorer, settings, top_k)  # type: ignore[arg-type]
    if mode == "pairs":
        return pairs(items, question, scorer, store, settings, top_k)  # type: ignore[arg-type]
    return diffusion(items, store, settings)  # type: ignore[arg-type]


def complete(
    final: Sequence[ScoredChunk],
    settings: RetrievalSettings,
    top_k: int,
    *,
    store: object | None = None,
) -> list[ScoredChunk]:
    """Достраивает выдачу после отсечки (К8). Прочим режимам нечего делать."""
    if settings.selection_mode != "closure":
        return list(final)
    check_ready(settings, None, store)
    return closure(final, store, settings, top_k)  # type: ignore[arg-type]
