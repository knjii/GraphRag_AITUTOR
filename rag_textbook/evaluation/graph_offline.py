"""Оценка графового канала без сервера.

Зачем это нужно. Каждый замер графа до сих пор стоил аренды машины с картой:
поднять Neo4j, поднять модель, пересобрать граф, прогнать набор. При личном
бюджете это означало, что перебрать десяток вариантов ранжирования нельзя —
проверялся один-два, остальные оставались догадками.

Между тем всё необходимое лежит на диске. Кэш извлечения хранит сущности
и связи по каждому фрагменту, разбор хранит тексты, эталонный набор хранит
пары фрагментов. Этого хватает, чтобы восстановить граф в памяти и померить
на нём то единственное, чего система не умеет.

Что именно меряется. Разбор промахов дал однозначную картину: из 34 неудач
на многошаговых вопросах все 34 — «нашёл один фрагмент из двух», ни одной
«не нашёл ни одного». Вход в тему находится всегда, не находится переход.
Поэтому мерой служит место второго фрагмента пары, если графу дан первый.

Чего проверка не заменяет. Здесь нет ни стартовых сущностей по тексту вопроса
(их даёт полнотекстовый индекс Neo4j), ни реранкера, ни слияния каналов.
Это измеритель одной подсистемы, а не продукта: он говорит, какой вариант
обхода лучше, но не какой будет итоговый recall. Итог по-прежнему меряется
прогоном на сервере — просто теперь туда едут проверенные варианты, а не все.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import content_hash

logger = get_logger("evaluation.graph_offline")

NOT_FOUND = 10**6


@dataclass
class OfflineGraph:
    """Граф, восстановленный из кэша извлечения."""

    # фрагмент → сущность → число упоминаний
    mentions: dict[str, dict[str, int]] = field(default_factory=dict)
    neighbours: dict[str, set[str]] = field(default_factory=dict)
    chunks_of_entity: dict[str, set[str]] = field(default_factory=dict)
    idf: dict[str, float] = field(default_factory=dict)
    names: dict[str, str] = field(default_factory=dict)
    edges: int = 0
    passages: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    pruned_entities: int = 0

    @property
    def entities(self) -> int:
        return len(self.chunks_of_entity)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passages": self.passages,
            "entities": self.entities,
            "edges": self.edges,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "pruned_entities": self.pruned_entities,
        }


def _extraction_key(
    settings: Settings,
    text: str,
    text_hash: str,
    *,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    """Тот же ключ, что считает экстрактор при индексации.

    Совпадение ключей и есть проверка того, что восстановленный граф —
    это граф с сервера, а не что-то похожее. Расхождение видно сразу:
    оно превращается в промахи кэша, а не в тихо неверные числа.

    Модель и режим размышления задаются отдельно не для удобства: один файл
    кэша переживает смену движка, и записи от Ollama лежат рядом с записями
    от SGLang. Без явного указания локальная настройка вытащит не тот прогон.
    """
    graph = settings.graph
    return content_hash(
        text_hash or content_hash(text),
        settings.llm.model if model is None else model,
        settings.llm.reasoning_effort_for("extraction")
        if reasoning_effort is None
        else reasoning_effort,
        graph.extraction_prompt_version,
        str(graph.max_entities_per_chunk),
        str(graph.max_relations_per_chunk),
    )


def _load_chunks(parsed_dir: Path, doc_id: str | None = None) -> list[dict[str, Any]]:
    pattern = f"{doc_id}_chunks.json" if doc_id else "*_chunks.json"
    chunks: list[dict[str, Any]] = []
    for path in sorted(parsed_dir.glob(pattern)):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def _load_extractions(cache_path: Path) -> dict[str, dict[str, Any]]:
    if not cache_path.exists():
        return {}
    conn = sqlite3.connect(str(cache_path))
    try:
        rows = conn.execute(
            "SELECT key, value FROM cache_entries WHERE namespace = 'extraction'"
        ).fetchall()
    finally:
        conn.close()
    return {key: json.loads(value) for key, value in rows}


def reconstruct(
    settings: Settings,
    *,
    max_entity_degree: int | None = None,
    doc_id: str | None = None,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> OfflineGraph:
    """Собирает граф в памяти по тем же правилам, что и запись в Neo4j.

    Отсечение хабов повторяется намеренно: в Neo4j оно выполняется после
    записи, и без него картина связности завышена вдвое.
    """
    chunks = _load_chunks(settings.paths.parsed_dir, doc_id)
    if not chunks:
        raise FileNotFoundError(
            f"В {settings.paths.parsed_dir} нет разобранных фрагментов — "
            "сначала выполните стадию parse"
        )
    extractions = _load_extractions(settings.paths.cache_dir / "extraction.sqlite3")

    graph = OfflineGraph(passages=len(chunks))
    mentions: dict[str, dict[str, int]] = defaultdict(dict)
    edges: set[tuple[str, str, str]] = set()

    for chunk in chunks:
        key = _extraction_key(
            settings,
            chunk.get("text", ""),
            chunk.get("text_hash", ""),
            model=model,
            reasoning_effort=reasoning_effort,
        )
        entry = extractions.get(key)
        if entry is None:
            graph.cache_misses += 1
            continue
        graph.cache_hits += 1
        for entity in entry.get("entities", []):
            mentions[chunk["id"]][entity["id"]] = int(entity.get("count", 1) or 1)
            graph.names[entity["id"]] = entity.get("name", "")
        for relation in entry.get("relations", []):
            edges.add(
                (relation["source_id"], relation["target_id"], relation.get("label", ""))
            )

    if graph.cache_hits == 0:
        raise RuntimeError(
            "Ни один фрагмент не найден в кэше извлечения. Обычно это значит, "
            "что настройки разошлись с теми, при которых кэш собран: модель, "
            "reasoning_effort, версия промпта или лимиты сущностей и связей."
        )
    if graph.cache_misses > graph.cache_hits:
        # Частичное попадание опаснее полного промаха: числа получаются,
        # но по куску корпуса, и это надо видеть, а не выяснять потом.
        logger.warning(
            "В кэше нашлось только %s фрагментов из %s. Скорее всего указана "
            "не та модель или не тот режим размышления: кэш хранит записи всех "
            "прогонов, включая прежний движок.",
            graph.cache_hits,
            graph.passages,
        )

    degree_limit = (
        settings.graph.max_entity_degree if max_entity_degree is None else max_entity_degree
    )
    if degree_limit:
        degree: dict[str, int] = defaultdict(int)
        for source, target, _label in edges:
            degree[source] += 1
            degree[target] += 1
        pruned = {entity for entity, value in degree.items() if value > degree_limit}
        graph.pruned_entities = len(pruned)
        edges = {e for e in edges if e[0] not in pruned and e[1] not in pruned}
        for entities in mentions.values():
            for entity_id in [e for e in entities if e in pruned]:
                del entities[entity_id]

    neighbours: dict[str, set[str]] = defaultdict(set)
    for source, target, _label in edges:
        neighbours[source].add(target)
        neighbours[target].add(source)

    chunks_of: dict[str, set[str]] = defaultdict(set)
    for chunk_id, entities in mentions.items():
        for entity_id in entities:
            chunks_of[entity_id].add(chunk_id)

    total = max(1, len(chunks))
    graph.mentions = dict(mentions)
    graph.neighbours = dict(neighbours)
    graph.chunks_of_entity = dict(chunks_of)
    graph.idf = {
        entity_id: math.log(total / len(found))
        for entity_id, found in chunks_of.items()
        if found
    }
    graph.edges = len(edges)
    logger.info("Граф восстановлен из кэша: %s", graph.as_dict())
    return graph


def rank_from_passage(
    graph: OfflineGraph,
    anchor: str,
    *,
    hop_decay: float,
    use_idf: bool,
) -> list[str]:
    """Ранжирует фрагменты, отталкиваясь от сущностей опорного фрагмента.

    Повторяет то, что делают ``expand_entities`` и ``find_passages`` вместе,
    включая нормировку на насыщенность фрагмента терминами.
    """
    own = graph.mentions.get(anchor, {})
    weights: dict[str, float] = dict.fromkeys(own, 1.0)
    if hop_decay > 0:
        for entity_id in list(own):
            for neighbour in graph.neighbours.get(entity_id, ()):
                if neighbour not in own:
                    weights[neighbour] = max(weights.get(neighbour, 0.0), hop_decay)

    scores: dict[str, float] = defaultdict(float)
    for entity_id, weight in weights.items():
        boost = graph.idf.get(entity_id, 0.0) if use_idf else 1.0
        if boost <= 0:
            continue
        for chunk_id in graph.chunks_of_entity.get(entity_id, ()):
            count = graph.mentions[chunk_id].get(entity_id, 1)
            scores[chunk_id] += weight * boost * math.log(1 + count)

    for chunk_id in scores:
        size = max(1, len(graph.mentions.get(chunk_id, {})))
        scores[chunk_id] /= math.sqrt(size)
    scores.pop(anchor, None)
    return sorted(scores, key=lambda chunk_id: -scores[chunk_id])


def linked_pairs(settings: Settings, graph: OfflineGraph) -> list[tuple[str, str]]:
    """Пары фрагментов многошаговых вопросов эталонного набора."""
    goldset = load_goldset(settings.paths.goldset_dir / "goldset.json")
    pairs: list[tuple[str, str]] = []
    for question in goldset:
        ids = list(question.gold_chunk_ids)
        if len(ids) != 2:
            continue
        if ids[0] in graph.mentions and ids[1] in graph.mentions:
            pairs.append((ids[0], ids[1]))
    return pairs


def second_hop_ranks(
    graph: OfflineGraph,
    pairs: Iterable[tuple[str, str]],
    *,
    hop_decay: float,
    use_idf: bool,
) -> list[int]:
    """Место второго фрагмента пары при известном первом, в обе стороны."""
    ranks: list[int] = []
    for left, right in pairs:
        for anchor, target in ((left, right), (right, left)):
            order = rank_from_passage(graph, anchor, hop_decay=hop_decay, use_idf=use_idf)
            ranks.append(order.index(target) + 1 if target in order else NOT_FOUND)
    return ranks


def summarize(ranks: Sequence[int], cutoffs: Sequence[int] = (8, 16, 30)) -> dict[str, float]:
    if not ranks:
        return {"measurements": 0}
    count = len(ranks)
    result: dict[str, float] = {
        "measurements": count,
        "mrr": round(sum(1.0 / rank for rank in ranks) / count, 4),
    }
    for cutoff in cutoffs:
        result[f"hit@{cutoff}"] = round(
            sum(1 for rank in ranks if rank <= cutoff) / count, 4
        )
    found = sorted(rank for rank in ranks if rank < NOT_FOUND)
    result["median_rank"] = float(found[len(found) // 2]) if found else float(NOT_FOUND)
    result["unreachable"] = sum(1 for rank in ranks if rank >= NOT_FOUND)
    return result
