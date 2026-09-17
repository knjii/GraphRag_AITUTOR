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


PPRNode = tuple[str, str]


class PPRGraph:
    """Переиспользуемые переходы PPR; тип узла исключает коллизии ID.

    Рёбра неориентированные. IDF умножает вес mention-ребра на idf
    сущности, а entity-ребра — на геометрическое среднее двух idf.
    """

    def __init__(
        self, graph: OfflineGraph, *, entity_weight: float = 1.0, use_idf: bool = False
    ) -> None:
        if not math.isfinite(entity_weight) or entity_weight < 0:
            raise ValueError("Вес связей должен быть конечным и неотрицательным")
        adjacency: dict[PPRNode, dict[PPRNode, float]] = {}

        def add(left: PPRNode, right: PPRNode, weight: float) -> None:
            adjacency.setdefault(left, {})
            adjacency.setdefault(right, {})
            if not math.isfinite(weight) or weight < 0:
                raise ValueError("Вес ребра должен быть конечным и неотрицательным")
            if weight > 0:
                adjacency[left][right] = weight
                adjacency[right][left] = weight

        def boost(entity: str) -> float:
            return graph.idf.get(entity, 0.0) if use_idf else 1.0

        for passage, entities in graph.mentions.items():
            adjacency.setdefault(("passage", passage), {})
            for entity, count in entities.items():
                if count < 0:
                    raise ValueError("Число упоминаний не может быть отрицательным")
                add(("passage", passage), ("entity", entity), math.log1p(count) * boost(entity))
        for entity, neighbours in graph.neighbours.items():
            adjacency.setdefault(("entity", entity), {})
            for neighbour in neighbours:
                add(
                    ("entity", entity), ("entity", neighbour),
                    entity_weight * math.sqrt(boost(entity) * boost(neighbour)),
                )
        self.nodes = sorted(adjacency)
        self.index = {node: i for i, node in enumerate(self.nodes)}
        self.transitions: list[list[tuple[int, float]]] = []
        for node in self.nodes:
            total = sum(adjacency[node].values())
            self.transitions.append([
                (self.index[target], weight / total)
                for target, weight in sorted(adjacency[node].items())
            ])

    def probabilities(
        self, seeds: dict[PPRNode, float], *, alpha: float = 0.5,
        tolerance: float = 1e-10, max_iterations: int = 200,
    ) -> dict[PPRNode, float]:
        """Полная масса, включая затравки; висячие узлы возвращают её в seeds.

        alpha — вероятность возврата. Критерий остановки — L1 между
        итерациями. Несходимость явно прерывает замер вместо тихого усечения.
        Неизвестная затравка считается изолированным узлом (например, cache miss).
        """
        if not 0 < alpha <= 1 or not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Нужны 0 < alpha <= 1 и положительный конечный tolerance")
        if max_iterations < 1:
            raise ValueError("max_iterations должен быть положительным")
        if any(not math.isfinite(w) or w < 0 for w in seeds.values()):
            raise ValueError("Веса затравок должны быть конечными и неотрицательными")
        positive = {node: weight for node, weight in seeds.items() if weight > 0}
        if not positive:
            raise ValueError("Нужна хотя бы одна положительная затравка")
        nodes = self.nodes + sorted(node for node in positive if node not in self.index)
        transitions = self.transitions + [[] for _ in range(len(nodes) - len(self.nodes))]
        # Масштабирование не даёт сумме больших конечных весов переполниться.
        scale = max(positive.values())
        total = sum(w / scale for w in positive.values())
        reset = [positive.get(node, 0.0) / scale / total for node in nodes]
        current = reset[:]
        for _ in range(max_iterations):
            dangling = sum(current[i] for i, edges in enumerate(transitions) if not edges)
            next_values = [(alpha + (1 - alpha) * dangling) * w for w in reset]
            for i, edges in enumerate(transitions):
                mass = (1 - alpha) * current[i]
                if mass:
                    for target, probability in edges:
                        next_values[target] += mass * probability
            difference = sum(abs(a - b) for a, b in zip(current, next_values, strict=True))
            current = next_values
            if difference <= tolerance:
                return dict(zip(nodes, current, strict=True))
        raise RuntimeError(f"PPR не сошёлся за {max_iterations} итераций")

    def rank(self, seeds: dict[PPRNode, float], **kwargs: Any) -> list[tuple[str, float]]:
        """Вероятности фрагментов без положительных затравок, без перенормировки."""
        probabilities = self.probabilities(seeds, **kwargs)
        return sorted(
            ((node[1], probability) for node, probability in probabilities.items()
             if node[0] == "passage" and seeds.get(node, 0) <= 0 and probability > 0),
            key=lambda item: (-item[1], item[0]),
        )


def rank_ppr(
    graph: OfflineGraph, seeds: dict[PPRNode, float], *, alpha: float = 0.5,
    entity_weight: float = 1.0, use_idf: bool = False,
    tolerance: float = 1e-10, max_iterations: int = 200,
) -> list[tuple[str, float]]:
    """Ранжирует фрагменты степенным методом, без внешних зависимостей."""
    return PPRGraph(graph, entity_weight=entity_weight, use_idf=use_idf).rank(
        seeds, alpha=alpha, tolerance=tolerance, max_iterations=max_iterations,
    )


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
