"""Сборка графа из результатов извлечения.

Здесь живёт главное лекарство от прежней болезни: co-occurrence-рёбра больше не
строятся как «все пары терминов чанка». Пары считаются, но попадают в граф только
если их совместная встречаемость статистически значима (PMI выше порога) и они
достаточно часты. На корпусе, где раньше получалось 280 526 рёбер, остаётся
на порядки меньше — и это те рёбра, которые действительно что-то значат.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

from rag_textbook.config import GraphSettings
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, Entity, ExtractionResult, Relation
from rag_textbook.stores.graph_store import GraphStore, pointwise_mutual_information

logger = get_logger("graph.builder")


@dataclass
class GraphBuildResult:
    passages: int = 0
    entities: int = 0
    mentions: int = 0
    relations: int = 0
    # Из них полученных сопоставлением нескольких фрагментов: только такие
    # рёбра несут сведения, которых нет ни в одном отдельном фрагменте.
    cross_chunk_relations: int = 0
    cooccurrences: int = 0
    cooccurrence_candidates: int = 0
    pruned_hubs: int = 0
    extraction_status: Counter = field(default_factory=Counter)

    def as_dict(self) -> dict[str, Any]:
        return {
            "passages": self.passages,
            "entities": self.entities,
            "mentions": self.mentions,
            "relations": self.relations,
            "cross_chunk_relations": self.cross_chunk_relations,
            "cooccurrences": self.cooccurrences,
            "cooccurrence_candidates": self.cooccurrence_candidates,
            "cooccurrence_kept_ratio": (
                round(self.cooccurrences / self.cooccurrence_candidates, 4)
                if self.cooccurrence_candidates
                else 0.0
            ),
            "pruned_hubs": self.pruned_hubs,
            "extraction_status": dict(self.extraction_status),
        }


class GraphBuilder:
    def __init__(
        self,
        settings: GraphSettings,
        extractor: EntityExtractor,
        store: GraphStore | None = None,
        max_workers: int = 4,
    ) -> None:
        self.settings = settings
        self.extractor = extractor
        self.store = store
        self.max_workers = max(1, int(max_workers))

    # -------------------------------------------------------------- извлечение

    def _extract_all(self, chunks: Sequence[Chunk], model_name: str) -> list[ExtractionResult]:
        """Параллельное извлечение.

        Прежний цикл был строго последовательным: 1060 пассажей по ~39 с давали
        около 11.5 часов. Кэш плюс параллелизм убирают основную часть этого времени.
        """
        if self.max_workers == 1:
            return [self.extractor.extract(chunk, model_name) for chunk in chunks]

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            return list(pool.map(lambda chunk: self.extractor.extract(chunk, model_name), chunks))

    # ------------------------------------------------------------ co-occurrence

    def _build_cooccurrences(
        self,
        entity_ids_per_chunk: list[list[str]],
        entity_doc: dict[str, str],
    ) -> tuple[list[dict[str, Any]], int]:
        """Строит co-occurrence-рёбра с отсевом по PMI.

        Именно отсутствие этого шага делало прежний граф бесполезным:
        частотный термин связывался со всем подряд просто потому, что он частотный.
        """
        if not self.settings.cooccurrence_enabled:
            return [], 0

        total_chunks = len(entity_ids_per_chunk)
        if total_chunks < 2:
            return [], 0

        entity_freq: Counter[str] = Counter()
        pair_freq: Counter[tuple[str, str]] = Counter()
        for entity_ids in entity_ids_per_chunk:
            unique = sorted(set(entity_ids))
            entity_freq.update(unique)
            for left, right in combinations(unique, 2):
                pair_freq[(left, right)] += 1

        candidates = len(pair_freq)
        edges: list[dict[str, Any]] = []
        for (left, right), count in pair_freq.items():
            if count < self.settings.cooccurrence_min_count:
                continue
            pmi = pointwise_mutual_information(
                count, entity_freq[left], entity_freq[right], total_chunks
            )
            if pmi < self.settings.cooccurrence_min_pmi:
                continue
            edges.append(
                {
                    "source_id": left,
                    "target_id": right,
                    "doc_id": entity_doc.get(left, ""),
                    "count": int(count),
                    "pmi": round(float(pmi), 4),
                }
            )

        logger.info(
            "Co-occurrence: кандидатов %s, оставлено %s (%.1f%%), порог PMI=%s, min_count=%s",
            candidates,
            len(edges),
            100.0 * len(edges) / candidates if candidates else 0.0,
            self.settings.cooccurrence_min_pmi,
            self.settings.cooccurrence_min_count,
        )
        return edges, candidates

    # ------------------------------------------------ связи между фрагментами

    def _build_cross_chunk_relations(
        self,
        chunks: Sequence[Chunk],
        entity_ids_per_chunk: list[list[str]],
        merged_entities: dict[str, Entity],
    ) -> list[Relation]:
        """Извлекает связи, видимые только при сопоставлении разных фрагментов.

        Отбор понятий не случайный: берутся те, что упомянуты в достаточном
        числе **разных** фрагментов. Понятие из одного фрагмента сопоставлять
        не с чем, а самые частотные дают больше всего шансов найти связь между
        далёкими разделами. Стоимость — один вызов модели на понятие, поэтому
        их число ограничено настройкой.
        """
        if not self.settings.cross_chunk_relations_enabled:
            return []

        chunks_by_entity: dict[str, list[Chunk]] = defaultdict(list)
        for chunk, entity_ids in zip(chunks, entity_ids_per_chunk, strict=True):
            for entity_id in set(entity_ids):
                chunks_by_entity[entity_id].append(chunk)

        eligible = [
            (entity_id, items)
            for entity_id, items in chunks_by_entity.items()
            if len(items) >= self.settings.cross_chunk_min_chunks
        ]
        eligible.sort(key=lambda item: len(item[1]), reverse=True)
        eligible = eligible[: self.settings.cross_chunk_max_entities]
        if not eligible:
            logger.info("Связи между фрагментами: подходящих понятий нет")
            return []

        known = [entity.canonical for entity in merged_entities.values()]

        def one(item: tuple[str, list[Chunk]]) -> list[Relation]:
            entity_id, items = item
            entity = merged_entities.get(entity_id)
            if entity is None:
                return []
            # Выдержки берутся из максимально далёких друг от друга мест:
            # соседние фрагменты перекрываются, и сопоставлять их бессмысленно.
            ordered = sorted(items, key=lambda chunk: chunk.ordinal)
            step = max(1, len(ordered) // self.settings.cross_chunk_max_excerpts)
            excerpts = [chunk.text for chunk in ordered[::step]][
                : self.settings.cross_chunk_max_excerpts
            ]
            return self.extractor.extract_cross_chunk(entity.canonical, excerpts, known)

        logger.info(
            "Связи между фрагментами: понятий %s (порог %s фрагментов)",
            len(eligible),
            self.settings.cross_chunk_min_chunks,
        )
        if self.max_workers == 1:
            batches = [one(item) for item in eligible]
        else:
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                batches = list(pool.map(one, eligible))

        relations = [relation for batch in batches for relation in batch]
        logger.info("Связи между фрагментами: получено %s рёбер", len(relations))
        return relations

    # ---------------------------------------------------------------- публично

    def build(
        self,
        chunks: Sequence[Chunk],
        *,
        doc_id: str,
        doc_name: str,
        source_path: str,
        model_name: str = "",
        write: bool = True,
    ) -> GraphBuildResult:
        result = GraphBuildResult()
        if not chunks:
            return result

        logger.info("Построение графа: документ=%s, чанков=%s", doc_name, len(chunks))
        extractions = self._extract_all(chunks, model_name)

        merged_entities: dict[str, Entity] = {}
        mentions: list[dict[str, Any]] = []
        relations: list[Relation] = []
        entity_ids_per_chunk: list[list[str]] = []
        entity_doc: dict[str, str] = {}

        for chunk, extraction in zip(chunks, extractions, strict=True):
            result.extraction_status[extraction.status] += 1
            chunk_entity_ids: list[str] = []

            for entity in extraction.entities:
                existing = merged_entities.get(entity.id)
                if existing is None:
                    merged_entities[entity.id] = entity.model_copy(deep=True)
                else:
                    existing.count += entity.count
                    for alias in entity.aliases:
                        if alias not in existing.aliases:
                            existing.aliases.append(alias)
                entity_doc.setdefault(entity.id, chunk.doc_id)
                chunk_entity_ids.append(entity.id)
                mentions.append(
                    {
                        "chunk_id": chunk.id,
                        "entity_id": entity.id,
                        "doc_id": chunk.doc_id,
                        "count": int(entity.count),
                    }
                )

            entity_ids_per_chunk.append(chunk_entity_ids)
            relations.extend(extraction.relations)

        cross_chunk = self._build_cross_chunk_relations(
            chunks, entity_ids_per_chunk, merged_entities
        )
        relations.extend(cross_chunk)

        cooccurrences, candidates = self._build_cooccurrences(entity_ids_per_chunk, entity_doc)

        result.passages = len(chunks)
        result.entities = len(merged_entities)
        result.mentions = len(mentions)
        result.relations = len(relations)
        result.cross_chunk_relations = len(cross_chunk)
        result.cooccurrences = len(cooccurrences)
        result.cooccurrence_candidates = candidates

        # Статусы извлечения выводятся рядом с числом связей намеренно: без них
        # «RELATES=0» не отличить от «в тексте нет связей», хотя это могут быть
        # пустые ответы модели или невалидный JSON на каждом чанке.
        logger.info(
            "Граф собран: сущностей=%s, упоминаний=%s, RELATES=%s (из них между "
            "фрагментами %s), CO_OCCURS=%s, извлечение=%s",
            result.entities,
            result.mentions,
            result.relations,
            result.cross_chunk_relations,
            result.cooccurrences,
            dict(result.extraction_status),
        )

        if write and self.store is not None:
            self.store.upsert_document(doc_id, doc_name, source_path)
            self.store.upsert_passages(chunks)
            self.store.upsert_entities(merged_entities.values())
            self.store.upsert_mentions(mentions)
            self.store.upsert_relations(relations)
            self.store.upsert_cooccurrences(cooccurrences)
            if self.settings.max_entity_degree > 0:
                result.pruned_hubs = self.store.prune_high_degree_entities(
                    self.settings.max_entity_degree
                )

        return result


def group_chunks_by_document(chunks: Sequence[Chunk]) -> dict[str, list[Chunk]]:
    grouped: dict[str, list[Chunk]] = defaultdict(list)
    for chunk in chunks:
        grouped[chunk.doc_id].append(chunk)
    for items in grouped.values():
        items.sort(key=lambda chunk: chunk.ordinal)
    return dict(grouped)
