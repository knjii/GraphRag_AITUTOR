"""Заглушка графового хранилища.

Повторяет контракт :class:`GraphStore` в объёме, который использует
:class:`GraphRetriever`: поиск стартовых сущностей, расширение по типизированным
связям и взвешенный отбор пассажей. Благодаря ей графовый канал проверяется
целиком, без запущенного Neo4j.

Реализация намеренно повторяет **логику** боевого хранилища, а не подменяет её
константами: иначе тест доказывал бы работу заглушки, а не системы.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from rag_textbook.models import Chunk, Entity, Relation
from rag_textbook.utils.text import canonicalize_entity


class InMemoryGraphStore:
    """Граф знаний в памяти процесса."""

    def __init__(self) -> None:
        self.entities: dict[str, Entity] = {}
        self.relations: list[Relation] = []
        self.mentions: list[dict[str, Any]] = []
        self.cooccurrences: list[dict[str, Any]] = []
        self.passages: dict[str, Chunk] = {}
        self.schema_applied = False

    # ------------------------------------------------------------------ запись

    def verify(self) -> bool:
        return True

    def close(self) -> None:
        return None

    def ensure_schema(self) -> None:
        self.schema_applied = True

    def upsert_document(self, doc_id: str, doc_name: str, source_path: str) -> None:
        return None

    def upsert_passages(self, chunks: Sequence[Chunk]) -> int:
        for chunk in chunks:
            self.passages[chunk.id] = chunk
        return len(chunks)

    def upsert_entities(self, entities) -> int:
        count = 0
        for entity in entities:
            existing = self.entities.get(entity.id)
            if existing is None:
                self.entities[entity.id] = entity.model_copy(deep=True)
            else:
                existing.count += entity.count
            count += 1
        return count

    def upsert_mentions(self, mentions: Sequence[dict[str, Any]]) -> int:
        self.mentions.extend(mentions)
        return len(mentions)

    def upsert_relations(self, relations: Sequence[Relation]) -> int:
        self.relations.extend(relations)
        return len(relations)

    def upsert_cooccurrences(self, edges: Sequence[dict[str, Any]]) -> int:
        self.cooccurrences.extend(edges)
        return len(edges)

    def prune_high_degree_entities(self, max_degree: int) -> int:
        if max_degree <= 0:
            return 0
        degree: dict[str, int] = {}
        for relation in self.relations:
            degree[relation.source_id] = degree.get(relation.source_id, 0) + 1
            degree[relation.target_id] = degree.get(relation.target_id, 0) + 1
        hubs = [eid for eid, value in degree.items() if value > max_degree]
        for entity_id in hubs:
            self.entities.pop(entity_id, None)
        return len(hubs)

    # ------------------------------------------------------------------ чтение

    def find_seed_entities(self, terms: Sequence[str], limit: int) -> list[dict[str, Any]]:
        """Аналог полнотекстового поиска: совпадение по канонической форме."""
        wanted = {canonicalize_entity(term, lemmatize=False) for term in terms}
        wanted |= set(terms)
        found: list[dict[str, Any]] = []
        for entity in self.entities.values():
            score = 0.0
            if entity.canonical in wanted:
                score = 2.0
            elif any(term and term in entity.canonical for term in wanted):
                score = 1.0
            if score > 0:
                found.append(
                    {
                        "id": entity.id,
                        "canonical": entity.canonical,
                        "name": entity.name,
                        "count": entity.count,
                        "score": score,
                    }
                )
        found.sort(key=lambda row: (-row["score"], row["canonical"]))
        return found[: max(1, int(limit))]

    def entities_of_passages(self, chunk_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        """Сущности опорных фрагментов с обратной частотой по корпусу.

        Повторяет логику боевого запроса, а не подменяет её готовым ответом:
        иначе тест доказывал бы работоспособность заглушки.
        """
        wanted = {str(item) for item in chunk_ids}
        if not wanted:
            return []
        corpus = max(1, len(self.passages))

        local: dict[str, float] = {}
        document_frequency: dict[str, set[str]] = {}
        for mention in self.mentions:
            entity_id = str(mention["entity_id"])
            chunk_id = str(mention["chunk_id"])
            document_frequency.setdefault(entity_id, set()).add(chunk_id)
            if chunk_id in wanted:
                local[entity_id] = local.get(entity_id, 0.0) + math.log(
                    1 + int(mention.get("count", 1))
                )

        rows: list[dict[str, Any]] = []
        for entity_id, value in local.items():
            frequency = max(1, len(document_frequency.get(entity_id, ())))
            entity = self.entities.get(entity_id)
            rows.append(
                {
                    "id": entity_id,
                    "canonical": entity.canonical if entity else "",
                    "document_frequency": frequency,
                    "weight": value * math.log(corpus / frequency),
                }
            )
        rows.sort(key=lambda row: row["weight"], reverse=True)
        return rows[: max(1, int(limit))]

    def expand_entities(
        self, seed_ids: Sequence[str], hops: int, rel_types: Sequence[str], limit: int
    ) -> dict[str, float]:
        """Обход только по разрешённым типам связей, вес затухает с расстоянием."""
        allowed = {rel.upper() for rel in rel_types}
        weights: dict[str, float] = {entity_id: 1.0 for entity_id in seed_ids}
        frontier = set(seed_ids)

        for distance in range(1, max(1, int(hops)) + 1):
            next_frontier: set[str] = set()
            for relation in self.relations:
                if "RELATES" not in allowed:
                    continue
                for source, target in (
                    (relation.source_id, relation.target_id),
                    (relation.target_id, relation.source_id),
                ):
                    if source in frontier and target not in weights:
                        weights[target] = 1.0 / (1.0 + distance)
                        next_frontier.add(target)
            if "CO_OCCURS" in allowed:
                for edge in self.cooccurrences:
                    for source, target in (
                        (edge["source_id"], edge["target_id"]),
                        (edge["target_id"], edge["source_id"]),
                    ):
                        if source in frontier and target not in weights:
                            weights[target] = 1.0 / (1.0 + distance)
                            next_frontier.add(target)
            frontier = next_frontier
            if not frontier:
                break

        return dict(list(weights.items())[: max(1, int(limit))])

    def find_passages(self, entity_weights: dict[str, float], limit: int) -> list[dict[str, Any]]:
        """Взвешенный отбор с нормировкой на насыщенность пассажа терминами."""
        mentions_by_passage: dict[str, list[dict[str, Any]]] = {}
        for mention in self.mentions:
            mentions_by_passage.setdefault(str(mention["chunk_id"]), []).append(mention)

        scored: list[tuple[float, str, list[str]]] = []
        for chunk_id, mentions in mentions_by_passage.items():
            raw = 0.0
            matched: list[str] = []
            for mention in mentions:
                weight = entity_weights.get(str(mention["entity_id"]))
                if weight is None:
                    continue
                raw += weight * math.log(1 + int(mention.get("count", 1)))
                entity = self.entities.get(str(mention["entity_id"]))
                if entity is not None and entity.canonical not in matched:
                    matched.append(entity.canonical)
            if raw <= 0:
                continue
            score = raw / math.sqrt(max(1, len(mentions)))
            scored.append((score, chunk_id, matched[:6]))

        scored.sort(key=lambda item: item[0], reverse=True)
        rows: list[dict[str, Any]] = []
        for score, chunk_id, matched in scored[: max(1, int(limit))]:
            chunk = self.passages.get(chunk_id)
            if chunk is None:
                continue
            rows.append(
                {
                    "chunk_id": chunk.id,
                    "doc_id": chunk.doc_id,
                    "doc_name": chunk.doc_name,
                    "text": chunk.text,
                    "pages": chunk.pages,
                    "ordinal": chunk.ordinal,
                    "matched_entities": matched,
                    "score": score,
                }
            )
        return rows

    def stats(self) -> dict[str, int]:
        return {
            "passages": len(self.passages),
            "entities": len(self.entities),
            "relates": len(self.relations),
            "cooccurs": len(self.cooccurrences),
            "mentions": len(self.mentions),
        }
