from __future__ import annotations

import hashlib
import re
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List

from langchain_core.documents import Document

from settings import Settings
from utils import get_logger

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None

logger = get_logger("neo4j_client")

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_+\-/]{2,63}")
_STOPWORDS = {
    "and",
    "are",
    "for",
    "from",
    "that",
    "the",
    "this",
    "with",
    "как",
    "или",
    "для",
    "что",
    "это",
    "эти",
    "при",
    "под",
    "без",
    "над",
    "его",
    "её",
    "она",
    "они",
    "оно",
    "где",
    "когда",
    "если",
    "чтобы",
    "также",
    "очень",
    "между",
    "после",
    "перед",
    "который",
    "которая",
    "которые",
    "chapter",
    "figure",
    "table",
}


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_int(value: Any) -> int | None:
    try:
        return int(value)
    except Exception:
        return None


def _norm_token(token: str) -> str:
    normalized = token.strip(".,;:!?()[]{}\"'`").lower().replace("ё", "е")
    return normalized


def _extract_entities(text: str, max_per_passage: int) -> Dict[str, int]:
    counts: Counter[str] = Counter()
    for raw in _TOKEN_RE.findall(text or ""):
        token = _norm_token(raw)
        if len(token) < 3:
            continue
        if token in _STOPWORDS:
            continue
        if token.isdigit():
            continue
        counts[token] += 1
    if not counts:
        return {}
    top = counts.most_common(max(1, int(max_per_passage)))
    return {name: freq for name, freq in top}


def _passage_rows(docs: Iterable[Document]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    ts = _now_utc_iso()
    for idx, doc in enumerate(docs):
        meta = doc.metadata or {}
        source = str(meta.get("source", ""))
        source_id = hashlib.sha1(source.encode("utf-8", errors="ignore")).hexdigest()
        page = _safe_int(meta.get("page"))
        chunk_id = str(meta.get("chunk_id", "") or "").strip() or f"auto:{idx}"
        base = f"{source}|{page}|{chunk_id}"
        rows.append(
            {
                "id": hashlib.sha1(base.encode("utf-8", errors="ignore")).hexdigest(),
                "text": str(doc.page_content or ""),
                "source": source,
                "source_id": source_id,
                "page": page,
                "chunk_id": chunk_id,
                "order_in_source": idx,
                "created_at": ts,
                "updated_at": ts,
            }
        )
    return rows


def _relation_rows(
    passage_rows: List[Dict[str, Any]],
    *,
    max_entities_per_passage: int,
    max_cooccurs_per_passage: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    mention_rows: List[Dict[str, Any]] = []
    cooc_counter: Counter[tuple[str, str, str]] = Counter()

    for row in passage_rows:
        entities = _extract_entities(row.get("text", ""), max_entities_per_passage)
        if not entities:
            continue

        entity_ids: List[str] = []
        for entity_name, count in entities.items():
            entity_id = hashlib.sha1(entity_name.encode("utf-8", errors="ignore")).hexdigest()
            entity_ids.append(entity_id)
            mention_rows.append(
                {
                    "passage_id": row["id"],
                    "source_id": row["source_id"],
                    "chunk_id": row["chunk_id"],
                    "entity_id": entity_id,
                    "entity_name": entity_name,
                    "count": int(count),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                }
            )

        pair_count = 0
        for a, b in combinations(sorted(set(entity_ids)), 2):
            cooc_counter[(row["source_id"], a, b)] += 1
            pair_count += 1
            if pair_count >= max_cooccurs_per_passage:
                break

    cooccurs_rows: List[Dict[str, Any]] = []
    ts = _now_utc_iso()
    for (source_id, entity1_id, entity2_id), weight in cooc_counter.items():
        cooccurs_rows.append(
            {
                "source_id": source_id,
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "weight": int(weight),
                "updated_at": ts,
            }
        )
    return mention_rows, cooccurs_rows


class Neo4jGraphClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._driver = None

    @property
    def enabled(self) -> bool:
        return bool(self.settings.neo4j_enabled and self.settings.graph_write_enabled)

    def connect(self) -> bool:
        if not self.enabled:
            return False
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver is not installed. Run: pip install neo4j")
        if not self.settings.neo4j_password:
            raise RuntimeError("NEO4J_PASSWORD is empty in .env")
        if self._driver is None:
            self._driver = GraphDatabase.driver(
                self.settings.neo4j_uri,
                auth=(self.settings.neo4j_user, self.settings.neo4j_password),
            )
        self._driver.verify_connectivity()
        return True

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def ensure_schema(self) -> None:
        if self._driver is None:
            return
        queries = [
            "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
            "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX passage_source IF NOT EXISTS FOR (p:Passage) ON (p.source)",
        ]
        with self._driver.session(database=self.settings.neo4j_database) as session:
            for query in queries:
                session.run(query).consume()

    def upsert_passages(self, docs: List[Document]) -> int:
        if self._driver is None or not docs:
            return 0
        rows = _passage_rows(docs)
        mention_rows: List[Dict[str, Any]] = []
        cooccurs_rows: List[Dict[str, Any]] = []
        if self.settings.graph_relations_enabled:
            max_entities_per_passage = max(1, int(self.settings.graph_entity_max_per_passage))
            max_cooccurs_per_passage = max(1, int(self.settings.graph_cooccurs_max_per_passage))
            mention_rows, cooccurs_rows = _relation_rows(
                rows,
                max_entities_per_passage=max_entities_per_passage,
                max_cooccurs_per_passage=max_cooccurs_per_passage,
            )

        passage_query = """
        UNWIND $rows AS row
        MERGE (s:Source {id: row.source_id})
        SET s.path = row.source
        MERGE (p:Passage {id: row.id})
        ON CREATE SET p.created_at = row.created_at
        SET p.text = row.text,
            p.source = row.source,
            p.page = row.page,
            p.chunk_id = row.chunk_id,
            p.order_in_source = row.order_in_source,
            p.updated_at = row.updated_at
        MERGE (s)-[:HAS_PASSAGE]->(p)
        """
        mentions_query = """
        UNWIND $rows AS row
        MERGE (e:Entity {id: row.entity_id})
        ON CREATE SET e.created_at = row.created_at
        SET e.name = row.entity_name,
            e.updated_at = row.updated_at
        WITH row, e
        MATCH (p:Passage {id: row.passage_id})
        MERGE (p)-[r:MENTIONS]->(e)
        SET r.count = row.count,
            r.source_id = row.source_id,
            r.chunk_id = row.chunk_id,
            r.updated_at = row.updated_at
        """
        cooccurs_query = """
        UNWIND $rows AS row
        MATCH (e1:Entity {id: row.entity1_id})
        MATCH (e2:Entity {id: row.entity2_id})
        MERGE (e1)-[r:CO_OCCURS {source_id: row.source_id}]->(e2)
        SET r.weight = row.weight,
            r.updated_at = row.updated_at
        """
        with self._driver.session(database=self.settings.neo4j_database) as session:
            session.run(passage_query, rows=rows).consume()
            session.run(
                """
                UNWIND range(0, size($rows) - 2) AS i
                WITH $rows[i] AS a, $rows[i+1] AS b
                WHERE a.source_id = b.source_id
                MATCH (p1:Passage {id: a.id})
                MATCH (p2:Passage {id: b.id})
                MERGE (p1)-[:NEXT]->(p2)
                """,
                rows=rows,
            ).consume()
            if mention_rows:
                session.run(mentions_query, rows=mention_rows).consume()
            if cooccurs_rows:
                session.run(cooccurs_query, rows=cooccurs_rows).consume()
        logger.info(
            "Neo4j relations upsert: mentions=%s co_occurs=%s",
            len(mention_rows),
            len(cooccurs_rows),
        )
        return len(rows)


def write_passages_to_graph(docs: List[Document], settings: Settings) -> int:
    client = Neo4jGraphClient(settings)
    if not client.enabled:
        return 0
    try:
        client.connect()
        client.ensure_schema()
        written = client.upsert_passages(docs)
        if written:
            logger.info("Neo4j upsert passages: %s", written)
        return written
    finally:
        client.close()
