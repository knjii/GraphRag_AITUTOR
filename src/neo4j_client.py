from __future__ import annotations

import hashlib
import json
import re
import time
from collections import Counter
from itertools import combinations
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Tuple

from langchain_core.documents import Document

from settings import Settings
from utils import chat_with_ollama, get_logger

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


def _extract_json_candidate(text: str) -> str | None:
    if not text:
        return None
    stripped = text.strip()
    if stripped.startswith("```") and stripped.endswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3:
            fenced = "\n".join(lines[1:-1]).strip()
            if fenced:
                return fenced
    decoder = json.JSONDecoder()
    for idx, ch in enumerate(text):
        if ch not in "{[":
            continue
        try:
            obj, _ = decoder.raw_decode(text[idx:])
        except Exception:
            continue
        if isinstance(obj, (dict, list)):
            return json.dumps(obj, ensure_ascii=False)
    return None


def _parse_llm_json(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    candidates: List[str] = [text]
    extracted = _extract_json_candidate(text)
    if extracted and extracted != text:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return {}


def _normalize_entity_name(name: str, min_token_len: int) -> str:
    if not name:
        return ""
    parts: List[str] = []
    for raw in _TOKEN_RE.findall(str(name)):
        token = _norm_token(raw)
        if _is_valid_entity_token(token, min_token_len):
            parts.append(token)
    return " ".join(parts[:6]).strip()


def _build_graph_extract_prompt(
    text: str,
    *,
    max_entities: int,
    max_relations: int,
) -> str:
    return (
        "Ты извлекаешь сущности и связи из технического текста.\n"
        "Верни строго JSON без комментариев и markdown.\n"
        'Формат: {"entities":[{"name":"..."}],'
        '"relations":[{"source":"...","relation":"...","target":"..."}]}\n'
        f"Ограничения: entities <= {max_entities}, relations <= {max_relations}.\n"
        "Если данных нет, верни пустые списки.\n\n"
        f"TEXT:\n{text}"
    )


def _is_ollama_503(exc: Exception) -> bool:
    return "status code: 503" in str(exc).lower()


def _extract_entities_and_relations_llm(
    text: str,
    settings: Settings,
    *,
    max_entities_per_passage: int,
    max_relations_per_passage: int,
    min_token_len: int,
) -> Tuple[Dict[str, int], List[Tuple[str, str, str]]]:
    if not text.strip():
        return {}, []

    max_chars = max(500, int(settings.graph_llm_input_max_chars))
    prompt = _build_graph_extract_prompt(
        text[:max_chars],
        max_entities=max_entities_per_passage,
        max_relations=max_relations_per_passage,
    )
    model_name = settings.graph_llm_model or settings.ollama_model
    options = {
        "temperature": float(settings.graph_llm_temperature),
        "top_k": int(settings.ollama_top_k),
        "num_ctx": int(settings.ollama_num_ctx),
        "num_gpu": int(settings.ollama_num_gpu),
        "num_batch": int(settings.ollama_num_batch),
        "num_predict": int(settings.graph_llm_num_predict),
    }
    raw = None
    for attempt in range(2):
        try:
            raw = chat_with_ollama(
                message=prompt,
                model=model_name,
                options=options,
                settings=settings,
            )
            break
        except Exception as exc:
            if attempt == 0 and _is_ollama_503(exc):
                # Best-effort cleanup of stuck Ollama runners before one retry.
                try:
                    chat_with_ollama(turn_off=True, model=model_name, settings=settings)
                    if settings.ollama_embed_model and settings.ollama_embed_model != model_name:
                        chat_with_ollama(
                            turn_off=True,
                            model=settings.ollama_embed_model,
                            settings=settings,
                        )
                except Exception:
                    pass
                time.sleep(0.8)
                continue
            raise

    parsed = _parse_llm_json(str(raw or ""))

    entities_raw = parsed.get("entities", [])
    relations_raw = parsed.get("relations", [])

    entities_count: Counter[str] = Counter()
    if isinstance(entities_raw, list):
        for item in entities_raw:
            if isinstance(item, str):
                candidate = item
            elif isinstance(item, dict):
                candidate = item.get("name") or item.get("entity") or ""
            else:
                candidate = ""
            normalized = _normalize_entity_name(str(candidate), min_token_len=min_token_len)
            if normalized:
                entities_count[normalized] += 1
            if len(entities_count) >= max_entities_per_passage:
                break

    relations: List[Tuple[str, str, str]] = []
    if isinstance(relations_raw, list):
        for rel in relations_raw:
            if not isinstance(rel, dict):
                continue
            src = _normalize_entity_name(str(rel.get("source", "")), min_token_len=min_token_len)
            dst = _normalize_entity_name(str(rel.get("target", "")), min_token_len=min_token_len)
            rtype = str(rel.get("relation", "")).strip().lower()[:64]
            if not src or not dst or src == dst:
                continue
            if not rtype:
                rtype = "related_to"
            relations.append((src, rtype, dst))
            if len(relations) >= max_relations_per_passage:
                break

    return dict(entities_count), relations


def _is_valid_entity_token(token: str, min_len: int) -> bool:
    if len(token) < max(1, int(min_len)):
        return False
    if token in _STOPWORDS:
        return False
    if token.isdigit():
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return True


def _extract_entities(
    text: str,
    max_per_passage: int,
    *,
    min_token_len: int,
    use_bigrams: bool,
    max_bigrams_per_passage: int,
) -> Dict[str, int]:
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        token = _norm_token(raw)
        if _is_valid_entity_token(token, min_token_len):
            tokens.append(token)

    counts: Counter[str] = Counter(tokens)

    if use_bigrams and max_bigrams_per_passage > 0 and len(tokens) > 1:
        bigram_counts: Counter[str] = Counter()
        for left, right in zip(tokens, tokens[1:]):
            if left == right:
                continue
            bigram_counts[f"{left} {right}"] += 1

        for phrase, freq in bigram_counts.most_common(max(1, int(max_bigrams_per_passage))):
            counts[phrase] += int(freq)

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
    settings: Settings,
    max_entities_per_passage: int,
    max_cooccurs_per_passage: int,
    max_cooccurs_provenance_per_edge: int,
    entity_min_token_len: int,
    entity_use_bigrams: bool,
    entity_max_bigrams_per_passage: int,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    mention_rows: List[Dict[str, Any]] = []
    cooc_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    relates_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
    use_llm_extractor = str(settings.graph_entity_extractor).strip().lower() == "llm"
    llm_failures = 0
    llm_success = 0
    llm_entities_total = 0
    llm_relations_total = 0
    total_passages = len(passage_rows)
    progress_every = max(1, int(getattr(settings, "graph_llm_progress_every", 25)))
    started_at = time.monotonic()

    for idx, row in enumerate(passage_rows, start=1):
        entities: Dict[str, int] = {}
        llm_relations: List[Tuple[str, str, str]] = []

        if use_llm_extractor:
            try:
                entities, llm_relations = _extract_entities_and_relations_llm(
                    row.get("text", ""),
                    settings,
                    max_entities_per_passage=max_entities_per_passage,
                    max_relations_per_passage=max_cooccurs_per_passage,
                    min_token_len=entity_min_token_len,
                )
                llm_success += 1
                llm_entities_total += int(len(entities))
                llm_relations_total += int(len(llm_relations))
            except Exception as exc:
                llm_failures += 1
                logger.warning("Graph LLM extraction failed for chunk %s: %s", row.get("chunk_id"), exc)
                if not settings.graph_llm_fallback_to_rule:
                    entities = {}
                else:
                    entities = _extract_entities(
                        row.get("text", ""),
                        max_entities_per_passage,
                        min_token_len=entity_min_token_len,
                        use_bigrams=entity_use_bigrams,
                        max_bigrams_per_passage=entity_max_bigrams_per_passage,
                    )
            if (
                idx == 1
                or idx % progress_every == 0
                or idx == total_passages
            ):
                elapsed = time.monotonic() - started_at
                logger.info(
                    "Graph LLM extraction progress: %s/%s passages (ok=%s fail=%s entities=%s relations=%s elapsed=%.1fs)",
                    idx,
                    total_passages,
                    llm_success,
                    llm_failures,
                    llm_entities_total,
                    llm_relations_total,
                    elapsed,
                )
        else:
            entities = _extract_entities(
                row.get("text", ""),
                max_entities_per_passage,
                min_token_len=entity_min_token_len,
                use_bigrams=entity_use_bigrams,
                max_bigrams_per_passage=entity_max_bigrams_per_passage,
            )
        if not entities:
            continue

        entity_name_to_id: Dict[str, str] = {}
        entity_ids: List[str] = []
        for entity_name, count in entities.items():
            entity_id = hashlib.sha1(entity_name.encode("utf-8", errors="ignore")).hexdigest()
            entity_name_to_id[entity_name] = entity_id
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
        if llm_relations:
            relates_seen_in_passage: set[tuple[str, str, str]] = set()
            for src_name, relation_label, dst_name in llm_relations:
                src_id = entity_name_to_id.get(src_name)
                dst_id = entity_name_to_id.get(dst_name)
                if not src_id or not dst_id or src_id == dst_id:
                    continue
                normalized_label = str(relation_label or "related_to").strip().lower()[:64] or "related_to"
                relation_key = (src_id, dst_id, normalized_label)
                if relation_key not in relates_seen_in_passage:
                    relates_seen_in_passage.add(relation_key)
                    relates_counter[
                        (
                            row["source_id"],
                            row["id"],
                            row["chunk_id"],
                            src_id,
                            dst_id,
                            normalized_label,
                        )
                    ] += 1

                a, b = sorted((src_id, dst_id))
                key = (row["source_id"], a, b)
                bucket = cooc_map.setdefault(
                    key,
                    {"weight": 0, "passage_ids": set(), "chunk_ids": set(), "relation_labels": set()},
                )
                bucket["weight"] += 1
                if normalized_label:
                    bucket["relation_labels"].add(normalized_label)
                if len(bucket["passage_ids"]) < max_cooccurs_provenance_per_edge:
                    bucket["passage_ids"].add(row["id"])
                if len(bucket["chunk_ids"]) < max_cooccurs_provenance_per_edge:
                    bucket["chunk_ids"].add(row["chunk_id"])
                pair_count += 1
                if pair_count >= max_cooccurs_per_passage:
                    break
        else:
            for a, b in combinations(sorted(set(entity_ids)), 2):
                key = (row["source_id"], a, b)
                bucket = cooc_map.setdefault(
                    key,
                    {"weight": 0, "passage_ids": set(), "chunk_ids": set(), "relation_labels": set()},
                )
                bucket["weight"] += 1
                if len(bucket["passage_ids"]) < max_cooccurs_provenance_per_edge:
                    bucket["passage_ids"].add(row["id"])
                if len(bucket["chunk_ids"]) < max_cooccurs_provenance_per_edge:
                    bucket["chunk_ids"].add(row["chunk_id"])
                pair_count += 1
                if pair_count >= max_cooccurs_per_passage:
                    break

    cooccurs_rows: List[Dict[str, Any]] = []
    ts = _now_utc_iso()
    for (source_id, entity1_id, entity2_id), payload in cooc_map.items():
        cooccurs_rows.append(
            {
                "source_id": source_id,
                "entity1_id": entity1_id,
                "entity2_id": entity2_id,
                "weight": int(payload["weight"]),
                "passage_ids": sorted(payload["passage_ids"]),
                "chunk_ids": sorted(payload["chunk_ids"]),
                "relation_labels": sorted(payload["relation_labels"]),
                "updated_at": ts,
            }
        )

    relates_rows: List[Dict[str, Any]] = []
    for (source_id, passage_id, chunk_id, source_entity_id, target_entity_id, relation), rel_count in relates_counter.items():
        relates_rows.append(
            {
                "source_id": source_id,
                "passage_id": passage_id,
                "chunk_id": chunk_id,
                "source_entity_id": source_entity_id,
                "target_entity_id": target_entity_id,
                "relation": relation,
                "count": int(rel_count),
                "updated_at": ts,
            }
        )

    if use_llm_extractor:
        logger.info(
            "Graph LLM extraction stats: passages=%s success=%s failed=%s",
            len(passage_rows),
            llm_success,
            llm_failures,
        )
    relation_stats = {
        "extractor_mode": "llm" if use_llm_extractor else "rule",
        "llm_passages_success": int(llm_success),
        "llm_passages_failed": int(llm_failures),
        "relates": int(len(relates_rows)),
    }
    return mention_rows, cooccurs_rows, relates_rows, relation_stats


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

    def upsert_passages_stats(self, docs: List[Document]) -> Dict[str, Any]:
        if self._driver is None or not docs:
            return {
                "passages": 0,
                "mentions": 0,
                "co_occurs": 0,
                "relates": 0,
                "extractor_mode": str(self.settings.graph_entity_extractor or "rule"),
                "llm_passages_success": 0,
                "llm_passages_failed": 0,
            }
        rows = _passage_rows(docs)
        mention_rows: List[Dict[str, Any]] = []
        cooccurs_rows: List[Dict[str, Any]] = []
        relates_rows: List[Dict[str, Any]] = []
        relation_stats: Dict[str, Any] = {
            "extractor_mode": str(self.settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
            "relates": 0,
        }
        if self.settings.graph_relations_enabled:
            max_entities_per_passage = max(1, int(self.settings.graph_entity_max_per_passage))
            max_cooccurs_per_passage = max(1, int(self.settings.graph_cooccurs_max_per_passage))
            max_cooccurs_provenance_per_edge = max(
                1, int(self.settings.graph_cooccurs_provenance_limit)
            )
            entity_min_token_len = max(1, int(self.settings.graph_entity_min_token_len))
            entity_use_bigrams = bool(self.settings.graph_entity_use_bigrams)
            entity_max_bigrams_per_passage = max(
                0, int(self.settings.graph_entity_max_bigrams_per_passage)
            )
            mention_rows, cooccurs_rows, relates_rows, relation_stats = _relation_rows(
                rows,
                settings=self.settings,
                max_entities_per_passage=max_entities_per_passage,
                max_cooccurs_per_passage=max_cooccurs_per_passage,
                max_cooccurs_provenance_per_edge=max_cooccurs_provenance_per_edge,
                entity_min_token_len=entity_min_token_len,
                entity_use_bigrams=entity_use_bigrams,
                entity_max_bigrams_per_passage=entity_max_bigrams_per_passage,
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
            r.passage_ids = row.passage_ids,
            r.chunk_ids = row.chunk_ids,
            r.relation_labels = row.relation_labels,
            r.updated_at = row.updated_at
        """
        relates_query = """
        UNWIND $rows AS row
        MATCH (e1:Entity {id: row.source_entity_id})
        MATCH (e2:Entity {id: row.target_entity_id})
        MERGE (e1)-[r:RELATES {
            source_id: row.source_id,
            passage_id: row.passage_id,
            relation: row.relation
        }]->(e2)
        ON CREATE SET r.created_at = row.updated_at
        SET r.chunk_id = row.chunk_id,
            r.count = row.count,
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
            if relates_rows:
                session.run(relates_query, rows=relates_rows).consume()
        logger.info(
            "Neo4j relations upsert: mentions=%s co_occurs=%s relates=%s",
            len(mention_rows),
            len(cooccurs_rows),
            len(relates_rows),
        )
        return {
            "passages": int(len(rows)),
            "mentions": int(len(mention_rows)),
            "co_occurs": int(len(cooccurs_rows)),
            "relates": int(len(relates_rows)),
            "extractor_mode": str(relation_stats.get("extractor_mode") or "rule"),
            "llm_passages_success": int(relation_stats.get("llm_passages_success", 0)),
            "llm_passages_failed": int(relation_stats.get("llm_passages_failed", 0)),
        }

    def upsert_passages(self, docs: List[Document]) -> int:
        stats = self.upsert_passages_stats(docs)
        return int(stats.get("passages", 0))


def write_passages_to_graph_stats(docs: List[Document], settings: Settings) -> Dict[str, Any]:
    if not docs:
        return {
            "enabled": bool(settings.neo4j_enabled and settings.graph_write_enabled),
            "passages": 0,
            "mentions": 0,
            "co_occurs": 0,
            "relates": 0,
            "extractor_mode": str(settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
        }

    client = Neo4jGraphClient(settings)
    if not client.enabled:
        return {
            "enabled": False,
            "passages": 0,
            "mentions": 0,
            "co_occurs": 0,
            "relates": 0,
            "extractor_mode": str(settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
        }
    try:
        client.connect()
        client.ensure_schema()
        stats = client.upsert_passages_stats(docs)
        stats["enabled"] = True
        if int(stats.get("passages", 0)) > 0:
            logger.info("Neo4j upsert passages: %s", int(stats["passages"]))
        return stats
    finally:
        client.close()


def write_passages_to_graph(docs: List[Document], settings: Settings) -> int:
    stats = write_passages_to_graph_stats(docs, settings)
    return int(stats.get("passages", 0))
