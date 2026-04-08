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

from ket_selector import (
    embed_texts_resilient,
    select_ket_core_passage_ids,
    split_sentences,
)
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


def _parse_llm_json_with_status(text: str) -> tuple[Dict[str, Any], str]:
    if not text or not str(text).strip():
        return {}, "empty_response"
    candidates: List[str] = [text]
    extracted = _extract_json_candidate(text)
    if extracted and extracted != text:
        candidates.append(extracted)
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj, "ok"
        except Exception:
            continue
    return {}, "invalid_json"


def _pick_entities_relations(parsed: Dict[str, Any]) -> tuple[List[Any], List[Any]]:
    entities_raw: List[Any] = []
    relations_raw: List[Any] = []

    for key in ("entities", "entity", "nodes", "terms", "concepts"):
        value = parsed.get(key)
        if isinstance(value, list):
            entities_raw = value
            break

    for key in ("relations", "edges", "relationships", "links", "triples"):
        value = parsed.get(key)
        if isinstance(value, list):
            relations_raw = value
            break

    def _is_relation_like(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        has_relation = any(k in item for k in ("relation", "predicate", "label"))
        has_target = any(k in item for k in ("target", "object", "to"))
        has_source = any(k in item for k in ("source", "subject", "from", "name"))
        return has_relation and has_target and has_source

    # Some models return relation objects inside `entities`.
    if entities_raw and not relations_raw:
        relation_like = [item for item in entities_raw if _is_relation_like(item)]
        if relation_like:
            relations_raw = relation_like
            entities_raw = [item for item in entities_raw if item not in relation_like]

    # Some models return only triples; derive entities from triples when possible.
    if not entities_raw and relations_raw:
        derived: List[Dict[str, str]] = []
        for rel in relations_raw:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source") or rel.get("subject") or rel.get("from") or rel.get("name") or ""
            dst = rel.get("target") or rel.get("object") or rel.get("to") or ""
            if str(src).strip():
                derived.append({"name": str(src)})
            if str(dst).strip():
                derived.append({"name": str(dst)})
        entities_raw = derived

    return entities_raw, relations_raw


def _has_graph_top_level_structure(parsed: Dict[str, Any]) -> bool:
    if not isinstance(parsed, dict):
        return False
    for key in ("entities", "entity", "nodes", "terms", "concepts", "relations", "edges", "relationships", "links", "triples"):
        if isinstance(parsed.get(key), list):
            return True
    return False


def _normalize_entity_name(name: str, min_token_len: int) -> str:
    if not name:
        return ""
    parts: List[str] = []
    for raw in _TOKEN_RE.findall(str(name)):
        token = _norm_token(raw)
        if _is_valid_entity_token(token, min_token_len):
            parts.append(token)
    return " ".join(parts[:6]).strip()


def _normalize_entity_name_soft(name: str, min_token_len: int) -> str:
    text = str(name or "").strip()
    if not text:
        return ""
    text = text.replace("\n", " ").replace("\t", " ")
    text = re.sub(r"\s+", " ", text)
    text = text.strip(".,;:!?()[]{}\"'`")
    if len(text) < max(2, int(min_token_len)):
        return ""
    if not any(ch.isalnum() for ch in text):
        return ""
    # Keep formulas/symbol-bearing terms when tokenizer-based normalization drops them.
    return text.lower().replace("ё", "е")[:96]


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
) -> Tuple[Dict[str, int], List[Tuple[str, str, str]], Dict[str, Any]]:
    if not text.strip():
        return {}, [], {"parse_status": "empty_input", "raw_len": 0, "parsed_keys": []}

    max_chars = max(500, int(settings.graph_llm_input_max_chars))
    model_name = settings.graph_llm_model or settings.ollama_model
    options = {
        "temperature": float(settings.graph_llm_temperature),
        "top_k": int(settings.ollama_top_k),
        "num_ctx": int(settings.ollama_num_ctx),
        "num_gpu": int(settings.ollama_num_gpu),
        "num_batch": int(settings.ollama_num_batch),
        "num_predict": int(settings.graph_llm_num_predict),
    }

    schema = {
        "type": "object",
        "properties": {
            "entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"name": {"type": "string"}},
                    "required": ["name"],
                },
            },
            "relations": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "relation": {"type": "string"},
                        "target": {"type": "string"},
                    },
                    "required": ["source", "relation", "target"],
                },
            },
        },
        "required": ["entities", "relations"],
    }

    raw = ""
    parse_status = "invalid_json"
    parsed: Dict[str, Any] = {}
    attempt_configs = [
        {
            "max_chars": max_chars,
            "max_entities": max_entities_per_passage,
            "max_relations": max_relations_per_passage,
            "format": schema,
            "suffix": "",
        },
        {
            "max_chars": min(max_chars, 2500),
            "max_entities": min(max_entities_per_passage, 10),
            "max_relations": min(max_relations_per_passage, 10),
            "format": "json",
            "suffix": "\n\nВАЖНО: Ответ только JSON-объект, без текста до/после, без markdown, без code fence.",
        },
        {
            "max_chars": min(max_chars, 1800),
            "max_entities": min(max_entities_per_passage, 8),
            "max_relations": min(max_relations_per_passage, 8),
            "format": "json",
            "suffix": "\n\nЕсли в тексте есть связанные технические термины, постарайся вернуть хотя бы 1 relation.",
        },
    ]
    for attempt, cfg in enumerate(attempt_configs):
        source_text = text[: int(cfg["max_chars"])]
        prompt_now = _build_graph_extract_prompt(
            source_text,
            max_entities=int(cfg["max_entities"]),
            max_relations=int(cfg["max_relations"]),
        ) + str(cfg["suffix"])
        format_now: dict | str = cfg["format"]  # type: ignore[assignment]
        try:
            raw = chat_with_ollama(
                message=prompt_now,
                model=model_name,
                options=options,
                settings=settings,
                response_format=format_now,
                think=False,
            )
            parsed, parse_status = _parse_llm_json_with_status(str(raw or ""))
            if parse_status == "ok":
                has_structure = _has_graph_top_level_structure(parsed)
                entities_probe, relations_probe = _pick_entities_relations(parsed)
                if not has_structure:
                    parse_status = "invalid_structure"
                elif entities_probe or relations_probe:
                    break
                else:
                    parse_status = "empty_json"
            if attempt < len(attempt_configs) - 1 and parse_status in {
                "invalid_json",
                "invalid_structure",
                "empty_json",
                "empty_response",
            }:
                continue
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

    entities_raw, relations_raw = _pick_entities_relations(parsed)

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
            if not normalized:
                normalized = _normalize_entity_name_soft(str(candidate), min_token_len=min_token_len)
            if normalized:
                entities_count[normalized] += 1
            if len(entities_count) >= max_entities_per_passage:
                break

    relations: List[Tuple[str, str, str]] = []
    if isinstance(relations_raw, list):
        for rel in relations_raw:
            if not isinstance(rel, dict):
                continue
            src = rel.get("source") or rel.get("subject") or rel.get("from") or rel.get("name") or ""
            dst = rel.get("target") or rel.get("object") or rel.get("to") or ""
            rtype = rel.get("relation") or rel.get("predicate") or rel.get("label") or ""
            src = _normalize_entity_name(str(src), min_token_len=min_token_len)
            dst = _normalize_entity_name(str(dst), min_token_len=min_token_len)
            if not src:
                src = _normalize_entity_name_soft(str(rel.get("source") or rel.get("subject") or rel.get("from") or rel.get("name") or ""), min_token_len=min_token_len)
            if not dst:
                dst = _normalize_entity_name_soft(str(rel.get("target") or rel.get("object") or rel.get("to") or ""), min_token_len=min_token_len)
            rtype = str(rtype).strip().lower()[:64]
            if not src or not dst or src == dst:
                continue
            if not rtype:
                rtype = "related_to"
            relations.append((src, rtype, dst))
            if len(relations) >= max_relations_per_passage:
                break

    # Final status after normalization/parsing.
    if parse_status == "ok" and not entities_count and not relations:
        parse_status = "empty_json"

    diag = {
        "parse_status": parse_status,
        "raw_len": len(str(raw or "")),
        "parsed_keys": sorted(parsed.keys()) if isinstance(parsed, dict) else [],
        "entities_raw_count": len(entities_raw) if isinstance(entities_raw, list) else 0,
        "relations_raw_count": len(relations_raw) if isinstance(relations_raw, list) else 0,
        "entities_norm_count": len(entities_count),
        "relations_norm_count": len(relations),
        "raw_preview": str(raw or "")[:240].replace("\n", "\\n"),
    }
    return dict(entities_count), relations, diag


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
    llm_core_passage_ids: set[str] | None = None,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    mention_rows: List[Dict[str, Any]] = []
    cooc_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    relates_counter: Counter[tuple[str, str, str, str, str, str]] = Counter()
    use_llm_extractor = str(settings.graph_entity_extractor).strip().lower() == "llm"
    llm_failures = 0
    llm_success = 0
    llm_entities_total = 0
    llm_relations_total = 0
    llm_applied_passages = 0
    llm_parse_invalid = 0
    llm_parse_empty = 0
    llm_parse_nonempty = 0
    llm_parse_invalid_structure = 0
    llm_parse_invalid_samples = 0
    llm_parse_empty_samples = 0
    llm_parse_invalid_structure_samples = 0
    rule_applied_passages = 0
    total_passages = len(passage_rows)
    progress_every = max(1, int(getattr(settings, "graph_llm_progress_every", 25)))
    started_at = time.monotonic()

    for idx, row in enumerate(passage_rows, start=1):
        entities: Dict[str, int] = {}
        llm_relations: List[Tuple[str, str, str]] = []

        apply_llm = bool(use_llm_extractor)
        if apply_llm and llm_core_passage_ids is not None:
            apply_llm = str(row.get("id", "")) in llm_core_passage_ids

        if apply_llm:
            llm_applied_passages += 1
            try:
                entities, llm_relations, llm_diag = _extract_entities_and_relations_llm(
                    row.get("text", ""),
                    settings,
                    max_entities_per_passage=max_entities_per_passage,
                    max_relations_per_passage=max(
                        1, int(getattr(settings, "graph_llm_max_relations_per_passage", 20))
                    ),
                    min_token_len=entity_min_token_len,
                )
                llm_success += 1
                parse_status = str(llm_diag.get("parse_status", "ok"))
                if parse_status == "invalid_json":
                    llm_parse_invalid += 1
                    if llm_parse_invalid_samples < 3:
                        llm_parse_invalid_samples += 1
                        logger.warning(
                            "Graph LLM parse invalid JSON sample %s (chunk=%s, keys=%s, raw_preview=%s)",
                            llm_parse_invalid_samples,
                            row.get("chunk_id"),
                            llm_diag.get("parsed_keys", []),
                            llm_diag.get("raw_preview", ""),
                        )
                elif parse_status == "invalid_structure":
                    llm_parse_invalid_structure += 1
                    if llm_parse_invalid_structure_samples < 3:
                        llm_parse_invalid_structure_samples += 1
                        logger.warning(
                            "Graph LLM parse invalid structure sample %s (chunk=%s, keys=%s, raw_preview=%s)",
                            llm_parse_invalid_structure_samples,
                            row.get("chunk_id"),
                            llm_diag.get("parsed_keys", []),
                            llm_diag.get("raw_preview", ""),
                        )
                elif parse_status == "empty_json":
                    llm_parse_empty += 1
                    if llm_parse_empty_samples < 3:
                        llm_parse_empty_samples += 1
                        logger.warning(
                            "Graph LLM parse empty JSON sample %s (chunk=%s, keys=%s, raw_preview=%s)",
                            llm_parse_empty_samples,
                            row.get("chunk_id"),
                            llm_diag.get("parsed_keys", []),
                            llm_diag.get("raw_preview", ""),
                        )
                else:
                    llm_parse_nonempty += 1
                llm_entities_total += int(len(entities))
                llm_relations_total += int(len(llm_relations))
                if not entities and not llm_relations and settings.graph_llm_fallback_to_rule:
                    rule_applied_passages += 1
                    entities = _extract_entities(
                        row.get("text", ""),
                        max_entities_per_passage,
                        min_token_len=entity_min_token_len,
                        use_bigrams=entity_use_bigrams,
                        max_bigrams_per_passage=entity_max_bigrams_per_passage,
                    )
            except Exception as exc:
                llm_failures += 1
                logger.warning("Graph LLM extraction failed for chunk %s: %s", row.get("chunk_id"), exc)
                if not settings.graph_llm_fallback_to_rule:
                    entities = {}
                else:
                    rule_applied_passages += 1
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
            rule_applied_passages += 1
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
            "Graph LLM extraction stats: passages=%s success=%s failed=%s parse_nonempty=%s parse_empty=%s parse_invalid=%s parse_invalid_structure=%s",
            len(passage_rows),
            llm_success,
            llm_failures,
            llm_parse_nonempty,
            llm_parse_empty,
            llm_parse_invalid,
            llm_parse_invalid_structure,
        )
    ket_enabled = llm_core_passage_ids is not None
    relation_stats = {
        "extractor_mode": "llm" if use_llm_extractor else "rule",
        "llm_passages_success": int(llm_success),
        "llm_passages_failed": int(llm_failures),
        "llm_applied_passages": int(llm_applied_passages),
        "llm_parse_nonempty": int(llm_parse_nonempty),
        "llm_parse_empty": int(llm_parse_empty),
        "llm_parse_invalid": int(llm_parse_invalid),
        "llm_parse_invalid_structure": int(llm_parse_invalid_structure),
        "rule_applied_passages": int(rule_applied_passages),
        "ket_core_passages": int(len(llm_core_passage_ids or set())) if ket_enabled else 0,
        "ket_periphery_passages": int(max(0, total_passages - len(llm_core_passage_ids or set())))
        if ket_enabled
        else 0,
        "relates": int(len(relates_rows)),
    }
    return mention_rows, cooccurs_rows, relates_rows, relation_stats


def _normalize_for_match(text: str) -> str:
    return str(text or "").lower().replace("ё", "е")


def _keyword_rows(
    passage_rows: List[Dict[str, Any]],
    mention_rows: List[Dict[str, Any]],
    *,
    settings: Settings,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    if not bool(settings.graph_keyword_channel_enabled):
        return [], [], [], {"keywords": 0, "has_keyword": 0, "keyword_near": 0, "embedded_keywords": 0}
    if not mention_rows:
        return [], [], [], {"keywords": 0, "has_keyword": 0, "keyword_near": 0, "embedded_keywords": 0}

    min_count = max(1, int(settings.graph_keyword_min_count))
    max_keywords = max(1, int(settings.graph_keyword_max_keywords))
    max_sentences_per_keyword = max(1, int(settings.graph_keyword_max_sentences_per_keyword))
    embedding_dims = max(0, int(settings.graph_keyword_embedding_dims))
    keyword_neighbor_cap = max(0, int(settings.graph_keyword_neighbors_per_passage))

    by_keyword_mentions: Dict[str, List[Dict[str, Any]]] = {}
    total_count: Counter[str] = Counter()
    passage_count: Dict[str, set[str]] = {}
    for row in mention_rows:
        keyword = str(row.get("entity_name", "") or "").strip().lower()
        if not keyword:
            continue
        by_keyword_mentions.setdefault(keyword, []).append(row)
        total_count[keyword] += int(row.get("count", 1) or 1)
        passage_count.setdefault(keyword, set()).add(str(row.get("passage_id", "")))

    candidates = [
        (name, int(total_count[name]), len(passage_count.get(name, set())))
        for name in by_keyword_mentions.keys()
        if int(total_count.get(name, 0)) >= min_count
    ]
    candidates.sort(key=lambda item: (item[1], item[2], item[0]), reverse=True)
    selected = candidates[:max_keywords]
    selected_names = {name for name, _, _ in selected}
    if not selected_names:
        return [], [], [], {"keywords": 0, "has_keyword": 0, "keyword_near": 0, "embedded_keywords": 0}

    logger.info(
        "Keyword channel start: selected_keywords=%s total_mentions=%s",
        len(selected_names),
        len(mention_rows),
    )

    passage_by_id = {str(row.get("id", "")): row for row in passage_rows}
    keyword_sentences: Dict[str, List[str]] = {}
    for keyword in selected_names:
        normalized_kw = _normalize_for_match(keyword)
        seen_sentences: set[str] = set()
        bucket: List[str] = []
        for mention in by_keyword_mentions.get(keyword, []):
            passage_id = str(mention.get("passage_id", ""))
            text = str((passage_by_id.get(passage_id) or {}).get("text", "") or "")
            if not text:
                continue
            for sentence in split_sentences(text, max_sentences=256):
                normalized_sentence = _normalize_for_match(sentence)
                if normalized_kw not in normalized_sentence:
                    continue
                if sentence in seen_sentences:
                    continue
                seen_sentences.add(sentence)
                bucket.append(sentence)
                if len(bucket) >= max_sentences_per_keyword:
                    break
            if len(bucket) >= max_sentences_per_keyword:
                break
        keyword_sentences[keyword] = bucket

    all_sentences: List[str] = []
    for sents in keyword_sentences.values():
        all_sentences.extend(sents)
    unique_sentences = list(dict.fromkeys(all_sentences))
    sentence_embeddings: Dict[str, List[float]] = {}
    if unique_sentences:
        logger.info(
            "Keyword channel embedding sentences: %s (batch=%s)",
            len(unique_sentences),
            max(1, int(settings.graph_keyword_embed_batch_size)),
        )
        vectors = embed_texts_resilient(
            unique_sentences,
            settings,
            batch_size=max(1, int(settings.graph_keyword_embed_batch_size)),
            purpose="keyword_embeddings",
        )
        if vectors is not None and len(vectors) == len(unique_sentences):
            for sentence, vec in zip(unique_sentences, vectors):
                sentence_embeddings[sentence] = vec

    ts = _now_utc_iso()
    keyword_rows: List[Dict[str, Any]] = []
    has_keyword_rows: List[Dict[str, Any]] = []
    keyword_id_by_name: Dict[str, str] = {}
    embedded_keywords = 0

    for keyword_name, kw_total_count, kw_passage_count in selected:
        keyword_id = hashlib.sha1(keyword_name.encode("utf-8", errors="ignore")).hexdigest()
        keyword_id_by_name[keyword_name] = keyword_id
        vecs: List[List[float]] = []
        for sent in keyword_sentences.get(keyword_name, []):
            vec = sentence_embeddings.get(sent)
            if vec:
                vecs.append(vec)
        keyword_embedding: List[float] = []
        if vecs:
            dim = min(len(v) for v in vecs)
            if dim > 0:
                if embedding_dims > 0:
                    dim = min(dim, embedding_dims)
                accum = [0.0] * dim
                for vec in vecs:
                    for i in range(dim):
                        accum[i] += float(vec[i])
                scale = 1.0 / float(len(vecs))
                keyword_embedding = [round(val * scale, 6) for val in accum]
                embedded_keywords += 1
        keyword_rows.append(
            {
                "id": keyword_id,
                "name": keyword_name,
                "count": int(kw_total_count),
                "passage_count": int(kw_passage_count),
                "embedding": keyword_embedding,
                "created_at": ts,
                "updated_at": ts,
            }
        )
        for mention in by_keyword_mentions.get(keyword_name, []):
            has_keyword_rows.append(
                {
                    "passage_id": str(mention.get("passage_id", "")),
                    "source_id": str(mention.get("source_id", "")),
                    "chunk_id": str(mention.get("chunk_id", "")),
                    "keyword_id": keyword_id,
                    "count": int(mention.get("count", 1) or 1),
                    "updated_at": ts,
                }
            )

    near_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    mentions_by_passage: Dict[str, List[Dict[str, Any]]] = {}
    for row in has_keyword_rows:
        mentions_by_passage.setdefault(str(row["passage_id"]), []).append(row)
    for passage_id, rows in mentions_by_passage.items():
        if len(rows) < 2:
            continue
        row0 = rows[0]
        source_id = str(row0.get("source_id", ""))
        chunk_id = str(row0.get("chunk_id", ""))
        local_pairs = 0
        for a, b in combinations(sorted({str(r.get("keyword_id", "")) for r in rows}), 2):
            key = (source_id, a, b)
            bucket = near_map.setdefault(
                key,
                {
                    "weight": 0,
                    "passage_ids": set(),
                    "chunk_ids": set(),
                },
            )
            bucket["weight"] += 1
            bucket["passage_ids"].add(passage_id)
            if chunk_id:
                bucket["chunk_ids"].add(chunk_id)
            local_pairs += 1
            if keyword_neighbor_cap > 0 and local_pairs >= keyword_neighbor_cap:
                break

    keyword_near_rows: List[Dict[str, Any]] = []
    for (source_id, k1, k2), payload in near_map.items():
        keyword_near_rows.append(
            {
                "source_id": source_id,
                "keyword1_id": k1,
                "keyword2_id": k2,
                "weight": int(payload["weight"]),
                "passage_ids": sorted(payload["passage_ids"]),
                "chunk_ids": sorted(payload["chunk_ids"]),
                "updated_at": ts,
            }
        )

    stats = {
        "keywords": int(len(keyword_rows)),
        "has_keyword": int(len(has_keyword_rows)),
        "keyword_near": int(len(keyword_near_rows)),
        "embedded_keywords": int(embedded_keywords),
    }
    logger.info(
        "Keyword channel done: keywords=%s has_keyword=%s keyword_near=%s embedded_keywords=%s",
        stats["keywords"],
        stats["has_keyword"],
        stats["keyword_near"],
        stats["embedded_keywords"],
    )
    return keyword_rows, has_keyword_rows, keyword_near_rows, stats


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
            "CREATE CONSTRAINT keyword_id IF NOT EXISTS FOR (k:Keyword) REQUIRE k.id IS UNIQUE",
            "CREATE CONSTRAINT source_id IF NOT EXISTS FOR (s:Source) REQUIRE s.id IS UNIQUE",
            "CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX keyword_name IF NOT EXISTS FOR (k:Keyword) ON (k.name)",
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
                "keywords": 0,
                "has_keyword": 0,
                "keyword_near": 0,
                "embedded_keywords": 0,
                "extractor_mode": str(self.settings.graph_entity_extractor or "rule"),
                "llm_passages_success": 0,
                "llm_passages_failed": 0,
                "llm_applied_passages": 0,
                "llm_parse_nonempty": 0,
                "llm_parse_empty": 0,
                "llm_parse_invalid": 0,
                "llm_parse_invalid_structure": 0,
                "rule_applied_passages": 0,
                "ket_enabled": False,
                "ket_core_passages": 0,
                "ket_periphery_passages": 0,
            }
        logger.info("Neo4j upsert stage start: docs=%s", len(docs))
        rows = _passage_rows(docs)
        logger.info("Neo4j passage rows prepared: %s", len(rows))
        mention_rows: List[Dict[str, Any]] = []
        cooccurs_rows: List[Dict[str, Any]] = []
        relates_rows: List[Dict[str, Any]] = []
        keyword_rows: List[Dict[str, Any]] = []
        has_keyword_rows: List[Dict[str, Any]] = []
        keyword_near_rows: List[Dict[str, Any]] = []
        llm_core_passage_ids: set[str] | None = None
        ket_report: Dict[str, Any] = {
            "enabled": False,
            "selected_core": 0,
            "selected_ratio": 0.0,
            "total_passages": int(len(rows)),
        }
        use_llm_extractor = str(self.settings.graph_entity_extractor).strip().lower() == "llm"
        if self.settings.ket_rag_enabled and use_llm_extractor and self.settings.graph_relations_enabled:
            llm_core_passage_ids, ket_report = select_ket_core_passage_ids(rows, self.settings)
            logger.info(
                "KET mode for graph extraction: core=%s/%s (semantic=%s, beta=%.3f)",
                int(ket_report.get("selected_core", 0)),
                int(ket_report.get("total_passages", len(rows))),
                bool(ket_report.get("semantic_available", False)),
                float(getattr(self.settings, "ket_beta", 0.2)),
            )

        relation_stats: Dict[str, Any] = {
            "extractor_mode": str(self.settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
            "llm_applied_passages": 0,
            "llm_parse_nonempty": 0,
            "llm_parse_empty": 0,
            "llm_parse_invalid": 0,
            "llm_parse_invalid_structure": 0,
            "rule_applied_passages": 0,
            "ket_core_passages": int(len(llm_core_passage_ids or set())),
            "ket_periphery_passages": int(max(0, len(rows) - len(llm_core_passage_ids or set()))),
            "relates": 0,
        }
        if self.settings.graph_relations_enabled:
            logger.info(
                "Neo4j relation extraction start: extractor=%s ket_enabled=%s",
                str(self.settings.graph_entity_extractor or "rule"),
                bool(self.settings.ket_rag_enabled),
            )
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
                llm_core_passage_ids=llm_core_passage_ids,
            )
            logger.info(
                "Neo4j relation extraction done: mentions=%s co_occurs=%s relates=%s",
                len(mention_rows),
                len(cooccurs_rows),
                len(relates_rows),
            )
            keyword_rows, has_keyword_rows, keyword_near_rows, keyword_stats = _keyword_rows(
                rows,
                mention_rows,
                settings=self.settings,
            )
        else:
            keyword_stats = {"keywords": 0, "has_keyword": 0, "keyword_near": 0, "embedded_keywords": 0}

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
        keyword_query = """
        UNWIND $rows AS row
        MERGE (k:Keyword {id: row.id})
        ON CREATE SET k.created_at = row.created_at
        SET k.name = row.name,
            k.count = row.count,
            k.passage_count = row.passage_count,
            k.embedding = row.embedding,
            k.updated_at = row.updated_at
        """
        has_keyword_query = """
        UNWIND $rows AS row
        MATCH (p:Passage {id: row.passage_id})
        MATCH (k:Keyword {id: row.keyword_id})
        MERGE (p)-[r:HAS_KEYWORD]->(k)
        SET r.count = row.count,
            r.source_id = row.source_id,
            r.chunk_id = row.chunk_id,
            r.updated_at = row.updated_at
        """
        keyword_near_query = """
        UNWIND $rows AS row
        MATCH (k1:Keyword {id: row.keyword1_id})
        MATCH (k2:Keyword {id: row.keyword2_id})
        MERGE (k1)-[r:KEYWORD_NEAR {source_id: row.source_id}]->(k2)
        SET r.weight = row.weight,
            r.passage_ids = row.passage_ids,
            r.chunk_ids = row.chunk_ids,
            r.updated_at = row.updated_at
        """
        with self._driver.session(database=self.settings.neo4j_database) as session:
            logger.info("Neo4j write query: passages (%s rows)", len(rows))
            session.run(passage_query, rows=rows).consume()
            logger.info("Neo4j write query done: passages")
            logger.info("Neo4j write query: NEXT edges")
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
            logger.info("Neo4j write query done: NEXT edges")
            if mention_rows:
                logger.info("Neo4j write query: MENTIONS (%s rows)", len(mention_rows))
                session.run(mentions_query, rows=mention_rows).consume()
                logger.info("Neo4j write query done: MENTIONS")
            if cooccurs_rows:
                logger.info("Neo4j write query: CO_OCCURS (%s rows)", len(cooccurs_rows))
                session.run(cooccurs_query, rows=cooccurs_rows).consume()
                logger.info("Neo4j write query done: CO_OCCURS")
            if relates_rows:
                logger.info("Neo4j write query: RELATES (%s rows)", len(relates_rows))
                session.run(relates_query, rows=relates_rows).consume()
                logger.info("Neo4j write query done: RELATES")
            if keyword_rows:
                logger.info("Neo4j write query: KEYWORD nodes (%s rows)", len(keyword_rows))
                session.run(keyword_query, rows=keyword_rows).consume()
                logger.info("Neo4j write query done: KEYWORD nodes")
            if has_keyword_rows:
                logger.info("Neo4j write query: HAS_KEYWORD (%s rows)", len(has_keyword_rows))
                session.run(has_keyword_query, rows=has_keyword_rows).consume()
                logger.info("Neo4j write query done: HAS_KEYWORD")
            if keyword_near_rows:
                logger.info("Neo4j write query: KEYWORD_NEAR (%s rows)", len(keyword_near_rows))
                session.run(keyword_near_query, rows=keyword_near_rows).consume()
                logger.info("Neo4j write query done: KEYWORD_NEAR")
        logger.info(
            "Neo4j relations upsert: mentions=%s co_occurs=%s relates=%s keywords=%s has_keyword=%s keyword_near=%s",
            len(mention_rows),
            len(cooccurs_rows),
            len(relates_rows),
            len(keyword_rows),
            len(has_keyword_rows),
            len(keyword_near_rows),
        )
        return {
            "passages": int(len(rows)),
            "mentions": int(len(mention_rows)),
            "co_occurs": int(len(cooccurs_rows)),
            "relates": int(len(relates_rows)),
            "keywords": int(keyword_stats.get("keywords", 0)),
            "has_keyword": int(keyword_stats.get("has_keyword", 0)),
            "keyword_near": int(keyword_stats.get("keyword_near", 0)),
            "embedded_keywords": int(keyword_stats.get("embedded_keywords", 0)),
            "extractor_mode": str(relation_stats.get("extractor_mode") or "rule"),
            "llm_passages_success": int(relation_stats.get("llm_passages_success", 0)),
            "llm_passages_failed": int(relation_stats.get("llm_passages_failed", 0)),
            "llm_applied_passages": int(relation_stats.get("llm_applied_passages", 0)),
            "llm_parse_nonempty": int(relation_stats.get("llm_parse_nonempty", 0)),
            "llm_parse_empty": int(relation_stats.get("llm_parse_empty", 0)),
            "llm_parse_invalid": int(relation_stats.get("llm_parse_invalid", 0)),
            "llm_parse_invalid_structure": int(
                relation_stats.get("llm_parse_invalid_structure", 0)
            ),
            "rule_applied_passages": int(relation_stats.get("rule_applied_passages", 0)),
            "ket_enabled": bool(ket_report.get("enabled", False)),
            "ket_core_passages": int(relation_stats.get("ket_core_passages", 0)),
            "ket_periphery_passages": int(relation_stats.get("ket_periphery_passages", 0)),
            "ket_selected_ratio": float(ket_report.get("selected_ratio", 0.0)),
            "ket_semantic_available": bool(ket_report.get("semantic_available", False)),
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
            "keywords": 0,
            "has_keyword": 0,
            "keyword_near": 0,
            "embedded_keywords": 0,
            "extractor_mode": str(settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
            "llm_applied_passages": 0,
            "llm_parse_nonempty": 0,
            "llm_parse_empty": 0,
            "llm_parse_invalid": 0,
            "llm_parse_invalid_structure": 0,
            "rule_applied_passages": 0,
            "ket_enabled": False,
            "ket_core_passages": 0,
            "ket_periphery_passages": 0,
            "ket_selected_ratio": 0.0,
            "ket_semantic_available": False,
        }

    client = Neo4jGraphClient(settings)
    if not client.enabled:
        return {
            "enabled": False,
            "passages": 0,
            "mentions": 0,
            "co_occurs": 0,
            "relates": 0,
            "keywords": 0,
            "has_keyword": 0,
            "keyword_near": 0,
            "embedded_keywords": 0,
            "extractor_mode": str(settings.graph_entity_extractor or "rule"),
            "llm_passages_success": 0,
            "llm_passages_failed": 0,
            "llm_applied_passages": 0,
            "llm_parse_nonempty": 0,
            "llm_parse_empty": 0,
            "llm_parse_invalid": 0,
            "llm_parse_invalid_structure": 0,
            "rule_applied_passages": 0,
            "ket_enabled": False,
            "ket_core_passages": 0,
            "ket_periphery_passages": 0,
            "ket_selected_ratio": 0.0,
            "ket_semantic_available": False,
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
