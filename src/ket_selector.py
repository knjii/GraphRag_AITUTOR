from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict
from typing import Any, Dict, List, Sequence, Tuple

from embeddings import get_embeddings_model
from settings import Settings
from utils import chat_with_ollama, get_logger

logger = get_logger("ket_selector")

_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_+\-/]{2,63}")
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
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
    "ее",
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
}


def _norm_token(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'`").lower().replace("ё", "е")


def _is_token_ok(token: str, min_len: int) -> bool:
    if len(token) < max(1, int(min_len)):
        return False
    if token in _STOPWORDS:
        return False
    if token.isdigit():
        return False
    if not any(ch.isalpha() for ch in token):
        return False
    return True


def extract_terms(text: str, *, min_len: int, max_terms: int, use_bigrams: bool) -> List[str]:
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(text or ""):
        tok = _norm_token(raw)
        if _is_token_ok(tok, min_len):
            tokens.append(tok)
    if not tokens:
        return []

    counts = Counter(tokens)
    terms = [name for name, _ in counts.most_common(max(1, int(max_terms)))]
    if use_bigrams and len(tokens) > 1:
        bigrams = Counter()
        for a, b in zip(tokens, tokens[1:]):
            if a != b:
                bigrams[f"{a} {b}"] += 1
        for phrase, _ in bigrams.most_common(max(1, int(max_terms // 2))):
            if phrase not in terms:
                terms.append(phrase)
    return terms[: max(1, int(max_terms))]


def split_sentences(text: str, *, max_sentences: int = 256) -> List[str]:
    raw_sentences = _SENTENCE_SPLIT_RE.split(text or "")
    out: List[str] = []
    for sentence in raw_sentences:
        normalized = " ".join(str(sentence).split()).strip()
        if not normalized:
            continue
        out.append(normalized)
        if len(out) >= max(1, int(max_sentences)):
            break
    return out


def _is_retryable_ollama_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "status code: 503" in text or "ggml_assert" in text


def _release_ollama_runners(settings: Settings) -> None:
    # Best-effort cleanup to reduce 503/runner contention.
    try:
        chat_with_ollama(turn_off=True, model=settings.ollama_model, settings=settings)
    except Exception:
        pass
    try:
        if settings.ollama_embed_model and settings.ollama_embed_model != settings.ollama_model:
            chat_with_ollama(turn_off=True, model=settings.ollama_embed_model, settings=settings)
    except Exception:
        pass


def embed_texts_resilient(
    texts: Sequence[str],
    settings: Settings,
    *,
    batch_size: int,
    purpose: str,
) -> List[List[float]] | None:
    clean_texts = [str(t or "").strip() for t in texts]
    if not clean_texts:
        return []
    try:
        model = get_embeddings_model(settings)
    except Exception as exc:
        logger.warning("KET %s: failed to initialize embeddings model: %s", purpose, exc)
        return None

    vectors: List[List[float]] = []
    step = max(1, int(batch_size))
    total = len(clean_texts)
    for start in range(0, total, step):
        batch = clean_texts[start : start + step]
        batch_idx = start // step + 1
        if batch_idx == 1 or batch_idx % 10 == 0 or start + len(batch) >= total:
            logger.info(
                "KET %s embedding progress: batch=%s processed=%s/%s",
                purpose,
                batch_idx,
                start + len(batch),
                total,
            )
        result = None
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                result = model.embed_documents(batch)
                break
            except Exception as exc:
                last_exc = exc
                if attempt < 2 and _is_retryable_ollama_error(exc):
                    _release_ollama_runners(settings)
                    time.sleep(0.8 * (attempt + 1))
                    continue
                break
        if result is None:
            logger.warning(
                "KET %s: embedding batch failed at %s/%s: %s",
                purpose,
                start + len(batch),
                len(clean_texts),
                last_exc,
            )
            return None
        for vec in result:
            try:
                vectors.append([float(x) for x in vec])
            except Exception:
                vectors.append([])
    if len(vectors) != len(clean_texts):
        return None
    return vectors


def _build_lexical_neighbors(term_sets: List[set[str]], *, k: int) -> List[List[Tuple[int, float]]]:
    n = len(term_sets)
    if n == 0 or k <= 0:
        return [[] for _ in range(n)]

    inverted: Dict[str, List[int]] = defaultdict(list)
    for idx, terms in enumerate(term_sets):
        for term in terms:
            inverted[term].append(idx)

    neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
    for i, terms_i in enumerate(term_sets):
        if i == 0 or (i + 1) % 500 == 0 or (i + 1) == n:
            logger.info("KET lexical neighbors progress: %s/%s", i + 1, n)
        if not terms_i:
            continue
        overlaps: Counter[int] = Counter()
        for term in terms_i:
            for j in inverted.get(term, []):
                if j != i:
                    overlaps[j] += 1
        scored: List[Tuple[int, float]] = []
        for j, overlap in overlaps.items():
            union = len(terms_i) + len(term_sets[j]) - overlap
            if union <= 0:
                continue
            score = float(overlap) / float(union)
            if score > 0:
                scored.append((j, score))
        scored.sort(key=lambda item: item[1], reverse=True)
        neighbors[i] = scored[: max(1, int(k))]
    return neighbors


def _build_semantic_neighbors(
    vectors: List[List[float]] | None,
    *,
    k: int,
) -> Tuple[List[List[Tuple[int, float]]], bool]:
    n = len(vectors or [])
    if vectors is None or n == 0 or k <= 0:
        return ([[] for _ in range(n)] if n > 0 else []), False
    try:
        import numpy as np

        arr = np.array(vectors, dtype=np.float32)
        if arr.ndim != 2 or arr.shape[0] == 0:
            return [[] for _ in range(n)], False
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        arr = arr / norms
        sims = arr @ arr.T
        np.fill_diagonal(sims, -1.0)
        top_k = max(1, int(k))
        neighbors: List[List[Tuple[int, float]]] = [[] for _ in range(n)]
        for i in range(n):
            row = sims[i]
            if row.size <= top_k:
                idxs = np.argsort(-row)
            else:
                idxs = np.argpartition(row, -top_k)[-top_k:]
                idxs = idxs[np.argsort(-row[idxs])]
            docs: List[Tuple[int, float]] = []
            for j in idxs.tolist():
                score = float(row[j])
                if j == i or score <= 0:
                    continue
                docs.append((int(j), score))
            neighbors[i] = docs
        return neighbors, True
    except Exception as exc:
        logger.warning("KET semantic neighbors disabled (numpy/cosine failed): %s", exc)
        return [[] for _ in range(n)], False


def _pick_core_count(
    n: int,
    *,
    beta: float,
    min_core: int,
    max_core: int | None,
) -> int:
    if n <= 0:
        return 0
    beta = min(1.0, max(0.0, float(beta)))
    target = int(round(float(n) * beta))
    target = max(int(min_core), target)
    if max_core is not None:
        max_core_int = int(max_core)
        if max_core_int > 0:
            target = min(max_core_int, target)
    return max(1, min(n, target))


def select_ket_core_passage_ids(
    passage_rows: Sequence[Dict[str, Any]],
    settings: Settings,
) -> tuple[set[str], Dict[str, Any]]:
    started_at = time.monotonic()
    n = len(passage_rows)
    if n == 0:
        return set(), {"enabled": False, "reason": "empty_input", "total_passages": 0}

    if not bool(getattr(settings, "ket_rag_enabled", False)):
        return set(), {"enabled": False, "reason": "ket_disabled", "total_passages": n}

    logger.info("KET selection started: passages=%s", n)

    k_total = max(2, int(getattr(settings, "ket_k", 12)))
    lexical_ratio = float(getattr(settings, "ket_lexical_ratio", 0.5))
    semantic_ratio = float(getattr(settings, "ket_semantic_ratio", 0.5))
    ratio_sum = lexical_ratio + semantic_ratio
    if ratio_sum <= 0:
        lexical_ratio, semantic_ratio = 0.5, 0.5
    else:
        lexical_ratio /= ratio_sum
        semantic_ratio /= ratio_sum

    k_lex = max(1, int(round(k_total * lexical_ratio)))
    k_sem = max(1, k_total - k_lex)
    min_token_len = max(1, int(getattr(settings, "ket_keyword_min_token_len", 3)))
    max_terms = max(8, int(getattr(settings, "ket_max_terms_per_passage", 64)))
    use_bigrams = bool(getattr(settings, "ket_use_bigrams", True))

    term_lists: List[List[str]] = []
    term_sets: List[set[str]] = []
    texts: List[str] = []
    for row in passage_rows:
        text = str(row.get("text", "") or "")
        texts.append(text)
        terms = extract_terms(
            text,
            min_len=min_token_len,
            max_terms=max_terms,
            use_bigrams=use_bigrams,
        )
        term_lists.append(terms)
        term_sets.append(set(terms))

    lexical_neighbors = _build_lexical_neighbors(term_sets, k=k_lex)
    logger.info("KET lexical graph built: passages=%s k_lex=%s", n, k_lex)
    semantic_vectors = embed_texts_resilient(
        texts,
        settings,
        batch_size=max(1, int(getattr(settings, "ket_embedding_batch_size", 48))),
        purpose="semantic_selection",
    )
    semantic_neighbors, semantic_ok = _build_semantic_neighbors(semantic_vectors, k=k_sem)
    if not semantic_ok and not bool(getattr(settings, "ket_semantic_fallback_to_lexical", True)):
        logger.warning("KET: semantic neighbors unavailable and fallback disabled; using lexical channel only.")

    channel_weight_lex = lexical_ratio
    channel_weight_sem = semantic_ratio if semantic_ok else 0.0
    if channel_weight_lex + channel_weight_sem <= 0:
        channel_weight_lex = 1.0
        channel_weight_sem = 0.0
    weight_sum = channel_weight_lex + channel_weight_sem
    channel_weight_lex /= weight_sum
    channel_weight_sem /= weight_sum

    graph_scores: Dict[int, float] = Counter()
    lexical_edges = 0
    semantic_edges = 0
    for i in range(n):
        for j, score in lexical_neighbors[i]:
            graph_scores[i] += channel_weight_lex * float(score)
            graph_scores[j] += channel_weight_lex * float(score)
            lexical_edges += 1
        for j, score in semantic_neighbors[i]:
            graph_scores[i] += channel_weight_sem * float(score)
            graph_scores[j] += channel_weight_sem * float(score)
            semantic_edges += 1

    ranked = sorted(range(n), key=lambda idx: float(graph_scores.get(idx, 0.0)), reverse=True)
    core_count = _pick_core_count(
        n,
        beta=float(getattr(settings, "ket_beta", 0.2)),
        min_core=max(1, int(getattr(settings, "ket_min_core_per_source", 1))),
        max_core=getattr(settings, "ket_max_core_per_source", 100000),
    )
    core_ids = {str(passage_rows[idx].get("id", "")) for idx in ranked[:core_count]}
    core_ids.discard("")

    # Safety: if IDs are missing, fall back to positional hashes.
    if len(core_ids) < core_count:
        for idx in ranked[:core_count]:
            row = passage_rows[idx]
            fallback = str(row.get("id") or hashlib.sha1(str(idx).encode("utf-8")).hexdigest())
            core_ids.add(fallback)

    elapsed = time.monotonic() - started_at
    report = {
        "enabled": True,
        "total_passages": int(n),
        "selected_core": int(len(core_ids)),
        "selected_ratio": (float(len(core_ids)) / float(n)) if n else 0.0,
        "ket_k": int(k_total),
        "k_lexical": int(k_lex),
        "k_semantic": int(k_sem),
        "semantic_available": bool(semantic_ok),
        "lexical_edges": int(lexical_edges),
        "semantic_edges": int(semantic_edges),
        "beta": float(getattr(settings, "ket_beta", 0.2)),
        "elapsed_sec": float(round(elapsed, 3)),
    }
    logger.info(
        "KET selection: passages=%s core=%s (%.1f%%) k=%s (lex=%s sem=%s) semantic=%s elapsed=%.2fs",
        report["total_passages"],
        report["selected_core"],
        report["selected_ratio"] * 100.0,
        report["ket_k"],
        report["k_lexical"],
        report["k_semantic"],
        report["semantic_available"],
        report["elapsed_sec"],
    )
    return core_ids, report
