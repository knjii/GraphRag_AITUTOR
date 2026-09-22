"""Независимые от графа пары и сборка эталона второго поколения."""

from __future__ import annotations

import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from functools import lru_cache
from statistics import median

from rag_textbook.evaluation.ablation import AblationResult
from rag_textbook.evaluation.crossref import anchors_of, references_of
from rag_textbook.evaluation.goldset import (
    MULTIHOP_PROMPT,
    SINGLE_PROMPT,
    GoldsetBuilder,
    is_exercise,
    is_toc,
    looks_leaky,
    references_numbering,
)
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, GoldQuestion, content_hash
from rag_textbook.utils.text import content_terms, lemmatize_token, tokenize, truncate

logger = get_logger("evaluation.pairgen")
SOURCES = ("chapter_random", "bm25", "dense", "explicit_ref", "cross_book")
DENSE_WARNING = "dense: векторы не переданы, источник пропущен"


@dataclass
class PairCandidate:
    left: str
    right: str
    source: str
    same_doc: bool
    ordinal_distance: int | None
    lexical_overlap: float
    evidence: str


@lru_cache(maxsize=8192)
def _terms(text: str) -> frozenset[str]:
    return frozenset(content_terms(text, lemmatize=True))


def _eligible(chunks: Sequence[Chunk]) -> list[Chunk]:
    return sorted(
        (c for c in chunks if len(c.text) >= 300 and not is_toc(c) and not is_exercise(c)),
        key=lambda c: c.id,
    )


def _allowed(left: Chunk, right: Chunk) -> bool:
    return left.id != right.id and (
        left.doc_id != right.doc_id or abs(left.ordinal - right.ordinal) >= 3
    )


def _overlap(left: frozenset[str], right: frozenset[str]) -> float:
    return len(left & right) / len(left | right) if left or right else 0.0


def _pair(left: Chunk, right: Chunk, source: str, evidence: str) -> PairCandidate:
    left, right = sorted((left, right), key=lambda c: c.id)
    same = left.doc_id == right.doc_id
    return PairCandidate(
        left.id, right.id, source, same,
        abs(left.ordinal - right.ordinal) if same else None,
        _overlap(_terms(left.text), _terms(right.text)), evidence,
    )


def _unique(pairs: Sequence[PairCandidate]) -> list[PairCandidate]:
    result: dict[tuple[str, str], PairCandidate] = {}
    for pair in pairs:
        result.setdefault((pair.left, pair.right), pair)
    return list(result.values())


def chapter_random(chunks: Sequence[Chunk], rng: random.Random) -> list[PairCandidate]:
    groups = defaultdict(list)
    for chunk in _eligible(chunks):
        chapter = ("header", chunk.headers[0]) if chunk.headers else ("window", chunk.ordinal // 40)
        groups[(chunk.doc_id, chapter)].append(chunk)
    pairs = []
    for (_, chapter), group in groups.items():
        for left in group:
            choices = [right for right in group if _allowed(left, right)]
            if choices:
                pairs.append(_pair(left, rng.choice(choices), "chapter_random", f"Глава: {chapter}"))
    return _unique(pairs)


def bm25(chunks: Sequence[Chunk], rng: random.Random) -> list[PairCandidate]:
    corpus = _eligible(chunks)
    if not corpus:
        return []
    terms = [_terms(c.text) for c in corpus]
    postings: dict[str, set[int]] = defaultdict(set)
    frequencies = []
    for index, chunk in enumerate(corpus):
        frequencies.append(Counter(
            term for raw in tokenize(chunk.text)
            if (term := lemmatize_token(raw)) in terms[index]
        ))
        for term in terms[index]:
            postings[term].add(index)
    lengths = [sum(freq.values()) for freq in frequencies]
    average = sum(lengths) / len(corpus) or 1.0
    idf = {t: math.log(1 + (len(corpus) - len(ids) + 0.5) / (len(ids) + 0.5))
           for t, ids in postings.items()}
    pairs = []
    for index, left in enumerate(corpus):
        query = sorted(terms[index], key=lambda t: (-idf[t], t))[:3]
        candidates = sorted(set().union(*(postings[t] for t in query)))
        candidates = [j for j in candidates if _allowed(left, corpus[j])]
        if not candidates:
            continue

        def score(j: int, query: list[str] = query) -> float:
            return sum(
                idf[t] * frequencies[j][t] * 2.5
                / (frequencies[j][t] + 1.5 * (0.25 + 0.75 * lengths[j] / average))
                for t in query
            )

        rng.shuffle(candidates)
        best = max(candidates, key=score)
        shared = sorted(set(query) & terms[best])
        pairs.append(_pair(left, corpus[best], "bm25", f"Термины: {', '.join(shared)}"))
    return _unique(pairs)


def dense(
    chunks: Sequence[Chunk], rng: random.Random,
    vectors: dict[str, Sequence[float]] | None = None,
) -> list[PairCandidate]:
    if not vectors:
        logger.warning(DENSE_WARNING)
        return []
    import numpy as np

    corpus = [c for c in _eligible(chunks) if c.id in vectors]
    normalized = []
    valid = []
    dimension = None
    for chunk in corpus:
        vector = np.asarray(vectors[chunk.id], dtype=float)
        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            raise ValueError(f"Некорректный вектор: {chunk.id}")
        if dimension is not None and vector.size != dimension:
            raise ValueError("Размерности векторов не совпадают")
        dimension = vector.size
        norm = np.linalg.norm(vector)
        if norm:
            normalized.append(vector / norm)
            valid.append(chunk)
    if not valid:
        return []
    matrix = np.asarray(normalized)
    pairs = []
    # Одна строка сходств за раз: память не растёт квадратично с корпусом.
    for index, left in enumerate(valid):
        choices = [j for j, right in enumerate(valid) if _allowed(left, right)]
        if not choices:
            continue
        scores = matrix @ matrix[index]
        rng.shuffle(choices)
        best = max(choices, key=lambda j: scores[j])
        pairs.append(_pair(left, valid[best], "dense", f"Косинус: {scores[best]:.6f}"))
    return _unique(pairs)


def explicit_ref(chunks: Sequence[Chunk], rng: random.Random) -> list[PairCandidate]:
    documents = defaultdict(list)
    for chunk in _eligible(chunks):
        documents[chunk.doc_id].append(chunk)
    pairs = []
    for group in documents.values():
        by_id = {c.id: c for c in group}
        anchors = anchors_of([c.model_dump() for c in group])
        for left in group:
            for kind, number in references_of(left.model_dump()):
                for target in sorted(anchors[kind].get(number, ())):
                    right = by_id[target]
                    if _allowed(left, right):
                        pairs.append(_pair(left, right, "explicit_ref", f"Ссылка: {kind} {number}"))
    pairs = _unique(pairs)
    rng.shuffle(pairs)
    return pairs


def cross_book(
    chunks: Sequence[Chunk], rng: random.Random, min_title_overlap: float = 0.5,
) -> list[PairCandidate]:
    if not 0 <= min_title_overlap <= 1:
        raise ValueError("Порог сходства заголовков должен быть от 0 до 1")
    sections = defaultdict(list)
    for chunk in _eligible(chunks):
        if chunk.headers and _terms(chunk.headers[-1]):
            sections[(chunk.doc_id, tuple(chunk.headers))].append(chunk)
    groups = list(sections.items())
    pairs = []
    for index, ((doc, headers), lefts) in enumerate(groups):
        for (other_doc, other_headers), rights in groups[index + 1:]:
            if doc == other_doc:
                continue
            overlap = _overlap(_terms(headers[-1]), _terms(other_headers[-1]))
            if overlap >= min_title_overlap:
                pairs.append(_pair(
                    rng.choice(lefts), rng.choice(rights), "cross_book",
                    f"Заголовки: {headers[-1]} / {other_headers[-1]}; Жаккар: {overlap:.3f}",
                ))
    return _unique(pairs)


def sample_pairs(
    chunks: Sequence[Chunk], per_source: int, seed: int,
    vectors: dict[str, Sequence[float]] | None = None,
) -> list[PairCandidate]:
    if per_source < 0:
        raise ValueError("Квота не может быть отрицательной")
    rng = random.Random(seed)
    owned: dict[tuple[str, str], PairCandidate] = {}
    for source in SOURCES:
        generator = globals()[source]
        candidates = generator(chunks, rng, vectors=vectors) if source == "dense" else generator(chunks, rng)
        for pair in candidates:
            key = (pair.left, pair.right)
            if key in owned:
                owned[key].evidence += f"; также источник: {source}"
            else:
                owned[key] = pair
    result = []
    for source in SOURCES:
        candidates = [p for p in owned.values() if p.source == source]
        rng.shuffle(candidates)
        result.extend(candidates[:per_source])
    return result


def summarize_pairs(
    pairs: Sequence[PairCandidate], per_source: int, *, vectors_available: bool = True,
) -> dict[str, dict[str, int | float | str | None]]:
    summary = {}
    for source in SOURCES:
        selected = [p for p in pairs if p.source == source]
        distances = [p.ordinal_distance for p in selected if p.ordinal_distance is not None]
        summary[source] = {
            "count": len(selected),
            "lexical_overlap": median([p.lexical_overlap for p in selected]) if selected else None,
            "ordinal_distance": median(distances) if distances else None,
            "shortfall": max(0, per_source - len(selected)),
            "warning": DENSE_WARNING if source == "dense" and not vectors_available else "",
        }
    return summary


def build_v2(
    builder: GoldsetBuilder, chunks: Sequence[Chunk], pairs: Sequence[PairCandidate],
    single_count: int, formula_count: int, seed: int,
) -> list[GoldQuestion]:
    if min(single_count, formula_count) < 0:
        raise ValueError("Количество вопросов не может быть отрицательным")
    builder.random.seed(seed)
    eligible = _eligible(chunks)
    by_id = {c.id: c for c in eligible}
    questions = []
    jobs: list[tuple[list[Chunk], str, str, str]] = []
    for formula, count in ((False, single_count), (True, formula_count)):
        subset = [c for c in eligible if bool(c.has_formula or c.has_table) == formula]
        selected = builder._select_single(subset, count * 2 if formula else count)[:count]
        for chunk in selected:
            jobs.append(([chunk], "formula" if formula else "single", "",
                         SINGLE_PROMPT.format(text=truncate(chunk.text, 6000))))
    seen = set()
    for pair in pairs:
        if pair.source not in SOURCES:
            raise ValueError(f"Неизвестный источник: {pair.source}")
        left, right = by_id[pair.left], by_id[pair.right]
        if not _allowed(left, right):
            raise ValueError(f"Недопустимая пара: {pair.left}, {pair.right}")
        key = tuple(sorted((left.id, right.id)))
        if key in seen:
            continue
        seen.add(key)
        jobs.append(([left, right], "linking" if left.doc_id == right.doc_id else "cross_book",
                     pair.source, MULTIHOP_PROMPT.format(
                         text_a=truncate(left.text, 3000), text_b=truncate(right.text, 3000))))
    answers = builder._ask_many([prompt for _, _, _, prompt in jobs])
    for (group, slice_name, source, _), produced in zip(jobs, answers, strict=True):
        if produced is None:
            continue
        question, answer = produced
        if looks_leaky(question):
            builder.failures["leaky"] += 1
            continue
        if references_numbering(question):
            builder.failures["numbered_reference"] += 1
            continue
        candidate = GoldQuestion(
            id=content_hash("v2", *(c.id for c in group), question)[:16],
            question=question, answer=answer, gold_chunk_ids=[c.id for c in group],
            gold_doc_ids=sorted({c.doc_id for c in group}),
            question_type="multi_hop" if source else builder._classify(group[0]),
            expected_hops=len(group), pair_source=source, slice=slice_name,
        )
        accepted = builder._accept(candidate, None)
        if accepted is not None:
            questions.append(accepted)
    return assign_split(questions, seed)


def keep_two_hop(
    questions: Sequence[GoldQuestion], ablation_results: Sequence[AblationResult],
) -> list[GoldQuestion]:
    verdicts = {r.question_id: r.verdict for r in ablation_results}
    return [q for q in questions if q.expected_hops <= 1 or verdicts.get(q.id) == "ok"]


def assign_split(
    questions: Sequence[GoldQuestion], seed: int, dev_share: float = 0.4,
) -> list[GoldQuestion]:
    if not 0 <= dev_share <= 1:
        raise ValueError("Доля dev должна быть от 0 до 1")
    result = []
    for question in questions:
        # Независимый хэш внутри каждой страты сохраняет назначение при дозаписи.
        # Доля вероятностная: точная квота несовместима с этой устойчивостью.
        key = json.dumps([seed, question.slice, question.pair_source, question.id], ensure_ascii=False)
        value = int.from_bytes(hashlib.sha256(key.encode()).digest(), "big") / 2**256
        result.append(question.model_copy(update={"split": "dev" if value < dev_share else "test"}))
    return result
