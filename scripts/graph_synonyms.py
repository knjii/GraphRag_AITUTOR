"""Рёбра-синонимы между сущностями графа (гипотеза К11, схема HippoRAG 2).

Сущности сливаются в узел только при буквальном совпадении канонической
формы, поэтому разные книги об одном понятии остаются несвязанными:
«внутреннее произведение» (MML) и «скалярное произведение векторов»
(Гельфанд) — разные узлы. На межкнижном эталоне у 3 из 60 пар test есть
общий узел (день 4, 2026-09-28). Скрипт добавляет ребро ``SYNONYM``
между понятиями, чьи эмбеддинги названий близки: k ближайших соседей
с косинусом не ниже порога, вес ребра — косинус.

Обозначения (``notation``) не связываются: «B» одной книги и «B» другой —
разные объекты. Хабы (понятие в > ``--max-degree`` фрагментах) тоже:
через них синонимия соединила бы всё со всем.

    # на сервере, эмбеддер сервиса (bge-m3):
    python scripts/graph_synonyms.py --graph artifacts/graphs/v4.json.gz \
        --out artifacts/graphs/v4syn-080.json.gz --threshold 0.80

    # дома, без сервиса — только чтобы отбросить идею, не подтвердить:
    python scripts/graph_synonyms.py ... --embedder tfidf

Поиск ходит по новым рёбрам при ``GRAPH_EXPANSION_REL_TYPES=RELATES,SYNONYM``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.stores.graph_file import GraphFile, file_hash  # noqa: E402

REL = "SYNONYM"


def candidates(graph: GraphFile, max_degree: int) -> list[str]:
    """Понятия, которые можно связывать: не обозначения и не хабы."""
    degree: Counter[str] = Counter()
    for mentions in graph.mentions.values():
        degree.update(mentions.keys())
    return sorted(eid for eid, e in graph.entities.items()
                  if e["kind"] == "concept" and 0 < degree[eid] <= max_degree)


def synonym_edges(ids: Sequence[str], vectors: np.ndarray, threshold: float,
                  k: int) -> list[tuple[str, str, float]]:
    """Пары (a, b, косинус) с a < b: b среди k ближайших к a или наоборот."""
    if len(ids) < 2:
        return []
    norm = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    edges: dict[tuple[str, str], float] = {}
    step = 1024
    for start in range(0, len(ids), step):
        sims = norm[start:start + step] @ norm.T
        for row, i in enumerate(range(start, min(start + step, len(ids)))):
            sims[row, i] = -1.0
            top = np.argpartition(-sims[row], min(k, len(ids) - 1) - 1)[:k]
            for j in top:
                cos = float(sims[row, j])
                if cos >= threshold:
                    a, b = sorted((ids[i], ids[int(j)]))
                    edges[(a, b)] = max(cos, edges.get((a, b), -1.0))
    return sorted((a, b, round(c, 4)) for (a, b), c in edges.items())


def embed_tfidf(names: Sequence[str]) -> np.ndarray:
    from sklearn.feature_extraction.text import TfidfVectorizer
    matrix = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                             sublinear_tf=True).fit_transform(names)
    return matrix.toarray().astype(np.float32)


def embed_service(names: Sequence[str]) -> np.ndarray:
    from rag_textbook.clients.embeddings import build_embedding_client
    from rag_textbook.config import Settings
    settings = Settings()
    # Без префикса документа: сравниваются названия, а не фрагменты.
    client = build_embedding_client(settings.embedding.model_copy(update={"document_prefix": ""}))
    vectors = client.embed_documents(list(names))
    if any(not v for v in vectors):
        raise SystemExit("эмбеддер вернул пустые векторы — сервис поднят?")
    return np.asarray(vectors, dtype=np.float32)


def book_of(graph: GraphFile) -> dict[str, set[str]]:
    books: dict[str, set[str]] = defaultdict(set)
    for pid, mentions in graph.mentions.items():
        doc = graph.passages.get(pid, {}).get("doc_id", pid.split(":")[0])
        for eid in mentions:
            books[eid].add(doc)
    return books


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.80, help="косинус, HippoRAG 2: 0.8")
    parser.add_argument("--k", type=int, default=10, help="ближайших соседей на понятие")
    parser.add_argument("--max-degree", type=int, default=64, help="понятия-хабы не связываются")
    parser.add_argument("--embedder", choices=("service", "tfidf"), default="service")
    args = parser.parse_args(argv)

    graph = GraphFile.load(args.graph)
    if any(rel == REL for _, _, rel, _, _ in graph.relations):
        raise SystemExit(f"{args.graph} уже содержит {REL}: строить от графа без синонимов")
    ids = candidates(graph, args.max_degree)
    names = [graph.entities[eid]["canonical"] for eid in ids]
    vectors = embed_tfidf(names) if args.embedder == "tfidf" else embed_service(names)
    edges = synonym_edges(ids, vectors, args.threshold, args.k)
    books = book_of(graph)
    for a, b, cos in edges:
        graph.add_relation(a, b, REL, "синоним", cos)
    problems = graph.validate()
    if problems:
        raise SystemExit(f"граф испорчен: {problems[:5]}")
    graph.save(args.out)
    cross = sum(1 for a, b, _ in edges if books[a] != books[b])
    report = {"source": str(args.graph), "source_sha256": file_hash(args.graph),
              "embedder": args.embedder, "threshold": args.threshold, "k": args.k,
              "max_degree": args.max_degree, "candidates": len(ids), "edges": len(edges),
              "edges_between_books": cross, "out_sha256": file_hash(args.out),
              "examples": [(graph.entities[a]["canonical"], graph.entities[b]["canonical"], c)
                           for a, b, c in edges[:: max(1, len(edges) // 12)][:12]]}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    args.out.with_suffix("").with_suffix(".report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
