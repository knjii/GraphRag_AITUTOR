from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from ket_selector import embed_texts_resilient
from settings import Settings
from utils import get_logger
from vectorstore import load_vector_store

try:
    from neo4j import GraphDatabase
except Exception:  # pragma: no cover
    GraphDatabase = None

logger = get_logger("retriever")
_TOKEN_RE = re.compile(r"[A-Za-zА-Яа-яЁё][A-Za-zА-Яа-яЁё0-9_+\-/]{2,63}")


def _normalize_weights(dense_weight: float, sparse_weight: float) -> tuple[float, float]:
    dense = max(0.0, float(dense_weight))
    sparse = max(0.0, float(sparse_weight))
    total = dense + sparse
    if total <= 0:
        return 0.5, 0.5
    return dense / total, sparse / total


def _retrieve(retriever: BaseRetriever, query: str) -> List[Document]:
    if hasattr(retriever, "invoke"):
        return list(retriever.invoke(query))
    return list(retriever.get_relevant_documents(query))


def _norm_query_token(token: str) -> str:
    return token.strip(".,;:!?()[]{}\"'`").lower().replace("ё", "е")


def _query_candidates(query: str, limit: int) -> List[str]:
    tokens: List[str] = []
    for raw in _TOKEN_RE.findall(query or ""):
        tok = _norm_query_token(raw)
        if len(tok) >= 3 and tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= max(1, int(limit)):
            break
    bigrams: List[str] = []
    for left, right in zip(tokens, tokens[1:]):
        phrase = f"{left} {right}"
        if phrase not in bigrams:
            bigrams.append(phrase)
    return (bigrams + tokens)[: max(1, int(limit))]


def _doc_key(doc: Document) -> Tuple[str, str, str, str]:
    meta = doc.metadata or {}
    return (
        str(meta.get("source", "")),
        str(meta.get("page", "")),
        str(meta.get("chunk_id", "")),
        (doc.page_content or "").strip(),
    )


def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    dim = min(len(a), len(b))
    if dim <= 0:
        return 0.0
    dot = 0.0
    na = 0.0
    nb = 0.0
    for i in range(dim):
        av = float(a[i])
        bv = float(b[i])
        dot += av * bv
        na += av * av
        nb += bv * bv
    if na <= 0 or nb <= 0:
        return 0.0
    return dot / ((na ** 0.5) * (nb ** 0.5))


def _safe_text_for_embedding(text: str, max_chars: int) -> str:
    value = str(text or "").strip()
    if not value:
        return ""
    return value[: max(200, int(max_chars))]


def _cosine_rerank_documents(
    *,
    query: str,
    docs_by_key: Dict[Tuple[str, str, str, str], Document],
    key_weight_sum: Dict[Tuple[str, str, str, str], float],
    settings: Any,
    batch_size: int,
    max_chars: int,
    purpose: str,
    fallback_scores: Dict[Tuple[str, str, str, str], float] | None = None,
) -> List[Document]:
    if settings is None:
        return []
    keys = list(docs_by_key.keys())
    if not keys:
        return []

    query_vecs = embed_texts_resilient(
        [query],
        settings,
        batch_size=1,
        purpose=f"{purpose}_query",
    )
    if not query_vecs or not query_vecs[0]:
        return []
    query_vec = query_vecs[0]

    texts = [_safe_text_for_embedding(docs_by_key[key].page_content, max_chars) for key in keys]
    doc_vecs = embed_texts_resilient(
        texts,
        settings,
        batch_size=max(1, int(batch_size)),
        purpose=f"{purpose}_docs",
    )
    if not doc_vecs:
        return []

    scored: List[Tuple[float, Tuple[str, str, str, str]]] = []
    for idx, key in enumerate(keys):
        vec = doc_vecs[idx] if idx < len(doc_vecs) else []
        if not vec:
            continue
        sim = _cosine(query_vec, vec)
        channel_weight = max(0.0, float(key_weight_sum.get(key, 1.0)))
        score = sim * (channel_weight if channel_weight > 0 else 1.0)
        if fallback_scores is not None:
            score += float(fallback_scores.get(key, 0.0)) * 1e-6
        scored.append((score, key))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [docs_by_key[key] for _, key in scored]


class WeightedHybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    top_k: int = 4
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60
    fusion_mode: str = "rrf"
    settings: Any = None
    cosine_batch_size: int = 32
    cosine_max_chars: int = 4000

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        dense_docs = _retrieve(self.dense_retriever, query)
        sparse_docs = _retrieve(self.sparse_retriever, query)

        dense_weight, sparse_weight = _normalize_weights(self.dense_weight, self.sparse_weight)
        scored: Dict[Tuple[str, str, str, str], float] = {}
        by_key: Dict[Tuple[str, str, str, str], Document] = {}
        key_weight_sum: Dict[Tuple[str, str, str, str], float] = {}

        for idx, doc in enumerate(dense_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + dense_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + dense_weight
        for idx, doc in enumerate(sparse_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + sparse_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + sparse_weight

        if str(self.fusion_mode).strip().lower() == "cosine":
            reranked = _cosine_rerank_documents(
                query=query,
                docs_by_key=by_key,
                key_weight_sum=key_weight_sum,
                settings=self.settings,
                batch_size=max(1, int(self.cosine_batch_size)),
                max_chars=max(200, int(self.cosine_max_chars)),
                purpose="hybrid_fusion",
                fallback_scores=scored,
            )
            if reranked:
                return reranked[: max(1, int(self.top_k))]
            logger.warning("Cosine rerank failed in hybrid fusion. Falling back to RRF.")

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return [by_key[key] for key, _ in ranked[: max(1, int(self.top_k))]]


class Neo4jEntityRetriever(BaseRetriever):
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    hops: int = 1
    entity_limit: int = 30
    passage_limit: int = 30
    query_candidate_limit: int = 24

    def _find_seed_entity_ids(self, session, candidates: List[str]) -> List[str]:
        if not candidates:
            return []
        exact = session.run(
            """
            MATCH (e:Entity)
            WHERE e.name IN $candidates
            RETURN collect(DISTINCT e.id)[0..$entity_limit] AS ids
            """,
            candidates=candidates,
            entity_limit=max(1, int(self.entity_limit)),
        ).single()
        seed_ids = list((exact or {}).get("ids") or [])
        if seed_ids:
            return seed_ids

        fuzzy = session.run(
            """
            MATCH (e:Entity)
            WHERE any(c IN $candidates WHERE e.name CONTAINS c)
            RETURN collect(DISTINCT e.id)[0..$entity_limit] AS ids
            """,
            candidates=candidates,
            entity_limit=max(1, int(self.entity_limit)),
        ).single()
        return list((fuzzy or {}).get("ids") or [])

    def _expand_entity_ids(self, session, seed_ids: List[str]) -> List[str]:
        if not seed_ids:
            return []
        hops = max(1, int(self.hops))
        rec = session.run(
            f"""
            UNWIND $seed_ids AS seed_id
            MATCH (s:Entity {{id: seed_id}})
            OPTIONAL MATCH p=(s)-[:RELATES|CO_OCCURS*1..{hops}]-(n:Entity)
            WITH collect(DISTINCT s.id) + collect(DISTINCT n.id) AS ids
            UNWIND ids AS id
            RETURN collect(DISTINCT id)[0..$entity_limit] AS entity_ids
            """,
            seed_ids=seed_ids,
            entity_limit=max(1, int(self.entity_limit)),
        ).single()
        return list((rec or {}).get("entity_ids") or [])

    def _find_passages(self, session, entity_ids: List[str]) -> List[Document]:
        if not entity_ids:
            return []
        rows = session.run(
            """
            UNWIND $entity_ids AS entity_id
            MATCH (p:Passage)-[:MENTIONS]->(e:Entity {id: entity_id})
            WITH p, count(*) AS score, collect(DISTINCT e.name)[0..5] AS matched_entities
            RETURN
              p.text AS text,
              p.source AS source,
              p.order_in_source AS page,
              p.chunk_id AS chunk_id,
              score,
              matched_entities
            ORDER BY score DESC, p.order_in_source ASC
            LIMIT $passage_limit
            """,
            entity_ids=entity_ids,
            passage_limit=max(1, int(self.passage_limit)),
        ).data()
        docs: List[Document] = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(row.get("source") or ""),
                        "page": row.get("page"),
                        "chunk_id": str(row.get("chunk_id") or ""),
                        "graph_score": int(row.get("score") or 0),
                        "graph_matched_entities": list(row.get("matched_entities") or []),
                        "retrieval_source": "neo4j_graph",
                    },
                )
            )
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        if GraphDatabase is None:
            return []
        if not self.neo4j_password:
            return []
        candidates = _query_candidates(query, self.query_candidate_limit)
        if not candidates:
            return []
        driver = None
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            with driver.session(database=self.neo4j_database) as session:
                seed_ids = self._find_seed_entity_ids(session, candidates)
                entity_ids = self._expand_entity_ids(session, seed_ids) if seed_ids else []
                return self._find_passages(session, entity_ids)
        except Exception as exc:
            logger.warning("Graph retriever failed (fallback to base retriever): %s", exc)
            return []
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass


class Neo4jKeywordRetriever(BaseRetriever):
    neo4j_uri: str
    neo4j_user: str
    neo4j_password: str
    neo4j_database: str = "neo4j"
    keyword_limit: int = 40
    passage_limit: int = 30
    query_candidate_limit: int = 24
    embedding_enabled: bool = True
    embedding_batch_size: int = 32
    settings: Any = None

    def _find_keywords(self, session, candidates: List[str]) -> List[Dict]:
        if not candidates:
            return []
        rows = session.run(
            """
            MATCH (k:Keyword)
            WHERE k.name IN $candidates OR any(c IN $candidates WHERE k.name CONTAINS c)
            RETURN k.id AS id, k.name AS name, k.embedding AS embedding, k.count AS count
            ORDER BY coalesce(k.count, 0) DESC
            LIMIT $keyword_limit
            """,
            candidates=candidates,
            keyword_limit=max(1, int(self.keyword_limit * 3)),
        ).data()
        return rows

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        if not a or not b:
            return 0.0
        dim = min(len(a), len(b))
        if dim <= 0:
            return 0.0
        dot = 0.0
        na = 0.0
        nb = 0.0
        for i in range(dim):
            av = float(a[i])
            bv = float(b[i])
            dot += av * bv
            na += av * av
            nb += bv * bv
        if na <= 0 or nb <= 0:
            return 0.0
        return dot / ((na ** 0.5) * (nb ** 0.5))

    def _rank_keywords(self, keywords: List[Dict], query: str) -> List[Dict]:
        if not keywords:
            return []
        if not self.embedding_enabled or self.settings is None:
            return keywords[: max(1, int(self.keyword_limit))]

        query_vecs = embed_texts_resilient(
            [query],
            self.settings,
            batch_size=max(1, int(self.embedding_batch_size)),
            purpose="graph_keyword_query",
        )
        if not query_vecs or not query_vecs[0]:
            return keywords[: max(1, int(self.keyword_limit))]
        query_vec = query_vecs[0]

        scored: List[Tuple[float, Dict]] = []
        for row in keywords:
            emb = row.get("embedding")
            if not isinstance(emb, list):
                scored.append((0.0, row))
                continue
            try:
                kw_vec = [float(x) for x in emb]
            except Exception:
                kw_vec = []
            score = self._cosine(query_vec, kw_vec)
            scored.append((score, row))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [row for _, row in scored[: max(1, int(self.keyword_limit))]]

    def _find_passages(self, session, keyword_ids: List[str]) -> List[Document]:
        if not keyword_ids:
            return []
        rows = session.run(
            """
            UNWIND $keyword_ids AS keyword_id
            MATCH (p:Passage)-[hk:HAS_KEYWORD]->(k:Keyword {id: keyword_id})
            WITH p, sum(coalesce(hk.count, 1)) AS score, collect(DISTINCT k.name)[0..6] AS matched_keywords
            OPTIONAL MATCH (p)-[:NEXT]->(pn:Passage)
            WITH p, score, matched_keywords, collect(DISTINCT pn)[0..1] AS next_passages
            RETURN
              p.text AS text,
              p.source AS source,
              p.order_in_source AS page,
              p.chunk_id AS chunk_id,
              score,
              matched_keywords,
              [x IN next_passages WHERE x IS NOT NULL | x.text][0] AS next_text
            ORDER BY score DESC, p.order_in_source ASC
            LIMIT $passage_limit
            """,
            keyword_ids=keyword_ids,
            passage_limit=max(1, int(self.passage_limit)),
        ).data()
        docs: List[Document] = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(row.get("source") or ""),
                        "page": row.get("page"),
                        "chunk_id": str(row.get("chunk_id") or ""),
                        "graph_score": int(row.get("score") or 0),
                        "graph_matched_keywords": list(row.get("matched_keywords") or []),
                        "graph_next_text": str(row.get("next_text") or ""),
                        "retrieval_source": "neo4j_keyword_graph",
                    },
                )
            )
        return docs

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        if GraphDatabase is None or not self.neo4j_password:
            return []
        candidates = _query_candidates(query, self.query_candidate_limit)
        if not candidates:
            return []
        driver = None
        try:
            driver = GraphDatabase.driver(
                self.neo4j_uri,
                auth=(self.neo4j_user, self.neo4j_password),
            )
            with driver.session(database=self.neo4j_database) as session:
                keywords = self._find_keywords(session, candidates)
                ranked = self._rank_keywords(keywords, query)
                keyword_ids = [str(row.get("id") or "") for row in ranked if row.get("id")]
                return self._find_passages(session, keyword_ids)
        except Exception as exc:
            logger.warning("Keyword graph retriever failed (fallback to base retriever): %s", exc)
            return []
        finally:
            if driver is not None:
                try:
                    driver.close()
                except Exception:
                    pass


class ThetaGraphRetriever(BaseRetriever):
    entity_retriever: BaseRetriever
    keyword_retriever: BaseRetriever
    top_k: int = 30
    theta: float = 0.65
    rrf_k: int = 60
    fusion_mode: str = "rrf"
    settings: Any = None
    cosine_batch_size: int = 32
    cosine_max_chars: int = 4000

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        gs_docs = _retrieve(self.entity_retriever, query)
        gk_docs = _retrieve(self.keyword_retriever, query)
        theta = max(0.0, min(1.0, float(self.theta)))
        gk_weight = 1.0 - theta
        scored: Dict[Tuple[str, str, str, str], float] = {}
        by_key: Dict[Tuple[str, str, str, str], Document] = {}
        key_weight_sum: Dict[Tuple[str, str, str, str], float] = {}

        for idx, doc in enumerate(gs_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + theta / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + theta
        for idx, doc in enumerate(gk_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + gk_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + gk_weight

        if str(self.fusion_mode).strip().lower() == "cosine":
            reranked = _cosine_rerank_documents(
                query=query,
                docs_by_key=by_key,
                key_weight_sum=key_weight_sum,
                settings=self.settings,
                batch_size=max(1, int(self.cosine_batch_size)),
                max_chars=max(200, int(self.cosine_max_chars)),
                purpose="graph_channel_fusion",
                fallback_scores=scored,
            )
            if reranked:
                return reranked[: max(1, int(self.top_k))]
            logger.warning("Cosine rerank failed in graph channel fusion. Falling back to RRF.")

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return [by_key[key] for key, _ in ranked[: max(1, int(self.top_k))]]


class WeightedGraphFusionRetriever(BaseRetriever):
    base_retriever: BaseRetriever
    graph_retriever: BaseRetriever
    top_k: int = 4
    base_weight: float = 0.65
    graph_weight: float = 0.35
    rrf_k: int = 60
    fusion_mode: str = "rrf"
    settings: Any = None
    cosine_batch_size: int = 32
    cosine_max_chars: int = 4000
    min_graph_docs_in_final: int = 0

    @staticmethod
    def _dedup_keys(keys: List[Tuple[str, str, str, str]]) -> List[Tuple[str, str, str, str]]:
        seen: set[Tuple[str, str, str, str]] = set()
        out: List[Tuple[str, str, str, str]] = []
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
        return out

    def _apply_min_graph_docs(
        self,
        selected_keys: List[Tuple[str, str, str, str]],
        graph_ranked_keys: List[Tuple[str, str, str, str]],
        top_k: int,
    ) -> List[Tuple[str, str, str, str]]:
        top_k = max(1, int(top_k))
        min_graph = max(0, int(self.min_graph_docs_in_final))
        if min_graph <= 0:
            return selected_keys[:top_k]

        min_graph = min(min_graph, top_k)
        graph_ranked_keys = self._dedup_keys(graph_ranked_keys)
        graph_key_set = set(graph_ranked_keys)
        if not graph_key_set:
            return selected_keys[:top_k]

        selected = self._dedup_keys(selected_keys)[:top_k]
        graph_count = sum(1 for key in selected if key in graph_key_set)
        if graph_count >= min_graph:
            return selected

        for key in graph_ranked_keys:
            if key in selected:
                continue
            selected.append(key)
            graph_count += 1
            if graph_count >= min_graph:
                break

        idx = len(selected) - 1
        while len(selected) > top_k and idx >= 0:
            key = selected[idx]
            is_graph = key in graph_key_set
            if not is_graph:
                selected.pop(idx)
            else:
                current_graph = sum(1 for k in selected if k in graph_key_set)
                if current_graph > min_graph:
                    selected.pop(idx)
            idx -= 1

        while len(selected) > top_k:
            selected.pop()

        final_graph = sum(1 for key in selected if key in graph_key_set)
        if final_graph < min_graph:
            logger.warning(
                "Graph minimum in final context was not fully satisfied (%s/%s). Increase graph candidate budget.",
                final_graph,
                min_graph,
            )
        return selected[:top_k]

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        base_docs = _retrieve(self.base_retriever, query)
        graph_docs = _retrieve(self.graph_retriever, query)
        graph_key_set = {_doc_key(doc) for doc in graph_docs}

        base_weight, graph_weight = _normalize_weights(self.base_weight, self.graph_weight)
        scored: Dict[Tuple[str, str, str, str], float] = {}
        by_key: Dict[Tuple[str, str, str, str], Document] = {}
        key_weight_sum: Dict[Tuple[str, str, str, str], float] = {}

        for idx, doc in enumerate(base_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + base_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + base_weight
        for idx, doc in enumerate(graph_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + graph_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
            key_weight_sum[key] = key_weight_sum.get(key, 0.0) + graph_weight

        if str(self.fusion_mode).strip().lower() == "cosine":
            reranked = _cosine_rerank_documents(
                query=query,
                docs_by_key=by_key,
                key_weight_sum=key_weight_sum,
                settings=self.settings,
                batch_size=max(1, int(self.cosine_batch_size)),
                max_chars=max(200, int(self.cosine_max_chars)),
                purpose="graph_final_fusion",
                fallback_scores=scored,
            )
            if reranked:
                top_k = max(1, int(self.top_k))
                reranked_keys = self._dedup_keys([_doc_key(doc) for doc in reranked])
                graph_ranked_keys = [key for key in reranked_keys if key in graph_key_set]
                selected_keys = self._apply_min_graph_docs(
                    selected_keys=reranked_keys[:top_k],
                    graph_ranked_keys=graph_ranked_keys,
                    top_k=top_k,
                )
                return [by_key[key] for key in selected_keys if key in by_key]
            logger.warning("Cosine rerank failed in final graph fusion. Falling back to RRF.")

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        top_k = max(1, int(self.top_k))
        ranked_keys = [key for key, _ in ranked]
        graph_ranked_keys = [key for key in ranked_keys if key in graph_key_set]
        selected_keys = self._apply_min_graph_docs(
            selected_keys=ranked_keys[:top_k],
            graph_ranked_keys=graph_ranked_keys,
            top_k=top_k,
        )
        return [by_key[key] for key in selected_keys if key in by_key]


def _load_sparse_documents(store) -> List[Document]:
    payload = store.get(include=["documents", "metadatas"])
    docs = payload.get("documents", []) or []
    metas = payload.get("metadatas", []) or []

    sparse_docs: List[Document] = []
    for idx, text in enumerate(docs):
        content = str(text or "").strip()
        if not content:
            continue
        metadata = metas[idx] if idx < len(metas) and isinstance(metas[idx], dict) else {}
        sparse_docs.append(Document(page_content=content, metadata=metadata))
    return sparse_docs


def _build_base_retriever(settings: Settings) -> BaseRetriever:
    store = load_vector_store(settings)
    if store is None:
        raise RuntimeError("Vector store not initialized. Run ingest.py first.")

    # dense_retriever = store.as_retriever(search_kwargs={"k": max(1, int(settings.top_k))})
    dense_retriever = store.as_retriever(search_type='similarity', search_kwargs={"k": max(1, int(settings.top_k))})
    mode = str(settings.retriever_mode).strip().lower()
    if mode != "hybrid":
        logger.info("Retriever mode: dense (top_k=%s)", settings.top_k)
        return dense_retriever

    sparse_docs = _load_sparse_documents(store)
    if not sparse_docs:
        logger.warning("Hybrid mode requested, but sparse corpus is empty. Falling back to dense.")
        return dense_retriever

    sparse_retriever = BM25Retriever.from_documents(sparse_docs)
    sparse_retriever.k = max(1, int(settings.hybrid_sparse_k))

    dense_weight, sparse_weight = _normalize_weights(
        settings.hybrid_dense_weight,
        settings.hybrid_sparse_weight,
    )
    logger.info(
        "Retriever mode: hybrid (top_k=%s, sparse_k=%s, weights=%.2f/%.2f, fusion=%s)",
        settings.top_k,
        sparse_retriever.k,
        dense_weight,
        sparse_weight,
        str(settings.hybrid_fusion_mode),
    )
    return WeightedHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=max(1, int(settings.top_k)),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=max(1, int(settings.hybrid_rrf_k)),
        fusion_mode=str(settings.hybrid_fusion_mode),
        settings=settings,
        cosine_batch_size=max(1, int(settings.embed_batch_size)),
        cosine_max_chars=max(200, int(settings.graph_llm_input_max_chars)),
    )


def build_retriever(settings: Settings) -> BaseRetriever:
    base_retriever = _build_base_retriever(settings)
    graph_enabled = bool(settings.graph_retriever_enabled or settings.graph_rag_enabled)
    if not graph_enabled:
        return base_retriever
    if not settings.neo4j_enabled:
        logger.warning("Graph retriever requested, but NEO4J_ENABLED=0. Using base retriever.")
        return base_retriever
    if GraphDatabase is None:
        logger.warning("Graph retriever requested, but neo4j driver is unavailable. Using base retriever.")
        return base_retriever
    if not settings.neo4j_password:
        logger.warning("Graph retriever requested, but NEO4J_PASSWORD is empty. Using base retriever.")
        return base_retriever

    entity_retriever = Neo4jEntityRetriever(
        neo4j_uri=settings.neo4j_uri,
        neo4j_user=settings.neo4j_user,
        neo4j_password=settings.neo4j_password,
        neo4j_database=settings.neo4j_database,
        hops=max(1, int(settings.graph_retriever_hops)),
        entity_limit=max(1, int(settings.graph_retriever_entity_limit)),
        passage_limit=max(1, int(settings.graph_retriever_passage_limit)),
        query_candidate_limit=max(4, int(settings.graph_retriever_entity_limit)),
    )
    graph_retriever: BaseRetriever = entity_retriever
    if bool(settings.graph_keyword_channel_enabled):
        keyword_retriever = Neo4jKeywordRetriever(
            neo4j_uri=settings.neo4j_uri,
            neo4j_user=settings.neo4j_user,
            neo4j_password=settings.neo4j_password,
            neo4j_database=settings.neo4j_database,
            keyword_limit=max(1, int(settings.graph_keyword_query_limit)),
            passage_limit=max(1, int(settings.graph_retriever_passage_limit)),
            query_candidate_limit=max(4, int(settings.graph_retriever_entity_limit)),
            embedding_enabled=True,
            embedding_batch_size=max(1, int(settings.graph_keyword_embed_batch_size)),
            settings=settings,
        )
        graph_retriever = ThetaGraphRetriever(
            entity_retriever=entity_retriever,
            keyword_retriever=keyword_retriever,
            top_k=max(1, int(settings.graph_retriever_passage_limit)),
            theta=max(0.0, min(1.0, float(settings.graph_retrieval_theta))),
            rrf_k=max(1, int(settings.hybrid_rrf_k)),
            fusion_mode=str(settings.graph_channel_fusion_mode),
            settings=settings,
            cosine_batch_size=max(1, int(settings.embed_batch_size)),
            cosine_max_chars=max(200, int(settings.graph_llm_input_max_chars)),
        )

    graph_weight = max(0.0, float(settings.graph_retriever_weight))
    base_weight = max(0.0, 1.0 - graph_weight)
    logger.info(
        "Graph retriever enabled (hops=%s, entity_limit=%s, passage_limit=%s, keyword_channel=%s, theta=%.2f, weights=base:%.2f graph:%.2f, channel_fusion=%s, final_fusion=%s, min_graph_docs=%s)",
        max(1, int(settings.graph_retriever_hops)),
        max(1, int(settings.graph_retriever_entity_limit)),
        max(1, int(settings.graph_retriever_passage_limit)),
        bool(settings.graph_keyword_channel_enabled),
        max(0.0, min(1.0, float(settings.graph_retrieval_theta))),
        base_weight,
        graph_weight,
        str(settings.graph_channel_fusion_mode),
        str(settings.graph_final_fusion_mode),
        max(0, int(settings.graph_min_docs_in_final)),
    )
    return WeightedGraphFusionRetriever(
        base_retriever=base_retriever,
        graph_retriever=graph_retriever,
        top_k=max(1, int(settings.top_k)),
        base_weight=base_weight,
        graph_weight=graph_weight,
        rrf_k=max(1, int(settings.hybrid_rrf_k)),
        fusion_mode=str(settings.graph_final_fusion_mode),
        settings=settings,
        cosine_batch_size=max(1, int(settings.embed_batch_size)),
        cosine_max_chars=max(200, int(settings.graph_llm_input_max_chars)),
        min_graph_docs_in_final=max(0, int(settings.graph_min_docs_in_final)),
    )
