from __future__ import annotations

from typing import Dict, List, Tuple

from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from settings import Settings
from utils import get_logger
from vectorstore import load_vector_store

logger = get_logger("retriever")


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


def _doc_key(doc: Document) -> Tuple[str, str, str, str]:
    meta = doc.metadata or {}
    return (
        str(meta.get("source", "")),
        str(meta.get("page", "")),
        str(meta.get("chunk_id", "")),
        (doc.page_content or "").strip(),
    )


class WeightedHybridRetriever(BaseRetriever):
    dense_retriever: BaseRetriever
    sparse_retriever: BaseRetriever
    top_k: int = 4
    dense_weight: float = 0.6
    sparse_weight: float = 0.4
    rrf_k: int = 60

    def _get_relevant_documents(self, query: str, *, run_manager=None) -> List[Document]:
        dense_docs = _retrieve(self.dense_retriever, query)
        sparse_docs = _retrieve(self.sparse_retriever, query)

        dense_weight, sparse_weight = _normalize_weights(self.dense_weight, self.sparse_weight)
        scored: Dict[Tuple[str, str, str, str], float] = {}
        by_key: Dict[Tuple[str, str, str, str], Document] = {}

        for idx, doc in enumerate(dense_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + dense_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc
        for idx, doc in enumerate(sparse_docs):
            key = _doc_key(doc)
            scored[key] = scored.get(key, 0.0) + sparse_weight / (self.rrf_k + idx + 1)
            by_key[key] = doc

        ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
        return [by_key[key] for key, _ in ranked[: max(1, int(self.top_k))]]


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


def build_retriever(settings: Settings) -> BaseRetriever:
    store = load_vector_store(settings)
    if store is None:
        raise RuntimeError("Vector store not initialized. Run ingest.py first.")

    dense_retriever = store.as_retriever(search_kwargs={"k": max(1, int(settings.top_k))})
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
        "Retriever mode: hybrid (top_k=%s, sparse_k=%s, weights=%.2f/%.2f)",
        settings.top_k,
        sparse_retriever.k,
        dense_weight,
        sparse_weight,
    )
    return WeightedHybridRetriever(
        dense_retriever=dense_retriever,
        sparse_retriever=sparse_retriever,
        top_k=max(1, int(settings.top_k)),
        dense_weight=dense_weight,
        sparse_weight=sparse_weight,
        rrf_k=max(1, int(settings.hybrid_rrf_k)),
    )
