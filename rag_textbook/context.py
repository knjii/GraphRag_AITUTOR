"""Сборка компонентов приложения в одном месте.

Раньше каждый скрипт собирал зависимости сам: ``query.py`` создавал цепочку,
Streamlit — свою, скрипт бэкфилла — третью. Из-за этого настройки расходились,
а ресурсы вроде драйвера Neo4j создавались на каждый запрос.

Здесь ресурсы создаются один раз и переиспользуются. Для сервиса это критично:
именно повторное создание BM25-индекса и драйверов делало прежнюю систему
непригодной для нескольких одновременных пользователей.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
from typing import Any

from rag_textbook.clients.embeddings import EmbeddingClient, build_embedding_client
from rag_textbook.clients.llm import LLMClient, build_llm_client
from rag_textbook.clients.reranker import RerankerClient, build_reranker_client
from rag_textbook.config import Settings
from rag_textbook.generation.answering import AnswerGenerator
from rag_textbook.generation.history import ChatHistoryStore
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.logging_setup import configure_logging, get_logger
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.retrieval.pipeline import RetrievalPipeline
from rag_textbook.stores.graph_store import GraphStore
from rag_textbook.stores.vector_store import VectorStore, build_vector_store
from rag_textbook.utils.cache import ArtifactCache

logger = get_logger("context")


@dataclass
class AppContext:
    """Собранное приложение."""

    settings: Settings
    embeddings: EmbeddingClient
    llm: LLMClient
    reranker: RerankerClient
    vector_store: VectorStore
    graph_store: GraphStore | None
    graph_retriever: GraphRetriever | None
    retrieval: RetrievalPipeline
    generator: AnswerGenerator
    history: ChatHistoryStore
    embedding_cache: ArtifactCache
    enrichment_cache: ArtifactCache
    extraction_cache: ArtifactCache

    def entity_extractor(self) -> EntityExtractor:
        return EntityExtractor(
            self.settings.graph,
            llm=self.llm,
            cache=self.extraction_cache,
        )

    def health(self) -> dict[str, Any]:
        """Состояние зависимостей.

        Нужно и для ``/health`` сервиса, и для быстрой проверки после подъёма
        арендованного сервера: одна команда вместо угадывания, что не поднялось.
        """
        report: dict[str, Any] = {"status": "ok", "components": {}}

        try:
            count = self.vector_store.count()
            report["components"]["vector_store"] = {"status": "ok", "chunks": count}
        except Exception as exc:  # noqa: BLE001
            message = str(exc)
            # До первой индексации коллекции ещё нет, и Qdrant отвечает 404.
            # Это не поломка: помечать её ошибкой значит приучать не доверять
            # проверке связности — а тогда она перестаёт быть полезной.
            if "404" in message or "Not Found" in message or "doesn't exist" in message:
                report["components"]["vector_store"] = {
                    "status": "empty",
                    "detail": "коллекция ещё не создана — выполните ingest",
                }
            else:
                report["components"]["vector_store"] = {
                    "status": "error",
                    "error": message[:200],
                }
                report["status"] = "degraded"

        if self.graph_store is not None:
            try:
                report["components"]["graph"] = {"status": "ok", **self.graph_store.stats()}
            except Exception as exc:  # noqa: BLE001
                report["components"]["graph"] = {"status": "error", "error": str(exc)[:200]}
                report["status"] = "degraded"
        else:
            report["components"]["graph"] = {"status": "disabled"}

        try:
            vector = self.embeddings.embed_query("проверка доступности")
            report["components"]["embeddings"] = {"status": "ok", "dimensions": len(vector)}
        except Exception as exc:  # noqa: BLE001
            report["components"]["embeddings"] = {"status": "error", "error": str(exc)[:200]}
            report["status"] = "error"

        return report

    def close(self) -> None:
        for resource in (self.embeddings, self.llm, self.reranker):
            close = getattr(resource, "close", None)
            if callable(close):
                # Закрытие ресурсов не должно мешать закрытию остальных.
                with contextlib.suppress(Exception):
                    close()
        if self.graph_store is not None:
            self.graph_store.close()
        self.history.close()
        for cache in (self.embedding_cache, self.enrichment_cache, self.extraction_cache):
            cache.close()


def build_context(settings: Settings | None = None) -> AppContext:
    settings = settings or Settings()
    configure_logging(settings.log_level, settings.log_json)
    settings.paths.ensure()

    cache_dir = settings.paths.cache_dir
    embedding_cache = ArtifactCache(
        cache_dir / "embeddings.sqlite3", "embeddings", settings.embedding.cache_enabled
    )
    enrichment_cache = ArtifactCache(
        cache_dir / "enrichment.sqlite3", "enrichment", settings.chunking.enrich_cache_enabled
    )
    extraction_cache = ArtifactCache(
        cache_dir / "extraction.sqlite3", "extraction", settings.graph.extraction_cache_enabled
    )

    embeddings = build_embedding_client(settings.embedding, embedding_cache)
    llm = build_llm_client(settings.llm)
    reranker = build_reranker_client(settings.reranker)
    vector_store = build_vector_store(settings.vector_store)

    graph_store: GraphStore | None = None
    graph_retriever: GraphRetriever | None = None
    if settings.graph.enabled:
        graph_store = GraphStore(settings.graph)
        if settings.graph.retrieval_enabled:
            graph_retriever = GraphRetriever(settings.graph, graph_store)

    retrieval = RetrievalPipeline(
        settings=settings,
        vector_store=vector_store,
        embedding_client=embeddings,
        reranker=reranker,
        graph_retriever=graph_retriever,
        llm=llm,
    )
    generator = AnswerGenerator(settings, retrieval, llm)
    history = ChatHistoryStore(settings.service.history_db_path, settings.service.history_enabled)

    logger.info(
        "Контекст собран: эмбеддинги=%s, векторное хранилище=%s, граф=%s, реранкер=%s",
        settings.embedding.provider,
        settings.vector_store.provider,
        "включён" if settings.graph.enabled else "выключен",
        settings.reranker.provider if settings.reranker.enabled else "выключен",
    )

    return AppContext(
        settings=settings,
        embeddings=embeddings,
        llm=llm,
        reranker=reranker,
        vector_store=vector_store,
        graph_store=graph_store,
        graph_retriever=graph_retriever,
        retrieval=retrieval,
        generator=generator,
        history=history,
        embedding_cache=embedding_cache,
        enrichment_cache=enrichment_cache,
        extraction_cache=extraction_cache,
    )
