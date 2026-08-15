"""Полный конвейер поиска.

Порядок шагов: переписывание вопроса по истории → роутинг → базовый гибридный
канал (и графовый, если роутер сказал «да») → слияние → дедупликация →
кросс-энкодерный реранкинг → отбор top-k.

Явный конвейер вместо ``create_retrieval_chain`` выбран потому, что legacy-цепочка
не оставляла места ни реранкеру, ни роутеру, ни измерению стадий по отдельности,
и вдобавок скрывала отсутствие переписывания вопроса — из-за чего диалоговый режим
искал буквально по тексту последней реплики.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_textbook.clients.embeddings import EmbeddingClient
from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.clients.reranker import RerankerClient
from rag_textbook.config import Settings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import ScoredChunk
from rag_textbook.retrieval.fusion import (
    deduplicate,
    enforce_minimum_graph_documents,
    reciprocal_rank_fusion,
)
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.retrieval.router import QueryRouter, RouteDecision
from rag_textbook.stores.vector_store import VectorStore

logger = get_logger("retrieval.pipeline")


@dataclass
class RetrievalResult:
    question: str
    rewritten_question: str
    chunks: list[ScoredChunk] = field(default_factory=list)
    route: RouteDecision | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)
    channel_sizes: dict[str, int] = field(default_factory=dict)

    @property
    def graph_share(self) -> float:
        """Доля графовых документов в финальном контексте.

        Отдельная метрика, потому что «граф нашёл» и «граф попал в контекст» —
        разные события, и раньше их невозможно было различить.
        """
        if not self.chunks:
            return 0.0
        return sum(1 for item in self.chunks if item.from_graph) / len(self.chunks)


class RetrievalPipeline:
    def __init__(
        self,
        settings: Settings,
        vector_store: VectorStore,
        embedding_client: EmbeddingClient,
        reranker: RerankerClient,
        graph_retriever: GraphRetriever | None = None,
        llm: LLMClient | None = None,
    ) -> None:
        self.settings = settings
        self.vector_store = vector_store
        self.embeddings = embedding_client
        self.reranker = reranker
        self.graph_retriever = graph_retriever
        self.llm = llm
        self.router = QueryRouter(settings.retrieval, llm)

    # ------------------------------------------------------- переписывание

    def rewrite_question(self, question: str, history: Sequence[ChatMessage]) -> str:
        """Делает вопрос самодостаточным.

        Без этого шага реплика «а как это применить на практике?» уходила в поиск
        буквально: местоимение «это» не несёт информации, и релевантных чанков нет.
        """
        if not self.settings.retrieval.query_rewrite_enabled or not history or self.llm is None:
            return question

        turns = list(history)[-self.settings.retrieval.max_history_turns * 2 :]
        transcript = "\n".join(
            f"{'Студент' if message.role == 'user' else 'Ассистент'}: {message.content[:500]}"
            for message in turns
        )
        prompt = (
            f"{self.settings.prompts.query_rewrite_system}\n\n"
            f"История диалога:\n{transcript}\n\n"
            f"Текущий вопрос: {question}\n\n"
            "Самодостаточный вопрос:"
        )
        try:
            rewritten = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="chat",
                max_tokens=160,
                temperature=0.0,
            ).strip()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Переписывание вопроса не удалось: %s", exc)
            return question

        if not rewritten or len(rewritten) > len(question) * 6:
            return question
        return rewritten

    # ------------------------------------------------------------- каналы

    def _base_channel(self, query: str) -> list[ScoredChunk]:
        vector = self.embeddings.embed_query(query)
        limit = max(
            self.settings.reranker.candidates if self.settings.reranker.enabled else 0,
            self.settings.retrieval.top_k,
        )
        return self.vector_store.search(
            query_text=query,
            query_vector=vector,
            limit=limit,
            settings=self.settings.retrieval,
        )

    def _graph_channel(self, query: str) -> list[ScoredChunk]:
        if self.graph_retriever is None:
            return []
        return self.graph_retriever.retrieve(query)

    # ---------------------------------------------------------- реранкинг

    def _rerank(self, query: str, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not self.settings.reranker.enabled or not items:
            return items
        documents = [item.chunk.text for item in items]
        top_n = min(len(items), max(self.settings.reranker.top_n, self.settings.retrieval.top_k))
        pairs = self.reranker.rerank(query, documents, top_n)
        reranked: list[ScoredChunk] = []
        for index, score in pairs:
            if 0 <= index < len(items):
                item = items[index]
                item.rerank_score = float(score)
                reranked.append(item)
        return reranked or items

    # ---------------------------------------------------------------- run

    def retrieve(
        self, question: str, history: Sequence[ChatMessage] | None = None
    ) -> RetrievalResult:
        timings: dict[str, float] = {}
        started = time.perf_counter()

        stage = time.perf_counter()
        rewritten = self.rewrite_question(question, history or [])
        timings["rewrite"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        route = self.router.route(rewritten)
        timings["route"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        base_items = self._base_channel(rewritten)
        timings["base_channel"] = (time.perf_counter() - stage) * 1000

        graph_items: list[ScoredChunk] = []
        if route.use_graph and self.graph_retriever is not None:
            stage = time.perf_counter()
            graph_items = self._graph_channel(rewritten)
            timings["graph_channel"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        graph_weight = self.settings.graph.weight if graph_items else 0.0
        merged = reciprocal_rank_fusion(
            {"base": base_items, "graph": graph_items},
            weights={"base": 1.0 - graph_weight, "graph": graph_weight},
            rrf_k=self.settings.retrieval.rrf_k,
        )
        if self.settings.retrieval.dedup_enabled:
            merged = deduplicate(merged, self.settings.retrieval.dedup_similarity)
        timings["fusion"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        candidates = merged[: self.settings.reranker.candidates]
        reranked = self._rerank(rewritten, candidates)
        timings["rerank"] = (time.perf_counter() - stage) * 1000

        final = enforce_minimum_graph_documents(
            reranked,
            minimum=self.settings.retrieval.min_graph_docs,
            top_k=self.settings.retrieval.top_k,
        )

        timings["total"] = (time.perf_counter() - started) * 1000
        result = RetrievalResult(
            question=question,
            rewritten_question=rewritten,
            chunks=final,
            route=route,
            timings_ms={key: round(value, 1) for key, value in timings.items()},
            channel_sizes={
                "base": len(base_items),
                "graph": len(graph_items),
                "merged": len(merged),
                "final": len(final),
            },
        )
        logger.debug(
            "Поиск: граф=%s (%s), база=%s, финал=%s, доля графа=%.2f, %.0f мс",
            route.use_graph,
            route.reason,
            len(base_items),
            len(final),
            result.graph_share,
            timings["total"],
        )
        return result
