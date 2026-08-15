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

import json
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
    # Подвопросы, на которые был разложен связывающий вопрос. Пусто, если
    # разложение выключено или вопрос делению не поддался.
    sub_questions: list[str] = field(default_factory=list)

    @property
    def graph_share(self) -> float:
        """Доля фрагментов контекста, которые нашёл графовый канал.

        Считает и те, что нашёл заодно векторный канал, поэтому величина
        завышает вклад графа. Для оценки вклада смотрите ``graph_only_share``.
        """
        if not self.chunks:
            return 0.0
        return sum(1 for item in self.chunks if item.from_graph) / len(self.chunks)

    @property
    def graph_only_share(self) -> float:
        """Доля фрагментов, которых без графа в контексте не было бы.

        Это и есть вклад канала. Разница между двумя величинами оказалась
        решающей: ``graph_share`` показывал 16-23%, тогда как ``graph_only_share``
        равнялся нулю, и A/B давал нулевую разницу по всем метрикам.
        """
        if not self.chunks:
            return 0.0
        return sum(1 for item in self.chunks if item.only_from_graph) / len(self.chunks)


DECOMPOSITION_SCHEMA = {
    "type": "object",
    "properties": {"parts": {"type": "array", "items": {"type": "string"}}},
    "required": ["parts"],
}


def _strip_code_fence(raw: str) -> str:
    """Снимает обрамление ```json ... ```, если модель его добавила."""
    text = str(raw or "").strip()
    if not text.startswith("```"):
        return text
    lines = text.splitlines()
    if len(lines) < 2:
        return text
    body = lines[1:]
    if body and body[-1].strip().startswith("```"):
        body = body[:-1]
    return "\n".join(body).strip()


def _merge_preserving_order(
    primary: list[ScoredChunk], extra: Sequence[ScoredChunk]
) -> list[ScoredChunk]:
    """Дописывает новые фрагменты в конец, не меняя порядок уже найденных.

    Порядок основной выдачи сохраняется намеренно: подвопросы дополняют её,
    а не переопределяют. Итоговое ранжирование всё равно делает реранкер.
    """
    seen = {item.chunk.id for item in primary}
    result = list(primary)
    for item in extra:
        if item.chunk.id not in seen:
            seen.add(item.chunk.id)
            result.append(item)
    return result


def _reserve_graph_candidates(
    merged: Sequence[ScoredChunk], *, limit: int, quota: int
) -> list[ScoredChunk]:
    """Отрезает пул кандидатов, сохранив место за находками одного лишь графа.

    Без резерва отбор кандидатов устроен так, что вклад графа теряется ещё
    до реранкера: ранговое слияние ставит графовые фрагменты ниже плотной
    векторной выдачи, а до реранкера доезжают только первые ``limit``.
    Замерено: графовый канал находит 10-12 процентных пунктов эталонного
    материала, которого нет в векторной выдаче, а доля таких фрагментов
    в итоговом контексте равна нулю.

    Резерв ничего не навязывает ответу — он доводит фрагменты до реранкера,
    а решает по-прежнему реранкер.
    """
    head = list(merged[:limit])
    if quota <= 0:
        return head

    present = {item.chunk.id for item in head}
    graph_only = [
        item for item in head if item.only_from_graph
    ]
    missing = quota - len(graph_only)
    if missing <= 0:
        return head

    additions: list[ScoredChunk] = []
    for item in merged[limit:]:
        if len(additions) >= missing:
            break
        if item.chunk.id in present or not item.only_from_graph:
            continue
        present.add(item.chunk.id)
        additions.append(item)
    if not additions:
        return head

    # Вытесняем хвост векторной выдачи, а не начало: у нижних кандидатов
    # релевантность и так низкая, и реранкер их всё равно бы отбросил.
    keep = len(head) - len(additions)
    return head[:keep] + additions


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

    # --------------------------------------------------------- разложение

    def decompose_question(self, question: str) -> list[str]:
        """Разбивает связывающий вопрос на части, каждая со своим ответом.

        Возвращает пустой список, если разложение выключено, недоступно или
        не удалось: тогда работает обычный путь. Молчаливая деградация здесь
        уместна — разложение это улучшение, а не условие работоспособности.
        """
        if not self.settings.retrieval.decompose_enabled or self.llm is None:
            return []

        limit = self.settings.retrieval.decompose_max_parts
        prompt = (
            f"Раздели вопрос на {limit} самостоятельных подвопроса, ответы на которые "
            "вместе дают ответ на исходный.\n"
            "Требования:\n"
            "- каждый подвопрос должен быть понятен сам по себе, без исходного вопроса;\n"
            "- подвопросы не должны повторять друг друга;\n"
            "- если вопрос простой и делению не поддаётся, верни его без изменений "
            "одним элементом.\n\n"
            'Верни строго JSON: {"parts": ["...", "..."]}\n\n'
            f"Вопрос: {question}"
        )
        try:
            raw = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="utility",
                json_schema=DECOMPOSITION_SCHEMA,
                max_tokens=300,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Разложение вопроса не удалось: %s", exc)
            return []

        try:
            payload = json.loads(_strip_code_fence(raw))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Разложение вернуло неразбираемый ответ: %s", str(raw)[:120])
            return []

        parts = [
            str(item).strip()
            for item in (payload.get("parts") or [])
            if str(item or "").strip()
        ]
        # Разложение из одной части — это исходный вопрос, работать по общему пути.
        if len(parts) < 2:
            return []
        return parts[:limit]

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
                purpose="utility",
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

    def _base_channel(self, query: str, top_k: int | None = None) -> list[ScoredChunk]:
        vector = self.embeddings.embed_query(query)
        limit = max(
            self.settings.reranker.candidates if self.settings.reranker.enabled else 0,
            top_k or self.settings.retrieval.top_k,
        )
        return self.vector_store.search(
            query_text=query,
            query_vector=vector,
            limit=limit,
            settings=self.settings.retrieval,
        )

    def _graph_channel(self, query: str, base_items: Sequence[ScoredChunk]) -> list[ScoredChunk]:
        """Графовый канал.

        Опорные фрагменты передаются всегда: в режиме ``query`` они не
        используются, а в режимах ``passages`` и ``both`` обход начинается
        именно от них. Порядок вызова из-за этого фиксирован — графовый канал
        идёт после векторного, а не параллельно ему.
        """
        if self.graph_retriever is None:
            return []
        seed_ids = [item.chunk.id for item in base_items[: self.settings.graph.seed_passages]]
        return self.graph_retriever.retrieve(query, seed_chunk_ids=seed_ids)

    # ---------------------------------------------------------- реранкинг

    def _rerank(
        self, query: str, items: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        if not self.settings.reranker.enabled or not items:
            return items
        documents = [item.chunk.text for item in items]
        top_n = min(
            len(items),
            max(self.settings.reranker.top_n, top_k or self.settings.retrieval.top_k),
        )
        pairs = self.reranker.rerank(query, documents, top_n)
        reranked: list[ScoredChunk] = []
        for index, score in pairs:
            if 0 <= index < len(items):
                item = items[index]
                item.rerank_score = float(score)
                reranked.append(item)
        return reranked or items

    def _rerank_by_parts(
        self, parts: Sequence[str], items: list[ScoredChunk], top_k: int | None = None
    ) -> list[ScoredChunk]:
        """Ранжирует кандидатов отдельно под каждый подвопрос и объединяет.

        Смысл именно здесь. Реранкер оценивает пару «вопрос — фрагмент»
        целиком, поэтому на связывающем вопросе оба нужных фрагмента получают
        средний балл: каждый отвечает лишь на половину. Замер: из 118
        эталонных фрагментов многошаговых вопросов в пул кандидатов попадают
        103, а в финальную выдачу — 68. Ранжирование под каждую половину
        отдельно снимает именно эту причину.

        Объединение — поочерёдным отбором лучших от каждого подвопроса, а не
        слиянием рангов. Взаимно-ранговое слияние здесь вредит и это измерено:
        оно складывает ранги по всем подвопросам, поэтому фрагмент, идеально
        отвечающий на первую половину и не имеющий отношения ко второй,
        оказывается в середине списка. Но именно такой фрагмент и нужен —
        по построению связывающего вопроса каждый фрагмент отвечает лишь
        на одну его часть. Первая версия объединяла слиянием и дала
        MRR −0.031 при значимом интервале.

        Поочерёдный отбор гарантирует, что лучший фрагмент каждого подвопроса
        попадёт в выдачу независимо от его оценки по остальным.
        """
        if not self.settings.reranker.enabled or not items:
            return items

        rankings = [self._rerank(part, list(items)) for part in parts]
        rankings = [ranked for ranked in rankings if ranked]
        if not rankings:
            return items

        top_n = max(self.settings.reranker.top_n, top_k or self.settings.retrieval.top_k)
        result: list[ScoredChunk] = []
        seen: set[str] = set()
        for position in range(max(len(ranked) for ranked in rankings)):
            for ranked in rankings:
                if position >= len(ranked):
                    continue
                item = ranked[position]
                if item.chunk.id in seen:
                    continue
                seen.add(item.chunk.id)
                result.append(item)
                if len(result) >= top_n:
                    return result
        return result

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

        # Связывающему вопросу нужно два эталонных фрагмента вместо одного,
        # поэтому та же квота вмещает вдвое меньше ответов. Квота расширяется
        # только здесь: на простых вопросах recall насыщается к двенадцати
        # фрагментам, и лишний контекст стоил бы токенов впустую.
        top_k = self.settings.retrieval.top_k_for(route.use_graph)

        # Разложение имеет смысл только там, где вопрос признан связывающим:
        # на простых вопросах оно тратит вызов модели впустую.
        parts: list[str] = []
        if route.use_graph:
            stage = time.perf_counter()
            parts = self.decompose_question(rewritten)
            timings["decompose"] = (time.perf_counter() - stage) * 1000

        stage = time.perf_counter()
        base_items = self._base_channel(rewritten, top_k)
        # Каждый подвопрос ищется отдельно: фрагмент, отвечающий на вторую
        # половину вопроса, по формулировке целиком в выдачу может не попасть.
        for part in parts:
            base_items = _merge_preserving_order(base_items, self._base_channel(part, top_k))
        timings["base_channel"] = (time.perf_counter() - stage) * 1000

        graph_items: list[ScoredChunk] = []
        if route.use_graph and self.graph_retriever is not None:
            stage = time.perf_counter()
            graph_items = self._graph_channel(rewritten, base_items)
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
        candidates = _reserve_graph_candidates(
            merged,
            limit=self.settings.reranker.candidates,
            quota=self.settings.retrieval.graph_candidate_quota,
        )
        if parts:
            reranked = self._rerank_by_parts(parts, candidates, top_k)
        else:
            reranked = self._rerank(rewritten, candidates, top_k)
        timings["rerank"] = (time.perf_counter() - stage) * 1000

        final = enforce_minimum_graph_documents(
            reranked,
            minimum=self.settings.retrieval.min_graph_docs,
            top_k=top_k,
        )

        timings["total"] = (time.perf_counter() - started) * 1000
        result = RetrievalResult(
            question=question,
            rewritten_question=rewritten,
            chunks=final,
            route=route,
            timings_ms={key: round(value, 1) for key, value in timings.items()},
            sub_questions=list(parts),
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
