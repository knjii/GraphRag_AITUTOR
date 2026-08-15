"""Кросс-энкодерный реранкер.

Прежде реранкера не было вовсе: режим ``cosine`` переранжировал кандидатов тем же
bi-encoder'ом, который уже отработал в плотном поиске, то есть добавлял задержку,
почти не добавляя сигнала. Кросс-энкодер видит пару «запрос-документ» целиком
и обычно даёт больший прирост, чем любая настройка весов слияния.

Модель по умолчанию — ``BAAI/bge-reranker-v2-m3``: мультиязычная, знает русский.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import httpx

from rag_textbook.config import RerankerSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.utils.retry import retry_async, retry_sync

logger = get_logger("clients.reranker")


class RerankerClient(Protocol):
    def rerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]: ...

    async def arerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]: ...


class InfinityRerankerClient:
    """Реранкер через ``POST {base_url}/rerank`` (Infinity, TEI-совместимо)."""

    def __init__(self, settings: RerankerSettings) -> None:
        self.settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(timeout=self.settings.timeout_seconds)
        return self._client

    def _async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(timeout=self.settings.timeout_seconds)
        return self._aclient

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def _payload(self, query: str, documents: Sequence[str], top_n: int) -> dict:
        limit = self.settings.max_chars
        return {
            "model": self.settings.model,
            "query": query,
            "documents": [(doc or "")[:limit] for doc in documents],
            "top_n": max(1, int(top_n)),
            "return_documents": False,
        }

    @staticmethod
    def _parse(payload: dict) -> list[tuple[int, float]]:
        results = payload.get("results")
        if not isinstance(results, list):
            raise ValueError(f"Неожиданный ответ реранкера: {str(payload)[:200]}")
        pairs: list[tuple[int, float]] = []
        for item in results:
            index = item.get("index")
            score = item.get("relevance_score", item.get("score"))
            if index is None or score is None:
                continue
            pairs.append((int(index), float(score)))
        pairs.sort(key=lambda pair: pair[1], reverse=True)
        return pairs

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = self._payload(query, documents, top_n)

        def call() -> list[tuple[int, float]]:
            response = self._sync_client().post(f"{self._base_url}/rerank", json=payload)
            response.raise_for_status()
            return self._parse(response.json())

        try:
            return retry_sync(call, description="rerank", attempts=2)
        except Exception as exc:  # noqa: BLE001
            # Реранкер — улучшение, а не обязательный шаг: при сбое отдаём
            # исходный порядок, чтобы запрос пользователя не падал целиком.
            logger.warning("Реранкер недоступен, сохраняю исходный порядок: %s", exc)
            return [(index, 0.0) for index in range(min(len(documents), top_n))]

    async def arerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        if not documents:
            return []
        payload = self._payload(query, documents, top_n)

        async def call() -> list[tuple[int, float]]:
            response = await self._async_client().post(f"{self._base_url}/rerank", json=payload)
            response.raise_for_status()
            return self._parse(response.json())

        try:
            return await retry_async(call, description="rerank", attempts=2)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Реранкер недоступен, сохраняю исходный порядок: %s", exc)
            return [(index, 0.0) for index in range(min(len(documents), top_n))]


class FakeRerankerClient:
    """Заглушка: ранжирует по доле общих слов с запросом.

    Не претендует на качество, но детерминирована и позволяет тестировать
    логику конвейера без запущенного сервера.
    """

    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        query_terms = set(query.lower().split())
        scored: list[tuple[int, float]] = []
        for index, document in enumerate(documents):
            doc_terms = set((document or "").lower().split())
            overlap = len(query_terms & doc_terms)
            score = overlap / (len(query_terms) or 1)
            scored.append((index, float(score)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored[: max(1, int(top_n))]

    async def arerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        return self.rerank(query, documents, top_n)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


class NoopRerankerClient:
    def rerank(self, query: str, documents: Sequence[str], top_n: int) -> list[tuple[int, float]]:
        return [(index, 0.0) for index in range(min(len(documents), max(1, int(top_n))))]

    async def arerank(
        self, query: str, documents: Sequence[str], top_n: int
    ) -> list[tuple[int, float]]:
        return self.rerank(query, documents, top_n)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def build_reranker_client(settings: RerankerSettings) -> RerankerClient:
    if not settings.enabled or settings.provider == "none":
        return NoopRerankerClient()
    if settings.provider == "fake":
        return FakeRerankerClient()
    return InfinityRerankerClient(settings)
