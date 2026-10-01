"""Клиент эмбеддингов.

Что чинится по сравнению с прежней реализацией:

* эмбеддинги считались через Ollama и наблюдаемо давали ~0.73 с на чанк —
  здесь запросы идут батчами к OpenAI-совместимому серверу (Infinity/TEI/vLLM);
* один и тот же текст эмбеддился до четырёх раз (индекс, KET-отбор, keyword-канал,
  косинусный фьюжн) — теперь работает общий кэш по хешу текста и модели;
* Qwen3-Embedding обучена асимметрично, но префикс для запроса не применялся —
  теперь запрос и документ кодируются разными префиксами.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Protocol

import httpx

from rag_textbook.config import EmbeddingSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.utils.cache import ArtifactCache
from rag_textbook.utils.retry import retry_async, retry_sync

logger = get_logger("clients.embeddings")


class EmbeddingClient(Protocol):
    dimensions: int

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    def embed_query(self, text: str) -> list[float]: ...

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]: ...

    async def aembed_query(self, text: str) -> list[float]: ...


def _cache_key(model: str, prefix: str, text: str) -> str:
    digest = hashlib.sha256()
    digest.update(model.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(prefix.encode("utf-8"))
    digest.update(b"\x1f")
    digest.update(text.encode("utf-8", errors="ignore"))
    return digest.hexdigest()


class OpenAICompatibleEmbeddingClient:
    """Эмбеддинги через ``POST {base_url}/embeddings``."""

    def __init__(self, settings: EmbeddingSettings, cache: ArtifactCache | None = None) -> None:
        self.settings = settings
        self.dimensions = settings.dimensions
        self._cache = cache
        self._base_url = settings.base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None

    # ------------------------------------------------------------- соединения

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.timeout_seconds, headers=self._headers
            )
        return self._client

    def _async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            self._aclient = httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, headers=self._headers
            )
        return self._aclient

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    # ---------------------------------------------------------------- запросы

    def _prepare(self, texts: Sequence[str], prefix: str) -> list[str]:
        limit = self.settings.max_chars
        return [f"{prefix}{(text or '').strip()[:limit]}" for text in texts]

    @staticmethod
    def _parse(payload: dict) -> list[list[float]]:
        data = payload.get("data")
        if not isinstance(data, list):
            raise ValueError(f"Неожиданный ответ сервера эмбеддингов: {str(payload)[:200]}")
        # Сервер может вернуть элементы не по порядку — сортируем по index.
        ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
        return [[float(value) for value in item["embedding"]] for item in ordered]

    def _post_sync(self, batch: list[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            response = self._sync_client().post(
                f"{self._base_url}/embeddings",
                json={"model": self.settings.model, "input": batch},
            )
            response.raise_for_status()
            return self._parse(response.json())

        return retry_sync(call, description="embeddings", attempts=3)

    async def _post_async(self, batch: list[str]) -> list[list[float]]:
        async def call() -> list[list[float]]:
            response = await self._async_client().post(
                f"{self._base_url}/embeddings",
                json={"model": self.settings.model, "input": batch},
            )
            response.raise_for_status()
            return self._parse(response.json())

        return await retry_async(call, description="embeddings", attempts=3)

    # -------------------------------------------------------------- публичное

    def _embed(self, texts: Sequence[str], prefix: str, *, is_async: bool = False):
        """Общая часть: подстановка из кэша и разбиение на батчи."""
        prepared = self._prepare(texts, prefix)
        result: list[list[float] | None] = [None] * len(prepared)

        keys = [_cache_key(self.settings.model, prefix, text) for text in prepared]
        if self._cache is not None and self.settings.cache_enabled:
            cached = self._cache.get_many(keys)
            for idx, key in enumerate(keys):
                hit = cached.get(key)
                if isinstance(hit, list) and hit:
                    result[idx] = [float(value) for value in hit]

        pending = [idx for idx, value in enumerate(result) if value is None]
        return prepared, keys, result, pending

    def _store(self, keys, result, pending) -> None:
        if self._cache is None or not self.settings.cache_enabled or not pending:
            return
        self._cache.set_many({keys[idx]: result[idx] for idx in pending if result[idx]})

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared, keys, result, pending = self._embed(texts, self.settings.document_prefix)
        step = self.settings.batch_size
        for start in range(0, len(pending), step):
            indices = pending[start : start + step]
            vectors = self._post_sync([prepared[i] for i in indices])
            for position, idx in enumerate(indices):
                result[idx] = vectors[position]
            if start % (step * 10) == 0 or start + step >= len(pending):
                logger.debug("Эмбеддинги: %s/%s", min(start + step, len(pending)), len(pending))
        self._store(keys, result, pending)
        return [vector or [] for vector in result]

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        prepared, keys, result, pending = self._embed(texts, self.settings.document_prefix)
        step = self.settings.batch_size
        for start in range(0, len(pending), step):
            indices = pending[start : start + step]
            vectors = await self._post_async([prepared[i] for i in indices])
            for position, idx in enumerate(indices):
                result[idx] = vectors[position]
        self._store(keys, result, pending)
        return [vector or [] for vector in result]

    def embed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text], self.settings.query_prefix)
        return self._post_sync(prepared)[0]

    async def aembed_query(self, text: str) -> list[float]:
        prepared = self._prepare([text], self.settings.query_prefix)
        return (await self._post_async(prepared))[0]


class OllamaEmbeddingClient(OpenAICompatibleEmbeddingClient):
    """Совместимость с Ollama.

    Оставлена как запасной путь: у Ollama есть ``/v1/embeddings``, но он
    заметно медленнее и хуже батчится. Основной режим — Infinity.
    """

    def _post_sync(self, batch: list[str]) -> list[list[float]]:
        def call() -> list[list[float]]:
            response = self._sync_client().post(
                f"{self._base_url}/embeddings",
                json={"model": self.settings.model, "input": batch},
            )
            response.raise_for_status()
            return self._parse(response.json())

        return retry_sync(call, description="ollama-embeddings", attempts=3)


class FakeEmbeddingClient:
    """Детерминированные эмбеддинги для тестов без запущенных сервисов.

    Вектор строится из хеша текста, поэтому одинаковый текст всегда даёт
    одинаковый вектор, а разные тексты — разные. Этого достаточно, чтобы
    проверять логику индексации, слияния и метрик.
    """

    def __init__(self, dimensions: int = 64) -> None:
        self.dimensions = dimensions

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256((text or "").encode("utf-8", errors="ignore")).digest()
        raw = [digest[i % len(digest)] / 255.0 for i in range(self.dimensions)]
        norm = sum(value * value for value in raw) ** 0.5 or 1.0
        return [value / norm for value in raw]

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    async def aembed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        return self.embed_documents(texts)

    async def aembed_query(self, text: str) -> list[float]:
        return self.embed_query(text)

    def close(self) -> None:  # совместимость по интерфейсу
        return None

    async def aclose(self) -> None:
        return None


def build_embedding_client(
    settings: EmbeddingSettings, cache: ArtifactCache | None = None
) -> EmbeddingClient:
    if settings.provider == "fake":
        return FakeEmbeddingClient(dimensions=settings.dimensions)
    if settings.provider == "ollama":
        return OllamaEmbeddingClient(settings, cache)
    return OpenAICompatibleEmbeddingClient(settings, cache)
