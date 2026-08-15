"""Векторное хранилище.

Замена Chroma на Qdrant решает сразу четыре проблемы прежней реализации:

1. Chroma — файл SQLite; при конкурентном доступе он ломается, из-за чего в коде
   появился обработчик ``disk I/O error`` с удалением журнальных файлов.
2. Коллекция открывалась без указания метрики, то есть работал L2 по ненормированным
   эмбеддингам, хотя модель обучена под косинус.
3. Лексический канал строился как ``BM25Retriever`` в памяти процесса: весь корпус
   вычитывался из базы при каждом создании цепочки. Теперь BM25 живёт в Qdrant
   как sparse-вектор, со стеммингом и стоп-словами русского языка.
4. Документы добавлялись без явных идентификаторов, поэтому повторная индексация
   плодила дубликаты. Идентификатор чанка детерминирован.

``InMemoryVectorStore`` повторяет тот же контракт и позволяет прогонять тесты
и локальную отладку вообще без запущенных сервисов.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable, Sequence
from typing import Any, Protocol

from rag_textbook.config import RetrievalSettings, VectorStoreSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.utils.text import content_terms

logger = get_logger("stores.vector")

DENSE_VECTOR = "dense"
SPARSE_VECTOR = "bm25"


class VectorStore(Protocol):
    def ensure_collection(self, dimensions: int) -> None: ...

    def upsert(self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]]) -> int: ...

    def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
        settings: RetrievalSettings,
    ) -> list[ScoredChunk]: ...

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]: ...

    def iter_chunks(self, batch_size: int = 256) -> Iterable[Chunk]: ...

    def count(self) -> int: ...

    def delete_document(self, doc_id: str) -> int: ...


def _point_id(chunk_id: str) -> str:
    """Qdrant принимает UUID или целое; наш идентификатор — строка вида ``doc:00042``.

    Детерминированный UUID5 сохраняет идемпотентность: повторная запись того же
    чанка обновляет точку, а не создаёт новую.
    """
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantVectorStore:
    def __init__(self, settings: VectorStoreSettings) -> None:
        self.settings = settings
        self._client: Any = None
        self._sparse_model: Any = None
        self._sparse_ready = False

    # -------------------------------------------------------------- соединение

    @property
    def client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient

            api_key = self.settings.api_key.get_secret_value() if self.settings.api_key else None
            self._client = QdrantClient(
                url=self.settings.url,
                api_key=api_key,
                timeout=int(self.settings.timeout_seconds),
            )
        return self._client

    def _sparse_encoder(self) -> Any | None:
        """BM25 со стеммингом нужного языка.

        Именно это чинит слабость лексического канала на русском: прежний
        ``BM25Retriever`` резал текст по пробелам и не знал ни морфологии,
        ни стоп-слов.
        """
        if not self.settings.sparse_enabled:
            return None
        if self._sparse_ready:
            return self._sparse_model
        self._sparse_ready = True
        try:
            from fastembed import SparseTextEmbedding

            try:
                self._sparse_model = SparseTextEmbedding(
                    model_name=self.settings.sparse_model,
                    language=self.settings.sparse_language,
                )
            except TypeError:
                # Старые версии fastembed не принимают language.
                logger.warning(
                    "fastembed не принял параметр language=%s, использую модель по умолчанию",
                    self.settings.sparse_language,
                )
                self._sparse_model = SparseTextEmbedding(model_name=self.settings.sparse_model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Разреженный канал недоступен (%s); работаю только плотным", exc)
            self._sparse_model = None
        return self._sparse_model

    # ------------------------------------------------------------------- схема

    def ensure_collection(self, dimensions: int) -> None:
        from qdrant_client import models

        exists = self.client.collection_exists(self.settings.collection)
        if exists:
            return

        sparse_config = None
        if self._sparse_encoder() is not None:
            sparse_config = {
                SPARSE_VECTOR: models.SparseVectorParams(
                    modifier=models.Modifier.IDF  # IDF считает сервер, как и положено BM25
                )
            }

        self.client.create_collection(
            collection_name=self.settings.collection,
            vectors_config={
                DENSE_VECTOR: models.VectorParams(
                    size=int(dimensions),
                    # Косинус задан явно: под него обучена модель эмбеддингов.
                    distance=models.Distance.COSINE,
                    hnsw_config=models.HnswConfigDiff(
                        m=self.settings.hnsw_m,
                        ef_construct=self.settings.hnsw_ef_construct,
                    ),
                )
            },
            sparse_vectors_config=sparse_config,
        )
        # Индексы по полям, по которым реально фильтруем.
        for field_name, schema in (
            ("doc_id", "keyword"),
            ("chunk_id", "keyword"),
            ("has_formula", "bool"),
            ("has_table", "bool"),
        ):
            try:
                self.client.create_payload_index(
                    collection_name=self.settings.collection,
                    field_name=field_name,
                    field_schema=schema,
                )
            except Exception as exc:  # noqa: BLE001
                logger.debug("Индекс по полю %s не создан: %s", field_name, exc)
        logger.info(
            "Коллекция %s создана (dim=%s, sparse=%s)",
            self.settings.collection,
            dimensions,
            sparse_config is not None,
        )

    # ------------------------------------------------------------------ запись

    def upsert(self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]]) -> int:
        if not chunks:
            return 0
        if len(chunks) != len(dense_vectors):
            raise ValueError("Число чанков и векторов должно совпадать")

        from qdrant_client import models

        encoder = self._sparse_encoder()
        sparse_vectors: list[Any] = []
        if encoder is not None:
            sparse_vectors = list(encoder.embed([chunk.text for chunk in chunks]))

        points: list[Any] = []
        for index, chunk in enumerate(chunks):
            vector: dict[str, Any] = {DENSE_VECTOR: list(dense_vectors[index])}
            if encoder is not None and index < len(sparse_vectors):
                sparse = sparse_vectors[index]
                vector[SPARSE_VECTOR] = models.SparseVector(
                    indices=list(sparse.indices), values=list(sparse.values)
                )
            points.append(
                models.PointStruct(id=_point_id(chunk.id), vector=vector, payload=chunk.payload())
            )

        written = 0
        step = self.settings.upsert_batch_size
        for start in range(0, len(points), step):
            batch = points[start : start + step]
            self.client.upsert(collection_name=self.settings.collection, points=batch, wait=True)
            written += len(batch)
        return written

    # ------------------------------------------------------------------ чтение

    def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
        settings: RetrievalSettings,
    ) -> list[ScoredChunk]:
        from qdrant_client import models

        encoder = self._sparse_encoder()
        prefetch: list[Any] = [
            models.Prefetch(
                query=list(query_vector),
                using=DENSE_VECTOR,
                limit=settings.dense_candidates,
            )
        ]
        if encoder is not None:
            sparse = next(iter(encoder.embed([query_text])), None)
            if sparse is not None:
                prefetch.append(
                    models.Prefetch(
                        query=models.SparseVector(
                            indices=list(sparse.indices), values=list(sparse.values)
                        ),
                        using=SPARSE_VECTOR,
                        limit=settings.sparse_candidates,
                    )
                )

        fusion = models.Fusion.DBSF if settings.fusion == "dbsf" else models.Fusion.RRF
        response = self.client.query_points(
            collection_name=self.settings.collection,
            prefetch=prefetch,
            # Слияние плотного и разреженного каналов выполняет сервер:
            # раньше это делалось в процессе приложения поверх полного корпуса в RAM.
            query=models.FusionQuery(fusion=fusion),
            limit=int(limit),
            with_payload=True,
        )

        results: list[ScoredChunk] = []
        for point in response.points:
            payload = dict(point.payload or {})
            chunk = Chunk.from_payload(payload)
            channel = "hybrid" if encoder is not None else "dense"
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(point.score or 0.0),
                    channels=[channel],
                    channel_scores={channel: float(point.score or 0.0)},
                )
            )
        return results

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        if not chunk_ids:
            return {}
        records = self.client.retrieve(
            collection_name=self.settings.collection,
            ids=[_point_id(chunk_id) for chunk_id in chunk_ids],
            with_payload=True,
        )
        out: dict[str, Chunk] = {}
        for record in records:
            chunk = Chunk.from_payload(dict(record.payload or {}))
            if chunk.id:
                out[chunk.id] = chunk
        return out

    def iter_chunks(self, batch_size: int = 256) -> Iterable[Chunk]:
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.settings.collection,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for point in points:
                yield Chunk.from_payload(dict(point.payload or {}))
            if offset is None:
                break

    def count(self) -> int:
        return int(self.client.count(self.settings.collection, exact=True).count)

    def delete_document(self, doc_id: str) -> int:
        from qdrant_client import models

        self.client.delete(
            collection_name=self.settings.collection,
            points_selector=models.FilterSelector(
                filter=models.Filter(
                    must=[
                        models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id))
                    ]
                )
            ),
            wait=True,
        )
        return 1


class InMemoryVectorStore:
    """Реализация того же контракта в памяти.

    Нужна, чтобы весь конвейер — индексация, поиск, слияние, метрики — можно было
    прогнать в тестах и на ноутбуке без Qdrant. Плотный канал считается косинусом,
    разреженный — упрощённым BM25 по леммам.
    """

    def __init__(self) -> None:
        self._chunks: dict[str, Chunk] = {}
        self._vectors: dict[str, list[float]] = {}
        self._terms: dict[str, list[str]] = {}
        self._dimensions = 0

    def ensure_collection(self, dimensions: int) -> None:
        self._dimensions = int(dimensions)

    def upsert(self, chunks: Sequence[Chunk], dense_vectors: Sequence[Sequence[float]]) -> int:
        for chunk, vector in zip(chunks, dense_vectors, strict=True):
            self._chunks[chunk.id] = chunk
            self._vectors[chunk.id] = [float(value) for value in vector]
            self._terms[chunk.id] = content_terms(chunk.text, lemmatize=True, limit=400)
        return len(chunks)

    @staticmethod
    def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
        if not left or not right:
            return 0.0
        size = min(len(left), len(right))
        dot = sum(float(left[i]) * float(right[i]) for i in range(size))
        norm_left = math.sqrt(sum(float(left[i]) ** 2 for i in range(size)))
        norm_right = math.sqrt(sum(float(right[i]) ** 2 for i in range(size)))
        if norm_left <= 0 or norm_right <= 0:
            return 0.0
        return dot / (norm_left * norm_right)

    def _bm25_scores(self, query_terms: list[str]) -> dict[str, float]:
        if not query_terms or not self._terms:
            return {}
        total_docs = len(self._terms)
        doc_freq: dict[str, int] = {}
        for terms in self._terms.values():
            for term in set(terms) & set(query_terms):
                doc_freq[term] = doc_freq.get(term, 0) + 1

        avg_len = sum(len(terms) for terms in self._terms.values()) / total_docs
        k1, b = 1.5, 0.75
        scores: dict[str, float] = {}
        for chunk_id, terms in self._terms.items():
            length = len(terms) or 1
            score = 0.0
            for term in query_terms:
                freq = terms.count(term)
                if freq == 0:
                    continue
                df = doc_freq.get(term, 0) or 1
                idf = math.log(1 + (total_docs - df + 0.5) / (df + 0.5))
                score += (
                    idf * (freq * (k1 + 1)) / (freq + k1 * (1 - b + b * length / (avg_len or 1)))
                )
            if score > 0:
                scores[chunk_id] = score
        return scores

    def search(
        self,
        *,
        query_text: str,
        query_vector: Sequence[float],
        limit: int,
        settings: RetrievalSettings,
    ) -> list[ScoredChunk]:
        dense_ranked = sorted(
            (
                (chunk_id, self._cosine(query_vector, vector))
                for chunk_id, vector in self._vectors.items()
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )[: settings.dense_candidates]

        query_terms = content_terms(query_text, lemmatize=True, limit=64)
        sparse_ranked = sorted(
            self._bm25_scores(query_terms).items(), key=lambda pair: pair[1], reverse=True
        )[: settings.sparse_candidates]

        # Слияние взаимных рангов — тот же принцип, что использует Qdrant.
        fused: dict[str, float] = {}
        channels: dict[str, list[str]] = {}
        per_channel: dict[str, dict[str, float]] = {}
        for rank, (chunk_id, score) in enumerate(dense_ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            channels.setdefault(chunk_id, []).append("dense")
            per_channel.setdefault(chunk_id, {})["dense"] = float(score)
        for rank, (chunk_id, score) in enumerate(sparse_ranked):
            fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (settings.rrf_k + rank + 1)
            channels.setdefault(chunk_id, []).append("sparse")
            per_channel.setdefault(chunk_id, {})["sparse"] = float(score)

        ranked = sorted(fused.items(), key=lambda pair: pair[1], reverse=True)[: int(limit)]
        return [
            ScoredChunk(
                chunk=self._chunks[chunk_id],
                score=score,
                channels=channels.get(chunk_id, []),
                channel_scores=per_channel.get(chunk_id, {}),
            )
            for chunk_id, score in ranked
            if chunk_id in self._chunks
        ]

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        return {cid: self._chunks[cid] for cid in chunk_ids if cid in self._chunks}

    def iter_chunks(self, batch_size: int = 256) -> Iterable[Chunk]:
        yield from self._chunks.values()

    def count(self) -> int:
        return len(self._chunks)

    def delete_document(self, doc_id: str) -> int:
        removed = [cid for cid, chunk in self._chunks.items() if chunk.doc_id == doc_id]
        for chunk_id in removed:
            self._chunks.pop(chunk_id, None)
            self._vectors.pop(chunk_id, None)
            self._terms.pop(chunk_id, None)
        return len(removed)


def build_vector_store(settings: VectorStoreSettings) -> VectorStore:
    if settings.provider == "memory":
        return InMemoryVectorStore()
    return QdrantVectorStore(settings)
