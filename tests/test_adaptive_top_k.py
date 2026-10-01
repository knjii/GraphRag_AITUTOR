"""Размер выдачи, зависящий от типа вопроса.

Основание — измеренная кривая recall по размеру выдачи на полном корпусе,
172 вопроса:

    тип             @8     @12    @16    @24
    одношаговые     0.944  0.963  0.963  0.963
    с формулами     0.966  0.983  0.983  0.983
    многошаговые    0.576  0.686  0.737  0.822

Простые вопросы насыщаются к двенадцати фрагментам, связывающие растут
до двадцати четырёх: связывающему нужно два эталонных фрагмента вместо
одного, и та же квота вмещает вдвое меньше ответов. Расширять выдачу всем —
платить токенами там, где прироста нет.
"""

from __future__ import annotations

from rag_textbook.clients.embeddings import FakeEmbeddingClient
from rag_textbook.clients.reranker import FakeRerankerClient
from rag_textbook.config import RetrievalSettings, Settings
from rag_textbook.models import Chunk, content_hash
from rag_textbook.retrieval.pipeline import RetrievalPipeline
from rag_textbook.stores.vector_store import InMemoryVectorStore


def test_zero_means_same_as_usual() -> None:
    """По умолчанию поведение не меняется."""
    settings = RetrievalSettings(_env_file=None, top_k=8)  # type: ignore[arg-type]
    assert settings.top_k_linking == 0
    assert settings.top_k_for(linking=True) == 8
    assert settings.top_k_for(linking=False) == 8


def test_linking_questions_get_the_wider_quota() -> None:
    settings = RetrievalSettings(_env_file=None, top_k=8, top_k_linking=16)  # type: ignore[arg-type]
    assert settings.top_k_for(linking=True) == 16
    assert settings.top_k_for(linking=False) == 8


def test_quota_never_shrinks_below_the_base() -> None:
    """Меньшее значение не должно урезать выдачу связывающих вопросов."""
    settings = RetrievalSettings(_env_file=None, top_k=10, top_k_linking=4)  # type: ignore[arg-type]
    assert settings.top_k_for(linking=True) == 10


def _pipeline(top_k: int, top_k_linking: int, router_mode: str) -> RetrievalPipeline:
    settings = Settings(
        embedding={"provider": "fake", "dimensions": 64, "cache_enabled": False},
        llm={"provider": "fake"},
        reranker={"provider": "fake", "enabled": True, "candidates": 30, "top_n": 8},
        vector_store={"provider": "memory"},
        graph={"enabled": False, "retrieval_enabled": False},
        retrieval={
            "top_k": top_k,
            "top_k_linking": top_k_linking,
            "router_mode": router_mode,
            # Фрагменты корпуса намеренно однотипны: проверяется размер выдачи,
            # а схлопывание похожих оставило бы от них один.
            "dedup_enabled": False,
        },
    )
    store = InMemoryVectorStore()
    store.ensure_collection(64)
    embeddings = FakeEmbeddingClient(dimensions=64)
    chunks = [
        Chunk(
            id=f"d:{index:05d}",
            doc_id="d",
            doc_name="Учебник",
            source_path="/book.pdf",
            ordinal=index,
            text=f"Фрагмент номер {index} про линейную алгебру и матрицы.",
            pages=[index + 1],
            text_hash=content_hash(f"chunk-{index}"),
        )
        for index in range(40)
    ]
    store.upsert(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return RetrievalPipeline(
        settings=settings,
        vector_store=store,
        embedding_client=embeddings,
        reranker=FakeRerankerClient(),
        graph_retriever=None,
        llm=None,
    )


def test_wider_quota_reaches_the_final_context() -> None:
    """Настройка должна доходить до выдачи, а не только до конфигурации."""
    linking = _pipeline(top_k=8, top_k_linking=16, router_mode="always")
    plain = _pipeline(top_k=8, top_k_linking=16, router_mode="never")

    assert len(linking.retrieve("Как связаны матрицы и разложения?").chunks) == 16
    assert len(plain.retrieve("Что такое матрица?").chunks) == 8


def test_default_configuration_returns_the_base_quota() -> None:
    pipeline = _pipeline(top_k=8, top_k_linking=0, router_mode="always")
    assert len(pipeline.retrieve("Как связаны матрицы и разложения?").chunks) == 8
