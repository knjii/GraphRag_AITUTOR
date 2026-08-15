"""Общие фикстуры.

Все тесты работают без Qdrant, Neo4j, Ollama и Infinity: клиенты подменяются
детерминированными заглушками, векторное хранилище — реализацией в памяти.
Это принципиально: прежний проект нельзя было проверить, не подняв весь стек,
поэтому его никто и не проверял.
"""

from __future__ import annotations

import os

# ВАЖНО: до первого импорта rag_textbook. Иначе конфигурация подхватит рабочий
# .env, если он лежит рядом, и тесты начнут проверять не значения по умолчанию,
# а содержимое чужого файла. Именно так набор проходил на машине разработки и
# падал на развёрнутом сервере.
os.environ.setdefault("RAG_ENV_FILE", "tests-no-such-env-file")

import pytest  # noqa: E402

from rag_textbook.clients.embeddings import FakeEmbeddingClient  # noqa: E402
from rag_textbook.clients.llm import FakeLLMClient  # noqa: E402
from rag_textbook.clients.reranker import FakeRerankerClient  # noqa: E402
from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.models import Block, Chunk, content_hash  # noqa: E402
from rag_textbook.retrieval.pipeline import RetrievalPipeline  # noqa: E402
from rag_textbook.stores.vector_store import InMemoryVectorStore  # noqa: E402


@pytest.fixture
def settings(tmp_path) -> Settings:
    """Конфигурация в тестовом окружении, без внешних сервисов."""
    settings = Settings(
        embedding={"provider": "fake", "dimensions": 64, "cache_enabled": False},
        llm={"provider": "fake"},
        reranker={"provider": "fake", "enabled": True, "candidates": 20, "top_n": 8},
        vector_store={"provider": "memory"},
        graph={"enabled": False, "retrieval_enabled": False},
        retrieval={"top_k": 5, "router_mode": "heuristic"},
    )
    settings.paths.cache_dir = tmp_path / "cache"
    settings.paths.parsed_dir = tmp_path / "parsed"
    settings.paths.manifest_dir = tmp_path / "manifests"
    settings.paths.metrics_dir = tmp_path / "metrics"
    settings.paths.state_dir = tmp_path / "state"
    settings.paths.goldset_dir = tmp_path / "goldsets"
    settings.service.history_db_path = tmp_path / "state" / "history.sqlite3"
    settings.paths.ensure()
    return settings


@pytest.fixture
def sample_blocks() -> list[Block]:
    """Фрагмент учебника: текст, формула, таблица, иллюстрация."""
    return [
        Block(
            index=0, type="text", text="Глава 3. Сингулярное разложение", text_level=1, page_idx=0
        ),
        Block(
            index=1,
            type="text",
            text=(
                "Сингулярное разложение матрицы применяется для понижения размерности. "
                "Оно связано с методом главных компонент и используется при сжатии данных. "
            )
            * 4,
            page_idx=0,
        ),
        Block(
            index=2,
            type="equation",
            text=r"A = U \Sigma V^{T}",
            latex=r"A = U \Sigma V^{T}",
            page_idx=1,
        ),
        Block(
            index=3,
            type="table",
            table_html="<table><tr><th>k</th><th>sigma</th></tr><tr><td>1</td><td>5.2</td></tr></table>",
            caption="Таблица 1. Сингулярные числа",
            page_idx=1,
        ),
        Block(
            index=4,
            type="image",
            img_path="images/svd.jpg",
            caption="Рис. 2. Геометрическая интерпретация",
            page_idx=2,
        ),
        Block(
            index=5,
            type="text",
            text=(
                "Метод главных компонент строится на ковариационной матрице. "
                "Он позволяет выделить направления наибольшей дисперсии данных. "
            )
            * 4,
            page_idx=3,
        ),
    ]


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    def make(ordinal: int, text: str, **kwargs) -> Chunk:
        return Chunk(
            id=f"doc1:{ordinal:05d}",
            doc_id="doc1",
            doc_name="Линейная алгебра",
            source_path="/corpus/linal.pdf",
            ordinal=ordinal,
            text=text,
            pages=[ordinal + 1],
            text_hash=content_hash(text),
            **kwargs,
        )

    return [
        make(0, "Сингулярное разложение матрицы раскладывает её на три множителя."),
        make(
            1,
            r"Формула разложения имеет вид $$A = U \Sigma V^{T}$$ где сигма диагональна.",
            has_formula=True,
        ),
        make(2, "Метод главных компонент использует ковариационную матрицу данных."),
        make(3, "Собственные значения ковариационной матрицы задают дисперсию."),
        make(4, "Понижение размерности применяется для визуализации многомерных данных."),
    ]


@pytest.fixture
def populated_store(sample_chunks) -> InMemoryVectorStore:
    store = InMemoryVectorStore()
    store.ensure_collection(64)
    embeddings = FakeEmbeddingClient(dimensions=64)
    vectors = embeddings.embed_documents([chunk.text for chunk in sample_chunks])
    store.upsert(sample_chunks, vectors)
    return store


@pytest.fixture
def pipeline(settings, populated_store) -> RetrievalPipeline:
    return RetrievalPipeline(
        settings=settings,
        vector_store=populated_store,
        embedding_client=FakeEmbeddingClient(dimensions=64),
        reranker=FakeRerankerClient(),
        graph_retriever=None,
        llm=FakeLLMClient(),
    )
