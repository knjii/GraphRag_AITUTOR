"""Тесты конвейера поиска, слияния, роутера и построения графа."""

from __future__ import annotations

from rag_textbook.clients.embeddings import FakeEmbeddingClient
from rag_textbook.clients.llm import ChatMessage, FakeLLMClient
from rag_textbook.clients.reranker import FakeRerankerClient
from rag_textbook.config import GraphSettings, RetrievalSettings
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.models import Chunk, ScoredChunk, content_hash
from rag_textbook.retrieval.fusion import (
    deduplicate,
    enforce_minimum_graph_documents,
    reciprocal_rank_fusion,
)
from rag_textbook.retrieval.pipeline import RetrievalPipeline
from rag_textbook.retrieval.router import QueryRouter
from rag_textbook.stores.vector_store import InMemoryVectorStore


def _chunk(chunk_id: str, text: str, ordinal: int = 0) -> Chunk:
    return Chunk(
        id=chunk_id,
        doc_id="doc",
        doc_name="Учебник",
        source_path="/x.pdf",
        ordinal=ordinal,
        text=text,
        pages=[ordinal + 1],
        text_hash=content_hash(text),
    )


def _scored(chunk_id: str, text: str, channel: str, score: float = 1.0) -> ScoredChunk:
    return ScoredChunk(
        chunk=_chunk(chunk_id, text),
        score=score,
        channels=[channel],
        channel_scores={channel: score},
    )


# ------------------------------------------------------------------ слияние


def test_rrf_merges_channels_and_tracks_origin() -> None:
    base = [_scored("a", "текст а", "dense"), _scored("b", "текст б", "dense")]
    graph = [_scored("b", "текст б", "graph_entity"), _scored("c", "текст в", "graph_entity")]

    merged = reciprocal_rank_fusion({"base": base, "graph": graph}, {"base": 0.6, "graph": 0.4})

    assert {item.chunk.id for item in merged} == {"a", "b", "c"}
    found_in_both = next(item for item in merged if item.chunk.id == "b")
    assert found_in_both.score > merged[-1].score, "документ из двух каналов должен быть выше"
    assert any(item.from_graph for item in merged), "происхождение из графа должно сохраняться"


def test_zero_weight_channel_is_excluded() -> None:
    base = [_scored("a", "текст", "dense")]
    graph = [_scored("z", "графовый", "graph_entity")]
    merged = reciprocal_rank_fusion({"base": base, "graph": graph}, {"base": 1.0, "graph": 0.0})
    assert {item.chunk.id for item in merged} == {"a"}


def test_deduplicate_removes_overlapping_chunks() -> None:
    body = (
        "сингулярное разложение матрицы применяется понижения размерности данных "
        "собственные значения ковариационная матрица дисперсия проекция направление"
    )
    items = [
        _scored("a", body, "dense", 1.0),
        _scored("b", body + " небольшой хвост", "dense", 0.9),
        _scored("c", "определённый интеграл вычисляется формулой ньютона лейбница", "dense", 0.8),
    ]
    result = deduplicate(items, similarity_threshold=0.85)
    assert [item.chunk.id for item in result] == ["a", "c"]


def test_minimum_graph_documents_is_enforced() -> None:
    items = [
        _scored("a", "т1", "dense"),
        _scored("b", "т2", "dense"),
        _scored("c", "т3", "dense"),
        _scored("g", "т4", "graph_entity"),
    ]
    result = enforce_minimum_graph_documents(items, minimum=1, top_k=3)
    assert len(result) == 3
    assert any(item.from_graph for item in result), (
        "квота нужна, чтобы отличить «граф не нашёл» от «граф вытеснен»"
    )


def test_minimum_zero_keeps_natural_order() -> None:
    items = [_scored("a", "т1", "dense"), _scored("b", "т2", "dense")]
    assert len(enforce_minimum_graph_documents(items, minimum=0, top_k=1)) == 1


# ------------------------------------------------------------------- роутер


def test_router_sends_relational_questions_to_graph() -> None:
    router = QueryRouter(RetrievalSettings(router_mode="heuristic"))
    assert router.route("Как связаны SVD и метод главных компонент?").use_graph
    assert router.route("Чем отличается ковариационная матрица от корреляционной?").use_graph


def test_router_keeps_definitions_on_base_channel() -> None:
    router = QueryRouter(RetrievalSettings(router_mode="heuristic"))
    assert not router.route("Что такое сингулярное разложение?").use_graph


def test_router_modes_override_heuristic() -> None:
    assert QueryRouter(RetrievalSettings(router_mode="always")).route("что такое x").use_graph
    assert (
        not QueryRouter(RetrievalSettings(router_mode="never")).route("как связаны x и y").use_graph
    )


# ------------------------------------------------------------------ конвейер


def test_pipeline_returns_top_k_and_measures_stages(pipeline, settings) -> None:
    result = pipeline.retrieve("сингулярное разложение матрицы")

    assert result.chunks, "конвейер обязан что-то найти на непустом индексе"
    assert len(result.chunks) <= settings.retrieval.top_k
    for stage in ("rewrite", "route", "base_channel", "fusion", "rerank", "total"):
        assert stage in result.timings_ms, f"стадия {stage} должна измеряться"


def test_pipeline_rewrites_followup_question(settings, populated_store) -> None:
    """Регрессия: прежде вопрос-продолжение уходил в поиск буквально.

    В старой цепочке не было history-aware retriever, поэтому «а как это
    применить?» искалось по этой самой строке.
    """
    llm = FakeLLMClient(responses=["Как применить сингулярное разложение на практике?"])
    pipeline = RetrievalPipeline(
        settings=settings,
        vector_store=populated_store,
        embedding_client=FakeEmbeddingClient(dimensions=64),
        reranker=FakeRerankerClient(),
        graph_retriever=None,
        llm=llm,
    )
    history = [
        ChatMessage(role="user", content="Что такое сингулярное разложение?"),
        ChatMessage(role="assistant", content="Это разложение матрицы на три множителя."),
    ]
    result = pipeline.retrieve("а как это применить?", history)

    assert result.rewritten_question != "а как это применить?"
    assert "сингулярное" in result.rewritten_question.lower()


def test_pipeline_without_history_keeps_question(pipeline) -> None:
    result = pipeline.retrieve("метод главных компонент", history=[])
    assert result.rewritten_question == "метод главных компонент"


def test_vector_store_is_idempotent(sample_chunks) -> None:
    store = InMemoryVectorStore()
    store.ensure_collection(64)
    embeddings = FakeEmbeddingClient(dimensions=64)
    vectors = embeddings.embed_documents([chunk.text for chunk in sample_chunks])

    store.upsert(sample_chunks, vectors)
    first = store.count()
    store.upsert(sample_chunks, vectors)

    assert store.count() == first, "повторная запись не должна плодить дубликаты"


# --------------------------------------------------------------------- граф


def test_rule_extractor_does_not_create_relations() -> None:
    """Регрессия: правиловый экстрактор строил клики из всех пар терминов.

    Именно так получался граф, где 99.4% рёбер были co-occurrence.
    """
    settings = GraphSettings(extractor="rule", extraction_cache_enabled=False)
    extractor = EntityExtractor(settings, llm=None)
    result = extractor.extract(_chunk("c1", "матрица разложение собственные значения дисперсия"))

    assert result.entities, "сущности извлекаться должны"
    assert result.relations == [], "рёбра по совместной встречаемости строить нельзя"


def test_llm_extractor_drops_relations_to_unknown_entities() -> None:
    payload = (
        '{"entities": [{"name": "сингулярное разложение"}, {"name": "матрица"}],'
        ' "relations": [{"source": "сингулярное разложение", "relation": "определяется_через",'
        ' "target": "матрица"},'
        ' {"source": "сингулярное разложение", "relation": "используется_в", "target": "нечто"}]}'
    )
    settings = GraphSettings(extractor="llm", extraction_cache_enabled=False)
    extractor = EntityExtractor(settings, llm=FakeLLMClient(responses=[payload]))
    result = extractor.extract(_chunk("c1", "текст про разложение матрицы"))

    assert len(result.entities) == 2
    assert len(result.relations) == 1, "связь на неизвестную сущность должна отбрасываться"
    assert result.relations[0].label == "определяется_через"


def test_llm_extractor_normalizes_unknown_relation_label() -> None:
    payload = (
        '{"entities": [{"name": "матрица"}, {"name": "вектор"}],'
        ' "relations": [{"source": "матрица", "relation": "какая-то выдуманная связь",'
        ' "target": "вектор"}]}'
    )
    settings = GraphSettings(extractor="llm", extraction_cache_enabled=False)
    extractor = EntityExtractor(settings, llm=FakeLLMClient(responses=[payload]))
    result = extractor.extract(_chunk("c1", "текст"))

    from rag_textbook.graph.extractor import RELATION_LABELS

    assert result.relations[0].label in RELATION_LABELS, (
        "свободный текст в метке связи делает граф необходимым"
    )


def test_llm_extractor_falls_back_on_broken_json() -> None:
    settings = GraphSettings(extractor="llm", extraction_cache_enabled=False)
    extractor = EntityExtractor(settings, llm=FakeLLMClient(responses=["не json вовсе"]))
    result = extractor.extract(_chunk("c1", "матрица разложение собственные значения"))

    assert result.status == "rule_fallback"
    assert result.entities, "пассаж не должен остаться вне графа из-за сбоя разбора"


def test_extraction_cache_prevents_repeated_llm_calls(tmp_path) -> None:
    from rag_textbook.utils.cache import ArtifactCache

    payload = '{"entities": [{"name": "матрица"}], "relations": []}'
    llm = FakeLLMClient(responses=[payload])
    cache = ArtifactCache(tmp_path / "extraction.sqlite3", "extraction")
    settings = GraphSettings(extractor="llm", extraction_cache_enabled=True)
    extractor = EntityExtractor(settings, llm=llm, cache=cache)
    chunk = _chunk("c1", "текст про матрицу")

    first = extractor.extract(chunk, "model-x")
    second = extractor.extract(chunk, "model-x")

    assert len(llm.calls) == 1, "повторная индексация не должна снова платить за экстракцию"
    assert [e.canonical for e in first.entities] == [e.canonical for e in second.entities]
    cache.close()


def test_cooccurrence_is_filtered_by_pmi() -> None:
    """Регрессия на главную причину бесполезности прежнего графа.

    Термин, встречающийся в каждом чанке, связывался со всем подряд просто
    потому, что он частотный. PMI такие пары отсекает.
    """
    settings = GraphSettings(
        extractor="rule",
        extraction_cache_enabled=False,
        cooccurrence_enabled=True,
        cooccurrence_min_pmi=1.0,
        cooccurrence_min_count=2,
        max_entity_degree=0,
    )
    extractor = EntityExtractor(settings, llm=None)
    builder = GraphBuilder(settings, extractor, store=None, max_workers=1)

    # «матрица» встречается всюду, пара «ядро–гильберт» — только вместе.
    chunks = [
        _chunk("c0", "матрица дисперсия проекция", 0),
        _chunk("c1", "матрица регрессия ошибка", 1),
        _chunk("c2", "матрица градиент шаг", 2),
        _chunk("c3", "ядро гильберт пространство", 3),
        _chunk("c4", "ядро гильберт пространство", 4),
    ]
    result = builder.build(chunks, doc_id="d", doc_name="n", source_path="/p.pdf", write=False)

    assert result.cooccurrence_candidates > result.cooccurrences, "часть пар обязана отсеиваться"
    kept_ratio = result.as_dict()["cooccurrence_kept_ratio"]
    assert kept_ratio < 0.5, f"после фильтрации должно остаться меньшинство пар, а не {kept_ratio}"


def test_graph_builder_merges_duplicate_entities() -> None:
    settings = GraphSettings(extractor="rule", extraction_cache_enabled=False, max_entity_degree=0)
    extractor = EntityExtractor(settings, llm=None)
    builder = GraphBuilder(settings, extractor, store=None, max_workers=1)

    chunks = [
        _chunk("c0", "ковариационная матрица дисперсия", 0),
        _chunk("c1", "ковариационная матрица собственные значения", 1),
    ]
    result = builder.build(chunks, doc_id="d", doc_name="n", source_path="/p.pdf", write=False)

    assert result.entities < result.mentions, (
        "повторяющиеся термины должны схлопываться в один узел с несколькими упоминаниями"
    )
