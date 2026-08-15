"""Сквозные тесты графового канала.

Назначение: доказать, что граф — действующая часть системы, а не отключённый
код. Здесь проверяется весь путь целиком: извлечение сущностей и связей →
запись в граф → поиск стартовых сущностей по вопросу → расширение по
типизированным связям → попадание графовых документов в финальный контекст
и в цитаты ответа.

Neo4j для этого не нужен: заглушка из ``tests/fakes.py`` повторяет логику
боевого хранилища, а не подменяет её заранее заданными ответами.
"""

from __future__ import annotations

from rag_textbook.clients.embeddings import FakeEmbeddingClient
from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.clients.reranker import FakeRerankerClient
from rag_textbook.config import GraphSettings, Settings
from rag_textbook.generation.answering import AnswerGenerator
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.models import Chunk, content_hash
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.retrieval.pipeline import RetrievalPipeline
from rag_textbook.stores.vector_store import InMemoryVectorStore
from tests.fakes import InMemoryGraphStore

# Корпус построен так, чтобы ответ на связывающий вопрос требовал двух фрагментов:
# определение SVD лежит в одном месте, а связь с PCA — в другом, далеко по тексту.
CORPUS: list[tuple[int, str]] = [
    (0, "Сингулярное разложение матрицы раскладывает её на три множителя."),
    (1, "Ортогональные матрицы сохраняют длины векторов при умножении."),
    (2, "Ковариационная матрица описывает совместную изменчивость признаков."),
    (3, "Определённый интеграл вычисляется по формуле Ньютона и Лейбница."),
    (
        20,
        "Метод главных компонент выражается через сингулярное разложение "
        "ковариационной матрицы выборки.",
    ),
]

# Явный граф: связь SVD ↔ PCA, которую невозможно получить из одного фрагмента.
EXTRACTIONS: dict[str, str] = {
    "d:00000": (
        '{"entities": [{"name": "сингулярное разложение"}, {"name": "матрица"}],'
        ' "relations": [{"source": "сингулярное разложение", "relation": "определяется_через",'
        ' "target": "матрица"}]}'
    ),
    "d:00001": '{"entities": [{"name": "ортогональная матрица"}], "relations": []}',
    "d:00002": '{"entities": [{"name": "ковариационная матрица"}], "relations": []}',
    "d:00003": '{"entities": [{"name": "определённый интеграл"}], "relations": []}',
    "d:00020": (
        '{"entities": [{"name": "метод главных компонент"}, {"name": "сингулярное разложение"},'
        ' {"name": "ковариационная матрица"}],'
        ' "relations": [{"source": "метод главных компонент", "relation": "вычисляется_по",'
        ' "target": "сингулярное разложение"},'
        ' {"source": "метод главных компонент", "relation": "определяется_через",'
        ' "target": "ковариационная матрица"}]}'
    ),
}


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"d:{ordinal:05d}",
            doc_id="d",
            doc_name="Линейная алгебра",
            source_path="/linal.pdf",
            ordinal=ordinal,
            text=text,
            pages=[ordinal + 1],
            text_hash=content_hash(text),
        )
        for ordinal, text in CORPUS
    ]


class _ScriptedLLM(FakeLLMClient):
    """Отдаёт разметку, соответствующую конкретному фрагменту."""

    def __init__(self, mapping: dict[str, str]) -> None:
        super().__init__()
        self._mapping = mapping

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        if json_schema is not None:
            prompt = messages[-1].content
            for chunk_id, payload in self._mapping.items():
                marker = dict(CORPUS)[int(chunk_id.split(":")[1])][:40]
                if marker in prompt:
                    return payload
            return '{"entities": [], "relations": []}'
        return super().chat(
            messages,
            purpose=purpose,
            json_schema=json_schema,
            max_tokens=max_tokens,
            temperature=temperature,
        )


def _build_graph(graph_settings: GraphSettings) -> InMemoryGraphStore:
    store = InMemoryGraphStore()
    extractor = EntityExtractor(graph_settings, llm=_ScriptedLLM(EXTRACTIONS))
    builder = GraphBuilder(graph_settings, extractor, store, max_workers=1)
    builder.build(
        _chunks(), doc_id="d", doc_name="Линейная алгебра", source_path="/linal.pdf", write=True
    )
    return store


def _graph_settings(**overrides) -> GraphSettings:
    defaults = {
        "enabled": True,
        "retrieval_enabled": True,
        "extractor": "llm",
        "extraction_cache_enabled": False,
        "max_entity_degree": 0,
        "cooccurrence_enabled": False,
        "weight": 0.5,
        "expansion_hops": 1,
    }
    defaults.update(overrides)
    return GraphSettings(**defaults)


def _pipeline(settings: Settings, store: InMemoryGraphStore) -> RetrievalPipeline:
    vector_store = InMemoryVectorStore()
    vector_store.ensure_collection(64)
    embeddings = FakeEmbeddingClient(dimensions=64)
    chunks = _chunks()
    vector_store.upsert(chunks, embeddings.embed_documents([c.text for c in chunks]))

    return RetrievalPipeline(
        settings=settings,
        vector_store=vector_store,
        embedding_client=embeddings,
        reranker=FakeRerankerClient(),
        graph_retriever=GraphRetriever(settings.graph, store),
        llm=FakeLLMClient(),
    )


# ------------------------------------------------------------- построение графа


def test_graph_is_built_with_typed_relations() -> None:
    store = _build_graph(_graph_settings())

    assert store.entities, "сущности должны попасть в граф"
    assert store.relations, "типизированные связи должны попасть в граф"
    assert store.mentions, "упоминания должны связывать пассажи с сущностями"

    labels = {relation.label for relation in store.relations}
    assert "вычисляется_по" in labels, "связь SVD ↔ PCA должна быть извлечена"


def test_graph_is_not_dominated_by_cooccurrence() -> None:
    """Прежде 99.4% рёбер были co-occurrence, и граф работал как шумный BM25."""
    store = _build_graph(_graph_settings(cooccurrence_enabled=True, cooccurrence_min_pmi=1.0))
    typed = len(store.relations)
    noisy = len(store.cooccurrences)

    assert typed > 0
    assert typed >= noisy, (
        f"типизированных связей ({typed}) должно быть не меньше co-occurrence ({noisy})"
    )


# ------------------------------------------------------------- графовый канал


def test_graph_retriever_finds_passages_by_question() -> None:
    graph_settings = _graph_settings()
    store = _build_graph(graph_settings)
    retriever = GraphRetriever(graph_settings, store)

    results = retriever.retrieve("Как связаны сингулярное разложение и метод главных компонент?")

    assert results, "графовый канал обязан что-то вернуть на релевантном вопросе"
    assert all(item.from_graph for item in results)
    assert all(item.matched_entities for item in results), "должны быть видны совпавшие сущности"


def test_graph_expansion_reaches_related_passage() -> None:
    """Расширение по RELATES обязано доставать фрагмент, которого нет в вопросе дословно."""
    graph_settings = _graph_settings(expansion_hops=1)
    store = _build_graph(graph_settings)
    retriever = GraphRetriever(graph_settings, store)

    results = retriever.retrieve("метод главных компонент")
    found_ids = {item.chunk.id for item in results}

    assert "d:00020" in found_ids
    assert "d:00000" in found_ids, (
        "через связь «вычисляется_по» должен подтянуться фрагмент про SVD"
    )


def test_graph_retriever_ignores_irrelevant_passages() -> None:
    graph_settings = _graph_settings()
    store = _build_graph(graph_settings)
    retriever = GraphRetriever(graph_settings, store)

    results = retriever.retrieve("сингулярное разложение")
    assert "d:00003" not in {item.chunk.id for item in results}, (
        "фрагмент про интегралы не должен попадать в графовую выдачу"
    )


# --------------------------------------------------------- сквозной путь ответа


def test_graph_documents_reach_final_context(settings) -> None:
    graph_settings = _graph_settings()
    settings.graph = graph_settings
    settings.retrieval.router_mode = "always"
    store = _build_graph(graph_settings)

    result = _pipeline(settings, store).retrieve(
        "Как связаны сингулярное разложение и метод главных компонент?"
    )

    assert result.route is not None and result.route.use_graph
    assert result.channel_sizes["graph"] > 0, "графовый канал должен отработать"
    assert result.graph_share > 0, "графовые документы должны дойти до финального контекста"
    assert "graph_channel" in result.timings_ms, "время графового канала должно измеряться"


def test_router_activates_graph_on_relational_question(settings) -> None:
    graph_settings = _graph_settings()
    settings.graph = graph_settings
    settings.retrieval.router_mode = "heuristic"
    store = _build_graph(graph_settings)
    pipeline = _pipeline(settings, store)

    relational = pipeline.retrieve("Как связаны сингулярное разложение и метод главных компонент?")
    factual = pipeline.retrieve("Что такое ортогональная матрица?")

    assert relational.route.use_graph, "связывающий вопрос должен идти в граф"
    assert not factual.route.use_graph, "вопрос-определение граф звать не должен"
    assert factual.channel_sizes["graph"] == 0


def test_minimum_graph_quota_forces_graph_into_context(settings) -> None:
    """Квота отличает «граф не нашёл» от «граф вытеснен базовым каналом»."""
    graph_settings = _graph_settings(weight=0.01)
    settings.graph = graph_settings
    settings.retrieval.router_mode = "always"
    settings.retrieval.top_k = 3
    store = _build_graph(graph_settings)

    settings.retrieval.min_graph_docs = 0
    without_quota = _pipeline(settings, store).retrieve("метод главных компонент")

    settings.retrieval.min_graph_docs = 2
    with_quota = _pipeline(settings, store).retrieve("метод главных компонент")

    graph_in_quota = sum(1 for item in with_quota.chunks if item.from_graph)
    assert graph_in_quota >= 2
    assert graph_in_quota >= sum(1 for item in without_quota.chunks if item.from_graph)


def test_answer_cites_graph_sourced_fragment(settings) -> None:
    graph_settings = _graph_settings()
    settings.graph = graph_settings
    settings.retrieval.router_mode = "always"
    settings.retrieval.min_graph_docs = 1
    store = _build_graph(graph_settings)
    pipeline = _pipeline(settings, store)

    generator = AnswerGenerator(
        settings, pipeline, FakeLLMClient(responses=["Связь описана в [1] и [2]."])
    )
    answer = generator.answer("Как связаны сингулярное разложение и метод главных компонент?")

    assert answer.used_graph is True
    assert answer.citations, "ответ должен содержать цитаты"
    assert any(citation.from_graph for citation in answer.citations), (
        "происхождение цитаты из графа должно быть видно пользователю"
    )
    assert all(citation.pages for citation in answer.citations), "цитаты должны иметь страницы"


def test_ab_experiment_graph_on_off_produces_comparable_metrics(settings) -> None:
    """Именно этот прогон отвечает на главный вопрос проекта — даёт ли граф прирост."""
    from rag_textbook.evaluation.metrics import compare
    from rag_textbook.evaluation.runner import run_retrieval_evaluation
    from rag_textbook.models import GoldQuestion

    graph_settings = _graph_settings()
    settings.graph = graph_settings
    settings.retrieval.router_mode = "always"
    store = _build_graph(graph_settings)

    questions = [
        GoldQuestion(
            id="q1",
            question="Как связаны сингулярное разложение и метод главных компонент?",
            gold_chunk_ids=["d:00000", "d:00020"],
            question_type="multi_hop",
            expected_hops=2,
        ),
        GoldQuestion(
            id="q2",
            question="Что такое ковариационная матрица?",
            gold_chunk_ids=["d:00002"],
            question_type="single_chunk",
        ),
    ]

    class _Ctx:
        def __init__(self, pipeline, settings) -> None:
            self.retrieval = pipeline
            self.settings = settings

    with_graph = _pipeline(settings, store)
    without_graph = _pipeline(settings, store)
    without_graph.graph_retriever = None

    metrics_graph, _ = run_retrieval_evaluation(
        _Ctx(with_graph, settings), questions, max_workers=1
    )
    metrics_base, _ = run_retrieval_evaluation(
        _Ctx(without_graph, settings), questions, max_workers=1
    )

    result = compare(metrics_base, metrics_graph, settings.retrieval.top_k)
    assert "delta" in result and "recall" in result["delta"]
    assert result["warning"], "на двух вопросах вывод недостоверен, и это должно быть сказано"
    # Граф не обязан выигрывать на этом микрокорпусе; проверяем, что сравнение выполнимо.
    assert metrics_graph.graph_usage["routed_to_graph"] == 1.0
    assert metrics_base.graph_usage["avg_graph_share_in_context"] == 0.0
