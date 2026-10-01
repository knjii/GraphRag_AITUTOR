"""Тесты харнесса оценки, истории диалога, генерации ответа и сервиса."""

from __future__ import annotations

import pytest

from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.evaluation.metrics import (
    QueryOutcome,
    compare,
    evaluate_retrieval,
    hit_rate_at_k,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from rag_textbook.generation.answering import (
    AnswerGenerator,
    build_context_block,
    extract_citations,
)
from rag_textbook.generation.history import ChatHistoryStore
from rag_textbook.models import Chunk, ScoredChunk, content_hash

# ------------------------------------------------------------------ метрики


def test_recall_and_precision_basics() -> None:
    retrieved = ["a", "b", "c", "d"]
    relevant = ["b", "d", "z"]

    assert recall_at_k(retrieved, relevant, 4) == pytest.approx(2 / 3)
    assert recall_at_k(retrieved, relevant, 2) == pytest.approx(1 / 3)
    assert precision_at_k(retrieved, relevant, 4) == pytest.approx(0.5)
    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0
    assert hit_rate_at_k(retrieved, relevant, 1) == 0.0


def test_mrr_uses_first_relevant_position() -> None:
    assert mrr(["x", "y", "gold"], ["gold"]) == pytest.approx(1 / 3)
    assert mrr(["gold"], ["gold"]) == 1.0
    assert mrr(["x"], ["gold"]) == 0.0


def test_ndcg_rewards_higher_positions() -> None:
    top = ndcg_at_k(["gold", "x", "y"], ["gold"], 3)
    bottom = ndcg_at_k(["x", "y", "gold"], ["gold"], 3)
    assert top > bottom
    assert top == pytest.approx(1.0)


def test_evaluate_retrieval_splits_by_question_type() -> None:
    outcomes = [
        QueryOutcome("q1", "single_chunk", ["a", "b"], ["a"], latency_ms=10),
        QueryOutcome(
            "q2", "multi_hop", ["c", "d"], ["x"], used_graph=True, graph_share=0.5, latency_ms=30
        ),
        QueryOutcome(
            "q3", "multi_hop", ["y", "z"], ["y"], used_graph=True, graph_share=0.25, latency_ms=20
        ),
    ]
    metrics = evaluate_retrieval(outcomes, k_values=(1, 2))

    assert metrics.questions == 3
    assert "single_chunk" in metrics.by_type and "multi_hop" in metrics.by_type
    # Разрез по типам — то, ради чего всё затевалось: граф должен помогать
    # именно на multi_hop, а не «в среднем по больнице».
    assert metrics.by_type["multi_hop"]["questions"] == 2
    assert metrics.graph_usage["routed_to_graph"] == pytest.approx(2 / 3)
    assert metrics.latency["max"] == 30


def test_compare_warns_about_small_sample() -> None:
    baseline = evaluate_retrieval(
        [QueryOutcome(f"q{i}", "single_chunk", ["a"], ["a"]) for i in range(30)], (1,)
    )
    candidate = evaluate_retrieval(
        [QueryOutcome(f"q{i}", "single_chunk", ["a"], ["a"]) for i in range(30)], (1,)
    )
    result = compare(baseline, candidate, 1)

    assert result["questions"] == 30
    assert result["warning"], "на 30 вопросах различия недостоверны, и это должно быть сказано"
    assert result["confidence_margin"] > 0.05


def test_compare_detects_improvement() -> None:
    baseline = evaluate_retrieval(
        [QueryOutcome(f"q{i}", "t", ["x"], ["gold"]) for i in range(200)], (1,)
    )
    candidate = evaluate_retrieval(
        [QueryOutcome(f"q{i}", "t", ["gold"], ["gold"]) for i in range(200)], (1,)
    )
    result = compare(baseline, candidate, 1)

    assert result["delta"]["recall"] == pytest.approx(1.0)
    assert result["likely_significant"]["recall"] is True
    assert not result["warning"]


# ------------------------------------------------------------------ история


def test_history_isolates_users(tmp_path) -> None:
    """Регрессия: прежде сессию можно было прочитать, зная только её id."""
    store = ChatHistoryStore(tmp_path / "history.sqlite3")
    store.append("alice", "s1", "user", "вопрос Алисы")
    store.append("alice", "s1", "assistant", "ответ Алисе")

    assert len(store.recent("alice", "s1", 5)) == 2
    assert store.recent("bob", "s1", 5) == [], "чужая сессия не должна быть доступна"
    store.close()


def test_history_returns_last_turns_in_order(tmp_path) -> None:
    store = ChatHistoryStore(tmp_path / "history.sqlite3")
    for index in range(10):
        store.append("u", "s", "user", f"вопрос {index}")
        store.append("u", "s", "assistant", f"ответ {index}")

    messages = store.recent("u", "s", max_turns=2)
    assert len(messages) == 4
    assert messages[0].content == "вопрос 8"
    assert messages[-1].content == "ответ 9"
    store.close()


def test_history_clear_scoped_to_user(tmp_path) -> None:
    store = ChatHistoryStore(tmp_path / "history.sqlite3")
    store.append("u1", "s", "user", "раз")
    store.append("u2", "s", "user", "два")

    store.clear("u1", "s")
    assert store.recent("u1", "s", 5) == []
    assert len(store.recent("u2", "s", 5)) == 1
    store.close()


# ----------------------------------------------------------------- генерация


def _scored(chunk_id: str, text: str, pages: list[int], doc: str = "Учебник") -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=chunk_id,
            doc_id="d",
            doc_name=doc,
            source_path="/x.pdf",
            ordinal=0,
            text=text,
            pages=pages,
            text_hash=content_hash(text),
        ),
        score=1.0,
        channels=["dense"],
    )


def test_context_block_is_numbered_and_has_pages() -> None:
    block = build_context_block(
        [_scored("a", "Первый фрагмент", [12]), _scored("b", "Второй фрагмент", [40, 41])],
        max_chars_per_chunk=500,
    )
    assert "[1]" in block and "[2]" in block
    # Цитата без номера страницы бесполезна студенту; прежде страница терялась.
    assert "с. 12" in block
    assert "с. 40–41" in block


def test_extract_citations_returns_only_used_sources() -> None:
    chunks = [_scored("a", "текст", [1]), _scored("b", "текст", [2]), _scored("c", "текст", [3])]
    citations = extract_citations("Утверждение [1] и следствие [3].", chunks)

    assert [citation.index for citation in citations] == [1, 3]
    assert citations[1].pages == [3]


def test_extract_citations_ignores_out_of_range() -> None:
    chunks = [_scored("a", "текст", [1])]
    assert extract_citations("см. [7]", chunks) == []


def test_generator_reports_missing_context(settings, pipeline) -> None:
    from rag_textbook.clients.embeddings import FakeEmbeddingClient
    from rag_textbook.clients.reranker import FakeRerankerClient
    from rag_textbook.generation.answering import NO_CONTEXT_MESSAGE
    from rag_textbook.retrieval.pipeline import RetrievalPipeline
    from rag_textbook.stores.vector_store import InMemoryVectorStore

    empty_pipeline = RetrievalPipeline(
        settings=settings,
        vector_store=InMemoryVectorStore(),
        embedding_client=FakeEmbeddingClient(dimensions=64),
        reranker=FakeRerankerClient(),
        graph_retriever=None,
        llm=FakeLLMClient(),
    )
    generator = AnswerGenerator(settings, empty_pipeline, FakeLLMClient())
    answer = generator.answer("вопрос про пустой индекс")

    assert answer.answer == NO_CONTEXT_MESSAGE
    assert answer.contexts == []


def test_generator_produces_answer_with_timings(settings, pipeline) -> None:
    generator = AnswerGenerator(
        settings, pipeline, FakeLLMClient(responses=["Разложение описано в [1]."])
    )
    answer = generator.answer("сингулярное разложение")

    assert answer.answer
    assert answer.contexts
    assert "total" in answer.timings_ms and "generation" in answer.timings_ms
    assert [citation.index for citation in answer.citations] == [1]


# ------------------------------------------------------------------- сервис


def test_api_requires_user_header() -> None:
    fastapi_testclient = pytest.importorskip("fastapi.testclient")
    from fastapi import FastAPI

    from rag_textbook.api.app import get_user_id

    probe = FastAPI()

    @probe.get("/probe")
    def probe_endpoint(user: str = __import__("fastapi").Depends(get_user_id)) -> dict:
        return {"user": user}

    client = fastapi_testclient.TestClient(probe)
    assert client.get("/probe").status_code == 401
    assert client.get("/probe", headers={"X-User-Id": "student-1"}).json() == {"user": "student-1"}
