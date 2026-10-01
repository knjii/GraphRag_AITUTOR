"""Обязательная квота мест за графом должна действительно действовать.

История дефекта. Настройка `RETRIEVAL_MIN_GRAPH_DOCS` существовала, была
описана в `.env.example` и участвовала в A/B-сравнении. При переключении с 0
на 3 не менялся **ни один** вопрос из 163 — и это выглядело как честный вывод
«эффекта нет», а не как поломка.

Причина: реранкер возвращал ровно `top_k` элементов, а функция, которая
подставляет графовые фрагменты, берёт замену из хвоста списка за `top_k`.
Хвост был пуст всегда, поэтому подставить было нечего.

Тест проверяет не наличие настройки, а её следствие: фрагмент, который нашёл
только граф и который реранкер поставил ниже границы выдачи, обязан попасть
в контекст, вытеснив худший неграфовый.
"""

from __future__ import annotations

from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.fusion import enforce_minimum_graph_documents


def _chunk(index: int, *, from_graph: bool) -> ScoredChunk:
    chunk = Chunk(
        id=f"doc:{index:05d}",
        doc_id="doc",
        doc_name="doc",
        source_path="doc.pdf",
        ordinal=index,
        text=f"фрагмент {index}",
        pages=[index],
    )
    channels = ["graph_entity"] if from_graph else ["dense"]
    return ScoredChunk(
        chunk=chunk,
        score=1.0 / (index + 1),
        channels=channels,
        channel_scores={channels[0]: 1.0 / (index + 1)},
    )


def test_quota_needs_a_tail_to_draw_from():
    """Без запаса за границей выдачи подставлять нечего — корень дефекта."""
    exactly_top_k = [_chunk(i, from_graph=False) for i in range(8)]

    result = enforce_minimum_graph_documents(exactly_top_k, minimum=3, top_k=8)

    # Функция отработала честно и ничего не смогла сделать: списка за top_k нет.
    assert [item.chunk.id for item in result] == [item.chunk.id for item in exactly_top_k]


def test_graph_chunk_below_cutoff_reaches_context():
    """Тот же вызов, но с запасом: графовый фрагмент обязан подняться."""
    items = [_chunk(i, from_graph=False) for i in range(8)]
    items += [_chunk(100 + i, from_graph=True) for i in range(3)]

    result = enforce_minimum_graph_documents(items, minimum=2, top_k=8)

    assert len(result) == 8
    graph_ids = [item.chunk.id for item in result if item.from_graph]
    assert len(graph_ids) == 2, "квота не выполнена"
    # Вытесняются худшие неграфовые, лучшие остаются.
    assert "doc:00000" in [item.chunk.id for item in result]
    assert "doc:00007" not in [item.chunk.id for item in result]


def test_rerank_width_leaves_room_for_the_quota(settings):
    """Ширина выдачи реранкера должна расти вместе с квотой."""
    from rag_textbook.retrieval.pipeline import RetrievalPipeline

    settings.retrieval.min_graph_docs = 0
    settings.reranker.top_n = 8
    pipeline = RetrievalPipeline.__new__(RetrievalPipeline)
    pipeline.settings = settings

    without_quota = pipeline._rerank_width(top_k=8)

    settings.retrieval.min_graph_docs = 3
    with_quota = pipeline._rerank_width(top_k=8)

    assert without_quota == 8
    assert with_quota == 11, "нет запаса — квоте неоткуда брать замену"


def test_quota_does_not_disturb_output_when_disabled():
    """При выключенной квоте порядок и состав выдачи не меняются."""
    items = [_chunk(i, from_graph=i >= 8) for i in range(12)]

    result = enforce_minimum_graph_documents(items, minimum=0, top_k=8)

    assert [item.chunk.id for item in result] == [f"doc:{i:05d}" for i in range(8)]
