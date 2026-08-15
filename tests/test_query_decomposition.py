"""Разложение связывающего вопроса на подвопросы.

Основание — замер на полном корпусе. Из 118 эталонных фрагментов многошаговых
вопросов векторный канал приносит в пул кандидатов 103, а до финальной выдачи
доживают 68. Материал найден, но теряется при отборе: реранкер оценивает пару
«вопрос — фрагмент» целиком, и на двухшаговом вопросе оба нужных фрагмента
получают средний балл, потому что каждый отвечает лишь на половину.
"""

from __future__ import annotations

import json

from rag_textbook.clients.embeddings import FakeEmbeddingClient
from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.clients.reranker import RerankerClient
from rag_textbook.config import Settings
from rag_textbook.models import Chunk, content_hash
from rag_textbook.retrieval.pipeline import RetrievalPipeline
from rag_textbook.stores.vector_store import InMemoryVectorStore

CORPUS = [
    (0, "Сингулярное разложение матрицы раскладывает её на три множителя."),
    (1, "Метод главных компонент снижает размерность данных."),
    (2, "Ортогональные матрицы сохраняют длины векторов."),
    (3, "Определённый интеграл вычисляется по формуле Ньютона и Лейбница."),
]


class _DecomposingLLM(FakeLLMClient):
    """Возвращает заданное разложение; считает обращения."""

    def __init__(self, parts: list[str] | None, raw: str | None = None) -> None:
        super().__init__()
        self._parts = parts
        self._raw = raw
        self.purposes: list[str] = []

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        self.purposes.append(purpose)
        if self._raw is not None:
            return self._raw
        if self._parts is not None:
            return json.dumps({"parts": self._parts}, ensure_ascii=False)
        return super().chat(messages, purpose=purpose, json_schema=json_schema)


class _PartAwareReranker(RerankerClient):
    """Реранкер, различающий подвопросы.

    Высоко оценивает фрагмент, начало которого дословно совпадает с запросом.
    Это моделирует существенное свойство настоящего реранкера: он оценивает
    соответствие фрагмента **запросу**, а не вопросу вообще. Совпадение
    по отдельным словам для этого не годится — общий термин вроде «матрицы»
    встречается почти во всех фрагментах и делает оценки неразличимыми.
    """

    def rerank(self, query: str, documents, top_n: int):
        scored = [
            (index, 1.0 if text.startswith(query) else 0.1)
            for index, text in enumerate(documents)
        ]
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_n]


def _chunks() -> list[Chunk]:
    return [
        Chunk(
            id=f"d:{ordinal:05d}",
            doc_id="d",
            doc_name="Учебник",
            source_path="/book.pdf",
            ordinal=ordinal,
            text=text,
            pages=[ordinal + 1],
            text_hash=content_hash(text),
        )
        for ordinal, text in CORPUS
    ]


def _pipeline(settings: Settings, llm) -> RetrievalPipeline:
    store = InMemoryVectorStore()
    store.ensure_collection(64)
    embeddings = FakeEmbeddingClient(dimensions=64)
    chunks = _chunks()
    store.upsert(chunks, embeddings.embed_documents([c.text for c in chunks]))
    return RetrievalPipeline(
        settings=settings,
        vector_store=store,
        embedding_client=embeddings,
        reranker=_PartAwareReranker(),
        graph_retriever=None,
        llm=llm,
    )


def _settings(**retrieval) -> Settings:
    base = {"top_k": 4, "router_mode": "always", "decompose_enabled": True}
    base.update(retrieval)
    return Settings(
        embedding={"provider": "fake", "dimensions": 64, "cache_enabled": False},
        llm={"provider": "fake"},
        reranker={"provider": "fake", "enabled": True, "candidates": 10, "top_n": 4},
        vector_store={"provider": "memory"},
        graph={"enabled": False, "retrieval_enabled": False},
        retrieval=base,
    )


def test_decomposition_splits_the_question() -> None:
    llm = _DecomposingLLM(["Что такое сингулярное разложение?", "Что такое метод главных компонент?"])
    pipeline = _pipeline(_settings(), llm)

    result = pipeline.retrieve("Как связаны сингулярное разложение и метод главных компонент?")

    assert len(result.sub_questions) == 2
    assert "utility" in llm.purposes, "служебный вызов должен идти с выключенным размышлением"


def test_each_part_contributes_its_own_material() -> None:
    """Оба нужных фрагмента должны дойти до выдачи, а не один из двух."""
    llm = _DecomposingLLM(["Сингулярное разложение матрицы", "Метод главных компонент снижает"])
    pipeline = _pipeline(_settings(), llm)

    result = pipeline.retrieve("Как связаны сингулярное разложение и метод главных компонент?")
    ids = [item.chunk.id for item in result.chunks]

    assert "d:00000" in ids
    assert "d:00001" in ids


def test_best_of_each_part_survives_even_if_irrelevant_to_the_other() -> None:
    """Ключевое свойство объединения, ради которого оно и переписано.

    Фрагмент, отвечающий на первую половину вопроса и не имеющий отношения
    ко второй, обязан попасть в выдачу. Слияние рангов его топило: сумма
    рангов по обоим подвопросам ставила его в середину списка.
    """
    llm = _DecomposingLLM(["Ортогональные матрицы сохраняют", "Определённый интеграл вычисляется"])
    pipeline = _pipeline(_settings(), llm)

    result = pipeline.retrieve("Как связаны ортогональные матрицы и определённый интеграл?")
    ids = [item.chunk.id for item in result.chunks]

    assert ids[:2] == ["d:00002", "d:00003"] or ids[:2] == ["d:00003", "d:00002"], (
        f"лучшие фрагменты обоих подвопросов должны стоять первыми, получено {ids}"
    )


def test_disabled_by_default() -> None:
    """Изменение поведения поиска не должно включаться само собой."""
    assert Settings().retrieval.decompose_enabled is False


def test_single_part_falls_back_to_the_normal_path() -> None:
    """Неделимый вопрос обрабатывается как обычно, без лишней работы."""
    llm = _DecomposingLLM(["Что такое сингулярное разложение?"])
    pipeline = _pipeline(_settings(), llm)

    result = pipeline.retrieve("Что такое сингулярное разложение?")
    assert result.sub_questions == []
    assert result.chunks


def test_broken_answer_does_not_break_retrieval() -> None:
    """Поиск обязан работать, даже если разложение вернуло мусор."""
    llm = _DecomposingLLM(None, raw="это не json")
    pipeline = _pipeline(_settings(), llm)

    result = pipeline.retrieve("Как связаны SVD и PCA?")
    assert result.sub_questions == []
    assert result.chunks, "выдача не должна опустеть из-за отказа разложения"


def test_code_fence_is_tolerated() -> None:
    """Модели часто оборачивают JSON в ```json — это не повод терять разложение."""
    llm = _DecomposingLLM(None, raw='```json\n{"parts": ["первый вопрос", "второй вопрос"]}\n```')
    pipeline = _pipeline(_settings(), llm)

    assert pipeline.decompose_question("Как связаны A и B?") == ["первый вопрос", "второй вопрос"]


def test_parts_are_capped() -> None:
    llm = _DecomposingLLM(["один", "два", "три", "четыре"])
    pipeline = _pipeline(_settings(decompose_max_parts=2), llm)

    assert pipeline.decompose_question("сложный вопрос") == ["один", "два"]


def test_no_llm_means_no_decomposition() -> None:
    pipeline = _pipeline(_settings(), llm=None)
    assert pipeline.decompose_question("Как связаны A и B?") == []


def test_decomposition_is_skipped_for_simple_questions() -> None:
    """На вопросах, не признанных связывающими, вызов модели не тратится."""
    llm = _DecomposingLLM(["первый", "второй"])
    pipeline = _pipeline(_settings(router_mode="never"), llm)

    result = pipeline.retrieve("Что такое матрица?")
    assert result.sub_questions == []
    assert "utility" not in llm.purposes, "разложение не должно вызываться"
