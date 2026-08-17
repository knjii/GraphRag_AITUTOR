"""Две настройки обхода, добавленные по итогам офлайн-замера.

Затухание веса соседа прежде было зашито в код как 1/(1+расстояние), а вклад
сущности не зависел от её редкости. Замер второго шага показал, что оба решения
стоят качества, поэтому оба вынесены в настройки. Тесты закрепляют не числа
замера, а поведение: настройка действительно доходит до хранилища и меняет
результат в ту сторону, ради которой добавлена.
"""

from __future__ import annotations

from rag_textbook.config import GraphSettings
from rag_textbook.models import Chunk, Entity, Relation
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from tests.fakes import InMemoryGraphStore


def _entity(name: str) -> Entity:
    return Entity(id=Entity.make_id(name), name=name, canonical=name, aliases=[], count=1)


def _chunk(index: int, text: str) -> Chunk:
    return Chunk(
        id=f"doc:{index:05d}",
        doc_id="doc",
        doc_name="doc",
        source_path="doc.pdf",
        ordinal=index,
        text=text,
        pages=[index],
    )


def _store() -> InMemoryGraphStore:
    """Два фрагмента: во втором нет слов запроса, дойти можно только по связи."""
    store = InMemoryGraphStore()
    seed, far = _entity("сингулярное разложение"), _entity("метод главных компонент")
    store.upsert_entities([seed, far])
    store.upsert_passages(
        [
            _chunk(0, "Сингулярное разложение матрицы и его свойства."),
            _chunk(1, "Понижение размерности опирается на собственные векторы ковариации."),
        ]
    )
    store.upsert_mentions(
        [
            {"chunk_id": "doc:00000", "entity_id": seed.id, "doc_id": "doc", "count": 3},
            {"chunk_id": "doc:00001", "entity_id": far.id, "doc_id": "doc", "count": 3},
        ]
    )
    store.upsert_relations(
        [
            Relation(
                source_id=seed.id,
                target_id=far.id,
                label="используется_в",
                chunk_id="doc:00000",
                doc_id="doc",
                weight=1.0,
            )
        ]
    )
    return store


def _settings(**overrides) -> GraphSettings:
    base = {
        "enabled": True,
        "retrieval_enabled": True,
        "seed_mode": "query",
        "expansion_hops": 1,
        "seed_entity_limit": 10,
        "passage_limit": 10,
    }
    base.update(overrides)
    return GraphSettings(**base)


def test_hop_decay_controls_weight_of_neighbours():
    store = _store()
    question = "Что такое сингулярное разложение?"

    weak = GraphRetriever(_settings(hop_decay=0.1), store).retrieve(question)
    strong = GraphRetriever(_settings(hop_decay=1.0), store).retrieve(question)

    def score_of(results, chunk_id):
        return next((item.score for item in results if item.chunk.id == chunk_id), 0.0)

    # Соседний фрагмент достижим в обоих случаях, но весит по-разному.
    assert score_of(weak, "doc:00001") < score_of(strong, "doc:00001")


def test_zero_decay_turns_expansion_off():
    """Нужно для A/B: без расширения канал должен отдавать только своё."""
    store = _store()
    question = "Что такое сингулярное разложение?"

    results = GraphRetriever(_settings(hop_decay=0.0), store).retrieve(question)

    assert [item.chunk.id for item in results] == ["doc:00000"]


def test_idf_toggle_reaches_the_store():
    store = _store()
    question = "Что такое сингулярное разложение?"
    seen: list[bool] = []
    original = store.find_passages

    def spy(entity_weights, limit, use_idf=True):
        seen.append(use_idf)
        return original(entity_weights, limit, use_idf)

    store.find_passages = spy  # type: ignore[method-assign]

    GraphRetriever(_settings(passage_idf_enabled=False), store).retrieve(question)
    GraphRetriever(_settings(passage_idf_enabled=True), store).retrieve(question)

    assert seen == [False, True]


def test_defaults_match_what_was_measured_on_the_product():
    """Умолчания должны отражать проверку на продукте, а не офлайн-замер.

    Все три значения сначала были подобраны офлайн-харнессом второго шага.
    Проверка полным прогоном отклонила порог 40 — он оказался значимо хуже 64
    (recall −0.012, у связывающих вопросов −0.029, четыре вопроса хуже и ни
    одного лучше). Затухание 0.8 и вес по редкости эффекта не дали.

    Тест закрепляет именно это: молчаливый возврат к офлайн-значениям означал
    бы ухудшение качества, подкреплённое красивым, но неприменимым замером.
    """
    settings = GraphSettings()

    assert settings.max_entity_degree == 64
    # Затухание и вес редкости остаются настраиваемыми: на продукте они
    # нейтральны, но нужны для A/B и для корпусов, где картина может отличаться.
    assert 0.0 <= settings.hop_decay <= 1.0
    assert isinstance(settings.passage_idf_enabled, bool)
