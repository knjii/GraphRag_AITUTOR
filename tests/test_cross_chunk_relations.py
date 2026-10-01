"""Связи, извлекаемые сопоставлением нескольких фрагментов.

Корневая причина того, что графовый канал не давал прироста: связи
извлекались по одному фрагменту за раз, поэтому каждое ребро ``RELATES``
соединяло сущности, встретившиеся в одном и том же тексте. Обход такого
графа приводит туда же, куда лексический поиск, — измерено, что его
исключительный вклад в контекст равен нулю.

Здесь проверяется, что появился источник рёбер другой природы: связь
следует из сопоставления выдержек из разных мест книги.
"""

from __future__ import annotations

import json

from rag_textbook.clients.llm import FakeLLMClient
from rag_textbook.config import GraphSettings
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.models import Chunk, content_hash

CORPUS = [
    (0, "Сингулярное разложение раскладывает матрицу на три множителя."),
    (30, "Сингулярное разложение применяется для сжатия изображений."),
    (60, "Сингулярное разложение лежит в основе метода главных компонент."),
]


class _ScriptedLLM(FakeLLMClient):
    """Различает обычное извлечение и извлечение по нескольким фрагментам."""

    def __init__(self, cross_payload: str) -> None:
        super().__init__()
        self._cross = cross_payload
        self.cross_prompts: list[str] = []

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        prompt = messages[-1].content
        if "выдержки из разных разделов" in prompt:
            self.cross_prompts.append(prompt)
            return self._cross
        if json_schema is not None:
            names = ["сингулярное разложение", "матрица", "метод главных компонент"]
            return json.dumps(
                {"entities": [{"name": n} for n in names], "relations": []}, ensure_ascii=False
            )
        return super().chat(messages, purpose=purpose, json_schema=json_schema)


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


def _settings(**overrides) -> GraphSettings:
    defaults = {
        "enabled": True,
        "extractor": "llm",
        "extraction_cache_enabled": False,
        "cooccurrence_enabled": False,
        "max_entity_degree": 0,
        "cross_chunk_relations_enabled": True,
        "cross_chunk_min_chunks": 2,
    }
    defaults.update(overrides)
    return GraphSettings(**defaults)


CROSS_ANSWER = json.dumps(
    {"relations": [{"target": "метод главных компонент", "relation": "вычисляется_по"}]},
    ensure_ascii=False,
)


def _build(settings: GraphSettings, llm: _ScriptedLLM):
    builder = GraphBuilder(settings, EntityExtractor(settings, llm=llm), None, max_workers=1)
    return builder.build(
        _chunks(), doc_id="d", doc_name="Учебник", source_path="/book.pdf", write=False
    )


def test_cross_chunk_relations_are_produced() -> None:
    llm = _ScriptedLLM(CROSS_ANSWER)
    result = _build(_settings(), llm)

    assert result.cross_chunk_relations >= 1
    assert result.relations >= result.cross_chunk_relations


def test_excerpts_come_from_different_places() -> None:
    """Сопоставлять соседние фрагменты бессмысленно — они перекрываются."""
    llm = _ScriptedLLM(CROSS_ANSWER)
    _build(_settings(), llm)

    assert llm.cross_prompts, "извлечение по нескольким фрагментам не вызывалось"
    prompt = llm.cross_prompts[0]
    assert prompt.count("Фрагмент ") >= 2


def test_disabled_by_default() -> None:
    """Изменение стоимости индексации не должно включаться само собой."""
    assert GraphSettings(_env_file=None).cross_chunk_relations_enabled is False  # type: ignore[arg-type]


def test_rare_concepts_are_skipped() -> None:
    """Понятие из одного фрагмента сопоставлять не с чем."""
    llm = _ScriptedLLM(CROSS_ANSWER)
    result = _build(_settings(cross_chunk_min_chunks=50), llm)

    assert result.cross_chunk_relations == 0
    assert not llm.cross_prompts


def test_unknown_targets_are_dropped() -> None:
    """Связь с понятием, которого нет в графе, дала бы висячий узел."""
    llm = _ScriptedLLM(
        json.dumps({"relations": [{"target": "квантовая запутанность", "relation": "связан_с"}]})
    )
    result = _build(_settings(), llm)

    assert result.cross_chunk_relations == 0


def test_broken_answer_does_not_break_the_build() -> None:
    llm = _ScriptedLLM("это не json")
    result = _build(_settings(), llm)

    assert result.cross_chunk_relations == 0
    assert result.entities > 0, "остальная сборка графа должна пройти"


def test_relation_belongs_to_no_single_chunk() -> None:
    """Связь следует из нескольких фрагментов и ни одному не принадлежит."""
    settings = _settings()
    llm = _ScriptedLLM(CROSS_ANSWER)
    extractor = EntityExtractor(settings, llm=llm)

    relations = extractor.extract_cross_chunk(
        "сингулярное разложение",
        [text for _, text in CORPUS],
        ["сингулярное разложение", "метод главных компонент"],
    )

    assert relations
    assert all(relation.chunk_id == "" for relation in relations)
