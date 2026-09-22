"""Извлечение v4: роли упоминаний (К2) и обозначения (К3).

Версия v3 обязана остаться прежней: графы разных версий строятся каждый
своим промптом, и старый граф должен воспроизводиться.
"""

from __future__ import annotations

import json

from rag_textbook.config import GraphSettings
from rag_textbook.graph.builder import GraphBuilder
from rag_textbook.graph.extractor import (
    EXTRACTION_SCHEMA,
    EXTRACTION_SCHEMA_V4,
    NOTATION_LABEL,
    PROMPT_TEMPLATE,
    EntityExtractor,
)
from rag_textbook.models import Chunk

V4_ANSWER = {
    "entities": [
        {"name": "матрица", "role": "uses"},
        {"name": "определитель", "role": "defines"},
        {"name": "определитель", "role": "mentions"},
        {"name": "перестановка", "role": "непонятно"},
    ],
    "relations": [{"source": "определитель", "relation": "определяется_через", "target": "матрица"}],
    "notation": [
        {"symbol": "$\\det A$", "meaning": "определитель"},
        {"symbol": "S_n", "meaning": "симметрическая группа"},
        {"symbol": "", "meaning": "пустое"},
    ],
}


class _LLM:
    def __init__(self, answer: dict) -> None:
        self.answer = answer
        self.prompts: list[str] = []
        self.schemas: list[dict] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.prompts.append(messages[0].content)
        self.schemas.append(kwargs.get("json_schema"))
        return json.dumps(self.answer, ensure_ascii=False)


def _chunk() -> Chunk:
    return Chunk(
        id="doc:00001",
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=1,
        text="Определителем матрицы A назовём число; обозначим его через $\\det A$.",
    )


def _extract(version: str, answer: dict):
    llm = _LLM(answer)
    settings = GraphSettings(extraction_cache_enabled=False, extraction_prompt_version=version)
    result = EntityExtractor(settings, llm=llm).extract(_chunk(), "модель")
    return result, llm


def test_v3_prompt_and_schema_are_unchanged() -> None:
    result, llm = _extract("v3", {"entities": [{"name": "матрица"}], "relations": []})
    assert llm.schemas == [EXTRACTION_SCHEMA]
    assert "notation" not in llm.prompts[0]
    assert llm.prompts[0].startswith(PROMPT_TEMPLATE.split("{")[0])
    assert [entity.role for entity in result.entities] == [""]
    assert [entity.kind for entity in result.entities] == ["concept"]


def test_v4_reads_roles_and_keeps_the_strongest() -> None:
    result, llm = _extract("v4", V4_ANSWER)
    assert llm.schemas == [EXTRACTION_SCHEMA_V4]
    roles = {entity.name: entity.role for entity in result.entities if entity.kind == "concept"}
    assert roles["матрица"] == "uses"
    # Названо дважды — остаётся определение.
    assert roles["определитель"] == "defines"
    # Роль вне списка не выдумывается.
    assert roles["перестановка"] == ""


def test_v4_notation_is_symbol_meaning_pair() -> None:
    result, _ = _extract("v4", V4_ANSWER)
    notations = [entity for entity in result.entities if entity.kind == "notation"]
    assert [entity.name for entity in notations] == ["\\det A", "S_n"]
    assert all(entity.role == "defines" for entity in notations)
    # Понятие, которого не было среди сущностей, добавлено с ролью mentions.
    group = [entity for entity in result.entities if entity.name == "симметрическая группа"]
    assert group and group[0].role == "mentions"
    edges = [relation for relation in result.relations if relation.label == NOTATION_LABEL]
    by_id = {entity.id: entity for entity in result.entities}
    assert {(by_id[edge.source_id].name, by_id[edge.target_id].name) for edge in edges} == {
        ("\\det A", "определитель"),
        ("S_n", "симметрическая группа"),
    }


def test_same_symbol_with_other_meaning_is_another_node() -> None:
    first, _ = _extract("v4", {"entities": [], "relations": [], "notation": [{"symbol": "x", "meaning": "вектор признаков"}]})
    second, _ = _extract("v4", {"entities": [], "relations": [], "notation": [{"symbol": "x", "meaning": "случайная величина"}]})
    ids = {entity.id for result in (first, second) for entity in result.entities if entity.kind == "notation"}
    assert len(ids) == 2


def test_builder_passes_roles_to_mentions() -> None:
    llm = _LLM(V4_ANSWER)
    settings = GraphSettings(
        extraction_cache_enabled=False,
        extraction_prompt_version="v4",
        cross_chunk_relations_enabled=False,
    )
    class _Store:
        def __init__(self) -> None:
            self.mentions: list[dict] = []
            self.entities: list = []

        def upsert_mentions(self, rows):  # noqa: ANN001
            self.mentions.extend(rows)

        def upsert_entities(self, rows):  # noqa: ANN001
            self.entities.extend(rows)

        def __getattr__(self, name):  # noqa: ANN001
            return lambda *args, **kwargs: 0

    store = _Store()
    builder = GraphBuilder(settings, EntityExtractor(settings, llm=llm), store=store)  # type: ignore[arg-type]
    builder.build([_chunk()], doc_id="doc", doc_name="Учебник", source_path="", write=True)
    names = {entity.id: entity.name for entity in store.entities}
    roles = {names[row["entity_id"]]: row["role"] for row in store.mentions}
    assert roles["определитель"] == "defines"
    assert roles["матрица"] == "uses"
    assert roles["\\det A"] == "defines"
    kinds = {entity.name: entity.kind for entity in store.entities}
    assert kinds["\\det A"] == "notation" and kinds["матрица"] == "concept"


def test_default_limit_fits_v4_worst_case() -> None:
    """Предел ответа вмещает худший ответ v4 с отступами (см. test_extraction_token_limit)."""
    settings = GraphSettings()
    name = "довольно длинное имя сущности"
    answer = {
        "entities": [{"name": f"{name} {i}", "role": "mentions"} for i in range(settings.max_entities_per_chunk)],
        "relations": [
            {"source": f"{name} {i}", "target": f"{name} {i + 1}", "relation": "используется_в"}
            for i in range(settings.max_relations_per_chunk)
        ],
        "notation": [
            {"symbol": "\\mathbf{x}_{i}", "meaning": f"{name} {i}"}
            for i in range(settings.max_entities_per_chunk // 2)
        ],
    }
    estimated = len(json.dumps(answer, ensure_ascii=False, indent=2)) / 2.5
    assert settings.extraction_max_tokens > estimated * 1.2
