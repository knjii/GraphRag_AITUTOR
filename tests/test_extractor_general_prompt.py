"""Промпт en: английский текст, свой закрытый список меток, свой ключ кэша."""

import json

from rag_textbook.config import GraphSettings
from rag_textbook.graph.extractor import (
    PROMPT_TEMPLATE_EN,
    RELATION_LABELS_EN,
    EntityExtractor,
    _build_prompt,
    _normalize_label,
)
from rag_textbook.models import Chunk


class _FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.prompts = []

    def chat(self, messages, **kwargs):
        self.prompts.append(messages[0].content)
        return json.dumps(self.payload)


def _settings(version):
    return GraphSettings(GRAPH_EXTRACTION_PROMPT_VERSION=version)


def _chunk():
    return Chunk(id="c1", doc_id="d1", doc_name="WFBG", source_path="", ordinal=0,
                 text="WFBG is a radio station in Altoona, Pennsylvania.")


def test_prompt_en_is_english_with_en_labels():
    prompt = _build_prompt("text", _settings("en"))
    assert prompt.startswith("You extract a knowledge graph")
    assert all(label in prompt for label in RELATION_LABELS_EN)
    assert "определяется_через" not in prompt


def test_v4_prompt_unchanged():
    assert _build_prompt("text", _settings("v4")).startswith("Ты извлекаешь")


def test_normalize_label_en():
    assert _normalize_label("located in", RELATION_LABELS_EN) == "located_in"
    assert _normalize_label("founded", RELATION_LABELS_EN) == "related_to"
    assert _normalize_label("что-то") == "используется_в"


def test_extract_en_keeps_relations():
    llm = _FakeLLM({"entities": [{"name": "WFBG"}, {"name": "Altoona"}],
                    "relations": [{"source": "WFBG", "relation": "located_in", "target": "Altoona"}]})
    result = EntityExtractor(_settings("en"), llm=llm).extract_llm(_chunk())
    assert result.status == "ok"
    assert [r.label for r in result.relations] == ["located_in"]


def test_cache_key_differs_by_version():
    chunk = _chunk()
    en = EntityExtractor(_settings("en"))._cache_key(chunk, "m")
    v4 = EntityExtractor(_settings("v4"))._cache_key(chunk, "m")
    assert en != v4
    assert PROMPT_TEMPLATE_EN  # текст промпта входит в ключ
