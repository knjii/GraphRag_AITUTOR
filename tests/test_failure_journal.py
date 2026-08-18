"""Журнал отказов извлечения.

Разбор 37 фрагментов, на которых извлечение падает всегда, упёрся в то, что
причина не сохраняется нигде: результат отката к правилам намеренно
не кэшируется, и отказавшие фрагменты в кэше просто отсутствуют. Журнал
закрывает разрыв, не трогая кэш.

Проверяется главное свойство: журнал пишется при откате и **не** пишется,
когда модель ответила по существу. Журнал, пишущий на каждый вызов, был бы
бесполезен — в нём утонули бы 1100 удачных фрагментов.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_textbook.config import GraphSettings
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.graph.failure_journal import JsonlFailureJournal
from rag_textbook.models import Chunk


class _Journal:
    def __init__(self) -> None:
        self.entries: list[dict] = []

    def record(self, entry: dict) -> None:
        self.entries.append(entry)


class _BrokenLLM:
    """Модель, которая всегда отвечает мусором."""

    def __init__(self, settings=None) -> None:
        self.settings = settings

    def complete(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "не JSON вовсе"

    def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return "не JSON вовсе"


class _WorkingLLM:
    def __init__(self, settings=None) -> None:
        self.settings = settings

    def _answer(self) -> str:
        return json.dumps(
            {
                "entities": [{"name": "матрица"}, {"name": "определитель"}],
                "relations": [
                    {"source": "матрица", "target": "определитель", "type": "HAS_PROPERTY"}
                ],
            },
            ensure_ascii=False,
        )

    def complete(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return self._answer()

    def chat(self, *args, **kwargs) -> str:  # noqa: ANN002, ANN003
        return self._answer()


def _chunk() -> Chunk:
    return Chunk(
        id="doc:00042",
        doc_id="doc",
        doc_name="Учебник",
        source_path="documents/pdf_docs/учебник.pdf",
        ordinal=42,
        text="Определитель матрицы равен произведению её собственных значений.",
        pages=[17],
        headers=["3.2. Определители"],
    )


def _settings() -> GraphSettings:
    return GraphSettings(GRAPH_EXTRACTION_RETRIES=0, GRAPH_EXTRACTION_CACHE_ENABLED=False)


def test_failure_is_recorded_with_reason():
    journal = _Journal()
    extractor = EntityExtractor(_settings(), llm=_BrokenLLM(), journal=journal)

    result = extractor.extract(_chunk(), "модель-для-теста")

    assert result.status == "rule_fallback"
    assert len(journal.entries) == 1, "отказ не записан, причина снова потеряна"
    entry = journal.entries[0]
    assert entry["chunk_id"] == "doc:00042"
    assert entry["status"] in {"invalid_json", "invalid_structure", "empty_response", "error"}
    assert entry["text_length"] > 0
    assert entry["pages"] == [17]


def test_successful_extraction_is_not_journalled():
    journal = _Journal()
    extractor = EntityExtractor(_settings(), llm=_WorkingLLM(), journal=journal)

    result = extractor.extract(_chunk(), "модель-для-теста")

    assert result.status != "rule_fallback"
    assert journal.entries == [], "журнал засорён удачными фрагментами"


def test_extractor_works_without_journal():
    """Журнал необязателен: без него извлечение обязано вести себя как раньше."""
    extractor = EntityExtractor(_settings(), llm=_BrokenLLM())

    result = extractor.extract(_chunk(), "модель-для-теста")

    assert result.status == "rule_fallback"


def test_jsonl_journal_appends_readable_lines(tmp_path: Path):
    path = tmp_path / "вложенный" / "failures.jsonl"
    journal = JsonlFailureJournal(path)

    journal.record({"chunk_id": "a", "status": "empty_response"})
    journal.record({"chunk_id": "b", "status": "invalid_json"})

    lines = path.read_text(encoding="utf-8").strip().split("\n")
    assert [json.loads(line)["chunk_id"] for line in lines] == ["a", "b"]


def test_journal_failure_does_not_break_indexing(tmp_path: Path):
    """Сбой записи журнала не должен ронять индексацию: он вспомогательный."""
    # Каталог вместо файла — запись гарантированно не удастся.
    path = tmp_path / "занято"
    path.mkdir()
    journal = JsonlFailureJournal(path)

    journal.record({"chunk_id": "a"})  # не должно бросить
