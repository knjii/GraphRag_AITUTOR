"""Индексация готовых чанков, минуя разбор PDF.

Путь нужен публичным бенчмаркам: их корпус раздаётся текстом. Проверяется
здесь прежде всего одно — куда ложатся чанки чужого корпуса. Если они лягут
рядом с чанками учебника, инструменты, которые ищут выгрузку маской
``*_chunks.json``, возьмут первый попавшийся файл, и аудит эталонного набора
посчитается по новостям при полном внешнем благополучии.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_textbook.config import Settings
from rag_textbook.indexing.pipeline import IndexingPipeline
from rag_textbook.models import Chunk
from rag_textbook.observability.monitor import NullMonitor


class _Pipeline:
    """Конвейер без тяжёлых зависимостей: проверяется только раскладка файлов."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.monitor = NullMonitor()
        self.embedded: list[str] = []
        self.graph_calls: list[str] = []

    def _embed_and_store(self, chunks):  # noqa: ANN001, ANN202
        self.embedded.extend(chunk.id for chunk in chunks)
        return len(chunks)

    def _build_graph(self, chunks, *, doc_id, doc_name, source_path):  # noqa: ANN001, ANN202
        self.graph_calls.append(doc_id)
        return {"status": "ok"}

    index_chunks = IndexingPipeline.index_chunks


def _chunk(identifier: str, doc_id: str) -> Chunk:
    return Chunk(
        id=identifier,
        doc_id=doc_id,
        doc_name="статья",
        source_path="multihop-rag/статья",
        ordinal=0,
        text="Текст фрагмента.",
    )


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    settings.paths.parsed_dir = tmp_path / "parsed"
    settings.paths.parsed_dir.mkdir(parents=True, exist_ok=True)
    return settings


def test_foreign_chunks_go_into_their_own_subdirectory(tmp_path: Path):
    settings = _settings(tmp_path)
    # Выгрузка учебника лежит на верхнем уровне — так её кладёт обычный разбор.
    (settings.paths.parsed_dir / "учебник_chunks.json").write_text("[]", encoding="utf-8")
    pipeline = _Pipeline(settings)

    pipeline.index_chunks(
        [_chunk("a:0", "a"), _chunk("b:0", "b")], source_label="multihop-rag"
    )

    top_level = sorted(path.name for path in settings.paths.parsed_dir.glob("*_chunks.json"))
    assert top_level == ["учебник_chunks.json"], "чужой корпус не должен попадать наверх"
    nested = sorted(
        path.name for path in (settings.paths.parsed_dir / "multihop-rag").glob("*_chunks.json")
    )
    assert nested == ["a_chunks.json", "b_chunks.json"]


def test_chunks_are_grouped_by_document(tmp_path: Path):
    settings = _settings(tmp_path)
    pipeline = _Pipeline(settings)

    pipeline.index_chunks(
        [_chunk("a:0", "a"), _chunk("a:1", "a"), _chunk("b:0", "b")],
        source_label="набор",
    )

    payload = json.loads(
        (settings.paths.parsed_dir / "набор" / "a_chunks.json").read_text(encoding="utf-8")
    )
    assert [item["id"] for item in payload] == ["a:0", "a:1"]


def test_graph_can_be_skipped(tmp_path: Path):
    """Граф по чужому корпусу строится не всегда: базовая линия меряется
    без него, и лишние полчаса аренды на него тратить незачем."""
    pipeline = _Pipeline(_settings(tmp_path))

    result = pipeline.index_chunks([_chunk("a:0", "a")], source_label="набор", with_graph=False)

    assert pipeline.graph_calls == []
    assert result["граф"]["status"] == "skipped"
    assert result["векторизовано"] == 1


def test_empty_input_does_nothing(tmp_path: Path):
    pipeline = _Pipeline(_settings(tmp_path))

    assert pipeline.index_chunks([], source_label="набор") == {"чанков": 0}
    assert pipeline.embedded == []
