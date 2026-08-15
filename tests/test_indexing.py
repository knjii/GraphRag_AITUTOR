"""Тесты конвейера индексации: возобновляемость, идемпотентность, кэши."""

from __future__ import annotations

import json

from rag_textbook.context import build_context
from rag_textbook.indexing.manifest import IndexingManifest
from rag_textbook.indexing.pipeline import IndexingPipeline, document_id


def _write_pdf_stub(path, content: bytes = b"%PDF-1.4 stub") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


# ---------------------------------------------------------------- манифест


def test_manifest_tracks_stages(tmp_path) -> None:
    manifest = IndexingManifest(tmp_path / "m.json")
    state = manifest.get("doc1", "/a.pdf", "fp1")

    assert not state.is_done("parsed")
    state.mark("parsed")
    manifest.save()

    reloaded = IndexingManifest(tmp_path / "m.json")
    assert reloaded.get("doc1", "/a.pdf", "fp1").is_done("parsed"), (
        "прогресс должен переживать перезапуск"
    )


def test_manifest_resets_stages_when_file_changes(tmp_path) -> None:
    manifest = IndexingManifest(tmp_path / "m.json")
    state = manifest.get("doc1", "/a.pdf", "fingerprint-1")
    state.mark("parsed")
    state.mark("embedded")

    updated = manifest.get("doc1", "/a.pdf", "fingerprint-2")
    assert not updated.is_done("parsed"), "изменённый документ надо переиндексировать"
    assert not updated.is_done("embedded")


def test_manifest_survives_corrupted_file(tmp_path) -> None:
    path = tmp_path / "m.json"
    path.write_text("{ это не json", encoding="utf-8")
    manifest = IndexingManifest(path)
    assert manifest.documents == {}, "повреждённый манифест не должен ронять индексацию"


def test_document_id_is_stable_across_paths(tmp_path) -> None:
    """Идентификатор считается от имени файла.

    Иначе перенос корпуса на арендованный сервер сменил бы все идентификаторы
    и превратил обновление индекса в полную переиндексацию.
    """
    from pathlib import Path

    assert document_id(Path("/home/user/corpus/linal.pdf")) == document_id(
        Path("D:/other/place/linal.pdf")
    )


# ------------------------------------------------------------- сквозной путь


def test_indexing_pipeline_is_resumable_and_idempotent(
    settings, monkeypatch, tmp_path, sample_blocks
) -> None:
    """Полный путь индексации на заглушках, без MinerU и внешних сервисов."""
    pdf_path = tmp_path / "corpus" / "linal.pdf"
    _write_pdf_stub(pdf_path)
    settings.paths.pdf_dir = pdf_path.parent

    parse_calls = {"count": 0}

    def fake_parse(self, path, force: bool = False):
        parse_calls["count"] += 1
        return list(sample_blocks)

    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.parse", fake_parse, raising=True
    )
    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.images_dir_for",
        lambda self, path: None,
        raising=True,
    )

    context = build_context(settings)
    try:
        pipeline = IndexingPipeline(context)

        first = pipeline.run()
        assert first.failed == 0
        assert first.total_chunks > 0
        chunks_after_first = context.vector_store.count()
        assert chunks_after_first > 0

        # Повторный запуск: стадии уже отмечены, работа не делается заново.
        second = pipeline.run()
        assert second.total_chunks == first.total_chunks
        assert context.vector_store.count() == chunks_after_first, (
            "повторная индексация не должна плодить дубликаты"
        )
        assert second.documents[0].stage_seconds == {}, "стадии должны пропускаться"
    finally:
        context.close()


def test_indexing_report_is_saved(settings, monkeypatch, tmp_path, sample_blocks) -> None:
    pdf_path = tmp_path / "corpus" / "book.pdf"
    _write_pdf_stub(pdf_path)
    settings.paths.pdf_dir = pdf_path.parent

    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.parse",
        lambda self, path, force=False: list(sample_blocks),
        raising=True,
    )
    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.images_dir_for",
        lambda self, path: None,
        raising=True,
    )

    context = build_context(settings)
    try:
        IndexingPipeline(context).run()
    finally:
        context.close()

    reports = list(settings.paths.metrics_dir.glob("indexing_*.json"))
    assert reports, "отчёт индексации должен сохраняться для сравнения прогонов"
    payload = json.loads(reports[0].read_text(encoding="utf-8"))
    # Конфигурация внутри отчёта — иначе прогоны невозможно честно сравнивать.
    assert "config" in payload and payload["config"]["chunk_size"] > 0
    assert payload["documents"][0]["chunks"] > 0


def test_duplicate_filenames_are_indexed_once(
    settings, monkeypatch, tmp_path, sample_blocks
) -> None:
    """В корпусе лежат копии одного учебника в подкаталогах test/, test2/, test_prev/."""
    root = tmp_path / "corpus"
    for subdir in ("", "test", "test_prev"):
        _write_pdf_stub(root / subdir / "linal.pdf")
    settings.paths.pdf_dir = root

    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.parse",
        lambda self, path, force=False: list(sample_blocks),
        raising=True,
    )
    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.images_dir_for",
        lambda self, path: None,
        raising=True,
    )

    context = build_context(settings)
    try:
        pipeline = IndexingPipeline(context)
        discovered = pipeline.discover_documents()
    finally:
        context.close()

    assert len(discovered) == 1, "копии одного файла не должны индексироваться повторно"


def test_health_reports_component_status(settings) -> None:
    context = build_context(settings)
    try:
        report = context.health()
    finally:
        context.close()

    assert report["status"] in {"ok", "degraded"}
    assert report["components"]["vector_store"]["status"] == "ok"
    assert report["components"]["embeddings"]["status"] == "ok"
    assert report["components"]["graph"]["status"] == "disabled"
