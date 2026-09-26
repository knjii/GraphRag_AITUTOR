"""Стадия, запущенная отдельно, должна получать входные данные.

История дефекта. Команда `ingest --stages graph --force` собирала граф из нуля
фрагментов. Ошибки не было: прогон завершался успешно, счётчики показывали нули,
а метрики поиска выходили правдоподобными — ровно такими, как у конфигурации
без графа. Из-за этого шаг эксперимента «порог отсечения хабов 64 против 40»
выдал одинаковый результат для обоих порогов, и вывод «разницы нет» выглядел
как измерение.

Причина: чтение готовых чанков с диска было под условием `not force`.
Флаг `force` означает «переделать выбранные стадии», а не «остаться без входа»,
и стадия чанкинга при `--stages graph` не выбрана.
"""

from __future__ import annotations

import pytest

from rag_textbook.indexing.pipeline import IndexingPipeline


@pytest.fixture
def indexed(settings, sample_blocks, monkeypatch):
    """Готовый разбор и чанки на диске, как после прошлой сессии."""
    from rag_textbook.context import build_context

    settings.graph.enabled = True
    settings.graph.extractor = "rule"
    context = build_context(settings)
    pipeline = IndexingPipeline(context)

    monkeypatch.setattr(pipeline.parser, "parse", lambda path, force=False: sample_blocks)
    monkeypatch.setattr(
        pipeline.parser, "images_dir_for", lambda path: settings.paths.parsed_dir
    )
    source = settings.paths.pdf_dir / "учебник.pdf"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"%PDF-1.4 stub")

    report = pipeline.run(sources=[source])
    assert report.total_chunks > 0, "подготовка не удалась: чанков нет"
    yield pipeline, source, report.total_chunks
    context.close()


def test_graph_stage_alone_reads_ready_chunks(indexed):
    """Главная проверка: стадия графа отдельно и с --force видит чанки."""
    pipeline, source, expected = indexed

    report = pipeline.run(sources=[source], stages=["graph"], force=True)

    document = report.documents[0]
    # Суть проверки — что на вход стадии пришли чанки. Собрался ли граф,
    # зависит от доступности Neo4j и здесь не проверяется.
    assert document.chunks == expected, (
        "стадия графа осталась без чанков: именно так собирался пустой граф"
    )


def test_embedding_stage_alone_reads_ready_chunks(indexed):
    """Та же ловушка ждала бы и векторизацию."""
    pipeline, source, expected = indexed

    report = pipeline.run(sources=[source], stages=["embed"], force=True)

    assert report.documents[0].chunks == expected


def test_missing_chunks_are_reported_not_swallowed(indexed, caplog):
    """Если чанков действительно нет, это должно быть видно в логе.

    Прежде такой прогон завершался успешно с нулями во всех счётчиках,
    и отличить его от честного «графа нет» было нельзя.
    """
    import logging

    pipeline, source, _ = indexed
    # Файл с чанками убираем: читать нечего, а отметка о стадии остаётся —
    # ровно то состояние, в котором конвейер молча собирал пустой граф.
    for path in pipeline.settings.paths.parsed_dir.glob("*_chunks.json"):
        path.unlink()

    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="rag.indexing.pipeline"):
        report = pipeline.run(sources=[source], stages=["graph"], force=True)

    assert report.documents[0].chunks == 0
    messages = [record.getMessage() for record in caplog.records]
    assert any("нет чанков на входе" in message for message in messages), (
        f"пустой вход прошёл молча, в логе: {messages}"
    )


def test_chunk_stage_alone_reads_parsed_blocks(indexed, sample_blocks, monkeypatch):
    """`ingest --stages chunk` без parse берёт готовый разбор с диска.

    День 2, 2026-09-22: после сброса отметок нарезки все 15 книг русского
    блока нарезались в ноль — блоки читала только стадия разбора, а она не
    была выбрана. Книги отметились нарезанными, проверка коллекции прошла.
    """
    pipeline, source, expected = indexed
    monkeypatch.setattr(pipeline.parser, "load_cached", lambda path: sample_blocks)

    report = pipeline.run(sources=[source], stages=["chunk"], force=True)

    assert report.documents[0].chunks == expected


def test_chunk_stage_without_parsed_blocks_fails_loudly(indexed, monkeypatch):
    pipeline, source, _ = indexed
    monkeypatch.setattr(pipeline.parser, "load_cached", lambda path: None)

    report = pipeline.run(sources=[source], stages=["chunk"], force=True)

    document = report.documents[0]
    assert document.status == "failed"
    assert "Нет блоков разбора" in (document.error or "")
