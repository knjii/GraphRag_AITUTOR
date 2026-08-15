"""Тесты страховки по видеопамяти и режимов обхода конвейера."""

from __future__ import annotations

import pytest

from rag_textbook.config import IndexingSettings
from rag_textbook.context import build_context
from rag_textbook.indexing.pipeline import IndexingPipeline
from rag_textbook.indexing.resources import (
    VramGuard,
    is_out_of_memory,
    run_with_oom_backoff,
)

# --------------------------------------------------------------- распознавание


@pytest.mark.parametrize(
    "message",
    [
        "CUDA out of memory. Tried to allocate 2.00 GiB",
        "CUBLAS_STATUS_ALLOC_FAILED",
        "No KV cache space available",
        "RESOURCE_EXHAUSTED: failed to allocate",
    ],
)
def test_oom_messages_are_recognized(message: str) -> None:
    assert is_out_of_memory(RuntimeError(message))


def test_unrelated_errors_are_not_oom() -> None:
    assert not is_out_of_memory(ValueError("некорректная схема"))
    assert not is_out_of_memory(RuntimeError("status code: 503"))


# ------------------------------------------------------------------- страховка


def test_guard_is_transparent_when_disabled() -> None:
    guard = VramGuard(enabled=False)
    assert guard.wait_for_headroom(999_999, "parse") is True
    assert guard.batch_size("embed", 256, per_item_mib=100.0) == 256


def test_guard_survives_missing_nvidia_smi(monkeypatch) -> None:
    """Без nvidia-smi защита обязана вырождаться в пустую операцию, а не падать."""
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: None)
    guard = VramGuard(enabled=True)
    assert guard.wait_for_headroom(8000, "parse") is True
    assert guard.batch_size("embed", 128, per_item_mib=10.0) == 128


def test_batch_size_shrinks_when_memory_is_tight(monkeypatch) -> None:
    # Свободно 2048 МиБ, резерв 1536 → на батч остаётся 512 МиБ.
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: (2048, 24576))
    guard = VramGuard(min_free_mib=1536, enabled=True)
    assert guard.batch_size("embed", 256, per_item_mib=8.0) == 64


def test_batch_size_keeps_request_when_memory_is_plentiful(monkeypatch) -> None:
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: (20000, 24576))
    guard = VramGuard(min_free_mib=1536, enabled=True)
    assert guard.batch_size("embed", 256, per_item_mib=8.0) == 256


def test_batch_size_never_drops_below_one(monkeypatch) -> None:
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: (1600, 24576))
    guard = VramGuard(min_free_mib=1536, enabled=True)
    assert guard.batch_size("embed", 256, per_item_mib=1000.0) == 1


def test_oom_event_reduces_subsequent_batches(monkeypatch) -> None:
    """Один эпизод нехватки памяти должен снизить нагрузку на весь прогон."""
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: None)
    guard = VramGuard(enabled=True)

    assert guard.batch_size("embed", 256) == 256
    guard.record_oom("embed")
    assert guard.batch_size("embed", 256) == 128
    guard.record_oom("embed")
    assert guard.batch_size("embed", 256) == 64
    # Другая стадия не должна пострадать.
    assert guard.batch_size("graph", 256) == 256
    assert guard.oom_events == 2


def test_wait_gives_up_when_request_exceeds_card(monkeypatch) -> None:
    monkeypatch.setattr("rag_textbook.indexing.resources.query_vram_mib", lambda: (1000, 24576))
    guard = VramGuard(enabled=True, max_wait_seconds=0.1, poll_seconds=0.5)
    # Запрошено больше, чем есть на карте физически — ждать бессмысленно.
    assert guard.wait_for_headroom(100_000, "parse") is False


def test_wait_returns_when_memory_frees_up(monkeypatch) -> None:
    readings = iter([(500, 24576), (500, 24576), (9000, 24576)])
    monkeypatch.setattr(
        "rag_textbook.indexing.resources.query_vram_mib", lambda: next(readings, (9000, 24576))
    )
    guard = VramGuard(enabled=True, max_wait_seconds=5.0, poll_seconds=0.01)
    assert guard.wait_for_headroom(8000, "parse") is True


# --------------------------------------------------------- деление батча при OOM


def test_backoff_splits_batch_and_completes() -> None:
    """Ключевое свойство: нехватка памяти не теряет прогон, а уменьшает батч."""
    guard = VramGuard(enabled=False)
    processed: list[int] = []

    def handler(batch):
        if len(batch) > 2:
            raise RuntimeError("CUDA out of memory")
        processed.extend(batch)
        return len(batch)

    results = run_with_oom_backoff(
        list(range(8)), handler, guard=guard, stage="embed", batch_size=8
    )

    assert sorted(processed) == list(range(8)), "все элементы должны быть обработаны"
    assert sum(results) == 8


def test_backoff_preserves_order() -> None:
    guard = VramGuard(enabled=False)
    seen: list[int] = []

    def handler(batch):
        if len(batch) > 2:
            raise RuntimeError("out of memory")
        seen.extend(batch)
        return len(batch)

    run_with_oom_backoff(list(range(8)), handler, guard=guard, stage="embed", batch_size=8)
    assert seen == list(range(8)), "порядок обработки должен сохраняться"


def test_backoff_reraises_non_oom_errors() -> None:
    guard = VramGuard(enabled=False)

    def handler(batch):
        raise ValueError("схема неверна")

    with pytest.raises(ValueError):
        run_with_oom_backoff([1, 2, 3], handler, guard=guard, stage="embed", batch_size=3)


def test_backoff_gives_up_on_single_item() -> None:
    """Если не помещается даже один элемент — это не лечится делением."""
    guard = VramGuard(enabled=False)

    def handler(batch):
        raise RuntimeError("CUDA out of memory")

    with pytest.raises(RuntimeError):
        run_with_oom_backoff([1, 2], handler, guard=guard, stage="embed", batch_size=2)


# ------------------------------------------------------------------- режимы


def _prepare(settings, monkeypatch, tmp_path, sample_blocks, count: int = 3):
    root = tmp_path / "corpus"
    root.mkdir(parents=True, exist_ok=True)
    for index in range(count):
        (root / f"book_{index}.pdf").write_bytes(b"%PDF-1.4 stub " + str(index).encode())
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
    return root


def test_stage_major_indexes_all_documents(settings, monkeypatch, tmp_path, sample_blocks) -> None:
    _prepare(settings, monkeypatch, tmp_path, sample_blocks, count=3)
    settings.indexing = IndexingSettings(mode="stage", vram_guard_enabled=False)

    context = build_context(settings)
    try:
        report = IndexingPipeline(context).run()
        stored = context.vector_store.count()
    finally:
        context.close()

    assert report.failed == 0
    assert len(report.documents) == 3
    assert report.total_chunks > 0
    assert stored == report.total_chunks
    assert report.config["indexing_mode"] == "stage"


def test_both_modes_produce_identical_index(settings, monkeypatch, tmp_path, sample_blocks) -> None:
    """Смена режима обхода не должна менять результат — только скорость."""
    _prepare(settings, monkeypatch, tmp_path, sample_blocks, count=2)

    settings.indexing = IndexingSettings(mode="document", vram_guard_enabled=False)
    context = build_context(settings)
    try:
        by_document = IndexingPipeline(context).run()
        document_ids = sorted(chunk.id for chunk in context.vector_store.iter_chunks())
    finally:
        context.close()

    # Чистое состояние для второго прогона.
    for path in settings.paths.manifest_dir.glob("*.json"):
        path.unlink()
    settings.indexing = IndexingSettings(mode="stage", vram_guard_enabled=False)
    context = build_context(settings)
    try:
        by_stage = IndexingPipeline(context).run()
        stage_ids = sorted(chunk.id for chunk in context.vector_store.iter_chunks())
    finally:
        context.close()

    assert by_document.total_chunks == by_stage.total_chunks
    assert document_ids == stage_ids


def test_stage_major_is_resumable(settings, monkeypatch, tmp_path, sample_blocks) -> None:
    _prepare(settings, monkeypatch, tmp_path, sample_blocks, count=2)
    settings.indexing = IndexingSettings(mode="stage", vram_guard_enabled=False)

    context = build_context(settings)
    try:
        first = IndexingPipeline(context).run()
        count_after_first = context.vector_store.count()
        second = IndexingPipeline(context).run()
    finally:
        context.close()

    assert second.total_chunks == first.total_chunks
    assert context.vector_store.count() == count_after_first
    assert all(not doc.stage_seconds for doc in second.documents), "стадии должны пропускаться"


def test_stage_major_isolates_failed_document(
    settings, monkeypatch, tmp_path, sample_blocks
) -> None:
    """Сбой на одном документе не должен ронять остальные."""
    _prepare(settings, monkeypatch, tmp_path, sample_blocks, count=3)
    settings.indexing = IndexingSettings(mode="stage", vram_guard_enabled=False)

    from rag_textbook.parsing.pdf_parser import PdfParseError

    def flaky_parse(self, path, force=False):
        if "book_1" in str(path):
            raise PdfParseError("сломанный PDF")
        return list(sample_blocks)

    monkeypatch.setattr(
        "rag_textbook.parsing.pdf_parser.MineruPdfParser.parse", flaky_parse, raising=True
    )

    context = build_context(settings)
    try:
        report = IndexingPipeline(context).run()
    finally:
        context.close()

    assert report.failed == 1
    assert sum(1 for doc in report.documents if doc.status == "ok") == 2
    assert report.total_chunks > 0


def test_report_records_oom_events(settings, monkeypatch, tmp_path, sample_blocks) -> None:
    _prepare(settings, monkeypatch, tmp_path, sample_blocks, count=1)
    settings.indexing = IndexingSettings(mode="stage", vram_guard_enabled=False)

    context = build_context(settings)
    try:
        report = IndexingPipeline(context).run()
    finally:
        context.close()

    # Поле должно присутствовать всегда: по нему видно, срабатывала ли страховка.
    assert report.config["oom_events"] == 0
