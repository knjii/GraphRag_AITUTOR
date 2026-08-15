"""Тесты сохранности формул и таблиц.

Это регрессионные тесты на конкретный дефект прежней версии: описание от модели
зрения возвращалось *вместо* исходного представления блока, из-за чего LaTeX
формул и HTML таблиц в индекс не попадали.
"""

from __future__ import annotations

from rag_textbook.chunking.layout_chunker import LayoutAwareChunker
from rag_textbook.config import ChunkingSettings
from rag_textbook.models import Block, html_to_text
from rag_textbook.parsing.normalize import normalize_mineru_blocks


def test_equation_keeps_latex_when_enrichment_present() -> None:
    block = Block(
        index=0,
        type="equation",
        text=r"A = U \Sigma V^{T}",
        latex=r"A = U \Sigma V^{T}",
        enrichment="Разложение матрицы на три множителя.",
    )
    indexed = block.to_indexable_text()

    assert r"\Sigma" in indexed, "LaTeX формулы обязан оставаться в индексируемом тексте"
    assert "Разложение матрицы" in indexed, "описание должно дополнять, а не заменять"
    assert indexed.index("\\Sigma") < indexed.index("Разложение"), (
        "исходное представление должно идти первым"
    )


def test_table_keeps_structure_when_enrichment_present() -> None:
    block = Block(
        index=0,
        type="table",
        table_html="<table><tr><td>k</td><td>sigma</td></tr><tr><td>1</td><td>5.2</td></tr></table>",
        caption="Таблица 1",
        enrichment="Таблица сингулярных чисел.",
    )
    indexed = block.to_indexable_text()

    assert "sigma" in indexed and "5.2" in indexed, "значения ячеек должны быть в индексе"
    assert "Таблица сингулярных чисел" in indexed


def test_image_without_own_text_uses_enrichment() -> None:
    block = Block(index=0, type="image", caption="Рис. 1", enrichment="График параболы.")
    indexed = block.to_indexable_text()
    assert "Рис. 1" in indexed
    assert "График параболы" in indexed


def test_html_to_text_preserves_cell_separation() -> None:
    text = html_to_text("<table><tr><td>a</td><td>b</td></tr></table>")
    assert "a" in text and "b" in text
    assert "|" in text, "ячейки должны быть разделены, а не слипаться"


def test_normalize_drops_discarded_and_empty_blocks() -> None:
    blocks = normalize_mineru_blocks(
        [
            {"type": "discarded", "text": "колонтитул"},
            {"type": "text", "text": ""},
            {"type": "text", "text": "Содержательный абзац", "page_idx": 4},
        ]
    )
    assert len(blocks) == 1
    assert blocks[0].page_idx == 4


def test_chunker_assigns_human_page_numbers(sample_blocks) -> None:
    chunker = LayoutAwareChunker(ChunkingSettings(chunk_size=400, chunk_overlap=60))
    chunks = chunker.chunk(
        sample_blocks, doc_id="doc1", doc_name="Линал", source_path="/corpus/linal.pdf"
    )

    assert chunks, "чанкер обязан вернуть хотя бы один фрагмент"
    assert all(chunk.pages for chunk in chunks), "у каждого чанка должны быть страницы"
    # MinerU нумерует страницы с нуля; читателю нужна нумерация с единицы.
    assert min(page for chunk in chunks for page in chunk.pages) >= 1
    assert "с." in chunks[0].citation_label()


def test_chunker_marks_formula_and_table_chunks(sample_blocks) -> None:
    chunker = LayoutAwareChunker(ChunkingSettings(chunk_size=500, chunk_overlap=50))
    chunks = chunker.chunk(
        sample_blocks, doc_id="doc1", doc_name="Линал", source_path="/corpus/linal.pdf"
    )
    assert any(chunk.has_formula for chunk in chunks)
    assert any(chunk.has_table for chunk in chunks)
    joined = " ".join(chunk.text for chunk in chunks)
    assert r"\Sigma" in joined, "формула должна дойти до текста чанков"


def test_chunk_ids_are_deterministic(sample_blocks) -> None:
    chunker = LayoutAwareChunker(ChunkingSettings(chunk_size=400, chunk_overlap=60))
    first = chunker.chunk(sample_blocks, doc_id="d", doc_name="n", source_path="/p.pdf")
    second = chunker.chunk(sample_blocks, doc_id="d", doc_name="n", source_path="/p.pdf")
    assert [chunk.id for chunk in first] == [chunk.id for chunk in second], (
        "повторная индексация должна обновлять записи, а не плодить дубликаты"
    )


def test_chunk_size_is_bounded(sample_blocks) -> None:
    settings = ChunkingSettings(chunk_size=300, chunk_overlap=40)
    chunker = LayoutAwareChunker(settings)
    chunks = chunker.chunk(sample_blocks, doc_id="d", doc_name="n", source_path="/p.pdf")
    limit = int(settings.chunk_size * 1.5) + 100
    assert all(len(chunk.text) <= limit for chunk in chunks), (
        "расширение под спец-объект должно быть ограничено сверху"
    )
