"""Текстовый вход в конвейер, минуя разбор PDF.

Зачем понадобился. Публичные наборы раздают корпус готовым текстом, а наш
конвейер начинается с MinerU. Гонять чужой текст через печать в PDF и обратно
значило бы измерять свойства этой перегонки, а не свойства поиска.

Что здесь важно не потерять. Чанкер режет не сырой текст, а последовательность
блоков: он опирается на заголовки, страницы и особые блоки (формулы, таблицы),
и именно поэтому не рвёт формулу пополам. Поэтому текст переводится в блоки,
а не подсовывается чанкеру напрямую: тогда чужой корпус проходит ровно тот же
путь, что и учебник, и сравнение остаётся честным.

Разметка блоков делается по разметке Markdown, потому что публичные корпуса
раздаются либо в Markdown, либо в простом тексте, где заголовков нет вовсе —
и тогда документ просто становится одним потоком абзацев.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_textbook.chunking.layout_chunker import LayoutAwareChunker
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Block, Chunk, content_hash

logger = get_logger("benchmarks.text_corpus")

_HEADING = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
# Формула в отдельном абзаце: $$...$$ либо \[...\]. Такой блок помечается
# особым, и чанкер перестаёт резать его пополам.
_DISPLAY_MATH = re.compile(r"^\s*(?:\$\$.*\$\$|\\\[.*\\\])\s*$", re.DOTALL)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class TextDocument:
    """Документ публичного корпуса."""

    doc_id: str
    title: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)


def blocks_from_text(text: str) -> list[Block]:
    """Переводит текст в блоки так, как это сделал бы парсер.

    Абзац — единица блока. Заголовки Markdown становятся блоками с уровнем,
    таблицы и выключные формулы — особыми блоками: именно по ним чанкер
    решает, где резать нельзя.
    """
    blocks: list[Block] = []
    for paragraph in re.split(r"\n\s*\n", text or ""):
        body = paragraph.strip()
        if not body:
            continue
        heading = _HEADING.match(body)
        if heading is not None:
            blocks.append(
                Block(
                    index=len(blocks),
                    type="text",
                    text=heading.group(2),
                    text_level=len(heading.group(1)),
                )
            )
            continue
        if _DISPLAY_MATH.match(body):
            blocks.append(
                Block(index=len(blocks), type="equation", text=body, latex=body.strip("$ \n"))
            )
            continue
        if all(_TABLE_ROW.match(line) for line in body.splitlines()):
            blocks.append(Block(index=len(blocks), type="table", text=body, table_html=body))
            continue
        blocks.append(Block(index=len(blocks), type="text", text=body))
    return blocks


def chunk_documents(
    documents: Iterable[TextDocument],
    chunker: LayoutAwareChunker,
    *,
    source_label: str,
) -> list[Chunk]:
    """Режет корпус на чанки тем же чанкером, что и учебник.

    ``source_label`` попадает в путь источника: по нему видно, из какого
    набора взят фрагмент, если в хранилище лежит несколько корпусов сразу.
    """
    chunks: list[Chunk] = []
    for document in documents:
        blocks = blocks_from_text(document.text)
        if not blocks:
            logger.warning("Документ %s пуст, пропущен", document.doc_id)
            continue
        produced = chunker.chunk(
            blocks,
            doc_id=document.doc_id,
            doc_name=document.title or document.doc_id,
            source_path=f"{source_label}/{document.doc_id}",
        )
        chunks.extend(produced)
    logger.info(
        "Корпус %s: документов %s, чанков %s",
        source_label,
        len(list(documents)) if isinstance(documents, Sequence) else "?",
        len(chunks),
    )
    return chunks


def stable_doc_id(*parts: str) -> str:
    """Устойчивый идентификатор документа.

    Именно устойчивый, а не порядковый: набор пересобирается, порядок в файле
    может измениться, а идентификаторы обязаны совпасть — иначе эталонные
    фрагменты прошлого прогона перестанут находиться.
    """
    return content_hash(*parts)[:16]


def iter_windows(text: str, size: int, overlap: int) -> Iterator[str]:
    """Служебное окно по тексту. Нужно только для сопоставления цитат."""
    step = max(size - overlap, 1)
    for start in range(0, max(len(text), 1), step):
        window = text[start : start + size]
        if window:
            yield window
