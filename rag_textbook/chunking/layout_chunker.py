"""Layout-aware чанкер.

Логика скользящего окна с мягкими границами и «липкими» заголовками унаследована
из прежней реализации — она предметно осмысленная и работала. Исправлено четыре вещи:

1. **Формулы и таблицы больше не теряются.** Прежний ``_block_to_text`` возвращал
   описание от модели зрения *вместо* исходного текста блока, поэтому LaTeX формулы
   и HTML таблицы в индекс не попадали вообще. Теперь описание дополняет исходник.
2. **Появились номера страниц.** Раньше писался только ``page_spans`` в виде JSON-строки,
   а поле ``page`` не заполнялось, из-за чего цитаты выглядели как «p.?».
3. **Границы чанка ограничены сверху.** Расширение под «атомарный» блок могло
   растянуть чанк без верхнего предела; теперь предел явный.
4. **Детерминированные идентификаторы.** ``chunk_id`` строится из документа и порядкового
   номера, поэтому повторная индексация обновляет запись, а не плодит дубликаты.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_textbook.config import ChunkingSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import SPECIAL_BLOCK_TYPES, Block, Chunk, content_hash, normalize_text

logger = get_logger("chunking.layout")

_SENTENCE_END_RE = re.compile(r"[.!?…]\s")
HEADER_LEVELS: frozenset[int] = frozenset({1, 2})


@dataclass
class _Segment:
    """Кусок итогового текста документа с привязкой к исходному блоку."""

    text: str
    block_index: int
    block_type: str
    page_idx: int | None
    header: str
    is_special: bool
    start: int = 0
    end: int = 0
    bbox: list[float] | None = field(default=None)


class LayoutAwareChunker:
    def __init__(self, settings: ChunkingSettings) -> None:
        self.settings = settings
        self.chunk_size = settings.chunk_size
        self.chunk_overlap = min(settings.chunk_overlap, max(settings.chunk_size - 1, 0))
        self.soft_boundary_window = 60
        # Верхняя граница расширения чанка ради целостности спец-объекта.
        self.max_chunk_size = int(settings.chunk_size * 1.5)

    # ------------------------------------------------------------- построение

    def _build_segments(self, blocks: Sequence[Block]) -> list[_Segment]:
        segments: list[_Segment] = []
        current_header = ""

        for block in blocks:
            if block.type == "title" or (
                block.type == "text" and block.text_level in HEADER_LEVELS
            ):
                header = normalize_text(block.text)
                if header:
                    current_header = header

            text = block.to_indexable_text(include_enrichment=True)
            if not text:
                continue

            segments.append(
                _Segment(
                    text=text,
                    block_index=block.index,
                    block_type=block.type,
                    page_idx=block.page_idx,
                    header=current_header if self.settings.sticky_headers else "",
                    is_special=block.type in SPECIAL_BLOCK_TYPES,
                    bbox=block.bbox,
                )
            )
        return segments

    def _build_document_text(self, segments: list[_Segment]) -> str:
        """Склеивает сегменты, проставляя им абсолютные позиции.

        Заголовок вставляется перед первым сегментом раздела, чтобы чанк
        из середины главы всё равно нёс её название.
        """
        parts: list[str] = []
        position = 0
        last_header: str | None = None

        for segment in segments:
            if parts:
                parts.append(" ")
                position += 1

            prefix = ""
            header = segment.header
            if header and header != last_header:
                # Не дублируем заголовок, если сегмент с него и начинается.
                if header.lower() not in segment.text[: len(header) + 10].lower():
                    prefix = f"{header}. "
                last_header = header

            segment.start = position
            if prefix:
                parts.append(prefix)
                position += len(prefix)
            parts.append(segment.text)
            position += len(segment.text)
            segment.end = position

        return "".join(parts)

    # -------------------------------------------------------------- границы

    def _soft_boundary(self, text: str, start: int, end: int) -> int:
        """Сдвигает границу к концу предложения или к пробелу."""
        if end >= len(text):
            return len(text)

        window = self.soft_boundary_window
        left = max(start + 1, end - window)
        right = min(len(text), end + window)

        # Конец предложения предпочтительнее пробела.
        best: int | None = None
        for match in _SENTENCE_END_RE.finditer(text, left, right):
            candidate = match.start() + 1
            if candidate <= start:
                continue
            if best is None or abs(candidate - end) < abs(best - end):
                best = candidate
        if best is not None:
            return best

        space = text.rfind(" ", left, end)
        if space > start:
            return space + 1
        return end

    def _adjust_for_special(
        self, chunk_start: int, candidate_end: int, segments: list[_Segment]
    ) -> int:
        """Старается не рвать спец-объект пополам.

        В отличие от прежней версии, расширение ограничено ``max_chunk_size``,
        поэтому одна большая таблица не растягивает чанк произвольно.
        """
        for segment in segments:
            if segment.end <= candidate_end:
                continue
            if segment.start >= candidate_end:
                break
            if not segment.is_special:
                continue

            # Вариант А: дотянуть чанк до конца объекта, если влезает в лимит.
            if segment.end - chunk_start <= self.max_chunk_size:
                return segment.end
            # Вариант Б: закончить чанк до объекта, если уже набрали осмысленный размер.
            if (
                segment.start > chunk_start
                and (segment.start - chunk_start) >= self.chunk_size // 3
            ):
                return segment.start
            break
        return candidate_end

    # ------------------------------------------------------------- метаданные

    def _metadata_for(
        self, start: int, end: int, segments: list[_Segment]
    ) -> tuple[list[int], list[str], list[str]]:
        pages: set[int] = set()
        headers: list[str] = []
        special: set[str] = set()

        for segment in segments:
            if segment.end <= start or segment.start >= end:
                continue
            if segment.page_idx is not None:
                pages.add(int(segment.page_idx))
            if segment.header and segment.header not in headers:
                headers.append(segment.header)
            if segment.is_special:
                special.add(segment.block_type)

        return sorted(pages), headers, sorted(special)

    # ---------------------------------------------------------------- публично

    def chunk(
        self,
        blocks: Sequence[Block],
        *,
        doc_id: str,
        doc_name: str,
        source_path: str,
        page_offset: int = 1,
    ) -> list[Chunk]:
        """Режет документ на чанки.

        ``page_offset`` переводит нумерацию страниц MinerU (с нуля) в человеческую
        (с единицы) — иначе цитата «с. 0» выглядит как ошибка.
        """

        segments = self._build_segments(blocks)
        if not segments:
            logger.warning("Документ %s не дал ни одного сегмента", doc_name)
            return []

        document_text = self._build_document_text(segments)
        total = len(document_text)
        chunks: list[Chunk] = []
        start = 0
        ordinal = 0

        while start < total:
            candidate_end = min(start + self.chunk_size, total)
            adjusted = self._adjust_for_special(start, candidate_end, segments)
            end = (
                adjusted
                if adjusted != candidate_end
                else self._soft_boundary(document_text, start, candidate_end)
            )

            if end <= start:
                end = min(start + self.chunk_size, total)
                if end <= start:
                    break

            text = document_text[start:end].strip()
            if text:
                pages, headers, special = self._metadata_for(start, end, segments)
                chunk_id = f"{doc_id}:{ordinal:05d}"
                chunks.append(
                    Chunk(
                        id=chunk_id,
                        doc_id=doc_id,
                        doc_name=doc_name,
                        source_path=source_path,
                        ordinal=ordinal,
                        text=text,
                        pages=[page + page_offset for page in pages],
                        headers=headers,
                        special_types=special,
                        has_formula=("equation" in special) or ("$$" in text),
                        has_table="table" in special,
                        has_figure=bool({"image", "chart"} & set(special)),
                        char_start=start,
                        char_end=end,
                        text_hash=content_hash(text),
                    )
                )
                ordinal += 1

            if end >= total:
                break

            next_start = max(end - self.chunk_overlap, start + 1)
            start = next_start

        logger.info(
            "Документ %s: чанков=%s, из них с формулами=%s, с таблицами=%s, с иллюстрациями=%s",
            doc_name,
            len(chunks),
            sum(1 for chunk in chunks if chunk.has_formula),
            sum(1 for chunk in chunks if chunk.has_table),
            sum(1 for chunk in chunks if chunk.has_figure),
        )
        return chunks
