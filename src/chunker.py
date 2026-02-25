from __future__ import annotations
import copy
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Set

from utils import chat_with_ollama

# Настройка логгера
logger = logging.getLogger("LayoutAwareChunker")
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


class LayoutAwareChunker:
    """Класс для layout-aware чанкирования с поддержкой спец-объектов и метаданных."""

    def __init__(
        self,
        images_root: Optional[str | Path] = None,
        chunk_size: int = 1200,
        chunk_overlap: int = 200,
        context_window: int = 250,
        separator: str = " ",
        special_types: Optional[set[str]] = None,
        use_llm: bool = True,
        llm_fn=None,
        add_type_prefix: bool = False,
        keep_bbox_map: bool = True,
        prompt_templates: Optional[Dict[str, str]] = None,
        normalize_bbox: bool = False,
        page_size_map: Optional[Dict[int, Tuple[int, int]]] = None,
        sticky_headers: bool = True,
        header_levels: Optional[set[int]] = None,
        hard_cut_window: int = 40,
    ) -> None:
        if llm_fn is None and use_llm:
            llm_fn = chat_with_ollama
        if chunk_size <= 0:
            raise ValueError("chunk_size должен быть > 0")
        if chunk_overlap < 0:
            raise ValueError("chunk_overlap не может быть < 0")

        self.images_root = Path(images_root) if images_root else None
        self.chunk_size = chunk_size
        self.chunk_overlap = min(chunk_overlap, max(chunk_size - 1, 0))
        self.context_window = context_window
        self.separator = separator
        self.special_types = special_types or {"image", "table", "equation"}
        self.use_llm = use_llm
        self.llm_fn = llm_fn
        self.add_type_prefix = add_type_prefix
        self.keep_bbox_map = keep_bbox_map
        self.normalize_bbox = normalize_bbox
        self.page_size_map = page_size_map or {}
        self.sticky_headers = sticky_headers
        self.header_levels = header_levels or {1, 2}
        self.hard_cut_window = hard_cut_window

        self.prompt_templates = prompt_templates or {
            "image": (
                "Опиши изображение максимально кратко (до 50 слов). Перечисли только факты.\n"
                "{context_block}\nПодпись: {caption}\n"
            ),
            "table": (
                "Представь таблицу в формате Markdown. Не добавляй комментариев.\n"
                "{context_block}\nПодпись: {caption}\n"
            ),
            "equation": (
                "Объясни смысл формулы в одном предложении.\n"
                "{context_block}\nФормула (latex): {equation_text}\n"
            ),
        }

    def process_special_objects (self, blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обогащает спец-объекты описаниями от LLM/VLM."""

        if not self.use_llm:
            return blocks
        if self.llm_fn is None:
            raise ValueError("llm_fn не задан.")

        total = sum(1 for b in blocks if b.get("type") in self.special_types)
        if total == 0:
            return blocks

        logger.info("Спец-объектов для обработки: %s", total)
        processed = 0

        for idx, block in enumerate(blocks):
            btype = block.get("type")
            if btype not in self.special_types:
                continue
            
            # Если описание уже есть, не тратим токены
            if block.get("llm_description"):
                continue

            processed += 1
            if processed % 5 == 0:
                logger.info(f"Обработано {processed}/{total} спец. объектов...")

            left_ctx, right_ctx = self._collect_context(blocks, idx)
            context_block = self._format_context_block(left_ctx, right_ctx)
            caption = self._get_caption(block)
            equation_text = self._normalize_text(block.get("text", "")) if btype == "equation" else ""

            prompt = self.prompt_templates.get(btype, "{context_block}").format(
                context_block=context_block,
                caption=caption,
                equation_text=equation_text,
            ).strip()

            img_path = block.get("img_path")
            full_img_path = None
            if img_path and self.images_root:
                full_img_path = self.images_root / img_path
                if not full_img_path.is_file():
                    full_img_path = None

            try:
                if full_img_path:
                    description = self.llm_fn(message=prompt, img_path=str(full_img_path))
                else:
                    description = self.llm_fn(message=prompt)
            except Exception as exc:
                logger.error(f"Ошибка LLM на блоке {idx} ({btype}): {exc}")
                description = ""

            block["llm_description"] = self._normalize_text(description)
            
        self.llm_fn(turn_off=True)

        return blocks

    def chunk(
        self,
        blocks: List[Dict[str, Any]],
        doc_name: Optional[str] = None,
        doc_format: Optional[str] = None,
        mutate: bool = True,
        normalize_bbox: Optional[bool] = None,
        page_size_map: Optional[Dict[int, Tuple[int, int]]] = None,
    ) -> List[Dict[str, Any]]:
        """Строит спец-чанки и обычные чанки по скользящему окну."""
        if not mutate:
            blocks = copy.deepcopy(blocks)

        blocks = [b for b in blocks if b.get("type") != "discarded"]

        self.process_special_objects(blocks)

        use_norm = self.normalize_bbox if normalize_bbox is None else normalize_bbox
        use_page_map = page_size_map or self.page_size_map

        segments = self._build_segments(blocks, use_norm, use_page_map)
        if not segments:
            return []

        full_text, spans = self._build_full_text(segments)
        total_len = len(full_text)

        special_chunks = self._build_special_chunks(
            full_text, spans, segments, doc_name=doc_name, doc_format=doc_format
        )

        chunks: List[Dict[str, Any]] = []
        start = 0

        while start < total_len:
            end_candidate = min(start + self.chunk_size, total_len)

            atomic_end = self._adjust_boundary_for_atomic_blocks(start, end_candidate, segments, spans)

            if atomic_end != end_candidate:
                end = atomic_end
            else:
                end = self._find_soft_boundary(full_text, start, end_candidate)

            if end <= start:
                end = min(start + self.chunk_size, total_len)
                if end <= start:
                    break

            chunk_text = full_text[start:end]

            if not chunk_text.strip():
                start = end
                continue

            meta = self._build_metadata(
                start, end, segments, spans, doc_name=doc_name, doc_format=doc_format, mark_special=False
            )
            chunks.append({"text": chunk_text, **meta})

            if end >= total_len:
                break

            next_start = max(end - self.chunk_overlap, 0)
            if next_start <= start:
                next_start = end
            start = next_start

        return special_chunks + chunks

    def _adjust_boundary_for_atomic_blocks(
        self, 
        chunk_start: int, 
        candidate_end: int, 
        segments: List[Dict[str, Any]], 
        spans: List[Tuple[int, int]]
    ) -> int:
        
        """
        Проверяет, попадает ли candidate_end внутрь спец. объекта.
        Гарантированно возвращает int.
        """

        for i, (seg_start, seg_end) in enumerate(spans):
            # Сегмент закончился до границы среза
            if seg_end <= candidate_end:
                continue
            
            # Сегмент начинается после границы среза (мы прошли точку разреза)
            if seg_start >= candidate_end:
                break
                
            # Мы внутри сегмента (seg_start < candidate_end < seg_end)
            if segments[i].get("is_special"):
                # Вариант А: Расширить чанк, чтобы включить весь объект
                limit = self.chunk_size * 1.5 
                if (seg_end - chunk_start) <= limit:
                    return seg_end 
                
                # Вариант Б: Обрезать ДО начала объекта (перенести его в след. чанк)
                # Только если текущий чанк не станет слишком маленьким
                if seg_start > chunk_start and (seg_start - chunk_start) > 200:
                    return seg_start
                
                # Если объект огромный и мы уже набрали мало текста, придется резать (fallback)
                return candidate_end

        # Если пересечений со спец. объектами нет или они не требуют особого обращения
        return candidate_end

    def _build_special_chunks(
        self,
        full_text: str,
        spans: List[Tuple[int, int]],
        segments: List[Dict[str, Any]],
        doc_name: Optional[str],
        doc_format: Optional[str],
    ) -> List[Dict[str, Any]]:
        
        """Создает спец-чанки, центрируя окно на спец-объекте."""
        
        chunks: List[Dict[str, Any]] = []
        total_len = len(full_text)

        for seg, (seg_start, seg_end) in zip(segments, spans):
            if not seg.get("is_special"):
                continue

            start, end = self._calc_special_window(seg_start, seg_end, total_len)
            start = self._find_soft_boundary_start(full_text, start, seg_start)
            end = self._find_soft_boundary(full_text, start, end)

            if end <= start:
                continue

            chunk_text = full_text[start:end]
            if not chunk_text.strip():
                continue

            meta = self._build_metadata(
                start, end, segments, spans, doc_name=doc_name, doc_format=doc_format, mark_special=True, force_special_type=(seg.get("special_type") or seg.get("type"))
            )
            meta["special_focus"] = {
                "block_idx": seg.get("block_idx"),
                "special_type": seg.get("special_type") or seg.get("type"),
            }
            chunks.append({"text": chunk_text, **meta})

        logger.info("Спец-чанков: %s", len(chunks))
        return chunks

    def _calc_special_window(self, seg_start: int, seg_end: int, total_len: int) -> Tuple[int, int]:
        size = self.chunk_size
        mid = (seg_start + seg_end) // 2
        start = max(0, mid - size // 2)
        end = min(total_len, start + size)

        if seg_start < start:
            start = seg_start
            end = min(total_len, start + size)
        if seg_end > end:
            end = seg_end
            start = max(0, end - size)

        return start, end

    def _find_soft_boundary_start(self, text: str, start: int, limit: int) -> int:
        if start <= 0:
            return 0
        window = self.hard_cut_window
        search_end = min(len(text), max(start, min(limit, start + window)))
        if search_end <= start:
            return start
        slice_text = text[start:search_end]
        for i, ch in enumerate(slice_text):
            if ch.isspace():
                return start + i + 1
        return start

    def _build_segments(
        self,
        blocks: List[Dict[str, Any]],
        normalize_bbox: bool,
        page_size_map: Dict[int, Tuple[int, int]],
    ) -> List[Dict[str, Any]]:
        
        """Готовит список сегментов текста с bbox, типами и заголовками."""

        segments: List[Dict[str, Any]] = []
        current_header = ""

        for idx, block in enumerate(blocks):
            btype = block.get("type")
            text_level = block.get("text_level")

            # Обновляем sticky header
            if btype == "text" and text_level in self.header_levels:
                header_text = self._normalize_text(block.get("text", ""))
                if header_text:
                    current_header = header_text

            text = self._block_to_text(block)
            text = self._normalize_text(text)
            if not text:
                continue

            bbox = block.get("bbox")
            page_idx = block.get("page_idx")
            if normalize_bbox and bbox and page_idx in page_size_map:
                bbox = self._normalize_bbox(bbox, page_size_map[page_idx])

            is_special = btype in self.special_types

            segments.append(
                {
                    "text": text,
                    "page_idx": page_idx,
                    "bbox": bbox,
                    "block_idx": idx,
                    "type": btype,
                    "is_special": is_special,
                    # Сохраняем конкретный тип (table/image/equation)
                    "special_type": btype if is_special else None,
                    # Сохраняем заголовок отдельно, не впекая в текст
                    "header": current_header if self.sticky_headers else ""
                }
            )
        return segments

    def _build_metadata(
        self,
        start: int,
        end: int,
        segments: List[Dict[str, Any]],
        spans: List[Tuple[int, int]],
        doc_name: Optional[str],
        doc_format: Optional[str],
        mark_special: bool = False,
        force_special_type: Optional[str] = None,
    ) -> Dict[str, Any]:
        
        """Формирует метаданные чанка: page_spans, bbox, headers, special flags."""

        page_spans: Dict[str, Dict[str, int]] = {}
        bbox_map: Dict[str, List[int]] = {}

        headers_found = set()
        special_types_found = set()

        for seg, (seg_start, seg_end) in zip(segments, spans):
            if seg_end <= start or seg_start >= end:
                continue

            overlap_start = max(seg_start, start)
            overlap_end = min(seg_end, end)
            local_start = overlap_start - start
            local_end = overlap_end - start

            # Page Spans
            page_idx = seg.get("page_idx")
            if page_idx is not None:
                page_key = f"page_{page_idx}"
                if page_key not in page_spans:
                    page_spans[page_key] = {"from": local_start, "to": local_end}
                else:
                    page_spans[page_key]["from"] = min(page_spans[page_key]["from"], local_start)
                    page_spans[page_key]["to"] = max(page_spans[page_key]["to"], local_end)

            # BBox
            if self.keep_bbox_map and seg.get("bbox"):
                bbox_map[f"{local_start}:{local_end}"] = seg["bbox"]

            # Headers Collection
            if seg.get("header"):
                headers_found.add(seg["header"])

            # Special Types (for optional use)
            if seg.get("is_special") and seg.get("special_type"):
                special_types_found.add(seg["special_type"])

        meta: Dict[str, Any] = {
            "page_spans": page_spans,
            "bbox": bbox_map if self.keep_bbox_map else {},
            "doc_name": doc_name,
            "doc_format": doc_format,
            "headers": list(headers_found),
            "is_special": False,
            "special_type": None,
        }

        if mark_special:
            meta["is_special"] = True
            if force_special_type:
                meta["special_type"] = force_special_type
            elif special_types_found:
                # ?????????: table > equation > image
                if "table" in special_types_found:
                    meta["special_type"] = "table"
                elif "equation" in special_types_found:
                    meta["special_type"] = "equation"
                else:
                    meta["special_type"] = list(special_types_found)[0]

        return meta

    def _build_full_text(self, segments: List[Dict[str, Any]]) -> Tuple[str, List[Tuple[int, int]]]:
        parts: List[str] = []
        spans: List[Tuple[int, int]] = []
        pos = 0
        last_header = None

        for seg in segments:
            current_seg_header = seg.get("header", "")
            seg_text = seg["text"]

            prefix = ""
            if current_seg_header and current_seg_header != last_header:
                if current_seg_header.lower() not in seg_text.lower()[:len(current_seg_header)+10]:
                     prefix = f"{current_seg_header}.\n"
                last_header = current_seg_header

            if parts:
                parts.append(self.separator)
                pos += len(self.separator)

            start = pos
            if prefix:
                parts.append(prefix)
                pos += len(prefix)

            parts.append(seg_text)
            pos += len(seg_text)
            end = pos
            spans.append((start, end))

        return "".join(parts), spans

    def _find_soft_boundary(self, text: str, start: int, end: int) -> int:
        if end >= len(text):
            return end
        window = self.hard_cut_window
        
        # Границы поиска
        search_start = max(start, end - window)
        search_end = min(len(text), end + window)
        
        slice_text = text[search_start:search_end]
        
        # 1. Приоритет: Конец предложения (. ! ?)
        for match in re.finditer(r'[.!?]\s', slice_text):
            absolute_pos = search_start + match.start() + 1
            # Не уходим слишком далеко вперед
            if absolute_pos <= end + 10:
                return absolute_pos

        # 2. Приоритет: Пробел (только влево от end)
        left_slice = text[search_start:end]
        last_space = left_slice.rfind(' ')
        if last_space != -1:
            return search_start + last_space + 1

        return end

    # --- Вспомогательные методы без изменений ---
    def _collect_context(self, blocks: List[Dict[str, Any]], index: int) -> Tuple[str, str]:
        left_parts = []
        right_parts = []
        
        i = index - 1
        while i >= 0 and sum(len(p) for p in left_parts) < self.context_window:
            if blocks[i].get("type") != "discarded":
                text = self._normalize_text(self._block_to_text(blocks[i], use_description=False))
                if text: left_parts.append(text)
            i -= 1

        i = index + 1
        while i < len(blocks) and sum(len(p) for p in right_parts) < self.context_window:
            if blocks[i].get("type") != "discarded":
                text = self._normalize_text(self._block_to_text(blocks[i], use_description=False))
                if text: right_parts.append(text)
            i += 1

        left = self.separator.join(reversed(left_parts)).strip()
        right = self.separator.join(right_parts).strip()
        return left[-self.context_window:], right[:self.context_window]

    def _format_context_block(self, left: str, right: str) -> str:
        parts = []
        if left: parts.append(f"Контекст слева: {left}")
        if right: parts.append(f"Контекст справа: {right}")
        return "\n".join(parts)

    def _get_caption(self, block: Dict[str, Any]) -> str:
        btype = block.get("type")
        caption = ""
        if btype == "image":
            caption = self._join_list(block.get("image_caption"))
        elif btype == "table":
            caption = self._join_list(block.get("table_caption"))
            if not caption: caption = self._join_list(block.get("table_footnote"))
        return self._normalize_text(caption)

    def _block_to_text(self, block: Dict[str, Any], use_description: bool = True) -> str:
        btype = block.get("type")
        if btype in self.special_types and use_description and block.get("llm_description"):
            text = block.get("llm_description", "")
            return self._apply_prefix(text, btype)
        if btype == "text": return block.get("text", "")
        if btype == "equation": return block.get("text", "")
        if btype == "image":
            return self._apply_prefix(self._join_list([block.get("image_caption"), block.get("image_footnote")]), btype)
        if btype == "table":
            return self._apply_prefix(self._join_list([block.get("table_caption"), block.get("table_footnote"), self._strip_html(block.get("table_body", ""))]), btype)
        
        for key in ("text", "image_caption", "table_caption", "table_body"):
            val = block.get(key)
            if val:
                return str(self._join_list(val) if isinstance(val, list) else val)
        return ""

    def _apply_prefix(self, text: str, btype: Optional[str]) -> str:
        if not self.add_type_prefix or not btype: return text
        return f"[{btype.upper()}] {text}" if text else ""

    @staticmethod
    def _normalize_bbox(bbox: List[int], page_size: Tuple[int, int]) -> List[int]:
        if not bbox or len(bbox) != 4: return bbox
        page_width, page_height = page_size
        if not page_width or not page_height: return bbox
        x0, y0, x1, y1 = bbox
        return [int(x0 * 1000 / page_width), int(y0 * 1000 / page_height), int(x1 * 1000 / page_width), int(y1 * 1000 / page_height)]

    @staticmethod
    def _normalize_text(text: str) -> str:
        if not text: return ""
        text = text.replace("­", "")
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def _join_list(value: Any) -> str:
        if isinstance(value, list): return " ".join([str(v) for v in value if v])
        return str(value) if value is not None else ""

    @staticmethod
    def _strip_html(html: str) -> str:
        if not html: return ""
        text = re.sub(r"<[^>]+>", " ", html)
        return re.sub(r"\s+", " ", text).strip()




