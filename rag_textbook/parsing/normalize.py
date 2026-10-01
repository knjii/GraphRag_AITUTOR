"""Приведение вывода MinerU к нашему типу :class:`Block`.

Слой нужен по двум причинам. Во-первых, формат ``content_list`` менялся между
версиями MinerU, и без нормализации эти отличия расползаются по всему коду.
Во-вторых, именно здесь мы сохраняем LaTeX формул и HTML таблиц в отдельные поля,
чтобы дальше их нельзя было случайно затереть описанием от модели зрения.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from rag_textbook.models import Block, BlockType, normalize_text

_TYPE_ALIASES: dict[str, BlockType] = {
    "text": "text",
    "title": "title",
    "image": "image",
    "figure": "image",
    "chart": "chart",
    "table": "table",
    "equation": "equation",
    "interline_equation": "equation",
    "isolate_formula": "equation",
    "discarded": "discarded",
}


def _join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return " ".join(str(item) for item in value if item)
    return str(value)


def _as_float_list(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def _resolve_type(raw: Any) -> BlockType:
    key = str(raw or "text").strip().lower()
    return _TYPE_ALIASES.get(key, "other")


def normalize_mineru_block(index: int, raw: dict[str, Any]) -> Block:
    block_type = _resolve_type(raw.get("type"))
    text = normalize_text(_join(raw.get("text")))

    caption = ""
    footnote = ""
    if block_type == "image":
        caption = normalize_text(_join(raw.get("image_caption")))
        footnote = normalize_text(_join(raw.get("image_footnote")))
    elif block_type == "chart":
        caption = normalize_text(_join(raw.get("chart_caption") or raw.get("image_caption")))
        footnote = normalize_text(_join(raw.get("chart_footnote") or raw.get("image_footnote")))
    elif block_type == "table":
        caption = normalize_text(_join(raw.get("table_caption")))
        footnote = normalize_text(_join(raw.get("table_footnote")))

    # Формула: MinerU кладёт LaTeX в `text`. Сохраняем его отдельным полем,
    # чтобы дальше по конвейеру он гарантированно дошёл до индекса.
    latex = ""
    if block_type == "equation":
        latex = normalize_text(_join(raw.get("latex") or raw.get("text")))

    page_idx = raw.get("page_idx")
    try:
        page = int(page_idx) if page_idx is not None else None
    except (TypeError, ValueError):
        page = None

    text_level = raw.get("text_level")
    try:
        level = int(text_level) if text_level is not None else None
    except (TypeError, ValueError):
        level = None

    return Block(
        index=index,
        type=block_type,
        text=text,
        page_idx=page,
        bbox=_as_float_list(raw.get("bbox")),
        text_level=level,
        img_path=(str(raw.get("img_path")) if raw.get("img_path") else None),
        caption=caption,
        footnote=footnote,
        table_html=str(raw.get("table_body") or ""),
        latex=latex,
    )


def normalize_mineru_blocks(raw_blocks: Iterable[dict[str, Any]]) -> list[Block]:
    """Нормализует список блоков, отбрасывая служебные и пустые."""
    blocks: list[Block] = []
    for index, raw in enumerate(raw_blocks or []):
        if not isinstance(raw, dict):
            continue
        block = normalize_mineru_block(index, raw)
        if block.type == "discarded":
            continue
        # Пустые блоки без картинки не несут информации.
        if not any(
            (
                block.text,
                block.caption,
                block.footnote,
                block.table_html,
                block.latex,
                block.img_path,
            )
        ):
            continue
        blocks.append(block)
    return blocks


def page_range(blocks: Sequence[Block]) -> list[int]:
    pages = sorted({block.page_idx for block in blocks if block.page_idx is not None})
    return [int(page) for page in pages]
