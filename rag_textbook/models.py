"""Доменные типы, общие для всех слоёв.

Ключевое отличие от прежней схемы: у чанка есть поля ``latex_fragments`` и
``table_fragments``. Раньше описание от модели зрения **замещало** исходное
представление формулы или таблицы, и в индекс попадал пересказ вместо самой формулы.
Теперь исходное представление сохраняется всегда, а описание идёт дополнением.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BlockType = Literal["text", "title", "image", "chart", "table", "equation", "discarded", "other"]
SPECIAL_BLOCK_TYPES: frozenset[str] = frozenset({"image", "chart", "table", "equation"})

_WS_RE = re.compile(r"\s+")


def normalize_text(value: str) -> str:
    """Схлопывает пробелы и убирает мягкий перенос."""
    if not value:
        return ""
    return _WS_RE.sub(" ", value.replace("­", "")).strip()


def content_hash(*parts: str) -> str:
    digest = hashlib.sha256()
    for part in parts:
        digest.update(str(part).encode("utf-8", errors="ignore"))
        digest.update(b"\x1f")
    return digest.hexdigest()


class Block(BaseModel):
    """Нормализованный блок из парсера, независимый от версии MinerU."""

    model_config = ConfigDict(extra="ignore")

    index: int = 0
    type: BlockType = "text"
    text: str = ""
    page_idx: int | None = None
    bbox: list[float] | None = None
    text_level: int | None = None
    img_path: str | None = None
    caption: str = ""
    footnote: str = ""
    table_html: str = ""
    latex: str = ""
    # Описание от модели зрения. Дополняет, но не заменяет `text` / `latex` / `table_html`.
    enrichment: str = ""

    @property
    def is_special(self) -> bool:
        return self.type in SPECIAL_BLOCK_TYPES

    def to_indexable_text(self, *, include_enrichment: bool = True) -> str:
        """Текст блока для индексации.

        Порядок частей выбран так, чтобы исходное представление шло первым:
        поиск по формуле должен находить саму формулу, а не её пересказ.
        """

        parts: list[str] = []
        if self.type == "equation":
            latex = self.latex or self.text
            if latex:
                parts.append(f"$${normalize_text(latex)}$$")
        elif self.type == "table":
            if self.caption:
                parts.append(normalize_text(self.caption))
            if self.table_html:
                parts.append(normalize_text(html_to_text(self.table_html)))
            if self.footnote:
                parts.append(normalize_text(self.footnote))
        elif self.type in {"image", "chart"}:
            if self.caption:
                parts.append(normalize_text(self.caption))
            if self.footnote:
                parts.append(normalize_text(self.footnote))
        else:
            if self.text:
                parts.append(normalize_text(self.text))

        if include_enrichment and self.enrichment:
            parts.append(normalize_text(self.enrichment))

        return " ".join(part for part in parts if part).strip()

    def enrichment_key(self, prompt_version: str) -> str:
        """Ключ кэша обогащения: одна и та же картинка не описывается дважды."""
        return content_hash(self.type, self.img_path or "", self.caption, self.text, prompt_version)


_TAG_RE = re.compile(r"<[^>]+>")


def html_to_text(html: str) -> str:
    """Плоское представление HTML-таблицы.

    Структура сохраняется разделителями, чтобы лексический канал видел
    значения ячеек, а не слипшуюся строку.
    """
    if not html:
        return ""
    text = html.replace("</td>", " | ").replace("</th>", " | ")
    text = text.replace("</tr>", "\n")
    text = _TAG_RE.sub(" ", text)
    # Убираем разделители, оставшиеся по краям после вырезания тегов.
    return _WS_RE.sub(" ", text).strip().strip("|").strip()


class Chunk(BaseModel):
    """Единица индексации."""

    model_config = ConfigDict(extra="ignore")

    id: str
    doc_id: str
    doc_name: str
    source_path: str
    ordinal: int
    text: str
    pages: list[int] = Field(default_factory=list)
    headers: list[str] = Field(default_factory=list)
    special_types: list[str] = Field(default_factory=list)
    has_formula: bool = False
    has_table: bool = False
    has_figure: bool = False
    char_start: int = 0
    char_end: int = 0
    text_hash: str = ""

    @property
    def primary_page(self) -> int | None:
        return self.pages[0] if self.pages else None

    def citation_label(self) -> str:
        """Ссылка для ответа: без номера страницы цитата бесполезна студенту."""
        if not self.pages:
            return self.doc_name
        if len(self.pages) == 1:
            return f"{self.doc_name}, с. {self.pages[0]}"
        return f"{self.doc_name}, с. {self.pages[0]}–{self.pages[-1]}"

    def payload(self) -> dict[str, Any]:
        """Полезная нагрузка для векторного хранилища."""
        return {
            "chunk_id": self.id,
            "doc_id": self.doc_id,
            "doc_name": self.doc_name,
            "source_path": self.source_path,
            "ordinal": self.ordinal,
            "text": self.text,
            "pages": self.pages,
            "headers": self.headers,
            "special_types": self.special_types,
            "has_formula": self.has_formula,
            "has_table": self.has_table,
            "has_figure": self.has_figure,
            "text_hash": self.text_hash,
        }

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> Chunk:
        return cls(
            id=str(payload.get("chunk_id") or payload.get("id") or ""),
            doc_id=str(payload.get("doc_id") or ""),
            doc_name=str(payload.get("doc_name") or ""),
            source_path=str(payload.get("source_path") or ""),
            ordinal=int(payload.get("ordinal") or 0),
            text=str(payload.get("text") or ""),
            pages=[int(p) for p in (payload.get("pages") or [])],
            headers=[str(h) for h in (payload.get("headers") or [])],
            special_types=[str(s) for s in (payload.get("special_types") or [])],
            has_formula=bool(payload.get("has_formula")),
            has_table=bool(payload.get("has_table")),
            has_figure=bool(payload.get("has_figure")),
            text_hash=str(payload.get("text_hash") or ""),
        )


RetrievalChannel = Literal[
    "dense", "sparse", "hybrid", "graph_entity", "graph_keyword", "fused", "rerank"
]


class ScoredChunk(BaseModel):
    """Чанк с оценкой и происхождением.

    ``channels`` позволяет отвечать на вопрос «сколько документов в финальном
    контексте пришло из графа» — без этого доля графа в ответе неизмерима.
    """

    model_config = ConfigDict(extra="ignore")

    chunk: Chunk
    score: float = 0.0
    channels: list[str] = Field(default_factory=list)
    channel_scores: dict[str, float] = Field(default_factory=dict)
    rerank_score: float | None = None
    matched_entities: list[str] = Field(default_factory=list)

    @property
    def from_graph(self) -> bool:
        return any(channel.startswith("graph") for channel in self.channels)


class Entity(BaseModel):
    """Каноническая сущность графа."""

    model_config = ConfigDict(extra="ignore")

    id: str
    name: str
    canonical: str
    aliases: list[str] = Field(default_factory=list)
    count: int = 0

    @staticmethod
    def make_id(canonical: str) -> str:
        return hashlib.sha1(canonical.encode("utf-8", errors="ignore")).hexdigest()


class Relation(BaseModel):
    """Направленная типизированная связь.

    Именно такие рёбра, а не co-occurrence-клики, должны нести полезный сигнал
    при многохоповом обходе.
    """

    model_config = ConfigDict(extra="ignore")

    source_id: str
    target_id: str
    label: str
    chunk_id: str
    doc_id: str
    weight: float = 1.0


class ExtractionResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    status: str = "ok"
    raw_preview: str = ""


class Citation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    index: int
    doc_name: str
    pages: list[int] = Field(default_factory=list)
    chunk_id: str = ""
    label: str = ""
    from_graph: bool = False


class Answer(BaseModel):
    model_config = ConfigDict(extra="ignore")

    question: str
    rewritten_question: str = ""
    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    contexts: list[ScoredChunk] = Field(default_factory=list)
    used_graph: bool = False
    timings_ms: dict[str, float] = Field(default_factory=dict)


class GoldQuestion(BaseModel):
    """Вопрос эталонного набора.

    ``gold_chunk_ids`` — то, чего не было в прежнем наборе: без идентификаторов
    эталонных фрагментов метрику Recall@k посчитать невозможно.
    """

    model_config = ConfigDict(extra="ignore")

    id: str
    question: str
    gold_chunk_ids: list[str]
    gold_doc_ids: list[str] = Field(default_factory=list)
    answer: str = ""
    question_type: Literal["single_chunk", "multi_hop", "relation", "formula_table"] = (
        "single_chunk"
    )
    expected_hops: int = 1
    generator_model: str = ""
    verified: bool = False
    notes: str = ""
