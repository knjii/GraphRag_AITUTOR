"""Демо-сервис для веб-интерфейса «Матчасть».

Отдельный модуль, а не часть пакета: он только читает рабочий код
``rag_textbook`` и ничего в нём не меняет. Главное, что он добавляет, —
трассу поиска событиями. Пока конвейер идёт, интерфейс видит, какие понятия
нашлись в вопросе, куда ушёл обход графа, какие фрагменты принёс каждый
канал и какие из них отобраны в ответ.

Трасса снимается наблюдением, а не правкой конвейера: на каждый запрос
собирается свой ``RetrievalPipeline``. У этого экземпляра стадии обёрнуты
так, чтобы сообщать о себе, а хранилище графа подменено записывающей
обёрткой. Поведение поиска при этом то же, что у сервиса. Обёртки только
пересказывают, что прошло через стадию. Исключение одно — фильтр документов,
отключённых пользователем: он отбрасывает их фрагменты на выходе каналов.

Запуск:

* сервер: ``uvicorn demo_api:app`` из ``web/server`` при ``PYTHONPATH``
  на корень репозитория и прежних переменных окружения сервиса
  (Qdrant, Neo4j или файл графа, эмбеддинги, реранкер, модель);
* ноутбук без служб: ``DEMO_OFFLINE_GRAPH=artifacts/graphs/v4.json.gz``.
  Фрагменты и граф берутся из файла, поиск по словам идёт в памяти,
  вместо модели работает выписка из найденного (это видно в интерфейсе).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import threading
import time
from collections import defaultdict
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from rag_textbook.config import Settings
from rag_textbook.generation.answering import build_answer_messages, extract_citations
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.graph_retriever import GraphRetriever
from rag_textbook.retrieval.pipeline import RetrievalPipeline, RetrievalResult

HERE = Path(__file__).resolve().parent
MODE = os.environ.get("DEMO_MODE", "local")  # local | server
OFFLINE_GRAPH = os.environ.get("DEMO_OFFLINE_GRAPH", "")
TITLES_FILE = Path(os.environ.get("DEMO_TITLES", HERE / "titles.json"))
STATIC_DIR = Path(os.environ.get("DEMO_STATIC", HERE.parent / "dist"))

# Сколько узлов и фрагментов отдавать сцене. Больше на экране не прочесть,
# а сам поиск этим не ограничивается.
# Сцена — не полный обход, а его объяснение: показываем только траектории,
# которые привели к фрагментам. Больше узлов глаз не разбирает.
SCENE_SEEDS = 6
SCENE_NEIGHBOURS = 8
SCENE_CANDIDATES = 24


# ------------------------------------------------------------------ режимы


@dataclass(frozen=True)
class Preset:
    id: str
    name: str
    hint: str
    settings: dict[str, Any]


# Сбалансированный режим повторяет принятую конфигурацию продукта:
# маршрут всегда в граф и выдача 16 (docs/HYPOTHESES.md, «Маршрут всегда в граф»).
PRESETS = [
    Preset(
        "balanced",
        "Сбалансированный",
        "Поиск по смыслу, словам и графу понятий, затем переранжирование.",
        {"topK": 16, "graphWeight": 0.4, "router": "always", "reranker": True, "selection": "off"},
    ),
    Preset(
        "setr",
        "Точный отбор",
        "Модель отбирает фрагменты набором (SetR). Лучше на вопросах, связывающих разделы.",
        {"topK": 16, "graphWeight": 0.4, "router": "always", "reranker": True, "selection": "setr"},
    ),
    # Лучшее измеренное качество: SEAL с буфером 5, ровно как в межкнижной
    # проверке (ctx_recall +0.075 [+0.025; +0.125]) и на MuSiQue (EM +0.13).
    Preset(
        "seal",
        "Лучшее качество",
        "Дособирает недостающие звенья дополнительными запросами (SEAL) и отвечает по пяти фрагментам. "
        "Лучший по замерам режим, но в несколько раз медленнее.",
        {"topK": 5, "graphWeight": 0.4, "router": "always", "reranker": True, "selection": "seal"},
    ),
]

ROUTER_MODES = {"auto": "heuristic", "always": "always", "off": "never"}


class UiSettings(BaseModel):
    topK: int = Field(default=16, ge=3, le=16)
    graphWeight: float = Field(default=0.4, ge=0.0, le=1.0)
    router: str = Field(default="always", pattern="^(auto|always|off)$")
    reranker: bool = True
    selection: str = Field(default="off", pattern="^(off|setr|seal)$")
    model: str = ""


class AskBody(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    sourceIds: list[str] = Field(default_factory=list)
    presetId: str | None = None
    settings: UiSettings | None = None


# ------------------------------------------------------------------ службы


@dataclass
class Services:
    settings: Settings
    vector_store: Any
    embeddings: Any
    reranker: Any
    graph_store: Any | None
    llm: Any | None
    generator: str  # llm | extractive
    catalogue: Catalogue


@dataclass
class Document:
    id: str
    name: str
    title: str
    authors: str
    chunks: list[Chunk] = field(default_factory=list)

    @property
    def pages(self) -> int:
        return max((page for chunk in self.chunks for page in chunk.pages), default=0)

    @property
    def formulas(self) -> int:
        return sum(len(_FORMULA_RE.findall(chunk.text)) for chunk in self.chunks)


_FORMULA_RE = re.compile(r"\$\$.+?\$\$|\$[^$\n]+\$", re.S)


class Catalogue:
    """Документы библиотеки и их фрагменты по порядку.

    Читается один раз: просмотрщику нужны соседние фрагменты страницы,
    а векторное хранилище по порядку внутри документа не ищет.
    """

    def __init__(self, chunks: Sequence[Chunk], titles: dict[str, dict[str, str]]) -> None:
        documents: dict[str, Document] = {}
        for chunk in chunks:
            doc = documents.get(chunk.doc_id)
            if doc is None:
                meta = titles.get(chunk.doc_id) or titles.get(chunk.doc_name) or {}
                doc = documents[chunk.doc_id] = Document(
                    id=chunk.doc_id,
                    name=chunk.doc_name,
                    title=meta.get("title") or _pretty_name(chunk.doc_name),
                    authors=meta.get("authors", ""),
                )
            doc.chunks.append(chunk)
        for doc in documents.values():
            doc.chunks.sort(key=lambda item: item.ordinal)
        self.documents = documents
        self.by_chunk = {chunk.id: chunk for doc in documents.values() for chunk in doc.chunks}


def _pretty_name(doc_name: str) -> str:
    """Имя файла без даты выгрузки: «Matematika_v_mashinnom_241126_230954» → «Matematika v mashinnom»."""
    name = re.sub(r"(_\d{6}){1,2}$", "", doc_name or "")
    return name.replace("_", " ").strip() or "Без названия"


def _load_titles() -> dict[str, dict[str, str]]:
    try:
        return json.loads(TITLES_FILE.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_services() -> Services:
    settings = Settings()
    titles = _load_titles()
    if OFFLINE_GRAPH:
        return _offline_services(settings, Path(OFFLINE_GRAPH), titles)

    from rag_textbook.context import build_context

    context = build_context(settings)
    catalogue = Catalogue(list(context.vector_store.iter_chunks()), titles)
    return Services(
        settings=settings,
        vector_store=context.vector_store,
        embeddings=context.embeddings,
        reranker=context.reranker,
        graph_store=context.graph_store,
        llm=context.llm,
        generator="llm",
        catalogue=catalogue,
    )


def _offline_services(settings: Settings, path: Path, titles: dict) -> Services:
    """Всё из файла графа: показать сцену на ноутбуке без Qdrant, Neo4j и модели."""
    from rag_textbook.clients.embeddings import FakeEmbeddingClient
    from rag_textbook.clients.reranker import FakeRerankerClient
    from rag_textbook.stores.graph_file import MemoryGraphStore
    from rag_textbook.stores.vector_store import InMemoryVectorStore

    store = MemoryGraphStore.from_file(path)
    chunks = [
        Chunk(
            id=passage["id"],
            doc_id=passage.get("doc_id") or "",
            doc_name=passage.get("doc_name") or "",
            source_path="",
            ordinal=int(passage.get("ordinal") or 0),
            text=passage.get("text") or "",
            pages=list(passage.get("pages") or []),
        )
        for passage in store.graph.passages.values()
    ]
    # Эмбеддинги хэшевые: смысла в них нет, поиск держится на словах
    # (BM25 в памяти) и графе. Для показа обхода этого достаточно.
    embeddings = FakeEmbeddingClient()
    vectors = InMemoryVectorStore()
    vectors.upsert(chunks, embeddings.embed_documents([chunk.text for chunk in chunks]))
    graph = settings.graph.model_copy(
        update={"backend": "memory", "graph_file": path, "enabled": True, "retrieval_enabled": True}
    )
    settings = settings.model_copy(update={"graph": graph})
    return Services(
        settings=settings,
        vector_store=vectors,
        embeddings=embeddings,
        reranker=FakeRerankerClient(),
        graph_store=store,
        llm=None,
        generator="extractive",
        catalogue=Catalogue(chunks, titles),
    )


# ------------------------------------------------------------ источник из вопроса

# Слова, после которых название в вопросе — это ссылка на книгу, а не тема:
# «что такое теория вероятностей» не должно сужать поиск до Черновой.
_BOOK_CUE = re.compile(r"учебник|книг|пособи|лекци|курс|конспект|источник|у автора", re.I)
_QUOTED = re.compile(r"[«\"“„]([^»\"”]{3,120})[»\"”]")
_LECTURE = re.compile(r"лекци\w*\s*(?:№\s*)?(\d{1,2})\b|\b(\d{1,2})\s*-?\s*(?:я|й|ой|ей)?\s+лекци", re.I)
_STRUCTURE = re.compile(r"раздел|глав[аыуе]|оглавлен|содержани|структур|из чего состоит|какие темы|о ч[её]м", re.I)


def _stems(text: str) -> list[str]:
    return [word[:5] for word in re.findall(r"\w{3,}", text.lower())]


def _surnames(authors: str) -> list[str]:
    """«И. М. Гельфанд» → «гельфан»; «Дайзенрот, Фейсал, Он» → каждая фамилия длиннее трёх букв."""
    names = []
    for part in authors.split(","):
        words = [w for w in re.findall(r"[А-ЯЁA-Z][а-яёa-z]+", part) if len(w) > 3]
        if words:
            surname = words[-1].lower()
            names.append(surname[:-1] if len(surname) > 5 else surname)
    return names


def resolve_scope(question: str, documents: Sequence[Document]) -> tuple[list[Document], str]:
    """Источники, названные в вопросе: кавычками, автором, номером лекции или названием после «учебник»."""
    lowered = question.lower()
    found: dict[str, tuple[Document, str]] = {}

    def title_hits(text: str) -> list[Document]:
        words = set(_stems(text))
        scored = []
        for doc in documents:
            title = [w for w in _stems(re.sub(r"Лекция \d+:.*", "", doc.title)) if len(w) >= 4]
            matched = sum(w in words for w in title)
            if title and matched / len(title) >= 0.75:
                scored.append((matched, doc))
        # «Математика в машинном обучении» накрывает и «Машинное обучение»:
        # побеждает название, совпавшее большим числом слов.
        best = max((matched for matched, _ in scored), default=0)
        return [doc for matched, doc in scored if matched == best]

    for quoted in _QUOTED.findall(question):
        for doc in title_hits(quoted):
            found.setdefault(doc.id, (doc, "название в кавычках"))
    for doc in documents:
        if any(re.search(rf"\b{re.escape(name)}", lowered) for name in _surnames(doc.authors)):
            found.setdefault(doc.id, (doc, "назван автор"))
    numbers = {int(a or b) for a, b in _LECTURE.findall(question)}
    if numbers:
        for doc in documents:
            match = re.search(r"Лекция (\d+):", doc.title)
            if match and int(match.group(1)) in numbers:
                found[doc.id] = (doc, "назван номер лекции")
    if not found and _BOOK_CUE.search(question):
        for doc in title_hits(question):
            found.setdefault(doc.id, (doc, "назван учебник"))
    # Номер лекции уточняет автора: «лекция 5 Соколова» — одна лекция, не все двенадцать.
    if numbers and any(reason == "назван номер лекции" for _, reason in found.values()):
        found = {key: value for key, value in found.items() if value[1] != "назван автор" or "Лекция" not in value[0].title}
    if not found or len(found) == len(documents):
        return [], ""
    docs = [doc for doc, _ in found.values()]
    return docs, next(iter(found.values()))[1]


def _outline(doc: Document, limit: int = 60) -> str:
    """Оглавление из заголовков фрагментов: что разбор PDF распознал как разделы."""
    seen: dict[str, int | None] = {}
    for chunk in doc.chunks:
        for header in chunk.headers:
            header = re.sub(r"\s+", " ", header).strip()
            if header and header not in seen:
                seen[header] = chunk.pages[0] if chunk.pages else None
    lines = [f"- {header}" + (f" (с. {page})" if page else "") for header, page in seen.items()]
    if len(lines) > limit:
        lines = lines[:limit] + [f"- … и ещё {len(lines) - limit}"]
    return "\n".join(lines)


def _doc_label(doc: Document) -> str:
    return f"«{doc.title}»" + (f", {doc.authors}" if doc.authors else "")


def library_prompt(library: Sequence[Document], scope: Sequence[Document], question: str) -> str:
    """Что модель должна знать о библиотеке, чтобы понимать «в учебнике Иванова»."""
    lines = [
        "",
        "Библиотека пользователя — источники, по которым ведётся поиск:",
        *[f"- {_doc_label(doc)}" for doc in sorted(library, key=lambda doc: doc.title)],
        "Заголовок каждого фрагмента контекста называет его источник. Если вопрос "
        "называет книгу, автора или лекцию, отвечай по фрагментам этого источника. "
        "Не приписывай источникам других авторов и не привлекай сведения о книгах "
        "не из этого списка.",
    ]
    if scope:
        lines.append("Пользователь спрашивает про: " + "; ".join(_doc_label(doc) for doc in scope) + ".")
        if _STRUCTURE.search(question):
            for doc in scope:
                lines += [
                    f"Оглавление {_doc_label(doc)} — заголовки разделов в порядке следования "
                    "(справка каталога, не фрагмент: ссылку [номер] на него не ставь):",
                    _outline(doc),
                ]
    return "\n".join(lines)


# ------------------------------------------------------------ трасса поиска

Emit = Callable[[str, dict[str, Any]], None]


def _channel_of(item: ScoredChunk) -> str:
    """Канал для интерфейса: граф, если без графа фрагмента бы не было."""
    if item.only_from_graph:
        return "graph"
    if "sparse" in item.channels and "dense" not in item.channels:
        return "sparse"
    return "dense"


def _relation_label(label: str) -> str:
    return (label or "").replace("_", " ")


class GraphDescriber:
    """Имена узлов и связи между ними — то, чего обход наружу не отдаёт."""

    def __init__(self, store: Any) -> None:
        self.store = store
        self._edges: dict[str, list[tuple[str, str]]] | None = None
        self._lock = threading.Lock()

    def describe(self, ids: Sequence[str]) -> tuple[dict[str, dict], list[dict]]:
        ids = list(dict.fromkeys(str(item) for item in ids))
        if not ids:
            return {}, []
        graph = getattr(self.store, "graph", None)
        if graph is not None:
            return self._from_memory(graph, ids)
        return self._from_neo4j(ids)

    def _from_memory(self, graph: Any, ids: list[str]) -> tuple[dict[str, dict], list[dict]]:
        with self._lock:
            if self._edges is None:
                edges: dict[str, list[tuple[str, str]]] = defaultdict(list)
                for source, target, rel_type, label, _ in graph.relations:
                    if rel_type == "RELATES":
                        edges[source].append((target, label))
                self._edges = edges
        wanted = set(ids)
        nodes = {
            entity_id: {
                "name": graph.entities[entity_id].get("name") or "",
                "canonical": graph.entities[entity_id].get("canonical") or "",
                "kind": graph.entities[entity_id].get("kind") or "concept",
            }
            for entity_id in ids
            if entity_id in graph.entities
        }
        links = [
            {"source": source, "target": target, "label": _relation_label(label)}
            for source in ids
            for target, label in self._edges.get(source, ())
            if target in wanted and target != source
        ]
        return nodes, links

    def _from_neo4j(self, ids: list[str]) -> tuple[dict[str, dict], list[dict]]:
        session_factory = getattr(self.store, "_session", None)
        if session_factory is None:
            return {}, []
        with session_factory() as session:
            rows = session.run(
                "MATCH (e:Entity) WHERE e.id IN $ids "
                "RETURN e.id AS id, e.name AS name, e.canonical AS canonical, "
                "coalesce(e.kind, 'concept') AS kind",
                {"ids": ids},
            ).data()
            edges = session.run(
                "MATCH (a:Entity)-[r:RELATES]->(b:Entity) "
                "WHERE a.id IN $ids AND b.id IN $ids AND a <> b "
                "RETURN a.id AS source, b.id AS target, r.label AS label LIMIT 200",
                {"ids": ids},
            ).data()
        nodes = {
            str(row["id"]): {
                "name": row.get("name") or row.get("canonical") or "",
                "canonical": row.get("canonical") or "",
                "kind": row.get("kind") or "concept",
            }
            for row in rows
        }
        links = [
            {"source": row["source"], "target": row["target"], "label": _relation_label(row.get("label"))}
            for row in edges
        ]
        return nodes, links


class RecordingStore:
    """Хранилище графа, которое пересказывает обход, не меняя его."""

    def __init__(self, store: Any, tracer: Tracer) -> None:
        self._store = store
        self._tracer = tracer

    def __getattr__(self, name: str) -> Any:
        return getattr(self._store, name)

    def find_seed_entities(self, terms: Sequence[str], limit: int) -> list[dict[str, Any]]:
        rows = self._store.find_seed_entities(terms, limit)
        self._tracer.on_seeds(rows, origin="question", terms=list(terms))
        return rows

    def entities_of_passages(self, chunk_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        rows = self._store.entities_of_passages(chunk_ids, limit)
        self._tracer.on_seeds(rows, origin="passages", terms=[])
        return rows

    def expand_entities(self, seed_ids: Sequence[str], *args: Any, **kwargs: Any) -> dict[str, float]:
        weights = self._store.expand_entities(seed_ids, *args, **kwargs)
        self._tracer.on_expand(list(seed_ids), weights)
        return weights


class Tracer:
    """Собирает конвейер одного запроса и сообщает о каждой его стадии."""

    def __init__(self, services: Services, describer: GraphDescriber | None, emit: Emit, excluded: set[str]) -> None:
        self.services = services
        self.describer = describer
        self.emit = emit
        self.excluded = excluded
        self.hop = 0  # 0 — основной запрос, дальше микрозапросы SEAL
        self.seed_ids: list[str] = []
        # Обход графа копится до конца графового канала: что показать,
        # решают фрагменты, к которым он привёл, а они известны только в конце.
        self._seeds: list[tuple[list[dict], str, list[str]]] = []
        self._expands: list[tuple[list[str], dict[str, float]]] = []

    # --- граф

    def on_seeds(self, rows: Sequence[dict], *, origin: str, terms: list[str]) -> None:
        self._seeds.append((list(rows), origin, terms))

    def on_expand(self, seed_ids: list[str], weights: dict[str, float]) -> None:
        self._expands.append((list(seed_ids), dict(weights)))

    def _flush_graph(self, items: Sequence[ScoredChunk]) -> None:
        """Показывает обход, оставив только понятия, которые привели к кандидатам."""
        seeds, expands = self._seeds, self._expands
        self._seeds, self._expands = [], []
        if not seeds and not expands:
            return
        contribution: dict[str, int] = defaultdict(int)
        for item in items[:SCENE_CANDIDATES]:
            for name in item.matched_entities:
                contribution[name.lower()] += 1

        all_ids = [str(row.get("id")) for rows, _, _ in seeds for row in rows]
        nodes, _ = self.describer.describe(all_ids) if self.describer else ({}, [])

        def gain(entity_id: str, row: dict) -> int:
            node = nodes.get(entity_id, {})
            names = {
                str(node.get("canonical") or "").lower(),
                str(node.get("name") or "").lower(),
                str(row.get("canonical") or row.get("name") or "").lower(),
            }
            return max((contribution.get(name, 0) for name in names if name), default=0)

        # Затравки: сперва те, что привели к фрагментам; из вопроса — раньше,
        # чем из найденного текста. Пустую сцену не оставляем: минимум три.
        ranked = []
        for order, (rows, origin, _) in enumerate(seeds):
            for position, row in enumerate(rows):
                entity_id = str(row.get("id"))
                ranked.append((-gain(entity_id, row), order, position, entity_id, row, origin))
        ranked.sort(key=lambda entry: entry[:3])
        picked: list[tuple] = []
        seen: set[str] = set()
        for entry in ranked:
            if entry[3] in seen:
                continue
            if len(picked) >= SCENE_SEEDS or (entry[0] == 0 and len(picked) >= 3):
                break
            seen.add(entry[3])
            picked.append(entry)
        self.seed_ids = [entry[3] for entry in picked]
        for rows, origin, terms in seeds:
            mine = [(entry[3], entry[4]) for entry in picked if entry[5] == origin]
            if mine or origin == "question":
                self._emit_seeds(mine, nodes, origin, terms, total=len(rows))
        for seed_ids, weights in expands:
            self._emit_expand(seed_ids, weights, contribution)

    def _emit_seeds(self, picked: list[tuple[str, dict]], nodes: dict, origin: str, terms: list[str], *, total: int) -> None:
        key = "score" if origin == "question" else "weight"
        top = max((float(row.get(key) or 0.0) for _, row in picked), default=0.0) or 1.0
        self.emit(
            "seeds",
            {
                "hop": self.hop,
                "origin": origin,
                "terms": terms[:12],
                "total": total,
                "entities": [
                    {
                        "id": entity_id,
                        "name": nodes.get(entity_id, {}).get("name") or str(row.get("name") or row.get("canonical") or ""),
                        "canonical": nodes.get(entity_id, {}).get("canonical") or str(row.get("canonical") or ""),
                        "kind": nodes.get(entity_id, {}).get("kind", "concept"),
                        "weight": round(float(row.get(key) or 0.0) / top, 3),
                    }
                    for entity_id, row in picked
                ],
            },
        )

    def _emit_expand(self, seed_ids: list[str], weights: dict[str, float], contribution: dict[str, int]) -> None:
        seeds = set(seed_ids)
        neighbours = sorted(
            ((entity_id, weight) for entity_id, weight in weights.items() if entity_id not in seeds),
            key=lambda item: (-item[1], item[0]),
        )
        shown_seeds = self.seed_ids or seed_ids[:SCENE_SEEDS]
        if self.describer is None:
            return
        # Соседей десятки. Оставляем тех, через кого граф дошёл до фрагментов,
        # и среди них первыми — связанных с показанными затравками.
        pool = [entity_id for entity_id, _ in neighbours[: SCENE_NEIGHBOURS * 8]]
        nodes, links = self.describer.describe(shown_seeds + pool)
        shown = set(shown_seeds)

        def gain(entity_id: str) -> int:
            node = nodes.get(entity_id, {})
            return max(
                contribution.get(str(node.get("canonical") or "").lower(), 0),
                contribution.get(str(node.get("name") or "").lower(), 0),
            )

        def linked(entity_id: str) -> bool:
            return any(
                (link["source"] == entity_id and link["target"] in shown)
                or (link["target"] == entity_id and link["source"] in shown)
                for link in links
            )

        useful = sorted((e for e in pool if gain(e) > 0), key=lambda e: (not linked(e), -gain(e)))
        quiet = [e for e in pool if gain(e) == 0 and linked(e)]
        chosen = (useful + quiet[: max(0, 4 - len(useful))])[:SCENE_NEIGHBOURS]
        keep = shown | set(chosen)
        self.emit(
            "expand",
            {
                "hop": self.hop,
                "total": len(neighbours),
                "entities": [
                    {
                        "id": entity_id,
                        "name": nodes.get(entity_id, {}).get("name", ""),
                        "canonical": nodes.get(entity_id, {}).get("canonical", ""),
                        "kind": nodes.get(entity_id, {}).get("kind", "concept"),
                        "weight": round(float(weights.get(entity_id, 0.0)), 3),
                    }
                    for entity_id in chosen
                    if entity_id in nodes
                ],
                "links": [link for link in links if link["source"] in keep and link["target"] in keep],
            },
        )

    # --- конвейер

    def _keep(self, items: list[ScoredChunk]) -> list[ScoredChunk]:
        if not self.excluded:
            return items
        return [item for item in items if item.chunk.doc_id not in self.excluded]

    def _candidates(self, channel: str, items: Sequence[ScoredChunk]) -> None:
        self.emit(
            "candidates",
            {
                "hop": self.hop,
                "channel": channel,
                "total": len(items),
                "items": [_fragment_out(item, rank) for rank, item in enumerate(items[:SCENE_CANDIDATES])],
            },
        )

    def build(self, settings: Settings) -> RetrievalPipeline:
        services = self.services
        retriever = None
        if services.graph_store is not None and settings.graph.retrieval_enabled:
            retriever = GraphRetriever(settings.graph, RecordingStore(services.graph_store, self))
        pipeline = RetrievalPipeline(
            settings=settings,
            vector_store=services.vector_store,
            embedding_client=services.embeddings,
            reranker=services.reranker,
            graph_retriever=retriever,
            llm=services.llm,
        )
        # Отбор по рёбрам ходит в хранилище напрямую: ему запись не нужна.
        pipeline.link_store = services.graph_store

        route, base, graph, rerank, micro = (
            pipeline.router.route,
            pipeline._base_channel,
            pipeline._graph_channel,
            pipeline._rerank,
            pipeline._micro_search,
        )

        def traced_route(question: str):
            self.emit("stage", {"id": "route", "state": "start"})
            decision = route(question)
            self.emit("route", {"useGraph": bool(decision.use_graph), "reason": str(decision.reason)})
            return decision

        def traced_base(query: str, top_k: int | None = None):
            self.emit("stage", {"id": "text", "state": "start", "hop": self.hop})
            items = self._keep(base(query, top_k))
            self._candidates("text", items)
            return items

        def traced_graph(query: str, base_items):
            self.emit("stage", {"id": "graph", "state": "start", "hop": self.hop})
            items = self._keep(graph(query, base_items))
            self._flush_graph(items)
            self._candidates("graph", items)
            return items

        def traced_rerank(query: str, items, top_k: int | None = None):
            self.emit("stage", {"id": "rerank", "state": "start", "hop": self.hop})
            ranked = rerank(query, items, top_k)
            self.emit(
                "rerank",
                {
                    "hop": self.hop,
                    "order": [item.chunk.id for item in ranked[:SCENE_CANDIDATES]],
                },
            )
            return ranked

        def traced_micro(query: str):
            self.hop += 1
            self.emit("seal_hop", {"hop": self.hop, "query": query})
            try:
                return micro(query)
            finally:
                self.emit("stage", {"id": "selection", "state": "start"})

        pipeline.router.route = traced_route  # type: ignore[method-assign]
        pipeline._base_channel = traced_base  # type: ignore[method-assign]
        pipeline._graph_channel = traced_graph  # type: ignore[method-assign]
        pipeline._rerank = traced_rerank  # type: ignore[method-assign]
        pipeline._micro_search = traced_micro  # type: ignore[method-assign]
        return pipeline


def _fragment_out(item: ScoredChunk, rank: int | None = None) -> dict[str, Any]:
    chunk = item.chunk
    return {
        "id": chunk.id,
        "sourceId": chunk.doc_id,
        "page": chunk.pages[0] if chunk.pages else None,
        "section": chunk.headers[-1] if chunk.headers else "",
        "preview": re.sub(r"\s+", " ", chunk.text)[:160],
        "channel": _channel_of(item),
        "channels": list(item.channels),
        "entities": list(item.matched_entities[:4]),
        "rank": rank,
    }


# ------------------------------------------------------------ настройки


def resolve_settings(base: Settings, body: AskBody) -> tuple[Settings, UiSettings, str | None]:
    preset = next((p for p in PRESETS if p.id == body.presetId), None)
    if MODE == "server" or body.settings is None:
        # На сервере параметры задаёт только готовый режим: показ не должен
        # зависеть от того, что накрутил предыдущий зритель.
        preset = preset or PRESETS[0]
        ui = UiSettings(**preset.settings)
    else:
        ui = body.settings

    retrieval = base.retrieval.model_copy(
        update={
            "top_k": ui.topK,
            "top_k_linking": 0,
            "router_enabled": ui.router != "off",
            "router_mode": ROUTER_MODES[ui.router],
            "selection_mode": ui.selection,
        }
    )
    graph = base.graph.model_copy(update={"weight": ui.graphWeight})
    reranker = base.reranker.model_copy(update={"enabled": ui.reranker})
    settings = base.model_copy(update={"retrieval": retrieval, "graph": graph, "reranker": reranker})
    return settings, ui, preset.id if preset else None


# ------------------------------------------------------------ генерация


def stream_answer(
    services: Services,
    settings: Settings,
    question: str,
    chunks: list[ScoredChunk],
    emit: Emit,
    *,
    library: Sequence[Document] = (),
    scope: Sequence[Document] = (),
) -> tuple[list[ScoredChunk], str, float]:
    started = time.perf_counter()
    if services.llm is None:
        ordered, text = _extractive_answer(question, chunks)
        for partial in _word_prefixes(text):
            emit("token", {"text": partial})
            time.sleep(0.012)
        return ordered, text, (time.perf_counter() - started) * 1000

    # В заголовке фрагмента — название и автор вместо имени файла: иначе модель
    # видит «ru-rl-ivanov» и сама додумывает, что это за книга.
    documents = services.catalogue.documents
    labelled = [
        item.model_copy(update={"chunk": item.chunk.model_copy(update={"doc_name": _doc_label(documents[item.chunk.doc_id])})})
        if item.chunk.doc_id in documents
        else item
        for item in chunks
    ]
    ordered, messages = build_answer_messages(settings, question, labelled)
    if library:
        messages[0] = messages[0].model_copy(
            update={"content": messages[0].content + "\n" + library_prompt(library, scope, question)}
        )
    text = ""
    try:
        for text in _stream_llm(services.llm, messages):
            emit("token", {"text": text})
    except Exception as exc:  # noqa: BLE001
        if not text:
            # Потоковый путь не поддержан движком — отвечаем целиком.
            text = services.llm.chat(messages, purpose="chat")
            emit("token", {"text": text})
        else:
            raise RuntimeError(f"генерация оборвалась: {exc}") from exc
    return ordered, text, (time.perf_counter() - started) * 1000


def _stream_llm(llm: Any, messages: list) -> Iterator[str]:
    """Тот же запрос, что ``llm.chat``, но с ``stream``: модель печатает на глазах."""
    import httpx

    from rag_textbook.clients.llm import strip_reasoning

    payload = llm._payload(messages, "chat", None, None, None)
    payload["stream"] = True
    url = f"{llm.settings.base_url_for('chat')}/chat/completions"
    raw = ""
    with httpx.Client(timeout=llm.settings.timeout_seconds, headers=llm._headers) as client:
        with client.stream("POST", url, json=payload) as response:
            if response.status_code >= 400:
                raise RuntimeError(f"LLM вернул {response.status_code}")
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                delta = (json.loads(data).get("choices") or [{}])[0].get("delta") or {}
                piece = delta.get("content") or ""
                if piece:
                    raw += piece
                    shown = strip_reasoning(raw) if "<think>" in raw else raw
                    if shown:
                        yield shown


def _extractive_answer(question: str, chunks: list[ScoredChunk]) -> tuple[list[ScoredChunk], str]:
    """Ответ без модели: из трёх первых фрагментов — предложение, ближайшее к вопросу."""
    words = {word[:6] for word in re.findall(r"\w{4,}", question.lower())}
    ordered = list(chunks)
    parts = []
    for index, item in enumerate(ordered[:3], start=1):
        text = re.sub(r"\s+", " ", item.chunk.text).strip()
        # Фрагмент часто начинается с обрывка слова: начинаем с первой заглавной.
        start = re.search(r"[A-ZА-ЯЁ]", text)
        sentences = re.split(r"(?<=[.!?])\s+", text[start.start() if start else 0 :])
        best = max(
            [sentence for sentence in sentences if len(sentence) > 30] or sentences,
            key=lambda sentence: len(words & {word[:6] for word in re.findall(r"\w{4,}", sentence.lower())}),
        )
        parts.append(f"{best[:320].rstrip()} [{index}]")
    text = (
        "Модель ответа в этом запуске не подключена, поэтому вместо ответа выписка из найденного.\n\n"
        + "\n\n".join(parts)
    )
    return ordered, text


def _word_prefixes(text: str) -> Iterator[str]:
    position = 0
    for match in re.finditer(r"\S+\s*", text):
        position = match.end()
        yield text[:position]


# ------------------------------------------------------------ приложение

state: dict[str, Any] = {}


def services() -> Services:
    if "services" not in state:
        raise HTTPException(status_code=503, detail="Сервис ещё поднимается")
    return state["services"]


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    state["services"] = await asyncio.to_thread(build_services)
    srv: Services = state["services"]
    state["describer"] = GraphDescriber(srv.graph_store) if srv.graph_store is not None else None
    state["semaphore"] = asyncio.Semaphore(int(os.environ.get("DEMO_MAX_CONCURRENT", "2")))
    yield


app = FastAPI(title="Матчасть — демо", lifespan=lifespan)


def _served_model(llm: Any) -> str:
    """Какая модель на самом деле отвечает: имя в настройках может отставать.

    llama.cpp принимает любое имя в запросе и отвечает своей моделью, поэтому
    показываем то, что сервер называет сам, а настройку — только если он молчит.
    """
    import httpx

    configured = llm.settings.model_for("chat")
    try:
        body = httpx.get(f"{llm.settings.base_url_for('chat').rstrip('/')}/models", timeout=1.5).json()
        entries = body.get("data") or body.get("models") or []
        name = entries[0].get("id") or entries[0].get("name") or ""
    except Exception:  # noqa: BLE001 — подпись в шапке, не повод падать
        return configured
    name = re.sub(r"\.gguf$", "", name.rstrip("/").rsplit("/", 1)[-1])
    return name or configured


@app.get("/api/info")
def info() -> dict[str, Any]:
    srv = services()
    model = _served_model(srv.llm) if srv.llm is not None else "без модели (выписка)"
    selections = ["off", "setr", "seal"] if srv.llm is not None else ["off"]
    presets = [p for p in PRESETS if p.settings["selection"] in selections]
    return {
        "mode": MODE,
        "models": [model],
        "generator": srv.generator,
        "graph": srv.graph_store is not None,
        "uploads": False,
        "selections": selections,
        "presets": [
            {"id": p.id, "name": p.name, "hint": p.hint, "settings": {**p.settings, "model": model}}
            for p in presets
        ],
    }


@app.get("/api/sources")
def sources() -> list[dict[str, Any]]:
    docs = services().catalogue.documents.values()
    return [
        {
            "id": doc.id,
            "title": doc.title,
            "authors": doc.authors,
            "pages": doc.pages,
            "fragments": len(doc.chunks),
            "formulas": doc.formulas,
            "status": "ready",
            "enabled": True,
            "addedAt": "",
        }
        for doc in sorted(docs, key=lambda item: item.name.lower())
    ]


@app.get("/api/sources/{doc_id}/fragments")
def fragments(
    doc_id: str,
    around: str | None = None,
    offset: int = Query(default=0, ge=0),
    limit: int = Query(default=40, ge=1, le=120),
) -> dict[str, Any]:
    doc = services().catalogue.documents.get(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail="Документ не найден")
    if around:
        index = next((i for i, chunk in enumerate(doc.chunks) if chunk.id == around), 0)
        offset = max(0, index - limit // 3)
    window = doc.chunks[offset : offset + limit]
    return {
        "total": len(doc.chunks),
        "offset": offset,
        "items": [
            {
                "id": chunk.id,
                "sourceId": doc.id,
                "page": chunk.pages[0] if chunk.pages else 0,
                "section": chunk.headers[-1] if chunk.headers else "",
                "text": chunk.text,
            }
            for chunk in window
        ],
    }


@app.post("/api/ask")
async def ask(body: AskBody) -> StreamingResponse:
    srv = services()
    settings, ui, preset_id = resolve_settings(srv.settings, body)
    if ui.selection != "off" and srv.llm is None:
        raise HTTPException(status_code=400, detail="Отбор SetR и SEAL требует модель, а она не подключена")
    if ui.selection == "seal" and not ui.reranker:
        raise HTTPException(status_code=400, detail="SEAL оценивает пары реранкером: включите переранжирование")

    known = set(srv.catalogue.documents)
    chosen = set(body.sourceIds) & known
    if not chosen:
        raise HTTPException(status_code=400, detail="Не выбран ни один документ")
    excluded = known - chosen

    loop = asyncio.get_running_loop()
    queue: asyncio.Queue[tuple[str, dict] | None] = asyncio.Queue()

    def emit(kind: str, data: dict[str, Any]) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (kind, data))

    def work() -> None:
        try:
            run_question(srv, settings, body.question, emit, excluded, ui, preset_id)
        except Exception as exc:  # noqa: BLE001
            emit("error", {"message": str(exc)[:300]})
        finally:
            loop.call_soon_threadsafe(queue.put_nowait, None)

    semaphore: asyncio.Semaphore = state["semaphore"]

    async def events():
        async with semaphore:
            task = loop.run_in_executor(None, work)
            while True:
                item = await queue.get()
                if item is None:
                    break
                kind, data = item
                yield f"event: {kind}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
            await task

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def run_question(
    srv: Services,
    settings: Settings,
    question: str,
    emit: Emit,
    excluded: set[str],
    ui: UiSettings,
    preset_id: str | None,
) -> None:
    library = [doc for doc_id, doc in srv.catalogue.documents.items() if doc_id not in excluded]
    scope, reason = resolve_scope(question, library)
    emit("start", {"question": question, "preset": preset_id, "settings": ui.model_dump()})
    if scope:
        # Хранилище не фильтрует по документу, поэтому кандидатов берём шире
        # и отсекаем чужие на выходе каналов — как для выключенных источников.
        scope_ids = {doc.id for doc in scope}
        excluded = excluded | {doc.id for doc in library if doc.id not in scope_ids}
        retrieval = settings.retrieval.model_copy(
            update={
                "dense_candidates": settings.retrieval.dense_candidates * 4,
                "sparse_candidates": settings.retrieval.sparse_candidates * 4,
            }
        )
        graph = settings.graph.model_copy(update={"passage_limit": settings.graph.passage_limit * 4})
        settings = settings.model_copy(update={"retrieval": retrieval, "graph": graph})
        emit("scope", {"sourceIds": [doc.id for doc in scope], "titles": [doc.title for doc in scope], "reason": reason})
    tracer = Tracer(srv, state.get("describer"), emit, excluded)
    pipeline = tracer.build(settings)

    result: RetrievalResult = pipeline.retrieve(question, [])
    final = result.chunks
    emit(
        "selected",
        {
            "ids": [item.chunk.id for item in final],
            "pool": len(result.pool),
            "sealAdded": list(result.seal_added),
            "fragments": [_fragment_out(item) for item in final],
            "retrievalMs": round(result.timings_ms.get("total", 0.0)),
            "timings": result.timings_ms,
        },
    )
    if not final:
        emit("token", {"text": "В отмеченных документах ничего подходящего не нашлось."})
        emit("done", {"citations": [], "contexts": [], "timings": {"retrievalMs": result.timings_ms.get("total", 0.0), "generationMs": 0}, "multiHop": False})
        return

    emit("stage", {"id": "generation", "state": "start"})
    ordered, text, generation_ms = stream_answer(srv, settings, question, final, emit, library=library, scope=scope)
    citations = extract_citations(text, ordered)
    by_id = {item.chunk.id: item for item in ordered}
    cited = [by_id[c.chunk_id] for c in citations if c.chunk_id in by_id]
    emit(
        "done",
        {
            "text": text,
            "contexts": [item.chunk.id for item in ordered],
            "citations": [
                {
                    "n": c.index,
                    "fragmentId": c.chunk_id,
                    "sourceId": by_id[c.chunk_id].chunk.doc_id if c.chunk_id in by_id else "",
                    "page": c.pages[0] if c.pages else 0,
                    "section": (by_id[c.chunk_id].chunk.headers or [""])[-1] if c.chunk_id in by_id else "",
                    "channel": _channel_of(by_id[c.chunk_id]) if c.chunk_id in by_id else "dense",
                }
                for c in citations
            ],
            "timings": {"retrievalMs": result.timings_ms.get("total", 0.0), "generationMs": round(generation_ms)},
            "multiHop": len(cited) >= 2 and any(item.from_graph for item in cited),
        },
    )


if STATIC_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=STATIC_DIR / "assets"), name="assets")

    @app.get("/{path:path}", include_in_schema=False)
    def spa(path: str) -> FileResponse:
        target = STATIC_DIR / path
        if path and target.is_file():
            return FileResponse(target)
        return FileResponse(STATIC_DIR / "index.html")
