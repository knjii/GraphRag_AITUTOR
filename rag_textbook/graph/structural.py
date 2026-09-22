"""Структурный граф учебника (гипотеза К1, docs/HYPOTHESES.md).

Узлы и рёбра строятся правилами, без единого вызова модели:

* узлы — утверждения с номером (определения, теоремы, леммы, следствия,
  примеры, замечания), формулы с номером ``\\tag{}``, разделы, рисунки;
* упоминания — фрагмент, где объект введён, получает роль ``defines``,
  фрагмент со ссылкой на него — ``refers``. Одна ссылка «подставим (2.32)»
  соединяет два фрагмента через общий узел, и обход графа находит второй
  фрагмент от первого теми же запросами, что и на модельном графе;
* вложенность — рёбра ``STRUCT`` «в_разделе» от объекта к разделу и от
  раздела к родительскому разделу. Тип отдельный: в обход по умолчанию
  (``GRAPH_EXPANSION_REL_TYPES``) он не входит, иначе раздел стал бы
  хабом, связывающим всё своё содержимое.

Правила якорей и ссылок общие со скриптом ``crossref_offline.py``
(``rag_textbook.evaluation.crossref``): один и тот же разбор должен давать
одни и те же рёбра в проверке R8 и в графе.

Номера уникальны только внутри книги, поэтому узел — пара книги и номера.

Три графа опыта К1 при одном алгоритме поиска:

* A — ``build_structural``: только структурный;
* B — выгрузка модельного графа (``scripts/graph_export.py``);
* C — ``merge_graphs(B, A)``: объединение.
"""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from typing import Any

from rag_textbook.evaluation.crossref import (
    FIGURE_CAPTION,
    SECTION_HEAD,
    STATEMENT,
    STATEMENT_ANCHOR,
    TAG,
    references_of,
)
from rag_textbook.graph.notation import contains_symbol, find_notations
from rag_textbook.models import Chunk
from rag_textbook.stores.graph_file import GraphFile

STRUCT_REL = "STRUCT"
IN_SECTION = "в_разделе"
KIND_OF = {"equation": "formula", "statement": "statement", "section": "section", "figure": "figure"}
# Имя утверждения в скобках после номера: «Определение 2.10 (векторное
# подпространство)». Оно даёт узлу слова, по которым его найдёт поиск
# затравок; без имени узел находится только от фрагментов.
_NAMED_STATEMENT = re.compile(STATEMENT + r"\s+(\d+\.\d+)\s*\(([^()]{2,80})\)")
_ROLE_RANK = {"defines": 3, "uses": 2, "refers": 2, "mentions": 1, "in_section": 1, "": 0}
# Область обозначения (К3) — раздел, где оно введено. Предел числа
# фрагментов-пользователей не даёт однобуквенному символу стать хабом.
NOTATION_SCOPE_LIMIT = 20


def _entity_id(doc_id: str, kind: str, number: str) -> str:
    return "s:" + hashlib.sha1(f"{doc_id}\x00{kind}\x00{number}".encode()).hexdigest()


def _as_dict(chunk: Chunk | dict[str, Any]) -> dict[str, Any]:
    return chunk.model_dump() if isinstance(chunk, Chunk) else dict(chunk)


def _parent(number: str) -> str | None:
    return number.rsplit(".", 1)[0] if "." in number else None


def _section_of(chunk: dict[str, Any]) -> str | None:
    """Самый глубокий номер раздела в заголовках фрагмента."""
    best = None
    for header in chunk.get("headers") or ():
        match = SECTION_HEAD.match(header)
        if match and (best is None or match.group(1).count(".") >= best.count(".")):
            best = match.group(1)
    return best


def _names(kind: str, number: str, statement_names: dict[str, str]) -> tuple[str, str]:
    """Каноническое имя и отображаемое: по ним узел ищется полнотекстово."""
    if kind == "equation":
        return f"формула {number}", f"({number})"
    if kind == "statement":
        title = statement_names.get(number, "")
        return f"утверждение {number} {title}".strip(), f"{number} {title}".strip()
    if kind == "section":
        return f"раздел {number}", f"раздел {number}"
    return f"рисунок {number}", f"рис. {number}"


def _anchors(chunks: list[dict[str, Any]]) -> dict[str, dict[str, list[str]]]:
    """Как ``crossref.anchors_of``, но с порядком изложения.

    Раздел якорится в первом фрагменте, где встречен его заголовок: иначе
    одно упоминание «раздел 8.2» породило бы куст рёбер.
    """
    found: dict[str, dict[str, list[str]]] = {kind: defaultdict(list) for kind in KIND_OF}
    first_of_section: dict[str, dict[str, Any]] = {}
    for chunk in sorted(chunks, key=lambda item: item["ordinal"]):
        text = chunk["text"]
        for match in TAG.finditer(text):
            found["equation"][match.group(1)].append(chunk["id"])
        for match in STATEMENT_ANCHOR.finditer(text):
            found["statement"][match.group(1)].append(chunk["id"])
        for header in chunk.get("headers") or ():
            match = SECTION_HEAD.match(header)
            if match and match.group(1) not in first_of_section:
                first_of_section[match.group(1)] = chunk
        if chunk.get("has_figure"):
            for match in FIGURE_CAPTION.finditer(text):
                found["figure"][match.group(1)].append(chunk["id"])
    for number, chunk in first_of_section.items():
        found["section"][number].append(chunk["id"])
    return {kind: {number: list(dict.fromkeys(ids)) for number, ids in by.items()} for kind, by in found.items()}


def build_structural(
    chunks: Iterable[Chunk | dict[str, Any]],
    *,
    variant: str = "structural",
    notation: bool = False,
    notation_scope: str = "section",
) -> tuple[GraphFile, dict[str, Any]]:
    """Граф A опыта К1 и отчёт о покрытии по книгам.

    ``notation`` добавляет узлы обозначений по правилам (К3) с областью
    действия в разделе. По умолчанию выключено: К1 и К3 — разные гипотезы,
    и граф К1 не должен молча включать вторую. ``notation_scope`` —
    два варианта опыта К3: ``section`` (область — подраздел введения)
    и ``book`` (глобальные обозначения в пределах книги).
    """
    if notation_scope not in ("section", "book"):
        raise ValueError(f"Неизвестная область обозначений: {notation_scope!r}")
    rows = [_as_dict(chunk) for chunk in chunks]
    graph = GraphFile(variant=variant)
    by_doc: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        graph.add_passage(
            row["id"],
            doc_id=row.get("doc_id", ""),
            doc_name=row.get("doc_name", ""),
            ordinal=int(row.get("ordinal") or 0),
            text=row.get("text", ""),
            pages=row.get("pages") or [],
        )
        by_doc[row.get("doc_id", "")].append(row)

    report: dict[str, Any] = {"variant": variant, "books": {}}
    for doc_id, doc_chunks in sorted(by_doc.items()):
        report["books"][doc_id] = _build_book(graph, doc_id, doc_chunks)
        if notation:
            report["books"][doc_id]["notation"] = _add_notation(
                graph, doc_id, doc_chunks, scope=notation_scope
            )
    report["total"] = graph.summary()
    problems = graph.validate()
    if problems:
        raise ValueError(f"Структурный граф испорчен: {problems[:5]}")
    return graph, report


def _build_book(graph: GraphFile, doc_id: str, chunks: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = _anchors(chunks)
    statement_names: dict[str, str] = {}
    for chunk in chunks:
        for match in _NAMED_STATEMENT.finditer(chunk["text"]):
            statement_names.setdefault(match.group(1), match.group(2).strip())

    created: set[str] = set()

    def ensure(kind: str, number: str) -> str:
        entity_id = _entity_id(doc_id, kind, number)
        if entity_id not in created:
            canonical, name = _names(kind, number, statement_names)
            graph.add_entity(entity_id, canonical=canonical, name=name, kind=KIND_OF[kind])
            created.add(entity_id)
        return entity_id

    for kind, numbers in anchors.items():
        for number, ids in numbers.items():
            entity_id = ensure(kind, number)
            for chunk_id in ids:
                graph.add_mention(chunk_id, entity_id, 1, role="defines")

    seen: Counter[str] = Counter()
    resolved: Counter[str] = Counter()
    touched: set[str] = set()
    for chunk in chunks:
        for kind, number in references_of(chunk):
            seen[kind] += 1
            if number not in anchors.get(kind, {}):
                continue
            resolved[kind] += 1
            entity_id = ensure(kind, number)
            count, role = graph.mentions.get(chunk["id"], {}).get(entity_id, (0, ""))
            graph.add_mention(chunk["id"], entity_id, count + 1, role="refers" if role != "defines" else role)
            touched.add(chunk["id"])

    # Вложенность: объект → раздел, раздел → родитель. Раздел берётся
    # из заголовков фрагмента, где объект введён.
    section_of = {chunk["id"]: _section_of(chunk) for chunk in chunks}
    linked: set[tuple[str, str]] = set()
    for kind, numbers in anchors.items():
        for number, ids in numbers.items():
            source = _entity_id(doc_id, kind, number)
            targets: list[str] = []
            if kind == "section":
                parent = _parent(number)
                if parent and parent in anchors["section"]:
                    targets.append(parent)
            else:
                section = section_of.get(ids[0])
                if section and section in anchors["section"]:
                    targets.append(section)
            for target_number in targets:
                pair = (source, _entity_id(doc_id, "section", target_number))
                if pair not in linked:
                    linked.add(pair)
                    graph.add_relation(pair[0], pair[1], STRUCT_REL, IN_SECTION, 1.0)

    with_mentions = sum(1 for chunk in chunks if graph.mentions.get(chunk["id"]))
    degree = Counter(
        entity_id
        for chunk in chunks
        for entity_id in graph.mentions.get(chunk["id"], {})
    )
    total_seen = sum(seen.values())
    return {
        "chunks": len(chunks),
        "anchors": {kind: len(numbers) for kind, numbers in anchors.items()},
        "references": dict(seen),
        "resolved": dict(resolved),
        # Неразрешённая ссылка — слепота правила, а не отсутствие связи:
        # эта доля ограничивает выводы о К1 сверху.
        "unresolved_share": round(1 - sum(resolved.values()) / total_seen, 3) if total_seen else None,
        "chunks_with_nodes_share": round(with_mentions / len(chunks), 3) if chunks else 0.0,
        "chunks_with_references": len(touched),
        "struct_relations": len(linked),
        "max_passages_per_node": max(degree.values(), default=0),
    }


def _add_notation(
    graph: GraphFile, doc_id: str, chunks: list[dict[str, Any]], *, scope: str = "section"
) -> dict[str, Any]:
    """Обозначения К3: узел «символ := смысл», введён — defines, в области — uses."""
    ordered = sorted(chunks, key=lambda item: item["ordinal"])
    section_of = {chunk["id"]: _section_of(chunk) for chunk in ordered}
    nodes = 0
    users_total = 0
    for position, chunk in enumerate(ordered):
        for note in find_notations(chunk["text"]):
            entity_id = _entity_id(doc_id, "notation", f"{note.symbol} := {note.meaning.lower()}")
            if entity_id not in graph.entities:
                graph.add_entity(
                    entity_id,
                    canonical=f"{note.meaning.lower()} {note.symbol}",
                    name=note.meaning,
                    kind="notation",
                )
                nodes += 1
            graph.add_mention(chunk["id"], entity_id, 1, role="defines")
            area = section_of.get(chunk["id"])
            if scope == "section" and area is None:
                continue
            users = 0
            for later in ordered[position + 1 :]:
                if users >= NOTATION_SCOPE_LIMIT:
                    break
                if scope == "section" and section_of.get(later["id"]) != area:
                    break
                if contains_symbol(later["text"], note.symbol):
                    graph.add_mention(later["id"], entity_id, 1, role="uses")
                    users += 1
            users_total += users
    return {"nodes": nodes, "uses": users_total}


def merge_graphs(*graphs: GraphFile, variant: str = "union") -> GraphFile:
    """Граф C опыта К1: объединение узлов, упоминаний и рёбер.

    Фрагменты обязаны совпадать по идентификаторам: объединяются графы
    одного корпуса. Роль упоминания берётся сильнейшая, число упоминаний —
    наибольшее.
    """
    if not graphs:
        raise ValueError("Нечего объединять")
    merged = GraphFile(variant=variant)
    for graph in graphs:
        for passage_id, row in graph.passages.items():
            merged.passages.setdefault(passage_id, dict(row))
        for entity_id, row in graph.entities.items():
            merged.entities.setdefault(entity_id, dict(row))
        for passage_id, mentions in graph.mentions.items():
            target = merged.mentions.setdefault(passage_id, {})
            for entity_id, (count, role) in mentions.items():
                old_count, old_role = target.get(entity_id, (0, ""))
                best = role if _ROLE_RANK.get(role, 0) > _ROLE_RANK.get(old_role, 0) else old_role
                target[entity_id] = (max(count, old_count), best)
        merged.relations.extend(graph.relations)
    problems = merged.validate()
    if problems:
        raise ValueError(f"Объединение испорчено: {problems[:5]}")
    return merged


def missing_passages(graphs: Sequence[GraphFile]) -> set[str]:
    """Фрагменты, известные не всем графам: признак разных корпусов."""
    sets = [set(graph.passages) for graph in graphs]
    return set.union(*sets) - set.intersection(*sets) if sets else set()
