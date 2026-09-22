"""Граф знаний как сменный файл и хранилище в памяти поверх него.

Зачем. Гипотезы серии К (docs/HYPOTHESES.md) сравнивают разные способы
строить граф: структурный граф по ссылкам учебника, граф «использование →
определение», обозначения с областью действия. Если каждый вариант
заливать в Neo4j, сравнение стоит пересборки базы на арендованной машине.
Здесь вариант графа — это один файл, а графовый канал читает его через
те же четыре метода, что и у Neo4j:

    find_seed_entities   — стартовые узлы по терминам вопроса;
    entities_of_passages — узлы уже найденных фрагментов;
    expand_entities      — шаг обхода;
    find_passages        — фрагменты по весам узлов.

Каждый метод повторяет запрос Cypher из ``graph_store.py`` вплоть до
формулы балла. Это не украшение, а условие допуска: стенд годится, только
если на выгрузке нынешнего графа канал по файлу совпадает с каналом по
Neo4j хотя бы у 99% вопросов (``scripts/graph_fidelity.py``). Без этого
стенд мерил бы другую систему.

Самое тонкое место — полнотекстовый поиск стартовых узлов. Neo4j ищет их
индексом Lucene, поэтому здесь воспроизведены его разбор на слова
и формула BM25 версии Lucene 9 (без множителя k1+1, убранного в Lucene 8).
Расхождение на этом шаге и есть главный риск проверки совпадения.

Формат файла (JSON, по желанию сжатый gzip)::

    {"format": "rag-graph/1", "variant": "...", "meta": {...},
     "passages":  [{"id", "doc_id", "doc_name", "ordinal", "text", "pages"}],
     "entities":  [{"id", "canonical", "name", "count", "kind"}],
     "mentions":  [[passage_id, entity_id, count, role]],
     "relations": [[source_id, target_id, rel_type, label, weight]]}

``kind`` — тип узла: ``concept`` (понятие из извлечения), ``statement``
(определение, теорема), ``formula`` (формула с номером), ``section``
(раздел), ``notation`` (символ с областью действия). ``role`` — роль
фрагмента по отношению к узлу: ``defines``, ``uses``, ``mentions``,
``refers``, ``in_section`` или пусто. Нынешний граф в Neo4j ролей
не знает, при выгрузке у него все роли пустые.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import re
from collections import defaultdict, deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_textbook.logging_setup import get_logger

logger = get_logger("stores.graph_file")

FORMAT = "rag-graph/1"
NODE_KINDS = ("concept", "statement", "formula", "section", "notation", "figure")
ROLES = ("", "defines", "uses", "mentions", "refers", "in_section")

# Параметры BM25 в Lucene по умолчанию; Neo4j их не меняет.
_BM25_K1 = 1.2
_BM25_B = 0.75
# Поля полнотекстового индекса: ``ON EACH [e.canonical, e.name]``.
_FULLTEXT_FIELDS = ("canonical", "name")

# Приближение StandardTokenizer (UAX#29): буквы и цифры, склеенные
# подчёркиванием, а также точкой или апострофом внутри слова («3.14»).
_TOKEN = re.compile(r"\w+(?:[.'’]\w+)*", re.UNICODE)


def analyze(text: str) -> list[str]:
    """Разбор строки так, как его делает анализатор ``standard-no-stop-words``."""
    return [token.lower() for token in _TOKEN.findall(text or "")]


# --------------------------------------------------------------------- файл


@dataclass
class GraphFile:
    """Вариант графа целиком: узлы, упоминания, связи."""

    variant: str = ""
    meta: dict[str, Any] = field(default_factory=dict)
    passages: dict[str, dict[str, Any]] = field(default_factory=dict)
    entities: dict[str, dict[str, Any]] = field(default_factory=dict)
    # фрагмент → узел → (число упоминаний, роль)
    mentions: dict[str, dict[str, tuple[int, str]]] = field(default_factory=dict)
    # (исток, сток, тип связи Neo4j, метка, вес)
    relations: list[tuple[str, str, str, str, float]] = field(default_factory=list)

    # ------------------------------------------------------------ наполнение

    def add_passage(
        self,
        passage_id: str,
        *,
        doc_id: str = "",
        doc_name: str = "",
        ordinal: int = 0,
        text: str = "",
        pages: Sequence[int] = (),
    ) -> None:
        self.passages[passage_id] = {
            "id": passage_id,
            "doc_id": doc_id,
            "doc_name": doc_name,
            "ordinal": int(ordinal),
            "text": text,
            "pages": [int(page) for page in pages],
        }

    def add_entity(
        self,
        entity_id: str,
        *,
        canonical: str,
        name: str = "",
        count: int = 1,
        kind: str = "concept",
    ) -> None:
        if kind not in NODE_KINDS:
            raise ValueError(f"Неизвестный тип узла: {kind!r}")
        self.entities[entity_id] = {
            "id": entity_id,
            "canonical": canonical,
            "name": name or canonical,
            "count": int(count),
            "kind": kind,
        }

    def add_mention(self, passage_id: str, entity_id: str, count: int = 1, role: str = "") -> None:
        if role not in ROLES:
            raise ValueError(f"Неизвестная роль: {role!r}")
        # Упоминание сливается, как MERGE в Neo4j: одна связь на пару,
        # число упоминаний — последнее записанное. Роль «defines» сильнее
        # прочих: фрагмент, где понятие определено, остаётся местом определения.
        previous = self.mentions.get(passage_id, {}).get(entity_id)
        if previous is not None and previous[1] == "defines":
            role = "defines"
        self.mentions.setdefault(passage_id, {})[entity_id] = (int(count), role)

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        rel_type: str = "RELATES",
        label: str = "",
        weight: float = 1.0,
    ) -> None:
        self.relations.append((source_id, target_id, rel_type, label, float(weight)))

    # ------------------------------------------------------------- проверки

    def validate(self) -> list[str]:
        """Ссылки на несуществующие узлы. Пустой список — файл цел."""
        problems: list[str] = []
        for passage_id, entities in self.mentions.items():
            if passage_id not in self.passages:
                problems.append(f"упоминание из неизвестного фрагмента {passage_id}")
            for entity_id in entities:
                if entity_id not in self.entities:
                    problems.append(f"упоминание неизвестного узла {entity_id}")
        for source, target, *_ in self.relations:
            for node in (source, target):
                if node not in self.entities:
                    problems.append(f"связь с неизвестным узлом {node}")
        return problems[:50]

    def summary(self) -> dict[str, Any]:
        kinds: dict[str, int] = defaultdict(int)
        for entity in self.entities.values():
            kinds[entity.get("kind") or "concept"] += 1
        rel_types: dict[str, int] = defaultdict(int)
        for _, _, rel_type, _, _ in self.relations:
            rel_types[rel_type] += 1
        roles: dict[str, int] = defaultdict(int)
        for entities in self.mentions.values():
            for _, role in entities.values():
                roles[role or "-"] += 1
        return {
            "variant": self.variant,
            "passages": len(self.passages),
            "entities": len(self.entities),
            "mentions": sum(len(item) for item in self.mentions.values()),
            "relations": len(self.relations),
            "entity_kinds": dict(sorted(kinds.items())),
            "relation_types": dict(sorted(rel_types.items())),
            "mention_roles": dict(sorted(roles.items())),
        }

    # ---------------------------------------------------------- сериализация

    def to_json(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "variant": self.variant,
            "meta": self.meta,
            "passages": list(self.passages.values()),
            "entities": list(self.entities.values()),
            "mentions": [
                [passage_id, entity_id, count, role]
                for passage_id, entities in self.mentions.items()
                for entity_id, (count, role) in entities.items()
            ],
            "relations": [list(item) for item in self.relations],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> GraphFile:
        if payload.get("format") != FORMAT:
            raise ValueError(
                f"Файл графа в формате {payload.get('format')!r}, ожидался {FORMAT!r}"
            )
        graph = cls(variant=str(payload.get("variant") or ""), meta=dict(payload.get("meta") or {}))
        for row in payload.get("passages") or []:
            graph.add_passage(
                str(row["id"]),
                doc_id=str(row.get("doc_id") or ""),
                doc_name=str(row.get("doc_name") or ""),
                ordinal=int(row.get("ordinal") or 0),
                text=str(row.get("text") or ""),
                pages=row.get("pages") or [],
            )
        for row in payload.get("entities") or []:
            graph.add_entity(
                str(row["id"]),
                canonical=str(row.get("canonical") or ""),
                name=str(row.get("name") or ""),
                count=int(row.get("count") or 0),
                kind=str(row.get("kind") or "concept"),
            )
        for passage_id, entity_id, count, role in payload.get("mentions") or []:
            graph.add_mention(str(passage_id), str(entity_id), int(count or 0), str(role or ""))
        for source, target, rel_type, label, weight in payload.get("relations") or []:
            graph.add_relation(str(source), str(target), str(rel_type), str(label or ""), float(weight))
        return graph

    def save(self, path: Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(self.to_json(), ensure_ascii=False).encode("utf-8")
        if path.suffix == ".gz":
            # mtime=0: одинаковый граф даёт одинаковый файл и одинаковый хэш.
            data = gzip.compress(data, mtime=0)
        path.write_bytes(data)
        return path

    @classmethod
    def load(cls, path: Path) -> GraphFile:
        raw = Path(path).read_bytes()
        if raw[:2] == b"\x1f\x8b":
            raw = gzip.decompress(raw)
        return cls.from_json(json.loads(raw.decode("utf-8")))


def file_hash(path: Path) -> str:
    """Хэш файла графа: записывается в результаты, чтобы число было воспроизводимо."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------- хранилище


class MemoryGraphStore:
    """Хранилище графа в памяти с поведением ``GraphStore`` на чтение.

    Методы записи не нужны: файл собирается отдельно и не меняется
    во время замера.
    """

    def __init__(self, graph: GraphFile, *, source: str = "") -> None:
        self.graph = graph
        self.source = source
        self._order = {entity_id: index for index, entity_id in enumerate(graph.entities)}

        # Число фрагментов с упоминанием узла и число узлов у фрагмента.
        self._passages_of: dict[str, list[str]] = defaultdict(list)
        self._entity_count: dict[str, int] = {}
        for passage_id, entities in graph.mentions.items():
            self._entity_count[passage_id] = len(entities)
            for entity_id in entities:
                self._passages_of[entity_id].append(passage_id)

        # Смежность по типам связей, без направления, как в шаблоне
        # ``(s)-[:TYPE*1..d]-(n)``. Параллельные рёбра считаются: они дают
        # стартовому узлу путь длины 2 обратно к себе.
        self._adjacency: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(lambda: defaultdict(int))
        )
        for source_id, target_id, rel_type, _, _ in graph.relations:
            self._adjacency[rel_type][source_id][target_id] += 1
            if source_id != target_id:
                self._adjacency[rel_type][target_id][source_id] += 1

        self._build_fulltext()

    @classmethod
    def from_file(cls, path: Path) -> MemoryGraphStore:
        graph = GraphFile.load(path)
        problems = graph.validate()
        if problems:
            raise ValueError(f"Файл графа {path} испорчен: {problems[:5]}")
        logger.info("Граф из файла %s: %s", path, graph.summary())
        return cls(graph, source=str(path))

    # ------------------------------------------------ совместимость с Neo4j

    def verify(self) -> bool:
        return bool(self.graph.passages)

    def close(self) -> None:
        return None

    def stats(self) -> dict[str, int]:
        rel_types: dict[str, int] = defaultdict(int)
        for _, _, rel_type, _, _ in self.graph.relations:
            rel_types[rel_type] += 1
        return {
            "passages": len(self.graph.passages),
            "entities": len(self.graph.entities),
            "relates": rel_types.get("RELATES", 0),
            "cooccurs": rel_types.get("CO_OCCURS", 0),
            "mentions": sum(len(item) for item in self.graph.mentions.values()),
        }

    # ------------------------------------------------- полнотекстовый поиск

    def _build_fulltext(self) -> None:
        # Для каждого поля: позиции слов у каждого узла, частота слов
        # по узлам и средняя длина поля — всё, что нужно формуле BM25.
        self._field_tokens: dict[str, dict[str, list[str]]] = {}
        self._field_df: dict[str, dict[str, int]] = {}
        self._field_avg: dict[str, float] = {}
        self._field_docs: dict[str, int] = {}
        self._postings: dict[str, dict[str, set[str]]] = {}
        for field_name in _FULLTEXT_FIELDS:
            tokens_of: dict[str, list[str]] = {}
            df: dict[str, int] = defaultdict(int)
            postings: dict[str, set[str]] = defaultdict(set)
            for entity_id, entity in self.graph.entities.items():
                tokens = analyze(str(entity.get(field_name) or ""))
                if not tokens:
                    continue
                tokens_of[entity_id] = tokens
                for token in set(tokens):
                    df[token] += 1
                    postings[token].add(entity_id)
            self._field_tokens[field_name] = tokens_of
            self._field_df[field_name] = dict(df)
            self._postings[field_name] = dict(postings)
            self._field_docs[field_name] = len(tokens_of)
            total = sum(len(tokens) for tokens in tokens_of.values())
            self._field_avg[field_name] = total / len(tokens_of) if tokens_of else 1.0

    def _idf(self, field_name: str, token: str) -> float:
        docs = self._field_docs[field_name]
        df = self._field_df[field_name].get(token, 0)
        return math.log(1 + (docs - df + 0.5) / (df + 0.5))

    def _phrase_score(self, field_name: str, entity_id: str, phrase: Sequence[str]) -> float:
        tokens = self._field_tokens[field_name].get(entity_id)
        if not tokens or not phrase:
            return 0.0
        width = len(phrase)
        freq = sum(
            1
            for start in range(len(tokens) - width + 1)
            if tokens[start : start + width] == list(phrase)
        )
        if freq == 0:
            return 0.0
        idf = sum(self._idf(field_name, token) for token in phrase)
        norm = _BM25_K1 * (1 - _BM25_B + _BM25_B * len(tokens) / self._field_avg[field_name])
        return idf * freq / (freq + norm)

    def find_seed_entities(self, terms: Sequence[str], limit: int) -> list[dict[str, Any]]:
        """Аналог ``db.index.fulltext.queryNodes`` с запросом ``"t1" OR "t2"``.

        Каждая фраза ищется в обоих полях, баллы полей и фраз складываются:
        так Lucene считает дизъюнкцию.
        """
        phrases = [analyze(term) for term in terms if term and term.strip()]
        phrases = [phrase for phrase in phrases if phrase]
        if not phrases:
            return []
        scores: dict[str, float] = defaultdict(float)
        for phrase in phrases:
            for field_name in _FULLTEXT_FIELDS:
                postings = self._postings[field_name]
                candidates: set[str] | None = None
                for token in phrase:
                    found = postings.get(token, set())
                    candidates = set(found) if candidates is None else candidates & found
                    if not candidates:
                        break
                for entity_id in candidates or ():
                    score = self._phrase_score(field_name, entity_id, phrase)
                    if score > 0:
                        scores[entity_id] += score
        ranked = sorted(scores.items(), key=lambda item: (-item[1], self._order[item[0]]))
        rows = []
        for entity_id, score in ranked[: int(limit)]:
            entity = self.graph.entities[entity_id]
            rows.append(
                {
                    "id": entity_id,
                    "canonical": entity.get("canonical"),
                    "name": entity.get("name"),
                    "count": entity.get("count") or 1,
                    "score": score,
                }
            )
        return rows

    # ------------------------------------------------------------- обход

    def entities_of_passages(self, chunk_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        corpus = len(self.graph.passages)
        local: dict[str, float] = defaultdict(float)
        # Повтор идентификатора считается дважды, как ``UNWIND`` в Cypher.
        for chunk_id in chunk_ids:
            for entity_id, (count, _) in self.graph.mentions.get(str(chunk_id), {}).items():
                local[entity_id] += math.log(1 + (count or 1))
        rows = []
        for entity_id, value in local.items():
            df = len(self._passages_of.get(entity_id, ()))
            if df == 0:
                continue
            rows.append(
                {
                    "id": entity_id,
                    "canonical": self.graph.entities[entity_id].get("canonical"),
                    "document_frequency": df,
                    "weight": value * math.log(corpus / df),
                }
            )
        rows.sort(key=lambda row: (-row["weight"], self._order[row["id"]]))
        return rows[: int(limit)]

    def expand_entities(
        self,
        seed_ids: Sequence[str],
        hops: int,
        rel_types: Sequence[str],
        limit: int,
        decay: float = 0.5,
    ) -> dict[str, float]:
        if not seed_ids:
            return {}
        allowed = [rel for rel in rel_types if rel.isalpha() or "_" in rel] or ["RELATES"]
        depth = max(1, min(int(hops), 3))
        seeds = [str(item) for item in seed_ids if str(item) in self.graph.entities]

        def neighbours(node: str) -> Iterable[tuple[str, int]]:
            for rel in allowed:
                yield from self._adjacency.get(rel, {}).get(node, {}).items()

        # Кратчайшее расстояние от множества затравок до прочих узлов.
        distance: dict[str, int] = {}
        queue: deque[tuple[str, int]] = deque((seed, 0) for seed in seeds)
        visited = set(seeds)
        while queue:
            node, dist = queue.popleft()
            if dist >= depth:
                continue
            for neighbour, _ in neighbours(node):
                if neighbour in visited:
                    continue
                visited.add(neighbour)
                distance[neighbour] = dist + 1
                queue.append((neighbour, dist + 1))

        # Затравка сама попадает в выдачу Cypher, если до неё есть путь длины
        # не меньше 1: от другой затравки, петлёй или парой параллельных рёбер.
        # В веса она не идёт, но место в LIMIT занимает — это и повторяется.
        seed_set = set(seeds)
        for seed in seeds:
            best: int | None = None
            for neighbour, multiplicity in neighbours(seed):
                if neighbour == seed:
                    best = 1
                    break
                if neighbour in seed_set:
                    best = 1
                elif depth >= 2 and (
                    multiplicity > 1
                    or any(other in seed_set and other != seed for other, _ in neighbours(neighbour))
                ):
                    best = 2 if best is None else min(best, 2)
            if best is not None:
                distance[seed] = best

        ranked = sorted(distance.items(), key=lambda item: (item[1], self._order.get(item[0], 0)))
        weights: dict[str, float] = {entity_id: 1.0 for entity_id in seed_ids}
        step = max(0.0, min(float(decay), 1.0))
        for entity_id, dist in ranked[: int(limit)]:
            if entity_id in weights:
                continue
            weights[entity_id] = step ** max(1, dist)
        return weights

    def find_passages(
        self, entity_weights: dict[str, float], limit: int, use_idf: bool = True
    ) -> list[dict[str, Any]]:
        if not entity_weights:
            return []
        total = len(self.graph.passages)
        raw: dict[str, float] = defaultdict(float)
        contributions: dict[str, list[tuple[float, str]]] = defaultdict(list)
        for entity_id, weight in entity_weights.items():
            if entity_id not in self.graph.entities:
                continue
            passages = self._passages_of.get(entity_id, [])
            df = len(passages)
            idf = math.log(total / max(df, 1)) if use_idf else 1.0
            canonical = str(self.graph.entities[entity_id].get("canonical") or "")
            for passage_id in passages:
                count, _ = self.graph.mentions[passage_id][entity_id]
                value = float(weight) * idf * math.log(1 + (count or 1))
                raw[passage_id] += value
                contributions[passage_id].append((value, canonical))

        rows = []
        for passage_id, value in raw.items():
            passage = self.graph.passages[passage_id]
            entity_count = max(1, self._entity_count.get(passage_id, 0))
            matched: list[str] = []
            for _, canonical in sorted(contributions[passage_id], key=lambda item: -item[0]):
                if canonical not in matched:
                    matched.append(canonical)
            rows.append(
                {
                    "chunk_id": passage_id,
                    "doc_id": passage.get("doc_id"),
                    "doc_name": passage.get("doc_name"),
                    "text": passage.get("text"),
                    "pages": passage.get("pages") or [],
                    "ordinal": passage.get("ordinal"),
                    "matched_entities": matched[:6],
                    "score": value / math.sqrt(entity_count),
                }
            )
        rows.sort(key=lambda row: (-row["score"], row["chunk_id"]))
        return rows[: int(limit)]

    # ------------------------------------------- то, чего у Neo4j пока нет

    def passage_links(
        self,
        chunk_ids: Sequence[str],
        *,
        mode: str = "shared",
        max_entity_passages: int = 64,
    ) -> dict[tuple[str, str], float]:
        """Рёбра между фрагментами внутри заданного множества.

        Нужны отбору (К7, К8, парный отбор К6): он работает с пулом
        кандидатов, а не с понятиями.

        ``shared`` — вес пары равен сумме IDF общих узлов. Узлы, упомянутые
        больше чем в ``max_entity_passages`` фрагментах, пропускаются:
        это хабы, связывающие всё со всем.

        ``dependency`` — направленное ребро от фрагмента, который понятие
        использует, к фрагменту, где оно определено (роли ``uses`` и
        ``defines``, гипотеза К2). У выгрузки нынешнего графа ролей нет,
        и этот режим на ней вернёт пустой словарь — это ожидаемо.
        """
        wanted = [str(item) for item in chunk_ids if str(item) in self.graph.passages]
        wanted_set = set(wanted)
        total = max(1, len(self.graph.passages))
        links: dict[tuple[str, str], float] = defaultdict(float)
        if mode == "shared":
            for passages in self._passages_of.values():
                if len(passages) > max_entity_passages:
                    continue
                inside = sorted(set(passages) & wanted_set)
                if len(inside) < 2:
                    continue
                idf = math.log(total / len(passages))
                for index, left in enumerate(inside):
                    for right in inside[index + 1 :]:
                        links[(left, right)] += idf
                        links[(right, left)] += idf
        elif mode == "dependency":
            for entity_id, passages in self._passages_of.items():
                users = []
                definers = []
                for passage_id in passages:
                    role = self.graph.mentions[passage_id][entity_id][1]
                    if role == "defines":
                        definers.append(passage_id)
                    elif role in ("uses", "mentions", "refers"):
                        users.append(passage_id)
                for user in users:
                    if user not in wanted_set:
                        continue
                    for definer in definers:
                        if definer != user and definer in wanted_set:
                            links[(user, definer)] += 1.0
        else:
            raise ValueError(f"Неизвестный режим связей: {mode!r}")
        return dict(links)

    def definitions_for(self, chunk_ids: Sequence[str], limit_per_entity: int = 2) -> list[str]:
        """Места определений понятий, которые используют заданные фрагменты.

        Для замыкания по зависимостям (К8): определение добирается из графа,
        даже если его нет в пуле кандидатов. Порядок — по порядку изложения.
        """
        found: list[str] = []
        seen = set(str(item) for item in chunk_ids)
        for chunk_id in chunk_ids:
            for entity_id, (_, role) in self.graph.mentions.get(str(chunk_id), {}).items():
                if role not in ("uses", "mentions", "refers"):
                    continue
                definers = [
                    passage_id
                    for passage_id in self._passages_of.get(entity_id, [])
                    if self.graph.mentions[passage_id][entity_id][1] == "defines"
                ]
                definers.sort(
                    key=lambda pid: (
                        self.graph.passages[pid].get("doc_id") or "",
                        self.graph.passages[pid].get("ordinal") or 0,
                    )
                )
                for passage_id in definers[:limit_per_entity]:
                    if passage_id not in seen:
                        seen.add(passage_id)
                        found.append(passage_id)
        return found

    def passage_row(self, chunk_id: str) -> dict[str, Any] | None:
        return self.graph.passages.get(str(chunk_id))

    # -------------------------------------------------------------- PPR (К4)

    def _ppr_matrix(self, rel_types: Sequence[str], use_idf: bool) -> Any:
        """Переходы случайного блуждания по двудольному графу «фрагмент — узел».

        Строится один раз на набор типов связей и кэшируется: граф во время
        замера не меняется.
        """
        import numpy as np

        key = (tuple(sorted(rel_types)), bool(use_idf))
        cache = getattr(self, "_ppr_cache", {})
        if key in cache:
            return cache[key]

        passages = list(self.graph.passages)
        entities = list(self.graph.entities)
        index = {("p", pid): i for i, pid in enumerate(passages)}
        offset = len(passages)
        index.update({("e", eid): offset + i for i, eid in enumerate(entities)})
        total = max(1, len(passages))

        def idf(entity_id: str) -> float:
            if not use_idf:
                return 1.0
            return math.log(total / max(1, len(self._passages_of.get(entity_id, ()))))

        sources: list[int] = []
        targets: list[int] = []
        weights: list[float] = []

        def add(left: int, right: int, weight: float) -> None:
            if weight <= 0:
                return
            sources.extend((left, right))
            targets.extend((right, left))
            weights.extend((weight, weight))

        for passage_id, mentioned in self.graph.mentions.items():
            for entity_id, (count, _) in mentioned.items():
                add(
                    index[("p", passage_id)],
                    index[("e", entity_id)],
                    math.log1p(count or 1) * idf(entity_id),
                )
        allowed = set(rel_types)
        for source_id, target_id, rel_type, _, _ in self.graph.relations:
            if rel_type in allowed and source_id != target_id:
                add(
                    index[("e", source_id)],
                    index[("e", target_id)],
                    math.sqrt(max(idf(source_id), 0.0) * max(idf(target_id), 0.0)) if use_idf else 1.0,
                )

        size = len(index)
        src = np.asarray(sources, dtype=np.int64)
        dst = np.asarray(targets, dtype=np.int64)
        wgt = np.asarray(weights, dtype=np.float64)
        out = np.bincount(src, weights=wgt, minlength=size)
        prob = np.divide(wgt, out[src], out=np.zeros_like(wgt), where=out[src] > 0)
        matrix = (src, dst, prob, out > 0, size, index, len(passages), passages)
        cache[key] = matrix
        self._ppr_cache = cache
        return matrix

    def ppr_passages(
        self,
        entity_weights: dict[str, float],
        limit: int,
        *,
        alpha: float = 0.5,
        rel_types: Sequence[str] = ("RELATES",),
        use_idf: bool = True,
        tolerance: float = 1e-8,
        max_iterations: int = 100,
    ) -> list[dict[str, Any]]:
        """Фрагменты по массе персонализированного PageRank (HippoRAG 2).

        ``alpha`` — вероятность вернуться к затравкам. Масса висячих узлов
        возвращается в затравки, чтобы не утекать. Несходимость не прячется:
        она пишется в журнал с достигнутой невязкой.
        """
        import numpy as np

        src, dst, prob, has_out, size, index, passage_count, passages = self._ppr_matrix(
            rel_types, use_idf
        )
        seed = np.zeros(size)
        for entity_id, weight in entity_weights.items():
            position = index.get(("e", entity_id))
            if position is not None and weight > 0:
                seed[position] += float(weight)
        if seed.sum() <= 0:
            return []
        seed /= seed.sum()

        rank = seed.copy()
        residual = float("inf")
        for _ in range(max_iterations):
            moved = np.bincount(dst, weights=rank[src] * prob, minlength=size)
            dangling = float(rank[~has_out].sum())
            updated = alpha * seed + (1 - alpha) * (moved + dangling * seed)
            residual = float(np.abs(updated - rank).sum())
            rank = updated
            if residual < tolerance:
                break
        else:
            logger.warning("PPR не сошёлся за %s шагов: невязка %.2e", max_iterations, residual)

        scores = rank[:passage_count]
        order = np.argsort(-scores, kind="stable")
        rows = []
        for position in order[: int(limit)]:
            score = float(scores[position])
            if score <= 0:
                break
            passage = self.graph.passages[passages[position]]
            rows.append(
                {
                    "chunk_id": passage["id"],
                    "doc_id": passage.get("doc_id"),
                    "doc_name": passage.get("doc_name"),
                    "text": passage.get("text"),
                    "pages": passage.get("pages") or [],
                    "ordinal": passage.get("ordinal"),
                    "matched_entities": [],
                    "score": score,
                }
            )
        return rows
