"""Граф знаний в Neo4j.

Прежняя реализация давала граф, в котором 280 526 из 282 143 рёбер были
``CO_OCCURS``, построенные как «все пары топ-20 слов чанка». Обход
``RELATES|CO_OCCURS*1..2`` по такому графу доставал почти весь граф, а пассажи
ранжировались по ``count(*)``, то есть по терминологической плотности.
Результат — графовый канал работал как шумный BM25 и прироста не давал.

Что изменено:

* ``CO_OCCURS`` не участвует в многохоповом расширении и хранится только как
  статистика с посчитанным PMI; расширение идёт по типизированным ``RELATES``;
* сущности канонизированы (лемматизация), поэтому запрос в косвенном падеже
  находит узел, а не проваливается в ``CONTAINS`` с полным сканом;
* seed-поиск использует full-text индекс Neo4j;
* пассажи ранжируются взвешенно, с нормировкой на насыщенность чанка терминами;
* запись идёт батчами, а не одной транзакцией на 280 тысяч строк;
* драйвер создаётся один раз на приложение, а не на каждый запрос.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from typing import Any

from rag_textbook.config import GraphSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, Entity, Relation

logger = get_logger("stores.graph")

FULLTEXT_INDEX = "entity_fulltext"

SCHEMA_STATEMENTS: tuple[str, ...] = (
    "CREATE CONSTRAINT document_id IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE CONSTRAINT passage_id IF NOT EXISTS FOR (p:Passage) REQUIRE p.id IS UNIQUE",
    "CREATE CONSTRAINT entity_id IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE INDEX entity_canonical IF NOT EXISTS FOR (e:Entity) ON (e.canonical)",
    "CREATE INDEX passage_doc IF NOT EXISTS FOR (p:Passage) ON (p.doc_id)",
    # Полнотекстовый индекс — замена `CONTAINS`, который сканировал все узлы Entity.
    f"CREATE FULLTEXT INDEX {FULLTEXT_INDEX} IF NOT EXISTS "
    "FOR (e:Entity) ON EACH [e.canonical, e.name]",
)


class GraphStore:
    def __init__(self, settings: GraphSettings) -> None:
        self.settings = settings
        self._driver: Any = None

    # -------------------------------------------------------------- соединение

    @property
    def driver(self) -> Any:
        if self._driver is None:
            from neo4j import GraphDatabase

            password = self.settings.password.get_secret_value()
            if not password:
                raise RuntimeError("NEO4J_PASSWORD пуст — графовый слой недоступен")
            self._driver = GraphDatabase.driver(
                self.settings.uri,
                auth=(self.settings.user, password),
                # Пул переиспользуется: прежде драйвер создавался на каждый запрос.
                max_connection_pool_size=16,
            )
        return self._driver

    def verify(self) -> bool:
        try:
            self.driver.verify_connectivity()
            return True
        except Exception as exc:  # noqa: BLE001
            logger.warning("Neo4j недоступен: %s", exc)
            return False

    def close(self) -> None:
        if self._driver is not None:
            self._driver.close()
            self._driver = None

    def _session(self) -> Any:
        return self.driver.session(database=self.settings.database)

    @staticmethod
    def _run(session: Any, cypher: str, **params: Any) -> Any:
        """Выполняет запрос, передавая параметры словарём.

        Именованными аргументами их передавать нельзя: сигнатура драйвера —
        ``Session.run(query, parameters=None, **kwparameters)``, поэтому
        параметр Cypher с именем ``query`` или ``parameters`` перекрывает
        собственный аргумент метода. Ошибка при этом не синтаксическая, а
        времени выполнения, и всплывает только на том единственном запросе,
        где имена совпали: поиск стартовых сущностей падал на каждом вопросе,
        графовый канал молча отдавал пустоту, а замер показывал «граф не даёт
        прироста» — при том что граф просто не участвовал в поиске.
        """
        return session.run(cypher, params)

    # ------------------------------------------------------------------- схема

    def ensure_schema(self) -> None:
        with self._session() as session:
            for statement in SCHEMA_STATEMENTS:
                try:
                    self._run(session, statement).consume()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Не удалось применить схему (%s): %s", statement[:60], exc)
        logger.info("Схема графа применена")

    # ------------------------------------------------------------------ запись

    def _run_batched(self, query: str, rows: Sequence[dict[str, Any]], label: str) -> int:
        """Пишет батчами.

        Одна транзакция на сотни тысяч строк давала пик памяти и «всё или ничего».
        Батч фиксируется отдельно, поэтому сбой на середине не отменяет уже записанное.
        """
        if not rows:
            return 0
        step = self.settings.write_batch_size
        written = 0
        with self._session() as session:
            for start in range(0, len(rows), step):
                batch = rows[start : start + step]
                self._run(session, query, rows=batch).consume()
                written += len(batch)
                if start and start % (step * 10) == 0:
                    logger.debug("%s: записано %s/%s", label, written, len(rows))
        logger.info("%s: записано %s строк", label, written)
        return written

    def upsert_document(self, doc_id: str, doc_name: str, source_path: str) -> None:
        with self._session() as session:
            self._run(
                session,
                """
                MERGE (d:Document {id: $doc_id})
                SET d.name = $doc_name, d.path = $source_path, d.updated_at = timestamp()
                """,
                doc_id=doc_id,
                doc_name=doc_name,
                source_path=source_path,
            ).consume()

    def upsert_passages(self, chunks: Sequence[Chunk]) -> int:
        rows = [
            {
                "id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "text": chunk.text,
                "pages": chunk.pages,
                "ordinal": chunk.ordinal,
                "has_formula": chunk.has_formula,
                "has_table": chunk.has_table,
            }
            for chunk in chunks
        ]
        written = self._run_batched(
            """
            UNWIND $rows AS row
            MERGE (p:Passage {id: row.id})
            SET p.doc_id = row.doc_id,
                p.doc_name = row.doc_name,
                p.text = row.text,
                p.pages = row.pages,
                p.ordinal = row.ordinal,
                p.has_formula = row.has_formula,
                p.has_table = row.has_table,
                p.updated_at = timestamp()
            WITH p, row
            MATCH (d:Document {id: row.doc_id})
            MERGE (d)-[:HAS_PASSAGE]->(p)
            """,
            rows,
            "Passage",
        )
        # Последовательные связи внутри документа: нужны, чтобы отдавать соседний
        # фрагмент, когда ответ разрезан границей чанка.
        self._run_batched(
            """
            UNWIND $rows AS row
            MATCH (a:Passage {id: row.id})
            MATCH (b:Passage {doc_id: row.doc_id, ordinal: row.next_ordinal})
            MERGE (a)-[:NEXT]->(b)
            """,
            [
                {"id": chunk.id, "doc_id": chunk.doc_id, "next_ordinal": chunk.ordinal + 1}
                for chunk in chunks
            ],
            "NEXT",
        )
        return written

    def upsert_entities(self, entities: Iterable[Entity]) -> int:
        rows = [
            {
                "id": entity.id,
                "name": entity.name,
                "canonical": entity.canonical,
                "aliases": entity.aliases,
                "count": entity.count,
            }
            for entity in entities
        ]
        return self._run_batched(
            """
            UNWIND $rows AS row
            MERGE (e:Entity {id: row.id})
            ON CREATE SET e.created_at = timestamp(), e.count = 0
            SET e.name = row.name,
                e.canonical = row.canonical,
                e.count = coalesce(e.count, 0) + row.count,
                e.aliases = CASE
                    WHEN e.aliases IS NULL THEN row.aliases
                    ELSE apoc.coll.toSet(e.aliases + row.aliases)
                END,
                e.updated_at = timestamp()
            """.replace(
                # APOC может отсутствовать: тогда просто перезаписываем список.
                "apoc.coll.toSet(e.aliases + row.aliases)",
                "row.aliases",
            ),
            rows,
            "Entity",
        )

    def upsert_mentions(self, mentions: Sequence[dict[str, Any]]) -> int:
        return self._run_batched(
            """
            UNWIND $rows AS row
            MATCH (p:Passage {id: row.chunk_id})
            MATCH (e:Entity {id: row.entity_id})
            MERGE (p)-[m:MENTIONS]->(e)
            SET m.count = row.count,
                m.doc_id = row.doc_id,
                m.updated_at = timestamp()
            """,
            list(mentions),
            "MENTIONS",
        )

    def upsert_relations(self, relations: Sequence[Relation]) -> int:
        rows = [
            {
                "source_id": relation.source_id,
                "target_id": relation.target_id,
                "label": relation.label,
                "chunk_id": relation.chunk_id,
                "doc_id": relation.doc_id,
                "weight": relation.weight,
            }
            for relation in relations
        ]
        return self._run_batched(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {id: row.source_id})
            MATCH (b:Entity {id: row.target_id})
            MERGE (a)-[r:RELATES {label: row.label, doc_id: row.doc_id}]->(b)
            ON CREATE SET r.created_at = timestamp(), r.weight = 0.0, r.chunk_ids = []
            SET r.weight = coalesce(r.weight, 0.0) + row.weight,
                r.chunk_ids = CASE
                    WHEN size(coalesce(r.chunk_ids, [])) < 20
                    THEN coalesce(r.chunk_ids, []) + row.chunk_id
                    ELSE r.chunk_ids
                END,
                r.updated_at = timestamp()
            """,
            rows,
            "RELATES",
        )

    def upsert_cooccurrences(self, edges: Sequence[dict[str, Any]]) -> int:
        """Записывает co-occurrence как статистику.

        Эти рёбра сознательно не используются для расширения: они нужны как
        материал для анализа графа и как запасной сигнал в пределах одного хопа.
        """
        return self._run_batched(
            """
            UNWIND $rows AS row
            MATCH (a:Entity {id: row.source_id})
            MATCH (b:Entity {id: row.target_id})
            MERGE (a)-[c:CO_OCCURS {doc_id: row.doc_id}]->(b)
            SET c.count = row.count,
                c.pmi = row.pmi,
                c.updated_at = timestamp()
            """,
            list(edges),
            "CO_OCCURS",
        )

    def delete_document(self, doc_id: str) -> dict[str, int]:
        with self._session() as session:
            record = self._run(
                session,
                """
                MATCH (p:Passage {doc_id: $doc_id})
                DETACH DELETE p
                RETURN count(*) AS passages
                """,
                doc_id=doc_id,
            ).single()
            passages = int((record or {}).get("passages") or 0)
            # Сущности без единого упоминания больше не нужны.
            record = self._run(
                session,
                """
                MATCH (e:Entity)
                WHERE NOT (e)<-[:MENTIONS]-()
                DETACH DELETE e
                RETURN count(*) AS entities
                """
            ).single()
            entities = int((record or {}).get("entities") or 0)
        return {"passages": passages, "orphan_entities": entities}

    # ------------------------------------------------------------------ чтение

    def find_seed_entities(self, terms: Sequence[str], limit: int) -> list[dict[str, Any]]:
        """Ищет стартовые сущности по каноническим формам через full-text индекс."""
        if not terms:
            return []
        # Экранируем спецсимволы Lucene, иначе запрос вида «C++» уронит поиск.
        escaped = [
            "".join("\\" + ch if ch in '+-&|!(){}[]^"~*?:\\/' else ch for ch in term)
            for term in terms
            if term
        ]
        query_string = " OR ".join(f'"{term}"' for term in escaped if term.strip())
        if not query_string:
            return []
        with self._session() as session:
            rows = self._run(
                session,
                f"""
                CALL db.index.fulltext.queryNodes('{FULLTEXT_INDEX}', $search)
                YIELD node, score
                RETURN node.id AS id, node.canonical AS canonical, node.name AS name,
                       coalesce(node.count, 1) AS count, score
                ORDER BY score DESC
                LIMIT $limit
                """,
                search=query_string,
                limit=int(limit),
            ).data()
        return rows

    def linked_passage_pairs(
        self, limit: int, min_distance: int = 10
    ) -> list[dict[str, Any]]:
        """Пары фрагментов, соединённые типизированной связью через сущности.

        Нужна для сборки эталонного набора: пара, где связь существует
        в графе, но фрагменты стоят далеко друг от друга, — это ровно тот
        случай, ради которого граф и строится. Соседние фрагменты исключены:
        они перекрываются, и связь между ними ничего не доказывает.

        Возвращает только пары, связанные ``RELATES``: совместная встречаемость
        для этой цели не годится, она и есть переодетый лексический сигнал.
        """
        with self._session() as session:
            rows = self._run(
                session,
                """
                MATCH (left:Passage)-[:MENTIONS]->(a:Entity)
                      -[:RELATES]-(b:Entity)<-[:MENTIONS]-(right:Passage)
                WHERE left.doc_id = right.doc_id
                  AND left.ordinal + $min_distance < right.ordinal
                WITH left, right, count(DISTINCT [a.id, b.id]) AS links
                RETURN left.id AS left, right.id AS right, links
                ORDER BY links DESC
                LIMIT $limit
                """,
                min_distance=int(min_distance),
                limit=int(limit),
            ).data()
        return rows

    def entities_of_passages(self, chunk_ids: Sequence[str], limit: int) -> list[dict[str, Any]]:
        """Сущности, упомянутые в заданных фрагментах.

        Опора обхода не на формулировку вопроса, а на уже найденный текст.
        Вклад сущности взвешен числом упоминаний и **обратной** частотой по
        корпусу: термин, встречающийся в каждом втором фрагменте, ничего
        не сообщает о связях именно этого фрагмента, а вот редкий — сообщает.
        """
        if not chunk_ids:
            return []
        with self._session() as session:
            rows = self._run(
                session,
                """
                MATCH (total:Passage)
                WITH count(total) AS corpus
                UNWIND $chunk_ids AS cid
                MATCH (p:Passage {id: cid})-[m:MENTIONS]->(e:Entity)
                WITH corpus, e, sum(log(1 + coalesce(m.count, 1))) AS local
                MATCH (e)<-[:MENTIONS]-(other:Passage)
                WITH corpus, e, local, count(DISTINCT other) AS document_frequency
                RETURN e.id AS id,
                       e.canonical AS canonical,
                       document_frequency,
                       local * log(toFloat(corpus) / document_frequency) AS weight
                ORDER BY weight DESC
                LIMIT $limit
                """,
                chunk_ids=[str(item) for item in chunk_ids],
                limit=int(limit),
            ).data()
        return rows

    def expand_entities(
        self, seed_ids: Sequence[str], hops: int, rel_types: Sequence[str], limit: int
    ) -> dict[str, float]:
        """Расширяет множество сущностей по типизированным связям.

        Вес соседа затухает с расстоянием, поэтому дальние узлы не забивают
        стартовые — прежний обход всех соседей считал их равнозначными.
        """
        if not seed_ids:
            return {}
        allowed = [rel for rel in rel_types if rel.isalpha() or "_" in rel]
        if not allowed:
            allowed = ["RELATES"]
        rel_pattern = "|".join(allowed)
        depth = max(1, min(int(hops), 3))

        with self._session() as session:
            rows = self._run(
                session,
                f"""
                UNWIND $seed_ids AS seed_id
                MATCH (s:Entity {{id: seed_id}})
                OPTIONAL MATCH path = (s)-[:{rel_pattern}*1..{depth}]-(n:Entity)
                WITH n, min(length(path)) AS distance
                WHERE n IS NOT NULL
                RETURN n.id AS id, distance
                ORDER BY distance ASC
                LIMIT $limit
                """,
                seed_ids=list(seed_ids),
                limit=int(limit),
            ).data()

        weights: dict[str, float] = {entity_id: 1.0 for entity_id in seed_ids}
        for row in rows:
            entity_id = str(row.get("id") or "")
            distance = int(row.get("distance") or 1)
            if not entity_id or entity_id in weights:
                continue
            weights[entity_id] = 1.0 / (1.0 + distance)
        return weights

    def find_passages(self, entity_weights: dict[str, float], limit: int) -> list[dict[str, Any]]:
        """Взвешенное ранжирование пассажей.

        Прежняя версия считала ``count(*)`` — число совпавших сущностей, из-за чего
        наверх всплывали просто самые терминологически плотные чанки.
        Здесь вклад сущности взвешен её близостью к запросу и логарифмом числа
        упоминаний, а сумма нормируется на насыщенность пассажа терминами.
        """
        if not entity_weights:
            return []
        rows_input = [
            {"entity_id": entity_id, "weight": float(weight)}
            for entity_id, weight in entity_weights.items()
        ]
        with self._session() as session:
            rows = self._run(
                session,
                """
                UNWIND $entities AS item
                MATCH (p:Passage)-[m:MENTIONS]->(e:Entity {id: item.entity_id})
                WITH p,
                     sum(item.weight * log(1 + coalesce(m.count, 1))) AS raw_score,
                     collect(DISTINCT e.canonical)[0..6] AS matched
                MATCH (p)-[:MENTIONS]->(all_e:Entity)
                WITH p, raw_score, matched, count(all_e) AS entity_count
                RETURN p.id AS chunk_id,
                       p.doc_id AS doc_id,
                       p.doc_name AS doc_name,
                       p.text AS text,
                       p.pages AS pages,
                       p.ordinal AS ordinal,
                       matched AS matched_entities,
                       raw_score / sqrt(toFloat(CASE WHEN entity_count < 1 THEN 1 ELSE entity_count END)) AS score
                ORDER BY score DESC
                LIMIT $limit
                """,
                entities=rows_input,
                limit=int(limit),
            ).data()
        return rows

    def stats(self) -> dict[str, int]:
        with self._session() as session:
            record = self._run(
                session,
                """
                CALL () {MATCH (p:Passage) RETURN count(p) AS passages}
                CALL () {MATCH (e:Entity) RETURN count(e) AS entities}
                CALL () {MATCH ()-[r:RELATES]->() RETURN count(r) AS relates}
                CALL () {MATCH ()-[c:CO_OCCURS]->() RETURN count(c) AS cooccurs}
                CALL () {MATCH ()-[m:MENTIONS]->() RETURN count(m) AS mentions}
                RETURN passages, entities, relates, cooccurs, mentions
                """
            ).single()
        return {key: int(value) for key, value in dict(record or {}).items()}

    def prune_high_degree_entities(self, max_degree: int) -> int:
        """Убирает узлы-хабы.

        Термин, встречающийся почти в каждом чанке, связывает всё со всем и
        превращает обход в перебор всего графа. Такие узлы полезнее удалить.
        """
        with self._session() as session:
            record = self._run(
                session,
                """
                MATCH (e:Entity)
                WITH e, COUNT { (e)-[:RELATES]-() } + COUNT { (e)-[:CO_OCCURS]-() } AS degree
                WHERE degree > $max_degree
                DETACH DELETE e
                RETURN count(*) AS removed
                """,
                max_degree=int(max_degree),
            ).single()
        removed = int((record or {}).get("removed") or 0)
        if removed:
            logger.info("Удалено сущностей-хабов со степенью выше %s: %s", max_degree, removed)
        return removed


def pointwise_mutual_information(
    pair_count: int, left_count: int, right_count: int, total: int
) -> float:
    """PMI для пары терминов.

    Используется, чтобы отсеять co-occurrence, объясняемые просто частотностью:
    именно они раздували граф до 280 тысяч рёбер.
    """
    if pair_count <= 0 or left_count <= 0 or right_count <= 0 or total <= 0:
        return 0.0
    p_pair = pair_count / total
    p_left = left_count / total
    p_right = right_count / total
    denominator = p_left * p_right
    if denominator <= 0:
        return 0.0
    return math.log2(p_pair / denominator)
