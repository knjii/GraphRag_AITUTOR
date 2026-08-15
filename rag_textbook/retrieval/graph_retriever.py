"""Графовый канал поиска.

Устройство канала: термины вопроса → канонические формы → full-text поиск
стартовых сущностей → расширение на один хоп по типизированным ``RELATES`` →
взвешенное ранжирование пассажей.

Отличия от прежней версии, каждое соответствует конкретной поломке:

* термины запроса лемматизируются, поэтому «сингулярных разложений» находит узел
  «сингулярное разложение» — раньше точное совпадение почти всегда промахивалось,
  а fallback через ``CONTAINS`` тащил случайные подстроки;
* поиск идёт по full-text индексу, а не полным сканом по метке ``Entity``;
* расширение не ходит по ``CO_OCCURS``: на плотном графе это давало почти весь граф;
* вклад сущности взвешен близостью к запросу, а ранжирование нормировано на
  насыщенность пассажа терминами, вместо ``count(*)``.
"""

from __future__ import annotations

from rag_textbook.config import GraphSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.stores.graph_store import GraphStore
from rag_textbook.utils.text import canonicalize_entity, content_terms

logger = get_logger("retrieval.graph")


class GraphRetriever:
    def __init__(self, settings: GraphSettings, store: GraphStore) -> None:
        self.settings = settings
        self.store = store

    def _query_terms(self, question: str) -> list[str]:
        """Кандидаты для поиска сущностей: леммы и биграммы лемм."""
        unigrams = content_terms(
            question,
            min_length=self.settings.min_entity_length,
            lemmatize=self.settings.lemmatize_entities,
            limit=16,
        )
        bigrams: list[str] = []
        for left, right in zip(unigrams, unigrams[1:], strict=False):
            phrase = canonicalize_entity(f"{left} {right}", lemmatize=False)
            if phrase and phrase not in bigrams:
                bigrams.append(phrase)
        # Биграммы идут первыми: составной термин точнее одиночного слова.
        return (bigrams + unigrams)[:24]

    def retrieve(self, question: str, limit: int | None = None) -> list[ScoredChunk]:
        if not self.settings.retrieval_enabled:
            return []

        terms = self._query_terms(question)
        if not terms:
            return []

        try:
            seeds = self.store.find_seed_entities(terms, self.settings.seed_entity_limit)
        except Exception as exc:  # noqa: BLE001
            # Графовый канал — дополнение. Его недоступность не должна ронять запрос.
            logger.warning("Поиск стартовых сущностей не удался: %s", exc)
            return []

        if not seeds:
            logger.debug("Графовый канал: стартовые сущности не найдены для «%s»", question[:60])
            return []

        seed_ids = [str(row["id"]) for row in seeds if row.get("id")]
        try:
            weights = self.store.expand_entities(
                seed_ids,
                hops=self.settings.expansion_hops,
                rel_types=list(self.settings.expansion_rel_types),
                limit=self.settings.seed_entity_limit * 4,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Расширение графа не удалось: %s", exc)
            weights = {entity_id: 1.0 for entity_id in seed_ids}

        # Стартовые сущности взвешиваем релевантностью полнотекстового поиска.
        max_score = max((float(row.get("score") or 0.0) for row in seeds), default=1.0) or 1.0
        for row in seeds:
            entity_id = str(row.get("id") or "")
            if entity_id:
                weights[entity_id] = float(row.get("score") or 0.0) / max_score

        try:
            rows = self.store.find_passages(weights, limit or self.settings.passage_limit)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Поиск пассажей в графе не удался: %s", exc)
            return []

        results: list[ScoredChunk] = []
        for row in rows:
            text = str(row.get("text") or "").strip()
            if not text:
                continue
            chunk = Chunk(
                id=str(row.get("chunk_id") or ""),
                doc_id=str(row.get("doc_id") or ""),
                doc_name=str(row.get("doc_name") or ""),
                source_path="",
                ordinal=int(row.get("ordinal") or 0),
                text=text,
                pages=[int(page) for page in (row.get("pages") or [])],
            )
            results.append(
                ScoredChunk(
                    chunk=chunk,
                    score=float(row.get("score") or 0.0),
                    channels=["graph_entity"],
                    channel_scores={"graph_entity": float(row.get("score") or 0.0)},
                    matched_entities=[str(name) for name in (row.get("matched_entities") or [])],
                )
            )

        logger.debug(
            "Графовый канал: seed=%s, расширено до %s сущностей, пассажей=%s",
            len(seed_ids),
            len(weights),
            len(results),
        )
        return results
