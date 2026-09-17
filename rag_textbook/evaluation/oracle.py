"""Оракул отбора: сколько даёт идеальный выбор из уже найденного пула.

Условие запуска гипотезы R7 (обучаемый отбор). Если даже идеальный отбор
16 фрагментов из пула почти не поднимает recall, учить отборщика нечему:
узкое место — в самом пуле, и чинить надо поиск.

Считается по слепку, без модели: пул — объединение каналов и кандидатов
реранкера, выдача — итоговые 16. Это верхняя граница по **recall**;
верхнюю границу по формулам в ответе даст только генерация на карте.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any


def pool_of(row: dict[str, Any]) -> set[str]:
    pool = {item["chunk_id"] for items in row.get("channels", {}).values() for item in items}
    pool.update(row.get("rerank_scores", {}))
    pool.update(row.get("final", []))
    return pool


def graph_only_of(row: dict[str, Any]) -> set[str]:
    channels = row.get("channels", {})
    graph = {item["chunk_id"] for item in channels.get("graph", [])}
    others = {
        item["chunk_id"]
        for name, items in channels.items() if name != "graph"
        for item in items
    }
    return graph - others


@dataclass(frozen=True)
class OracleOutcome:
    question_id: str
    question_type: str
    gold: int
    in_pool: int
    in_final: int
    graph_only_gold: int
    graph_only_in_final: int
    pool_size: int

    @property
    def recall(self) -> float:
        return self.in_final / self.gold

    @property
    def oracle_recall(self) -> float:
        return min(self.in_pool, self.pool_capacity) / self.gold

    # Идеальный отбор ограничен размером выдачи; при 1–2 эталонах из 16 мест
    # ограничение не срабатывает, но оставлено явным.
    pool_capacity: int = 16


def evaluate(rows: Iterable[dict[str, Any]], gold: dict[str, set[str]], k: int = 16) -> list[OracleOutcome]:
    outcomes = []
    for row in rows:
        expected = gold.get(row["question_id"])
        if not expected:
            raise ValueError(f"Нет эталона для {row['question_id']}")
        pool = pool_of(row)
        final = set(row.get("final", [])[:k])
        graph_only = graph_only_of(row) & expected
        outcomes.append(OracleOutcome(
            question_id=row["question_id"],
            question_type=row.get("question_type", ""),
            gold=len(expected),
            in_pool=len(expected & pool),
            in_final=len(expected & final),
            graph_only_gold=len(graph_only),
            graph_only_in_final=len(graph_only & final),
            pool_size=len(pool),
            pool_capacity=k,
        ))
    return outcomes


def summarize(outcomes: list[OracleOutcome]) -> dict[str, Any]:
    def block(items: list[OracleOutcome]) -> dict[str, Any]:
        n = len(items)
        return {
            "questions": n,
            "recall": round(sum(o.recall for o in items) / n, 4),
            "oracle_recall": round(sum(o.oracle_recall for o in items) / n, 4),
            "headroom": round(sum(o.oracle_recall - o.recall for o in items) / n, 4),
            "questions_with_headroom": sum(o.in_pool > o.in_final for o in items),
            "graph_only_gold": sum(o.graph_only_gold for o in items),
            "graph_only_in_final": sum(o.graph_only_in_final for o in items),
            "median_pool": sorted(o.pool_size for o in items)[n // 2],
        }

    groups: dict[str, list[OracleOutcome]] = defaultdict(list)
    for outcome in outcomes:
        groups[outcome.question_type].append(outcome)
    return {"all": block(outcomes), "by_type": {k: block(v) for k, v in sorted(groups.items())}}
