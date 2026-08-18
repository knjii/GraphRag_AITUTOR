"""Слепок поиска: что видел конвейер на каждом вопросе.

Зачем. Прошлый офлайн-харнесс восстанавливал граф из кэша извлечения и указал
**в другую сторону**: порог хабов 40 подтверждался офлайн на отложенной
половине пар, а на сервере оказался значимо вредным. Восстановление
моделировало обход, но не моделировало полнотекстовый поиск затравок, на
который обрезка хабов влияет тоже. Отсюда правило: офлайн годится, чтобы
отбрасывать, но не подтверждать.

Слепок это правило не отменяет, а сужает область, где оно нужно. Мы ничего
не воспроизводим заново: за один серверный прогон сохраняются кандидаты всех
каналов с рангами и баллами, баллы реранкера по широкому окну и итоговый
отбор. После этого всё, что меняет **порядок и отбор** из этих кандидатов,
считается офлайн на тех же числах — это не приближение, а тот же расчёт.

Граница проходит по составу кандидатов. Всё, что его меняет — параметры
обхода графа, режим затравок, разложение вопроса, другая модель, — слепком
не проверяется. Граница закреплена проверкой ``assert_replayable``, а не
комментарием: комментарий про ту же ловушку уже один раз не сработал.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rag_textbook.config import Settings
from rag_textbook.logging_setup import get_logger

logger = get_logger("evaluation.trace")

# Настройки, меняющие СОСТАВ кандидатов. Их изменение делает слепок
# недействительным: сохранённые кандидаты получены при других значениях,
# и пересчёт по ним дал бы уверенный, но неверный ответ.
COMPOSITION_FIELDS: tuple[str, ...] = (
    "graph.expansion_hops",
    "graph.seed_entity_limit",
    "graph.passage_limit",
    "graph.max_entity_degree",
    "graph.hop_decay",
    "graph.passage_idf_enabled",
    "graph.seed_mode",
    "graph.seed_passages",
    "graph.expansion_rel_types",
    "graph.retrieval_enabled",
    "retrieval.dense_candidates",
    "retrieval.sparse_candidates",
    "retrieval.decompose_enabled",
    "retrieval.decompose_max_parts",
    "retrieval.query_rewrite_enabled",
    "retrieval.router_enabled",
    "retrieval.router_mode",
    "llm.model",
    "embedding.model",
    "reranker.model",
)


# Настройки, меняющие ПОРЯДОК и ОТБОР. Они и есть предмет офлайн-перебора,
# но записывать их в слепок всё равно надо: на сервере значения отличаются
# от значений по умолчанию в репозитории, и без них точкой отсчёта окажется
# не та конфигурация, что дала измеренные 0.754.
ORDERING_FIELDS: tuple[str, ...] = (
    "retrieval.top_k",
    "retrieval.top_k_linking",
    "retrieval.rrf_k",
    "retrieval.dedup_enabled",
    "retrieval.dedup_similarity",
    "retrieval.min_graph_docs",
    "retrieval.graph_candidate_quota",
    "retrieval.diversity_mode",
    "retrieval.diversity_lambda",
    "retrieval.diversity_reserve_slots",
    "reranker.enabled",
    "reranker.mode",
    "reranker.blend_alpha",
    "reranker.top_n",
    "reranker.candidates",
    "graph.weight",
)


class NotReplayable(RuntimeError):
    """Настройку нельзя проверить по слепку — только прогоном на сервере."""


@dataclass
class TracedCandidate:
    """Кандидат, каким его вернул канал."""

    chunk_id: str
    rank: int
    score: float

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TracedCandidate:
        return cls(
            chunk_id=str(raw["chunk_id"]),
            rank=int(raw["rank"]),
            score=float(raw.get("score", 0.0)),
        )


@dataclass
class QueryTrace:
    """Всё, что конвейер видел и решил по одному вопросу."""

    question_id: str
    question: str
    rewritten_question: str = ""
    question_type: str = ""
    used_graph: bool = False
    route_reason: str = ""
    sub_questions: list[str] = field(default_factory=list)
    # Каналы хранятся раздельно: слияние — как раз то, что мы будем менять.
    channels: dict[str, list[TracedCandidate]] = field(default_factory=dict)
    # Баллы реранкера по ШИРОКОМУ окну: рабочее окно 30, но без запаса нельзя
    # проверить гипотезу о том, что окно обрезает нужное до ранжирования.
    rerank_scores: dict[str, float] = field(default_factory=dict)
    # Итоговый отбор рабочей конфигурации: по нему сверяется честность слепка.
    final: list[str] = field(default_factory=list)

    def candidate_ids(self) -> list[str]:
        """Все кандидаты без повторов, в порядке первого появления."""
        seen: dict[str, None] = {}
        for items in self.channels.values():
            for item in items:
                seen.setdefault(item.chunk_id, None)
        return list(seen)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["channels"] = {
            name: [asdict(item) for item in items] for name, items in self.channels.items()
        }
        return payload

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> QueryTrace:
        return cls(
            question_id=str(raw["question_id"]),
            question=str(raw.get("question", "")),
            rewritten_question=str(raw.get("rewritten_question", "")),
            question_type=str(raw.get("question_type", "")),
            used_graph=bool(raw.get("used_graph", False)),
            route_reason=str(raw.get("route_reason", "")),
            sub_questions=list(raw.get("sub_questions", [])),
            channels={
                name: [TracedCandidate.from_dict(item) for item in items]
                for name, items in (raw.get("channels") or {}).items()
            },
            rerank_scores={
                str(key): float(value) for key, value in (raw.get("rerank_scores") or {}).items()
            },
            final=list(raw.get("final", [])),
        )


@dataclass
class TraceSet:
    """Слепок целиком плюс настройки, при которых он снят."""

    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    # Настройки порядка на момент снятия — это точка отсчёта для перебора.
    ordering_snapshot: dict[str, Any] = field(default_factory=dict)
    rerank_window: int = 0
    traces: list[QueryTrace] = field(default_factory=list)

    def save(self, path: Path) -> None:
        """Пишет построчно: прогон может прерваться, и хвост не должен
        обесценивать начало."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            header = {
                "kind": "trace-header",
                "settings_snapshot": self.settings_snapshot,
                "ordering_snapshot": self.ordering_snapshot,
                "rerank_window": self.rerank_window,
            }
            handle.write(json.dumps(header, ensure_ascii=False) + "\n")
            for trace in self.traces:
                handle.write(json.dumps(trace.to_dict(), ensure_ascii=False) + "\n")

    @classmethod
    def load(cls, path: Path) -> TraceSet:
        result = cls()
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                if raw.get("kind") == "trace-header":
                    result.settings_snapshot = raw.get("settings_snapshot", {})
                    result.ordering_snapshot = raw.get("ordering_snapshot", {})
                    result.rerank_window = int(raw.get("rerank_window", 0))
                    continue
                result.traces.append(QueryTrace.from_dict(raw))
        logger.info("Загружен слепок: %s вопросов", len(result.traces))
        return result


def read_setting(settings: Settings, path: str) -> Any:
    """Достаёт значение настройки по пути вида ``graph.hop_decay``."""
    current: Any = settings
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def snapshot_ordering(settings: Settings) -> dict[str, Any]:
    """Запоминает значения, определяющие порядок и отбор."""
    snapshot: dict[str, Any] = {}
    for path in ORDERING_FIELDS:
        value = read_setting(settings, path)
        snapshot[path] = list(value) if isinstance(value, tuple) else value
    return snapshot


def snapshot_settings(settings: Settings) -> dict[str, Any]:
    """Запоминает значения, определяющие состав кандидатов."""
    snapshot: dict[str, Any] = {}
    for path in COMPOSITION_FIELDS:
        value = read_setting(settings, path)
        snapshot[path] = list(value) if isinstance(value, tuple) else value
    return snapshot


def assert_replayable(snapshot: dict[str, Any], settings: Settings) -> None:
    """Отказывается считать, если изменена настройка из-за границы.

    Это главная защита харнесса. Прошлый раз офлайн-замер уверенно указал
    в другую сторону именно потому, что менял то, чего его модель не описывала.
    Здесь такая попытка заканчивается ошибкой, а не правдоподобным числом.
    """
    changed: list[str] = []
    for path, recorded in snapshot.items():
        current = read_setting(settings, path)
        if isinstance(current, tuple):
            current = list(current)
        if current != recorded:
            changed.append(f"{path}: слепок {recorded!r}, запрошено {current!r}")
    if changed:
        raise NotReplayable(
            "Эти настройки меняют состав кандидатов, а не их порядок, поэтому "
            "по слепку не проверяются — нужен прогон на сервере:\n  "
            + "\n  ".join(changed)
        )


def missing_chunks(traces: Iterable[QueryTrace], known: Sequence[str]) -> set[str]:
    """Кандидаты, для которых нет текста или вектора.

    Разнообразие выдачи считается по векторам фрагментов; если вектора нет,
    расчёт молча выродится в обычный порядок.
    """
    available = set(known)
    unknown: set[str] = set()
    for trace in traces:
        unknown.update(item for item in trace.candidate_ids() if item not in available)
    return unknown


def align_to_snapshot(settings: Settings, snapshot: dict[str, Any]) -> tuple[Settings, list[str]]:
    """Приводит настройки состава к тем, при которых снят слепок.

    Слепок — источник истины о том, КАК получены кандидаты. Локальный `.env`
    об этом ничего не знает и знать не может: он живёт на другой машине
    и вообще на другой стадии работы. Поэтому поля состава берутся из слепка,
    а не сверяются с локальными.

    Это не ослабление защиты, а её правильная точка приложения. Проверка
    ``assert_replayable`` остаётся и срабатывает там, где ей место: когда
    перебор пытается **изменить** поле состава относительно слепка.

    Расхождение здесь обычно и не ошибка вовсе: на сервере в `.env` могут
    стоять значения, отличные от значений по умолчанию в репозитории, — и
    именно они породили точку отсчёта.
    """
    aligned = settings.model_copy(deep=True)
    changes: list[str] = []
    updates: dict[str, dict[str, Any]] = {}
    for path, recorded in snapshot.items():
        section, _, field_name = path.partition(".")
        if not field_name or not hasattr(aligned, section):
            continue
        current = read_setting(aligned, path)
        if isinstance(current, tuple):
            current = list(current)
        if current == recorded:
            continue
        updates.setdefault(section, {})[field_name] = recorded
        changes.append(f"{path}: {current!r} → {recorded!r}")

    for section, values in updates.items():
        setattr(aligned, section, getattr(aligned, section).model_copy(update=values))
    if changes:
        logger.info("Настройки приведены к слепку: %s", "; ".join(changes))
    return aligned, changes
