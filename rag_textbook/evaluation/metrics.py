"""Метрики качества поиска.

Зачем это существует: прежде качество ретривера оценивалось только через
LLM-судью на конце пайплайна. Судьёй была та же локальная модель 4B, разброс
между прогонами (Contextual Precision 1.0 против 0.269) превышал искомый эффект,
а сами метрики считались по двум контекстам из восьми.

Здесь метрики детерминированы, считаются за секунды и не требуют ни GPU, ни LLM.
Именно по ним принимаются решения о ретривере и графе.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Доля эталонных фрагментов, попавших в top-k."""
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    hits = sum(1 for item in set(relevant) if item in top)
    return hits / len(set(relevant))


def full_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """1, если в top-k пришли **все** эталонные фрагменты вопроса.

    Главная мера серии К (аналог Full@k из NaturalProofs): связывающему
    вопросу половина пары не помогает, а recall засчитывает её как 0.5.
    """
    if not relevant:
        return 0.0
    top = set(retrieved[:k])
    return 1.0 if all(item in top for item in set(relevant)) else 0.0


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    if k <= 0:
        return 0.0
    top = retrieved[:k]
    if not top:
        return 0.0
    relevant_set = set(relevant)
    return sum(1 for item in top if item in relevant_set) / len(top)


def hit_rate_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Есть ли хотя бы один эталонный фрагмент в top-k."""
    relevant_set = set(relevant)
    return 1.0 if any(item in relevant_set for item in retrieved[:k]) else 0.0


def mrr(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    relevant_set = set(relevant)
    for position, item in enumerate(retrieved, start=1):
        if item in relevant_set:
            return 1.0 / position
    return 0.0


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """nDCG с бинарной релевантностью."""
    relevant_set = set(relevant)
    if not relevant_set:
        return 0.0
    dcg = 0.0
    for position, item in enumerate(retrieved[:k], start=1):
        if item in relevant_set:
            dcg += 1.0 / math.log2(position + 1)
    ideal_hits = min(len(relevant_set), k)
    idcg = sum(1.0 / math.log2(position + 1) for position in range(1, ideal_hits + 1))
    return dcg / idcg if idcg > 0 else 0.0


@dataclass
class QueryOutcome:
    """Результат одного вопроса."""

    question_id: str
    question_type: str
    retrieved: list[str]
    relevant: list[str]
    used_graph: bool = False
    graph_share: float = 0.0
    # Доля фрагментов, которых без графового канала в контексте не было бы.
    graph_only_share: float = 0.0
    latency_ms: float = 0.0
    # Длина текста каждого выданного фрагмента, в знаках, по порядку выдачи.
    # Нужна, чтобы уравнять бюджет контекста: при равном числе мест вариант,
    # выдающий длинные фрагменты, получает больше текста (SetCE, табл. 1).
    context_chars: list[int] = field(default_factory=list)
    # Этап 2. Множество, выбранное моделью (setr/picker) — оно же контекст
    # при переменном числе мест; recall@k по ``retrieved`` его не видит.
    selected: list[str] = field(default_factory=list)
    # SEAL: что пришло микрозапросами; ``pool`` — кандидаты до отбора, чтобы
    # отличить починку доступа (эталон пришёл извне пула) от перестановки.
    seal_added: list[str] = field(default_factory=list)
    pool: list[str] = field(default_factory=list)
    selection_status: str = ""


@dataclass
class RetrievalMetrics:
    k_values: tuple[int, ...]
    per_k: dict[int, dict[str, float]] = field(default_factory=dict)
    mrr: float = 0.0
    questions: int = 0
    by_type: dict[str, dict[str, float]] = field(default_factory=dict)
    graph_usage: dict[str, float] = field(default_factory=dict)
    latency: dict[str, float] = field(default_factory=dict)
    selection: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "questions": self.questions,
            "mrr": round(self.mrr, 4),
            "per_k": {
                str(k): {name: round(value, 4) for name, value in metrics.items()}
                for k, metrics in sorted(self.per_k.items())
            },
            "by_type": {
                qtype: {name: round(value, 4) for name, value in metrics.items()}
                for qtype, metrics in sorted(self.by_type.items())
            },
            "graph_usage": {name: round(value, 4) for name, value in self.graph_usage.items()},
            "latency_ms": {name: round(value, 1) for name, value in self.latency.items()},
            "selection": {name: round(value, 4) for name, value in self.selection.items()},
        }

    def summary_line(self, k: int) -> str:
        metrics = self.per_k.get(k, {})
        return (
            f"n={self.questions} "
            f"recall@{k}={metrics.get('recall', 0):.3f} "
            f"ndcg@{k}={metrics.get('ndcg', 0):.3f} "
            f"hit@{k}={metrics.get('hit_rate', 0):.3f} "
            f"mrr={self.mrr:.3f}"
        )


def evaluate_retrieval(
    outcomes: Sequence[QueryOutcome], k_values: Sequence[int] = (1, 3, 5, 8, 10)
) -> RetrievalMetrics:
    k_values = tuple(sorted(set(int(k) for k in k_values)))
    result = RetrievalMetrics(k_values=k_values, questions=len(outcomes))
    if not outcomes:
        return result

    for k in k_values:
        result.per_k[k] = {
            "recall": statistics.fmean(
                recall_at_k(item.retrieved, item.relevant, k) for item in outcomes
            ),
            "precision": statistics.fmean(
                precision_at_k(item.retrieved, item.relevant, k) for item in outcomes
            ),
            "ndcg": statistics.fmean(
                ndcg_at_k(item.retrieved, item.relevant, k) for item in outcomes
            ),
            "hit_rate": statistics.fmean(
                hit_rate_at_k(item.retrieved, item.relevant, k) for item in outcomes
            ),
            "full": statistics.fmean(
                full_at_k(item.retrieved, item.relevant, k) for item in outcomes
            ),
        }
        if any(item.context_chars for item in outcomes):
            result.per_k[k]["chars"] = statistics.fmean(
                sum(item.context_chars[:k]) for item in outcomes
            )

    result.mrr = statistics.fmean(mrr(item.retrieved, item.relevant) for item in outcomes)

    # Разрез по типам вопросов: именно он покажет, помогает ли граф там,
    # где он должен помогать (multi_hop, relation), а не «в среднем».
    by_type: dict[str, list[QueryOutcome]] = {}
    for item in outcomes:
        by_type.setdefault(item.question_type, []).append(item)
    main_k = max(k_values)
    for qtype, items in by_type.items():
        result.by_type[qtype] = {
            "questions": float(len(items)),
            "recall": statistics.fmean(
                recall_at_k(item.retrieved, item.relevant, main_k) for item in items
            ),
            "ndcg": statistics.fmean(
                ndcg_at_k(item.retrieved, item.relevant, main_k) for item in items
            ),
            "mrr": statistics.fmean(mrr(item.retrieved, item.relevant) for item in items),
            "full": statistics.fmean(
                full_at_k(item.retrieved, item.relevant, main_k) for item in items
            ),
        }

    result.graph_usage = {
        "routed_to_graph": statistics.fmean(1.0 if item.used_graph else 0.0 for item in outcomes),
        "avg_graph_share_in_context": statistics.fmean(item.graph_share for item in outcomes),
        # Вклад канала: доля фрагментов, которых без него не было бы.
        "avg_graph_only_share": statistics.fmean(item.graph_only_share for item in outcomes),
    }

    latencies = sorted(item.latency_ms for item in outcomes)
    if latencies:
        result.latency = {
            "mean": statistics.fmean(latencies),
            "p50": latencies[len(latencies) // 2],
            "p95": latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))],
            "max": latencies[-1],
        }
    result.selection = selection_metrics(outcomes)
    return result


def selection_metrics(outcomes: Sequence[QueryOutcome]) -> dict[str, float]:
    """Сводка этапа 2: размер и полнота выбранного множества, вклад SEAL.

    Пусто, если ни один вопрос не прошёл через отбор моделью или SEAL.
    """
    result: dict[str, float] = {}
    statuses = [item.selection_status for item in outcomes if item.selection_status]
    if not statuses and not any(item.selected or item.seal_added for item in outcomes):
        return result
    n = len(outcomes)
    for name in sorted({part for status in statuses for part in status.split(",") if part}):
        result[f"status_{name}"] = sum(name in s.split(",") for s in statuses) / n
    chosen = [item for item in outcomes if item.selected]
    if chosen:
        result["set_share"] = len(chosen) / n
        result["set_size"] = statistics.fmean(len(item.selected) for item in chosen)
        # Полнота по всем вопросам: не выбрал ничего — полнота ноль.
        result["set_recall"] = statistics.fmean(
            recall_at_k(item.selected, item.relevant, len(item.selected) or 1)
            if item.selected
            else 0.0
            for item in outcomes
        )
        result["set_full"] = statistics.fmean(
            full_at_k(item.selected, item.relevant, len(item.selected) or 1)
            if item.selected
            else 0.0
            for item in outcomes
        )
    sealed = [item for item in outcomes if item.seal_added]
    if sealed or any("seal" in s for s in statuses):
        result["seal_touched"] = len(sealed) / n
        result["seal_added"] = statistics.fmean(len(item.seal_added) for item in outcomes)
        outside = 0
        gold_outside = 0
        for item in outcomes:
            pool = set(item.pool)
            relevant = set(item.relevant)
            fresh = [cid for cid in item.seal_added if cid not in pool]
            outside += len(fresh)
            gold_outside += sum(cid in relevant for cid in fresh)
        result["seal_outside_pool"] = outside / n
        # Главное число SEAL: эталон, которого не было среди кандидатов.
        result["seal_gold_outside_pool"] = gold_outside / n
    return result


_METRIC_FUNCS: dict[str, Any] = {
    "recall": recall_at_k,
    "precision": precision_at_k,
    "ndcg": ndcg_at_k,
    "hit_rate": hit_rate_at_k,
    "full": full_at_k,
}


def _per_question(outcomes: Sequence[QueryOutcome], name: str, k: int) -> dict[str, float]:
    if name == "mrr":
        return {item.question_id: mrr(item.retrieved, item.relevant) for item in outcomes}
    func = _METRIC_FUNCS[name]
    return {item.question_id: func(item.retrieved, item.relevant, k) for item in outcomes}


def paired_bootstrap(
    differences: Sequence[float], *, resamples: int = 10000, seed: int = 20260815
) -> dict[str, float]:
    """Доверительный интервал среднего различия по парному бутстрэпу.

    Парный, потому что обе конфигурации оцениваются на одних и тех же вопросах.
    Вопросы различаются по трудности гораздо сильнее, чем конфигурации между
    собой, и если считать выборки независимыми, эта общая дисперсия попадает
    в оценку шума целиком. Тогда реальный эффект тонет: при 140 вопросах
    независимый критерий не увидит прироста меньше 8 процентных пунктов,
    хотя типичный эффект от графа — единицы пунктов.
    """
    values = list(differences)
    n = len(values)
    if n == 0:
        return {"mean": 0.0, "low": 0.0, "high": 0.0, "p_value": 1.0}
    mean = statistics.fmean(values)

    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    low = means[int(0.025 * resamples)]
    high = means[min(resamples - 1, int(0.975 * resamples))]

    # Двусторонняя доля повторных выборок по ту сторону нуля от наблюдаемого
    # среднего. Это не точное p-значение, а его бутстрэп-приближение.
    if mean >= 0:
        tail = sum(1 for value in means if value <= 0.0)
    else:
        tail = sum(1 for value in means if value >= 0.0)
    p_value = min(1.0, 2.0 * tail / resamples)

    return {"mean": mean, "low": low, "high": high, "p_value": p_value}


def compare_paired(
    baseline_outcomes: Sequence[QueryOutcome],
    candidate_outcomes: Sequence[QueryOutcome],
    k: int,
) -> dict[str, Any]:
    """Парное сравнение двух конфигураций по каждому вопросу.

    Возвращает не только средние, но и разбор по вопросам: сколько вопросов
    улучшилось, сколько ухудшилось, и доверительный интервал среднего различия.
    Счёт «выиграло/проиграло» важен сам по себе: прирост, собранный из
    +12 и −9 вопросов, требует другого решения, чем прирост из +3 и −0.
    """
    base_by_id = {item.question_id: item for item in baseline_outcomes}
    cand_by_id = {item.question_id: item for item in candidate_outcomes}
    shared = [qid for qid in base_by_id if qid in cand_by_id]

    def _block(ids: Sequence[str]) -> dict[str, Any]:
        base_items = [base_by_id[qid] for qid in ids]
        cand_items = [cand_by_id[qid] for qid in ids]
        block: dict[str, Any] = {}
        for name in ("recall", "full", "precision", "ndcg", "hit_rate", "mrr"):
            base_values = _per_question(base_items, name, k)
            cand_values = _per_question(cand_items, name, k)
            differences = [cand_values[qid] - base_values[qid] for qid in ids]
            stats = paired_bootstrap(differences)
            block[name] = {
                "baseline": round(statistics.fmean(base_values.values()) if ids else 0.0, 4),
                "candidate": round(statistics.fmean(cand_values.values()) if ids else 0.0, 4),
                "delta": round(stats["mean"], 4),
                "ci_low": round(stats["low"], 4),
                "ci_high": round(stats["high"], 4),
                "p_value": round(stats["p_value"], 4),
                # Значимость — это интервал, не покрывающий ноль. Формулировка
                # «доверительный интервал» честнее, чем голое «да/нет».
                "significant": stats["low"] > 0.0 or stats["high"] < 0.0,
                "improved": sum(1 for value in differences if value > 1e-9),
                "worsened": sum(1 for value in differences if value < -1e-9),
                "unchanged": sum(1 for value in differences if abs(value) <= 1e-9),
            }
        # Бюджет контекста — не мера качества, а условие честного сравнения:
        # прирост при заметно большем тексте надо проверять при равных знаках.
        if any(base_by_id[qid].context_chars or cand_by_id[qid].context_chars for qid in ids):
            block["chars"] = {
                "baseline": round(
                    statistics.fmean(sum(item.context_chars[:k]) for item in base_items), 1
                ),
                "candidate": round(
                    statistics.fmean(sum(item.context_chars[:k]) for item in cand_items), 1
                ),
            }
        return block

    # Разбор по типам обязателен, а не приятен. Реранкер на полном корпусе
    # поднял recall вопросов с формулами на 5 пунктов и уронил многошаговые
    # на 13 — в среднем по набору это выглядело как чистый выигрыш, и решение
    # «принято» было принято по среднему. Такое больше не должно проходить молча.
    by_type: dict[str, Any] = {}
    types: dict[str, list[str]] = {}
    for qid in shared:
        types.setdefault(base_by_id[qid].question_type, []).append(qid)
    for name, ids in sorted(types.items()):
        by_type[name] = {"questions": len(ids), "metrics": _block(ids)}

    return {
        "k": k,
        "questions": len(shared),
        "metrics": _block(shared),
        "by_type": by_type,
    }


def compare(baseline: RetrievalMetrics, candidate: RetrievalMetrics, k: int) -> dict[str, Any]:
    """Сравнение двух конфигураций.

    Отдельно возвращаем предупреждение о размере выборки: на 30 вопросах
    доверительный интервал шире типичного эффекта, и разницу нельзя считать
    установленной — прежние выводы страдали именно этим.
    """
    base = baseline.per_k.get(k, {})
    cand = candidate.per_k.get(k, {})
    deltas = {
        name: round(cand.get(name, 0.0) - base.get(name, 0.0), 4)
        for name in ("recall", "full", "precision", "ndcg", "hit_rate")
    }
    deltas["mrr"] = round(candidate.mrr - baseline.mrr, 4)

    n = min(baseline.questions, candidate.questions)
    # Грубая оценка половины 95% доверительного интервала для доли.
    margin = 1.96 * 0.5 / math.sqrt(n) if n > 0 else 1.0
    significant = {name: abs(value) > margin for name, value in deltas.items() if name != "mrr"}
    return {
        "k": k,
        "baseline": {name: round(value, 4) for name, value in base.items()},
        "candidate": {name: round(value, 4) for name, value in cand.items()},
        "delta": deltas,
        "questions": n,
        "confidence_margin": round(margin, 4),
        "likely_significant": significant,
        "warning": (
            "Выборка мала: различия меньше доверительного интервала считать нельзя"
            if n < 100
            else ""
        ),
    }
