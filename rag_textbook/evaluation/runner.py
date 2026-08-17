"""Прогон оценки и A/B-сравнение конфигураций.

Прежний прогон DeepEval выполнялся строго последовательно
(``AsyncConfig(run_async=False, max_concurrent=1)``), из-за чего оценка занимала
часы и упиралась в таймауты судьи. Здесь измерение поиска не требует LLM вообще,
поэтому полный прогон по 150 вопросам занимает секунды, а сравнение двух
конфигураций становится рутинной операцией, а не событием на день.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path
from typing import Any

from rag_textbook.config import Settings
from rag_textbook.context import AppContext, build_context
from rag_textbook.evaluation.metrics import (
    QueryOutcome,
    RetrievalMetrics,
    compare,
    compare_paired,
    evaluate_retrieval,
)
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import GoldQuestion

logger = get_logger("evaluation.runner")


def run_retrieval_evaluation(
    context: AppContext,
    questions: Sequence[GoldQuestion],
    *,
    max_workers: int = 4,
) -> tuple[RetrievalMetrics, list[QueryOutcome]]:
    settings = context.settings
    k_values = settings.evaluation.k_values
    max_k = max(k_values)

    def evaluate_one(question: GoldQuestion) -> QueryOutcome:
        result = context.retrieval.retrieve(question.question, history=[])
        retrieved = [item.chunk.id for item in result.chunks][:max_k]
        return QueryOutcome(
            question_id=question.id,
            question_type=question.question_type,
            retrieved=retrieved,
            relevant=list(question.gold_chunk_ids),
            used_graph=bool(result.route and result.route.use_graph),
            graph_share=result.graph_share,
            graph_only_share=result.graph_only_share,
            latency_ms=result.timings_ms.get("total", 0.0),
        )

    logger.info("Оценка поиска: вопросов=%s, параллелизм=%s", len(questions), max_workers)
    if max_workers <= 1:
        outcomes = [evaluate_one(question) for question in questions]
    else:
        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            outcomes = list(pool.map(evaluate_one, questions))

    metrics = evaluate_retrieval(outcomes, k_values)
    logger.info("Результат: %s", metrics.summary_line(settings.retrieval.top_k))
    return metrics, outcomes


def save_evaluation(
    metrics: RetrievalMetrics,
    outcomes: Sequence[QueryOutcome],
    settings: Settings,
    *,
    label: str,
    metrics_dir: Path | None = None,
) -> Path:
    directory = Path(metrics_dir or settings.paths.metrics_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = directory / f"retrieval_eval_{label}_{stamp}.json"
    payload: dict[str, Any] = {
        "label": label,
        "created_at": stamp,
        # Конфигурация сохраняется вместе с метриками: без неё результат
        # невоспроизводим и его нельзя честно сравнить с другим прогоном.
        "config": {
            "top_k": settings.retrieval.top_k,
            "fusion": settings.retrieval.fusion,
            "router_mode": settings.retrieval.router_mode,
            "reranker_enabled": settings.reranker.enabled,
            "reranker_model": settings.reranker.model if settings.reranker.enabled else "",
            "graph_enabled": settings.graph.enabled and settings.graph.retrieval_enabled,
            "graph_weight": settings.graph.weight,
            "graph_hops": settings.graph.expansion_hops,
            "graph_rel_types": list(settings.graph.expansion_rel_types),
            "embedding_model": settings.embedding.model,
            "min_graph_docs": settings.retrieval.min_graph_docs,
        },
        "metrics": metrics.as_dict(),
        "outcomes": [
            {
                "question_id": item.question_id,
                "type": item.question_type,
                "retrieved": item.retrieved,
                "relevant": item.relevant,
                "used_graph": item.used_graph,
                "graph_share": round(item.graph_share, 3),
                "graph_only_share": round(item.graph_only_share, 3),
                "latency_ms": round(item.latency_ms, 1),
            }
            for item in outcomes
        ],
    }
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Результаты оценки сохранены: %s", path)
    return path


def load_outcomes(path: Path) -> tuple[str, list[QueryOutcome]]:
    """Читает сохранённый прогон обратно в объекты.

    Нужно там, где ``eval ab`` бессилен. Он переключает настройки на лету
    и потому умеет сравнивать только то, что влияет на запрос. Порог отсечения
    хабов влияет на индекс: граф с ним пересобирается, и две конфигурации
    физически не могут существовать одновременно. Остаётся снять два прогона
    и сравнить их по файлам — парно, по одним и тем же вопросам.
    """
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    outcomes = [
        QueryOutcome(
            question_id=str(row.get("question_id") or ""),
            question_type=str(row.get("type") or ""),
            retrieved=[str(item) for item in (row.get("retrieved") or [])],
            relevant=[str(item) for item in (row.get("relevant") or [])],
            used_graph=bool(row.get("used_graph")),
            graph_share=float(row.get("graph_share") or 0.0),
            graph_only_share=float(row.get("graph_only_share") or 0.0),
            latency_ms=float(row.get("latency_ms") or 0.0),
        )
        for row in payload.get("outcomes", [])
    ]
    return str(payload.get("label") or Path(path).stem), outcomes


def run_ab_comparison(
    questions: Sequence[GoldQuestion],
    baseline_overrides: dict[str, Any],
    candidate_overrides: dict[str, Any],
    *,
    base_settings: Settings | None = None,
    labels: tuple[str, str] = ("baseline", "candidate"),
) -> dict[str, Any]:
    """Сравнивает две конфигурации на одном наборе вопросов.

    Типовое применение — доказать или опровергнуть пользу графа: базовая
    конфигурация с ``graph.retrieval_enabled=False`` против кандидата с графом.
    Раньше такой прогон занимал часы и упирался в таймауты судьи, поэтому
    вопрос «даёт ли граф прирост» оставался без ответа.
    """

    results: dict[str, RetrievalMetrics] = {}
    outcomes_by_label: dict[str, list[QueryOutcome]] = {}
    for label, overrides in zip(labels, (baseline_overrides, candidate_overrides), strict=True):
        settings = (base_settings or Settings()).model_copy(deep=True)
        for path, value in overrides.items():
            section_name, _, field_name = path.partition(".")
            section = getattr(settings, section_name, None)
            if section is None or not field_name:
                raise ValueError(f"Неизвестный параметр конфигурации: {path}")
            setattr(section, field_name, value)

        context = build_context(settings)
        try:
            metrics, outcomes = run_retrieval_evaluation(
                context, questions, max_workers=settings.evaluation.max_concurrency
            )
            save_evaluation(metrics, outcomes, settings, label=label)
            results[label] = metrics
            outcomes_by_label[label] = list(outcomes)
        finally:
            context.close()

    baseline_metrics = results[labels[0]]
    candidate_metrics = results[labels[1]]
    # Сравнивать надо по всему контексту, который реально отдаётся. Для
    # связывающих вопросов квота выдачи шире, и при жёстком k=top_k прирост,
    # приходящий в позиции с девятой по шестнадцатую, невидим: A/B показывал
    # нулевую разницу там, где прямой замер давал +0.044 recall.
    retrieval = (base_settings or Settings()).retrieval
    k = max(retrieval.top_k, retrieval.top_k_linking)
    return {
        "labels": list(labels),
        # Основной вывод делается по парному сравнению: конфигурации оценены
        # на одних и тех же вопросах, и учитывать это обязательно.
        "paired": compare_paired(outcomes_by_label[labels[0]], outcomes_by_label[labels[1]], k),
        "comparison": compare(baseline_metrics, candidate_metrics, k),
        labels[0]: baseline_metrics.as_dict(),
        labels[1]: candidate_metrics.as_dict(),
    }
