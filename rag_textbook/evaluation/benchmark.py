"""Замер пропускной способности сервера инференса.

Зачем: выбор движка инференса (Ollama, vLLM, SGLang) и уровня параллелизма
нельзя делать по обзорам — он зависит от конкретной карты, модели, длины промпта
и формы нагрузки. Этот модуль меряет то, что нас действительно интересует:
**сколько чанков в секунду мы способны обработать на стадии извлечения графа**.

Нагрузка имитирует реальную: длинный общий префикс инструкции плюс переменный
текст фрагмента, структурированный вывод по схеме. Именно на таком профиле
разница между движками максимальна — из-за кэширования общего префикса.
"""

from __future__ import annotations

import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.graph.extractor import EXTRACTION_SCHEMA, RELATION_LABELS
from rag_textbook.logging_setup import get_logger

logger = get_logger("evaluation.benchmark")

# Текст-донор: осмысленный технический фрагмент нужного порядка длины.
_SAMPLE_PARAGRAPH = (
    "Сингулярное разложение матрицы A размера m на n представляет её в виде "
    "произведения трёх матриц: ортогональной матрицы U, диагональной матрицы "
    "сингулярных чисел Sigma и транспонированной ортогональной матрицы V. "
    "Сингулярные числа неотрицательны и упорядочены по убыванию. Разложение "
    "применяется для понижения размерности, приближения матрицы матрицей "
    "меньшего ранга и вычисления псевдообратной матрицы. Метод главных "
    "компонент выражается через сингулярное разложение центрированной матрицы "
    "данных, а квадраты сингулярных чисел пропорциональны дисперсиям вдоль "
    "главных направлений. "
)

_PROMPT_PREFIX = (
    "Ты извлекаешь граф знаний из фрагмента учебника по математике.\n"
    "Извлеки сущности и связи между ними.\n"
    "Поле relation выбирай строго из списка:\n"
    + "\n".join(f"- {label}" for label in RELATION_LABELS)
    + "\n\nФрагмент:\n"
)


def _make_prompt(index: int, target_chars: int) -> str:
    """Промпт с общим префиксом и уникальным телом.

    Уникальность обязательна: одинаковые запросы движок вернёт из кэша ответов,
    и замер покажет скорость кэша, а не скорость модели.
    """
    body = (f"[фрагмент {index}] " + _SAMPLE_PARAGRAPH * 8)[:target_chars]
    return _PROMPT_PREFIX + body


@dataclass
class BenchmarkPoint:
    concurrency: int
    requests: int
    ok: int
    failed: int
    wall_seconds: float
    latencies_ms: list[float] = field(default_factory=list)

    @property
    def throughput_rps(self) -> float:
        return self.ok / self.wall_seconds if self.wall_seconds > 0 else 0.0

    @property
    def latency_p50_ms(self) -> float:
        return statistics.median(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def latency_p95_ms(self) -> float:
        if not self.latencies_ms:
            return 0.0
        ordered = sorted(self.latencies_ms)
        return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]

    def chunks_per_hour(self) -> float:
        return self.throughput_rps * 3600

    def as_dict(self) -> dict[str, Any]:
        return {
            "concurrency": self.concurrency,
            "requests": self.requests,
            "ok": self.ok,
            "failed": self.failed,
            "wall_seconds": round(self.wall_seconds, 2),
            "throughput_rps": round(self.throughput_rps, 3),
            "chunks_per_hour": round(self.chunks_per_hour()),
            "latency_p50_ms": round(self.latency_p50_ms, 1),
            "latency_p95_ms": round(self.latency_p95_ms, 1),
        }


def run_concurrency_point(
    llm: LLMClient,
    *,
    concurrency: int,
    requests: int,
    prompt_chars: int,
    max_tokens: int,
    structured: bool,
) -> BenchmarkPoint:
    prompts = [_make_prompt(index, prompt_chars) for index in range(requests)]
    latencies: list[float] = []
    ok = 0
    failed = 0

    def one(prompt: str) -> float | None:
        started = time.perf_counter()
        try:
            llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="extraction",
                json_schema=EXTRACTION_SCHEMA if structured else None,
                max_tokens=max_tokens,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Запрос не удался: %s", str(exc)[:120])
            return None
        return (time.perf_counter() - started) * 1000

    wall_started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(one, prompt) for prompt in prompts]
        for future in as_completed(futures):
            latency = future.result()
            if latency is None:
                failed += 1
            else:
                ok += 1
                latencies.append(latency)
    wall = time.perf_counter() - wall_started

    return BenchmarkPoint(
        concurrency=concurrency,
        requests=requests,
        ok=ok,
        failed=failed,
        wall_seconds=wall,
        latencies_ms=latencies,
    )


def run_throughput_sweep(
    llm: LLMClient,
    *,
    concurrency_levels: tuple[int, ...] = (1, 2, 4, 8, 16),
    requests_per_level: int = 24,
    prompt_chars: int = 3500,
    max_tokens: int = 512,
    structured: bool = True,
    warmup_requests: int = 2,
) -> dict[str, Any]:
    """Прогоняет нагрузку на нескольких уровнях параллелизма.

    Смысл развёртки: найти точку, после которой рост параллелизма перестаёт
    давать пропускную способность и начинает только увеличивать задержку.
    Именно это значение и надо ставить в ``LLM_MAX_CONCURRENCY``.
    """

    if warmup_requests > 0:
        logger.info("Прогрев: %s запросов", warmup_requests)
        run_concurrency_point(
            llm,
            concurrency=1,
            requests=warmup_requests,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            structured=structured,
        )

    points: list[BenchmarkPoint] = []
    for concurrency in concurrency_levels:
        logger.info("Замер при параллелизме %s (%s запросов)", concurrency, requests_per_level)
        point = run_concurrency_point(
            llm,
            concurrency=concurrency,
            requests=requests_per_level,
            prompt_chars=prompt_chars,
            max_tokens=max_tokens,
            structured=structured,
        )
        points.append(point)
        logger.info(
            "  пропускная способность %.2f зап/с (%s чанков/час), p50 %.0f мс, ошибок %s",
            point.throughput_rps,
            round(point.chunks_per_hour()),
            point.latency_p50_ms,
            point.failed,
        )
        # Дальше наращивать параллелизм бессмысленно: движок уже отказывает.
        if point.failed > point.ok:
            logger.warning("Больше половины запросов упало — прекращаю развёртку")
            break

    best = max(points, key=lambda item: item.throughput_rps) if points else None
    scaling = 0.0
    if points and points[0].throughput_rps > 0 and best is not None:
        scaling = best.throughput_rps / points[0].throughput_rps

    return {
        "config": {
            "prompt_chars": prompt_chars,
            "max_tokens": max_tokens,
            "structured_output": structured,
            "requests_per_level": requests_per_level,
        },
        "points": [point.as_dict() for point in points],
        "best_concurrency": best.concurrency if best else None,
        "best_throughput_rps": round(best.throughput_rps, 3) if best else 0.0,
        "best_chunks_per_hour": round(best.chunks_per_hour()) if best else 0,
        # Во сколько раз параллелизм ускорил работу против одного потока.
        # Значение около единицы означает, что движок не батчит запросы.
        "scaling_factor": round(scaling, 2),
    }


def estimate_indexing_time(chunks: int, throughput_rps: float) -> dict[str, Any]:
    """Пересчитывает пропускную способность в время индексации корпуса."""
    if throughput_rps <= 0:
        return {"chunks": chunks, "hours": None, "note": "нулевая пропускная способность"}
    seconds = chunks / throughput_rps
    return {
        "chunks": chunks,
        "seconds": round(seconds),
        "hours": round(seconds / 3600, 2),
        "note": "только стадия извлечения графа, без разбора PDF и эмбеддингов",
    }
