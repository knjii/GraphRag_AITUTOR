"""Разбор собранных метрик и вывод об узких местах.

Задача модуля — превратить сырые замеры в утверждение вида «стадия графа
простаивает 70% времени, ограничение не в карте, а в ожидании ответов сервера
инференса». Именно такое утверждение позволяет принять решение, а не гадать.

Классификация стадии выполняется по простым порогам. Это сознательный выбор:
прозрачное правило, которое можно проверить глазами по тем же числам,
полезнее непрозрачной эвристики.
"""

from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag_textbook.logging_setup import get_logger

logger = get_logger("observability.analyze")

Bottleneck = Literal["gpu", "cpu", "disk", "waiting", "mixed", "unknown"]

# Пороги подобраны так, чтобы «занят» означало действительно занят.
GPU_BUSY_PCT = 70.0
CPU_BUSY_PCT = 70.0
IDLE_PCT = 10.0
DISK_BUSY_MIB_PER_S = 80.0


@dataclass
class StageVerdict:
    """Итог по одной стадии конвейера."""

    stage: str
    samples: int
    duration_seconds: float
    gpu_util_mean: float | None = None
    gpu_util_p95: float | None = None
    gpu_idle_share_pct: float | None = None
    gpu_mem_peak_mib: float | None = None
    gpu_mem_total_mib: float | None = None
    cpu_util_mean: float | None = None
    cpu_count: int | None = None
    ram_peak_mib: float | None = None
    disk_read_mib_per_s: float | None = None
    disk_write_mib_per_s: float | None = None
    gpu_processes_peak: dict[str, float] = field(default_factory=dict)
    bottleneck: Bottleneck = "unknown"
    explanation: str = ""
    recommendation: str = ""

    def as_dict(self) -> dict[str, Any]:
        payload = {
            "stage": self.stage,
            "samples": self.samples,
            "duration_seconds": round(self.duration_seconds, 1),
            "duration_minutes": round(self.duration_seconds / 60, 1),
            "gpu_util_mean_pct": _round(self.gpu_util_mean),
            "gpu_util_p95_pct": _round(self.gpu_util_p95),
            "gpu_idle_share_pct": _round(self.gpu_idle_share_pct),
            "gpu_mem_peak_mib": _round(self.gpu_mem_peak_mib),
            "gpu_mem_total_mib": _round(self.gpu_mem_total_mib),
            "cpu_util_mean_pct": _round(self.cpu_util_mean),
            "cpu_count": self.cpu_count,
            "ram_peak_mib": _round(self.ram_peak_mib),
            "disk_read_mib_per_s": _round(self.disk_read_mib_per_s),
            "disk_write_mib_per_s": _round(self.disk_write_mib_per_s),
            "gpu_processes_peak_mib": {k: _round(v) for k, v in self.gpu_processes_peak.items()},
            "bottleneck": self.bottleneck,
            "explanation": self.explanation,
            "recommendation": self.recommendation,
        }
        return payload


def _round(value: float | None, digits: int = 1) -> float | None:
    return None if value is None else round(float(value), digits)


def _mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def _p95(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * 0.95))]


def load_run(run_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Читает замеры и границы стадий."""
    run_dir = Path(run_dir)
    samples: list[dict[str, Any]] = []
    stages: list[dict[str, Any]] = []

    samples_path = run_dir / "samples.jsonl"
    if samples_path.is_file():
        for line in samples_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    samples.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    stages_path = run_dir / "stages.jsonl"
    if stages_path.is_file():
        for line in stages_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    stages.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return samples, stages


def _classify(verdict: StageVerdict) -> tuple[Bottleneck, str, str]:
    """Определяет узкое место стадии и что с ним делать."""
    gpu = verdict.gpu_util_mean
    cpu = verdict.cpu_util_mean
    disk = max(verdict.disk_read_mib_per_s or 0.0, verdict.disk_write_mib_per_s or 0.0)
    idle = verdict.gpu_idle_share_pct

    gpu_busy = gpu is not None and gpu >= GPU_BUSY_PCT
    cpu_busy = cpu is not None and cpu >= CPU_BUSY_PCT
    disk_busy = disk >= DISK_BUSY_MIB_PER_S

    if gpu_busy and cpu_busy:
        return (
            "mixed",
            f"И карта ({gpu:.0f}%), и процессор ({cpu:.0f}%) загружены.",
            "Стадия использует ресурсы полно. Ускорение — только сменой алгоритма или железа.",
        )
    if gpu_busy:
        return (
            "gpu",
            f"Карта загружена на {gpu:.0f}%, процессор на {cpu or 0:.0f}%.",
            "Упираемся в карту. Помогут квантование, батчи побольше или карта помощнее. "
            "Конвейеризация стадий выигрыша не даст — карта уже занята.",
        )
    if cpu_busy:
        return (
            "cpu",
            f"Процессор загружен на {cpu:.0f}%, карта на {gpu or 0:.0f}%.",
            "Упираемся в процессор. Помогут распараллеливание по ядрам "
            "и вынос этой стадии в отдельные процессы.",
        )
    if disk_busy:
        return (
            "disk",
            f"Диск: чтение {verdict.disk_read_mib_per_s or 0:.0f}, "
            f"запись {verdict.disk_write_mib_per_s or 0:.0f} МиБ/с.",
            "Упираемся в диск. Помогут батчи покрупнее при записи и меньше промежуточных файлов.",
        )
    if idle is not None and idle >= 50.0:
        return (
            "waiting",
            f"Карта простаивает {idle:.0f}% времени, процессор на {cpu or 0:.0f}%.",
            "Ресурсы свободны, а стадия идёт — значит мы ждём. Наиболее вероятно: "
            "запросы к серверу инференса выполняются по очереди. "
            "Поднимите LLM_MAX_CONCURRENCY и сравните с `rag-textbook bench`; "
            "если движок не батчит — переходите на vLLM или SGLang.",
        )
    return (
        "mixed",
        f"Карта {gpu or 0:.0f}%, процессор {cpu or 0:.0f}% — ни один ресурс не насыщен.",
        "Узкое место не выражено. Смотрите долю простоя карты и задержки стадии по отдельности.",
    )


def analyze_run(run_dir: Path) -> dict[str, Any]:
    """Полный отчёт по прогону."""
    samples, stages = load_run(run_dir)
    if not samples:
        return {"error": f"Замеры не найдены в {run_dir}"}

    by_stage: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        by_stage.setdefault(str(sample.get("stage") or "(вне стадий)"), []).append(sample)

    durations: dict[str, float] = {}
    for record in stages:
        name = str(record.get("stage") or "")
        durations[name] = durations.get(name, 0.0) + float(record.get("duration_seconds") or 0.0)

    verdicts: list[StageVerdict] = []
    for stage_name, stage_samples in by_stage.items():
        gpu_util = [
            float(s["gpu_util_pct"]) for s in stage_samples if s.get("gpu_util_pct") is not None
        ]
        gpu_mem = [
            float(s["gpu_mem_used_mib"])
            for s in stage_samples
            if s.get("gpu_mem_used_mib") is not None
        ]
        cpu_util = [
            float(s["cpu_util_pct"]) for s in stage_samples if s.get("cpu_util_pct") is not None
        ]
        ram = [float(s["ram_used_mib"]) for s in stage_samples if s.get("ram_used_mib") is not None]

        # Счётчики диска монотонно растут: скорость получаем как разность
        # на границах интервала стадии.
        read_rate = write_rate = None
        timestamps = [float(s["timestamp"]) for s in stage_samples if s.get("timestamp")]
        span = (max(timestamps) - min(timestamps)) if len(timestamps) > 1 else 0.0
        if span > 0:
            reads = [
                float(s["disk_read_mib"])
                for s in stage_samples
                if s.get("disk_read_mib") is not None
            ]
            writes = [
                float(s["disk_write_mib"])
                for s in stage_samples
                if s.get("disk_write_mib") is not None
            ]
            if len(reads) > 1:
                read_rate = max(0.0, (max(reads) - min(reads)) / span)
            if len(writes) > 1:
                write_rate = max(0.0, (max(writes) - min(writes)) / span)

        processes_peak: dict[str, float] = {}
        for sample in stage_samples:
            for name, memory in (sample.get("gpu_processes") or {}).items():
                processes_peak[name] = max(processes_peak.get(name, 0.0), float(memory))

        verdict = StageVerdict(
            stage=stage_name,
            samples=len(stage_samples),
            duration_seconds=durations.get(stage_name, span),
            gpu_util_mean=_mean(gpu_util),
            gpu_util_p95=_p95(gpu_util),
            gpu_idle_share_pct=(
                100.0 * sum(1 for value in gpu_util if value < IDLE_PCT) / len(gpu_util)
                if gpu_util
                else None
            ),
            gpu_mem_peak_mib=max(gpu_mem) if gpu_mem else None,
            gpu_mem_total_mib=next(
                (
                    float(s["gpu_mem_total_mib"])
                    for s in stage_samples
                    if s.get("gpu_mem_total_mib")
                ),
                None,
            ),
            cpu_util_mean=_mean(cpu_util),
            cpu_count=next(
                (int(s["cpu_count"]) for s in stage_samples if s.get("cpu_count")), None
            ),
            ram_peak_mib=max(ram) if ram else None,
            disk_read_mib_per_s=read_rate,
            disk_write_mib_per_s=write_rate,
            gpu_processes_peak=processes_peak,
        )
        verdict.bottleneck, verdict.explanation, verdict.recommendation = _classify(verdict)
        verdicts.append(verdict)

    verdicts.sort(key=lambda item: item.duration_seconds, reverse=True)
    total_duration = sum(item.duration_seconds for item in verdicts) or 1.0

    return {
        "run_dir": str(run_dir),
        "total_samples": len(samples),
        "total_duration_seconds": round(total_duration, 1),
        "stages": [
            {
                **verdict.as_dict(),
                "share_of_time_pct": round(100.0 * verdict.duration_seconds / total_duration, 1),
            }
            for verdict in verdicts
        ],
        "summary": _summarize(verdicts, total_duration),
    }


def _summarize(verdicts: list[StageVerdict], total_duration: float) -> dict[str, Any]:
    """Главный вывод: на что тратится время и что чинить первым."""
    if not verdicts:
        return {}

    heaviest = verdicts[0]
    peak_memory = max((v.gpu_mem_peak_mib or 0.0) for v in verdicts)
    total_memory = next((v.gpu_mem_total_mib for v in verdicts if v.gpu_mem_total_mib), None)

    waiting_time = sum(v.duration_seconds for v in verdicts if v.bottleneck == "waiting")
    gpu_time = sum(v.duration_seconds for v in verdicts if v.bottleneck == "gpu")

    # Конвейеризация стадий помогает, только когда они упираются в разные ресурсы.
    # На одной карте «GPU + GPU» не ускорится: стадии просто поделят её по времени.
    bottlenecks = {v.bottleneck for v in verdicts if v.duration_seconds > total_duration * 0.05}
    pipelining_useful = len(bottlenecks - {"unknown"}) > 1

    notes: list[str] = [
        f"Больше всего времени занимает стадия «{heaviest.stage}» "
        f"({heaviest.duration_seconds / 60:.0f} мин, "
        f"{100 * heaviest.duration_seconds / total_duration:.0f}% прогона). "
        f"{heaviest.explanation} {heaviest.recommendation}"
    ]

    if waiting_time > total_duration * 0.3:
        notes.append(
            f"Около {100 * waiting_time / total_duration:.0f}% времени ресурсы простаивают "
            "в ожидании. Это самый дешёвый резерв: обычно решается повышением "
            "параллелизма запросов или сменой движка инференса."
        )
    if gpu_time > total_duration * 0.6:
        notes.append(
            "Карта занята большую часть прогона — конвейеризация стадий заметного "
            "выигрыша не даст, стадии будут делить одну карту по времени."
        )
    if total_memory and peak_memory:
        headroom = total_memory - peak_memory
        notes.append(
            f"Пик видеопамяти {peak_memory:.0f} из {total_memory:.0f} МиБ, "
            f"свободно {headroom:.0f} МиБ. "
            + (
                "Запаса хватает на модель существенно крупнее."
                if headroom > 8000
                else "Запас невелик, модель крупнее потребует разделения стадий во времени."
            )
        )

    return {
        "heaviest_stage": heaviest.stage,
        "heaviest_stage_share_pct": round(100 * heaviest.duration_seconds / total_duration, 1),
        "primary_bottleneck": heaviest.bottleneck,
        "gpu_memory_peak_mib": _round(peak_memory),
        "gpu_memory_total_mib": _round(total_memory),
        "gpu_memory_headroom_mib": _round((total_memory - peak_memory) if total_memory else None),
        # Ответ на вопрос «стоит ли конвейеризовать стадии»: да, только если
        # тяжёлые стадии упираются в разные ресурсы.
        "pipelining_likely_useful": pipelining_useful,
        "notes": notes,
    }
