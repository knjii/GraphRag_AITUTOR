"""Тесты мониторинга ресурсов и поиска узких мест."""

from __future__ import annotations

import json
import time

from rag_textbook.observability.analyze import analyze_run, load_run
from rag_textbook.observability.monitor import NullMonitor, ResourceMonitor


def _write_samples(run_dir, samples: list[dict], stages: list[dict]) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    with (run_dir / "samples.jsonl").open("w", encoding="utf-8") as handle:
        for sample in samples:
            handle.write(json.dumps(sample, ensure_ascii=False) + "\n")
    with (run_dir / "stages.jsonl").open("w", encoding="utf-8") as handle:
        for stage in stages:
            handle.write(json.dumps(stage, ensure_ascii=False) + "\n")


def _sample(stage: str, ts: float, gpu: float, cpu: float, **extra) -> dict:
    return {
        "timestamp": ts,
        "stage": stage,
        "gpu_util_pct": gpu,
        "cpu_util_pct": cpu,
        "gpu_mem_used_mib": extra.get("vram", 8000.0),
        "gpu_mem_total_mib": 24576.0,
        "cpu_count": 18,
        "ram_used_mib": extra.get("ram", 12000.0),
        "disk_read_mib": extra.get("read", 100.0),
        "disk_write_mib": extra.get("write", 100.0),
        "gpu_processes": extra.get("processes", {}),
    }


# --------------------------------------------------------------------- монитор


def test_monitor_records_stage_boundaries(tmp_path) -> None:
    monitor = ResourceMonitor(tmp_path / "run", interval_seconds=0.5)
    with monitor:
        with monitor.stage("parse", "book"):
            time.sleep(0.05)
        with monitor.stage("graph", "book"):
            time.sleep(0.05)

    samples, stages = load_run(tmp_path / "run")
    assert len(stages) == 2
    assert [record["stage"] for record in stages] == ["parse", "graph"]
    assert all(record["duration_seconds"] > 0 for record in stages)
    # Каждая стадия обязана оставить хотя бы один замер, иначе короткие стадии
    # выпадут из анализа целиком.
    assert {sample["stage"] for sample in samples} >= {"parse", "graph"}


def test_monitor_restores_previous_stage_on_nesting(tmp_path) -> None:
    monitor = ResourceMonitor(tmp_path / "run", interval_seconds=5.0)
    with monitor, monitor.stage("outer"):
        with monitor.stage("inner"):
            pass
        monitor._write_sample(monitor._collect())  # noqa: SLF001

    samples, _ = load_run(tmp_path / "run")
    assert samples[-1]["stage"] == "outer", "после вложенной стадии метка должна вернуться"


def test_monitor_survives_missing_tools(tmp_path, monkeypatch) -> None:
    """Отсутствие nvidia-smi не должно ломать прогон — только обеднять метрики."""
    monkeypatch.setattr("rag_textbook.observability.monitor._query_gpu", lambda: {}, raising=True)
    monitor = ResourceMonitor(tmp_path / "run", interval_seconds=0.5)
    with monitor, monitor.stage("parse"):
        time.sleep(0.05)

    samples, stages = load_run(tmp_path / "run")
    assert samples and stages
    assert samples[0].get("gpu_util_pct") is None


def test_null_monitor_is_transparent() -> None:
    monitor = NullMonitor()
    with monitor, monitor.stage("anything", "doc"):
        pass
    assert monitor.enabled is False


# ---------------------------------------------------------------------- анализ


def test_gpu_bound_stage_is_detected(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [_sample("parse", 1000 + i, gpu=94.0, cpu=25.0) for i in range(10)],
        [{"stage": "parse", "duration_seconds": 600.0}],
    )
    report = analyze_run(run_dir)
    stage = report["stages"][0]

    assert stage["bottleneck"] == "gpu"
    assert "карт" in stage["recommendation"].lower()


def test_cpu_bound_stage_is_detected(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [_sample("chunk", 1000 + i, gpu=3.0, cpu=88.0) for i in range(10)],
        [{"stage": "chunk", "duration_seconds": 300.0}],
    )
    report = analyze_run(run_dir)
    assert report["stages"][0]["bottleneck"] == "cpu"


def test_waiting_stage_is_detected(tmp_path) -> None:
    """Ключевой случай: ресурсы свободны, а стадия идёт — значит мы ждём ответов."""
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [_sample("graph", 1000 + i, gpu=4.0, cpu=12.0) for i in range(20)],
        [{"stage": "graph", "duration_seconds": 3600.0}],
    )
    report = analyze_run(run_dir)
    stage = report["stages"][0]

    assert stage["bottleneck"] == "waiting"
    assert stage["gpu_idle_share_pct"] == 100.0
    assert (
        "vllm" in stage["recommendation"].lower() or "параллел" in stage["recommendation"].lower()
    )


def test_pipelining_useful_only_with_different_bottlenecks(tmp_path) -> None:
    """Конвейеризация помогает, только если стадии упираются в разные ресурсы."""
    same = tmp_path / "same"
    _write_samples(
        same,
        [_sample("parse", 1000 + i, gpu=92.0, cpu=20.0) for i in range(10)]
        + [_sample("graph", 2000 + i, gpu=90.0, cpu=20.0) for i in range(10)],
        [
            {"stage": "parse", "duration_seconds": 1800.0},
            {"stage": "graph", "duration_seconds": 1800.0},
        ],
    )
    assert analyze_run(same)["summary"]["pipelining_likely_useful"] is False

    different = tmp_path / "different"
    _write_samples(
        different,
        [_sample("parse", 1000 + i, gpu=92.0, cpu=20.0) for i in range(10)]
        + [_sample("graph", 2000 + i, gpu=5.0, cpu=10.0) for i in range(10)],
        [
            {"stage": "parse", "duration_seconds": 1800.0},
            {"stage": "graph", "duration_seconds": 1800.0},
        ],
    )
    assert analyze_run(different)["summary"]["pipelining_likely_useful"] is True


def test_summary_reports_memory_headroom(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [_sample("graph", 1000 + i, gpu=50.0, cpu=30.0, vram=11000.0) for i in range(5)],
        [{"stage": "graph", "duration_seconds": 100.0}],
    )
    summary = analyze_run(run_dir)["summary"]

    assert summary["gpu_memory_peak_mib"] == 11000.0
    assert summary["gpu_memory_headroom_mib"] == 24576.0 - 11000.0
    # Запас больше 8 ГБ означает, что модель крупнее поместится.
    assert any("крупнее" in note for note in summary["notes"])


def test_heaviest_stage_is_first(tmp_path) -> None:
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [_sample("parse", 1000 + i, gpu=90.0, cpu=20.0) for i in range(5)]
        + [_sample("embed", 2000 + i, gpu=40.0, cpu=20.0) for i in range(5)],
        [
            {"stage": "parse", "duration_seconds": 600.0},
            {"stage": "embed", "duration_seconds": 60.0},
        ],
    )
    report = analyze_run(run_dir)

    assert report["stages"][0]["stage"] == "parse"
    assert report["summary"]["heaviest_stage"] == "parse"
    assert report["summary"]["heaviest_stage_share_pct"] > 80


def test_per_process_vram_is_tracked(tmp_path) -> None:
    """Показывает, кто именно держит память — например, что MinerU не выгрузился."""
    run_dir = tmp_path / "run"
    _write_samples(
        run_dir,
        [
            _sample(
                "graph",
                1000 + i,
                gpu=30.0,
                cpu=20.0,
                processes={"python": 8000.0, "ollama": 3000.0},
            )
            for i in range(3)
        ],
        [{"stage": "graph", "duration_seconds": 120.0}],
    )
    stage = analyze_run(run_dir)["stages"][0]

    assert stage["gpu_processes_peak_mib"]["python"] == 8000.0
    assert stage["gpu_processes_peak_mib"]["ollama"] == 3000.0


def test_analyze_missing_run(tmp_path) -> None:
    assert "error" in analyze_run(tmp_path / "nope")
