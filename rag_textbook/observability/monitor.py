"""Сбор метрик ресурсов с привязкой к стадиям конвейера.

Зачем именно так, а не просто ``nvidia-smi -l``: голая загрузка карты по времени
не говорит ничего, пока неизвестно, какая стадия в этот момент работала.
Число «карта загружена на 45%» бесполезно; «на стадии разбора карта загружена
на 92%, а на стадии графа — на 12%» уже прямо указывает, что оптимизировать.

Монитор пишет два потока данных:

* ``samples.jsonl`` — замеры каждые N секунд (GPU, CPU, память, диск);
* ``stages.jsonl`` — границы стадий с метками времени.

Анализ выполняется отдельно (:mod:`rag_textbook.observability.analyze`),
чтобы сбор данных был максимально дешёвым и не влиял на замеряемое.
"""

from __future__ import annotations

import json
import subprocess
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook.logging_setup import get_logger

logger = get_logger("observability.monitor")

_NVIDIA_QUERY = (
    "utilization.gpu,utilization.memory,memory.used,memory.total,"
    "temperature.gpu,power.draw,clocks.sm"
)


@dataclass
class ResourceSample:
    """Один замер состояния машины."""

    timestamp: float
    stage: str = ""
    document: str = ""

    gpu_util_pct: float | None = None
    gpu_mem_util_pct: float | None = None
    gpu_mem_used_mib: float | None = None
    gpu_mem_total_mib: float | None = None
    gpu_temp_c: float | None = None
    gpu_power_w: float | None = None
    gpu_clock_mhz: float | None = None
    # Кто именно держит видеопамять: позволяет увидеть, что MinerU не выгрузился.
    gpu_processes: dict[str, float] = field(default_factory=dict)

    cpu_util_pct: float | None = None
    cpu_count: int | None = None
    ram_used_mib: float | None = None
    ram_total_mib: float | None = None
    disk_read_mib: float | None = None
    disk_write_mib: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _query_gpu() -> dict[str, Any]:
    """Состояние карты через nvidia-smi.

    Через subprocess, а не через pynvml: одна зависимость меньше, а накладные
    расходы в пару десятков миллисекунд при интервале в секунды несущественны.
    """
    result: dict[str, Any] = {}
    try:
        raw = subprocess.run(  # noqa: S603
            ["nvidia-smi", f"--query-gpu={_NVIDIA_QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return result

    line = raw.splitlines()[0] if raw else ""
    parts = [part.strip() for part in line.split(",")]
    keys = [
        "gpu_util_pct",
        "gpu_mem_util_pct",
        "gpu_mem_used_mib",
        "gpu_mem_total_mib",
        "gpu_temp_c",
        "gpu_power_w",
        "gpu_clock_mhz",
    ]
    for key, value in zip(keys, parts, strict=False):
        try:
            result[key] = float(value)
        except (TypeError, ValueError):
            result[key] = None

    try:
        apps = subprocess.run(  # noqa: S603
            [
                "nvidia-smi",
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
        processes: dict[str, float] = {}
        for row in apps.splitlines():
            name, _, memory = row.partition(",")
            name = Path(name.strip()).name or name.strip()
            try:
                processes[name] = processes.get(name, 0.0) + float(memory.strip())
            except ValueError:
                continue
        result["gpu_processes"] = processes
    except (OSError, subprocess.SubprocessError):
        result["gpu_processes"] = {}

    return result


def _query_host() -> dict[str, Any]:
    try:
        import psutil
    except ImportError:
        return {}

    memory = psutil.virtual_memory()
    payload: dict[str, Any] = {
        # interval=None — мгновенное значение с прошлого вызова, не блокирует поток.
        "cpu_util_pct": psutil.cpu_percent(interval=None),
        "cpu_count": psutil.cpu_count(logical=True),
        "ram_used_mib": round(memory.used / 1024 / 1024, 1),
        "ram_total_mib": round(memory.total / 1024 / 1024, 1),
    }
    counters = psutil.disk_io_counters()
    if counters is not None:
        payload["disk_read_mib"] = round(counters.read_bytes / 1024 / 1024, 1)
        payload["disk_write_mib"] = round(counters.write_bytes / 1024 / 1024, 1)
    return payload


class ResourceMonitor:
    """Фоновый сбор метрик с разметкой по стадиям."""

    def __init__(
        self,
        output_dir: Path,
        interval_seconds: float = 2.0,
        enabled: bool = True,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.interval = max(0.5, float(interval_seconds))
        self.enabled = bool(enabled)
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._stage = ""
        self._document = ""
        self._samples_path = self.output_dir / "samples.jsonl"
        self._stages_path = self.output_dir / "stages.jsonl"
        self._started_at: float | None = None

    # ------------------------------------------------------------ жизненный цикл

    def start(self) -> ResourceMonitor:
        if not self.enabled or self._thread is not None:
            return self
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._loop, name="resource-monitor", daemon=True)
        self._thread.start()
        logger.info(
            "Мониторинг ресурсов запущен: %s (интервал %.1f с)", self.output_dir, self.interval
        )
        return self

    def stop(self) -> None:
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=self.interval * 3)
        self._thread = None
        logger.info("Мониторинг ресурсов остановлен")

    def __enter__(self) -> ResourceMonitor:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # ---------------------------------------------------------------- стадии

    @contextmanager
    def stage(self, name: str, document: str = "") -> Iterator[None]:
        """Отмечает границы стадии.

        Всё, что происходит внутри блока, попадает в замеры с этой меткой,
        поэтому потом видно, чем именно была занята машина на каждом этапе.
        """
        started = time.time()
        with self._lock:
            previous_stage, previous_document = self._stage, self._document
            self._stage, self._document = name, document
        # Замер сразу на входе: короткие стадии иначе не попадут в выборку.
        self._write_sample(self._collect())
        try:
            yield
        finally:
            finished = time.time()
            self._write_sample(self._collect())
            with self._lock:
                self._stage, self._document = previous_stage, previous_document
            self._append(
                self._stages_path,
                {
                    "stage": name,
                    "document": document,
                    "started_at": started,
                    "finished_at": finished,
                    "duration_seconds": round(finished - started, 3),
                    "started_iso": datetime.fromtimestamp(started, UTC).isoformat(),
                },
            )

    # ----------------------------------------------------------------- сбор

    def _collect(self) -> ResourceSample:
        with self._lock:
            stage, document = self._stage, self._document
        payload: dict[str, Any] = {"timestamp": time.time(), "stage": stage, "document": document}
        payload.update(_query_gpu())
        payload.update(_query_host())
        known = set(ResourceSample.__dataclass_fields__)
        return ResourceSample(**{k: v for k, v in payload.items() if k in known})

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self._write_sample(self._collect())
            except Exception as exc:  # noqa: BLE001
                # Мониторинг не имеет права ронять прогон, который он наблюдает.
                logger.debug("Замер не удался: %s", exc)
            self._stop.wait(self.interval)

    def _write_sample(self, sample: ResourceSample) -> None:
        self._append(self._samples_path, sample.as_dict())

    @staticmethod
    def _append(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- сервис

    @property
    def samples_path(self) -> Path:
        return self._samples_path

    @property
    def stages_path(self) -> Path:
        return self._stages_path


class NullMonitor:
    """Заглушка для случаев, когда мониторинг выключен."""

    enabled = False

    def start(self) -> NullMonitor:
        return self

    def stop(self) -> None:
        return None

    def __enter__(self) -> NullMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        return None

    @contextmanager
    def stage(self, name: str, document: str = "") -> Iterator[None]:
        yield
