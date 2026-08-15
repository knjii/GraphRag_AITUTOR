"""Управление памятью на стадиях индексации.

Задача модуля — сделать повышение утилизации безопасным. Наращивать батчи и
параллелизм имеет смысл только если есть страховка: иначе однажды на длинном
документе прогон упадёт с нехваткой видеопамяти, и часы работы пропадут.

Две страховки:

* **упреждающая** — перед стадией проверяем свободную память и уменьшаем размер
  батча, если запаса не хватает; лучше отработать медленнее, чем упасть;
* **реактивная** — при возникновении ошибки нехватки памяти батч делится пополам
  и повторяется, а размер батча для последующих вызовов снижается.

Обе деградируют мягко: без ``nvidia-smi`` защита превращается в пустую операцию,
и конвейер работает как раньше.
"""

from __future__ import annotations

import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from typing import TypeVar

from rag_textbook.logging_setup import get_logger

logger = get_logger("indexing.resources")

T = TypeVar("T")
R = TypeVar("R")

# Маркеры нехватки памяти в сообщениях разных библиотек и серверов.
_OOM_MARKERS: tuple[str, ...] = (
    "out of memory",
    "cuda oom",
    "cublas_status_alloc_failed",
    "no kv cache space",
    "insufficient memory",
    "failed to allocate",
    "resource exhausted",
)


def is_out_of_memory(exc: BaseException) -> bool:
    text = str(exc).lower()
    return any(marker in text for marker in _OOM_MARKERS)


def query_vram_mib() -> tuple[int, int] | None:
    """Свободная и общая видеопамять в МиБ.

    Спрашиваем у ``nvidia-smi``, а не у torch: часть памяти держат посторонние
    процессы (Infinity, Ollama, контейнеры), и torch о них не знает.
    """
    try:
        raw = subprocess.run(  # noqa: S603
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    line = raw.splitlines()[0] if raw else ""
    free, _, total = line.partition(",")
    try:
        return int(free.strip()), int(total.strip())
    except ValueError:
        return None


class VramGuard:
    """Страховка от нехватки видеопамяти.

    Хранит подобранные размеры батчей по стадиям, поэтому один эпизод нехватки
    памяти снижает нагрузку на всю оставшуюся часть прогона, а не повторяется
    на каждом документе.
    """

    def __init__(
        self,
        *,
        min_free_mib: int = 1536,
        poll_seconds: float = 2.0,
        max_wait_seconds: float = 120.0,
        enabled: bool = True,
    ) -> None:
        self.min_free_mib = max(0, int(min_free_mib))
        self.poll_seconds = max(0.5, float(poll_seconds))
        self.max_wait_seconds = max(0.0, float(max_wait_seconds))
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._scale: dict[str, float] = {}
        self._oom_events = 0

    # ------------------------------------------------------------------ замеры

    def snapshot(self) -> tuple[int, int] | None:
        if not self.enabled:
            return None
        return query_vram_mib()

    @property
    def oom_events(self) -> int:
        return self._oom_events

    # ------------------------------------------------------------- ожидание

    def wait_for_headroom(self, required_mib: int, stage: str = "") -> bool:
        """Ждёт освобождения памяти перед тяжёлой стадией.

        Нужно потому, что соседние сервисы (Infinity, Ollama) освобождают память
        не мгновенно. Прежняя версия проекта решала это безусловными паузами
        на минуты; здесь выход происходит по факту, а таймер — только верхняя граница.
        """
        if not self.enabled or required_mib <= 0:
            return True

        deadline = time.monotonic() + self.max_wait_seconds
        first = True
        while True:
            vram = query_vram_mib()
            if vram is None:
                return True
            free, total = vram
            if free >= required_mib:
                if not first:
                    logger.info("Память освободилась: %s МиБ свободно (%s)", free, stage or "—")
                return True
            if required_mib > total:
                logger.warning(
                    "Стадии %s требуется %s МиБ, а на карте всего %s МиБ — ждать бессмысленно",
                    stage or "—",
                    required_mib,
                    total,
                )
                return False
            if time.monotonic() >= deadline:
                logger.warning(
                    "Не дождались памяти для %s: свободно %s из требуемых %s МиБ. Продолжаю осторожно.",
                    stage or "—",
                    free,
                    required_mib,
                )
                return False
            if first:
                logger.info(
                    "Жду освобождения памяти для %s: свободно %s, нужно %s МиБ",
                    stage or "—",
                    free,
                    required_mib,
                )
                first = False
            time.sleep(self.poll_seconds)

    # ---------------------------------------------------------------- батчи

    def batch_size(self, stage: str, requested: int, per_item_mib: float = 0.0) -> int:
        """Размер батча с учётом свободной памяти и прошлых сбоев."""
        requested = max(1, int(requested))
        with self._lock:
            scale = self._scale.get(stage, 1.0)
        size = max(1, int(requested * scale))

        if not self.enabled or per_item_mib <= 0:
            return size

        vram = query_vram_mib()
        if vram is None:
            return size
        free, _ = vram
        usable = max(0, free - self.min_free_mib)
        affordable = int(usable / per_item_mib) if per_item_mib > 0 else size
        if affordable < size:
            adjusted = max(1, affordable)
            logger.info(
                "Стадия %s: батч уменьшен с %s до %s (свободно %s МиБ)",
                stage,
                size,
                adjusted,
                free,
            )
            return adjusted
        return size

    def record_oom(self, stage: str) -> float:
        """Запоминает эпизод нехватки памяти и снижает нагрузку стадии."""
        with self._lock:
            self._oom_events += 1
            scale = max(0.125, self._scale.get(stage, 1.0) * 0.5)
            self._scale[stage] = scale
        logger.warning(
            "Нехватка видеопамяти на стадии %s. Нагрузка стадии снижена до %.0f%% от запрошенной.",
            stage,
            scale * 100,
        )
        return scale

    @contextmanager
    def guarded(self, stage: str, required_mib: int = 0) -> Iterator[None]:
        """Контекст стадии: ожидание памяти на входе, диагностика на выходе."""
        if required_mib > 0:
            self.wait_for_headroom(required_mib, stage)
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            if is_out_of_memory(exc):
                self.record_oom(stage)
            raise


def run_with_oom_backoff(
    items: Sequence[T],
    handler: Callable[[Sequence[T]], R],
    *,
    guard: VramGuard,
    stage: str,
    batch_size: int,
    min_batch_size: int = 1,
) -> list[R]:
    """Обрабатывает элементы батчами, деля батч пополам при нехватке памяти.

    Именно это делает наращивание батчей безопасным: при ошибке памяти мы не
    теряем прогон, а автоматически переходим к меньшему батчу и продолжаем.
    """
    results: list[R] = []
    if not items:
        return results

    pending: list[Sequence[T]] = []
    step = max(1, int(batch_size))
    for start in range(0, len(items), step):
        pending.append(items[start : start + step])

    while pending:
        batch = pending.pop(0)
        try:
            results.append(handler(batch))
        except Exception as exc:  # noqa: BLE001
            if not is_out_of_memory(exc) or len(batch) <= max(1, min_batch_size):
                raise
            guard.record_oom(stage)
            middle = len(batch) // 2
            # Половинки возвращаются в начало очереди, порядок обработки сохраняется.
            pending.insert(0, batch[middle:])
            pending.insert(0, batch[:middle])
            logger.warning(
                "Стадия %s: батч из %s элементов разделён на %s и %s после нехватки памяти",
                stage,
                len(batch),
                middle,
                len(batch) - middle,
            )
    return results
