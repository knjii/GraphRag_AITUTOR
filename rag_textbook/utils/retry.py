"""Единая политика повторов.

Прежде повторы были разбросаны по трём модулям в виде ``for attempt in range(2)``
с фиксированным ``time.sleep(0.8)``: разное поведение в разных местах и
невозможность настроить. Здесь одна реализация с экспоненциальной задержкой
и джиттером, синхронная и асинхронная.
"""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

from rag_textbook.logging_setup import get_logger

logger = get_logger("retry")

T = TypeVar("T")

# Ошибки, которые имеет смысл повторять: перегрузка сервера инференса,
# временная недоступность, разрыв соединения.
RETRYABLE_MARKERS: tuple[str, ...] = (
    "503",
    "502",
    "504",
    "429",
    "timeout",
    "timed out",
    "connection reset",
    "connection refused",
    "temporarily unavailable",
    "service unavailable",
    "ggml_assert",
)


def is_retryable(exc: BaseException, extra_markers: Iterable[str] = ()) -> bool:
    text = str(exc).lower()
    markers = (*RETRYABLE_MARKERS, *tuple(extra_markers))
    return any(marker in text for marker in markers)


def _delay(attempt: int, base: float, cap: float) -> float:
    raw = min(cap, base * (2**attempt))
    # Джиттер разводит одновременные повторы нескольких воркеров.
    return raw * (0.5 + random.random() * 0.5)


def retry_sync(
    func: Callable[[], T],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 20.0,
    description: str = "operation",
    on_retry: Callable[[int, BaseException], None] | None = None,
) -> T:
    last: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return func()
        except Exception as exc:  # noqa: BLE001 - решение о повторе принимаем ниже
            last = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            wait = _delay(attempt, base_delay, max_delay)
            logger.warning(
                "%s: попытка %s/%s не удалась (%s). Повтор через %.1f с",
                description,
                attempt + 1,
                attempts,
                exc,
                wait,
            )
            if on_retry is not None:
                on_retry(attempt, exc)
            time.sleep(wait)
    assert last is not None
    raise last


async def retry_async(
    func: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.8,
    max_delay: float = 20.0,
    description: str = "operation",
) -> T:
    last: BaseException | None = None
    for attempt in range(max(1, attempts)):
        try:
            return await func()
        except Exception as exc:  # noqa: BLE001
            last = exc
            if attempt >= attempts - 1 or not is_retryable(exc):
                raise
            wait = _delay(attempt, base_delay, max_delay)
            logger.warning(
                "%s: попытка %s/%s не удалась (%s). Повтор через %.1f с",
                description,
                attempt + 1,
                attempts,
                exc,
                wait,
            )
            await asyncio.sleep(wait)
    assert last is not None
    raise last
