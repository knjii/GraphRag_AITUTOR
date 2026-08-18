"""Журнал отказов извлечения.

Зачем отдельная сущность, а не запись в кэш. Результат отката к правилам
намеренно не кэшируется: иначе разовый сбой модели стал бы постоянным —
следующий прогон взял бы из кэша пустой граф и даже не попробовал ещё раз.
Решение верное, но у него есть цена, которую мы обнаружили при разборе
37 фрагментов, падающих всегда: **причина отказа не сохраняется нигде**.
Диагностика упёрлась в то, что отказавшие фрагменты в кэше просто
отсутствуют, а чем именно ответила модель — не знает никто.

Журнал закрывает разрыв, не трогая кэш: он пишется в стороне, читается только
человеком и на поведение конвейера не влияет. Ответ модели сохраняется
обрезанным — целиком он не нужен, а размер журнала при 1151 фрагменте
имеет значение.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol

from rag_textbook.logging_setup import get_logger

logger = get_logger("graph.failure_journal")

# Сколько символов ответа модели сохранять. Хватает, чтобы отличить пустой
# ответ от оборванного JSON и от связного текста вместо разметки.
PREVIEW_CHARS = 600


class FailureJournal(Protocol):
    """Куда извлекатель сообщает об отказах."""

    def record(self, entry: dict[str, Any]) -> None: ...


class NullJournal:
    """Ничего не пишет. Значение по умолчанию, чтобы вызов был безусловным."""

    def record(self, entry: dict[str, Any]) -> None:  # noqa: D102
        return


class JsonlFailureJournal:
    """Пишет по записи на строку, дописывая в конец файла.

    Формат построчный намеренно: прогон может прерваться, и частично
    записанный журнал должен оставаться читаемым. JSON-массив такого
    не позволяет.
    """

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._prepared = False

    def _prepare(self) -> None:
        if not self._prepared:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._prepared = True

    def record(self, entry: dict[str, Any]) -> None:
        try:
            self._prepare()
            with self.path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except OSError as error:
            # Журнал — вспомогательный: сбой записи не должен ронять индексацию.
            logger.warning("Не удалось записать журнал отказов: %s", error)


def build_entry(
    *,
    chunk_id: str,
    status: str,
    attempts: int,
    raw_preview: str,
    text: str,
    pages: Any = None,
    headers: Any = None,
) -> dict[str, Any]:
    """Собирает запись журнала.

    Признаки фрагмента складываются сюда же: разбор отказов начинается
    со сравнения отказавших фрагментов с удачными, и держать их в одном месте
    дешевле, чем каждый раз сшивать журнал с корпусом.
    """
    return {
        "chunk_id": chunk_id,
        "status": status,
        "attempts": attempts,
        "raw_preview": (raw_preview or "")[:PREVIEW_CHARS],
        "text_length": len(text or ""),
        "text_preview": (text or "")[:200],
        "pages": list(pages) if pages else [],
        "header": (list(headers)[0] if headers else ""),
    }
