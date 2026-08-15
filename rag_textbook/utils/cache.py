"""Кэш результатов дорогих стадий, адресуемый по содержимому.

Зачем: в прежнем пайплайне повторный прогон заново вызывал модель зрения на каждой
картинке, заново эмбеддил уже посчитанные чанки и заново дёргал LLM-экстрактор.
При стоимости индексации в часы это главный источник потерь.

Ключ — sha256 от входа, поэтому кэш автоматически инвалидируется при смене
текста, модели или версии промпта: они входят в ключ.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from rag_textbook.logging_setup import get_logger

logger = get_logger("cache")


class ArtifactCache:
    """Потокобезопасный кэш «ключ → JSON» на SQLite.

    SQLite выбран сознательно: кэш переживает перезапуск, не требует отдельного
    сервиса и переносится вместе с диском арендованной машины одним файлом.
    """

    def __init__(self, path: Path, namespace: str, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.namespace = namespace
        self.path = Path(path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        # WAL позволяет читать во время записи — важно, когда стадии идут параллельно.
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS cache_entries (
                namespace TEXT NOT NULL,
                key       TEXT NOT NULL,
                value     TEXT NOT NULL,
                created_at REAL NOT NULL DEFAULT (strftime('%s','now')),
                PRIMARY KEY (namespace, key)
            )
            """
        )
        self._conn.commit()

    def get(self, key: str) -> Any | None:
        if not self.enabled or self._conn is None:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT value FROM cache_entries WHERE namespace = ? AND key = ?",
                (self.namespace, key),
            ).fetchone()
        if not row:
            return None
        try:
            return json.loads(row[0])
        except json.JSONDecodeError:
            logger.warning("Повреждённая запись кэша %s/%s, игнорирую", self.namespace, key)
            return None

    def get_many(self, keys: Iterable[str]) -> dict[str, Any]:
        keys = list(keys)
        if not self.enabled or self._conn is None or not keys:
            return {}
        found: dict[str, Any] = {}
        # SQLite ограничивает число параметров, поэтому читаем частями.
        chunk_size = 500
        with self._lock:
            for start in range(0, len(keys), chunk_size):
                batch = keys[start : start + chunk_size]
                placeholders = ",".join("?" * len(batch))
                rows = self._conn.execute(
                    f"SELECT key, value FROM cache_entries "
                    f"WHERE namespace = ? AND key IN ({placeholders})",
                    (self.namespace, *batch),
                ).fetchall()
                for key, value in rows:
                    try:
                        found[key] = json.loads(value)
                    except json.JSONDecodeError:
                        continue
        return found

    def set(self, key: str, value: Any) -> None:
        if not self.enabled or self._conn is None:
            return
        payload = json.dumps(value, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO cache_entries (namespace, key, value) VALUES (?, ?, ?)",
                (self.namespace, key, payload),
            )
            self._conn.commit()

    def set_many(self, items: dict[str, Any]) -> None:
        if not self.enabled or self._conn is None or not items:
            return
        rows = [
            (self.namespace, key, json.dumps(value, ensure_ascii=False))
            for key, value in items.items()
        ]
        with self._lock:
            self._conn.executemany(
                "INSERT OR REPLACE INTO cache_entries (namespace, key, value) VALUES (?, ?, ?)",
                rows,
            )
            self._conn.commit()

    def stats(self) -> dict[str, int]:
        if not self.enabled or self._conn is None:
            return {"entries": 0}
        with self._lock:
            row = self._conn.execute(
                "SELECT count(*) FROM cache_entries WHERE namespace = ?", (self.namespace,)
            ).fetchone()
        return {"entries": int(row[0]) if row else 0}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None

    def __enter__(self) -> ArtifactCache:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
