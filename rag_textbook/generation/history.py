"""Хранилище истории диалога.

Прежняя реализация писала JSONL-файл на сессию: полное чтение файла на каждый ход,
никаких блокировок при конкурентной записи и никакой привязки к пользователю —
идентификатор сессии передавался аргументом, так что чужую переписку можно было
прочитать, просто угадав имя.

Здесь SQLite: атомарные записи, блокировка на уровне БД, обязательный ``user_id``
и выборка последних N ходов запросом, а не чтением всего файла.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from rag_textbook.clients.llm import ChatMessage
from rag_textbook.logging_setup import get_logger

logger = get_logger("generation.history")


class ChatHistoryStore:
    def __init__(self, db_path: Path, enabled: bool = True) -> None:
        self.enabled = bool(enabled)
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._conn: sqlite3.Connection | None = None
        if self.enabled:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._connect()

    def _connect(self) -> None:
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute(
            """
            CREATE TABLE IF NOT EXISTS turns (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id    TEXT NOT NULL,
                session_id TEXT NOT NULL,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        # Индекс под основной запрос: последние ходы конкретной сессии пользователя.
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS turns_lookup ON turns (user_id, session_id, id DESC)"
        )
        self._conn.commit()

    def append(self, user_id: str, session_id: str, role: str, content: str) -> None:
        if not self.enabled or self._conn is None:
            return
        with self._lock:
            self._conn.execute(
                "INSERT INTO turns (user_id, session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    user_id,
                    session_id,
                    role,
                    content,
                    datetime.now(UTC).isoformat(),
                ),
            )
            self._conn.commit()

    def recent(self, user_id: str, session_id: str, max_turns: int) -> list[ChatMessage]:
        """Последние ходы диалога.

        Владелец проверяется в самом запросе: без этого сессию можно было бы
        прочитать, просто зная её идентификатор.
        """
        if not self.enabled or self._conn is None or max_turns <= 0:
            return []
        limit = max_turns * 2
        with self._lock:
            rows = self._conn.execute(
                "SELECT role, content FROM turns "
                "WHERE user_id = ? AND session_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, session_id, limit),
            ).fetchall()
        messages = [
            ChatMessage(role=("user" if role == "user" else "assistant"), content=content)
            for role, content in reversed(rows)
        ]
        return messages

    def clear(self, user_id: str, session_id: str) -> int:
        if not self.enabled or self._conn is None:
            return 0
        with self._lock:
            cursor = self._conn.execute(
                "DELETE FROM turns WHERE user_id = ? AND session_id = ?",
                (user_id, session_id),
            )
            self._conn.commit()
            return int(cursor.rowcount or 0)

    def sessions(self, user_id: str) -> list[str]:
        if not self.enabled or self._conn is None:
            return []
        with self._lock:
            rows = self._conn.execute(
                "SELECT DISTINCT session_id FROM turns WHERE user_id = ? ORDER BY session_id",
                (user_id,),
            ).fetchall()
        return [str(row[0]) for row in rows]

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None


def to_chat_messages(pairs: Sequence[tuple[str, str]]) -> list[ChatMessage]:
    return [
        ChatMessage(role=("user" if role == "user" else "assistant"), content=content)
        for role, content in pairs
    ]
