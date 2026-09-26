"""Журнал ответов модели: прерванная сборка продолжается с места обрыва.

Сборка эталона v2 — тысячи обращений к модели и полтора часа карты, а
сама сборка с середины не возобновляется. 2026-09-22 её пришлось прервать
дважды, и каждый раз сделанное пропадало целиком. Журнал пишет каждый
ответ строкой JSONL сразу по получении; при повторном запуске тот же
запрос берётся из журнала, а не у модели.

Ключ — хэш всего, от чего зависит ответ: модель, сообщения, назначение,
схема, пределы и температура. Другой промпт или другая модель дают другой
ключ, поэтому старый журнал не подменит новые ответы. Ошибки не пишутся:
упавший запрос при повторе уйдёт к модели заново.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.logging_setup import get_logger

logger = get_logger("clients.llm_journal")


class JournaledLLM:
    """Обёртка над ``LLMClient``: ответы из журнала, новые — в журнал."""

    def __init__(self, inner: LLMClient, path: Path, model: str = "") -> None:
        self.inner = inner
        self.path = Path(path)
        self.model = model
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()
        self._answers: dict[str, str] = {}
        if self.path.is_file():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    # Последняя строка могла оборваться вместе с процессом.
                    continue
                self._answers[record["key"]] = record["answer"]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")
        logger.info("Журнал ответов %s: %s записей", self.path, len(self._answers))

    def key(self, messages: Sequence[ChatMessage], **params: Any) -> str:
        payload = {
            "model": self.model,
            "messages": [[m.role, m.content] for m in messages],
            **params,
        }
        blob = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        key = self.key(messages, purpose=purpose, json_schema=json_schema,
                       max_tokens=max_tokens, temperature=temperature)
        with self._lock:
            if key in self._answers:
                self.hits += 1
                return self._answers[key]
        answer = self.inner.chat(messages, purpose=purpose, json_schema=json_schema,
                                 max_tokens=max_tokens, temperature=temperature)
        with self._lock:
            self.misses += 1
            if key not in self._answers:
                self._answers[key] = answer
                self._file.write(json.dumps({"key": key, "answer": answer}, ensure_ascii=False) + "\n")
                self._file.flush()
        return answer

    def close(self) -> None:
        with self._lock:
            if not self._file.closed:
                self._file.close()
        close = getattr(self.inner, "close", None)
        if close is not None:
            close()
