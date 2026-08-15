"""Манифест индексации: состояние документов между запусками.

Прежний чекпоинт умел только «файл обработан / не обработан», был выключен по
умолчанию и не отличал изменившийся документ от неизменившегося. Здесь состояние
хранится по стадиям и по отпечатку файла, поэтому:

* повторный запуск не переделывает уже сделанное;
* изменённый PDF переиндексируется, а неизменённый — нет;
* прогон, упавший на стадии графа, при перезапуске продолжается с неё,
  а не с разбора PDF.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag_textbook.logging_setup import get_logger

logger = get_logger("indexing.manifest")

STAGES: tuple[str, ...] = ("parsed", "chunked", "embedded", "graphed")


def _now() -> str:
    return datetime.now(UTC).isoformat()


@dataclass
class DocumentState:
    doc_id: str
    source_path: str
    fingerprint: str
    stages: dict[str, str] = field(default_factory=dict)
    chunks: int = 0
    error: str = ""
    updated_at: str = field(default_factory=_now)

    def is_done(self, stage: str) -> bool:
        return self.stages.get(stage) == "done"

    def mark(self, stage: str, status: str = "done", error: str = "") -> None:
        self.stages[stage] = status
        self.error = error
        self.updated_at = _now()


class IndexingManifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.documents: dict[str, DocumentState] = {}
        self.load()

    def load(self) -> None:
        if not self.path.is_file():
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            logger.warning("Манифест повреждён, начинаю с чистого состояния: %s", self.path)
            return
        for doc_id, raw in (payload.get("documents") or {}).items():
            self.documents[doc_id] = DocumentState(
                doc_id=doc_id,
                source_path=str(raw.get("source_path", "")),
                fingerprint=str(raw.get("fingerprint", "")),
                stages=dict(raw.get("stages") or {}),
                chunks=int(raw.get("chunks") or 0),
                error=str(raw.get("error") or ""),
                updated_at=str(raw.get("updated_at") or _now()),
            )

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 2,
            "updated_at": _now(),
            "documents": {doc_id: asdict(state) for doc_id, state in self.documents.items()},
        }
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def get(self, doc_id: str, source_path: str, fingerprint: str) -> DocumentState:
        state = self.documents.get(doc_id)
        if state is None:
            state = DocumentState(doc_id=doc_id, source_path=source_path, fingerprint=fingerprint)
            self.documents[doc_id] = state
            return state

        if state.fingerprint != fingerprint:
            # Файл изменился — весь прогресс по нему недействителен.
            logger.info("Документ изменился, сбрасываю стадии: %s", source_path)
            state.fingerprint = fingerprint
            state.stages = {}
            state.chunks = 0
            state.error = ""
        return state

    def summary(self) -> dict[str, Any]:
        totals = {stage: 0 for stage in STAGES}
        for state in self.documents.values():
            for stage in STAGES:
                if state.is_done(stage):
                    totals[stage] += 1
        return {
            "documents": len(self.documents),
            "stages_done": totals,
            "failed": sum(1 for state in self.documents.values() if state.error),
        }
