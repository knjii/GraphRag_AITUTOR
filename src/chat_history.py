from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from settings import Settings


def _normalize_session_id(session_id: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9._-]+", "_", str(session_id or "").strip())
    return value or "default"


def _history_path(settings: Settings, session_id: str) -> Path:
    history_dir = Path(settings.chat_history_dir)
    if not history_dir.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        history_dir = project_root / history_dir
    history_dir.mkdir(parents=True, exist_ok=True)
    return history_dir / f"{_normalize_session_id(session_id)}.jsonl"


def load_chat_turns(settings: Settings, session_id: str) -> List[Dict[str, str]]:
    path = _history_path(settings, session_id)
    if not path.exists():
        return []

    turns: List[Dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        payload = line.strip()
        if not payload:
            continue
        try:
            row = json.loads(payload)
        except json.JSONDecodeError:
            continue
        user_text = str(row.get("user") or "").strip()
        assistant_text = str(row.get("assistant") or "").strip()
        if user_text or assistant_text:
            turns.append({"user": user_text, "assistant": assistant_text})
    return turns


def build_chat_history_messages(
    settings: Settings,
    session_id: str,
    max_turns: int | None = None,
) -> List[BaseMessage]:
    turns = load_chat_turns(settings, session_id)
    limit = settings.chat_history_max_turns if max_turns is None else int(max_turns)
    if limit > 0:
        turns = turns[-limit:]
    else:
        turns = []

    messages: List[BaseMessage] = []
    for turn in turns:
        user_text = str(turn.get("user") or "").strip()
        assistant_text = str(turn.get("assistant") or "").strip()
        if user_text:
            messages.append(HumanMessage(content=user_text))
        if assistant_text:
            messages.append(AIMessage(content=assistant_text))
    return messages


def append_chat_turn(
    settings: Settings,
    session_id: str,
    user_query: str,
    assistant_answer: str,
    metadata: Dict[str, Any] | None = None,
) -> None:
    path = _history_path(settings, session_id)
    row: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "user": str(user_query or "").strip(),
        "assistant": str(assistant_answer or "").strip(),
    }
    if metadata:
        row["metadata"] = metadata

    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def clear_chat_history(settings: Settings, session_id: str) -> None:
    path = _history_path(settings, session_id)
    if path.exists():
        path.unlink()
