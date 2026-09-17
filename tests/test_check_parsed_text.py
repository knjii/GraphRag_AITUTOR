"""Проверка разобранных книг на мусор кодировки."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

_spec = importlib.util.spec_from_file_location(
    "check_parsed_text", Path(__file__).resolve().parents[1] / "scripts" / "check_parsed_text.py"
)
assert _spec is not None and _spec.loader is not None
check = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(check)


def _book(root: Path, name: str, text: str) -> None:
    folder = root / f"{name}__abc"
    folder.mkdir(parents=True)
    (folder / "blocks.json").write_text(
        json.dumps({"blocks": [{"type": "text", "text": text}]}, ensure_ascii=False), encoding="utf-8"
    )


def test_shifted_cyrillic_is_flagged(tmp_path):
    # Так Гельфанд извлекается из текстового слоя PDF.
    _book(tmp_path, "ru-la-gelfand", "¥¢ª«¨¤®¢® ¯°®±²° ­±²¢® (x; y) " * 20)
    _book(tmp_path, "ru-prob-chernova", "Случайная величина $\\xi$ имеет распределение Пуассона. " * 20)
    _book(tmp_path, "en-la-axler", "A vector space $V$ over $\\mathbf F$ is a set. " * 20)
    rows = {name: problem for name, _, _, problem in check.collect(tmp_path)}
    assert rows == {"en-la-axler": "", "ru-la-gelfand": "мусор кодировки", "ru-prob-chernova": ""}


def test_russian_book_without_cyrillic_is_flagged(tmp_path):
    _book(tmp_path, "ru-rl-ivanov", "Reinforcement learning text without Russian letters. " * 10)
    [(_, _, _, problem)] = check.collect(tmp_path)
    assert problem == "нет кириллицы"


def test_non_library_documents_are_ignored(tmp_path):
    _book(tmp_path, "Dayzenrot_Feyzal", "текст")
    assert check.collect(tmp_path) == []
