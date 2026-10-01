"""Проверка разобранных книг: живой ли текст, а не мусор кодировки.

У Гельфанда текстовый слой PDF есть, но кириллица в нём лежит в старой
8-битной кодировке со сдвигом: «Евклидово» извлекается как «¥¢ª«¨¤®¢®».
MinerU в режиме ``auto`` может взять такой слой — и в индекс молча
попадёт мусор, который никто не найдёт поиском. Проверка запускается
сразу после стадии разбора, до векторизации.

    python scripts/check_parsed_text.py --parsed artifacts/parsed

Язык книги берётся из префикса имени файла (``ru-``/``en-``, как пишет
``scripts/fetch_library.py``). Код выхода 1 — есть подозрительные книги.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.rewards.formula import strip_math  # noqa: E402

_CYRILLIC = re.compile(r"[а-яА-ЯёЁ]")
_LATIN = re.compile(r"[a-zA-Z]")
# Латиница-1 «¡…ÿ» — след сдвинутой кириллицы; в нормальном тексте почти не бывает.
_MOJIBAKE = re.compile(r"[¡-ÿ]")

MIN_CYRILLIC_SHARE = 0.5
MAX_MOJIBAKE_SHARE = 0.02


def text_stats(text: str) -> dict[str, float]:
    prose = strip_math(text)
    cyrillic = len(_CYRILLIC.findall(prose))
    latin = len(_LATIN.findall(prose))
    letters = cyrillic + latin
    visible = sum(1 for c in prose if not c.isspace())
    return {
        "cyrillic_share": cyrillic / letters if letters else 0.0,
        "mojibake_share": len(_MOJIBAKE.findall(prose)) / visible if visible else 0.0,
        "chars": visible,
    }


def language_of(name: str) -> str | None:
    for prefix in ("ru-", "en-"):
        if name.startswith(prefix):
            return prefix[:2]
    return None


def verdict(lang: str | None, stats: dict[str, float]) -> str:
    if stats["chars"] == 0:
        return "пусто"
    if stats["mojibake_share"] > MAX_MOJIBAKE_SHARE:
        return "мусор кодировки"
    if lang == "ru" and stats["cyrillic_share"] < MIN_CYRILLIC_SHARE:
        return "нет кириллицы"
    return ""


def collect(parsed_dir: Path) -> list[tuple[str, str | None, dict[str, float], str]]:
    rows = []
    for blocks_path in sorted(parsed_dir.glob("*/blocks.json")):
        name = blocks_path.parent.name.split("__", 1)[0]
        lang = language_of(name)
        if lang is None:
            continue
        payload = json.loads(blocks_path.read_text(encoding="utf-8"))
        blocks = payload["blocks"] if isinstance(payload, dict) else payload
        text = " ".join(
            str(block.get("text") or "") for block in blocks if block.get("type") in {"text", "title"}
        )
        stats = text_stats(text)
        rows.append((name, lang, stats, verdict(lang, stats)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", type=Path, default=ROOT / "artifacts" / "parsed")
    args = parser.parse_args()
    rows = collect(args.parsed)
    if not rows:
        print(f"В {args.parsed} нет разобранных книг библиотеки (имена ru-*/en-*)")
        return 1
    bad = 0
    for name, lang, stats, problem in rows:
        bad += bool(problem)
        print(f"{name:44s} {lang}  кириллица {stats['cyrillic_share']:.2f}  "
              f"мусор {stats['mojibake_share']:.3f}  знаков {int(stats['chars']):8d}  {problem or 'ок'}")
    if bad:
        print(f"\nПодозрительных книг: {bad}. Русские книги с мусором — переразобрать с MINERU_METHOD=ocr.")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
