"""Что за файл ячейки: сколько ответов, чем сделан, не мусор ли.

Понадобился после случая 2026-09-04: остановленный замер оставил после себя
дочерний процесс, тот дописал файл `answers_model-muse30b-w16384.json`
уже во время пробы движка, когда контейнеры менялись под ним. Файл выглядел
законным, а содержал ответы вперемешку от разных моделей.

    python scripts/inspect_cell.py artifacts/metrics/answers_model-*.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(path: Path, *, expected: int | None, max_empty_share: float) -> list[str]:
    print(f"{path.name}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"не удалось прочитать JSON: {exc}"]

    if not isinstance(data, dict):
        return ["неверная структура файла: корень должен быть объектом"]
    outcomes = data.get("outcomes", [])
    if not isinstance(outcomes, list):
        return ["неверная структура файла: outcomes должен быть списком"]
    for index, item in enumerate(outcomes):
        if not isinstance(item, dict):
            return [f"неверная структура файла: outcomes[{index}] должен быть объектом"]
    summary = data.get("summary", {})
    if not isinstance(summary, dict):
        return ["неверная структура файла: summary должен быть объектом"]
    made = summary.get("чем сделано", {})
    if not isinstance(made, dict):
        return ["неверная структура файла: summary / чем сделано должен быть объектом"]

    empty = sum(1 for item in outcomes if not item.get("answer", "").strip())
    print(f"   ответов {len(outcomes)}, пустых {empty}")
    print(f"   модель ответа: {made.get('модель ответа', '?')}")
    print(f"   по настройке:  {made.get('модель по настройке', '—')}")
    print(f"   промпт: {made.get('промпт', '?')}, окно: {made.get('окно контекста', '?')}")

    problems = []
    if not outcomes:
        problems.append("нет ни одного ответа")
    if expected is not None and len(outcomes) != expected:
        problems.append(f"число ответов {len(outcomes)} не равно ожидаемому {expected}")
    if outcomes and empty / len(outcomes) > max_empty_share:
        problems.append(f"доля пустых ответов {empty / len(outcomes):.2%} больше {max_empty_share:.2%}")
    if not (made.get("модель ответа") or "").strip():
        problems.append("в метаданных нет модели ответа")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("files", nargs="+", type=Path, help="файлы ячеек замера")
    parser.add_argument("--expected", type=int, help="ожидаемое число ответов")
    parser.add_argument(
        "--max-empty-share", type=float, default=0.05,
        help="допустимая доля пустых ответов (по умолчанию 0.05)",
    )
    args = parser.parse_args(argv)
    failed = False
    for path in args.files:
        problems = inspect(path, expected=args.expected, max_empty_share=args.max_empty_share)
        if problems:
            print(f"   НЕГОДЕН: {'; '.join(problems)}")
            failed = True
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
