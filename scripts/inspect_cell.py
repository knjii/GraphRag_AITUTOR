"""Что за файл ячейки: сколько ответов, чем сделан, не мусор ли.

Понадобился после случая 2026-09-04: остановленный замер оставил после себя
дочерний процесс, тот дописал файл `answers_model-muse30b-w16384.json`
уже во время пробы движка, когда контейнеры менялись под ним. Файл выглядел
законным, а содержал ответы вперемешку от разных моделей.

    python scripts/inspect_cell.py artifacts/metrics/answers_model-*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def main() -> int:
    for name in sys.argv[1:]:
        path = Path(name)
        data = json.loads(path.read_text(encoding="utf-8"))
        outcomes = data.get("outcomes", [])
        empty = sum(1 for item in outcomes if not item.get("answer", "").strip())
        made = data.get("summary", {}).get("чем сделано", {})
        print(f"{path.name}")
        print(f"   ответов {len(outcomes)}, пустых {empty}")
        print(f"   модель ответа: {made.get('модель ответа', '?')}")
        print(f"   по настройке:  {made.get('модель по настройке', '—')}")
        print(f"   промпт: {made.get('промпт', '?')}, окно: {made.get('окно контекста', '?')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
