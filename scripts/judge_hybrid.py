"""Гибридный судья (M6, редакция r1): балл v1, поправленный атомарными проверками v2.

Судья v1 ранжирует ответы лучше (Спирмен на across 0.617 против 0.412 у v2),
но проиграл допуск на within из-за двух систематических ошибок дня 3:
верный отказ при фрагментах без ответа получал 0, а правдоподобный ответ
по теме без сверки с эталоном — 2. Эти две ветки v2 проверяет атомарно,
поэтому правило берёт балл v1 и заменяет его только там:

- v2 видит отказ → 2, если ни одного ключевого факта нет во фрагментах
  (отказ верный), иначе 0;
- v2 видит противоречие эталону → 0;
- v1 ставит 2, а v2 не нашёл в ответе ни одного ключевого факта → 1.

Правило выведено из разбора ошибок дня 3, а не подобрано по across:
на across оно не меняет ни одного балла (Спирмен 0.617 = v1).

    python scripts/judge_hybrid.py --v1 artifacts/runs/day3/judged/calibration/calibration.json \\
        --v2 artifacts/runs/day5/judge-v2/within-final/calibration.json --group within \\
        --json artifacts/runs/day5/judge-hybrid-within.json

Печатаются только сводные числа: оценки within по одной не выводятся —
это отложенная группа допуска.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from judge_saved import calibration_metrics  # noqa: E402


def hybrid(v1: int | None, checks: dict[str, Any] | None) -> int | None:
    if v1 is None:
        return None
    if not checks:
        return v1  # v2 не ответил — поправлять нечем, остаётся v1
    if checks["refusal"]:
        return 0 if any(checks["facts_in_context"]) else 2
    if checks["contradicts"]:
        return 0
    if v1 == 2 and not any(checks["facts_in_answer"]):
        return 1
    return v1


def combine(v1_rows: list[dict[str, Any]], v2_rows: list[dict[str, Any]],
            group: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    key = lambda r: (r["question_id"], r.get("model"), r.get("group"))
    v1 = {key(r): r for r in v1_rows if r.get("group") == group}
    rows, stats = [], Counter()
    for r in v2_rows:
        if r.get("group") != group:
            continue
        base = v1.get(key(r))
        if base is None:
            stats["нет пары v1"] += 1
            continue
        if base.get("human_grade") != r.get("human_grade"):
            raise SystemExit("ручные оценки v1 и v2 расходятся — прогоны от разных листов")
        score = hybrid(base.get("correctness"), r.get("judge_checks"))
        stats["изменено v2"] += int(score is not None and score != base.get("correctness"))
        stats["без проверок v2"] += int(not r.get("judge_checks"))
        rows.append({"question_id": r["question_id"], "group": group,
                     "human_grade": r["human_grade"], "correctness": score})
    if len(rows) < len(v1):
        stats["v1 без пары v2"] = len(v1) - len(rows)
    return rows, dict(stats)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--v1", type=Path, required=True, help="calibration.json судьи v1")
    parser.add_argument("--v2", type=Path, required=True, help="calibration.json судьи v2")
    parser.add_argument("--group", choices=("within", "across"), required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    v1 = json.loads(args.v1.read_text(encoding="utf-8"))["outcomes"]
    v2 = json.loads(args.v2.read_text(encoding="utf-8"))["outcomes"]
    rows, stats = combine(v1, v2, args.group)
    if not rows:
        raise SystemExit(f"нет общих ответов группы {args.group}")
    metrics = calibration_metrics(rows)
    report = {"group": args.group, "rows": len(rows), "stats": stats, "calibration": metrics,
              "v1": str(args.v1), "v2": str(args.v2)}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
