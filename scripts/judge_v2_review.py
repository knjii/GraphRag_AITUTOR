"""Разбор прогона судьи v2 на группе across между редакциями (M6).

Подбор инструкции идёт только на across; within — отложенная группа для
допуска, и её результат этим скриптом не читается вовсе: файл с другой
группой отвергается до разбора.

Что печатается:
- согласованность с человеком (Спирмен), доля невалидных оценок;
- таблица «человек 0–3 × судья 0–2»;
- расхождения двух видов, из-за которых судья 27B не прошёл J0 дня 3:
  «занижен» (человек ≥ 2, судья 0) и «завышен» (человек ≤ 1, судья 2),
  с веткой правила, давшей балл, и атомарными ответами судьи;
- частоты признаков (отказ, противоречие, доля найденных фактов).

    python scripts/judge_v2_review.py artifacts/runs/day4/judge-v2/across-r0/calibration.json \
        [--facts evaluation/judge_facts/calibration.json] [--show 12] [--json отчёт.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def branch(checks: dict[str, Any] | None) -> str:
    """Какая ветка score_judge_v2 дала балл — чтобы править нужную проверку."""
    if not checks:
        return "невалидно"
    if checks["refusal"]:
        return "отказ при факте во фрагментах" if any(checks["facts_in_context"]) else "верный отказ"
    if checks["contradicts"]:
        return "противоречие"
    found, total = sum(checks["facts_in_answer"]), len(checks["facts_in_answer"])
    return f"факты {found}/{total}"


def review(data: dict[str, Any], facts: dict[str, list[str]], show: int) -> dict[str, Any]:
    group = data.get("judge_provenance", {}).get("calibration_group")
    rows = data["outcomes"]
    if group != "across" or any(row.get("group") != "across" for row in rows):
        raise SystemExit("разбор только для группы across: within — отложенная группа допуска (M6)")
    valid = [r for r in rows if r.get("correctness") is not None]
    table = Counter((r["human_grade"], r["correctness"]) for r in valid)
    under = [r for r in valid if r["human_grade"] >= 2 and r["correctness"] == 0]
    over = [r for r in valid if r["human_grade"] <= 1 and r["correctness"] == 2]
    checks = [r["judge_checks"] for r in valid if r.get("judge_checks")]
    found = [sum(c["facts_in_answer"]) / len(c["facts_in_answer"]) for c in checks]
    summary = {
        "calibration": data.get("calibration", {}),
        "rows": len(rows), "invalid": len(rows) - len(valid),
        "invalid_share": round((len(rows) - len(valid)) / max(1, len(rows)), 3),
        "table": {f"{h}->{j}": n for (h, j), n in sorted(table.items())},
        "under": len(under), "over": len(over),
        "under_branches": dict(Counter(branch(r.get("judge_checks")) for r in under)),
        "over_branches": dict(Counter(branch(r.get("judge_checks")) for r in over)),
        "refusal_rate": round(sum(c["refusal"] for c in checks) / max(1, len(checks)), 3),
        "contradicts_rate": round(sum(c["contradicts"] for c in checks) / max(1, len(checks)), 3),
        "facts_found_mean": round(sum(found) / max(1, len(found)), 3),
    }
    cases = []
    for kind, items in (("занижен", under), ("завышен", over)):
        for r in sorted(items, key=lambda r: (r["question_id"], r.get("model", "")))[:show]:
            cases.append({"kind": kind, "question_id": r["question_id"], "model": r.get("model"),
                          "human": r["human_grade"], "judge": r["correctness"],
                          "branch": branch(r.get("judge_checks")), "checks": r.get("judge_checks"),
                          "facts": facts.get(r["question_id"], []),
                          "answer": str(r.get("answer", ""))[:400]})
    summary["cases"] = cases
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("calibration", type=Path)
    parser.add_argument("--facts", type=Path, default=ROOT / "evaluation/judge_facts/calibration.json")
    parser.add_argument("--show", type=int, default=12, help="сколько расхождений каждого вида печатать")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    data = json.loads(args.calibration.read_text(encoding="utf-8"))
    facts_data = json.loads(args.facts.read_text(encoding="utf-8")) if args.facts.exists() else {}
    facts = facts_data.get("facts", facts_data) if isinstance(facts_data, dict) else {}
    result = review(data, facts, args.show)
    cal = result["calibration"]
    print(f"строк {result['rows']}, невалидных {result['invalid']} ({result['invalid_share']:.1%}); "
          f"Спирмен across {cal.get('spearman_across')}")
    print("человек→судья:", result["table"])
    print(f"занижен {result['under']}: {result['under_branches']}")
    print(f"завышен {result['over']}: {result['over_branches']}")
    print(f"отказ {result['refusal_rate']}, противоречие {result['contradicts_rate']}, "
          f"найдено фактов в среднем {result['facts_found_mean']}")
    for case in result["cases"]:
        print(f"\n--- {case['kind']}: {case['question_id']} {case['model']} "
              f"человек {case['human']} / судья {case['judge']} ({case['branch']})")
        for i, fact in enumerate(case["facts"], 1):
            print(f"  факт {i}: {fact}")
        print(f"  проверки: {json.dumps(case['checks'], ensure_ascii=False)}")
        print(f"  ответ: {case['answer']}")
    if args.json:
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
