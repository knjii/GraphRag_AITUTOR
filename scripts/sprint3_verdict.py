"""Вердикт спринта 3 по критериям, записанным до прогона.

Критерии — раздел 4.5 `docs/RESEARCH-2026-09.md`. Программа их применяет,
а не пересказывает: после двух-трёх дней оплаченной карты соблазн
подвинуть порог на сотую слишком велик, чтобы полагаться на честность
в моменте.

    python scripts/sprint3_verdict.py --metrics artifacts/metrics \\
        --base base-4b --main rl-main --controls rl-random rl-format

Мера — «хотя бы одна формула»: доля вопросов, у которых в эталонном
фрагменте формулы были и хотя бы одна дошла до ответа дословно. Она
объективна и не зависит от награды: награда считает перенос по своей
канонической форме, метрика — по своей (`latex_recall` замера ответов).

**Принять**, если все четыре условия выполнены:

1. прирост основного прогона над базой ≥ +0.055;
2. интервал парного бутстрапа не покрывает ноль;
3. прирост каждого контроля меньше половины основного;
4. опора на контекст не ниже базы, доля взлома ≤ 0.10.

**Опровергнуть**, если прирост неотличим от случайной награды или доля
взлома больше 0.10.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.metrics import paired_bootstrap  # noqa: E402

MIN_GAIN = 0.055
MAX_CONTROL_SHARE = 0.5
MAX_HACK_SHARE = 0.10
# Ответ, больше чем наполовину состоящий из формул, — не ответ, а выписка;
# награда за перенос формул толкает ровно туда, поэтому доля меряется.
HACK_FORMULA_SHARE = 0.6
MATH = re.compile(r"\$\$?.+?\$\$?", re.DOTALL)


def load(metrics: Path, label: str) -> dict[str, Any]:
    matches = sorted(metrics.glob(f"answers_{label}.json"))
    matches += sorted(metrics.glob(f"answers_{label}_*.json"))
    if not matches:
        raise FileNotFoundError(f"нет замера с меткой {label} в {metrics}")
    return json.loads(matches[-1].read_text(encoding="utf-8"))


def at_least_one(run: dict[str, Any]) -> dict[str, float]:
    """По вопросу: 1, если хотя бы одна эталонная формула дошла до ответа."""
    return {
        item["question_id"]: float(item.get("latex_found", 0) > 0)
        for item in run["outcomes"]
        if item.get("latex_expected", 0) > 0
    }


def hack_share(run: dict[str, Any]) -> float:
    answers = [item.get("answer") or "" for item in run["outcomes"]]
    if not answers:
        return 0.0
    hacked = 0
    for text in answers:
        length = len(text.strip())
        if not length:
            continue
        math = sum(len(m.group(0)) for m in MATH.finditer(text))
        if math / length > HACK_FORMULA_SHARE:
            hacked += 1
    return hacked / len(answers)


def support(run: dict[str, Any]) -> float:
    return float(run["summary"]["всего"].get("опора на контекст", 0.0))


def paired(candidate: dict[str, float], baseline: dict[str, float]) -> dict[str, float]:
    if set(candidate) != set(baseline):
        # Совпадения подмножества мало: замер на половине вопросов дал бы
        # «прирост», который на деле — разница выборок. Этот прогон
        # у нас уже был, когда сохранность формул посчиталась по чужому
        # корпусу при полной с виду сводке.
        only_candidate = len(set(candidate) - set(baseline))
        only_base = len(set(baseline) - set(candidate))
        raise ValueError(
            f"замеры считались по разным вопросам: только в кандидате "
            f"{only_candidate}, только в базе {only_base}"
        )
    shared = sorted(candidate)
    return paired_bootstrap([candidate[q] - baseline[q] for q in shared])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    runs = {label: load(args.metrics, label)
            for label in [args.base, args.main, *args.controls]}
    scores = {label: at_least_one(run) for label, run in runs.items()}
    base = scores[args.base]

    report: dict[str, Any] = {"вопросов с формулами": len(base), "прогоны": {}}
    print(f"вопросов с формулами в эталоне: {len(base)}\n")
    print(f"{'прогон':<12} {'хотя бы одна':>13} {'прирост':>9} {'интервал':>20} "
          f"{'опора':>7} {'взлом':>7}")
    for label, values in scores.items():
        mean = sum(values.values()) / max(1, len(values))
        row: dict[str, Any] = {"хотя бы одна": round(mean, 4),
                               "опора на контекст": round(support(runs[label]), 4),
                               "взлом": round(hack_share(runs[label]), 4)}
        if label != args.base:
            stats = paired(values, base)
            row["прирост"] = round(stats["mean"], 4)
            row["интервал"] = [round(stats["low"], 4), round(stats["high"], 4)]
            interval = f"[{stats['low']:+.3f}; {stats['high']:+.3f}]"
            gain = f"{stats['mean']:+.3f}"
        else:
            interval, gain = "", ""
        print(f"{label:<12} {mean:>13.3f} {gain:>9} {interval:>20} "
              f"{row['опора на контекст']:>7.3f} {row['взлом']:>7.3f}")
        report["прогоны"][label] = row

    main_row = report["прогоны"][args.main]
    checks = {
        f"прирост ≥ {MIN_GAIN}": main_row["прирост"] >= MIN_GAIN,
        "интервал не покрывает ноль": main_row["интервал"][0] > 0,
        "контроли меньше половины основного": all(
            report["прогоны"][c]["прирост"] < MAX_CONTROL_SHARE * main_row["прирост"]
            for c in args.controls
        ) if args.controls else False,
        "опора не ниже базы": (main_row["опора на контекст"]
                               >= report["прогоны"][args.base]["опора на контекст"]),
        f"взлом ≤ {MAX_HACK_SHARE}": main_row["взлом"] <= MAX_HACK_SHARE,
    }
    print("\nкритерии, записанные до прогона")
    for name, passed in checks.items():
        print(f"  {'да ' if passed else 'НЕТ'} {name}")
    if not args.controls:
        print("  (контролей не передано — без них результат не принимается)")

    accepted = all(checks.values())
    report["критерии"] = checks
    report["вердикт"] = "принято" if accepted else "не принято"
    print(f"\nвердикт: {report['вердикт']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"записано: {args.out}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
