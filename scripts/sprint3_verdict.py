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

Коды выхода: 0 — принято, 2 — не принято (штатный отрицательный итог),
1 — входы негодны, вердикта нет. Входы проверяются до расчёта (задача 021):
ровно два разных контроля, одинаковый полный набор вопросов без повторов,
один промпт и один источник контекста, а движок у каждого замера свой —
иначе «кандидат» мог отвечать базовой моделью.
"""

from __future__ import annotations

import argparse
import json
import math
import os
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


class BadInputs(Exception):
    """Входы не позволяют вынести вердикт — это не «не принято»."""


def load(metrics: Path, label: str) -> dict[str, Any]:
    # Только точное имя: замер пишет ровно answers_<метка>.json. Прежний
    # запасной поиск по маске «answers_<метка>_*» брал последний по алфавиту
    # файл и принимал старый замер вместо свежего (задача 021).
    path = metrics / f"answers_{label}.json"
    if not path.is_file():
        raise BadInputs(f"нет замера {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def validate(runs: dict[str, dict[str, Any]], controls: list[str]) -> None:
    """Сопоставимость замеров — до любых чисел."""
    if len(controls) != 2 or len(set(controls)) != 2:
        raise BadInputs(f"нужны ровно два разных контроля (random и format), получено {controls}")
    question_sets = {}
    for label, run in runs.items():
        ids = [str(item["question_id"]) for item in run["outcomes"]]
        if len(ids) != len(set(ids)):
            raise BadInputs(f"{label}: вопросы повторяются ({len(ids) - len(set(ids))})")
        question_sets[label] = set(ids)
        value = run.get("summary", {}).get("всего", {}).get("опора на контекст")
        # Наличия ключа мало: строка "inf" или NaN прошли бы любое «не ниже
        # базы» (задача 022).
        if (isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or not 0 <= value <= 1):
            raise BadInputs(f"{label}: опора на контекст не число из [0, 1]: {value!r}")
        made = run.get("summary", {}).get("чем сделано")
        if not isinstance(made, dict):
            raise BadInputs(f"{label}: замер без провенанса («чем сделано»)")
        # Отсутствующее поле у всех замеров иначе совпало бы как None == None.
        for field in ("промпт", "контекст", "модель ответа", "модель по настройке"):
            if not isinstance(made.get(field), str) or not made[field].strip():
                raise BadInputs(f"{label}: в провенансе нет «{field}»")
        window = made.get("окно контекста")
        if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
            raise BadInputs(f"{label}: «окно контекста» не целое: {window!r}")
    first = next(iter(question_sets.values()))
    for label, ids in question_sets.items():
        if ids != first:
            raise BadInputs(f"{label}: другой набор вопросов ({len(ids)} против {len(first)})")
    made = {label: run["summary"]["чем сделано"] for label, run in runs.items()}
    for field in ("промпт", "контекст", "окно контекста"):
        values = {label: info.get(field) for label, info in made.items()}
        if len(set(values.values())) != 1:
            raise BadInputs(f"замеры сделаны с разным «{field}»: {values}")
    served = {label: info.get("модель ответа") for label, info in made.items()}
    for label, info in made.items():
        if info.get("модель ответа") != info.get("модель по настройке"):
            raise BadInputs(
                f"{label}: отвечала {info.get('модель ответа')}, а настроена "
                f"{info.get('модель по настройке')} — запросы шли не туда"
            )
    if len(set(served.values())) != len(served):
        raise BadInputs(f"одна модель отвечала за разные прогоны: {served}")


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


def write_atomic(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--base", required=True)
    parser.add_argument("--main", required=True)
    parser.add_argument("--controls", nargs="*", default=[])
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)
    # Прежний вердикт убирается сразу: при сбое ниже он остался бы лежать
    # и сошёл бы за итог этого прогона (задача 021).
    if args.out is not None and args.out.exists():
        args.out.unlink()
    try:
        labels = [args.base, args.main, *args.controls]
        # Словарь по меткам схлопнул бы повтор, и контроль молча совпал бы
        # с базой или кандидатом (задача 022: KeyError вместо отказа).
        if len(set(labels)) != len(labels):
            raise BadInputs(f"метки прогонов обязаны различаться: {labels}")
        runs = {label: load(args.metrics, label)
                for label in [args.base, args.main, *args.controls]}
        validate(runs, args.controls)
        return decide(args, runs)
    except (BadInputs, ValueError) as error:
        print(f"ВХОДЫ НЕГОДНЫ, вердикта нет: {error}", file=sys.stderr)
        return 1


def decide(args: argparse.Namespace, runs: dict[str, dict[str, Any]]) -> int:
    scores = {label: at_least_one(run) for label, run in runs.items()}
    base = scores[args.base]

    report: dict[str, Any] = {"вопросов с формулами": len(base), "прогоны": {}}
    # Критерии сравниваются по точным значениям; округление — только для
    # показа. Прирост 21/382 = 0.05497 округлялся до 0.055 и проходил.
    raw: dict[str, dict[str, float]] = {}
    print(f"вопросов с формулами в эталоне: {len(base)}\n")
    print(f"{'прогон':<12} {'хотя бы одна':>13} {'прирост':>9} {'интервал':>20} "
          f"{'опора':>7} {'взлом':>7}")
    for label, values in scores.items():
        mean = sum(values.values()) / max(1, len(values))
        raw[label] = {"опора": support(runs[label]), "взлом": hack_share(runs[label])}
        row: dict[str, Any] = {"хотя бы одна": round(mean, 4),
                               "опора на контекст": round(raw[label]["опора"], 4),
                               "взлом": round(raw[label]["взлом"], 4)}
        if label != args.base:
            stats = paired(values, base)
            raw[label]["прирост"] = stats["mean"]
            raw[label]["низ"] = stats["low"]
            row["прирост"] = round(stats["mean"], 4)
            row["интервал"] = [round(stats["low"], 4), round(stats["high"], 4)]
            interval = f"[{stats['low']:+.3f}; {stats['high']:+.3f}]"
            gain = f"{stats['mean']:+.3f}"
        else:
            interval, gain = "", ""
        print(f"{label:<12} {mean:>13.3f} {gain:>9} {interval:>20} "
              f"{row['опора на контекст']:>7.3f} {row['взлом']:>7.3f}")
        report["прогоны"][label] = row

    main_raw = raw[args.main]
    checks = {
        f"прирост ≥ {MIN_GAIN}": main_raw["прирост"] >= MIN_GAIN,
        "интервал не покрывает ноль": main_raw["низ"] > 0,
        "контроли меньше половины основного": all(
            raw[c]["прирост"] < MAX_CONTROL_SHARE * main_raw["прирост"] for c in args.controls
        ),
        "опора не ниже базы": main_raw["опора"] >= raw[args.base]["опора"],
        f"взлом ≤ {MAX_HACK_SHARE}": main_raw["взлом"] <= MAX_HACK_SHARE,
    }
    print("\nкритерии, записанные до прогона")
    for name, passed in checks.items():
        print(f"  {'да ' if passed else 'НЕТ'} {name}")

    accepted = all(checks.values())
    report["критерии"] = checks
    report["вердикт"] = "принято" if accepted else "не принято"
    print(f"\nвердикт: {report['вердикт']}")
    if args.out:
        write_atomic(args.out, report)
        print(f"записано: {args.out}")
    return 0 if accepted else 2


if __name__ == "__main__":
    raise SystemExit(main())
