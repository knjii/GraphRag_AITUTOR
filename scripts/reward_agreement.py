"""Согласие награды с ручной оценкой внутри вопроса.

GRPO видит только порядок наград внутри группы ответов на один вопрос,
поэтому и проверяется порядок: доля пар с разной ручной оценкой, которые
награда упорядочивает так же (ничья в награде — половина). Случайная
награда даёт 0.5; критерий приёмки — 0.75 (``tasks/013-report.md``, 6.3).

    python scripts/reward_agreement.py --key runs/groups-4b-key.json --grades grades.json

``grades.json`` — ``{"1.1": 3, "1.2": 0, ...}`` по меткам слепого листа.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

THRESHOLD = 0.75


def concordance(groups: Sequence[Sequence[tuple[float, float]]]) -> tuple[float, int]:
    """(согласие, число пар) по группам пар «оценка, награда»."""
    agree = 0.0
    total = 0
    for items in groups:
        for (g1, r1), (g2, r2) in itertools.combinations(items, 2):
            if g1 == g2:
                continue
            total += 1
            if r1 == r2:
                agree += 0.5
            elif (g1 > g2) == (r1 > r2):
                agree += 1.0
    return (agree / total if total else float("nan")), total


def bootstrap(groups: Sequence[Sequence[tuple[float, float]]], rounds: int = 2000,
              seed: int = 7) -> tuple[float, float]:
    rng = random.Random(seed)
    values = sorted(
        concordance([rng.choice(groups) for _ in groups])[0] for _ in range(rounds)
    )
    return values[int(0.025 * rounds)], values[int(0.975 * rounds) - 1]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--key", type=Path, required=True)
    parser.add_argument("--grades", type=Path, required=True)
    args = parser.parse_args()

    key = json.loads(args.key.read_text(encoding="utf-8"))
    grades = json.loads(args.grades.read_text(encoding="utf-8"))
    missing = [row["id"] for row in key if row["id"] not in grades]
    if missing:
        raise SystemExit(f"Нет оценок для {len(missing)} ответов: {missing[:5]}")
    by_question: dict[object, list[tuple[float, float]]] = defaultdict(list)
    for row in key:
        by_question[row["q"]].append((float(grades[row["id"]]), float(row["reward"])))
    groups = list(by_question.values())
    value, pairs = concordance(groups)
    low, high = bootstrap(groups)
    verdict = "принята" if low >= 0.5 and value >= THRESHOLD else "НЕ принята"
    print(f"согласие пар внутри вопроса: {value:.3f} на {pairs} парах, 95% [{low:.3f}; {high:.3f}]")
    print(f"награда {verdict} (порог {THRESHOLD}, нижняя граница выше случайной 0.5)")
    return 0 if verdict == "принята" else 1


if __name__ == "__main__":
    raise SystemExit(main())
