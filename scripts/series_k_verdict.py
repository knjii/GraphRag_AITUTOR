"""Вердикт серии К по сохранённым прогонам: критерии записаны до замера.

    python scripts/series_k_verdict.py --goldset evaluation/goldsets/goldset-v2.json --split test \\
        --run base=artifacts/metrics/retrieval_eval_k-base_<время>.json --run K2=... ... \\
        --json artifacts/runs/day3/verdict-test.json

Каждый прогон — файл ``eval run`` (с ``outcomes``). Имена прогонов
фиксированы, потому что критерий знает, что с чем сравнивать:

- ``base``   — модельный граф v4 файлом, обход по совместному упоминанию;
- ``K1A``    — структурный граф, ``K1C`` — объединение со структурным;
- ``K2``     — обход по зависимостям (``GRAPH_WALK=dependency``);
- ``K3none`` / ``K3book`` / ``K3sec`` — обозначения: нет, на всю книгу, на раздел;
- ``K4walk`` / ``K4ppr`` — лучший граф, обход против PPR при тех же затравках;
- ``K8``     — замыкание контекста по зависимостям;
- ``K6a`` / ``K6b`` — условный отбор жадный и парами; ``K7`` — распространение.

Гипотеза принята, если хотя бы один её вариант прошёл все свои проверки
(К1: объединение лучше **или** структурный не хуже; К6: жадный **или** пары).

Отсутствующий прогон — «не ставился», а не провал. Пороги взяты
из docs/HYPOTHESES.md, серия К; мера — recall@16 на срезе. Полнота пары
(full@16) и объём контекста печатаются рядом: протокол требует равного
бюджета, и вариант, выигравший длиной фрагментов, не принимается.

Проверки превосходства одной семьи корректируются по Холму. Принимает
гипотезу только отложенная часть (``--split test``); часть подбора
(``dev``) годится для выбора варианта, и вердикт на ней помечается.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from rag_textbook.evaluation.metrics import (  # noqa: E402
    QueryOutcome,
    full_at_k,
    paired_bootstrap,
    recall_at_k,
)
from rag_textbook.evaluation.runner import load_outcomes  # noqa: E402

K = 16
ALPHA = 0.05
# Объём контекста кандидата больше базового больше чем на 10 % — бюджет
# не уравнен (SetCE): прирост мог дать просто лишний текст.
MAX_CHARS_RATIO = 1.10
LINKING_TYPES = {"graph_linked", "multi_hop", "relation"}


@dataclass(frozen=True)
class Test:
    hypothesis: str
    candidate: str
    reference: str
    slice: str
    kind: str  # superiority | noninferiority | guard
    threshold: float
    note: str


TESTS: tuple[Test, ...] = (
    Test("К1", "K1C", "base", "linking", "superiority", 0.0,
         "объединение лучше модельного"),
    Test("К1", "K1A", "base", "linking", "noninferiority", -0.02,
         "структурный не хуже модельного больше чем на 0.02"),
    Test("К2", "K2", "base", "linking", "superiority", 0.02, "связывающие ≥ +0.02"),
    Test("К2", "K2", "base", "cross_book", "superiority", 0.03, "межкнижные ≥ +0.03"),
    Test("К3", "K3sec", "K3book", "linking", "superiority", 0.02,
         "область «раздел» лучше глобальных на 0.02"),
    Test("К4", "K4ppr", "K4walk", "cross_book", "superiority", 0.03,
         "PPR лучше обхода на межкнижных на 0.03"),
    Test("К8", "K8", "base", "linking", "superiority", 0.02, "связывающие ≥ +0.02"),
    Test("К8", "K8", "base", "single", "guard", -0.005, "одношаговые не хуже −0.005"),
    # Запас оракула у связывающих +0.126 (слепок 2026-08-19): К6 обязана
    # забрать треть, К7 — четверть. Пороги абсолютные, как записаны.
    Test("К6", "K6a", "base", "linking", "superiority", 0.042, "треть запаса оракула"),
    Test("К6", "K6a", "base", "single", "guard", -0.005, "одношаговые не хуже −0.005"),
    Test("К6", "K6b", "base", "linking", "superiority", 0.042, "треть запаса оракула"),
    Test("К6", "K6b", "base", "single", "guard", -0.005, "одношаговые не хуже −0.005"),
    Test("К7", "K7", "base", "linking", "superiority", 0.0315, "четверть запаса оракула"),
)


def slice_of(question: Any) -> set[str]:
    """Срезы вопроса: v2 размечает slice, прежний набор — только тип."""
    slices = set()
    if question.slice == "cross_book":
        slices |= {"cross_book", "linking"}
    elif question.slice == "linking" or question.question_type in LINKING_TYPES:
        slices.add("linking")
    if question.question_type == "single_chunk" and question.expected_hops <= 1:
        slices.add("single")
    return slices


def paired(
    base: dict[str, QueryOutcome], cand: dict[str, QueryOutcome], ids: list[str]
) -> dict[str, Any]:
    ids = [qid for qid in ids if qid in base and qid in cand]
    if not ids:
        return {"questions": 0}
    rec = [recall_at_k(cand[q].retrieved, cand[q].relevant, K)
           - recall_at_k(base[q].retrieved, base[q].relevant, K) for q in ids]
    full = [full_at_k(cand[q].retrieved, cand[q].relevant, K)
            - full_at_k(base[q].retrieved, base[q].relevant, K) for q in ids]
    stats = paired_bootstrap(rec)
    chars_base = [sum(base[q].context_chars[:K]) for q in ids]
    chars_cand = [sum(cand[q].context_chars[:K]) for q in ids]
    ratio = (fmean(chars_cand) / fmean(chars_base)) if chars_base and fmean(chars_base) else None
    return {
        "questions": len(ids),
        "delta": stats["mean"],
        "ci": [stats["low"], stats["high"]],
        "p": stats["p_value"],
        "full_delta": fmean(full),
        "improved": sum(1 for d in rec if d > 0),
        "worsened": sum(1 for d in rec if d < 0),
        "chars_ratio": ratio,
    }


def holm(pvalues: dict[int, float]) -> dict[int, float]:
    """Поправка Холма: скорректированные p, монотонные по порядку."""
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    m = len(ordered)
    adjusted: dict[int, float] = {}
    running = 0.0
    for rank, (index, p) in enumerate(ordered):
        running = max(running, min(1.0, (m - rank) * p))
        adjusted[index] = running
    return adjusted


def verdict(
    runs: dict[str, dict[str, QueryOutcome]], questions: list[Any], split: str
) -> dict[str, Any]:
    ids_by_slice: dict[str, list[str]] = {"linking": [], "cross_book": [], "single": []}
    for question in questions:
        if split and question.split != split:
            continue
        for name in slice_of(question):
            ids_by_slice[name].append(question.id)

    rows: list[dict[str, Any]] = []
    for test in TESTS:
        row: dict[str, Any] = {**test.__dict__}
        if test.candidate not in runs or test.reference not in runs:
            row["status"] = "не ставился"
        else:
            row.update(paired(runs[test.reference], runs[test.candidate], ids_by_slice[test.slice]))
            if not row["questions"]:
                row["status"] = "нет вопросов среза"
        rows.append(row)

    family = {i: row["p"] for i, row in enumerate(rows)
              if row["kind"] == "superiority" and "status" not in row}
    adjusted = holm(family)
    for i, row in enumerate(rows):
        if "status" in row:
            continue
        budget_ok = row["chars_ratio"] is None or row["chars_ratio"] <= MAX_CHARS_RATIO
        if row["kind"] == "superiority":
            row["p_holm"] = adjusted[i]
            passed = row["delta"] >= row["threshold"] and row["p_holm"] < ALPHA and row["ci"][0] > 0
        elif row["kind"] == "noninferiority":
            passed = row["ci"][0] > row["threshold"]
        else:
            passed = row["delta"] >= row["threshold"]
        if passed and not budget_ok:
            row["status"] = "не принято: бюджет контекста не уравнен"
        else:
            row["status"] = "выполнено" if passed else "не выполнено"

    # Гипотеза принята, если хотя бы один её вариант прошёл все свои проверки.
    groups: dict[str, dict[str, list[str]]] = {}
    for row in rows:
        groups.setdefault(row["hypothesis"], {}).setdefault(row["candidate"], []).append(row["status"])
    decisions = {}
    for hypothesis, variants in groups.items():
        statuses = [status for items in variants.values() for status in items]
        if all(status == "не ставился" for status in statuses):
            decisions[hypothesis] = "не ставилась"
        elif any(all(status == "выполнено" for status in items) for items in variants.values()):
            decisions[hypothesis] = "принята"
        else:
            decisions[hypothesis] = "отвергнута"
    if split != "test":
        decisions = {h: f"{d} (часть «{split or 'вся'}» — только выбор варианта)"
                     for h, d in decisions.items()}
    return {"split": split, "k": K, "slices": {n: len(v) for n, v in ids_by_slice.items()},
            "tests": rows, "decisions": decisions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goldset", type=Path, required=True)
    parser.add_argument("--split", default="test", help="test | dev | пусто — весь набор")
    parser.add_argument("--run", action="append", default=[], help="имя=файл прогона")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    known = {t.candidate for t in TESTS} | {t.reference for t in TESTS}
    runs: dict[str, dict[str, QueryOutcome]] = {}
    for item in args.run:
        name, _, path = item.partition("=")
        if name not in known:
            print(f"неизвестный прогон {name}; ожидаются {sorted(known)}", file=sys.stderr)
            return 2
        runs[name] = {o.question_id: o for o in load_outcomes(Path(path))[1]}
    if "base" not in runs:
        print("нужен прогон base", file=sys.stderr)
        return 2
    questions = load_goldset(args.goldset)
    result = verdict(runs, questions, args.split)
    for row in result["tests"]:
        if "delta" in row:
            print(f"{row['hypothesis']:3} {row['candidate']:>6} против {row['reference']:<6} "
                  f"{row['slice']:<10} n={row['questions']:<4} Δ={row['delta']:+.3f} "
                  f"[{row['ci'][0]:+.3f}; {row['ci'][1]:+.3f}] full Δ={row['full_delta']:+.3f} "
                  f"— {row['status']}  ({row['note']})")
        else:
            print(f"{row['hypothesis']:3} {row['candidate']:>6} {row['slice']:<10} — {row['status']}")
    print(json.dumps(result["decisions"], ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
