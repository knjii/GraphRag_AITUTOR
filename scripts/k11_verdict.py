"""Вердикт К11 (рёбра-синонимы): критерии записаны в docs/HYPOTHESES.md до замера.

    python scripts/k11_verdict.py \\
        --x-goldset evaluation/goldsets/goldset-x.json --x-base X0.json --x-cand XS.json \\
        --v2-goldset evaluation/goldsets/goldset-v2.json --v2-base V0.json --v2-cand VS.json \\
        --json artifacts/runs/day5/k11-verdict.json

Два набора — два разных вопроса, поэтому срезы не смешиваются:

- основной критерий — goldset-x, test, cross_book: recall@16 ≥ +0.03,
  нижняя граница 95 % > 0, p < 0.05 (одна проверка превосходства — без поправки);
- страховки — внутрикнижный goldset-v2, test: связывающие не хуже −0.01,
  одношаговые не хуже −0.005 (по среднему, как guard серии К).

Объём контекста кандидата не больше базового на 10 %, как в серии К.
Dev goldset-x печатается только как картина.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from rag_textbook.evaluation.runner import load_outcomes  # noqa: E402
from series_k_verdict import ALPHA, MAX_CHARS_RATIO, paired, slice_of  # noqa: E402

PRIMARY = 0.03
GUARDS = {"linking": -0.01, "single": -0.005}


def outcomes(path: Path) -> dict[str, Any]:
    return {o.question_id: o for o in load_outcomes(path)[1]}


def ids(questions: list[Any], split: str, name: str, *, only_x: bool = False) -> list[str]:
    return [q.id for q in questions if q.split == split and name in slice_of(q)
            and (not only_x or q.slice == "cross_book")]


def decide(primary: dict[str, Any], guards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    checks = {}
    budget = lambda row: row.get("chars_ratio") is None or row["chars_ratio"] <= MAX_CHARS_RATIO
    checks["межкнижные ≥ +0.03"] = bool(
        primary.get("questions") and primary["delta"] >= PRIMARY and primary["ci"][0] > 0
        and primary["p"] < ALPHA and budget(primary))
    for name, threshold in GUARDS.items():
        row = guards[name]
        checks[f"внутрикнижные {name} ≥ {threshold:+}"] = bool(
            row.get("questions") and row["delta"] >= threshold)
    return {"checks": checks, "decision": "принята" if all(checks.values()) else "отвергнута"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    for key in ("x-goldset", "x-base", "x-cand", "v2-goldset", "v2-base", "v2-cand"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)

    xq, vq = load_goldset(args.x_goldset), load_goldset(args.v2_goldset)
    xb, xc = outcomes(args.x_base), outcomes(args.x_cand)
    vb, vc = outcomes(args.v2_base), outcomes(args.v2_cand)
    primary = paired(xb, xc, ids(xq, "test", "cross_book", only_x=True))
    dev = paired(xb, xc, ids(xq, "dev", "cross_book", only_x=True))
    guards = {name: paired(vb, vc, ids(vq, "test", name)) for name in GUARDS}
    result = {"primary": primary, "dev": dev, "guards": guards, **decide(primary, guards)}

    def line(title: str, row: dict[str, Any]) -> str:
        if not row.get("questions"):
            return f"{title:34} нет вопросов"
        return (f"{title:34} n={row['questions']:<4} Δ={row['delta']:+.3f} "
                f"[{row['ci'][0]:+.3f}; {row['ci'][1]:+.3f}] p={row['p']:.3f} "
                f"full Δ={row['full_delta']:+.3f} лучше/хуже {row['improved']}/{row['worsened']} "
                f"контекст ×{(row['chars_ratio'] or 0):.2f}")
    print(line("К11 межкнижные test (критерий)", primary))
    print(line("К11 межкнижные dev (картина)", dev))
    for name, row in guards.items():
        print(line(f"К11 внутрикнижные {name} (страховка)", row))
    for name, passed in result["checks"].items():
        print(f"  {'выполнено' if passed else 'НЕ выполнено'}: {name}")
    print(f"К11: {result['decision']}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
