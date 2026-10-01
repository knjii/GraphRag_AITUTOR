"""Оракул отбора по слепку (условие запуска R7).

    python scripts/oracle_selection.py
    python scripts/oracle_selection.py --trace capture/session-0819/trace-always.jsonl --json out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.oracle import evaluate, summarize  # noqa: E402
from rag_textbook.rl.env import load_trace  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, default=ROOT / "capture/session-0819/trace-always.jsonl")
    parser.add_argument("--goldset", type=Path, default=ROOT / "evaluation/goldsets/goldset.json")
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()

    questions = json.loads(args.goldset.read_text(encoding="utf-8"))["questions"]
    gold = {q["id"]: set(q["gold_chunk_ids"]) for q in questions}
    summary = summarize(evaluate(load_trace(args.trace), gold, k=args.k))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
