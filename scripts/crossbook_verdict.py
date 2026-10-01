"""Вердикт межкнижной проверки «не хуже» (серия S, мера утверждена 2026-10-01).

На goldset-x эталонные ответы — развёрнутые русские объяснения, EM вырождена.
Решает мера, утверждённая владельцем до замера (docs/HYPOTHESES.md, «Межкнижная
проверка: смена меры»):

* главная — ``ctx_recall`` контекста генератора (доля эталонных фрагментов
  среди тех k, что прочёл генератор), не ниже базы более чем на 0.03 точечно;
* вспомогательная — F1 ответа (русский промпт), тот же запас;
* EM показывается, но не решает.

На 60 вопросах значимости не будет, поэтому интервал бутстрепа печатается
только как пояснение. Ответы берутся из ``answers-<имя>.jsonl``, которые пишет
``bench_answers.py`` (поле ``ids`` — что реально прочёл генератор).

    python scripts/crossbook_verdict.py --bundle artifacts/bench/goldset-x \\
        --answers artifacts/runs/bench/goldset-x/answers --base x-current
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from bench_answers import exact_match, f1_score, final_answer, load_jsonl  # noqa: E402

MARGIN = 0.03
BOOTSTRAP = 10_000


def per_question(path: Path, questions: dict[str, dict]) -> dict[str, dict[str, float]]:
    out = {}
    for record in load_jsonl(path):
        q = questions.get(record["qid"])
        if q is None:
            continue
        gold = set(q["gold_chunk_ids"])
        prediction = final_answer(record["raw"])
        golds = [q["answer"], *q.get("answer_aliases", [])]
        out[record["qid"]] = {
            "ctx_recall": len(gold & set(record["ids"])) / len(gold) if gold else 0.0,
            "ctx_all": float(bool(gold) and gold <= set(record["ids"])),
            "f1": f1_score(prediction, golds),
            "em": exact_match(prediction, golds),
            "answered": float(bool(prediction)),
            "error": float(bool(record.get("error"))),
        }
    return out


def bootstrap_ci(diffs: list[float], seed: int = 0) -> tuple[float, float]:
    rng = random.Random(seed)
    n = len(diffs)
    means = sorted(statistics.fmean(rng.choices(diffs, k=n)) for _ in range(BOOTSTRAP))
    return means[int(0.025 * BOOTSTRAP)], means[int(0.975 * BOOTSTRAP) - 1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--answers", type=Path, required=True)
    parser.add_argument("--base", default="x-current")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    questions = {q["qid"]: q for q in load_jsonl(args.bundle / "questions.jsonl")}
    runs = {p.stem.removeprefix("answers-"): per_question(p, questions)
            for p in sorted(args.answers.glob("answers-*.jsonl"))}
    if args.base not in runs:
        print(f"СТОП: нет ответов базы {args.base} в {args.answers}", file=sys.stderr)
        return 1
    base = runs[args.base]
    report: dict[str, dict] = {"base": args.base, "margin": MARGIN, "questions": len(questions), "systems": {}}
    metrics = ("ctx_recall", "ctx_all", "f1", "em", "answered")
    print(f"{'система':26s} {'n':>3s} " + " ".join(f"{m:>10s}" for m in metrics))
    for name, rows in runs.items():
        common = sorted(set(rows) & set(base))
        entry: dict = {"n": len(rows), "errors": int(sum(r["error"] for r in rows.values()))}
        for m in metrics:
            entry[m] = round(statistics.fmean(r[m] for r in rows.values()), 4) if rows else None
        print(f"{name:26s} {len(rows):3d} " + " ".join(f"{entry[m]:10.3f}" for m in metrics))
        if name != args.base and common:
            for m in ("ctx_recall", "f1", "em"):
                diffs = [rows[q][m] - base[q][m] for q in common]
                low, high = bootstrap_ci(diffs)
                entry[f"{m}_diff"] = {"point": round(statistics.fmean(diffs), 4),
                                      "ci95": [round(low, 4), round(high, 4)],
                                      "better": sum(d > 0 for d in diffs), "worse": sum(d < 0 for d in diffs)}
            entry["not_worse_ctx_recall"] = entry["ctx_recall_diff"]["point"] >= -MARGIN
            entry["not_worse_f1"] = entry["f1_diff"]["point"] >= -MARGIN
            entry["verdict"] = ("не хуже" if entry["not_worse_ctx_recall"] and entry["not_worse_f1"]
                                else "хуже" if not entry["not_worse_ctx_recall"]
                                else "не хуже по контексту, хуже по F1")
        report["systems"][name] = entry
        if len(rows) < len(questions):
            print(f"  ! {name}: ответов {len(rows)} из {len(questions)}")
        if entry["errors"] > 0.02 * max(1, len(rows)):
            print(f"  ! {name}: ошибок генератора {entry['errors']} — больше 2%, замер недействителен")
    print(f"\nпротив {args.base}, запас {MARGIN} (точечно; интервал — пояснение):")
    for name, entry in report["systems"].items():
        if "verdict" not in entry:
            continue
        c, f, e = entry["ctx_recall_diff"], entry["f1_diff"], entry["em_diff"]
        print(f"  {name:24s} ctx_recall {c['point']:+.3f} {c['ci95']}  F1 {f['point']:+.3f} {f['ci95']}"
              f"  EM {e['point']:+.3f}  → {entry['verdict']}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
