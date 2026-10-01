"""Метрики этапа 1 по выдачам систем на общем наборе (bench_bundle.py).

Одна функция на все системы: разница в числах должна происходить из выдачи,
а не из того, что две системы мерились разным кодом.

Вход — каталог набора и файлы ``rankings-<система>.jsonl`` со строками
``{"qid", "ranked": [chunk_id, …], "pool": [chunk_id, …]?}``. Поле ``pool``
необязательно: у нашего конвейера это кандидаты до отбора, у HippoRAG 2 и
плотного поиска пула отдельно нет — пулом считается вся сохранённая выдача.

Что считается (для каждого k из --k):

``recall@k``  доля эталонных фрагментов в первых k;
``all@k``     доля вопросов, у которых **все** эталонные в первых k — главная
              для многошаговых: один найденный из двух ответа не даёт;
``any@k``     доля вопросов хотя бы с одним;
``last_rank`` медиана места последнего найденного эталонного (второй шаг);
разбивка по полю ``type`` набора (у MuSiQue — 2hop, 3hop1, …).

Сравнение двух систем (--pair A,B) — парный бутстрап по вопросам для
разности all@k и recall@k: интервал, а не одно число.

    python scripts/bench_metrics.py --bundle artifacts/bench/musique-300 \\
        --run dense=runs/.../rankings-dense.jsonl \\
        --run hipporag2=runs/.../rankings-hipporag2.jsonl --pair dense,hipporag2
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def per_question(gold: list[str], ranked: list[str], ks: list[int]) -> dict:
    position = {cid: i + 1 for i, cid in enumerate(ranked)}
    ranks = sorted(position[g] for g in gold if g in position)
    row: dict = {"n_gold": len(gold), "found_ranks": ranks}
    for k in ks:
        hit = sum(1 for r in ranks if r <= k)
        row[f"recall@{k}"] = hit / len(gold)
        row[f"all@{k}"] = float(hit == len(gold))
        row[f"any@{k}"] = float(hit > 0)
    row["recall@pool"] = len(ranks) / len(gold)
    row["all@pool"] = float(len(ranks) == len(gold))
    row["last_rank"] = ranks[-1] if len(ranks) == len(gold) else None
    return row


def summarize(rows: list[dict], ks: list[int]) -> dict:
    out: dict = {"n": len(rows)}
    keys = [f"{m}@{k}" for k in ks for m in ("recall", "all", "any")] + ["recall@pool", "all@pool"]
    for key in keys:
        out[key] = round(sum(r[key] for r in rows) / max(1, len(rows)), 4)
    last = [r["last_rank"] for r in rows if r["last_rank"] is not None]
    out["last_rank_median"] = statistics.median(last) if last else None
    return out


def evaluate(bundle: Path, rankings: Path, ks: list[int]) -> tuple[dict, dict[str, dict]]:
    questions = {q["qid"]: q for q in load_jsonl(bundle / "questions.jsonl") if q["gold_chunk_ids"]}
    per_q: dict[str, dict] = {}
    for item in load_jsonl(rankings):
        q = questions.get(item["qid"])
        if q is None:
            continue
        ranked = item["ranked"]
        row = per_question(q["gold_chunk_ids"], ranked, ks)
        if item.get("pool") is not None:
            pool = set(item["pool"])
            hit = sum(1 for g in q["gold_chunk_ids"] if g in pool)
            row["recall@pool"] = hit / len(q["gold_chunk_ids"])
            row["all@pool"] = float(hit == len(q["gold_chunk_ids"]))
        row["type"] = q.get("type", "")
        per_q[item["qid"]] = row
    missing = len(questions) - len(per_q)
    summary = {"overall": summarize(list(per_q.values()), ks), "missing_questions": missing}
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in per_q.values():
        groups[row["type"]].append(row)
    summary["by_type"] = {t: summarize(rows, ks) for t, rows in sorted(groups.items())}
    return summary, per_q


def paired_bootstrap(a: dict[str, dict], b: dict[str, dict], key: str, n: int = 2000,
                     seed: int = 0) -> dict:
    common = sorted(set(a) & set(b))
    diffs = [b[q][key] - a[q][key] for q in common]
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [diffs[rng.randrange(len(diffs))] for _ in diffs]
        means.append(sum(sample) / len(sample))
    means.sort()
    return {"n": len(common), "delta": round(sum(diffs) / max(1, len(diffs)), 4),
            "ci95": [round(means[int(0.025 * n)], 4), round(means[int(0.975 * n) - 1], 4)],
            "better": sum(1 for d in diffs if d > 0), "worse": sum(1 for d in diffs if d < 0)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True, help="имя=путь к rankings-*.jsonl")
    parser.add_argument("--k", default="5,16,50")
    parser.add_argument("--pair", action="append", default=[], help="A,B — бутстрап разности B−A")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    ks = [int(k) for k in args.k.split(",")]

    results: dict = {"bundle": str(args.bundle), "systems": {}, "pairs": {}}
    per_system: dict[str, dict] = {}
    for spec in args.run:
        name, path = spec.split("=", 1)
        summary, per_q = evaluate(args.bundle, Path(path), ks)
        results["systems"][name] = summary
        per_system[name] = per_q

    header = ["система", "n"] + [f"all@{k}" for k in ks] + [f"recall@{k}" for k in ks] + ["all@пул", "место посл."]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    for name, summary in results["systems"].items():
        o = summary["overall"]
        cells = [name, str(o["n"])] + [f"{o[f'all@{k}']:.3f}" for k in ks] \
            + [f"{o[f'recall@{k}']:.3f}" for k in ks] + [f"{o['all@pool']:.3f}", str(o["last_rank_median"])]
        print("| " + " | ".join(cells) + " |")
        if summary["missing_questions"]:
            print(f"  внимание: {name} — нет выдачи для {summary['missing_questions']} вопросов")

    focus = 16 if 16 in ks else ks[-1]
    for spec in args.pair:
        a, b = spec.split(",")
        results["pairs"][f"{b}-{a}"] = {
            key: paired_bootstrap(per_system[a], per_system[b], key)
            for key in (f"all@{focus}", f"recall@{focus}", f"all@{ks[0]}")
        }
        for key, value in results["pairs"][f"{b}-{a}"].items():
            print(f"{b} − {a}, {key}: {value['delta']:+.3f} [{value['ci95'][0]:+.3f}; {value['ci95'][1]:+.3f}]"
                  f", лучше {value['better']}, хуже {value['worse']}")

    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
