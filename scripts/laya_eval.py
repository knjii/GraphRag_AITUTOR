"""Классификатор решений в работе: проверка, гейт L2 и цепочка L1 (серия L).

Три подкоманды, все с одной моделью (базовой или дообученной):

``heldout``  AUC и точность по 500 отложенным вопросам MuSiQue train —
             условие действительности замера (AUC < 0.70 — не обучилось);
``gate``     вероятность «пятёрки хватает» по первым пяти выдачи системы;
``chain``    жадная сборка цепочки поверх выдачи (правило записано в
             docs/HYPOTHESES.md до замера): якорь — первое место, кандидаты —
             пул и первые 16, добавляется лучший при вероятности ≥ 0.5,
             следующий шаг — от добавленного; свободные места — порядком
             реранкера. Пишет rankings-*.jsonl того же вида, что у бенча.
             ``--seed selected`` (второй вопрос L1, записан 2026-10-01):
             цепочка растёт от множества SetR, кандидат оценивается против
             каждого члена цепочки (берётся максимум), свободные места —
             порядком «SetR, затем реранкер» (это же контроль setr-fill);
``setrfill`` контроль без модели: множество SetR, добитое до k порядком
             реранкера — та же длина контекста, что у цепочки поверх SetR;
``testpairs`` объясняющая, не критерий: AUC связи на парах из нашего пула
             по разметке теста (эталон — эталон против эталон — не эталон);
             по ней ничего не настраивается.

    python scripts/laya_eval.py heldout --model artifacts/laya/model --data artifacts/laya/data
    python scripts/laya_eval.py gate --model … --bundle artifacts/bench/musique-300 \\
        --rankings …/rankings-ours-current.jsonl --out …/gate-ours-current.jsonl
    python scripts/laya_eval.py chain --model … --bundle … --rankings … --out …/rankings-ours-current+chain.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from laya_data import QUESTIONS, pair_state, set_state  # noqa: E402

THRESHOLD = 0.5
K = 5
EXTRA_FROM_RANKED = 16
NEGATIVES_PER_ANCHOR = 5
GATE_TIMING_N = 20


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_agent(model: str, device: str):
    import laya

    return laya.load(model, device=device)


def probabilities(agent, states: list[dict], task: str, batch_size: int) -> list[float]:
    question = {task: QUESTIONS[task]}
    results = agent.predict_batch(states, question, batch_size=batch_size, sort_by_length=True)
    return [float(result["answers"][task]["noul"]) for result in results]


def auc(scores: list[float], labels: list[int]) -> float:
    """Площадь под ROC через ранги (Манна — Уитни), ничьи — пополам."""
    pairs = sorted(zip(scores, labels))
    pos = sum(labels)
    neg = len(labels) - pos
    if pos == 0 or neg == 0:
        return float("nan")
    rank_sum, i = 0.0, 0
    while i < len(pairs):
        j = i
        while j < len(pairs) and pairs[j][0] == pairs[i][0]:
            j += 1
        mid = (i + j + 1) / 2
        rank_sum += mid * sum(label for _, label in pairs[i:j])
        i = j
    return (rank_sum - pos * (pos + 1) / 2) / (pos * neg)


def cmd_heldout(args) -> int:
    agent = load_agent(args.model, args.device)
    rows = load_jsonl(args.data / "heldout.jsonl")
    report = {"model": args.model}
    for task in ("pair", "enough"):
        part = [row for row in rows if row["task"] == task]
        labels = [int(row["answers"][task]["noul"] > 0.5) for row in part]
        started = time.perf_counter()
        scores = probabilities(agent, [row["state"] for row in part], task, args.batch)
        seconds = time.perf_counter() - started
        accuracy = sum(int((s >= THRESHOLD) == bool(y)) for s, y in zip(scores, labels)) / len(part)
        report[task] = {
            "n": len(part), "positive": sum(labels), "auc": round(auc(scores, labels), 4),
            "accuracy@0.5": round(accuracy, 4),
            "mean_p_pos": round(sum(s for s, y in zip(scores, labels) if y) / max(1, sum(labels)), 4),
            "mean_p_neg": round(sum(s for s, y in zip(scores, labels) if not y)
                                / max(1, len(labels) - sum(labels)), 4),
            "ms_per_decision": round(1000 * seconds / len(part), 2),
        }
        print(f"{task}: {report[task]}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def _bundle(bundle: Path) -> tuple[dict[str, str], dict[str, dict]]:
    texts = {row["chunk_id"]: row["text"] for row in load_jsonl(bundle / "chunks.jsonl")}
    questions = {row["qid"]: row for row in load_jsonl(bundle / "questions.jsonl")}
    return texts, questions


def cmd_gate(args) -> int:
    agent = load_agent(args.model, args.device)
    texts, questions = _bundle(args.bundle)
    rankings = load_jsonl(args.rankings)
    if args.limit:
        rankings = rankings[: args.limit]
    states = [set_state(questions[row["qid"]]["question"], [texts[c] for c in row["ranked"][:K]])
              for row in rankings]
    started = time.perf_counter()
    scores = probabilities(agent, states, "enough", args.batch)
    seconds = time.perf_counter() - started
    # Задержка в работе — по одному вопросу, а не пакетом: она идёт в критерий
    # «быстрее SEAL» у Б1.
    single = []
    for state in states[:GATE_TIMING_N]:
        begun = time.perf_counter()
        probabilities(agent, [state], "enough", 1)
        single.append(1000 * (time.perf_counter() - begun))
    single.sort()
    summary = {"questions": len(scores), "batched_ms_per_question": round(1000 * seconds / len(scores), 1),
               "single_median_ms": round(single[len(single) // 2], 1) if single else None}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"гейт: {summary}", flush=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row, score in zip(rankings, scores):
            gold = set(questions[row["qid"]]["gold_chunk_ids"])
            handle.write(json.dumps({
                "qid": row["qid"], "p_enough": round(score, 6),
                "all_gold_in_top5": gold <= set(row["ranked"][:K]),
            }) + "\n")
    return 0


def fill_order(row: dict, seed: str) -> list[str]:
    """Порядок, которым добиваются свободные места: реранкер или «SetR, затем реранкер»."""
    ranked = list(row["ranked"])
    if seed == "selected":
        chosen = [cid for cid in row.get("selected") or [] if cid]
        return chosen + [cid for cid in ranked if cid not in chosen]
    return ranked


def build_chain(agent, question: str, row: dict, texts: dict[str, str], batch: int,
                seed: str = "top1") -> tuple[list[str], list[float], int]:
    order = fill_order(row, seed)
    candidates = list(dict.fromkeys(list(row.get("pool") or []) + list(row["ranked"])[:EXTRA_FROM_RANKED]))
    if seed == "selected":
        chain = [cid for cid in (row.get("selected") or []) if cid in texts][:K] or order[:1]
    else:
        chain = order[:1]
    start = len(chain)
    chain_p: list[float] = []
    calls = 0
    while len(chain) < K:
        options = [cid for cid in candidates if cid not in chain and cid in texts]
        if not options:
            break
        # top1: от последнего добавленного (правило L1); selected: множество
        # без порядка — кандидат против каждого члена, берётся максимум.
        anchors = chain[-1:] if seed == "top1" else chain
        scores = [0.0] * len(options)
        for anchor in anchors:
            states = [pair_state(question, texts[anchor], texts[cid]) for cid in options]
            got = probabilities(agent, states, "pair", batch)
            calls += len(states)
            scores = [max(a, b) for a, b in zip(scores, got)]
        best = max(range(len(options)), key=lambda i: (scores[i], -i))
        if scores[best] < THRESHOLD:
            break
        chain.append(options[best])
        chain_p.append(round(scores[best], 6))
    rest = [cid for cid in order if cid not in chain]
    return chain + rest, chain_p, calls, start


def cmd_chain(args) -> int:
    agent = load_agent(args.model, args.device)
    texts, questions = _bundle(args.bundle)
    rankings = load_jsonl(args.rankings)
    if args.limit:
        rankings = rankings[: args.limit]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    total_calls, started = 0, time.perf_counter()
    with args.out.open("w", encoding="utf-8") as handle:
        for n, row in enumerate(rankings, 1):
            begun = time.perf_counter()
            ranked, chain_p, calls, start = build_chain(
                agent, questions[row["qid"]]["question"], row, texts, args.batch, args.seed)
            total_calls += calls
            out = dict(row)
            out["ranked"] = ranked
            out["chain"] = ranked[: start + len(chain_p)]
            out.pop("selected", None)  # генератор читает первые k, а не множество SetR
            out["chain_p"] = chain_p
            out["chain_ms"] = round(1000 * (time.perf_counter() - begun), 1)
            out["latency_ms"] = round(float(row.get("latency_ms") or 0.0) + out["chain_ms"], 1)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
            if n % 50 == 0:
                print(f"  {n}/{len(rankings)}", flush=True)
    seconds = time.perf_counter() - started
    print(f"цепочка: {len(rankings)} вопросов, решений {total_calls}, "
          f"{1000 * seconds / max(1, len(rankings)):.0f} мс на вопрос", flush=True)
    return 0


def cmd_setrfill(args) -> int:
    rankings = load_jsonl(args.rankings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rankings:
            out = dict(row)
            out["ranked"] = fill_order(row, "selected")
            out.pop("selected", None)
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"SetR, добитый реранкером: {len(rankings)} вопросов", flush=True)
    return 0


def cmd_testpairs(args) -> int:
    agent = load_agent(args.model, args.device)
    texts, questions = _bundle(args.bundle)
    states, labels = [], []
    rankings = load_jsonl(args.rankings)
    if args.limit:
        rankings = rankings[: args.limit]
    for row in rankings:
        gold = set(questions[row["qid"]]["gold_chunk_ids"])
        candidates = [cid for cid in dict.fromkeys(list(row.get("pool") or [])
                                                   + list(row["ranked"])[:EXTRA_FROM_RANKED]) if cid in texts]
        present = [cid for cid in candidates if cid in gold]
        others = [cid for cid in candidates if cid not in gold][:NEGATIVES_PER_ANCHOR]
        question = questions[row["qid"]]["question"]
        for anchor in present:
            for cid in present:
                if cid != anchor:
                    states.append(pair_state(question, texts[anchor], texts[cid])); labels.append(1)
            for cid in others:
                states.append(pair_state(question, texts[anchor], texts[cid])); labels.append(0)
    scores = probabilities(agent, states, "pair", args.batch)
    report = {"pairs": len(labels), "positive": sum(labels), "auc": round(auc(scores, labels), 4),
              "note": "объясняющая, по разметке теста; ничего по ней не настраивается"}
    print(json.dumps(report, ensure_ascii=False), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--model", required=True)
    common.add_argument("--device", default="cuda")
    common.add_argument("--batch", type=int, default=64)
    common.add_argument("--out", type=Path, required=True)

    held = sub.add_parser("heldout", parents=[common])
    held.add_argument("--data", type=Path, required=True)
    for name in ("gate", "chain", "testpairs"):
        p = sub.add_parser(name, parents=[common])
        p.add_argument("--bundle", type=Path, required=True)
        p.add_argument("--rankings", type=Path, required=True)
        p.add_argument("--limit", type=int, default=0)
        if name == "chain":
            p.add_argument("--seed", choices=("top1", "selected"), default="top1")
    fill = sub.add_parser("setrfill")
    fill.add_argument("--rankings", type=Path, required=True)
    fill.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    return {"heldout": cmd_heldout, "gate": cmd_gate, "chain": cmd_chain,
            "testpairs": cmd_testpairs, "setrfill": cmd_setrfill}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
