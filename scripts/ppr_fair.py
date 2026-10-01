"""Честное сравнение: PPR против обхода на один шаг на ОДНОМ офлайн-графе.

Codex сравнил PPR (офлайн-граф из кэша, 47 пропусков) с каналом graph
из слепка (рабочий граф Neo4j). Разница графов и затравок смешана с
разницей алгоритмов. Здесь оба алгоритма получают один граф и одни затравки.
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
import tempfile
from collections import defaultdict
from contextlib import closing
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from rag_textbook.evaluation.graph_offline import PPRGraph, reconstruct  # noqa: E402
from rag_textbook.evaluation.metrics import paired_bootstrap  # noqa: E402
from scripts.ppr_offline import normalized, question_metrics, question_seeds  # noqa: E402


def one_hop(graph, seeds, hop_decay=0.5):
    """Обход на один шаг, как rank_from_passage, но от набора затравок."""
    weights: dict[str, float] = {}
    for (kind, node), _ in seeds.items():
        if kind == "entity":
            weights[node] = 1.0
        else:
            for entity in graph.mentions.get(node, {}):
                weights[entity] = 1.0
    own = dict(weights)
    for entity in own:
        for nb in graph.neighbours.get(entity, ()):
            if nb not in own:
                weights[nb] = max(weights.get(nb, 0.0), hop_decay)
    scores: dict[str, float] = defaultdict(float)
    for entity, w in weights.items():
        for chunk in graph.chunks_of_entity.get(entity, ()):
            scores[chunk] += w * math.log(1 + graph.mentions[chunk].get(entity, 1))
    for chunk in scores:
        scores[chunk] /= math.sqrt(max(1, len(graph.mentions.get(chunk, {}))))
    for (kind, node) in seeds:
        if kind == "passage":
            scores.pop(node, None)
    return sorted(scores, key=lambda c: (-scores[c], c))


def main() -> None:
    settings = Settings()
    root = Path(__file__).resolve().parents[1]
    trace = root / "capture/session-0819/trace-always.jsonl"
    records = [json.loads(x) for x in trace.read_text(encoding="utf-8").splitlines()]
    snap, rows = records[0]["settings_snapshot"], records[1:]
    with tempfile.TemporaryDirectory() as d:
        src = settings.paths.cache_dir / "extraction.sqlite3"
        with closing(sqlite3.connect(src.resolve().as_uri() + "?mode=ro", uri=True)) as s, \
             closing(sqlite3.connect(str(Path(d) / src.name))) as t:
            s.backup(t)
        settings.paths.cache_dir = Path(d)
        graph = reconstruct(settings, model=snap["llm.model"], reasoning_effort="none",
                            max_entity_degree=snap["graph.max_entity_degree"])
    gold = {q.id: set(q.gold_chunk_ids)
            for q in load_goldset(root / "evaluation/goldsets/goldset.json")}
    names = {e: normalized(n) for e, n in graph.names.items()
             if e in graph.chunks_of_entity or e in graph.neighbours}
    k = snap["graph.seed_passages"]
    seeds = {r["question_id"]: question_seeds(r, names, k) for r in rows}

    trace_orders = {r["question_id"]: [v["chunk_id"] for v in sorted(r["channels"]["graph"], key=lambda v: v["rank"])] for r in rows}
    hop_orders = {q: one_hop(graph, s) for q, s in seeds.items()}
    ppr = PPRGraph(graph, entity_weight=1.0)
    ppr_orders = {q: [c for c, _ in ppr.rank(s, alpha=0.3)] if s else [] for q, s in seeds.items()}

    res = {name: question_metrics(rows, o, gold)
           for name, o in (("trace", trace_orders), ("hop_offline", hop_orders), ("ppr_0.3_1.0", ppr_orders))}
    for name, out in res.items():
        print(f"{name:12s} recall@30={sum(v['recall@30'] for v in out)/len(out):.4f} "
              f"gold={sum(v['gold_found'] for v in out)} graph_only={sum(v['graph_only_found'] for v in out)}")
    for a, b in (("ppr_0.3_1.0", "hop_offline"), ("hop_offline", "trace")):
        for m in ("recall@30", "graph_only_found"):
            diffs = [x[m] - y[m] for x, y in zip(res[a], res[b], strict=True)]
            print(f"{a} - {b} {m}: {paired_bootstrap(diffs)}")


if __name__ == "__main__":
    main()
