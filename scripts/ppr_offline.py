"""Офлайн-сравнение PPR со слепком; запуск: python -m scripts.ppr_offline.

CACHE_DIR и PARSED_DIR указывают на исходные данные. Перед запуском задайте
RAG_ENV_FILE=tests-no-such-env-file. SQLite читается через mode=ro и backup
в TEMP: существующий reconstruct работает только с этой отдельной копией.
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import statistics
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import closing
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset
from rag_textbook.evaluation.graph_offline import (
    NOT_FOUND,
    OfflineGraph,
    PPRGraph,
    PPRNode,
    linked_pairs,
    reconstruct,
    second_hop_ranks,
    summarize,
)
from rag_textbook.evaluation.metrics import paired_bootstrap
from rag_textbook.utils.text import lemmatize_token, tokenize


def normalized(text: str) -> str:
    """Границы слов предотвращают совпадение сущности внутри другого слова."""
    return " " + " ".join(lemmatize_token(token) for token in tokenize(text)) + " "


def question_seeds(
    row: dict[str, Any], names: dict[str, str], k: int,
) -> dict[PPRNode, float]:
    seeds: dict[PPRNode, float] = {}
    for item in sorted(row["channels"]["base"], key=lambda item: item["rank"])[:k]:
        score = float(item["score"])
        if score > 0:
            seeds[("passage", item["chunk_id"])] = score
    question = normalized(row["question"])
    for entity, name in names.items():
        if name.strip() and name in question:
            seeds[("entity", entity)] = 1.0
    return seeds


def question_metrics(
    rows: list[dict[str, Any]], orders: dict[str, list[str]],
    gold: dict[str, set[str]],
) -> list[dict[str, Any]]:
    outcomes = []
    for row in rows:
        qid = row["question_id"]
        expected = gold[qid]
        base = {item["chunk_id"] for item in row["channels"]["base"]}
        found = set(orders[qid][:30]) & expected
        outcomes.append({
            "question_id": qid, "question_type": row["question_type"],
            "recall@30": len(found) / len(expected),
            "precision@30": len(found) / 30,
            "gold_found": len(found), "graph_only_found": len(found - base),
            "graph_only_total": len(expected - base),
        })
    return outcomes


def aggregate(outcomes: list[dict[str, Any]]) -> dict[str, Any]:
    def group(values: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(values),
            "recall@30": statistics.fmean(v["recall@30"] for v in values),
            "precision@30": statistics.fmean(v["precision@30"] for v in values),
            **{key: sum(v[key] for v in values) for key in (
                "gold_found", "graph_only_found", "graph_only_total",
            )},
        }
    return {
        "all": group(outcomes),
        "by_type": {
            kind: group([v for v in outcomes if v["question_type"] == kind])
            for kind in sorted({v["question_type"] for v in outcomes})
        },
    }


def compare(
    candidate: list[dict[str, Any]], baseline: list[dict[str, Any]],
) -> dict[str, Any]:
    if [v["question_id"] for v in candidate] != [v["question_id"] for v in baseline]:
        raise ValueError("Парный бутстрап требует одинакового порядка вопросов")
    return {
        metric: paired_bootstrap([
            a[metric] - b[metric] for a, b in zip(candidate, baseline, strict=True)
        ])
        for metric in ("recall@30", "precision@30", "graph_only_found")
    }


def evaluate_variant(
    graph: OfflineGraph, pairs: list[tuple[str, str]], rows: list[dict[str, Any]],
    gold: dict[str, set[str]], seeds: dict[str, dict[PPRNode, float]],
    baseline: list[dict[str, Any]], weight: float, alpha: float, use_idf: bool,
) -> dict[str, Any]:
    """Варианты независимы, поэтому процессы сокращают время полного перебора."""
    ppr = PPRGraph(graph, entity_weight=weight, use_idf=use_idf)
    ranks = ppr_second_hop_ranks(ppr, pairs, alpha)
    orders = {
        qid: [chunk for chunk, _ in ppr.rank(s, alpha=alpha)] if s else []
        for qid, s in seeds.items()
    }
    outcomes = question_metrics(rows, orders, gold)
    return {
        "alpha": alpha, "entity_weight": weight, "hops": summarize(ranks),
        "questions": aggregate(outcomes), "paired_difference": compare(outcomes, baseline),
        "outcomes": outcomes,
    }


def run_worker(path: Path) -> dict[str, Any]:
    """Файловый обмен работает и в Windows-песочнице без именованных каналов."""
    with path.with_suffix(".log").open("w", encoding="utf-8") as log:
        subprocess.run(
            [sys.executable, "-m", "scripts.ppr_offline", "--worker-input", str(path)],
            stdout=log, stderr=subprocess.STDOUT, check=True,
        )
    return json.loads(path.with_suffix(".result.json").read_text(encoding="utf-8"))


def run(args: argparse.Namespace) -> dict[str, Any]:
    settings = Settings()
    records = [json.loads(line) for line in args.trace.read_text(encoding="utf-8").splitlines()]
    header = records[0]
    rows = records[1:]
    if not rows or len({r["question_id"] for r in rows}) != len(rows):
        raise ValueError("Слепок пуст или содержит повторные вопросы")
    snapshot = header["settings_snapshot"]
    k = args.k if args.k is not None else snapshot["graph.seed_passages"]
    if k < 1:
        raise ValueError("k должен быть положительным")
    model = args.model or snapshot["llm.model"]
    with tempfile.TemporaryDirectory(prefix="cache011-") as directory:
        source = settings.paths.cache_dir / "extraction.sqlite3"
        # backup учитывает WAL и даёт согласованный снимок без изменения источника.
        with (
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(str(Path(directory) / source.name))) as dst,
        ):
            src.backup(dst)
        settings.paths.cache_dir = Path(directory)
        graph = reconstruct(
            settings, model=model, reasoning_effort=args.effort,
            max_entity_degree=snapshot["graph.max_entity_degree"],
        )
    gold = {q.id: set(q.gold_chunk_ids) for q in load_goldset(settings.paths.goldset_dir / "goldset.json")}
    if any(row["question_id"] not in gold or not gold[row["question_id"]] for row in rows):
        raise ValueError("Для вопроса слепка нет непустого эталона")
    pairs = linked_pairs(settings, graph)
    names = {entity: normalized(name) for entity, name in graph.names.items()
             if entity in graph.chunks_of_entity or entity in graph.neighbours}
    seeds = {r["question_id"]: question_seeds(r, names, k) for r in rows}
    baseline = question_metrics(rows, {
        r["question_id"]: [v["chunk_id"] for v in sorted(
            r["channels"]["graph"], key=lambda v: v["rank"],
        )] for r in rows
    }, gold)
    result: dict[str, Any] = {
        "protocol": {
            "trace": str(args.trace), "k": k, "model": model, "effort": args.effort,
            "use_idf": args.idf, "alpha_is_restart": True, "entity_seed_weight": 1.0,
            "tolerance": 1e-10, "max_iterations": 200, "bootstrap_resamples": 10000,
            "bootstrap_seed": 20260815, "selection": "graph_only_found, then recall@30",
            "seedless_questions": sum(not v for v in seeds.values()),
            "questions_with_entity_seeds": sum(any(n[0] == "entity" for n in s) for s in seeds.values()),
            "unknown_passage_seeds": sum(n[0] == "passage" and n[1] not in graph.mentions
                                         for s in seeds.values() for n in s),
        },
        "graph": graph.as_dict(), "pairs": len(pairs),
        "baseline_hops": summarize(second_hop_ranks(graph, pairs, hop_decay=0.5, use_idf=False)),
        "baseline_questions": aggregate(baseline), "baseline_outcomes": baseline,
        "variants": [],
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"graph": result["graph"], "pairs": len(pairs), "questions": len(rows)}, ensure_ascii=False), flush=True)
    with (
        tempfile.TemporaryDirectory(prefix="ppr011-workers-") as directory,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        payload = {
            "graph": asdict(graph), "pairs": pairs, "rows": rows, "gold": gold,
            "seeds": {qid: [[list(node), weight] for node, weight in values.items()]
                      for qid, values in seeds.items()},
            "baseline": baseline, "use_idf": args.idf,
        }
        futures = []
        for weight in (0.0, 0.5, 1.0):
            for alpha in (0.3, 0.5, 0.7, 0.85):
                path = Path(directory) / f"{weight}-{alpha}.json"
                path.write_text(json.dumps({**payload, "weight": weight, "alpha": alpha}, default=list), encoding="utf-8")
                futures.append(pool.submit(run_worker, path))
        for future in as_completed(futures):
            variant = future.result()
            result["variants"].append(variant)
            result["variants"].sort(key=lambda v: (v["entity_weight"], v["alpha"]))
            args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps({key: variant[key] for key in ("alpha", "entity_weight", "hops", "questions")}, ensure_ascii=False), flush=True)
    best = max(result["variants"], key=lambda v: (
        v["questions"]["all"]["graph_only_found"], v["questions"]["all"]["recall@30"],
    ))
    old, new = result["baseline_questions"]["all"], best["questions"]["all"]
    result["decision"] = {
        "best_alpha": best["alpha"], "best_entity_weight": best["entity_weight"],
        "reject": new["graph_only_found"] <= old["graph_only_found"] and new["recall@30"] < old["recall@30"],
        "limitation": "Выбор лучшего на тех же вопросах; CI без поправки на перебор. Офлайн не подтверждает эффект на сервере.",
    }
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def ppr_second_hop_ranks(
    ppr: PPRGraph, pairs: list[tuple[str, str]], alpha: float,
) -> list[int]:
    cached: dict[str, dict[str, int]] = {}
    ranks = []
    for left, right in pairs:
        for anchor, target in ((left, right), (right, left)):
            if anchor not in cached:
                cached[anchor] = {chunk: i for i, (chunk, _) in enumerate(
                    ppr.rank({("passage", anchor): 1.0}, alpha=alpha), start=1,
                )}
            ranks.append(cached[anchor].get(target, NOT_FOUND))
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, default=Path("capture/session-0819/trace-always.jsonl"))
    parser.add_argument("--output", type=Path, default=Path(tempfile.gettempdir()) / "ppr_offline.json")
    parser.add_argument("--k", type=int)
    parser.add_argument("--model")
    parser.add_argument("--effort", default="none")
    parser.add_argument("--idf", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--worker-input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers должен быть положительным")
    # Результаты разрешены только во временной папке, включая пользовательский путь.
    if not args.output.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
        parser.error("--output должен находиться в TEMP")
    if os.environ.get("RAG_ENV_FILE") != "tests-no-such-env-file":
        parser.error("Перед запуском задайте RAG_ENV_FILE=tests-no-such-env-file")
    if args.worker_input:
        if not args.worker_input.resolve().is_relative_to(Path(tempfile.gettempdir()).resolve()):
            parser.error("Вход рабочего процесса должен находиться в TEMP")
        payload = json.loads(args.worker_input.read_text(encoding="utf-8"))
        payload["graph"] = OfflineGraph(**payload["graph"])
        payload["gold"] = {qid: set(ids) for qid, ids in payload["gold"].items()}
        payload["seeds"] = {qid: {tuple(node): w for node, w in values}
                            for qid, values in payload["seeds"].items()}
        result = evaluate_variant(**payload)
        args.worker_input.with_suffix(".result.json").write_text(
            json.dumps(result, ensure_ascii=False), encoding="utf-8",
        )
        return
    run(args)


if __name__ == "__main__":
    main()
