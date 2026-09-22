"""Проверка стенда: совпадает ли графовый канал по файлу с каналом по Neo4j.

    # по слепку, снятому на том же графе (можно на ноутбуке):
    python scripts/graph_fidelity.py --graph-file current.json.gz --trace trace.jsonl

    # и напрямую с живым Neo4j (на сервере):
    python scripts/graph_fidelity.py --graph-file current.json.gz --trace trace.jsonl --live

Правило допуска записано до проверки (docs/HYPOTHESES.md, серия К): стенд
годится, если у не меньше 99% вопросов канал по файлу возвращает то же
множество фрагментов, что и Neo4j. Иначе стенд мерит другую систему,
и гипотезы К1–К5 на нём не ставятся.

Входы канала берутся из слепка: переписанный вопрос и опорные фрагменты —
верх векторной выдачи. Так сравнивается только сам граф, а не вся цепочка.
Поэтому слепок обязан быть снят на том же графе, что выгружен в файл;
иначе расхождение покажет разницу графов, а не разницу хранилищ.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.evaluation.trace import TraceSet, align_to_snapshot  # noqa: E402
from rag_textbook.retrieval.graph_retriever import GraphRetriever  # noqa: E402
from rag_textbook.stores.graph_file import MemoryGraphStore, file_hash  # noqa: E402

THRESHOLD = 0.99


def compare(reference: list[str], candidate: list[str]) -> dict[str, Any]:
    left, right = set(reference), set(candidate)
    union = left | right
    return {
        "same_list": reference == candidate,
        "same_set": left == right,
        "jaccard": len(left & right) / len(union) if union else 1.0,
        "top10_overlap": len(set(reference[:10]) & set(candidate[:10])) / max(1, min(10, len(reference))),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"questions": 0}
    return {
        "questions": len(rows),
        "same_set_share": sum(row["same_set"] for row in rows) / len(rows),
        "same_list_share": sum(row["same_list"] for row in rows) / len(rows),
        "mean_jaccard": mean(row["jaccard"] for row in rows),
        "mean_top10_overlap": mean(row["top10_overlap"] for row in rows),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph-file", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--live", action="store_true", help="сравнить ещё и с живым Neo4j")
    parser.add_argument("--json", type=Path, help="куда записать отчёт")
    parser.add_argument("--examples", type=int, default=5, help="сколько расхождений показать")
    args = parser.parse_args(argv)

    traces = TraceSet.load(args.trace)
    # Настройки обхода берутся из слепка: на сервере они могли отличаться
    # от значений по умолчанию, и тогда сравнение шло бы с другим обходом.
    settings, changes = align_to_snapshot(Settings(), traces.settings_snapshot)
    # Канал по файлу включается явно: без пароля Neo4j ``Settings`` выключает
    # графовый слой, а здесь Neo4j для канала по файлу не нужен.
    graph_settings = settings.graph.model_copy(
        update={
            "backend": "memory",
            "graph_file": args.graph_file,
            "ranker": "walk",
            "enabled": True,
            "retrieval_enabled": True,
        }
    )
    memory = GraphRetriever(graph_settings, MemoryGraphStore.from_file(args.graph_file))

    live = None
    if args.live:
        from rag_textbook.stores.graph_store import GraphStore

        live = GraphRetriever(settings.graph, GraphStore(settings.graph))

    against_trace: list[dict[str, Any]] = []
    against_live: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    skipped = 0
    for trace in traces.traces:
        recorded = [item.chunk_id for item in trace.channels.get("graph", [])]
        if not trace.used_graph:
            skipped += 1
            continue
        question = trace.rewritten_question or trace.question
        seeds = [item.chunk_id for item in trace.channels.get("base", [])][: settings.graph.seed_passages]
        produced = [item.chunk.id for item in memory.retrieve(question, seed_chunk_ids=seeds)]

        row = {"question_id": trace.question_id, **compare(recorded, produced)}
        against_trace.append(row)
        if not row["same_set"] and len(examples) < args.examples:
            examples.append(
                {
                    "question_id": trace.question_id,
                    "only_neo4j": sorted(set(recorded) - set(produced))[:5],
                    "only_file": sorted(set(produced) - set(recorded))[:5],
                    "jaccard": round(row["jaccard"], 3),
                }
            )
        if live is not None:
            reference = [item.chunk.id for item in live.retrieve(question, seed_chunk_ids=seeds)]
            against_live.append({"question_id": trace.question_id, **compare(reference, produced)})

    report = {
        "graph_file": str(args.graph_file),
        "graph_sha256": file_hash(args.graph_file),
        "trace": str(args.trace),
        "settings_aligned": changes,
        "skipped_without_graph": skipped,
        "threshold_same_set": THRESHOLD,
        "against_trace": summarize(against_trace),
        "against_live": summarize(against_live) if live is not None else None,
        "examples": examples,
    }
    decisive = report["against_live"] or report["against_trace"]
    passed = decisive.get("questions", 0) > 0 and decisive["same_set_share"] >= THRESHOLD
    report["verdict"] = "годен" if passed else "НЕ ГОДЕН"
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(text, encoding="utf-8")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
