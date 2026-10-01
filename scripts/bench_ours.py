"""Наша система на общем наборе этапа 1 (bench_bundle.py) — выдачи для bench_metrics.py.

Две команды:

``index``  фрагменты набора — в отдельную коллекцию Qdrant (векторы и BM25)
           и, с ``--graph``, граф в Neo4j тем же построителем, что у учебника.
           Идентификаторы фрагментов — идентификаторы набора: эталон
           сопоставляется без пересчёта. Нарезка не выполняется: фрагменты
           набора общие для всех систем (иначе сравниваются нарезки, а не графы).
``rank``   для каждого вопроса — ``context.retrieval.retrieve`` и строка
           ``{"qid", "ranked", "pool", …}`` в ``rankings-<система>.jsonl``.

``ranked`` — итоговая выдача (top_k), за ней остальные кандидаты по баллу
реранкера: отборщикам этапа 2 (bench_select.py) нужно окно шире top_k.
``pool`` — кандидаты до реранкера: доступ этапа 1. Поля ``selected``,
``seal_added``, ``status`` пишутся, когда включён отбор (RETRIEVAL_SELECTION).

Конфигурация — переменными окружения, как у всего конвейера; сценарий
``deploy/bench-ours.sh`` задаёт «старую» и «текущую» явно. Файл дописывается
построчно, повторный запуск пропускает готовые вопросы.

    QDRANT_COLLECTION=bench_musique_300 python scripts/bench_ours.py index \\
        --bundle artifacts/bench/musique-300 --graph
    QDRANT_COLLECTION=bench_musique_300 GRAPH_BACKEND=memory GRAPH_FILE=… \\
        python scripts/bench_ours.py rank --bundle artifacts/bench/musique-300 \\
        --out artifacts/runs/bench/musique-300/ours --system ours-current
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.context import build_context  # noqa: E402
from rag_textbook.models import Chunk  # noqa: E402

# Коллекция учебника: сюда чужой корпус писать нельзя (eval public, та же защита).
TEXTBOOK_COLLECTION = "textbook_chunks"


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def to_chunks(rows: list[dict]) -> list[Chunk]:
    return [
        Chunk(
            id=row["chunk_id"], doc_id=str(row.get("doc_id") or row["chunk_id"]),
            doc_name=str(row.get("title") or ""), source_path="", ordinal=index,
            text=row["text"], headers=[row["title"]] if row.get("title") else [],
        )
        for index, row in enumerate(rows)
    ]


def ranked_list(result, top_k: int) -> list[str]:
    """Выдача, затем остальные кандидаты по баллу реранкера, затем без балла."""
    final = [item.chunk.id for item in result.chunks][:top_k]
    seen = set(final)
    scored = sorted(
        ((cid, score) for cid, score in result.rerank_scores.items() if cid not in seen),
        key=lambda pair: -pair[1],
    )
    rest = [cid for cid, _ in scored]
    seen.update(rest)
    tail = [cid for cid in result.pool if cid not in seen]
    return final + rest + tail


def cmd_index(args, settings: Settings) -> int:
    if settings.vector_store.collection == TEXTBOOK_COLLECTION:
        print("QDRANT_COLLECTION — коллекция учебника: чужой корпус испортил бы прежние замеры",
              file=sys.stderr)
        return 1
    chunks = to_chunks(load_jsonl(args.bundle / "chunks.jsonl"))
    print(f"набор {args.bundle.name}: {len(chunks)} фрагментов → {settings.vector_store.collection}"
          f", граф {'да' if args.graph else 'нет'}")
    from rag_textbook.indexing.pipeline import IndexingPipeline

    context = build_context(settings)
    try:
        started = time.perf_counter()
        result = IndexingPipeline(context).index_chunks(
            chunks, source_label=f"bench-{args.bundle.name}", with_graph=args.graph
        )
        result["seconds"] = round(time.perf_counter() - started, 1)
    finally:
        context.close()
    print(json.dumps(result, ensure_ascii=False, default=str))
    if args.report:
        args.report.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=str),
                               encoding="utf-8")
    graph = result.get("граф") or {}
    if args.graph and isinstance(graph, dict) and graph.get("status") in {"disabled", "unavailable"}:
        print(f"граф не построен: {graph.get('status')}", file=sys.stderr)
        return 1
    if args.graph and isinstance(graph, dict):
        # Откат к правилам молча даёт граф из мешка слов без единой связи —
        # так прошёл первый запуск I1 (4657 из 4657 откатов, relations=0).
        statuses = graph.get("extraction_status") or {}
        total = sum(statuses.values()) or 1
        fallback = statuses.get("rule_fallback", 0) / total
        if graph.get("relations", 0) == 0 or fallback > 0.10:
            print(f"извлечение провалено: relations={graph.get('relations')}, "
                  f"откат к правилам {fallback:.1%} (журнал state/extraction_failures.jsonl)",
                  file=sys.stderr)
            return 1
    return 0


def cmd_rank(args, settings: Settings) -> int:
    questions = load_jsonl(args.bundle / "questions.jsonl")
    if args.limit:
        questions = questions[: args.limit]
    known = {row["chunk_id"] for row in load_jsonl(args.bundle / "chunks.jsonl")}
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"rankings-{args.system}.jsonl"
    done = {row["qid"] for row in load_jsonl(path)} if path.exists() else set()
    todo = [q for q in questions if q["qid"] not in done]
    top_k = settings.retrieval.top_k
    print(f"{args.system}: вопросов {len(questions)}, готово {len(done)}, top_k {top_k}, "
          f"граф {settings.graph.backend}:{settings.graph.graph_file or settings.graph.database}, "
          f"отбор {settings.retrieval.selection_mode}")

    context = build_context(settings)
    stats = {"unknown": 0, "graph_routed": 0, "graph_share": 0.0, "n": 0}
    try:
        def one(question: dict) -> dict:
            result = context.retrieval.retrieve(question["question"], history=[])
            ranked = ranked_list(result, top_k)
            row = {
                "qid": question["qid"], "ranked": ranked, "pool": list(result.pool),
                "latency_ms": result.timings_ms.get("total", 0.0),
                "used_graph": bool(result.route and result.route.use_graph),
                "graph_share": round(result.graph_share, 3),
                "graph_only_share": round(result.graph_only_share, 3),
            }
            if result.selection_status:
                row.update(selected=list(result.selected), seal_added=list(result.seal_added),
                           seal_gaps=list(result.seal_gaps), status=result.selection_status)
            return row

        with path.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
            for index, row in enumerate(pool.map(one, todo), start=1):
                stats["unknown"] += sum(1 for cid in row["ranked"] if cid not in known)
                stats["graph_routed"] += int(row["used_graph"])
                stats["graph_share"] += row["graph_share"]
                stats["n"] += 1
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 25 == 0:
                    print(f"  {index}/{len(todo)}", flush=True)
    finally:
        context.close()

    n = max(1, stats["n"])
    summary = {"system": args.system, "questions": stats["n"],
               "graph_routed": round(stats["graph_routed"] / n, 3),
               "graph_share": round(stats["graph_share"] / n, 3),
               "unknown_ids": stats["unknown"]}
    print(json.dumps(summary, ensure_ascii=False))
    # Выдача вне набора — сломано сопоставление (не та коллекция, остатки учебника).
    if stats["unknown"]:
        print("СТОП: в выдаче фрагменты, которых нет в наборе — не та коллекция?", file=sys.stderr)
        return 1
    # Граф включён, но не участвовал — отказ канала, а не отсутствие эффекта.
    if stats["n"] and settings.graph.retrieval_enabled and stats["graph_share"] == 0.0:
        print("СТОП: графовый канал не дал ни одного фрагмента", file=sys.stderr)
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ix = sub.add_parser("index")
    ix.add_argument("--bundle", type=Path, required=True)
    ix.add_argument("--graph", action="store_true", help="строить граф в Neo4j")
    ix.add_argument("--report", type=Path, default=None)
    rk = sub.add_parser("rank")
    rk.add_argument("--bundle", type=Path, required=True)
    rk.add_argument("--out", type=Path, required=True)
    rk.add_argument("--system", required=True)
    rk.add_argument("--workers", type=int, default=8)
    rk.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    settings = Settings()
    return cmd_index(args, settings) if args.command == "index" else cmd_rank(args, settings)


if __name__ == "__main__":
    raise SystemExit(main())
