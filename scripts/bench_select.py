"""Отбор множеством (SetR, Context-Picker) поверх выдачи любой системы на бенче.

Этап 2 сравнивается поверх двух систем этапа 1 — нашей текущей и HippoRAG 2,
поэтому отборщик работает с их общим форматом (bench_metrics.py), а не
с внутренностями конвейера. Вход — ``rankings-<система>.jsonl``, выход —
``rankings-<система>+<режим>.jsonl`` того же формата: выбранное первым,
остаток выдачи за ним в прежнем порядке, ``pool`` без изменений. Поэтому
recall@k и all@k считаются тем же bench_metrics.py, а размер и полнота самого
множества — здесь, по полю ``selected``.

Подсказки и разбор — те же функции, что в конвейере
(``rag_textbook.retrieval.set_selection``); модель — ``LLM_*`` из окружения,
назначение utility. Файл дописывается построчно, повторный запуск пропускает
готовые вопросы: обрыв связи не стоит пересчёта.

    python scripts/bench_select.py --bundle artifacts/bench/musique-300 \\
        --rankings runs/bench/rankings-hipporag2.jsonl --mode setr --workers 8
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_textbook.clients.llm import build_llm_client
from rag_textbook.config import Settings
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval import set_selection


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def as_scored(row: dict) -> ScoredChunk:
    chunk = Chunk(
        id=row["chunk_id"], doc_id=str(row.get("doc_id") or ""), doc_name=str(row.get("title") or ""),
        source_path="", ordinal=0, text=row["text"],
    )
    return ScoredChunk(chunk=chunk)


def set_summary(rows: list[dict], gold: dict[str, list[str]]) -> dict:
    """Размер множества, доля вопросов со всем эталоном в нём, статусы."""
    n = len(rows)
    statuses = Counter(row["status"] for row in rows)
    chosen = [row for row in rows if row["selected"]]
    full = [
        float(set(gold.get(row["qid"], [])) <= set(row["selected"])) if row["selected"] else 0.0
        for row in rows
    ]
    recall = [
        len(set(gold[row["qid"]]) & set(row["selected"])) / len(gold[row["qid"]])
        for row in rows
        if gold.get(row["qid"])
    ]
    return {
        "n": n,
        "status": {name: count / n for name, count in sorted(statuses.items())},
        "set_size": statistics.fmean(len(row["selected"]) for row in chosen) if chosen else 0.0,
        "set_recall": statistics.fmean(recall) if recall else 0.0,
        "set_all": statistics.fmean(full) if full else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--rankings", type=Path, required=True)
    parser.add_argument("--mode", choices=set_selection.MODES, required=True)
    parser.add_argument("--pool", type=int, default=20, help="сколько верхних кандидатов видит модель")
    parser.add_argument("--chars", type=int, default=2400, help="предел знаков на фрагмент")
    parser.add_argument("--top-k", type=int, default=16, help="предел размера множества")
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="только первые N вопросов (проба)")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    settings = Settings()
    retrieval = settings.retrieval.model_copy(update={
        "selection_mode": args.mode,
        "selection_llm_pool": args.pool,
        "selection_llm_chars": args.chars,
        "selection_llm_max_tokens": args.max_tokens,
    })
    llm = build_llm_client(settings.llm)

    chunks = {row["chunk_id"]: row for row in load_jsonl(args.bundle / "chunks.jsonl")}
    questions = {row["qid"]: row for row in load_jsonl(args.bundle / "questions.jsonl")}
    rankings = load_jsonl(args.rankings)
    if args.limit:
        rankings = rankings[: args.limit]
    missing = {cid for row in rankings for cid in row["ranked"][: args.pool] if cid not in chunks}
    if missing:
        raise SystemExit(f"В выдаче {len(missing)} фрагментов, которых нет в наборе: не тот набор?")

    system = args.rankings.stem.removeprefix("rankings-")
    out = args.out or args.rankings.with_name(f"rankings-{system}+{args.mode}.jsonl")
    done = {row["qid"] for row in load_jsonl(out)} if out.exists() else set()
    todo = [row for row in rankings if row["qid"] not in done]
    print(f"{system}+{args.mode}: вопросов {len(rankings)}, готово {len(done)}, "
          f"модель {settings.llm.model_for('utility')}")

    def one(row: dict) -> dict:
        items = [as_scored(chunks[cid]) for cid in row["ranked"] if cid in chunks]
        outcome = set_selection.select_set(
            items, questions[row["qid"]]["question"], llm, retrieval, args.top_k, mode=args.mode
        )
        return {
            "qid": row["qid"],
            "ranked": [item.chunk.id for item in outcome.ordered],
            "pool": row.get("pool") or row["ranked"],
            "selected": outcome.chosen,
            "status": outcome.status,
        }

    with out.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
        for index, result in enumerate(pool.map(one, todo), start=1):
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            handle.flush()
            if index % 25 == 0:
                print(f"  {index}/{len(todo)}")

    rows = load_jsonl(out)
    gold = {qid: row["gold_chunk_ids"] for qid, row in questions.items()}
    summary = set_summary(rows, gold)
    out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))
    if summary["status"].get("fallback", 0.0) > 0.1:
        print("ВНИМАНИЕ: больше 10% ответов без строки выбора — число меряет отказы, а не метод")


if __name__ == "__main__":
    main()
