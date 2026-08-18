"""Где теряется недостающий эталонный фрагмент.

Гипотеза П1 предполагала, что второй нужный фрагмент лежит в пуле сразу
под отсечкой, и разнообразие выдачи его достанет. Критерий отказа был назван
заранее: если медианный ранг недостающего фрагмента в пуле выше 40,
разнообразие его не достанет.

Перебор показал, что разнообразие не помогает. Этот скрипт отвечает почему —
и отвечает по слепку, то есть по тем самым кандидатам, которые видел сервер.

    python scripts/analyze_misses.py capture/trace.jsonl capture/goldset.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

from rag_textbook.evaluation.trace import TraceSet
from rag_textbook.models import GoldQuestion


def main() -> int:
    trace_path = Path(sys.argv[1] if len(sys.argv) > 1 else "capture/trace.jsonl")
    goldset_path = Path(sys.argv[2] if len(sys.argv) > 2 else "capture/goldset.json")

    traces = TraceSet.load(trace_path)
    raw = json.loads(goldset_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("questions", [])
    gold = {
        item["id"]: GoldQuestion.model_validate(item) for item in items
    }

    verdicts_path = Path("evaluation/goldsets/verdicts.json")
    verdicts: dict[str, str] = {}
    if verdicts_path.exists():
        payload = json.loads(verdicts_path.read_text(encoding="utf-8"))
        for entry in payload.get("verdicts", []):
            verdicts[entry["question_id"]] = entry["verdict"]

    ranks: list[int] = []
    outside_pool = 0
    misses = 0
    by_verdict: Counter[str] = Counter()

    for trace in traces.traces:
        question = gold.get(trace.question_id)
        if question is None or question.question_type != "graph_linked":
            continue
        found = set(trace.final)
        missing = [item for item in question.gold_chunk_ids if item not in found]
        if not missing:
            continue
        misses += 1
        by_verdict[verdicts.get(trace.question_id, "не проверялся")] += 1

        # Ранг в объединённом пуле кандидатов: если фрагмента там нет вовсе,
        # никакой перестановкой его не достать — дело не в отборе, а в доступе.
        pool = trace.candidate_ids()
        for chunk_id in missing:
            if chunk_id in pool:
                ranks.append(pool.index(chunk_id))
            else:
                outside_pool += 1

    print(f"связывающих вопросов с промахом: {misses}")
    print(f"недостающих фрагментов вне пула кандидатов: {outside_pool}")
    print(f"недостающих фрагментов в пуле: {len(ranks)}")

    if ranks:
        ranks.sort()
        print(f"\nранг недостающего фрагмента в пуле:")
        print(f"  медиана   {statistics.median(ranks):.0f}")
        print(f"  четверти  {ranks[len(ranks) // 4]} / {ranks[3 * len(ranks) // 4]}")
        print(f"  минимум   {ranks[0]}, максимум {ranks[-1]}")
        for bound in (8, 16, 30, 40, 60):
            share = sum(1 for value in ranks if value < bound) / len(ranks)
            print(f"  рангом ниже {bound:>3}: {share:.1%}")

    print("\nпромахи по вердикту ручной проверки:")
    for verdict, count in by_verdict.most_common():
        print(f"  {verdict:<20} {count}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
