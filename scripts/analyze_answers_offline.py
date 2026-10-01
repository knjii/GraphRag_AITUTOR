"""Разбор сохранённого прогона ответов, без обращения к серверу.

Отвечает на три вопроса, ради которых иначе пришлось бы арендовать карту:

1. Что мы на самом деле мерили. Прежний прогон считал качество по тексту,
   в котором ответа могло не быть вовсе: модель отдавала поток размышлений.
   Здесь пересчитываются признаки «размышление вместо ответа» и «ответ
   не по-русски» по уже сохранённым текстам.
2. Помогает ли место эталонного фрагмента в контексте. Гипотеза А3 говорит,
   что модель хуже использует середину длинного контекста. Проверяется
   сопоставлением судейской верности с позицией эталонного фрагмента
   в итоговой выдаче из слепка.
3. Различает ли что-нибудь новая мера опоры на контекст — там, где прежняя
   давала 0.98 во всех прогонах.

    python scripts/analyze_answers_offline.py capture/session-0819/answers_current.json
"""

from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_textbook.evaluation.answers import latin_share, looks_like_reasoning  # noqa: E402

TRACE = Path("capture/trace.jsonl")
GOLDSET = Path("capture/goldset.json")


def load_outcomes(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload["outcomes"] if isinstance(payload, dict) else payload


def load_final_positions() -> dict[str, list[str]]:
    if not TRACE.exists():
        return {}
    out: dict[str, list[str]] = {}
    for line in TRACE.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") == "trace-header":
            continue
        out[record["question_id"]] = list(record.get("final", []))
    return out


def load_gold() -> dict[str, list[str]]:
    if not GOLDSET.exists():
        return {}
    payload = json.loads(GOLDSET.read_text(encoding="utf-8"))
    questions = payload["questions"] if isinstance(payload, dict) else payload
    return {item["id"]: list(item.get("gold_chunk_ids", [])) for item in questions}


def share(items: list[bool]) -> float:
    return round(sum(items) / len(items), 4) if items else 0.0


def main() -> None:
    path = Path(sys.argv[1] if len(sys.argv) > 1 else "capture/session-0819/answers_current.json")
    outcomes = load_outcomes(path)
    print(f"Файл: {path}, вопросов {len(outcomes)}\n")

    # ------------------------------------------------ 1. что мерили на самом деле
    leaks = [looks_like_reasoning(item.get("answer", "")) for item in outcomes]
    latin = [latin_share(item.get("answer", "")) > 0.5 for item in outcomes]
    print("Ответ ли это вообще")
    print(f"  размышление вместо ответа  {share(leaks)}")
    print(f"  ответ не по-русски         {share(latin)}")

    # Разбор по существу: чем именно был текст, по которому считались метрики.
    kinds: dict[str, list[dict]] = defaultdict(list)
    for item in outcomes:
        text = item.get("answer", "")
        if "Не удалось получить ответ модели" in text:
            kind = "отказ инференса"
        elif looks_like_reasoning(text):
            kind = "размышление"
        elif not text.strip():
            kind = "пусто"
        elif item.get("refused"):
            kind = "отказ по контексту"
        else:
            kind = "ответ"
        kinds[kind].append(item)
    print(chr(10) + "  чем был текст:")
    for kind, items in sorted(kinds.items(), key=lambda pair: -len(pair[1])):
        graded = [i["correctness"] for i in items if i.get("correctness") is not None]
        tail = f", верность {statistics.fmean(graded):.3f}" if graded else ""
        print(f"    {kind:<18} {len(items):>4} ({len(items) / len(outcomes):.1%}){tail}")

    print(chr(10) + "  доля размышлений по типам вопросов:")
    by_type: dict[str, list[dict]] = defaultdict(list)
    for item in outcomes:
        by_type[item.get("question_type", "?")].append(item)
    for name, items in sorted(by_type.items()):
        flags = [looks_like_reasoning(i.get("answer", "")) for i in items]
        print(f"    {name:<14} {share(flags)} ({len(items)} вопросов)")

    # Судейские числа считались по этим же текстам, поэтому важно, как они
    # соотносятся: если у размышлений верность не ниже, судья мерил не ответ.
    judged = [i for i in outcomes if i.get("correctness") is not None]
    good = [i["correctness"] for i in judged if not looks_like_reasoning(i.get("answer", ""))]
    bad = [i["correctness"] for i in judged if looks_like_reasoning(i.get("answer", ""))]
    if good and bad:
        print(
            chr(10) + f"  верность: у текстов без размышления {statistics.fmean(good):.3f} "
            f"({len(good)}), у размышлений {statistics.fmean(bad):.3f} ({len(bad)}) — "
            f"судья оценил размышление выше."
        )

    # ------------------------------------- 2. верность против места фрагмента
    finals, gold = load_final_positions(), load_gold()
    if finals and gold and judged:
        buckets: dict[str, list[int]] = defaultdict(list)
        for item in judged:
            order = finals.get(item["question_id"], [])
            wanted = set(gold.get(item["question_id"], []))
            positions = [i for i, chunk in enumerate(order) if chunk in wanted]
            if not positions:
                buckets["эталона нет в контексте"].append(item["correctness"])
                continue
            first = positions[0]
            key = "1-2" if first < 2 else "3-6" if first < 6 else "7 и дальше"
            buckets[key].append(item["correctness"])
        print("\nВерность против места эталонного фрагмента в контексте")
        for key in ["1-2", "3-6", "7 и дальше", "эталона нет в контексте"]:
            values = buckets.get(key, [])
            if values:
                print(f"  {key:<24} {statistics.fmean(values):.3f} ({len(values)} вопросов)")

    # --------------------------------------------- 3. прежняя мера выдумки
    unsupported = [item.get("unsupported", 0.0) for item in outcomes]
    if unsupported:
        print(
            f"\nПрежняя мера выдумки: среднее {statistics.fmean(unsupported):.4f}, "
            f"разброс {min(unsupported):.4f}-{max(unsupported):.4f}"
        )
        print("  Столько же во всех прогонах — различать конфигурации ею нельзя.")


if __name__ == "__main__":
    main()
