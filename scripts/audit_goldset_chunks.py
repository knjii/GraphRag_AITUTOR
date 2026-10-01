"""Какие фрагменты попадают в эталон и годятся ли они на эту роль.

Ручная проверка двадцати связывающих вопросов показала повторяющийся изъян:
вторым «эталонным» фрагментом часто оказывается оглавление или страница
упражнений. Такие фрагменты делят с вопросом лексику, поэтому отбор пар
по общим сущностям их охотно выбирает, но содержания в них нет — ответить
по ним нельзя, и вопрос становится одношаговым при разметке «двухшаговый».

Ручная выборка говорит, что изъян есть; этот скрипт говорит, насколько
он распространён — по всему набору сразу.

    python scripts/audit_goldset_chunks.py
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

from rag_textbook.models import Chunk, GoldQuestion

# Оглавление опознаётся по «отточию» — цепочке точек перед номером страницы.
# Признак устойчивее, чем номер страницы: оглавление встречается и в середине
# книги, перед началом частей.
TOC_LEADER = re.compile(r"\.{2,}\s*\d{1,3}")

# Упражнения опознаются по заголовку: у MinerU он выделен, и это надёжнее,
# чем угадывать по тексту.
EXERCISE_HEADERS = ("УПРАЖНЕНИЯ", "УПРАЖНЕНИЕ", "ЗАДАЧИ")


def is_toc(chunk: Chunk) -> bool:
    """Оглавление: много строк вида «... 123» на небольшой текст."""
    hits = len(TOC_LEADER.findall(chunk.text))
    return hits >= 8


def is_exercise(chunk: Chunk) -> bool:
    return any(
        str(header).upper().startswith(EXERCISE_HEADERS)
        for header in (chunk.headers or [])
    )


def classify(chunk: Chunk) -> str:
    if is_toc(chunk):
        return "оглавление"
    if is_exercise(chunk):
        return "упражнения"
    return "содержательный"


def main() -> int:
    chunks_path = next(Path("artifacts/parsed").glob("*_chunks.json"), None)
    if chunks_path is None:
        print("не найден файл с фрагментами")
        return 1
    chunks = {
        item.id: item
        for item in (
            Chunk.model_validate(raw)
            for raw in json.loads(chunks_path.read_text(encoding="utf-8"))
        )
    }
    kinds = {chunk_id: classify(chunk) for chunk_id, chunk in chunks.items()}

    print("=== состав корпуса ===")
    for kind, count in Counter(kinds.values()).most_common():
        print(f"  {kind:<16} {count:>5}  ({count / len(chunks):.1%})")

    goldset_path = Path("evaluation/goldsets/goldset.json")
    raw = json.loads(goldset_path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("questions", [])
    questions = [GoldQuestion.model_validate(item) for item in items]
    print(f"\nвопросов в наборе: {len(questions)}")

    print("\n=== эталонные фрагменты по типам вопросов ===")
    by_type: dict[str, Counter] = {}
    tainted: dict[str, int] = {}
    totals: dict[str, int] = {}
    for question in questions:
        bucket = by_type.setdefault(question.question_type, Counter())
        totals[question.question_type] = totals.get(question.question_type, 0) + 1
        bad = False
        for chunk_id in question.gold_chunk_ids:
            kind = kinds.get(chunk_id, "нет в корпусе")
            bucket[kind] += 1
            if kind in {"оглавление", "упражнения", "нет в корпусе"}:
                bad = True
        if bad:
            tainted[question.question_type] = tainted.get(question.question_type, 0) + 1

    for question_type, counter in sorted(by_type.items()):
        total = sum(counter.values())
        print(f"\n  {question_type} ({totals[question_type]} вопросов, {total} фрагментов)")
        for kind, count in counter.most_common():
            print(f"    {kind:<18} {count:>4}  ({count / total:.1%})")
        share = tainted.get(question_type, 0) / totals[question_type]
        print(
            f"    вопросов, где хотя бы один эталонный фрагмент негоден: "
            f"{tainted.get(question_type, 0)} из {totals[question_type]} ({share:.1%})"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
