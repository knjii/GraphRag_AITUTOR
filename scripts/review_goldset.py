"""Выгрузка вопросов эталонного набора для ручной проверки.

Набор сгенерирован моделью и не вычитан ни разу. Пока неизвестно, какая доля
«связывающих» вопросов действительно требует двух фрагментов, отставание этого
типа (0.734 против 0.927 у одношаговых) нельзя отнести ни к системе,
ни к разметке — а значит, нельзя и осмысленно улучшать.

Скрипт печатает вопрос вместе с его эталонными фрагментами, чтобы проверка
шла по тексту, а не по памяти.

    python scripts/review_goldset.py --type graph_linked --start 0 --count 10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_textbook.models import Chunk, GoldQuestion


def load_chunks() -> dict[str, Chunk]:
    path = next(Path("artifacts/parsed").glob("*_chunks.json"), None)
    if path is None:
        raise SystemExit("не найден файл с фрагментами в artifacts/parsed")
    raw = json.loads(path.read_text(encoding="utf-8"))
    chunks = [Chunk.model_validate(item) for item in raw]
    return {chunk.id: chunk for chunk in chunks}


def load_questions(path: Path) -> list[GoldQuestion]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    items = raw if isinstance(raw, list) else raw.get("questions", [])
    return [GoldQuestion.model_validate(item) for item in items]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--goldset", default="evaluation/goldsets/goldset.json")
    parser.add_argument("--type", default="", help="фильтр по типу вопроса")
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--chars", type=int, default=1400, help="сколько символов фрагмента печатать")
    args = parser.parse_args()

    chunks = load_chunks()
    questions = load_questions(Path(args.goldset))
    if args.type:
        questions = [item for item in questions if item.question_type == args.type]

    window = questions[args.start : args.start + args.count]
    print(f"# всего по фильтру: {len(questions)}, показаны {args.start}..{args.start + len(window) - 1}\n")

    for position, question in enumerate(window, start=args.start):
        print("=" * 78)
        print(f"[{position}] id={question.id}  тип={question.question_type}  шагов={question.expected_hops}")
        print(f"ВОПРОС: {question.question}")
        print(f"ОТВЕТ ГЕНЕРАТОРА: {question.answer}")
        for number, chunk_id in enumerate(question.gold_chunk_ids, start=1):
            chunk = chunks.get(chunk_id)
            print(f"\n--- эталонный фрагмент {number}/{len(question.gold_chunk_ids)}: {chunk_id} ---")
            if chunk is None:
                print("ФРАГМЕНТ НЕ НАЙДЕН В КОРПУСЕ")
                continue
            header = " / ".join(chunk.headers or [])
            print(f"страницы {chunk.pages}, заголовки: {header}")
            print(chunk.text[: args.chars])
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
