"""Вернуть косую черту командам LaTeX в вопросах и ответах эталона.

Эталон v2 собран до ``loads_llm_json``: у 21 вопроса из 277 ``\\b``, ``\\t``,
``\\f``, ``\\r`` и ``\\n`` перед командами стали управляющими символами
(``rag_textbook/clients/llm_json.py``). Скрипт правит только поля ``question``
и ``answer``, печатает каждую правку и без ``--write`` ничего не пишет.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.clients.llm_json import restore_latex  # noqa: E402

FIELDS = ("question", "answer")
CONTROL = "\b\f\t\r"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("goldset", type=Path)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)

    data = json.loads(args.goldset.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    changed = 0
    for question in questions:
        touched = False
        for field in FIELDS:
            value = question.get(field)
            if not isinstance(value, str):
                continue
            fixed = restore_latex(value)
            if fixed != value:
                touched = True
                print(f"{question['id']} {field}: {fixed[:160]!r}")
        if touched:
            changed += 1
            for field in FIELDS:
                if isinstance(question.get(field), str):
                    question[field] = restore_latex(question[field])
    left = sum(any(c in str(q.get(f, "")) for c in CONTROL for f in FIELDS) for q in questions)
    print(f"Исправлено вопросов: {changed} из {len(questions)}; с управляющими символами осталось: {left}")
    if args.write and changed:
        args.goldset.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"Записано: {args.goldset}")
    return 1 if left else 0


if __name__ == "__main__":
    raise SystemExit(main())
