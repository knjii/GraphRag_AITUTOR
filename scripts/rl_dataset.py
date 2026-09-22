"""Выгрузка эпизодов RL по слепку поиска в JSONL.

    python scripts/rl_dataset.py --trace capture/session-0819/trace-always.jsonl \\
        --goldset evaluation/goldsets/goldset.json --out rl-test.jsonl \\
        --prompt deploy/prompts/qa-v4.txt --test-docs 0690bb81b7e3c831

Эпизоды, чей контекст касается книг из ``--test-docs``, пишутся в файл
``*-test.jsonl``, остальные — в ``*-train.jsonl``. Нынешний набор из 388
вопросов целиком относится к одной книге и уходит в тест; обучающие
эпизоды появятся со слепком по библиотеке (спринт 2).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--goldset", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True, help="префикс: <out>-train.jsonl и <out>-test.jsonl")
    parser.add_argument("--prompt", type=Path, required=True, help="текст системного промпта")
    parser.add_argument("--window", type=int, default=16384)
    parser.add_argument("--test-docs", nargs="*", default=[])
    parser.add_argument("--verdicts", type=Path, default=None,
                        help="вердикты ручной проверки; отклонённые не идут в обучение")
    args = parser.parse_args()
    # Загрузчик вердиктов молча возвращает пустой набор, если файла нет:
    # опечатка в пути выключила бы ручную отбраковку без следа (задача 020).
    if args.verdicts is not None and not args.verdicts.is_file():
        parser.error(f"нет файла вердиктов: {args.verdicts}")

    # Промпт и окно задаются явно: RL учит политику на конкретном промпте,
    # и тот же промпт должен стоять в сервисе. Значение по умолчанию из
    # настроек здесь было бы ловушкой (см. историю PROMPT_VERSION).
    os.environ["QA_SYSTEM_PROMPT"] = args.prompt.read_text(encoding="utf-8")
    os.environ["LLM_CONTEXT_WINDOW"] = str(args.window)

    from rag_textbook.config import Settings
    from rag_textbook.evaluation.verdicts import VerdictSet
    from rag_textbook.rl.env import (
        build_examples,
        drop_unfit,
        load_chunks,
        load_trace,
        save_jsonl,
        split_by_docs,
    )

    settings = Settings()
    examples = build_examples(
        settings, load_trace(args.trace), load_chunks(settings.paths.parsed_dir), args.goldset
    )
    train, test = split_by_docs(examples, set(args.test_docs))
    train, dropped = drop_unfit(train, VerdictSet.load(args.verdicts) if args.verdicts else None)
    for reason, count in dropped.most_common():
        print(f"из обучения исключено ({reason}): {count}")
    stem = str(args.out.with_suffix(""))
    n_train = save_jsonl(train, Path(f"{stem}-train.jsonl"))
    n_test = save_jsonl(test, Path(f"{stem}-test.jsonl"))
    print(f"обучение: {n_train}, тест: {n_test}, "
          f"эталон вне контекста: {sum(not e.gold_in_context for e in examples)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
