"""Отпечаток награды: одинаково ли она считает на ноутбуке и на сервере.

Награда зависит от лемматизации (pymorphy3 подключается лениво, без него
молча работает запасной путь) и от разбора формул. В отдельном окружении
обучения любой пропущенный пакет дал бы другую награду, и обучение шло бы
не по той функции, что проверена вручную. Отпечаток снимается на ноутбуке
и сверяется на сервере до обучения.

    python scripts/reward_fingerprint.py --dataset artifacts/rl/mml-test.jsonl --write fp.json
    python scripts/reward_fingerprint.py --dataset artifacts/rl/mml-test.jsonl --check fp.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.rewards.composite import compute_reward  # noqa: E402
from rag_textbook.rl.env import load_jsonl  # noqa: E402
from rag_textbook.utils.text import content_terms  # noqa: E402

EPISODES = 12


def fingerprint(dataset: Path) -> dict:
    examples = load_jsonl(dataset)[:EPISODES]
    rows = []
    for example in examples:
        # Три ответа на эпизод: эталон, первые предложения контекста, пустой.
        answers = [example.reference[:1200], example.context[:900], ""]
        rows.append([
            round(compute_reward(a, context=example.context, reference=example.reference,
                                 question=example.question,
                                 gold_in_context=example.gold_in_context).total, 6)
            for a in answers
        ])
    return {
        # Лемматизация видна прямо: «матрицы» → «матрица» только с pymorphy3.
        "lemmas": content_terms("ортогональные матрицы сохраняют длины векторов"),
        "rewards": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", type=Path)
    group.add_argument("--check", type=Path)
    args = parser.parse_args()
    current = fingerprint(args.dataset)
    if args.write:
        args.write.write_text(json.dumps(current, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"отпечаток записан: {args.write}")
        return 0
    expected = json.loads(args.check.read_text(encoding="utf-8"))
    if current == expected:
        print("награда совпадает с проверенной на ноутбуке")
        return 0
    print("НАГРАДА РАСХОДИТСЯ с проверенной на ноутбуке:")
    if current["lemmas"] != expected["lemmas"]:
        print(f"  леммы: {current['lemmas']} против {expected['lemmas']} — нет pymorphy3?")
    for index, (got, want) in enumerate(zip(current["rewards"], expected["rewards"], strict=False)):
        if got != want:
            print(f"  эпизод {index}: {got} против {want}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
