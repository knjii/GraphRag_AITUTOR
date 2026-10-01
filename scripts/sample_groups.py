"""Группы генераций на вопрос: приёмка награды и точка отсчёта R6.

GRPO учится на разнице наград внутри группы ответов на один вопрос.
До обучения нужно знать две вещи, и обе меряются здесь, на сервере
с поднятой моделью:

1. **Есть ли сигнал.** Если у всех ответов группы одна награда
   (все за воротами, все с одной формулой), преимущество нулевое и шаг
   обучения пустой. Доля таких групп — первое число сводки.
2. **Согласна ли награда с человеком внутри группы.** Скрипт пишет
   слепой лист (``--sheet``): ответы перемешаны, награда в отдельном
   ключе. Критерий приёмки — согласие пар ≥ 0.75
   (``tasks/013-report.md``, раздел 6.3).

    python scripts/sample_groups.py --dataset rl/test.jsonl --questions 20 --n 8 \\
        --out runs/groups-4b.jsonl --sheet runs/groups-4b

Промпт берётся из набора как есть (``Example.messages``) — тот же, что
получит обучение. Размышление гасится так же, как в сервисе
(``LLM_CHAT_REASONING_EFFORT``); для Qwen3.5 на llama.cpp — аргументом
шаблона чата при запуске сервера.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.clients.llm import ChatMessage, OpenAICompatibleLLMClient  # noqa: E402
from rag_textbook.rewards.composite import compute_reward  # noqa: E402
from rag_textbook.rl.env import Example, load_jsonl  # noqa: E402


async def sample_one(
    client: OpenAICompatibleLLMClient,
    example: Example,
    *,
    temperature: float,
    max_tokens: int,
) -> dict[str, Any]:
    """Одна генерация с причиной остановки: обрыв закрывается воротами."""
    messages = [ChatMessage(role=m["role"], content=m["content"]) for m in example.messages]
    return await client.acomplete_raw(messages, max_tokens=max_tokens, temperature=temperature)


def score(example: Example, sample: dict[str, Any]) -> dict[str, Any]:
    result = compute_reward(
        sample["answer"],
        context=example.context,
        reference=example.reference,
        question=example.question,
        gold_in_context=example.gold_in_context,
        truncated=sample["finish_reason"] == "length",
    )
    return {"reward": result.total, "gate": result.gate, "parts": result.parts}


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        groups.setdefault(row["question_id"], []).append(row["reward"])
    spreads = [statistics.pstdev(values) for values in groups.values()]
    gates = Counter(row["gate"] or "прошёл" for row in rows)
    return {
        "вопросов": len(groups),
        "ответов": len(rows),
        "средняя награда": round(statistics.fmean(r["reward"] for r in rows), 4),
        # Группа без разброса не даёт GRPO градиента.
        "доля групп без разброса": round(sum(s < 1e-9 for s in spreads) / len(spreads), 4),
        "медианный разброс в группе": round(statistics.median(spreads), 4),
        "оборвано пределом": sum(r["finish_reason"] == "length" for r in rows),
        "ворота": dict(gates),
    }


def write_sheet(rows: list[dict[str, Any]], examples: dict[str, Example], prefix: Path,
                questions: int, seed: int) -> None:
    """Слепой лист: ответы перемешаны внутри вопроса, награда — в ключе."""
    rng = random.Random(seed)
    by_question: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_question.setdefault(row["question_id"], []).append(row)
    chosen = sorted(by_question)[:questions]
    sheet, key = [], []
    for qn, qid in enumerate(chosen, start=1):
        example = examples[qid]
        items = by_question[qid][:]
        rng.shuffle(items)
        sheet.append(f"## Вопрос {qn} ({example.question_type})\n{example.question}\n"
                     f"ЭТАЛОН:\n{example.reference[:1500]}\n")
        for index, row in enumerate(items, start=1):
            label = f"{qn}.{index}"
            sheet.append(f"### {label} ({len(row['answer'])} зн., {row['finish_reason']})\n{row['answer']}\n")
            key.append({"id": label, "q": qn, "qid": qid, "sample": row["sample"],
                        "reward": row["reward"], "gate": row["gate"], "parts": row["parts"]})
    prefix.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{prefix}-sheet.md").write_text("\n".join(sheet), encoding="utf-8")
    Path(f"{prefix}-key.json").write_text(json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")


async def run(args: argparse.Namespace) -> int:
    from rag_textbook.config import Settings

    examples = load_jsonl(args.dataset)
    rng = random.Random(args.seed)
    rng.shuffle(examples)
    examples = examples[: args.questions]
    by_id = {e.question_id: e for e in examples}
    client = OpenAICompatibleLLMClient(Settings().llm)

    async def task(example: Example, sample: int) -> dict[str, Any]:
        generated = await sample_one(client, example, temperature=args.temperature,
                                     max_tokens=args.max_tokens)
        return {"question_id": example.question_id, "sample": sample, **generated,
                **score(example, generated)}

    try:
        rows = await asyncio.gather(*(task(e, i) for e in examples for i in range(args.n)))
    finally:
        await client.aclose()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary = summarize(list(rows))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    args.out.with_suffix(".summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.sheet:
        write_sheet(list(rows), by_id, args.sheet, args.sheet_questions, args.seed)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--questions", type=int, default=20)
    parser.add_argument("--n", type=int, default=8)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--max-tokens", type=int, default=768)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--sheet", type=Path, help="префикс слепого листа для ручной приёмки")
    parser.add_argument("--sheet-questions", type=int, default=20)
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
