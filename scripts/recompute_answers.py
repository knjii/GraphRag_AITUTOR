"""Пересчёт метрик ответов по сохранённым текстам, без обращения к серверу.

Нужен, потому что измеритель чинится чаще, чем идут прогоны. Тексты ответов
сохраняются целиком, поэтому любую объективную величину можно пересчитать
задним числом — и сравнить прогоны, сделанные разными версиями метрики,
одной и той же меркой.

    python scripts/recompute_answers.py capture/session-0903/answers_*.json
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_textbook.evaluation.answers import (  # noqa: E402
    latex_overlap,
    latin_share,
    looks_like_reasoning,
    sentence_support,
)

CHUNKS = Path("capture/0690bb81b7e3c831_chunks.json")
GOLDSET = Path("capture/goldset.json")


def load_chunks() -> dict[str, str]:
    payload = json.loads(CHUNKS.read_text(encoding="utf-8"))
    items = payload["chunks"] if isinstance(payload, dict) else payload
    return {item["id"]: item["text"] for item in items}


def load_gold() -> dict[str, dict]:
    payload = json.loads(GOLDSET.read_text(encoding="utf-8"))
    items = payload["questions"] if isinstance(payload, dict) else payload
    return {item["id"]: item for item in items}


def recompute(path: Path, chunks: dict[str, str], gold: dict[str, dict]) -> dict:
    outcomes = json.loads(path.read_text(encoding="utf-8"))["outcomes"]
    acc: dict[str, float] = defaultdict(float)
    for item in outcomes:
        answer = item.get("answer", "")
        question = gold.get(item["question_id"])
        acc["вопросов"] += 1
        acc["размышление"] += 1 if looks_like_reasoning(answer) else 0
        acc["не по-русски"] += 1 if latin_share(answer) > 0.5 else 0
        judged, supported = sentence_support(answer, "")  # контекст не сохранён
        acc["предложений"] += judged
        if not question:
            continue
        reference = "\n".join(chunks.get(cid, "") for cid in question["gold_chunk_ids"])
        expected, found = latex_overlap(reference, answer)
        if expected:
            acc["с формулами"] += 1
            acc["формул"] += expected
            acc["дошло"] += found
            acc["хотя бы одна"] += 1 if found else 0
    return dict(acc)


def main() -> None:
    chunks, gold = load_chunks(), load_gold()
    paths = [Path(arg) for arg in sys.argv[1:]] or sorted(
        Path("capture/session-0903").glob("answers_*.json")
    )
    header = f'{"ячейка":<28}{"вопросов":>9}{"формул":>8}{"доля":>7}{"хотя бы одна":>14}{"размышл.":>10}{"не рус.":>9}'
    print(header)
    print("-" * len(header))
    for path in paths:
        acc = recompute(path, chunks, gold)
        with_formulas = acc.get("с формулами", 0) or 1
        total = acc.get("вопросов", 0) or 1
        print(
            f'{path.stem.replace("answers_", ""):<28}'
            f'{int(acc["вопросов"]):>9}'
            f'{int(acc.get("формул", 0)):>8}'
            f'{acc.get("дошло", 0) / max(acc.get("формул", 0), 1):>7.3f}'
            f'{acc.get("хотя бы одна", 0) / with_formulas:>14.3f}'
            f'{acc.get("размышление", 0) / total:>10.3f}'
            f'{acc.get("не по-русски", 0) / total:>9.3f}'
        )


if __name__ == "__main__":
    main()
