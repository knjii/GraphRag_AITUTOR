"""Пакеты для приёмки эталона v2 проверяющим агентом.

Каждый вопрос выборки — отдельный JSON: вопрос, ответ генератора, тексты
эталонных фрагментов и вердикт абляции, который агент подтверждает или
опровергает по правилам ``evaluation/goldsets/review-v2/RULES.md``.
Вердикты владельца в пакеты не попадают: на них агент проверяется вслепую.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from scripts.goldset_review import group_of, load_ablation, load_texts  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--goldset", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True, help="шаблон с выборкой (verdicts.json)")
    parser.add_argument("--parsed", type=Path, required=True)
    parser.add_argument("--ablation", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    questions = {q.id: q for q in load_goldset(args.goldset)}
    texts = load_texts(args.parsed)
    ablation = load_ablation(args.ablation or Path(f"{args.goldset}.ablation.jsonl"))
    sample = [row["question_id"] for row in json.loads(args.verdicts.read_text(encoding="utf-8"))["verdicts"]]
    args.out.mkdir(parents=True, exist_ok=True)
    for number, qid in enumerate(sample, 1):
        question = questions[qid]
        row = ablation.get(qid, {})
        packet = {
            "question_id": qid,
            "group": group_of(question),
            "question": question.question,
            "answer": question.answer,
            "fragments": [{"chunk_id": cid, "text": texts.get(cid, "(текста нет в разборе)")}
                          for cid in question.gold_chunk_ids],
            "ablation": {
                "proposed_verdict": row.get("verdict"),
                "answered_by_each_alone": row.get("single_matches"),
                "answered_by_both": row.get("joint_match"),
            },
        }
        path = args.out / f"{number:02d}_{qid}.json"
        path.write_text(json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Пакетов: {len(sample)} в {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
