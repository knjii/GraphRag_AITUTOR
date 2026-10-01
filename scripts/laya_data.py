"""Данные для классификатора решений (гипотезы L1 и L2, docs/HYPOTHESES.md).

Один классификатор (laya-multilingual) учится двум вопросам сразу:

L1 «нужен ли фрагмент B для ответа, если A уже найден»: состояние — вопрос,
   A и B. Метка — B поддерживающий. A — поддерживающий в 80 % примеров
   и отвлекающий в 20 %: в работе якорем служит первое место выдачи,
   а оно не всегда верно. Отрицательные B — отвлекающие абзацы того же
   вопроса MuSiQue: составители набора подбирали их похожими на вопрос.
L2 «хватает ли этих пяти фрагментов для ответа»: положительный пример —
   все поддерживающие плюс отвлекающие до пяти, отрицательный — те же без
   одного поддерживающего. Это форма наших промахов «нашёл один из двух».

Формат строки — как у рецепта Laya: ``{"state", "questions", "answers"}``,
плюс ``qid`` и ``task`` для отчёта. Вопросы определены здесь один раз
(``QUESTIONS``) и импортируются оценкой и сборкой цепочки: обучение и работа
обязаны видеть одну и ту же формулировку.

    python scripts/laya_data.py --src data/musique/musique_ans_v1.0_train.jsonl \\
        --train-questions 6000 --heldout 500 --out artifacts/laya/data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

# Пределы знаков на фрагмент. Окно laya-multilingual — 1024 токена вместе
# с вопросом и вариантами; пять фрагментов L2 обязаны поместиться целиком,
# иначе последний молча обрежется и метка «хватает» станет ложью.
CHARS_PAIR = 1500
CHARS_SET = 650

QUESTIONS = {
    "pair": {
        "type": "noul",
        "instructions": (
            "The state holds a multi-hop question, passage A that was already retrieved "
            "for it, and a candidate passage B. Is passage B also needed to answer the "
            "question, for example because it continues the reasoning chain from A?"
        ),
        "criteria": {
            "false": "B is not needed: it does not provide a fact required by the question",
            "true": "B is needed: it provides a fact required to answer the question",
        },
    },
    "enough": {
        "type": "noul",
        "instructions": (
            "The state holds a multi-hop question and five retrieved passages. "
            "Do these passages together contain every fact needed to answer the question?"
        ),
        "criteria": {
            "false": "at least one required fact is missing from the passages",
            "true": "all facts required for the answer are present in the passages",
        },
    },
}


def clip(text: str, limit: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def passage_text(title: str, body: str) -> str:
    """Как фрагмент выглядит в наборе MuSiQue-300: «заголовок\\nабзац»."""
    return f"{title}\n{body}"


def pair_state(question: str, a: str, b: str) -> dict:
    return {"question": question, "passage_A": clip(a, CHARS_PAIR), "passage_B": clip(b, CHARS_PAIR)}


def set_state(question: str, passages: list[str]) -> dict:
    return {
        "question": question,
        "passages": [clip(text, CHARS_SET) for text in passages],
    }


def noul_row(qid: str, task: str, state: dict, label: bool) -> dict:
    return {
        "qid": qid,
        "task": task,
        "state": state,
        "questions": {task: QUESTIONS[task]},
        "answers": {task: {"noul": 1.0 if label else 0.0}},
    }


def _bucket(qid: str) -> float:
    """Детерминированное место вопроса в [0, 1): отбор не зависит от порядка файла."""
    return int(hashlib.sha1(qid.encode("utf-8")).hexdigest()[:8], 16) / 0x100000000


def rows_for(record: dict, rng: random.Random) -> list[dict]:
    qid = record["id"]
    question = record["question"]
    texts = [passage_text(p["title"], p["paragraph_text"]) for p in record["paragraphs"]]
    support = [i for i, p in enumerate(record["paragraphs"]) if p["is_supporting"]]
    distract = [i for i, p in enumerate(record["paragraphs"]) if not p["is_supporting"]]
    if len(support) < 2 or len(distract) < 3:
        return []
    rows: list[dict] = []

    # L1: один положительный B и два отрицательных при одном якоре.
    anchor = rng.choice(support) if rng.random() < 0.8 else rng.choice(distract)
    positive = rng.choice([i for i in support if i != anchor])
    rows.append(noul_row(qid, "pair", pair_state(question, texts[anchor], texts[positive]), True))
    for negative in rng.sample([i for i in distract if i != anchor], 2):
        rows.append(noul_row(qid, "pair", pair_state(question, texts[anchor], texts[negative]), False))

    # L2: полная пятёрка и та же без одного поддерживающего.
    if len(support) <= 5:
        fill = rng.sample(distract, 5 - len(support) + 1)
        full = support + fill[: 5 - len(support)]
        dropped = rng.choice(support)
        short = [i for i in support if i != dropped] + fill[: 5 - len(support) + 1]
        for chosen, label in ((full, True), (short, False)):
            order = list(chosen)
            rng.shuffle(order)
            rows.append(noul_row(qid, "enough", set_state(question, [texts[i] for i in order]), label))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--train-questions", type=int, default=6000)
    parser.add_argument("--heldout", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    records = []
    with args.src.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("answerable", True):
                records.append(record)
    records.sort(key=lambda r: (_bucket(r["id"]), r["id"]))
    heldout = records[: args.heldout]
    train = records[args.heldout : args.heldout + args.train_questions]

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)
    summary = {"src": str(args.src), "seed": args.seed}
    for name, part in (("train", train), ("heldout", heldout)):
        rows = [row for record in part for row in rows_for(record, rng)]
        path = args.out / f"{name}.jsonl"
        with path.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        counts: dict[str, list[int]] = {}
        for row in rows:
            stat = counts.setdefault(row["task"], [0, 0])
            stat[0] += 1
            stat[1] += int(row["answers"][row["task"]]["noul"] > 0.5)
        summary[name] = {
            "questions": len(part),
            "rows": len(rows),
            "by_task": {task: {"rows": n, "positive": p} for task, (n, p) in counts.items()},
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        print(f"{name}: вопросов {len(part)}, строк {len(rows)}, {summary[name]['by_task']}")
    (args.out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                            encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
