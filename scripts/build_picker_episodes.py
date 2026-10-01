"""Эпизоды обучения отборщика (Context-Picker) для scripts/train_picker.py.

Эпизод — вопрос, окно кандидатов и номера эталонных в окне:

    {"question_id", "source", "type", "question", "messages": [...],
     "window": [chunk_id, …], "gold": [номер в окне, с нуля]}

``messages`` собираются той же функцией, что и подсказка отбора в работе
(``set_selection.build_messages``, режим picker): обученный отборщик видит
в работе ровно то, на чём учился.

Источники:

``musique``  MuSiQue train (musique_ans_v1.0_train.jsonl): у каждого вопроса
             20 абзацев, из них 2–4 опорных (is_supporting) — готовое окно
             и минимальное множество по построению, без поиска и без судьи.
             Берутся только answerable. С замером не пересекается: MuSiQue-300
             собран из dev (файлы HippoRAG 2).
``bundle``   Выдача любой системы в формате бенча (bench_metrics.py): окно —
             верхние ``--window`` из ``ranked``. Эпизоды без всего эталона
             в окне по умолчанию отбрасываются: там награда ограничена сверху
             не отбором, а доступом, и учит не тому. ``--inject`` вместо этого
             ставит недостающие эталонные на места худших кандидатов.

    python scripts/build_picker_episodes.py musique --src data/musique/musique_ans_v1.0_train.jsonl \\
        --n 4000 --out artifacts/picker/musique-train.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.models import Chunk, ScoredChunk  # noqa: E402
from rag_textbook.retrieval.set_selection import build_messages  # noqa: E402


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _scored(chunk_id: str, text: str) -> ScoredChunk:
    chunk = Chunk(id=chunk_id, doc_id="", doc_name="", source_path="", ordinal=0, text=text)
    return ScoredChunk(chunk=chunk)


def episode(qid: str, source: str, qtype: str, question: str, window: list[tuple[str, str]],
            gold: list[int], max_chars: int) -> dict:
    items = [_scored(cid, text) for cid, text in window]
    messages = build_messages("picker", question, items, max_chars)
    return {
        "question_id": qid, "source": source, "type": qtype, "question": question,
        "messages": [{"role": m.role, "content": m.content} for m in messages],
        "window": [cid for cid, _ in window], "gold": sorted(gold),
    }


def from_musique(path: Path, max_chars: int) -> list[dict]:
    rows = []
    for row in load_jsonl(path):
        if not row.get("answerable", True):
            continue
        window = [
            (f"{row['id']}:{p['idx']}", f"{p['title']}\n{p['paragraph_text']}")
            for p in row["paragraphs"]
        ]
        gold = [i for i, p in enumerate(row["paragraphs"]) if p.get("is_supporting")]
        if gold:
            qtype = row["id"].split("__")[0]  # 2hop, 3hop1, …
            rows.append(episode(row["id"], "musique-train", qtype, row["question"], window, gold,
                                max_chars))
    return rows


def from_bundle(bundle: Path, rankings: Path, size: int, inject: bool, seed: int,
                max_chars: int) -> tuple[list[dict], Counter]:
    chunks = {row["chunk_id"]: row["text"] for row in load_jsonl(bundle / "chunks.jsonl")}
    questions = {row["qid"]: row for row in load_jsonl(bundle / "questions.jsonl")}
    source = bundle.name
    stats: Counter = Counter()
    rows = []
    for ranking in load_jsonl(rankings):
        question = questions[ranking["qid"]]
        gold_ids = [cid for cid in question["gold_chunk_ids"] if cid in chunks]
        ids = [cid for cid in ranking["ranked"] if cid in chunks][:size]
        missing = [cid for cid in gold_ids if cid not in ids]
        if missing and not inject:
            stats["без всего эталона в окне"] += 1
            continue
        if missing:
            rng = random.Random(f"{seed}:{ranking['qid']}")
            # Вытесняются худшие неэталонные, эталон встаёт на случайное место:
            # иначе он всегда был бы в хвосте, и отборщик выучил бы позицию.
            keep = [cid for cid in ids if cid in gold_ids]
            others = [cid for cid in ids if cid not in gold_ids]
            others = others[: max(0, size - len(keep) - len(missing))]
            ids = keep + others
            for cid in missing:
                ids.insert(rng.randrange(len(ids) + 1), cid)
            stats["эталон подставлен"] += 1
        gold = [i for i, cid in enumerate(ids) if cid in gold_ids]
        if not gold:
            stats["эталона нет"] += 1
            continue
        window = [(cid, chunks[cid]) for cid in ids]
        rows.append(episode(ranking["qid"], source, question.get("type", ""), question["question"],
                            window, gold, max_chars))
    return rows, stats


def pick(rows: list[dict], n: int) -> list[dict]:
    """Детерминированная подвыборка по хэшу идентификатора."""
    if not n or n >= len(rows):
        return rows
    return sorted(rows, key=lambda r: hashlib.sha1(r["question_id"].encode()).hexdigest())[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="source", required=True)
    mq = sub.add_parser("musique")
    mq.add_argument("--src", type=Path, required=True)
    bd = sub.add_parser("bundle")
    bd.add_argument("--bundle", type=Path, required=True)
    bd.add_argument("--rankings", type=Path, required=True)
    bd.add_argument("--window", type=int, default=20)
    bd.add_argument("--inject", action="store_true")
    bd.add_argument("--seed", type=int, default=20260929)
    for p in (mq, bd):
        p.add_argument("--n", type=int, default=0, help="сколько эпизодов оставить (0 — все)")
        p.add_argument("--chars", type=int, default=2400)
        p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    stats: Counter = Counter()
    if args.source == "musique":
        rows = from_musique(args.src, args.chars)
    else:
        rows, stats = from_bundle(args.bundle, args.rankings, args.window, args.inject, args.seed,
                                  args.chars)
    rows = pick(rows, args.n)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    types = Counter(row["type"] for row in rows)
    gold = Counter(len(row["gold"]) for row in rows)
    print(f"эпизодов {len(rows)}: {args.out}")
    print(f"  типы: {dict(sorted(types.items()))}; эталонных в окне: {dict(sorted(gold.items()))}")
    if stats:
        print(f"  {dict(stats)}")


if __name__ == "__main__":
    main()
