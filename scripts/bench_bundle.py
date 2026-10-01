"""Сборка публичного набора в общий формат для сравнения систем этапа 1.

Все четыре системы (обычный RAG, наш граф в старой и текущей конфигурации,
HippoRAG 2) получают **один и тот же список фрагментов** и один и тот же
эталон. Иначе разница в выдаче смешается с разницей в нарезке, и сравнение
графов перестанет быть сравнением графов.

Формат набора (каталог):

``chunks.jsonl``     {"chunk_id", "doc_id", "title", "text"} — фрагмент
                     целиком, как его видит поиск (у MuSiQue это
                     «заголовок\\nабзац», ровно как в протоколе HippoRAG 2);
``questions.jsonl``  {"qid", "question", "answer", "type", "gold_chunk_ids"};
``manifest.json``    откуда собран, сколько чего, sha256 обоих файлов.

MuSiQue берётся из файлов HippoRAG 2 (osunlp/HippoRAG_2 на Hugging Face):
вопросы — ``musique.json``, корпус — ``musique_corpus.json``. Подвыборка
детерминирована (sha1 идентификатора) и стратифицирована по типу вопроса
(2hop, 3hop1, …), а корпус сужается до абзацев выбранных вопросов — вместе
с их отвлекающими абзацами. Числа поэтому сравнимы между нашими системами,
но с таблицей HippoRAG 2 — только как ориентир: отвлекающих абзацев меньше.

    python scripts/bench_bundle.py musique --src data/hipporag2 --n 300 \\
        --out artifacts/bench/musique-300
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def passage_text(title: str, text: str) -> str:
    # Протокол HippoRAG 2 (reproduce/main.py): документ — заголовок и абзац
    # через перевод строки. Заголовок несёт сущность, без него второй шаг
    # MuSiQue часто неразрешим.
    return f"{title}\n{text}"


def stratified_pick(items: list[dict], n: int, key) -> list[dict]:
    """Берёт n элементов пропорционально группам, внутри группы — по sha1."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    total = len(items)
    quotas = {g: n * len(v) / total for g, v in groups.items()}
    taken = {g: int(q) for g, q in quotas.items()}
    # Остаток раздаём по наибольшей дробной части: сумма ровно n.
    rest = n - sum(taken.values())
    for g in sorted(quotas, key=lambda g: quotas[g] - taken[g], reverse=True)[:rest]:
        taken[g] += 1
    picked: list[dict] = []
    for g, members in sorted(groups.items()):
        members = sorted(members, key=lambda item: _sha1(item["id"]))
        picked.extend(members[: taken[g]])
    return sorted(picked, key=lambda item: _sha1(item["id"]))


def build_musique(src: Path, n: int) -> tuple[list[dict], list[dict], dict]:
    questions = json.loads((src / "musique.json").read_text(encoding="utf-8"))
    corpus = json.loads((src / "musique_corpus.json").read_text(encoding="utf-8"))
    by_key = {(p["title"], p["text"]): i for i, p in enumerate(corpus)}

    hop_type = lambda q: q["id"].split("__")[0]  # noqa: E731
    picked = stratified_pick(questions, n, hop_type) if n else questions

    used: dict[int, None] = {}
    rows: list[dict] = []
    missing = 0
    for q in picked:
        gold: list[str] = []
        for para in q["paragraphs"]:
            idx = by_key.get((para["title"], para["paragraph_text"]))
            if idx is None:
                missing += 1
                continue
            used[idx] = None
            if para.get("is_supporting"):
                gold.append(f"musique-{idx}")
        rows.append({
            "qid": q["id"], "question": q["question"], "answer": q["answer"],
            "type": hop_type(q), "hops": len(q["question_decomposition"]),
            "gold_chunk_ids": gold,
        })
    chunks = [
        {"chunk_id": f"musique-{i}", "doc_id": f"musique-{i}", "title": corpus[i]["title"],
         "text": passage_text(corpus[i]["title"], corpus[i]["text"])}
        for i in sorted(used)
    ]
    stats = {"paragraphs_not_in_corpus": missing,
             "questions_without_gold": sum(1 for r in rows if not r["gold_chunk_ids"])}
    return chunks, rows, stats


def write_bundle(out: Path, chunks: list[dict], rows: list[dict], meta: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for name, items in (("chunks.jsonl", chunks), ("questions.jsonl", rows)):
        with (out / name).open("w", encoding="utf-8", newline="\n") as handle:
            for item in items:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    types: dict[str, int] = defaultdict(int)
    for r in rows:
        types[r["type"]] += 1
    manifest = {
        **meta, "chunks": len(chunks), "questions": len(rows), "types": dict(sorted(types.items())),
        "gold_per_question": round(sum(len(r["gold_chunk_ids"]) for r in rows) / max(1, len(rows)), 3),
        "sha256": {name: _sha256(out / name) for name in ("chunks.jsonl", "questions.jsonl")},
    }
    (out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                                       encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="dataset", required=True)
    mq = sub.add_parser("musique", help="MuSiQue из файлов HippoRAG 2")
    mq.add_argument("--src", type=Path, required=True, help="каталог с musique.json и musique_corpus.json")
    mq.add_argument("--n", type=int, default=300, help="вопросов (0 — все 1000)")
    mq.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    if args.dataset == "musique":
        chunks, rows, stats = build_musique(args.src, args.n)
        write_bundle(args.out, chunks, rows, {
            "dataset": "musique", "source": "osunlp/HippoRAG_2", "n": args.n,
            "selection": "стратификация по типу, порядок sha1(id)", **stats,
        })


if __name__ == "__main__":
    main()
