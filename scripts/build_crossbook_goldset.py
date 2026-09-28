"""Межкнижный эталон (гипотеза К10) из принятых голосованием кандидатов.

Берутся кандидаты ``tasks/035/out/*.jsonl`` с итоговым вердиктом ``ok``
в ``tasks/036/verdicts.json`` (``crossbook_bundles.py tally --write``).
Остальные вердикты в набор не попадают: годен только вопрос, для которого
нужны оба фрагмента. Ручная проверка (``tasks/036/review.json``, поле
``exclude``: вопрос → причина) может исключить принятый вопрос, но не принять
отклонённый.

Деление на test и dev записано до замера (решение владельца 2026-09-27):
в test ровно ``--test-size`` вопросов (не меньше 60), dev — всё сверх, для
выбора вариантов. Деление стратифицировано по паре книг (все лекции Соколова —
одна книга), чтобы ни одна пара не ушла целиком в одну часть; порядок внутри
пары — случайный с фиксированным зерном. Если принятых меньше ``--test-size``,
набор не собирается: меньший test не различит заданный эффект.

    python scripts/build_crossbook_goldset.py              проверка без записи
    python scripts/build_crossbook_goldset.py --write      evaluation/goldsets/goldset-x.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

from rag_textbook.models import GoldQuestion

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tasks/035/out"
VERDICTS = ROOT / "tasks/036/verdicts.json"
REVIEW = ROOT / "tasks/036/review.json"
CHUNKS = ROOT / "tasks/035/chunks"
OUTPUT = ROOT / "evaluation/goldsets/goldset-x.json"

BOOKS = {
    "0690bb81b7e3c831": "MML",
    "0b3569d862ab1589": "Гельфанд",
    "567b09c6f1443478": "Чернова",
    "822a7f22a3de0829": "Иванов",
}


def book(doc_id: str) -> str:
    return BOOKS.get(doc_id, "Соколов")


def pair_of(candidate: dict) -> str:
    return " × ".join(sorted({book(d) for d in candidate["gold_doc_ids"]}))


def load_candidates() -> dict[str, dict]:
    rows: dict[str, dict] = {}
    for path in sorted(CANDIDATES.glob("*.jsonl")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                row = json.loads(line)
                if row["id"] in rows:
                    raise SystemExit(f"повтор идентификатора {row['id']} в {path.name}")
                rows[row["id"]] = row
    return rows


def split_stratified(accepted: list[dict], test_size: int, seed: int) -> dict[str, str]:
    """Сколько вопросов пары идёт в test — пропорционально её доле, остатки по наибольшей дробной части."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for c in sorted(accepted, key=lambda c: c["id"]):
        groups[pair_of(c)].append(c)
    rng = random.Random(seed)
    for name in sorted(groups):
        rng.shuffle(groups[name])
    total = len(accepted)
    quota = {n: len(g) * test_size / total for n, g in groups.items()}
    take = {n: int(q) for n, q in quota.items()}
    rest = test_size - sum(take.values())
    for n in sorted(groups, key=lambda n: (-(quota[n] - take[n]), n))[:rest]:
        take[n] += 1
    split = {}
    for n, g in groups.items():
        for i, c in enumerate(g):
            split[c["id"]] = "test" if i < take[n] else "dev"
    return split


def build(test_size: int, seed: int, verdicts_path: Path = VERDICTS, review_path: Path | None = None) -> dict:
    candidates = load_candidates()
    verdicts = {v["question_id"]: v for v in json.loads(verdicts_path.read_text(encoding="utf-8"))["verdicts"]}
    unknown = sorted(set(verdicts) - set(candidates))
    if unknown:
        raise SystemExit(f"вердикты без кандидатов: {unknown[:5]}")
    # Ручная проверка поверх голосования может только исключить принятый вопрос,
    # принять отклонённый голосующими она не вправе.
    excluded = json.loads(review_path.read_text(encoding="utf-8"))["exclude"] if review_path else {}
    for q in excluded:
        if verdicts.get(q, {}).get("verdict") != "ok":
            raise SystemExit(f"ручная проверка исключает {q}, но голосование его не приняло")
        verdicts[q] = dict(verdicts[q], verdict="excluded_by_review")
    accepted = [candidates[q] for q, v in verdicts.items() if v["verdict"] == "ok"]
    if len(accepted) < test_size:
        raise SystemExit(f"принято {len(accepted)}, а в test нужно {test_size}: набор не собирается")
    known: dict[str, str] = {}
    for doc in {d for c in accepted for d in c["gold_doc_ids"]}:
        known |= {r["id"]: r["text_hash"] for r in json.loads((CHUNKS / f"{doc}_chunks.json").read_text(encoding="utf-8"))}
    for c in accepted:
        missing = [cid for cid in c["gold_chunk_ids"] if cid not in known]
        if missing or len(set(c["gold_doc_ids"])) < 2:
            raise SystemExit(f"{c['id']}: нет фрагментов {missing} или одна книга")
    split = split_stratified(accepted, test_size, seed)
    questions = []
    for c in sorted(accepted, key=lambda c: c["id"]):
        v = verdicts[c["id"]]
        questions.append({
            "id": c["id"], "question": c["question"], "gold_chunk_ids": c["gold_chunk_ids"],
            "gold_doc_ids": c["gold_doc_ids"], "answer": c["answer"], "question_type": "multi_hop",
            "expected_hops": 2, "pair_source": "crossbook-035", "slice": "cross_book",
            "split": split[c["id"]], "generator_model": "", "verified": True,
            "notes": f"{pair_of(c)}; голоса {json.dumps(v['votes'], ensure_ascii=False)}",
        })
    return {"version": 1, "count": len(questions), "seed": seed, "test_size": test_size,
            "verdicts_sha256": hashlib.sha256(verdicts_path.read_bytes()).hexdigest(),
            "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest() if review_path else None,
            "excluded_by_review": sorted(excluded),
            # Отпечаток текстов: на сервере совпадение id ещё не значит совпадение нарезки.
            "chunk_hashes": {cid: known[cid] for q in questions for cid in sorted(q["gold_chunk_ids"])},
            "questions": questions}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--test-size", type=int, default=60)
    parser.add_argument("--seed", type=int, default=20260927)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--verdicts", type=Path, default=VERDICTS)
    parser.add_argument("--review", type=Path, default=REVIEW if REVIEW.exists() else None,
                        help="исключения ручной проверки; по умолчанию tasks/036/review.json, если он есть")
    args = parser.parse_args(argv)
    if args.test_size < 60:
        parser.error("test меньше 60 — решение владельца, не параметр")
    data = build(args.test_size, args.seed, args.verdicts, args.review)
    for q in data["questions"]:
        GoldQuestion.model_validate(q)
    by = Counter((pair_of(q), q["split"]) for q in data["questions"])
    print(f"принято {data['count']}: test {sum(q['split'] == 'test' for q in data['questions'])}, "
          f"dev {sum(q['split'] == 'dev' for q in data['questions'])}")
    for pair in sorted({p for p, _ in by}):
        print(f"  {pair:22} test {by[(pair, 'test')]:3}  dev {by[(pair, 'dev')]:3}")
    if args.write:
        payload = (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
        args.output.write_bytes(payload)
        # Заморозка до замера: день 4 (deploy/day4.sh) сверяет файл по sha256sum -c
        # из корня репозитория, поэтому путь в отметке — относительный.
        try:
            name = args.output.resolve().relative_to(ROOT).as_posix()
        except ValueError:
            name = args.output.name
        accepted = args.output.with_suffix(".accepted")
        accepted.write_bytes(f"{hashlib.sha256(payload).hexdigest()}  {name}\n".encode("utf-8"))
        print(f"записано: {args.output}; отметка {accepted.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
