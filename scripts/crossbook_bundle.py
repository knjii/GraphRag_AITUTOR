"""Межкнижный эталон (goldset-x) в формате бенча — для проверки «не хуже» серии S.

Протокол S требует от SetR и SEAL не потерять качество на нашем межкнижном
наборе. Чтобы мерить теми же скриптами, что MuSiQue (bench_ours.py rank,
bench_select.py, bench_answers.py), эталон переводится в набор:

``questions.jsonl``  вопросы части ``--split`` (по умолчанию test, 60):
                     qid, question, gold_chunk_ids, answer, type;
``chunks.jsonl``     все фрагменты коллекции учебника (chunk_id, doc_id, text) —
                     выдача может вернуть любой фрагмент, генератору нужен текст;
``manifest.json``    sha256 файлов, sha256 эталона, число фрагментов.

Эталон сверяется с замороженным (goldset-x.accepted). Каждый эталонный
фрагмент обязан быть в коллекции: иначе нарезка разошлась с эталоном,
и замер мерил бы пустоту — тогда стоп.

    QDRANT_COLLECTION=library_ru python scripts/crossbook_bundle.py --out artifacts/bench/goldset-x
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.stores.vector_store import build_vector_store  # noqa: E402

GOLDSET = ROOT / "evaluation" / "goldsets" / "goldset-x.json"
ACCEPTED = ROOT / "evaluation" / "goldsets" / "goldset-x.accepted"


def as_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    return [str(v) for v in ast.literal_eval(value)] if value else []


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    frozen = ACCEPTED.read_text(encoding="utf-8").split()[0]
    if sha256(GOLDSET) != frozen:
        print("СТОП: goldset-x.json отличается от замороженного", file=sys.stderr)
        return 1
    data = json.loads(GOLDSET.read_text(encoding="utf-8"))
    questions = [q for q in data["questions"] if q.get("split") == args.split]

    settings = Settings()
    store = build_vector_store(settings.vector_store)
    args.out.mkdir(parents=True, exist_ok=True)
    known = set()
    with (args.out / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in store.iter_chunks():
            known.add(chunk.id)
            handle.write(json.dumps({"chunk_id": chunk.id, "doc_id": chunk.doc_id, "text": chunk.text},
                                    ensure_ascii=False) + "\n")
    missing = []
    with (args.out / "questions.jsonl").open("w", encoding="utf-8") as handle:
        for q in questions:
            gold = as_list(q["gold_chunk_ids"])
            missing += [cid for cid in gold if cid not in known]
            handle.write(json.dumps({
                "qid": q["id"], "question": q["question"], "gold_chunk_ids": gold,
                "answer": q["answer"], "answer_aliases": [], "type": q.get("slice", ""),
            }, ensure_ascii=False) + "\n")
    if missing:
        print(f"СТОП: {len(missing)} эталонных фрагментов нет в коллекции "
              f"{settings.vector_store.collection if hasattr(settings.vector_store, 'collection') else ''}: "
              f"{missing[:5]}", file=sys.stderr)
        return 1
    manifest = {
        "source": "evaluation/goldsets/goldset-x.json", "goldset_sha256": frozen, "split": args.split,
        "questions": len(questions), "chunks": len(known),
        "sha256": {name: sha256(args.out / name) for name in ("questions.jsonl", "chunks.jsonl")},
    }
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"набор goldset-x ({args.split}): {len(questions)} вопросов, {len(known)} фрагментов коллекции")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
