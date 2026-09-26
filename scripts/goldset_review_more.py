"""Дополнительные пачки на голосование субагентов по эталону v2 (указание владельца 2026-09-24).

Берёт ещё не размеченные двухшаговые вопросы: сначала часть test, потом dev,
в случайном порядке с фиксированным зерном. Режет их на пачки по 10.
На каждый прогон пишется своя копия пачки: другой порядок пакетов,
а фрагменты переставлены случайно. Так пять прогонов получают разный
вход, и ответы расходятся: температуру у субагента задать нельзя.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from scripts.goldset_review import group_of, load_ablation, load_texts  # noqa: E402

TASK = """# Задача {name}: голос по приёмке эталона (пачка {b}, прогон {r})

Это разметка, не код. Ничего не менять, кроме файла результата.

1. Прочитай правила целиком: `tasks/032/RULES.md`. Следуй им строго:
   порядок проверки (первый сработавший вердикт), правило про синтез,
   правило «итоговая формула в одном фрагменте — single_hop_enough»,
   чтение сквозь шум OCR. Раздел «Допуск агента» к тебе не относится.
2. Прочитай `{bundle}` — 10 пакетов (вопрос, эталонный ответ, фрагменты,
   группа, предложенный вердикт абляции). Каждый пакет читай целиком.
   Порядок фрагментов в пакете случаен и ничего не значит.
3. Проверка слепая: НЕ открывай ничего, кроме этих двух файлов.
4. Для каждого пакета: какие содержательные части есть в эталонном ответе
   и в каком фрагменте стоит каждая; можно ли получить весь ответ из одного
   фрагмента.
5. Запиши `{out}` — ровно 10 строк JSON, формат из раздела «Ответ
   проверяющего» правил. Фрагменты в заметке называй по chunk_id.
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--goldset", type=Path, default=ROOT / "evaluation/goldsets/goldset-v2.json")
    parser.add_argument("--parsed", type=Path, default=ROOT / "artifacts/runs/day2/ru-parsed")
    parser.add_argument("--reviewed", type=Path, default=ROOT / "evaluation/goldsets/review-v2/verdicts.json")
    parser.add_argument("--out", type=Path, default=ROOT / "tasks/033")
    parser.add_argument("--batches", type=int, required=True, help="сколько пачек по 10 выдать")
    parser.add_argument("--start", type=int, default=1, help="номер первой пачки")
    parser.add_argument("--runs", type=int, default=3, help="3 голоса: так решил владелец 2026-09-25 ради лимитов")
    parser.add_argument("--seed", type=int, default=20260924)
    args = parser.parse_args(argv)

    done = {r["question_id"] for r in json.loads(args.reviewed.read_text(encoding="utf-8"))["verdicts"]}
    for path in (args.out / "packets").glob("*.json") if (args.out / "packets").exists() else []:
        done.add(json.loads(path.read_text(encoding="utf-8"))["question_id"])
    questions = load_goldset(args.goldset)
    rng = random.Random(args.seed)
    pool = []
    for split in ("test", "dev"):
        part = [q for q in questions if q.split == split and group_of(q) != "single" and q.id not in done]
        rng.shuffle(part)
        pool += part
    texts = load_texts(args.parsed)
    ablation = load_ablation(Path(f"{args.goldset}.ablation.jsonl"))
    (args.out / "packets").mkdir(parents=True, exist_ok=True)
    (args.out / "out").mkdir(exist_ok=True)
    for index in range(args.batches):
        b = args.start + index
        chunk = pool[index * 10:(index + 1) * 10]
        if not chunk:
            break
        packets = []
        for q in chunk:
            row = ablation.get(q.id, {})
            packet = {"question_id": q.id, "group": group_of(q), "question": q.question, "answer": q.answer,
                      "fragments": [{"chunk_id": cid, "text": texts.get(cid, "(текста нет в разборе)")}
                                    for cid in q.gold_chunk_ids],
                      "ablation": {"proposed_verdict": row.get("verdict"),
                                   "answered_by_each_alone": row.get("single_matches"),
                                   "answered_by_both": row.get("joint_match")}}
            (args.out / "packets" / f"b{b:02d}_{q.id}.json").write_text(
                json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
            packets.append(packet)
        for r in range(1, args.runs + 1):
            local = random.Random(f"{args.seed}-{b}-{r}")
            order = packets[:]
            local.shuffle(order)
            lines = [f"# Пачка {b}, прогон {r}: {len(order)} пакетов", ""]
            for p in order:
                p = dict(p, fragments=local.sample(p["fragments"], len(p["fragments"])))
                lines += [f"## Пакет {p['question_id']}", "", "```json",
                          json.dumps(p, ensure_ascii=False, indent=1), "```", ""]
            bundle = args.out / f"bundle-b{b:02d}-r{r}.md"
            bundle.write_text("\n".join(lines), encoding="utf-8")
            name = f"033/sub-r{r}-b{b:02d}"
            (args.out / f"sub-r{r}-b{b:02d}.md").write_text(TASK.format(
                name=name, b=b, r=r, bundle=bundle.relative_to(ROOT).as_posix(),
                out=f"tasks/033/out/sub-r{r}-b{b:02d}.jsonl"), encoding="utf-8")
        print(f"пачка {b}: {len(chunk)} вопросов ({', '.join(sorted({q.split for q in chunk}))})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
