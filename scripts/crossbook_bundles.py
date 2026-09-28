"""Приёмка межкнижных кандидатов (гипотеза К10) голосованием Codex.

``bundle``: новые кандидаты из ``tasks/035/out/*.jsonl`` режутся на пачки
по 10 (порядок случаен с фиксированным зерном). На каждый прогон своя копия
пачки: другой порядок пакетов и фрагментов. Поля ``roles`` и ``topic`` в пакет
не попадают: проверяющий сам решает, нужны ли оба фрагмента, а не сверяет
заявление составителя.

``tally``: свод голосов ``tasks/036/out/*.jsonl``. Два совпавших голоса — итог,
два разных ждут третьего (``--pending`` печатает задачи на третий голос),
иначе большинство. Годен только ``ok``.
"""

from __future__ import annotations

import argparse
import io
import json
import random
from collections import Counter, defaultdict
from contextlib import redirect_stdout
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ROOT / "tasks/035/out"
CHUNKS = ROOT / "tasks/035/chunks"
OUT = ROOT / "tasks/036"

TASK = """# Задача 036/{name}: голос по межкнижному вопросу (пачка {b}, прогон {r})

Это разметка, не код. Ничего не менять, кроме файла результата.

1. Прочитай правила целиком: `tasks/032/RULES.md`. Следуй им строго:
   порядок проверки (первый сработавший вердикт), правило про синтез,
   правило «итоговая формула в одном фрагменте — single_hop_enough»,
   чтение сквозь шум OCR. Раздел «Допуск агента» к тебе не относится.
   Все пакеты здесь — группа `cross_book`, вердикта абляции нет.
2. Прочитай `{bundle}` — пакеты (вопрос, эталонный ответ, два фрагмента
   из разных книг). Каждый пакет читай целиком. Порядок фрагментов случаен.
3. Проверка слепая: НЕ открывай ничего, кроме этих двух файлов.
4. Для каждого пакета: какие содержательные части есть в эталонном ответе
   и в каком фрагменте стоит каждая; можно ли получить весь ответ из одного
   фрагмента.
5. Запиши `{out}` — по строке JSON на пакет, формат из раздела «Ответ
   проверяющего» правил. Фрагменты в заметке называй по chunk_id.

## Можно менять

- `{out}` (новый)
"""


def load_candidates() -> list[dict]:
    rows = []
    for path in sorted(CANDIDATES.glob("*.jsonl")):
        rows += [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    return rows


def load_texts(ids: set[str]) -> dict[str, str]:
    texts = {}
    for doc in {i.split(":")[0] for i in ids}:
        for row in json.loads((CHUNKS / f"{doc}_chunks.json").read_text(encoding="utf-8")):
            if row["id"] in ids:
                texts[row["id"]] = row["text"]
    return texts


def bundle(args: argparse.Namespace) -> int:
    packets_dir = OUT / "packets"
    packets_dir.mkdir(parents=True, exist_ok=True)
    (OUT / "out").mkdir(exist_ok=True)
    done = {json.loads(p.read_text(encoding="utf-8"))["question_id"] for p in packets_dir.glob("*.json")}
    start = 1 + max([int(p.name[1:3]) for p in packets_dir.glob("b*.json")] or [0])
    pool = [c for c in load_candidates() if c["id"] not in done]
    random.Random(args.seed).shuffle(pool)
    texts = load_texts({cid for c in pool for cid in c["gold_chunk_ids"]})
    for index in range(0, len(pool), 10):
        b = start + index // 10
        packets = []
        for c in pool[index:index + 10]:
            missing = [cid for cid in c["gold_chunk_ids"] if cid not in texts]
            if missing:
                raise SystemExit(f"{c['id']}: нет фрагментов {missing}")
            packet = {"question_id": c["id"], "group": "cross_book", "question": c["question"],
                      "answer": c["answer"],
                      "fragments": [{"chunk_id": cid, "text": texts[cid]} for cid in c["gold_chunk_ids"]]}
            (packets_dir / f"b{b:02d}_{c['id']}.json").write_text(
                json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8")
            packets.append(packet)
        for r in (1, 2, 3):  # третий прогон пишется заранее, запускается только при расхождении
            local = random.Random(f"{args.seed}-{b}-{r}")
            order = packets[:]
            local.shuffle(order)
            lines = [f"# Пачка {b}, прогон {r}: {len(order)} пакетов", ""]
            for p in order:
                p = dict(p, fragments=local.sample(p["fragments"], len(p["fragments"])))
                lines += [f"## Пакет {p['question_id']}", "", "```json",
                          json.dumps(p, ensure_ascii=False, indent=1), "```", ""]
            path = OUT / f"bundle-b{b:02d}-r{r}.md"
            path.write_text("\n".join(lines), encoding="utf-8")
            name = f"codex-r{r}-b{b:02d}"
            (OUT / f"{name}.md").write_text(TASK.format(
                name=name, b=b, r=r, bundle=path.relative_to(ROOT).as_posix(),
                out=f"tasks/036/out/{name}.jsonl"), encoding="utf-8")
        print(f"пачка {b}: {len(packets)} кандидатов")
    return 0


def tally(args: argparse.Namespace) -> int:
    groups = {}
    for p in (OUT / "packets").glob("*.json"):
        groups[json.loads(p.read_text(encoding="utf-8"))["question_id"]] = p.name[:3]
    votes: dict[str, list[str]] = defaultdict(list)
    for path in sorted((OUT / "out").glob("codex-r*.jsonl")):
        for line in path.read_text(encoding="utf-8-sig").splitlines():  # Codex иногда пишет BOM
            if line.strip():
                row = json.loads(line)
                votes[row["question_id"]].append(row["verdict"])
    final, pending = {}, defaultdict(list)
    for qid, b in groups.items():
        v = votes[qid]
        if len(v) < 2:
            continue
        if len(v) == 2 and v[0] != v[1]:
            pending[b].append(qid)
            continue
        verdict, count = Counter(v).most_common(1)[0]
        final[qid] = verdict if count * 2 > len(v) else "ambiguous"
    if args.pending:
        for b, ids in sorted(pending.items()):
            print(f"{b}: {len(ids)} — {' '.join(ids)}")
        return 0
    print(f"кандидатов в пачках: {len(groups)}, решено: {len(final)}, ждут третьего голоса: "
          f"{sum(map(len, pending.values()))}")
    for verdict, count in Counter(final.values()).most_common():
        print(f"  {verdict:18} {count}")
    if args.write:
        rows = [{"question_id": q, "verdict": v, "votes": dict(Counter(votes[q]))} for q, v in sorted(final.items())]
        (OUT / "verdicts.json").write_text(json.dumps({"verdicts": rows}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
        print(f"записано: {OUT / 'verdicts.json'}")
    return 0


def third(args: argparse.Namespace) -> int:
    """Спорные вопросы всех пачек — в одну пачку на третий голос (не перепроверять пачку целиком)."""
    votes: dict[str, int] = Counter()
    for path in (OUT / "out").glob("codex-r*.jsonl"):
        for line in path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip():
                votes[json.loads(line)["question_id"]] += 1
    packets = {}
    for p in sorted((OUT / "packets").glob("*.json")):
        packet = json.loads(p.read_text(encoding="utf-8"))
        packets[packet["question_id"]] = packet
    buf = io.StringIO()
    with redirect_stdout(buf):
        tally(argparse.Namespace(pending=True, write=False))
    pending = [q for line in buf.getvalue().splitlines() for q in line.split(" — ")[1].split()]
    pending = [q for q in pending if votes[q] == 2]
    if not pending:
        print("спорных нет")
        return 0
    t = 1 + max([int(p.name[len("bundle-t"):][:2]) for p in OUT.glob("bundle-t*-r3.md")] or [0])
    local = random.Random(f"{args.seed}-t{t}")
    lines = [f"# Третий голос, пачка t{t:02d}: {len(pending)} пакетов", ""]
    for q in local.sample(pending, len(pending)):
        p = packets[q]
        p = dict(p, fragments=local.sample(p["fragments"], len(p["fragments"])))
        lines += [f"## Пакет {q}", "", "```json", json.dumps(p, ensure_ascii=False, indent=1), "```", ""]
    path = OUT / f"bundle-t{t:02d}-r3.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    name = f"codex-r3-t{t:02d}"
    (OUT / f"{name}.md").write_text(TASK.format(
        name=name, b=f"t{t:02d}", r=3, bundle=path.relative_to(ROOT).as_posix(),
        out=f"tasks/036/out/{name}.jsonl"), encoding="utf-8")
    print(f"пачка t{t:02d}: {len(pending)} спорных — задача 036/{name}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bundle")
    b.add_argument("--seed", type=int, default=20260927)
    t = sub.add_parser("tally")
    t.add_argument("--pending", action="store_true")
    t.add_argument("--write", action="store_true")
    th = sub.add_parser("third")
    th.add_argument("--seed", type=int, default=20260927)
    args = parser.parse_args(argv)
    return {"bundle": bundle, "tally": tally, "third": third}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
