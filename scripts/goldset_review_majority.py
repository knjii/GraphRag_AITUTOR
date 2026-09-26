"""Свод голосования субагентов по эталону v2 правилом большинства (указание владельца 2026-09-24).

Голоса: выборка ``review-v2`` (``agent/r2`` и ``tasks/032/out/sub-r*``), дополнительные
пачки ``tasks/033/out/sub-r*``, dev доразмечен Codex (``codex-*``). Вердикты владельца
не учитываются. Вопрос ``ok``, если за ``ok`` большинство (3 из 5, 2 из 3 или 2 из 2).
Вопрос с двумя разными голосами ждёт третьего: ``--pending`` печатает такие вопросы.
Цель — не меньше 50 двухшаговых вопросов ``ok``.

С ``--write`` пишет ``review-v2/verdicts-majority.json`` по всем размеченным вопросам:
вердикт большинства. Если большинства нет, пишется ``ambiguous``: это
в любом случае не надёжный ``ok``, и вопрос уходит из набора.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = 50


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--pending", action="store_true", help="напечатать id вопросов с двумя разными голосами")
    args = parser.parse_args(argv)

    groups: dict[str, str] = {}
    for folder in (ROOT / "evaluation/goldsets/review-v2/packets", ROOT / "tasks/033/packets"):
        for path in folder.glob("*.json"):
            packet = json.loads(path.read_text(encoding="utf-8"))
            groups[packet["question_id"]] = packet["group"]
    files = list((ROOT / "evaluation/goldsets/review-v2/agent/r2").glob("*.jsonl"))
    files += list((ROOT / "tasks/032/out").glob("sub-r*.jsonl")) + list((ROOT / "tasks/033/out").glob("sub-r*.jsonl"))
    # dev доразмечен Codex (решение владельца 2026-09-25): 2 голоса, третий только при расхождении.
    files += list((ROOT / "tasks/033/out").glob("codex-*.jsonl"))
    votes: dict[str, list[str]] = defaultdict(list)
    for path in files:
        for line in path.read_text(encoding="utf-8-sig").splitlines():  # Codex иногда пишет BOM
            if line.strip():
                row = json.loads(line)
                votes[row["question_id"]].append(row["verdict"])

    final, table = {}, Counter()
    for qid, group in groups.items():
        # Два совпавших голоса — уже большинство из трёх; два разных ждут третьего.
        if len(votes[qid]) < 2 or (len(votes[qid]) == 2 and len(set(votes[qid])) > 1):
            continue
        verdict, count = Counter(votes[qid]).most_common(1)[0]
        final[qid] = verdict if count * 2 > len(votes[qid]) else "ambiguous"
        table[("двухшаговый" if group != "single" else "одношаговый", final[qid])] += 1
    if args.pending:
        for qid in groups:
            if len(votes[qid]) == 2 and len(set(votes[qid])) > 1:
                print(qid)
        return 0
    unanimous = sum(Counter(votes[q]).most_common(1)[0][1] == len(votes[q]) for q in final)
    print(f"размечено: {len(final)} из {len(groups)}; единогласно: {unanimous}")
    for key, count in sorted(table.items()):
        print(f"  {key[0]:12} {key[1]:18} {count}")
    ok_two_hop = table[("двухшаговый", "ok")]
    print(f"двухшаговых ok: {ok_two_hop} (цель {TARGET}) — {'достигнута' if ok_two_hop >= TARGET else 'не достигнута'}")
    if args.write:
        rows = [{"question_id": q, "verdict": v, "note": f"большинство голосов: {dict(Counter(votes[q]))}"}
                for q, v in final.items()]
        out = ROOT / "evaluation/goldsets/review-v2/verdicts-majority.json"
        out.write_text(json.dumps({"verdicts": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
