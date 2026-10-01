"""Свод вердиктов агентов по приёмке v2: итог по порогам и устойчивость.

Читает вердикты владельца (``verdicts.json``) и два прогона агентов
(``agent/r1``, ``agent/r2``). Владелец главнее агента. Печатает доли
по порогам приёмки, какие вопросы решают исход и где прогоны расходятся.
С ``--sheet`` пишет короткий лист только этих вопросов для человека.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.goldset_review import (  # noqa: E402
    MAX_SINGLE_HOP_SHARE,
    MAX_UNUSABLE_SHARE,
    fragment_markdown,
)

UNUSABLE = {"unanswerable", "ambiguous", "leaky"}


def read_jsonl(folder: Path) -> dict[str, dict]:
    rows = {}
    for path in sorted(folder.glob("*.jsonl")):
        if path.name.startswith("calibration"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, default=ROOT / "evaluation/goldsets/review-v2")
    parser.add_argument("--sheet", type=Path)
    args = parser.parse_args(argv)

    owner = {r["question_id"]: r for r in json.loads((args.review / "verdicts.json").read_text(encoding="utf-8"))["verdicts"]
             if r["verdict"]}
    r1, r2 = read_jsonl(args.review / "agent/r1"), read_jsonl(args.review / "agent/r2")
    packets = {json.loads(p.read_text(encoding="utf-8"))["question_id"]: json.loads(p.read_text(encoding="utf-8"))
               for p in sorted((args.review / "packets").glob("*.json"))}

    final = {}
    for qid in packets:
        final[qid] = owner[qid]["verdict"] if qid in owner else r2[qid]["verdict"]
    two_hop = [q for q, p in packets.items() if p["group"] != "single"]
    single = sum(final[q] == "single_hop_enough" for q in two_hop)
    unusable = sum(final[q] in UNUSABLE for q in packets)
    print(f"Владелец: {len(owner)}, агент r2: {len(packets) - len(owner)}")
    print(f"single_hop_enough: {single}/{len(two_hop)} = {single / len(two_hop):.0%} (порог {MAX_SINGLE_HOP_SHARE:.0%})")
    print(f"негодных: {unusable}/{len(packets)} = {unusable / len(packets):.0%} (порог {MAX_UNUSABLE_SHARE:.0%})")
    need_single = single - int(MAX_SINGLE_HOP_SHARE * len(two_hop))
    need_unusable = unusable - int(MAX_UNUSABLE_SHARE * len(packets))
    print(f"для прохода надо опровергнуть: single — {max(need_single, 0)}, негодных — {max(need_unusable, 0)}")

    disputed = []
    for qid in packets:
        if qid in owner:
            continue
        a, b = r1.get(qid, {}).get("verdict"), r2[qid]["verdict"]
        low = r2[qid].get("confidence") == "low"
        if b != "ok" or low or a != b:
            disputed.append(qid)

    def tier(qid: str) -> int:
        # 0 — решают исход: не ok и агент колебался; 1 — не ok уверенно; 2 — ok, но колебался.
        verdict, low = r2[qid]["verdict"], r2[qid].get("confidence") == "low"
        if verdict != "ok":
            return 0 if low or r1.get(qid, {}).get("verdict") != verdict else 1
        return 2

    disputed.sort(key=tier)
    stable = sum(r1.get(q, {}).get("verdict") == r2[q]["verdict"] for q in r2)
    print(f"r1 и r2 совпали: {stable}/{len(r2)}; на решение человека: {len(disputed)}")
    for qid in disputed:
        print(f"  {qid} {packets[qid]['group']:10} r1={r1.get(qid, {}).get('verdict')} r2={r2[qid]['verdict']}"
              f" ({r2[qid].get('confidence')})")

    if args.sheet:
        lines = ["# Приёмка v2: вопросы, которые решает человек", "",
                 "Здесь только вопросы, где агент поставил не `ok`, колебался (`low`) или два прогона",
                 "агентов разошлись. Остальные агент принял как `ok` уверенно и одинаково дважды.",
                 "Вердикт вписать в `verdicts.json` (строка с тем же id). Правила — `RULES.md`.", ""]
        titles = {0: "# А. Решают исход: агент поставил не ok и колебался или прогоны разошлись",
                  1: "# Б. Агент уверенно и дважды одинаково: не ok",
                  2: "# В. Агент поставил ok, но колебался (важно, только если набор проходит)"}
        current = None
        for number, qid in enumerate(disputed, 1):
            p = packets[qid]
            if tier(qid) != current:
                current = tier(qid)
                lines += [titles[current], ""]
            lines += [f"## {number}. `{qid}` — {p['group']}", "",
                      f"**Агент:** r2 `{r2[qid]['verdict']}` ({r2[qid].get('confidence')}), "
                      f"r1 `{r1.get(qid, {}).get('verdict')}`. {r2[qid].get('note', '')}", "",
                      f"**Вопрос.** {p['question']}", "", f"**Ответ.** {p['answer']}", ""]
            for fragment in p["fragments"]:
                lines += [f"#### Фрагмент `{fragment['chunk_id']}`", "", fragment_markdown(fragment["text"]), "", "---", ""]
        args.sheet.write_text("\n".join(lines), encoding="utf-8")
        print(f"Лист: {args.sheet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
