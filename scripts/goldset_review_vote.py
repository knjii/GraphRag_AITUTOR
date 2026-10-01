"""Голосование по приёмке v2: пять прогонов Codex и пять прогонов субагента.

Голоса: ``tasks/032/out/codex-r*-b*.jsonl`` и ``sub-r*-b*.jsonl``, первым
прогоном субагента считается ``review-v2/agent/r2`` (те же правила).
Правило записано до свода (2026-09-24, указание владельца): вердикт
принимается, если за него не меньше 4/5 голосов, иначе вопрос спорный
и уходит владельцу. Две системы голосуют отдельно: вердикт принят, только
если каждая дала за него ≥ 4/5 своих голосов. Вердикт владельца главнее.

Пишет ``verdicts-vote.json`` (принятые вердикты, спорные пусты) и лист
``sheet-vote.md`` со спорными вопросами. Печатает, решают ли спорные исход.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.goldset_review import (  # noqa: E402
    MAX_SINGLE_HOP_SHARE,
    MAX_UNUSABLE_SHARE,
    fragment_markdown,
)

UNUSABLE = {"unanswerable", "ambiguous", "leaky"}
QUORUM = 0.8


def read_rows(paths: list[Path]) -> dict[str, dict]:
    rows = {}
    for path in paths:
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                row = json.loads(line)
                rows[row["question_id"]] = row
    return rows


def collect(votes_dir: Path, agent_r2: Path) -> dict[str, list[dict[str, dict]]]:
    runs: dict[str, list[dict[str, dict]]] = {"codex": [], "sub": [read_rows(sorted(agent_r2.glob("*.jsonl")))]}
    for system in runs:
        for r in range(1, 6):
            files = sorted(votes_dir.glob(f"{system}-r{r}-b*.jsonl"))
            if files:
                runs[system].append(read_rows(files))
    return runs


def consensus(votes: list[str]) -> str | None:
    if not votes:
        return None
    verdict, count = Counter(votes).most_common(1)[0]
    return verdict if count >= QUORUM * len(votes) else None


def shares(final: dict[str, str | None], packets: dict[str, dict]) -> tuple[int, int, int, int]:
    two_hop = [q for q, p in packets.items() if p["group"] != "single"]
    return (sum(final[q] == "single_hop_enough" for q in two_hop), len(two_hop),
            sum(final[q] in UNUSABLE for q in packets), len(packets))


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--review", type=Path, default=ROOT / "evaluation/goldsets/review-v2")
    parser.add_argument("--votes", type=Path, default=ROOT / "tasks/032/out")
    parser.add_argument("--min-runs", type=int, default=5, help="сколько прогонов нужно каждой системе")
    parser.add_argument("--write", action="store_true", help="записать verdicts-vote.json и sheet-vote.md")
    args = parser.parse_args(argv)

    packets = {}
    for path in sorted((args.review / "packets").glob("*.json")):
        packet = json.loads(path.read_text(encoding="utf-8"))
        packet["number"] = path.name[:2]
        packets[packet["question_id"]] = packet
    owner = {r["question_id"]: r for r in json.loads((args.review / "verdicts.json").read_text(encoding="utf-8"))["verdicts"]
             if r["verdict"]}
    runs = collect(args.votes, args.review / "agent/r2")
    for system, found in runs.items():
        full = sum(len(run) == len(packets) for run in found)
        print(f"{system}: прогонов {len(found)}, полных {full}")
        if full < args.min_runs:
            print(f"  мало полных прогонов (нужно {args.min_runs})", file=sys.stderr)

    votes = {q: {s: [run[q]["verdict"] for run in found if q in run] for s, found in runs.items()} for q in packets}
    final: dict[str, str | None] = {}
    source: dict[str, str] = {}
    for qid in packets:
        by_system = {s: consensus(v) for s, v in votes[qid].items()}
        agreed = by_system["codex"] if by_system["codex"] == by_system["sub"] else None
        if qid in owner:
            final[qid], source[qid] = owner[qid]["verdict"], "владелец"
            mark = "совпало" if agreed == final[qid] else f"голосование: {agreed or 'спорно'}"
            print(f"  {packets[qid]['number']} {qid} владелец={final[qid]:18} {mark}  {dict(votes[qid])}")
        else:
            final[qid], source[qid] = agreed, "голосование" if agreed else "спорно"

    # Первыми — спорные, где за ok не голосовал никто: они решают исход быстрее всего.
    disputed = sorted((q for q in packets if final[q] is None),
                      key=lambda q: sum(v == "ok" for v in sum(votes[q].values(), [])))
    single, two_hop, unusable, total = shares(final, packets)
    print(f"Принято голосованием: {sum(s == 'голосование' for s in source.values())}, спорных: {len(disputed)}")
    print(f"single_hop_enough (без спорных): {single}/{two_hop} = {single / two_hop:.0%} (порог {MAX_SINGLE_HOP_SHARE:.0%})")
    print(f"негодных (без спорных): {unusable}/{total} = {unusable / total:.0%} (порог {MAX_UNUSABLE_SHARE:.0%})")
    # Худший и лучший исход по спорным: решают ли они приёмку.
    worst = dict(final)
    for q in disputed:
        worst[q] = "single_hop_enough" if packets[q]["group"] != "single" else "unanswerable"
    best = {q: final[q] or "ok" for q in packets}
    for name, case in (("лучший", best), ("худший", worst)):
        s, th, u, t = shares(case, packets)
        passed = s / th <= MAX_SINGLE_HOP_SHARE and u / t <= MAX_UNUSABLE_SHARE
        print(f"  {name} исход по спорным: single {s / th:.0%}, негодных {u / t:.0%} — {'проходит' if passed else 'не проходит'}")
    for q in disputed:
        print(f"  спорно {packets[q]['number']} {q} {packets[q]['group']:10} {dict(votes[q])}")

    if args.write:
        rows = [{"question_id": q, "verdict": final[q] or "",
                 "note": owner[q].get("note", "") if q in owner else
                 (f"голосование: {dict(Counter(sum(votes[q].values(), [])))}" if final[q] else "")}
                for q in packets]
        (args.review / "verdicts-vote.json").write_text(
            json.dumps({"verdicts": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
        lines = ["# Приёмка v2: спорные вопросы после голосования", "",
                 "Каждый вопрос размечен пять раз Codex и пять раз субагентом. Здесь только те,",
                 "где хотя бы одна система не дала одному вердикту 4/5 голосов или системы разошлись.",
                 "Первыми идут вопросы, где за `ok` не голосовал никто.",
                 "Вердикт вписать в `verdicts-vote.json` (строка с тем же id). Правила — `RULES.md`.", ""]
        for number, qid in enumerate(disputed, 1):
            p = packets[qid]
            lines += [f"## {number}. `{qid}` — {p['group']} (пакет {p['number']})", ""]
            for system, found in runs.items():
                counts = Counter(votes[qid][system])
                lines.append(f"- **{system}:** " + ", ".join(f"`{v}` ×{c}" for v, c in counts.most_common()))
            for system, found in runs.items():
                notes = {run[qid]["verdict"]: run[qid].get("note", "") for run in found if qid in run}
                for verdict, note in notes.items():
                    lines.append(f"  - {system} за `{verdict}`: {note}")
            lines += ["", f"**Вопрос.** {p['question']}", "", f"**Ответ.** {p['answer']}", ""]
            for fragment in p["fragments"]:
                lines += [f"#### Фрагмент `{fragment['chunk_id']}`", "", fragment_markdown(fragment["text"]), "", "---", ""]
        (args.review / "sheet-vote.md").write_text("\n".join(lines), encoding="utf-8")
        print(f"Записано: verdicts-vote.json, sheet-vote.md ({len(disputed)} спорных)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
