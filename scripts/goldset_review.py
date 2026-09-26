"""Приёмка эталона v2 владельцем между днями аренды (M4, ручная выборка).

    # лист для чтения и шаблон вердиктов
    python scripts/goldset_review.py sheet --goldset evaluation/goldsets/goldset-v2.json \\
        --parsed artifacts/runs/day2/ru-parsed --out evaluation/goldsets/review-v2

    # после заполнения review-v2/verdicts.json
    python scripts/goldset_review.py accept --goldset evaluation/goldsets/goldset-v2.json \\
        --verdicts evaluation/goldsets/review-v2/verdicts.json

Выборка: 30 связывающих, 10 межкнижных, 10 одношаговых (по умолчанию),
только из части test — её принимает вердикт серии К. Вердикты — те же,
что у r2 (``rag_textbook/evaluation/verdicts.py``): ok, single_hop_enough,
unanswerable, ambiguous, leaky.

Порог принятия записан до чтения (docs/SERVER-DAY-2.md, «Между днями»):

- среди просмотренных двухшаговых (связывающие и межкнижные)
  ``single_hop_enough`` не больше 30 % — у r2 было 50 %, абляция
  обязана это исправить;
- негодных (unanswerable, ambiguous, leaky) по всей выборке
  не больше 10 %.

Принятие применяет вердикты, **удаляет** негодные вопросы из набора
(у двухшаговых с single_hop_enough ставит expected_hops=1 и снимает срез
связывающих — такой вопрос мерит не то) и пишет рядом отметку
``<набор>.accepted`` в формате sha256sum: день 3 сверяет её в E0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, get_args

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.goldset import load_goldset, save_goldset  # noqa: E402
from rag_textbook.evaluation.verdicts import (  # noqa: E402
    USABLE,
    Verdict,
    VerdictSet,
    apply_verdicts,
)

MAX_SINGLE_HOP_SHARE = 0.30
MAX_UNUSABLE_SHARE = 0.10
TWO_HOP_SLICES = ("linking", "cross_book")
# Блок $$…$$ посреди строки KaTeX не разбирает («Can't use function '$'»),
# поэтому формула выносится в отдельный абзац. Фрагменты не обрезаются:
# обрезка рвала формулы, а самый длинный фрагмент — около 1800 знаков.
DISPLAY_MATH = re.compile(r"\$\$(.+?)\$\$", re.DOTALL)


def group_of(question: Any) -> str:
    if question.slice == "cross_book":
        return "cross_book"
    if question.slice == "linking" or question.expected_hops > 1:
        return "linking"
    return "single"


def pick(questions: list[Any], counts: dict[str, int], seed: int, split: str) -> list[Any]:
    rng = random.Random(seed)
    chosen = []
    for group, count in counts.items():
        pool = [q for q in questions if group_of(q) == group and (not split or q.split == split)]
        pool.sort(key=lambda q: q.id)
        chosen.extend(rng.sample(pool, min(count, len(pool))))
    return chosen


def load_texts(parsed: Path) -> dict[str, str]:
    texts: dict[str, str] = {}
    for path in sorted(parsed.glob("*_chunks.json")):
        raw = json.loads(path.read_text(encoding="utf-8"))
        for row in raw if isinstance(raw, list) else raw.get("chunks", []):
            texts[row["id"]] = row["text"]
    return texts


def load_ablation(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return {row["question_id"]: row for row in rows}


def _looks_like_formula_tail(text: str) -> bool:
    return not re.search(r"[А-Яа-яЁё]{3}", text) and bool(re.search(r"[{}^_\\]", text))


def fragment_markdown(text: str) -> str:
    """Текст фрагмента для просмотра: каждая формула $$…$$ — отдельным абзацем.

    Нарезка иногда начинает или обрывает фрагмент посреди строчной формулы.
    Непарный ``$`` тогда сдвигает разметку всего фрагмента, поэтому он
    экранируется, а место помечается для проверяющего.
    """
    warnings = []
    blocks: list[str] = []

    def block(match: re.Match[str]) -> str:
        blocks.append(re.sub(r"\n\s*\n", "\n", match.group(1).strip()))
        return f"\x00{len(blocks) - 1}\x00"

    text = DISPLAY_MATH.sub(block, text)
    if "$$" in text:
        warnings.append("блок формулы не закрыт")
        text = text.replace("$$", r"\$\$")
    singles = [m.start() for m in re.finditer(r"(?<![\\$])\$(?!\$)", text)]
    if singles and _looks_like_formula_tail(text[:singles[0]]):
        warnings.append("начинается посреди формулы")
        text = text[:singles[0]] + "\\" + text[singles[0]:]
        singles = [m.start() for m in re.finditer(r"(?<![\\$])\$(?!\$)", text)]
    if len(singles) % 2:
        warnings.append("обрывается посреди формулы")
        text = text[:singles[-1]] + "\\" + text[singles[-1]:]
    text = re.sub("\x00(\\d+)\x00", lambda m: "\n\n$$\n" + blocks[int(m.group(1))] + "\n$$\n\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    if warnings:
        text = f"*⚠ Нарезка: {', '.join(warnings)}.*\n\n" + text
    return text


def render(questions: list[Any], texts: dict[str, str], ablation: dict[str, dict[str, Any]]) -> str:
    lines = [
        "# Приёмка эталона v2: ручная выборка",
        "",
        "Для каждого вопроса впишите вердикт в `verdicts.json` рядом:",
        "`ok` — годен как есть; `single_hop_enough` — хватает одного фрагмента;",
        "`unanswerable` — из фрагментов не следует; `ambiguous` — ответов несколько;",
        "`leaky` — ответ подсказан в вопросе. Проверьте, что второй фрагмент —",
        "не оглавление и не список упражнений.",
        "",
    ]
    for number, question in enumerate(questions, 1):
        lines += [
            f"## {number}. `{question.id}` — {group_of(question)}, {question.question_type}, "
            f"источник пары: {question.pair_source or '—'}",
            "",
            f"**Вопрос.** {question.question}",
            "",
            f"**Ответ.** {question.answer}",
            "",
        ]
        if question.id in ablation:
            row = ablation[question.id]
            lines += [f"Абляция: `{row.get('verdict')}`, по отдельности {row.get('single_matches')}, "
                      f"вместе {row.get('joint_match')}", ""]
        for chunk_id in question.gold_chunk_ids:
            text = texts.get(chunk_id, "(текста нет в разборе)")
            lines += [f"#### Фрагмент `{chunk_id}`", "", fragment_markdown(text), "", "---", ""]
    return "\n".join(lines)


def sheet(args: argparse.Namespace) -> int:
    questions = load_goldset(args.goldset)
    counts = {"linking": args.linking, "cross_book": args.cross, "single": args.single}
    chosen = pick(questions, counts, args.seed, args.split)
    if not chosen:
        print("Выборка пуста: нет вопросов нужной части", file=sys.stderr)
        return 1
    texts = load_texts(args.parsed)
    missing = [cid for q in chosen for cid in q.gold_chunk_ids if cid not in texts]
    if missing:
        print(f"Нет текстов {len(missing)} фрагментов в {args.parsed}: {missing[:5]}", file=sys.stderr)
        return 1
    ablation = load_ablation(args.ablation or Path(str(args.goldset) + ".ablation.jsonl"))
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "sheet.md").write_text(render(chosen, texts, ablation), encoding="utf-8")
    template = args.out / "verdicts.json"
    if template.exists():
        print(f"{template} уже есть — не перезаписываю", file=sys.stderr)
    else:
        payload = {"verdicts": [{"question_id": q.id, "verdict": "", "note": ""} for q in chosen]}
        template.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    by_group: dict[str, int] = {}
    for question in chosen:
        by_group[group_of(question)] = by_group.get(group_of(question), 0) + 1
    print(f"Выборка {len(chosen)}: {by_group}; лист {args.out / 'sheet.md'}")
    return 0


def decide(questions: list[Any], verdicts: dict[str, str]) -> dict[str, Any]:
    seen = [q for q in questions if q.id in verdicts]
    two_hop = [q for q in seen if group_of(q) in TWO_HOP_SLICES]
    single_hop = sum(1 for q in two_hop if verdicts[q.id] == "single_hop_enough")
    unusable = sum(1 for q in seen if verdicts[q.id] not in USABLE)
    single_share = single_hop / len(two_hop) if two_hop else 0.0
    unusable_share = unusable / len(seen) if seen else 1.0
    problems = []
    if single_share > MAX_SINGLE_HOP_SHARE:
        problems.append(f"одного фрагмента хватает {single_share:.0%} двухшаговых > {MAX_SINGLE_HOP_SHARE:.0%}")
    if unusable_share > MAX_UNUSABLE_SHARE:
        problems.append(f"негодных {unusable_share:.0%} > {MAX_UNUSABLE_SHARE:.0%}")
    return {"просмотрено": len(seen), "двухшаговых": len(two_hop),
            "single_hop_enough": single_hop, "негодных": unusable,
            "доля одношаговых среди двухшаговых": round(single_share, 3),
            "доля негодных": round(unusable_share, 3),
            "проблемы": problems, "принят": bool(seen) and not problems}


def accept(args: argparse.Namespace) -> int:
    questions = load_goldset(args.goldset)
    raw = json.loads(args.verdicts.read_text(encoding="utf-8"))
    empty = [row["question_id"] for row in raw.get("verdicts", []) if not row.get("verdict")]
    if empty:
        print(f"Не заполнено {len(empty)} вердиктов: {empty[:5]}", file=sys.stderr)
        return 1
    allowed = set(get_args(Verdict))
    wrong = [row["question_id"] for row in raw.get("verdicts", []) if row["verdict"] not in allowed]
    if wrong:
        print(f"Вердикты вне {sorted(allowed)}: {wrong[:5]}", file=sys.stderr)
        return 1
    verdict_set = VerdictSet.load(args.verdicts)
    known = {q.id for q in questions}
    unknown = [qid for qid in verdict_set.verdicts if qid not in known]
    if unknown:
        print(f"Вердикты к вопросам, которых нет в наборе: {unknown[:5]}", file=sys.stderr)
        return 1
    verdicts = {qid: v.verdict for qid, v in verdict_set.verdicts.items()}
    marker = Path(str(args.goldset).removesuffix(".json") + ".accepted")
    if args.filtered:
        # Решение владельца 2026-09-25: пороги по выборке не применяются. Вместо этого
        # набор фильтруется по вердиктам, а непроверенные двухшаговые удаляются
        # (годна примерно половина таких). Непроверенные одношаговые остаются.
        print(f"Фильтрация по {len(verdicts)} вердиктам, пороги выборки не применяются")
    else:
        result = decide(questions, verdicts)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["принят"]:
            marker.unlink(missing_ok=True)
            print("Эталон НЕ принят: день 3 не ставить (docs/SERVER-DAY-2.md)")
            return 1
    updated, _ = apply_verdicts(questions, verdict_set)
    kept, dropped = [], Counter()
    for question in updated:
        verdict = verdicts.get(question.id)
        if verdict is None and args.filtered and group_of(question) in TWO_HOP_SLICES:
            dropped["двухшаговый без вердикта"] += 1
            continue
        if verdict is not None and verdict not in USABLE:
            dropped[verdict] += 1
            continue
        if verdict == "single_hop_enough" and group_of(question) in TWO_HOP_SLICES:
            # Тип тоже сбрасывается: K4 в day3.sh и series_k_verdict.py считают
            # связывающим и вопрос с question_type=multi_hop, даже если срез single.
            question = question.model_copy(update={"expected_hops": 1, "slice": "single",
                                                   "question_type": "single_chunk"})
        kept.append(question)
    composition = Counter((q.split, group_of(q)) for q in kept)
    print(f"Останется {len(kept)} из {len(questions)}; удалено: {dict(dropped)}")
    print("Состав:", {f"{s}/{g}": n for (s, g), n in sorted(composition.items())})
    if args.dry_run:
        print("Сухой прогон: набор и отметка не записаны")
        return 0
    save_goldset(kept, args.goldset)
    digest = hashlib.sha256(args.goldset.read_bytes()).hexdigest()
    # Формат sha256sum: путь относительно корня репозитория, как его сверяет day3.sh.
    try:
        relative = args.goldset.resolve().relative_to(ROOT.resolve()).as_posix()
    except ValueError:
        relative = args.goldset.as_posix()
    marker.write_text(f"{digest}  {relative}\n", encoding="utf-8", newline="\n")
    print(f"Принят: {len(kept)} вопросов (удалено {len(updated) - len(kept)}), отметка {marker}")
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # консоль Windows — cp1251
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    make = sub.add_parser("sheet")
    make.add_argument("--goldset", type=Path, required=True)
    make.add_argument("--parsed", type=Path, required=True, help="каталог *_chunks.json")
    make.add_argument("--out", type=Path, required=True)
    make.add_argument("--ablation", type=Path)
    make.add_argument("--linking", type=int, default=30)
    make.add_argument("--cross", type=int, default=10)
    make.add_argument("--single", type=int, default=10)
    make.add_argument("--split", default="test")
    make.add_argument("--seed", type=int, default=20260922)
    take = sub.add_parser("accept")
    take.add_argument("--goldset", type=Path, required=True)
    take.add_argument("--verdicts", type=Path, required=True)
    take.add_argument("--dry-run", action="store_true")
    take.add_argument("--filtered", action="store_true",
                      help="фильтровать по вердиктам без порогов выборки (решение владельца 2026-09-25)")
    args = parser.parse_args(argv)
    return sheet(args) if args.mode == "sheet" else accept(args)


if __name__ == "__main__":
    raise SystemExit(main())
