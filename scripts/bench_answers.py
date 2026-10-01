"""Ответы генератора по выдачам систем и сравнение по протоколу статей этапа 2.

Протокол записан в docs/HYPOTHESES.md (серия S, 2026-09-30) и собран из
того, как доказывали сами авторы:

* главная метрика — точность ответа: EM и F1 (SetR, 2507.06838), нормализация
  SQuAD, максимум по ответу и его синонимам (как в оценке MuSiQue);
* равный бюджет: базовые выдачи отвечают по первым ``--k`` (5, как у SetR;
  SEAL сравнивает при одинаковом k), отборщики — по выбранному множеству
  (``name=path:selected``); размер множества пишется рядом, как у SetR;
* значимость — как у SEAL-RAG (2512.10787): McNemar (точный, биномиальный)
  для EM, парный t-тест для F1, поправка Холма — Бонферрони на все сравнения
  с базой.

Генератор один на все системы: ``LLM_*`` из окружения (в сценарии —
Qwen3.5-9B в кванте на llama.cpp), температура 0. Ответы кэшируются по
системе в ``answers-<имя>.jsonl``; повторный запуск спрашивает только новое.

    python scripts/bench_answers.py --bundle artifacts/bench/musique-300 \\
        --run dense=…/rankings-dense.jsonl --run ours-current=…/rankings-ours-current.jsonl \\
        --run setr=…/rankings-ours-current+setr.jsonl:selected \\
        --baseline ours-current --out artifacts/runs/bench/musique-300/answers
"""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
import string
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Подсказка ответа короткой фразой, как в протоколах MuSiQue/HippoRAG:
# EM и F1 осмысленны только для короткого ответа.
SYSTEM = (
    "You are a reading comprehension assistant. Answer the question using the passages. "
    "Reason briefly if needed, then give the final answer on the last line as "
    "'Answer: <short answer>'. The short answer is an entity, a date, a number or yes/no — "
    "a few words, never a sentence."
)
USER = "{context}\n\nQuestion: {question}"
# Межкнижный эталон (goldset-x): ответы — развёрнутые объяснения по-русски,
# короткий ответ MuSiQue для них не годится. Строка «Answer:» сохранена,
# чтобы разбор и строгий подсчёт были одни и те же.
SYSTEM_RU = (
    "Ты отвечаешь на вопросы по фрагментам учебников математики. Опирайся только на фрагменты. "
    "При необходимости коротко порассуждай, затем дай итог последней строкой в виде "
    "'Answer: <ответ в одном-трёх предложениях по-русски>'."
)
USER_RU = "{context}\n\nВопрос: {question}"
PROMPTS = {"musique": (SYSTEM, USER), "ru": (SYSTEM_RU, USER_RU)}
_ANSWER = re.compile(r"answer\s*:\s*(.+)", re.IGNORECASE)


# ------------------------------------------------------------------ метрики

def normalize(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in set(string.punctuation))
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    return " ".join(text.split())


def exact_match(prediction: str, golds: list[str]) -> float:
    return float(any(normalize(prediction) == normalize(g) for g in golds))


def f1_score(prediction: str, golds: list[str]) -> float:
    best = 0.0
    pred = normalize(prediction).split()
    for gold in golds:
        ref = normalize(gold).split()
        common = Counter(pred) & Counter(ref)
        same = sum(common.values())
        if not pred or not ref or same == 0:
            continue
        precision, recall = same / len(pred), same / len(ref)
        best = max(best, 2 * precision * recall / (precision + recall))
    return best


def extract_answer(raw: str) -> str:
    matches = _ANSWER.findall(raw or "")
    if matches:
        text = matches[-1]
    else:
        lines = [line for line in (raw or "").strip().splitlines() if line.strip()]
        text = lines[-1] if lines else ""
    # Разметка и точка вокруг ответа: «**Leo Tolstoy**.» → «Leo Tolstoy».
    return text.strip(" \t*_.`\"'")


def final_answer(raw: str, strict: bool = True) -> str:
    """Ответ для подсчёта EM/F1.

    Строгий режим (по умолчанию, так посчитаны таблицы серии S 2026-09-30):
    ответ без строки «Answer:» считается пустым. Нестрогий берёт последнюю
    строку рассуждения — у зациклившейся модели это обрывок мысли, который
    изредка совпадает с эталоном по F1 и завышает счёт.
    """
    if strict and "answer:" not in (raw or "").lower():
        return ""
    return extract_answer(raw)


# ------------------------------------------------------------- статистика

def mcnemar_exact(base: list[float], cand: list[float]) -> dict:
    """Точный McNemar: биномиальный тест по несогласным парам."""
    b = sum(1 for x, y in zip(base, cand, strict=True) if x == 1 and y == 0)
    c = sum(1 for x, y in zip(base, cand, strict=True) if x == 0 and y == 1)
    n = b + c
    if n == 0:
        return {"worse": b, "better": c, "p": 1.0}
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return {"worse": b, "better": c, "p": min(1.0, 2 * tail)}


def _betacf(a: float, b: float, x: float) -> float:
    # Цепная дробь неполной бета-функции (Numerical Recipes, betacf).
    qab, qap, qam = a + b, a + 1, a - 1
    c, d = 1.0, 1 - qab * x / qap
    d = 1 / (d if abs(d) > 1e-300 else 1e-300)
    h = d
    for m in range(1, 300):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1 + aa * d
        d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = 1 + aa / c if abs(1 + aa / c) > 1e-300 else 1e-300
        delta = d * c
        h *= delta
        if abs(delta - 1) < 3e-12:
            break
    return h


def _betai(a: float, b: float, x: float) -> float:
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log(1 - x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1 - front * _betacf(b, a, 1 - x) / b


def paired_t(base: list[float], cand: list[float]) -> dict:
    diffs = [y - x for x, y in zip(base, cand, strict=True)]
    n = len(diffs)
    mean = statistics.fmean(diffs) if diffs else 0.0
    if n < 2:
        return {"mean_diff": mean, "t": 0.0, "p": 1.0}
    sd = statistics.stdev(diffs)
    if sd == 0:
        return {"mean_diff": mean, "t": 0.0, "p": 1.0 if mean == 0 else 0.0}
    t = mean / (sd / math.sqrt(n))
    df = n - 1
    p = _betai(df / 2, 0.5, df / (df + t * t))  # двусторонний
    return {"mean_diff": mean, "t": t, "p": p}


def holm(pvalues: dict[str, float], alpha: float = 0.05) -> dict[str, dict]:
    """Поправка Холма — Бонферрони: скорректированные p и решение."""
    order = sorted(pvalues, key=pvalues.get)
    m = len(order)
    out, running = {}, 0.0
    for rank, key in enumerate(order):
        adjusted = min(1.0, (m - rank) * pvalues[key])
        running = max(running, adjusted)
        out[key] = {"p": pvalues[key], "p_holm": running, "significant": running < alpha}
    return out


def retrieval_row(gold: set[str], ranked: list[str], context: list[str]) -> dict[str, float]:
    """Объясняющие метрики поиска, как в статьях: P/R@1,3,5 (SEAL),
    NDCG@10 и MRR@10 (SetR) по порядку системы; ctx_* — по тому, что
    реально прочёл генератор (множество отборщика или первые k)."""
    row: dict[str, float] = {}
    for k in (1, 3, 5):
        hit = len(gold & set(ranked[:k]))
        row[f"P@{k}"] = hit / k
        row[f"R@{k}"] = hit / len(gold)
    dcg = sum(1 / math.log2(i + 2) for i, cid in enumerate(ranked[:10]) if cid in gold)
    ideal = sum(1 / math.log2(i + 2) for i in range(min(10, len(gold))))
    row["NDCG@10"] = dcg / ideal
    first = next((i + 1 for i, cid in enumerate(ranked[:10]) if cid in gold), None)
    row["MRR@10"] = 1 / first if first else 0.0
    hit = len(gold & set(context))
    row["ctx_precision"] = hit / len(context) if context else 0.0
    row["ctx_recall"] = hit / len(gold)
    row["ctx_all"] = float(hit == len(gold))
    return row


def load_aliases(path: Path | None) -> dict[str, list[str]]:
    """Синонимы ответа из исходного файла MuSiQue (``answer_aliases``);
    в набор бенча они не входят, а оценка MuSiQue берёт максимум по ним."""
    if path is None:
        return {}
    items = json.loads(path.read_text(encoding="utf-8"))
    return {item["id"]: list(item.get("answer_aliases") or []) for item in items}


# ------------------------------------------------------------------ прогон

def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def context_ids(row: dict, k: int, selected: bool) -> tuple[list[str], str]:
    """Что читает генератор и откуда: множество отборщика или первые k."""
    if selected:
        chosen = row.get("selected") or []
        if chosen:
            return list(chosen), "selected"
        # Отказ или пустой выбор — порядок реранкера, как в конвейере;
        # доля таких вопросов пишется в сводку.
        return row["ranked"][:k], "fallback"
    return row["ranked"][:k], "top_k"


def _mean_retrieval(rows) -> dict[str, float]:
    measured = [r["retrieval"] for r in rows if "retrieval" in r]
    if not measured:
        return {}
    return {key: round(statistics.fmean(r[key] for r in measured), 4) for key in measured[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--run", action="append", required=True,
                        help="имя=rankings.jsonl, с суффиксом :selected — по множеству отборщика")
    parser.add_argument("--baseline", required=True, help="с чем сравнивать остальные")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--chars", type=int, default=3000, help="предел знаков на фрагмент")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--aliases", type=Path, default=None,
                        help="musique.json с answer_aliases (HippoRAG 2, dataset/)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt", choices=sorted(PROMPTS), default="musique",
                        help="musique — короткий ответ; ru — развёрнутый по-русски (goldset-x)")
    parser.add_argument("--lenient-answer", action="store_true",
                        help="без строки «Answer:» брать последнюю строку (так считалось до 2026-09-30)")
    args = parser.parse_args()

    from rag_textbook.clients.llm import ChatMessage, build_llm_client
    from rag_textbook.config import Settings

    settings = Settings()
    llm = build_llm_client(settings.llm)
    model = settings.llm.model_for("chat")
    chunks = {row["chunk_id"]: row for row in load_jsonl(args.bundle / "chunks.jsonl")}
    questions = {row["qid"]: row for row in load_jsonl(args.bundle / "questions.jsonl")}
    aliases = load_aliases(args.aliases)
    if aliases:
        print(f"синонимы ответа: {sum(1 for q in questions if aliases.get(q))} вопросов из {len(questions)}")
    args.out.mkdir(parents=True, exist_ok=True)

    runs: dict[str, tuple[Path, bool]] = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        selected = path.endswith(":selected")
        runs[name] = (Path(path.removesuffix(":selected")), selected)
    if args.baseline not in runs:
        parser.error(f"база {args.baseline} не среди --run")
    print(f"генератор {model}, k={args.k}, систем {len(runs)}")

    def ask(question: str, ids: list[str]) -> str:
        context = "\n\n".join(
            f"[{i}] {chunks[cid]['text'][: args.chars]}" for i, cid in enumerate(ids, start=1)
        )
        system, user = PROMPTS[args.prompt]
        messages = [ChatMessage(role="system", content=system),
                    ChatMessage(role="user", content=user.format(context=context, question=question))]
        return llm.chat(messages, purpose="chat", max_tokens=args.max_tokens, temperature=0.0)

    scores: dict[str, dict[str, dict]] = {}
    summary: dict[str, dict] = {}
    for name, (path, selected) in runs.items():
        rankings = [r for r in load_jsonl(path) if r["qid"] in questions]
        if args.limit:
            rankings = rankings[: args.limit]
        cache = args.out / f"answers-{name}.jsonl"
        done = {r["qid"]: r for r in load_jsonl(cache)} if cache.exists() else {}
        todo = [r for r in rankings if r["qid"] not in done]
        print(f"{name}: вопросов {len(rankings)}, из кэша {len(done)}", flush=True)

        def one(row: dict, selected: bool = selected) -> dict:
            q = questions[row["qid"]]
            ids, source = context_ids(row, args.k, selected)
            ids = [cid for cid in ids if cid in chunks]
            started = time.perf_counter()
            try:
                raw = ask(q["question"], ids)
                error = ""
            except Exception as exc:  # noqa: BLE001
                raw, error = "", str(exc)[:200]
            return {"qid": row["qid"], "ids": ids, "source": source, "raw": raw[-2000:],
                    "error": error, "ms": round(1000 * (time.perf_counter() - started), 1)}

        with cache.open("a", encoding="utf-8") as handle, ThreadPoolExecutor(args.workers) as pool:
            for index, record in enumerate(pool.map(one, todo), start=1):
                done[record["qid"]] = record
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                handle.flush()
                if index % 50 == 0:
                    print(f"  {index}/{len(todo)}", flush=True)

        per_q: dict[str, dict] = {}
        for row in rankings:
            record = done[row["qid"]]
            q = questions[row["qid"]]
            golds = [q["answer"], *q.get("answer_aliases", []), *aliases.get(row["qid"], [])]
            prediction = final_answer(record["raw"], strict=not args.lenient_answer)
            per_q[row["qid"]] = {"em": exact_match(prediction, golds), "f1": f1_score(prediction, golds),
                                 "size": len(record["ids"]), "source": record["source"],
                                 "error": bool(record["error"]), "type": q.get("type", ""),
                                 "prediction": prediction}
            gold = set(q.get("gold_chunk_ids") or [])
            if gold:
                per_q[row["qid"]]["retrieval"] = retrieval_row(gold, row["ranked"], record["ids"])
        scores[name] = per_q
        n = max(1, len(per_q))
        sources = Counter(v["source"] for v in per_q.values())
        by_type: dict[str, list[dict]] = defaultdict(list)
        for v in per_q.values():
            by_type[v["type"]].append(v)
        summary[name] = {
            "n": len(per_q),
            "em": round(sum(v["em"] for v in per_q.values()) / n, 4),
            "f1": round(sum(v["f1"] for v in per_q.values()) / n, 4),
            "context_size": round(sum(v["size"] for v in per_q.values()) / n, 2),
            "sources": {k: round(c / n, 3) for k, c in sources.items()},
            "errors": sum(1 for v in per_q.values() if v["error"]),
            "retrieval": _mean_retrieval(per_q.values()),
            "by_type": {t: {"n": len(vs), "em": round(statistics.fmean(v["em"] for v in vs), 4),
                            "f1": round(statistics.fmean(v["f1"] for v in vs), 4)}
                        for t, vs in sorted(by_type.items())},
        }
        # Правило проекта: читать генерации до замера — три ответа в журнал.
        for record in list(done.values())[:3]:
            print(f"  образец [{name}] {record['qid']}: {record['raw'][-200:]!r}")

    base = scores[args.baseline]
    comparisons: dict[str, dict] = {}
    raw_p: dict[str, float] = {}
    for name, per_q in scores.items():
        if name == args.baseline:
            continue
        shared = sorted(set(base) & set(per_q))
        em = mcnemar_exact([base[q]["em"] for q in shared], [per_q[q]["em"] for q in shared])
        f1 = paired_t([base[q]["f1"] for q in shared], [per_q[q]["f1"] for q in shared])
        comparisons[name] = {"shared": len(shared), "em": em, "f1": f1,
                             "em_diff": round(statistics.fmean(per_q[q]["em"] - base[q]["em"]
                                                               for q in shared), 4) if shared else 0.0}
        raw_p[f"{name}:em"] = em["p"]
        raw_p[f"{name}:f1"] = f1["p"]
    corrected = holm(raw_p)
    for key, value in corrected.items():
        name, metric = key.rsplit(":", 1)
        comparisons[name][metric]["p_holm"] = value["p_holm"]
        comparisons[name][metric]["significant"] = value["significant"]

    report = {"generator": model, "k": args.k, "baseline": args.baseline,
              "summary": summary, "vs_baseline": comparisons}
    (args.out / "answers-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n{'система':24s} {'EM':>6s} {'F1':>6s} {'|ctx|':>6s}  против {args.baseline}")
    for name, s in summary.items():
        line = f"{name:24s} {s['em']:6.3f} {s['f1']:6.3f} {s['context_size']:6.2f}"
        if name in comparisons:
            c = comparisons[name]
            line += (f"  EM {c['em_diff']:+.3f} (+{c['em']['better']}/−{c['em']['worse']},"
                     f" p_holm {c['em']['p_holm']:.3g}{' *' if c['em']['significant'] else ''})"
                     f"  F1 {c['f1']['mean_diff']:+.3f} (p_holm {c['f1']['p_holm']:.3g}"
                     f"{' *' if c['f1']['significant'] else ''})")
        print(line)
    print(f"\n{'поиск':24s} {'R@5':>6s} {'P@5':>6s} {'NDCG10':>7s} {'MRR10':>6s} {'ctxR':>6s} {'ctxAll':>6s}")
    for name, s in summary.items():
        r = s["retrieval"]
        if r:
            print(f"{name:24s} {r['R@5']:6.3f} {r['P@5']:6.3f} {r['NDCG@10']:7.3f} {r['MRR@10']:6.3f}"
                  f" {r['ctx_recall']:6.3f} {r['ctx_all']:6.3f}")
    # Отказ вызова засчитывается неверным ответом и сдвигает сравнение:
    # больше 2% у любой системы — прогон недействителен (сервер генерации падал).
    bad = {name: s["errors"] for name, s in summary.items() if s["errors"] > 0.02 * max(1, s["n"])}
    if bad:
        print(f"СТОП: ошибки вызова генератора {bad} — прогон недействителен", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
