"""Офлайн-часть серии L: отсев A1, контроль по графу, смесь Б1 и итоговый вердикт.

Всё считается на ноутбуке по сохранённым выдачам и ответам, без карты.

``coverage``    доля вопросов, у которых весь эталон в первых k (ctxAll@k),
                и сколько из «оборванных цепочек» (часть эталона в k,
                остальное в пуле) закрыто — отсев A1 (≥ 0.430);
``graphchain``  контроль без модели: та же жадная цепочка, что у L1,
                но кандидат — фрагмент с наибольшим весом общих сущностей
                с последним добавленным (IDF по графу, узлы-хабы степени
                > 64 не считаются — тот же порог, что у обхода);
``gatemix``     Б1: SEAL на доле 0.67 вопросов с наименьшей вероятностью
                «хватает», остальным — ответ базы. Ответы сохранены,
                генерация детерминирована — смесь точна. Критерий владельца:
                не хуже SEAL-на-всех с запасом 0.05, EM выше SetR, медианная
                задержка поиска ниже SEAL (чистый замер P1);
``verdict``     все вердикты серии L по записанным до замера критериям —
                по файлам, вернувшимся с сервера.

    python scripts/laya_offline.py coverage --bundle artifacts/bench/musique-300 \
        --run base=…/rankings-ours-current.jsonl --run chain=…/rankings-ours-current+chain.jsonl
    python scripts/laya_offline.py graphchain --bundle … --graph artifacts/graphs/bench-musique-300/current.json.gz \
        --rankings …/rankings-ours-current.jsonl --out …/rankings-ours-current+graphchain.jsonl
    python scripts/laya_offline.py verdict --bundle artifacts/bench/musique-300 \
        --run artifacts/runs/bench/musique-300 --aliases artifacts/bench/musique-300-aliases.json
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import bench_answers as B  # noqa: E402

K = 5
EXTRA_FROM_RANKED = 16
HUB_DEGREE = 64
FRACTION = 0.25  # L2, снята
B1_FRACTION = 0.67  # Б1: 201 из 300, решение владельца 2026-10-01
NONINFERIORITY = 0.05  # Б1: «не сильно хуже SEAL», решение владельца 2026-10-01
VALID_AUC = 0.70
CHAIN_SECONDS = 2.0
DRAWS = 10_000
SCREEN = 0.430  # отсев L1, поднят с 0.380 решением владельца 2026-10-01 (мощность)


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def questions_of(bundle: Path) -> dict[str, dict]:
    return {row["qid"]: row for row in load_jsonl(bundle / "questions.jsonl")}


# ------------------------------------------------------------- coverage

def coverage(rows: list[dict], questions: dict[str, dict], base: dict[str, dict] | None) -> dict:
    full = broken = repaired = 0
    for row in rows:
        gold = set(questions[row["qid"]]["gold_chunk_ids"])
        top = set(row["ranked"][:K])
        full += gold <= top
        if base is not None:
            ref = base[row["qid"]]
            ref_top = set(ref["ranked"][:K])
            if gold & ref_top and not gold <= ref_top and gold <= set(ref.get("pool") or []):
                broken += 1
                repaired += gold <= top
    n = max(1, len(rows))
    out = {"n": len(rows), "ctx_all": round(full / n, 4)}
    if base is not None:
        out["broken_chains"] = broken
        out["repaired"] = repaired
    return out


def cmd_coverage(args) -> int:
    questions = questions_of(args.bundle)
    runs = {}
    for spec in args.run:
        name, _, path = spec.partition("=")
        runs[name] = {row["qid"]: row for row in load_jsonl(Path(path))}
    base_name = next(iter(runs))
    base = runs[base_name]
    report = {}
    for name, rows in runs.items():
        report[name] = coverage(list(rows.values()), questions, base)
        extra = ""
        if name != base_name:
            extra = f"  отсев L1 (≥ {SCREEN}) " + ("пройден" if report[name]["ctx_all"] >= SCREEN else "НЕ пройден")
        print(f"{name:32s} ctx_all@{K} {report[name]['ctx_all']:.3f}  "
              f"оборванных {report[name]['broken_chains']}, закрыто {report[name]['repaired']}{extra}")
    if args.out:
        args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# ----------------------------------------------------------- graphchain

def load_mentions(path: Path) -> tuple[dict[str, dict[str, float]], int]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        graph = json.load(handle)
    passages = defaultdict(set)
    for passage_id, entity_id, *_ in graph["mentions"]:
        passages[str(passage_id)].add(str(entity_id))
    df = defaultdict(int)
    for entities in passages.values():
        for entity_id in entities:
            df[entity_id] += 1
    total = len(graph["passages"])
    weights = {
        pid: {e: math.log(total / df[e]) for e in entities if df[e] <= HUB_DEGREE}
        for pid, entities in passages.items()
    }
    return weights, total


def cmd_graphchain(args) -> int:
    weights, _ = load_mentions(args.graph)
    rows = load_jsonl(args.rankings)
    lengths = []
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            ranked = list(row["ranked"])
            candidates = list(dict.fromkeys(list(row.get("pool") or []) + ranked[:EXTRA_FROM_RANKED]))
            chain = [ranked[0]]
            while len(chain) < K:
                last = weights.get(chain[-1], {})
                scored = []
                for index, cid in enumerate(candidates):
                    if cid in chain:
                        continue
                    shared = sum(w for e, w in weights.get(cid, {}).items() if e in last)
                    scored.append((shared, -index, cid))
                if not scored:
                    break
                best = max(scored)
                if best[0] <= 0:
                    break
                chain.append(best[2])
            lengths.append(len(chain))
            out = dict(row)
            out["ranked"] = chain + [cid for cid in ranked if cid not in chain]
            out["chain"] = chain
            handle.write(json.dumps(out, ensure_ascii=False) + "\n")
    print(f"контроль по графу: {len(rows)} вопросов, средняя длина цепочки "
          f"{sum(lengths) / max(1, len(lengths)):.2f}")
    return 0


# -------------------------------------------------------------- gatemix

def scored_answers(path: Path, questions: dict[str, dict], aliases: dict) -> dict[str, tuple[float, float]]:
    out = {}
    for row in load_jsonl(path):
        q = questions[row["qid"]]
        golds = [q["answer"], *q.get("answer_aliases", []), *aliases.get(row["qid"], [])]
        pred = B.final_answer(row["raw"], strict=True)
        out[row["qid"]] = (B.exact_match(pred, golds), B.f1_score(pred, golds))
    return out


def gatemix(bundle: Path, gate_path: Path, answers: Path, aliases_path: Path | None,
            fraction: float, latency: dict | None, gate_summary: dict | None,
            base_name: str = "ours-current", seal_name: str = "ours-current+seal",
            setr_name: str = "ours-current+setr") -> dict:
    """Б1: SEAL на доле ``fraction`` вопросов с наименьшим p_enough, остальным — база.

    Критерий владельца (2026-10-01, до расчёта смеси): не хуже SEAL-на-всех
    с запасом NONINFERIORITY (нижняя граница одностороннего 95%-го бутстрапа
    разности EM), EM выше SetR и медианная задержка поиска ниже SEAL.
    """
    questions = questions_of(bundle)
    aliases = B.load_aliases(aliases_path)
    base = scored_answers(answers / f"answers-{base_name}.jsonl", questions, aliases)
    seal = scored_answers(answers / f"answers-{seal_name}.jsonl", questions, aliases)
    setr = scored_answers(answers / f"answers-{setr_name}.jsonl", questions, aliases)
    gate = {row["qid"]: row for row in load_jsonl(gate_path)}
    ids = sorted(set(base) & set(seal) & set(setr) & set(gate))
    n = len(ids)
    if n < 300:
        raise SystemExit(f"смесь по {n} вопросам из 300 — гейт или ответы неполны")
    m = round(fraction * n)

    def mix(chosen: set[str]) -> tuple[list[float], list[float]]:
        em = [seal[q][0] if q in chosen else base[q][0] for q in ids]
        f1 = [seal[q][1] if q in chosen else base[q][1] for q in ids]
        return em, f1

    order = sorted(ids, key=lambda q: (gate[q]["p_enough"], q))
    chosen = set(order[:m])
    em_mix, f1_mix = mix(chosen)
    em_base = [base[q][0] for q in ids]
    f1_base = [base[q][1] for q in ids]
    em_seal = [seal[q][0] for q in ids]
    em_setr = [setr[q][0] for q in ids]
    holm = B.holm({"em": B.mcnemar_exact(em_base, em_mix)["p"], "f1": B.paired_t(f1_base, f1_mix)["p"]})
    vs_setr = B.mcnemar_exact(em_setr, em_mix)

    rng = random.Random(20261001)
    random_em = sorted(
        sum(seal[q][0] if q in picked else base[q][0] for q in ids) / n
        for picked in (set(rng.sample(ids, m)) for _ in range(DRAWS))
    )
    gain = {q: seal[q][0] - base[q][0] for q in ids}
    by_gain = sorted(ids, key=lambda q: -gain[q])
    curve = []
    for share in [i / 20 for i in range(21)]:
        cut = round(share * n)
        curve.append({"share": share,
                      "gate": round(sum(mix(set(order[:cut]))[0]) / n, 4),
                      "random_expected": round(sum(em_base) / n + share * sum(gain.values()) / n, 4),
                      "oracle": round(sum(mix(set(by_gain[:cut]))[0]) / n, 4)})

    em_value = sum(em_mix) / n
    noninf = _noninferiority(em_mix, em_seal)
    speed = _mix_latency(latency, gate_summary, fraction)
    checks = {
        "noninferior_to_seal": noninf["lower95"] > -NONINFERIORITY,
        "above_setr": em_value > sum(em_setr) / n,
        "faster_than_seal": speed.get("faster_than_seal"),
    }
    accept = all(v is True for v in checks.values())
    return {
        "n": n, "fraction": fraction, "seal_questions": m,
        "em_base": round(sum(em_base) / n, 4), "em_seal_all": round(sum(em_seal) / n, 4),
        "em_setr": round(sum(em_setr) / n, 4),
        "em_mix": round(em_value, 4), "f1_mix": round(sum(f1_mix) / n, 4),
        "noninferiority_vs_seal": noninf, "margin": NONINFERIORITY,
        "mix_vs_setr_mcnemar": vs_setr, "mix_vs_base_holm": holm,
        "random_p95": round(random_em[int(0.95 * DRAWS) - 1], 4),
        "random_mean": round(sum(random_em) / DRAWS, 4),
        "oracle_at_fraction": round(sum(mix(set(by_gain[:m]))[0]) / n, 4),
        "seal_helped_in_chosen": sum(1 for q in chosen if gain[q] > 0),
        "seal_hurt_in_chosen": sum(1 for q in chosen if gain[q] < 0),
        "auc_gate_vs_all_gold": _auc_gold(gate, ids),
        "latency": speed,
        "checks": checks,
        "accept_B1": accept,
        "curve": curve,
    }


def _mix_latency(latency: dict | None, gate_summary: dict | None, fraction: float) -> dict:
    """Медианная задержка поиска смеси по чистому замеру P1: каждый вопрос
    проходит базу и гейт, доля ``fraction`` затем ещё и SEAL целиком
    (консервативно: SEAL не переиспользует поиск базы)."""
    if not latency or "ours-current" not in latency or "ours-current+seal" not in latency:
        return {"faster_than_seal": None, "note": "нет latency-clean.json (P1) — критерий не проверен"}
    base_s = latency["ours-current"]["median_s"]
    seal_s = latency["ours-current+seal"]["median_s"]
    gate_s = ((gate_summary or {}).get("single_median_ms") or 0.0) / 1000
    mix_s = base_s + gate_s + fraction * seal_s
    out = {"base_s": base_s, "seal_s": seal_s, "gate_s": round(gate_s, 3), "mix_s": round(mix_s, 2),
           "mix_over_base": round(mix_s / base_s, 2), "seal_over_base": round(seal_s / base_s, 2),
           "faster_than_seal": mix_s < seal_s}
    if gate_summary is None:
        out["note"] = "нет сводки гейта — время гейта принято за 0"
    return out


def _noninferiority(em_mix: list[float], em_seal: list[float]) -> dict:
    """Разность EM смеси и SEAL-на-всех и нижняя граница одностороннего
    95%-го бутстрапа (10 000 парных выборок вопросов)."""
    diffs = [a - b for a, b in zip(em_mix, em_seal, strict=True)]
    n = len(diffs)
    rng = random.Random(20261001)
    boot = sorted(sum(diffs[rng.randrange(n)] for _ in range(n)) / n for _ in range(DRAWS))
    return {"diff": round(sum(diffs) / n, 4), "lower95": round(boot[int(0.05 * DRAWS)], 4)}


def _auc_gold(gate: dict[str, dict], ids: list[str]) -> float | None:
    """Насколько гейт различает «весь эталон в пятёрке» на самом тесте (объясняющая)."""
    from laya_eval import auc

    labels = [int(bool(gate[q].get("all_gold_in_top5"))) for q in ids]
    scores = [gate[q]["p_enough"] for q in ids]
    value = auc(scores, labels)
    return None if value != value else round(value, 4)


def _read_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def cmd_gatemix(args) -> int:
    report = gatemix(args.bundle, args.gate, args.answers, args.aliases, args.fraction,
                     _read_json(args.latency) if args.latency else None,
                     _read_json(args.gate.with_suffix(".summary.json")))
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, ensure_ascii=False, indent=1))
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


# -------------------------------------------------------------- verdict

def _answers_verdict(report: dict | None, candidate: str) -> dict:
    """Принять: EM или F1 выше базы при p_holm < 0.05 и ни одна метрика значимо не ниже."""
    if report is None:
        return {"status": "нет ответов"}
    vs = report["vs_baseline"][candidate]
    em, f1 = vs["em"], vs["f1"]
    em_up = em["significant"] and em["better"] > em["worse"]
    em_down = em["significant"] and em["better"] < em["worse"]
    f1_up = f1["significant"] and f1["mean_diff"] > 0
    f1_down = f1["significant"] and f1["mean_diff"] < 0
    summary = report["summary"]
    return {
        "em": summary[candidate]["em"], "em_base": summary[report["baseline"]]["em"],
        "f1": summary[candidate]["f1"], "f1_base": summary[report["baseline"]]["f1"],
        "em_better_worse": [em["better"], em["worse"]], "em_p_holm": em["p_holm"],
        "f1_mean_diff": round(f1["mean_diff"], 4), "f1_p_holm": f1["p_holm"],
        "status": "принять" if (em_up or f1_up) and not (em_down or f1_down) else "отвергнуть",
    }


def _chain_seconds(path: Path) -> float | None:
    if not path.exists():
        return None
    ms = sorted(row["chain_ms"] for row in load_jsonl(path))
    return round(ms[len(ms) // 2] / 1000, 3)


def cmd_verdict(args) -> int:
    run, out, ours = args.run, args.run / "laya", args.run / "ours"
    held = _read_json(out / "heldout-ft.json")
    if held is None:
        raise SystemExit(f"нет {out / 'heldout-ft.json'} — обучение не проверено")
    auc_pair, auc_enough = held["pair"]["auc"], held["enough"]["auc"]
    verdict: dict = {"validity": {"auc_pair": auc_pair, "auc_enough": auc_enough,
                                  "base_model": _read_json(out / "heldout-base.json"),
                                  "A_valid": auc_pair >= VALID_AUC, "B1_valid": auc_enough >= VALID_AUC}}

    cov = _read_json(out / "coverage.json")
    a1 = {"ctx_all": cov["chain"]["ctx_all"] if cov else None,
          "repaired": cov["chain"].get("repaired") if cov else None,
          "chain_median_s": _chain_seconds(ours / "rankings-ours-current+chain.jsonl")}
    a1["screen"] = a1["ctx_all"] is not None and a1["ctx_all"] >= SCREEN
    a1["speed"] = a1["chain_median_s"] is not None and a1["chain_median_s"] <= CHAIN_SECONDS
    answers_a1 = _answers_verdict(_read_json(run / "answers-L-A1.json"), "ours-current+chain")
    a1["answers"] = answers_a1
    if not verdict["validity"]["A_valid"]:
        a1["status"] = "замер недействителен (AUC связи < 0.70)"
    elif not a1["screen"]:
        a1["status"] = "отвергнуть на отсеве"
    elif not a1["speed"]:
        a1["status"] = "отвергнуть: цепочка дольше 2 с"
    else:
        a1["status"] = answers_a1["status"]
    verdict["A1"] = a1

    cov2 = _read_json(out / "coverage-setr.json")
    a2 = {"ctx_all_fill": cov2["setr-fill"]["ctx_all"] if cov2 else None,
          "ctx_all_chain": cov2["setr+chain"]["ctx_all"] if cov2 else None,
          "chain_median_s": _chain_seconds(ours / "rankings-ours-current+setr+chain.jsonl"),
          "answers": _answers_verdict(_read_json(run / "answers-L-A2.json"), "ours-current+setr+chain")}
    status = a2["answers"]["status"]
    if not verdict["validity"]["A_valid"]:
        status = "замер недействителен (AUC связи < 0.70)"
    elif status == "отвергнуть":
        status = "не доказано (мощность ≤ 35 вопросов)"
    a2["status"] = status
    verdict["A2"] = a2

    gate_path = out / "gate-ours-current.jsonl"
    if gate_path.exists():
        b1 = gatemix(args.bundle, gate_path, run / "answers", args.aliases, B1_FRACTION,
                     # холодный замер (кэши выключены, как у нового вопроса) главнее тёплого
                     _read_json(run / "latency-cold.json") or _read_json(run / "latency-clean.json"),
                     _read_json(gate_path.with_suffix(".summary.json")))
        (out / "gatemix-B1.json").write_text(json.dumps(b1, ensure_ascii=False, indent=2), encoding="utf-8")
        b1 = {k: v for k, v in b1.items() if k != "curve"}
        b1["status"] = ("замер недействителен (AUC «хватает» < 0.70)" if not verdict["validity"]["B1_valid"]
                        else "принять" if b1["accept_B1"] else "отвергнуть")
    else:
        b1 = {"status": "нет файла гейта"}
    verdict["B1"] = b1
    verdict["latency_clean"] = _read_json(run / "latency-clean.json")
    verdict["testpairs"] = _read_json(out / "testpairs.json")

    (out / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=2), encoding="utf-8")
    v = verdict
    print(f"действительность: AUC связи {auc_pair:.3f}, «хватает» {auc_enough:.3f} (порог {VALID_AUC})")
    print(f"A1: ctx_all {a1['ctx_all']} (отсев {SCREEN}), цепочка {a1['chain_median_s']} с — {a1['status']}")
    if "em" in answers_a1:
        print(f"    EM {answers_a1['em_base']} → {answers_a1['em']} (+{answers_a1['em_better_worse'][0]}/"
              f"−{answers_a1['em_better_worse'][1]}, p_holm {answers_a1['em_p_holm']:.3g}), "
              f"F1 {answers_a1['f1_base']} → {answers_a1['f1']} (p_holm {answers_a1['f1_p_holm']:.3g})")
    print(f"A2: ctx_all {a2['ctx_all_fill']} → {a2['ctx_all_chain']} — {a2['status']}")
    if "em" in a2["answers"]:
        ans = a2["answers"]
        print(f"    EM {ans['em_base']} → {ans['em']} (+{ans['em_better_worse'][0]}/−{ans['em_better_worse'][1]}, "
              f"p_holm {ans['em_p_holm']:.3g}), F1 {ans['f1_base']} → {ans['f1']} (p_holm {ans['f1_p_holm']:.3g})")
    if "em_mix" in b1:
        print(f"Б1: EM смеси {b1['em_mix']} (SEAL {b1['em_seal_all']}, SetR {b1['em_setr']}, база {b1['em_base']}), "
              f"нижняя граница разности с SEAL {b1['noninferiority_vs_seal']['lower95']} (запас −{NONINFERIORITY}), "
              f"задержка {b1['latency'].get('mix_s')} с против SEAL {b1['latency'].get('seal_s')} с — {b1['status']}")
        print(f"    проверки: {b1['checks']}; AUC гейта по флагу эталона {b1['auc_gate_vs_all_gold']}")
    else:
        print(f"Б1: {b1['status']}")
    if v["testpairs"]:
        print(f"объясняющая: AUC связи на парах нашего пула {v['testpairs']['auc']}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    cov = sub.add_parser("coverage")
    cov.add_argument("--bundle", type=Path, required=True)
    cov.add_argument("--run", action="append", required=True, help="имя=rankings.jsonl; первый — база")
    cov.add_argument("--out", type=Path)
    gc = sub.add_parser("graphchain")
    gc.add_argument("--bundle", type=Path, required=True)
    gc.add_argument("--graph", type=Path, required=True)
    gc.add_argument("--rankings", type=Path, required=True)
    gc.add_argument("--out", type=Path, required=True)
    gm = sub.add_parser("gatemix")
    gm.add_argument("--bundle", type=Path, required=True)
    gm.add_argument("--gate", type=Path, required=True)
    gm.add_argument("--answers", type=Path, required=True)
    gm.add_argument("--aliases", type=Path, default=None)
    gm.add_argument("--fraction", type=float, default=B1_FRACTION,
                    help="доля вопросов, уходящих в SEAL (Б1: 0.67; снятая L2: 0.25)")
    gm.add_argument("--latency", type=Path, default=None, help="latency-clean.json (P1/P2)")
    gm.add_argument("--out", type=Path, required=True)
    vd = sub.add_parser("verdict")
    vd.add_argument("--bundle", type=Path, required=True)
    vd.add_argument("--run", type=Path, required=True, help="artifacts/runs/bench/musique-300")
    vd.add_argument("--aliases", type=Path, default=None)
    args = parser.parse_args()
    return {"coverage": cmd_coverage, "graphchain": cmd_graphchain, "gatemix": cmd_gatemix,
            "verdict": cmd_verdict}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
