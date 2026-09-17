"""Сравнение проб GRPO: какую связку брать в спринт 3 и нужна ли вторая карта.

Правило выбора записано здесь **до** прогона — чтобы после замера не было
соблазна подогнать критерий под понравившееся число. Читает `probe.json`
проб (`scripts/train_grpo.py --probe`) и печатает разбор шага, вердикт
и цену обучения.

    python scripts/probe_compare.py runs/probe-4b-*/probe.json

Пробы бывают двух видов:

* **полная** — `max_completion_length` рабочий (768);
* **короткая** — тот же прогон с коротким ответом (96).

Разность их времён шага даёт нижнюю оценку доли генерации: за что именно
мы платим время карты. Это число решает, имеет ли смысл вторая карта
(движок генерации отдельным процессом) — её выгода ограничена сверху
долей генерации.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

# Правило выбора (записано 2026-09-18, до аренды).
MIN_SPEEDUP = 0.15  # 20 шагов — слишком короткий прогон, чтобы верить меньшей разнице
MIN_GENERATIONS = 2  # связка, не выдержавшая двух генераций, выбывает
HEADROOM_GIB = 2.0  # запас памяти, ниже которого связка считается впритык
SECOND_CARD_SHARE = 0.35  # ниже этой доли генерации вторая карта не окупается


def load(paths: list[Path]) -> list[dict[str, Any]]:
    probes = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data["path"] = str(path)
        data.setdefault("stack", "?")
        probes.append(data)
    return probes


def split_by_stack(probes: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Для каждой связки: самая длинная проба и самая короткая."""
    stacks: dict[str, dict[str, Any]] = {}
    for probe in probes:
        key = f"{probe['stack']}:g{probe.get('num_generations', '?')}"
        entry = stacks.setdefault(key, {"stack": probe["stack"], "probes": []})
        entry["probes"].append(probe)
    for entry in stacks.values():
        by_len = sorted(entry["probes"], key=lambda p: p.get("max_completion_length", 0))
        entry["full"] = by_len[-1]
        entry["short"] = None
        # Короткой считается только проба, у которой ответ действительно
        # короче и всё остальное совпадает. Иначе разность времён — шум
        # между двумя одинаковыми прогонами, выданный за долю генерации.
        for probe in by_len[:-1]:
            if comparable(probe, entry["full"]):
                entry["short"] = probe
                break
    return stacks


def comparable(short: dict[str, Any], full: dict[str, Any]) -> bool:
    """Отличаться пробы должны ровно длиной ответа."""
    if short.get("max_completion_length", 0) >= full.get("max_completion_length", 0):
        return False
    return all(
        short.get(key) == full.get(key)
        for key in ("stack", "num_generations", "grad_accum", "max_seq_length", "steps")
    )


def generation_share(entry: dict[str, Any]) -> float | None:
    """Нижняя оценка доли генерации в шаге.

    Короткая проба всё ещё генерирует (96 токенов), поэтому разность
    занижает истинную долю — оценка нижняя, и так её и надо читать.
    """
    short, full = entry.get("short"), entry["full"]
    if not short:
        return None
    t_full, t_short = full["seconds_per_step"], short["seconds_per_step"]
    if t_full <= 0:
        return None
    return max(0.0, (t_full - t_short) / t_full)


def verdict(stacks: dict[str, dict[str, Any]]) -> list[str]:
    lines: list[str] = []
    usable = {k: v for k, v in stacks.items()
              if v["full"].get("num_generations", 0) >= MIN_GENERATIONS}
    if not usable:
        return [f"ни одна связка не прошла пробу при {MIN_GENERATIONS} генерациях — "
                "брать карту больше или сокращать окно"]
    # 1. Больше генераций — важнее скорости: при двух ответах в группе
    #    преимущество GRPO считается по паре, и сигнал заметно беднее.
    best_gens = max(v["full"].get("num_generations", 0) for v in usable.values())
    finalists = {k: v for k, v in usable.items()
                 if v["full"].get("num_generations", 0) == best_gens}
    if len(finalists) < len(usable):
        lines.append(f"отсеяны связки, не вытянувшие {best_gens} генераций: "
                     + ", ".join(sorted(set(usable) - set(finalists))))
    # 2. Скорость шага, если разница больше шума короткого прогона.
    ranked = sorted(finalists.items(), key=lambda kv: kv[1]["full"]["seconds_per_step"])
    (top_key, top), rest = ranked[0], ranked[1:]
    if not rest:
        lines.append(f"выбор: {top_key} — сравнивать не с чем")
        return lines
    second_key, second = rest[0]
    gain = 1 - top["full"]["seconds_per_step"] / second["full"]["seconds_per_step"]
    if gain >= MIN_SPEEDUP:
        lines.append(f"выбор: {top_key} — шаг короче на {gain:.0%} "
                     f"({top['full']['seconds_per_step']} с против "
                     f"{second['full']['seconds_per_step']} с), это больше порога {MIN_SPEEDUP:.0%}")
        return lines
    # 3. Разница в пределах шума — решает запас памяти: он покупает
    #    генерации и длину контекста в следующих спринтах.
    lines.append(f"разница по времени {gain:.0%} — меньше порога {MIN_SPEEDUP:.0%}, "
                 "решает запас памяти")
    by_free = sorted(finalists.items(), key=lambda kv: -free_gib(kv[1]["full"]))
    winner_key, winner = by_free[0]
    lines.append(f"выбор: {winner_key} — свободно {free_gib(winner['full']):.1f} ГиБ "
                 f"против {free_gib(by_free[1][1]['full']):.1f} ГиБ")
    return lines


def free_gib(probe: dict[str, Any]) -> float:
    total = probe.get("device_memory_gib")
    used = probe.get("reserved_memory_gib") or probe.get("peak_memory_gib")
    if total is None or used is None:
        return 0.0
    return total - used


def second_card(stacks: dict[str, dict[str, Any]]) -> list[str]:
    """Стоит ли добирать вторую карту под движок генерации."""
    shares = {k: generation_share(v) for k, v in stacks.items()}
    known = {k: s for k, s in shares.items() if s is not None}
    if not known:
        return ["доля генерации не измерена: нет короткой пробы "
                "(прогнать ту же связку с --max-completion-length 96)"]
    lines = ["доля генерации в шаге (нижняя оценка): "
             + ", ".join(f"{k} {s:.0%}" for k, s in sorted(known.items()))]
    best = max(known.values())
    tight = [k for k, v in stacks.items()
             if free_gib(v["full"]) < HEADROOM_GIB or v["full"].get("num_generations", 0) < 4]
    if best >= SECOND_CARD_SHARE:
        lines.append(f"вторая карта окупается: генерация занимает до {best:.0%} шага, "
                     "движок выносится отдельным процессом "
                     "(`trl vllm-serve` + --vllm-mode server)")
    else:
        lines.append(f"вторая карта под движок не окупается: генерация {best:.0%} шага, "
                     "выигрыш ограничен этой долей, а цена карты удваивается")
    if tight:
        lines.append("но память впритык у: " + ", ".join(sorted(set(tight)))
                     + " — вторая карта (или одна на 48 ГБ) нужна не ради скорости, "
                       "а чтобы поднять число генераций и длину контекста")
    return lines


def price(stacks: dict[str, dict[str, Any]], steps: int, rate: float) -> list[str]:
    lines = []
    for key, entry in sorted(stacks.items()):
        hours = entry["full"]["seconds_per_step"] * steps / 3600
        lines.append(f"{key}: {steps} шагов ~ {hours:.1f} ч ~ {hours * rate:.0f} руб "
                     f"при {rate:.0f} руб/ч")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("probes", nargs="+", type=Path)
    parser.add_argument("--steps", type=int, default=1000, help="шагов в полном прогоне спринта 3")
    parser.add_argument("--rate", type=float, default=36.0, help="рублей за час карты")
    args = parser.parse_args()

    stacks = split_by_stack(load(args.probes))
    print("проба               шаг, с   пик, ГиБ  свободно  ген.  ответ")
    for key, entry in sorted(stacks.items()):
        for probe in sorted(entry["probes"], key=lambda p: -p.get("max_completion_length", 0)):
            print(f"{key:<18} {probe['seconds_per_step']:>8} "
                  f"{probe.get('peak_memory_gib', '—'):>10} "
                  f"{free_gib(probe):>9.1f} {probe.get('num_generations', '—'):>5} "
                  f"{probe.get('max_completion_length', '—'):>6}")
    print()
    for line in verdict(stacks):
        print("  " + line)
    print()
    for line in second_card(stacks):
        print("  " + line)
    print()
    for line in price(stacks, args.steps, args.rate):
        print("  " + line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
