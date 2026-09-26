"""Проба извлечения на малой выборке до полного прогона графа.

    python scripts/extraction_smoke.py --chunks artifacts/parsed/0690bb81b7e3c831_chunks.json \
        --sample 40 --json artifacts/runs/day2/smoke-v4.json

Полное извлечение v4 по русскому блоку — около трёх часов карты. Если
модель не заполняет роли, отвечает пустотой или ломает JSON, это видно
на сорока фрагментах за две минуты. Кэш не читается и не пишется: проба
не должна ни подсунуть старый результат, ни оставить свой.

Критерии записаны до запуска (docs/SERVER-DAY-2.md, шаг G0):
- откатов к правилам после повторов не больше 5 %;
- роль заполнена у ≥ 95 % сущностей (иначе К2 мерить нечем);
- фрагментов хотя бы с одним «defines» от 10 % до 90 %: ноль значит,
  что модель роль не различает, почти все — что она ставит её всем.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.clients.llm import build_llm_client  # noqa: E402
from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.graph.extractor import EntityExtractor  # noqa: E402
from rag_textbook.models import Chunk  # noqa: E402

MAX_FALLBACK = 0.05
MIN_ROLE_FILLED = 0.95
DEFINES_RANGE = (0.10, 0.90)


def summarize(results: list[tuple[Chunk, object]]) -> dict:
    statuses = Counter(str(result.status) for _, result in results)
    entities = [entity for _, result in results for entity in result.entities]
    roles = Counter(str(getattr(entity, "role", "") or "") for entity in entities)
    kinds = Counter(str(getattr(entity, "kind", "concept")) for entity in entities)
    with_defines = sum(
        1 for _, result in results if any(getattr(e, "role", "") == "defines" for e in result.entities)
    )
    total = max(1, len(results))
    filled = sum(count for role, count in roles.items() if role)
    fallback = statuses.get("rule_fallback", 0) / total
    role_share = filled / max(1, len(entities))
    defines_share = with_defines / total
    problems = []
    if fallback > MAX_FALLBACK:
        problems.append(f"откатов {fallback:.0%} > {MAX_FALLBACK:.0%}")
    if role_share < MIN_ROLE_FILLED:
        problems.append(f"роль заполнена у {role_share:.0%} сущностей < {MIN_ROLE_FILLED:.0%}")
    if not DEFINES_RANGE[0] <= defines_share <= DEFINES_RANGE[1]:
        problems.append(f"фрагментов с defines {defines_share:.0%} вне {DEFINES_RANGE}")
    return {
        "фрагментов": len(results),
        "статусы": dict(statuses),
        "сущностей на фрагмент": round(len(entities) / total, 2),
        "связей на фрагмент": round(sum(len(r.relations) for _, r in results) / total, 2),
        "роли": dict(roles),
        "типы узлов": dict(kinds),
        "доля откатов": round(fallback, 3),
        "доля с ролью": round(role_share, 3),
        "доля фрагментов с defines": round(defines_share, 3),
        "проблемы": problems,
        "годен": not problems,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--sample", type=int, default=40)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--json", type=Path)
    parser.add_argument("--examples", type=int, default=3, help="сколько ответов напечатать целиком")
    args = parser.parse_args(argv)

    payload = json.loads(args.chunks.read_text(encoding="utf-8"))
    items = payload["chunks"] if isinstance(payload, dict) else payload
    chunks = [Chunk.model_validate(item) for item in items]
    sample = random.Random(args.seed).sample(chunks, min(args.sample, len(chunks)))

    settings = Settings()
    if settings.graph.extraction_prompt_version != "v4":
        print(f"GRAPH_EXTRACTION_PROMPT_VERSION={settings.graph.extraction_prompt_version}, нужна v4")
        return 1
    llm = build_llm_client(settings.llm)
    extractor = EntityExtractor(settings.graph, llm=llm, cache=None)
    try:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            results = list(zip(sample, pool.map(extractor.extract, sample), strict=True))
    finally:
        close = getattr(llm, "close", None)
        if close is not None:
            close()

    summary = summarize(results)
    # Прочитать глазами до выводов: число «годен» не отличит осмысленные
    # роли от случайных.
    for chunk, result in results[: args.examples]:
        print(f"\n--- {chunk.id}: {chunk.text[:300]!r}")
        for entity in result.entities:
            print(f"    {getattr(entity, 'role', '') or '—':8} {getattr(entity, 'kind', ''):9} {entity.name}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if summary["годен"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
