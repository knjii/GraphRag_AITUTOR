"""Почему извлечение связей падает всегда на одних и тех же фрагментах.

Повторы (`GRAPH_EXTRACTION_RETRIES=2`) вернули 8 фрагментов из 45, оставшиеся
37 падают детерминированно. Повтор помогает от сетевых сбоев и случайных срывов
генерации; то, что остаётся, — свойство самого входа, и разбирается оно
по кэшу, без сервера и без GPU.

Скрипт сопоставляет записи кэша с фрагментами по ключу и сравнивает удавшиеся
фрагменты с неудавшимися: длина, доля формул, таблицы, заголовки. Вывод —
не список подозрений, а числа, по которым видно, есть ли системная причина
вообще.

    uv run python scripts/diagnose_extraction_failures.py
"""

from __future__ import annotations

import json
import re
import sqlite3
import statistics
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any

from rag_textbook.config import get_settings
from rag_textbook.models import Chunk, content_hash

# Статусы, означающие, что содержательного ответа не получено. Совпадают
# с теми, при которых извлечение уходит в правиловый откат.
FAILED_STATUSES = {"error", "invalid_json", "invalid_structure", "empty_response"}


def load_chunks(path: Path) -> list[Chunk]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return [Chunk.model_validate(item) for item in raw]


def load_cache(path: Path) -> dict[str, dict[str, Any]]:
    db = sqlite3.connect(path)
    try:
        rows = db.execute(
            "SELECT key, value FROM cache_entries WHERE namespace = 'extraction'"
        ).fetchall()
    finally:
        db.close()
    entries: dict[str, dict[str, Any]] = {}
    for key, value in rows:
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries[key] = parsed
    return entries


def formula_share(text: str) -> float:
    """Доля символов внутри формульной разметки."""
    if not text:
        return 0.0
    inside = sum(len(match) for match in re.findall(r"\$\$.*?\$\$|\$[^$]+\$", text, re.S))
    return inside / len(text)


def cache_key(
    chunk: Chunk, model: str, variant: str, prompt: str, max_ent: str, max_rel: str
) -> str:
    """Повторяет EntityExtractor._cache_key, но с параметрами извне.

    Собственный экземпляр извлекателя тут не годится: он взял бы настройки
    из локального окружения, а кэш собран на сервере с другими.
    """
    return content_hash(
        chunk.text_hash or content_hash(chunk.text),
        model,
        variant,
        prompt,
        max_ent,
        max_rel,
    )


def detect_cache_params(
    chunks: list[Chunk], cache: dict[str, dict[str, Any]], probe: int = 60
) -> tuple[str, str, str, str, str] | None:
    """Подбирает сочетание параметров, которым собран кэш."""
    models = [
        "Qwen/Qwen3.5-4B",
        "qwen3.5:4b",
        "Qwen/Qwen3.5-4B-Instruct",
        "qwen3.5:4b-instruct",
    ]
    variants = ["none", "", "low", "medium", "high"]
    prompts = ["v3", "v2", "v1"]
    limits = ["12", "10", "8", "16"]

    sample = chunks[:probe]
    best: tuple[int, tuple[str, str, str, str, str]] | None = None
    for model, variant, prompt, max_ent, max_rel in product(
        models, variants, prompts, limits, limits
    ):
        hits = sum(
            1
            for chunk in sample
            if cache_key(chunk, model, variant, prompt, max_ent, max_rel) in cache
        )
        if hits and (best is None or hits > best[0]):
            best = (hits, (model, variant, prompt, max_ent, max_rel))
    return best[1] if best else None


def describe(name: str, chunks: list[Chunk]) -> dict[str, Any]:
    if not chunks:
        return {"группа": name, "фрагментов": 0}
    lengths = [len(item.text) for item in chunks]
    shares = [formula_share(item.text) for item in chunks]
    return {
        "группа": name,
        "фрагментов": len(chunks),
        "длина медиана": int(statistics.median(lengths)),
        "длина максимум": max(lengths),
        "с формулами": sum(1 for item in chunks if item.has_formula),
        "доля формул медиана": round(statistics.median(shares), 3),
        "с таблицами": sum(1 for item in chunks if item.has_table),
        "с рисунками": sum(1 for item in chunks if item.has_figure),
    }


def main() -> int:
    settings = get_settings()
    chunks_path = next(Path("artifacts/parsed").glob("*_chunks.json"), None)
    if chunks_path is None:
        print("не найден файл с фрагментами в artifacts/parsed")
        return 1

    chunks = load_chunks(chunks_path)
    cache = load_cache(Path("artifacts/cache/extraction.sqlite3"))
    print(f"фрагментов: {len(chunks)}, записей в кэше: {len(cache)}")

    # Ключ кэша складывается из модели, глубины размышления, версии промпта
    # и пределов на сущности и связи. Локально этих настроек нет: кэш собран
    # на сервере. Поэтому не угадываем, а подбираем — сочетание, которое
    # совпало с кэшем, и есть то, которым кэш собран. Подбор сам себя проверяет:
    # неверное сочетание не даёт ни одного совпадения.
    params = detect_cache_params(chunks, cache)
    if params is None:
        print("ни одно сочетание параметров не совпало с кэшем")
        return 1
    model, variant, prompt, max_ent, max_rel = params
    print(
        f"параметры кэша: модель={model}, размышление={variant!r}, "
        f"промпт={prompt}, сущностей={max_ent}, связей={max_rel}"
    )

    matched: dict[str, tuple[Chunk, dict[str, Any]]] = {}
    for chunk in chunks:
        key = cache_key(chunk, model, variant, prompt, max_ent, max_rel)
        entry = cache.get(key)
        if entry is not None:
            matched[chunk.id] = (chunk, entry)

    print(f"сопоставлено фрагментов с кэшем: {len(matched)} из {len(chunks)}")
    if not matched:
        print("ни один ключ не совпал")
        return 1

    statuses = Counter(entry.get("status", "?") for _, entry in matched.values())
    print("\n=== статусы ===")
    for status, count in statuses.most_common():
        print(f"  {status:<20} {count}")

    # Откат к правилам НЕ кэшируется намеренно: иначе разовый сбой модели стал бы
    # постоянным. Побочное следствие — отказавшие фрагменты в кэше отсутствуют
    # вовсе, и опознать их можно только по отсутствию. Заодно это объясняет,
    # почему причина отказа до сих пор не найдена: она нигде не записана.
    missing = [chunk for chunk in chunks if chunk.id not in matched]
    failed = [chunk for chunk, entry in matched.values() if entry.get("status") in FAILED_STATUSES]
    failed = failed + missing
    # Пустой результат при благополучном статусе — тоже потеря: связей нет.
    empty_ok = [
        chunk
        for chunk, entry in matched.values()
        if entry.get("status") not in FAILED_STATUSES and not entry.get("relations")
    ]
    good = [
        chunk
        for chunk, entry in matched.values()
        if entry.get("status") not in FAILED_STATUSES and entry.get("relations")
    ]

    print("\n=== сравнение групп ===")
    rows = [
        describe("удачные", good),
        describe("без связей", empty_ok),
        describe("отказ", failed),
    ]
    columns = [key for key in rows[0] if key != "группа"]
    print(f"{'группа':<14}" + "".join(f"{name:>22}" for name in columns))
    for row in rows:
        print(f"{row['группа']:<14}" + "".join(f"{row.get(name, '—')!s:>22}" for name in columns))

    if failed:
        print("\n=== отказавшие фрагменты ===")
        for chunk in sorted(failed, key=lambda item: len(item.text), reverse=True)[:20]:
            head = (chunk.headers or ["—"])[0][:34]
            print(
                f"  длина {len(chunk.text):>5}  формул {formula_share(chunk.text):.2f}  "
                f"табл {int(chunk.has_table)}  рис {int(chunk.has_figure)}  "
                f"стр {str(chunk.pages)[:12]:<12} {head}"
            )

    # Обрезка входа — первый подозреваемый: если отказы начинаются ровно там,
    # где текст упирается в предел, значит модели достаётся оборванная формула.
    limit = settings.graph.extraction_max_chars
    near_limit_failed = sum(1 for item in failed if len(item.text) >= limit * 0.95)
    near_limit_good = sum(1 for item in good if len(item.text) >= limit * 0.95)
    print(f"\n=== предел обрезки {limit} символов ===")
    print(f"  отказов у предела: {near_limit_failed} из {len(failed)}")
    print(f"  удачных у предела: {near_limit_good} из {len(good)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
