"""Выгрузка графа из Neo4j в файл варианта графа.

    python scripts/graph_export.py --out artifacts/graphs/current.json.gz --variant current

Запускается на сервере, где поднят Neo4j. Файл потом едет на ноутбук
и читается графовым каналом при ``GRAPH_BACKEND=memory``. Пароль Neo4j
берётся из окружения, как у остального конвейера, и в файл не пишется.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.config import Settings  # noqa: E402
from rag_textbook.stores.graph_file import file_hash  # noqa: E402
from rag_textbook.stores.graph_store import GraphStore  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--variant", default="current")
    args = parser.parse_args(argv)

    settings = Settings()
    store = GraphStore(settings.graph)
    try:
        graph = store.export_graph(args.variant)
        live = store.stats()
    finally:
        store.close()

    problems = graph.validate()
    if problems:
        print(f"Выгрузка испорчена: {problems[:5]}", file=sys.stderr)
        return 1
    summary = graph.summary()
    # Сверка с подсчётом самой базы: выгрузка, потерявшая часть узлов,
    # дала бы стенд, который мерит другой граф.
    mismatches = [
        f"{name}: база {live[name]}, файл {summary[key]}"
        for name, key in (("passages", "passages"), ("entities", "entities"), ("mentions", "mentions"))
        if live.get(name) != summary[key]
    ]
    if mismatches:
        print("Выгрузка не совпала с базой: " + "; ".join(mismatches), file=sys.stderr)
        return 1

    path = graph.save(args.out)
    print(json.dumps({**summary, "file": str(path), "sha256": file_hash(path)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
