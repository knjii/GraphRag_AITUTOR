"""Структурный граф учебника (К1) в файл варианта графа.

    # граф A — только структура:
    python scripts/graph_structural.py --parsed artifacts/parsed --out artifacts/graphs/structural.json.gz

    # граф C — объединение с выгрузкой модельного графа:
    python scripts/graph_structural.py --parsed artifacts/parsed --out artifacts/graphs/union.json.gz \
        --merge-with artifacts/graphs/current.json.gz

Модель, карта и сеть не нужны. Отчёт о покрытии по книгам печатается
и при ``--report`` сохраняется: доля неразрешённых ссылок ограничивает
выводы о К1 сверху.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.graph.structural import (  # noqa: E402
    build_structural,
    merge_graphs,
    missing_passages,
)
from rag_textbook.stores.graph_file import GraphFile, file_hash  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--parsed", type=Path, required=True, help="каталог *_chunks.json")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--merge-with", type=Path, help="модельный граф для варианта C")
    parser.add_argument("--report", type=Path, help="куда сохранить отчёт о покрытии")
    parser.add_argument("--notation", action="store_true", help="добавить обозначения по правилам (К3)")
    parser.add_argument("--notation-scope", choices=("section", "book"), default="section")
    args = parser.parse_args(argv)

    chunks = [
        item
        for path in sorted(args.parsed.glob("*_chunks.json"))
        for item in json.loads(path.read_text(encoding="utf-8"))
    ]
    if not chunks:
        print(f"В {args.parsed} нет фрагментов", file=sys.stderr)
        return 1
    graph, report = build_structural(
        chunks,
        variant=f"structural+notation-{args.notation_scope}" if args.notation else "structural",
        notation=args.notation,
        notation_scope=args.notation_scope,
    )
    if args.merge_with is not None:
        model = GraphFile.load(args.merge_with)
        # Граф другого корпуса дал бы объединение, которое мерит не то.
        foreign = missing_passages([model, graph])
        if foreign:
            print(
                f"Графы построены на разных корпусах: {len(foreign)} фрагментов известны не обоим",
                file=sys.stderr,
            )
            return 1
        graph = merge_graphs(model, graph, variant="union")
        report["merged_with"] = {"file": str(args.merge_with), "sha256": file_hash(args.merge_with)}
        report["total"] = graph.summary()

    path = graph.save(args.out)
    report["file"] = str(path)
    report["sha256"] = file_hash(path)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
