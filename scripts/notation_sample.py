"""Ручная проверка К3 без модели: sheet создаёт лист, score оценивает разметку."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag_textbook.graph.extractor import NOTATION_LABEL  # noqa: E402
from rag_textbook.graph.notation import find_notations  # noqa: E402
from rag_textbook.stores.graph_file import GraphFile  # noqa: E402

MARKERS = re.compile(r"\b(?:где\b|обознач\w*|через\b|назовём\b|пусть\b|denote\w*)", re.I)


def normalize_symbol(symbol: str) -> str:
    """Сохраняем различия LaTeX: нормализуем только пробелы и внешние доллары."""
    return re.sub(r"\s+", "", symbol).strip("$")


def candidates(parsed: Path) -> list[dict[str, Any]]:
    result = []
    seen: set[str] = set()
    for path in sorted(parsed.glob("*_chunks.json")):
        for chunk in json.loads(path.read_text(encoding="utf-8-sig")):
            if chunk["id"] in seen:
                raise ValueError(f"Повторный id фрагмента: {chunk['id']}")
            seen.add(chunk["id"])
            if "$" in chunk["text"] and MARKERS.search(chunk["text"]):
                result.append(chunk)
    return result


def stratified_sample(
    chunks: list[dict[str, Any]], sample: int = 50, seed: int = 0,
) -> list[dict[str, Any]]:
    if sample < 1:
        raise ValueError("Размер выборки должен быть положительным")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for chunk in chunks:
        groups[chunk["doc_id"]].append(chunk)
    if not groups:
        return []
    rng = random.Random(seed)
    books = sorted(groups)
    rng.shuffle(books)
    for book in books:
        rng.shuffle(groups[book])
    cap = math.ceil(sample / len(books)) + 1
    selected = []
    # Круги дают каждой книге место, а предел защищает от доминирования большой книги.
    for _ in range(cap):
        for book in books:
            if groups[book]:
                selected.append(groups[book].pop())
                if len(selected) == sample:
                    return selected
    return selected


def graph_notations(graph: GraphFile, chunk_id: str) -> list[str]:
    links: dict[str, list[str]] = defaultdict(list)
    for source, target, _, label, _ in graph.relations:
        if label == NOTATION_LABEL:
            links[source].append(target)
    result: set[str] = set()
    for entity_id, (_, role) in graph.mentions.get(chunk_id, {}).items():
        if role not in ("defines", "", "mentions"):
            continue
        entity = graph.entities[entity_id]
        if entity.get("kind") != "notation" and entity_id not in links:
            continue
        canonical = entity.get("canonical", "")
        if ":=" in canonical:
            result.add(canonical)
        else:
            symbol = entity.get("name") or canonical
            for target in links.get(entity_id, []):
                meaning = graph.entities[target]
                result.add(f"{symbol} := {meaning.get('canonical') or meaning['name']}")
            if not links.get(entity_id):
                result.add(f"{symbol} := (смысл отсутствует в графе)")
    return sorted(result)


def make_sheet(parsed: Path, sample: int, seed: int, graph: GraphFile | None) -> dict[str, Any]:
    pool = candidates(parsed)
    rows = []
    for chunk in stratified_sample(pool, sample, seed):
        full_text = chunk["text"]
        rows.append({
            "chunk_id": chunk["id"], "doc_id": chunk["doc_id"],
            "doc_name": chunk.get("doc_name", ""), "pages": chunk.get("pages", []),
            "text": full_text[:1500], "full_text": full_text,
            "rules": [f"{n.symbol} := {n.meaning}" for n in find_notations(full_text)],
            "graph": graph_notations(graph, chunk["id"]) if graph is not None else None,
            "human": None,
        })
    return {"format": "notation-sample/1", "seed": seed, "requested": sample,
            "candidate_count": len(pool), "candidate_books": len({c['doc_id'] for c in pool}),
            "graph_available": graph is not None, "items": rows}


def markdown_sheet(key: dict[str, Any]) -> str:
    lines = ["# Проверка обозначений К3", "",
             f"Кандидатов: {key['candidate_count']}; выбрано: {len(key['items'])}.", "",
             'Заполните human в JSON списком строк «символ := смысл». '
             'null — не проверено, [] — обозначений нет. '
             'Проверяйте полный фрагмент full_text: ниже превью до 1500 знаков.', ""]
    for row in key["items"]:
        lines.extend([f"## {row['chunk_id']} — {row['doc_name']}", "",
                      f"Страницы: {row['pages']}", "", row["text"], "",
                      "Правила: " + json.dumps(row["rules"], ensure_ascii=False),
                      "", "Граф v4: " + (json.dumps(row["graph"], ensure_ascii=False)
                                          if row["graph"] is not None else "не предоставлен"),
                      "", "Человек (символ := смысл): ____________________", ""])
    return "\n".join(lines)


def symbols(answer: Any) -> set[str]:
    if not isinstance(answer, list):
        raise ValueError("Ответ должен быть списком строк «символ := смысл»; null не размечен")
    result = set()
    for item in answer:
        if not isinstance(item, str) or ":=" not in item:
            raise ValueError("Ожидается строка «символ := смысл»")
        symbol, meaning = item.split(":=", 1)
        normalized = normalize_symbol(symbol)
        if not normalized or not meaning.strip():
            raise ValueError("Символ и смысл не должны быть пустыми")
        result.add(normalized)
    return result


def threshold_decision(interval: list[float] | None) -> str:
    if interval is None or interval[0] <= 0.8 <= interval[1]:
        return "не различимо"
    return "правила достаточны" if interval[0] > 0.8 else "нужен запасной путь v4"


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


def score(key: dict[str, Any], bootstrap: int = 2000, seed: int = 0) -> dict[str, Any]:
    if bootstrap < 1:
        raise ValueError("Число повторов бутстрэпа должно быть положительным")
    rows = key["items"]
    if not rows:
        raise ValueError("Выборка пуста")
    gold = [symbols(row["human"]) for row in rows]
    result: dict[str, Any] = {"fragments": len(rows), "confidence": 0.95,
                              "bootstrap": bootstrap, "seed": seed}
    for method in ("rules", "graph"):
        if method == "graph" and not key["graph_available"]:
            result[method] = None
            continue
        predicted = [symbols(row[method]) for row in rows]
        counts = [(len(p & g), len(p), len(g)) for p, g in zip(predicted, gold, strict=True)]
        totals = [sum(c[i] for c in counts) for i in range(3)]
        draws: dict[str, list[float]] = {"precision": [], "recall": []}
        rng = random.Random(seed)
        for _ in range(bootstrap):
            sampled = rng.choices(counts, k=len(counts))
            summed = [sum(c[i] for c in sampled) for i in range(3)]
            for name, denominator in (("precision", 1), ("recall", 2)):
                if summed[denominator]:
                    draws[name].append(summed[0] / summed[denominator])
        metrics: dict[str, Any] = {"tp": totals[0], "predicted": totals[1], "human": totals[2]}
        for name, denominator in (("precision", 1), ("recall", 2)):
            values = draws[name]
            metrics[name] = totals[0] / totals[denominator] if totals[denominator] else None
            metrics[name + "_ci"] = (
                [percentile(values, 0.025), percentile(values, 0.975)] if values else None
            )
            metrics[name + "_valid_bootstrap"] = len(values)
        result[method] = metrics
    result["decision"] = threshold_decision(result["rules"]["recall_ci"])
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    sheet = modes.add_parser("sheet", help="Создать Markdown и JSON для разметки")
    sheet.add_argument("--parsed", type=Path, required=True)
    sheet.add_argument("--sample", type=int, default=50)
    sheet.add_argument("--seed", type=int, default=0)
    sheet.add_argument("--graph", type=Path)
    sheet.add_argument("--output", type=Path, required=True, help="Префикс файлов .md и .json")
    scoring = modes.add_parser("score", help="Оценить полностью заполненный JSON")
    scoring.add_argument("--key", type=Path, required=True)
    scoring.add_argument("--bootstrap", type=int, default=2000)
    scoring.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.mode == "sheet":
            if not args.parsed.is_dir():
                raise ValueError("--parsed должен указывать на существующий каталог")
            graph = GraphFile.load(args.graph) if args.graph else None
            key = make_sheet(args.parsed, args.sample, args.seed, graph)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.with_suffix(".json").write_text(
                json.dumps(key, ensure_ascii=False, indent=2), encoding="utf-8")
            args.output.with_suffix(".md").write_text(markdown_sheet(key), encoding="utf-8")
            print(f"Кандидатов: {key['candidate_count']}; выбрано: {len(key['items'])}")
        else:
            key = json.loads(args.key.read_text(encoding="utf-8-sig"))
            print(json.dumps(score(key, args.bootstrap, args.seed), ensure_ascii=False, indent=2))
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
