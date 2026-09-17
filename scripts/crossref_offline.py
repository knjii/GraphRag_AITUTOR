"""R8: перекрёстные ссылки учебника как рёбра графа — офлайн-проверка.

Автор учебника уже расставил связи: «подставим (2.32) в (2.47)», «см.
определение 3.8», «раздел 8.2». Это рёбра, за которые не надо платить
извлечением сущностей и которые не страдают от главной болезни нашего
графа — связи вырождаются внутри одного фрагмента (R2).

Проверяется **без карты и без сети**: разобранные фрагменты уже лежат
в `artifacts/parsed`, ссылки достаются правилами.

    python scripts/crossref_offline.py --json artifacts/metrics/crossref.json

## Критерий отказа (записан до первого прогона, 2026-09-18)

Мера — пары эталонных фрагментов связывающих вопросов, которые
**нынешний граф не соединяет** (второй фрагмент не попадает в top-30
обхода от первого). R8 стоит проверять на сервере, только если
ссылочные рёбра соединяют **≥ 10 %** таких пар при **≤ 2 шагах**.
Меньше — направление закрывается: значит, автор связывает не то, что
спрашивают.

Отдельно печатается доля ссылок, которые не удалось разрешить. Она
ограничивает выводы сверху: неразрешённая ссылка — не отсутствие связи,
а слепота правила.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from collections import deque
from collections.abc import Iterable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Номер формулы MinerU отдаёт разметкой \tag{2.32} внутри выражения —
# однозначный якорь: именно здесь формула определена, а не упомянута.
TAG = re.compile(r"\\tag\{\(?(\d+\.\d+[a-z]?)\)?\}")
# Ссылка на формулу в прозе: «подставив (2.32) в (2.47)».
EQ_REF = re.compile(r"\((\d+\.\d+[a-z]?)\)")

# Падежи перечислены явно: «[Пп]ример\w*» поймало бы «примерно 2.5 раза».
STATEMENT = (r"(?:[Оо]пределени(?:е|я|и|ю|ем)|[Тт]еорем(?:а|ы|е|у|ой)|"
             r"[Лл]емм(?:а|ы|е|у|ой)|[Сс]ледстви(?:е|я|и|ю|ем)|"
             r"[Пп]ример(?:а|е|у|ы|ов|ах)?|[Зз]амечани(?:е|я|и|ю|ем))")
# «Определение 2.10 (векторное подпространство).» — имя в скобках сразу
# после номера встречается только в месте определения, не в ссылке.
STATEMENT_ANCHOR = re.compile(STATEMENT + r"\s+(\d+\.\d+)\s*\(")
STATEMENT_REF = re.compile(STATEMENT + r"\s+(\d+\.\d+)")

SECTION_REF = re.compile(r"[Рр]азд(?:ел[аеу]?|\.)\s*(\d+(?:\.\d+)+)")
FIGURE_REF = re.compile(r"[Рр]ис(?:унок|унке|унка|\.)\s*(\d+\.\d+)")
# Подпись рисунка: номер, точка, прописная буква — в отличие от ссылки
# «на рис. 1.1 показано».
FIGURE_CAPTION = re.compile(r"[Рр]ис(?:унок|\.)\s*(\d+\.\d+)\.\s+[А-ЯЁ]")
# Номер раздела живёт в заголовках фрагмента: «1.1. ПОИСК ИНТУИТИВНО…».
SECTION_HEAD = re.compile(r"^\s*(\d+(?:\.\d+)+)\.")


def load_chunks(parsed_dir: Path, doc_id: str | None) -> list[dict[str, Any]]:
    pattern = f"{doc_id}_chunks.json" if doc_id else "*_chunks.json"
    chunks: list[dict[str, Any]] = []
    for path in sorted(parsed_dir.glob(pattern)):
        chunks.extend(json.loads(path.read_text(encoding="utf-8")))
    return chunks


def anchors_of(chunks: list[dict[str, Any]]) -> dict[str, dict[str, set[str]]]:
    """Где определён объект: вид → номер → фрагменты."""
    found: dict[str, dict[str, set[str]]] = {
        "equation": {}, "statement": {}, "section": {}, "figure": {},
    }
    sections: dict[str, dict[str, Any]] = {}
    for chunk in chunks:
        text = chunk["text"]
        for match in TAG.finditer(text):
            found["equation"].setdefault(match.group(1), set()).add(chunk["id"])
        for match in STATEMENT_ANCHOR.finditer(text):
            found["statement"].setdefault(match.group(1), set()).add(chunk["id"])
        # Раздел — это десятки фрагментов. Ссылка ведёт к его началу,
        # иначе одно упоминание «раздел 8.2» породило бы куст рёбер,
        # то есть ровно тот хаб, от которого мы лечили граф.
        for header in chunk.get("headers") or ():
            match = SECTION_HEAD.match(header)
            if not match:
                continue
            first = sections.get(match.group(1))
            if first is None or chunk["ordinal"] < first["ordinal"]:
                sections[match.group(1)] = chunk
        if chunk.get("has_figure"):
            for match in FIGURE_CAPTION.finditer(text):
                found["figure"].setdefault(match.group(1), set()).add(chunk["id"])
    for number, chunk in sections.items():
        found["section"][number] = {chunk["id"]}
    return found


def references_of(chunk: dict[str, Any]) -> list[tuple[str, str]]:
    """Ссылки фрагмента: (вид, номер). Разметка тегов вырезана."""
    text = TAG.sub(" ", chunk["text"])
    refs = [("equation", m.group(1)) for m in EQ_REF.finditer(text)]
    refs += [("statement", m.group(1)) for m in STATEMENT_REF.finditer(text)]
    refs += [("section", m.group(1)) for m in SECTION_REF.finditer(text)]
    refs += [("figure", m.group(1)) for m in FIGURE_REF.finditer(text)]
    return refs


def build(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    anchors = anchors_of(chunks)
    edges: dict[str, set[str]] = collections.defaultdict(set)
    seen = collections.Counter()
    resolved = collections.Counter()
    self_only = collections.Counter()
    for chunk in chunks:
        for kind, number in references_of(chunk):
            seen[kind] += 1
            targets = anchors.get(kind, {}).get(number, set())
            if not targets:
                continue
            outside = targets - {chunk["id"]}
            resolved[kind] += 1
            if not outside:
                # Ссылка на формулу в том же фрагменте: связь есть,
                # но ребра она не даёт — считаем отдельно, чтобы не
                # выдавать её за связность (ровно ошибка R2).
                self_only[kind] += 1
                continue
            for target in outside:
                edges[chunk["id"]].add(target)
                edges[target].add(chunk["id"])
    return {"anchors": anchors, "edges": edges, "seen": seen,
            "resolved": resolved, "self_only": self_only}


def distance(edges: dict[str, set[str]], source: str, target: str, limit: int) -> int | None:
    """Число шагов от фрагмента до фрагмента по ссылочным рёбрам."""
    if source == target:
        return 0
    seen = {source}
    queue = deque([(source, 0)])
    while queue:
        node, depth = queue.popleft()
        if depth >= limit:
            continue
        for neighbour in edges.get(node, ()):  # noqa: SIM118
            if neighbour == target:
                return depth + 1
            if neighbour not in seen:
                seen.add(neighbour)
                queue.append((neighbour, depth + 1))
    return None


def gold_pairs(goldset_path: Path, known: set[str]) -> list[tuple[str, str, str]]:
    data = json.loads(goldset_path.read_text(encoding="utf-8"))
    questions = data["questions"] if isinstance(data, dict) else data
    pairs = []
    for question in questions:
        ids = list(dict.fromkeys(question.get("gold_chunk_ids") or []))
        if len(ids) != 2 or not set(ids) <= known:
            continue
        pairs.append((question["id"], ids[0], ids[1]))
    return pairs


def degree_summary(edges: dict[str, set[str]], chunks: int) -> dict[str, Any]:
    degrees = sorted((len(v) for v in edges.values()), reverse=True)
    return {
        "фрагментов со ссылочным ребром": len(degrees),
        "доля фрагментов": round(len(degrees) / max(1, chunks), 3),
        "рёбер": sum(degrees) // 2,
        "медианная степень": degrees[len(degrees) // 2] if degrees else 0,
        "максимальная степень": degrees[0] if degrees else 0,
    }


def graph_reach(settings_model: str, effort: str | None, doc: str,
                pairs: list[tuple[str, str, str]], cutoff: int) -> dict[str, int]:
    """Достаёт ли нынешний граф второй фрагмент пары из первого.

    Граф восстанавливается из кэша извлечения теми же правилами, что
    и запись в Neo4j (`rag_textbook.evaluation.graph_offline`). Копия
    базы делается через backup: источник не трогаем.
    """
    import os
    import sqlite3
    import tempfile
    from contextlib import closing

    os.environ.setdefault("RAG_ENV_FILE", "tests-no-such-env-file")
    from rag_textbook.config import Settings
    from rag_textbook.evaluation.graph_offline import rank_from_passage, reconstruct

    settings = Settings()
    with tempfile.TemporaryDirectory(prefix="crossref-") as directory:
        source = settings.paths.cache_dir / "extraction.sqlite3"
        with (
            closing(sqlite3.connect(source.resolve().as_uri() + "?mode=ro", uri=True)) as src,
            closing(sqlite3.connect(str(Path(directory) / source.name))) as dst,
        ):
            src.backup(dst)
        settings.paths.cache_dir = Path(directory)
        graph = reconstruct(settings, model=settings_model, reasoning_effort=effort,
                            doc_id=doc,
                            max_entity_degree=settings.graph.max_entity_degree)
    reach: dict[str, int] = {}
    orders: dict[str, list[str]] = {}
    for _, left, right in pairs:
        for anchor, target in ((left, right), (right, left)):
            if anchor not in orders:
                orders[anchor] = rank_from_passage(
                    graph, anchor, hop_decay=settings.graph.hop_decay,
                    use_idf=settings.graph.passage_idf_enabled)
            order = orders[anchor]
            place = order.index(target) + 1 if target in order else None
            key = f"{anchor}->{target}"
            reach[key] = place if place is not None and place <= cutoff else 0
    return reach


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed", type=Path,
                        default=Path("C:/python/rag_textbook/artifacts/parsed"))
    parser.add_argument("--goldset", type=Path,
                        default=Path("C:/python/rag_textbook/evaluation/goldsets/goldset.json"))
    parser.add_argument("--doc", default="0690bb81b7e3c831", help="документ (по умолчанию MML)")
    parser.add_argument("--hops", type=int, default=2)
    parser.add_argument("--compare-graph", action="store_true",
                        help="сравнить со связностью нынешнего графа (кэш извлечения)")
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--effort", default=None)
    parser.add_argument("--cutoff", type=int, default=30,
                        help="место, дальше которого фрагмент считается ненайденным")
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(list(argv) if argv is not None else None)

    chunks = load_chunks(args.parsed, args.doc)
    if not chunks:
        print(f"нет разобранных фрагментов в {args.parsed}", file=sys.stderr)
        return 1
    graph = build(chunks)
    edges = graph["edges"]

    report: dict[str, Any] = {"фрагментов": len(chunks)}
    print(f"фрагментов {len(chunks)}")
    print("\nссылки")
    for kind in ("equation", "statement", "section", "figure"):
        seen = graph["seen"][kind]
        if not seen:
            continue
        done, inside = graph["resolved"][kind], graph["self_only"][kind]
        print(f"  {kind:10s} найдено {seen:5d}  разрешено {done:5d} ({done / seen:.2f})"
              f"  из них внутри фрагмента {inside}")
        report[f"ссылки.{kind}"] = {"найдено": seen, "разрешено": done, "внутри": inside}

    summary = degree_summary(edges, len(chunks))
    print("\nграф ссылок")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    report["граф"] = summary

    pairs = gold_pairs(args.goldset, {c["id"] for c in chunks})
    reached = collections.Counter()
    for _, left, right in pairs:
        steps = distance(edges, left, right, args.hops)
        reached[steps if steps is not None else "нет"] += 1
    print(f"\nпары эталонных фрагментов: {len(pairs)}")
    for key in sorted(reached, key=lambda k: (k == "нет", k)):
        share = reached[key] / max(1, len(pairs))
        label = "не соединены" if key == "нет" else f"соединены за {key}"
        print(f"  {label}: {reached[key]} ({share:.3f})")
    report["пары"] = {"всего": len(pairs),
                      "соединено": sum(v for k, v in reached.items() if k != "нет")}

    if args.compare_graph:
        reach = graph_reach(args.model, args.effort, args.doc, pairs, args.cutoff)
        missed = both = 0
        for _, left, right in pairs:
            # Пара считается пропущенной графом, если он не достаёт второй
            # фрагмент ни в одну сторону: одностороннего доступа хватает.
            graph_ok = any(reach.get(f"{a}->{b}") for a, b in ((left, right), (right, left)))
            if graph_ok:
                continue
            missed += 1
            if distance(edges, left, right, args.hops) is not None:
                both += 1
        share = both / missed if missed else 0.0
        print(f"\nнынешний граф не соединяет: {missed} пар из {len(pairs)}")
        print(f"  из них ссылочные рёбра соединяют: {both} ({share:.3f})")
        print(f"  критерий, записанный до прогона: >= 0.10 — "
              f"{'выполнен' if share >= 0.10 else 'НЕ выполнен, R8 закрывается'}")
        report["сравнение"] = {"пропущено графом": missed, "спасено ссылками": both,
                               "доля": round(share, 3), "критерий": 0.10}

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\nзаписано: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
