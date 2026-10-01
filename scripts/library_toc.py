"""Компактная навигация по headers локальной нарезки книг."""

import argparse
import json
import re
from pathlib import Path


CHUNKS_DIR = Path(__file__).resolve().parents[1] / "tasks" / "035" / "chunks"


def table_of_contents(chunks: list[dict], limit: int = 149) -> list[str]:
    # headers перечисляют встреченные заголовки, а не путь в дереве разделов.
    headers = [h for chunk in chunks for h in chunk.get("headers", [])]
    sections = any(re.match(r"^§\s*\d+", h) for h in headers)
    numbered = any(re.match(r"^\d+\.\d+\.\s", h) for h in headers)
    rows: list[tuple[str, int, int]] = []
    current = "Без заголовка"
    for chunk in chunks:
        number = int(chunk["id"].rsplit(":", 1)[1])
        for header in chunk.get("headers", []):
            if sections:
                keep = bool(re.match(r"^§\s*\d+", header))
            elif numbered:
                keep = bool(re.match(r"^\d+\.\d+\.\s", header))
            else:
                keep = True
            if keep or current == "Без заголовка":
                current = header
        # На границе один фрагмент может содержать конец предыдущего раздела.
        if rows and rows[-1][0].casefold() == current.casefold():
            title, start, _ = rows[-1]
            rows[-1] = (title, start, number)
        else:
            if rows:
                title, start, _ = rows[-1]
                rows[-1] = (title, start, number)
            rows.append((current, number, number))
    # Для книг без распознанной иерархии сохраняем все диапазоны навигации.
    size = max(1, (len(rows) + limit - 1) // limit)
    lines = []
    for offset in range(0, len(rows), size):
        group = rows[offset : offset + size]
        title = group[0][0]
        if len(group) > 1:
            title += " … " + group[-1][0]
        lines.append(f"{group[0][1]:05d}–{group[-1][2]:05d}  {title}")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--doc", required=True, help="Идентификатор книги")
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-f]{16}", args.doc):
        parser.error("Нужен идентификатор из 16 шестнадцатеричных символов")
    path = CHUNKS_DIR / f"{args.doc}_chunks.json"
    if not path.is_file():
        parser.error(f"Нет нарезки: {path}")
    chunks = json.loads(path.read_text(encoding="utf-8"))
    print(f"{args.doc}: {len(chunks)} фрагментов; вложенные заголовки свёрнуты")
    print("\n".join(table_of_contents(chunks)))


if __name__ == "__main__":
    main()
