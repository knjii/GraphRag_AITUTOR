"""Скачивает публичный набор MultiHop-RAG.

    python scripts/fetch_multihop_rag.py [--target evaluation/public/multihop-rag]

Набор раздаётся авторами на Hugging Face двумя файлами: корпус (6.8 МБ)
и вопросы (5.2 МБ). В git проекта он не кладётся — это чужие данные весом
в десятки мегабайт, и версионировать их вместе с кодом незачем.

Отдельно про доверие к загрузке: файлы читаются как есть и разбираются
нашим переходником, который считает долю сопоставленных свидетельств.
Если файл окажется другой версии или повреждённым, это будет видно
по покрытию, а не по молчаливому падению метрик.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

BASE = "https://huggingface.co/datasets/yixuantt/MultiHopRAG/resolve/main"
FILES = ("corpus.json", "MultiHopRAG.json")


def fetch(name: str, target: Path) -> Path:
    destination = target / name
    if destination.exists():
        print(f"уже есть: {destination}")
        return destination
    url = f"{BASE}/{name}"
    print(f"скачиваю {url}")
    target.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response:  # noqa: S310
        destination.write_bytes(response.read())
    print(f"сохранено: {destination} ({destination.stat().st_size / 1e6:.1f} МБ)")
    return destination


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", type=Path, default=Path("evaluation/public/multihop-rag"))
    args = parser.parse_args()

    for name in FILES:
        path = fetch(name, args.target)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(f"[ошибка] {path} не разбирается как JSON: {error}")
            return 1
        print(f"  записей: {len(payload)}")

    print(
        "\nДальше — на сервере, обязательно в отдельную коллекцию:\n"
        "  QDRANT_COLLECTION=multihop_rag NEO4J_DATABASE=multihop \\\n"
        "    uv run rag-textbook eval public \\\n"
        "      --corpus evaluation/public/multihop-rag/corpus.json \\\n"
        "      --questions evaluation/public/multihop-rag/MultiHopRAG.json"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
