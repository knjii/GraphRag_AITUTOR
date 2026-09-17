"""Скачивание комплектов книг по манифесту с контрольными суммами.

По умолчанию только показывает план. Скачивает с ``--download``; книги
со статусом ``check`` пропускаются, пока не указаны явно через ``--only``.

    python scripts/fetch_library.py                       # план
    python scripts/fetch_library.py --download            # книги со статусом ready
    python scripts/fetch_library.py --download --only ru-la-gelfand

Суммы пишутся в ``<target>/SHA256SUMS``: замер привязан к конкретному
файлу, а черновики на авторских страницах обновляются без смены адреса
(у Murphy черновик 2025-04-18 уже не совпадает с изданием 2022).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "evaluation" / "library" / "manifest.json"


def plan(books: list[dict], only: set[str]) -> list[dict]:
    selected = []
    for book in books:
        if only and book["id"] not in only:
            continue
        if not only and book["status"] != "ready":
            continue
        if book["url"].startswith("https://github.com/") and "/tree/" in book["url"]:
            # Каталог, а не файл: скачивать нечего, нужен ручной выбор.
            continue
        selected.append(book)
    return selected


def target_path(base: Path, book: dict) -> Path:
    return base / book["lang"] / f"{book['id']}.pdf"


def download(url: str, path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    partial = path.with_suffix(".part")
    with httpx.stream("GET", url, follow_redirects=True, timeout=120) as response:
        response.raise_for_status()
        with partial.open("wb") as handle:
            for block in response.iter_bytes():
                digest.update(block)
                handle.write(block)
    head = partial.read_bytes()[:5]
    if head != b"%PDF-":
        partial.unlink()
        raise RuntimeError(f"{url}: не PDF (начало файла {head!r})")
    partial.replace(path)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--only", nargs="*", default=[])
    parser.add_argument("--target", type=Path)
    args = parser.parse_args()

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    base = args.target or (ROOT / manifest["target_dir"])
    selected = plan(manifest["books"], set(args.only))
    skipped = [b["id"] for b in manifest["books"] if b not in selected]
    for book in selected:
        print(f"{book['id']:28s} → {target_path(base, book)}\n    {book['url']}")
    if skipped:
        print(f"пропущены (status=check или каталог): {', '.join(skipped)}")
    if not args.download:
        print("\nтолько план; для скачивания добавьте --download")
        return 0

    sums = base / "SHA256SUMS"
    lines = sums.read_text(encoding="utf-8").splitlines() if sums.exists() else []
    failed = 0
    for book in selected:
        path = target_path(base, book)
        try:
            checksum = download(book["url"], path)
        except Exception as error:  # noqa: BLE001 — один сбой не должен останавливать остальные
            print(f"ОШИБКА {book['id']}: {error}", file=sys.stderr)
            failed += 1
            continue
        relative = path.relative_to(base).as_posix()
        lines = [line for line in lines if not line.endswith(f"  {relative}")]
        lines.append(f"{checksum}  {relative}")
        print(f"{book['id']}: {path.stat().st_size / 2**20:.1f} МБ, sha256 {checksum[:12]}")
    sums.parent.mkdir(parents=True, exist_ok=True)
    sums.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
