"""Парсер PDF поверх MinerU.

Три отличия от прежней реализации, каждое снимает конкретную проблему.

1. **Вызов через CLI, а не через внутренние модули MinerU.** Прежний код импортировал
   ``mineru.backend.pipeline.pipeline_analyze`` и работал только с backend'ом ``pipeline``.
   CLI — стабильный контракт, одинаковый для ``pipeline`` / ``hybrid`` / ``vlm``,
   поэтому backend становится параметром конфигурации.
2. **Изоляция процесса получается бесплатно.** Отдельный процесс завершается и
   освобождает CUDA-контекст сам. Это убирает самодельные heartbeat-вотчдоги,
   принудительные ``kill`` и безусловный сон в барьере выгрузки моделей.
3. **Кэш разбора по хешу файла.** Повторный запуск не парсит заново то,
   что уже разобрано, — при стоимости парсинга в часы это принципиально.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from rag_textbook.config import ParsingSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Block, content_hash
from rag_textbook.parsing.normalize import normalize_mineru_blocks

logger = get_logger("parsing.pdf")


class PdfParseError(RuntimeError):
    """Разбор документа не удался."""


def file_fingerprint(path: Path) -> str:
    """Отпечаток файла для кэша.

    Считаем по размеру и первым/последним мегабайтам: полный sha256
    двадцатимегабайтного PDF на каждом запуске — лишние секунды без пользы,
    а коллизия на нашем масштабе практически исключена.
    """

    stat = path.stat()
    with path.open("rb") as handle:
        head = handle.read(1024 * 1024)
        if stat.st_size > 2 * 1024 * 1024:
            handle.seek(-1024 * 1024, os.SEEK_END)
            tail = handle.read(1024 * 1024)
        else:
            tail = b""
    return content_hash(
        path.name,
        str(stat.st_size),
        head.hex()[:256],
        tail.hex()[:256],
    )


class MineruPdfParser:
    """Обёртка над CLI MinerU."""

    def __init__(self, settings: ParsingSettings, parsed_dir: Path) -> None:
        self.settings = settings
        self.parsed_dir = Path(parsed_dir)
        self.parsed_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------- служебное

    def _doc_cache_dir(self, pdf_path: Path) -> Path:
        name = f"{pdf_path.stem}__{file_fingerprint(pdf_path)[:16]}"
        # Разбор части документа кэшируется отдельно от полного: иначе пробный
        # прогон по первым страницам выдавал бы себя за разбор всей книги.
        if self.settings.page_start > 0 or self.settings.page_end >= 0:
            end = self.settings.page_end if self.settings.page_end >= 0 else "end"
            name = f"{name}__p{self.settings.page_start}-{end}"
        return self.parsed_dir / name

    def _blocks_file(self, cache_dir: Path) -> Path:
        return cache_dir / "blocks.json"

    def _build_command(self, pdf_path: Path, output_dir: Path) -> list[str]:
        # `python -m mineru.cli.client` надёжнее голого `mineru`: не зависит от того,
        # попал ли каталог скриптов окружения в PATH арендованной машины.
        command = [
            sys.executable,
            "-m",
            "mineru.cli.client",
            "-p",
            str(pdf_path),
            "-o",
            str(output_dir),
            "-b",
            self.settings.backend,
            "-l",
            self.settings.lang,
        ]
        if self.settings.backend == "pipeline":
            command += ["-m", self.settings.method]
        if self.settings.page_start > 0:
            command += ["-s", str(self.settings.page_start)]
        if self.settings.page_end >= 0:
            command += ["-e", str(self.settings.page_end)]
        command += ["-f", "true" if self.settings.formula_enable else "false"]
        command += ["-t", "true" if self.settings.table_enable else "false"]
        return command

    def _environment(self) -> dict[str, str]:
        env = dict(os.environ)
        env["MINERU_MODEL_SOURCE"] = self.settings.model_source
        if self.settings.tools_config_json:
            config_path = Path(self.settings.tools_config_json)
            if not config_path.is_absolute():
                config_path = Path.home() / config_path
            env["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)
        return env

    @staticmethod
    def _find_content_list(output_dir: Path, stem: str) -> Path | None:
        """Ищет `*_content_list.json`.

        Точный путь зависит от backend'а (`auto` / `ocr` / `txt` / `vlm`),
        поэтому не угадываем структуру каталогов, а ищем по шаблону.
        """
        candidates = sorted(output_dir.rglob(f"{stem}*content_list.json"))
        if not candidates:
            candidates = sorted(output_dir.rglob("*content_list.json"))
        return candidates[0] if candidates else None

    # ------------------------------------------------------------------- разбор

    def parse(self, pdf_path: Path, *, force: bool = False) -> list[Block]:
        pdf_path = Path(pdf_path)
        if not pdf_path.is_file():
            raise PdfParseError(f"Файл не найден: {pdf_path}")

        cache_dir = self._doc_cache_dir(pdf_path)
        blocks_file = self._blocks_file(cache_dir)

        if blocks_file.is_file() and not force:
            logger.info("Разбор взят из кэша: %s", pdf_path.name)
            payload = json.loads(blocks_file.read_text(encoding="utf-8"))
            return [Block.model_validate(item) for item in payload]

        cache_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = cache_dir / "mineru"
        if raw_dir.exists():
            shutil.rmtree(raw_dir, ignore_errors=True)
        raw_dir.mkdir(parents=True, exist_ok=True)

        command = self._build_command(pdf_path, raw_dir)
        logger.info(
            "Запускаю MinerU: %s (backend=%s, lang=%s)",
            pdf_path.name,
            self.settings.backend,
            self.settings.lang,
        )
        started = time.monotonic()
        try:
            completed = subprocess.run(  # noqa: S603 - команда собирается нами, не из ввода
                command,
                env=self._environment(),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=(self.settings.stall_timeout_seconds or None),
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise PdfParseError(
                f"MinerU не уложился в {self.settings.stall_timeout_seconds} с на {pdf_path.name}"
            ) from exc

        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            tail = (completed.stderr or completed.stdout or "")[-2000:]
            raise PdfParseError(
                f"MinerU завершился с кодом {completed.returncode} на {pdf_path.name}.\n{tail}"
            )

        content_list_path = self._find_content_list(raw_dir, pdf_path.stem)
        if content_list_path is None:
            raise PdfParseError(
                f"MinerU отработал, но не нашёлся *_content_list.json для {pdf_path.name}"
            )

        raw_blocks = json.loads(content_list_path.read_text(encoding="utf-8"))
        blocks = normalize_mineru_blocks(raw_blocks)

        # Каталог с картинками нужен на стадии обогащения: запоминаем его рядом с блоками.
        images_dir = content_list_path.parent / "images"
        meta = {
            "source": str(pdf_path),
            "backend": self.settings.backend,
            "elapsed_seconds": round(elapsed, 2),
            "blocks": len(blocks),
            "images_dir": str(images_dir) if images_dir.is_dir() else "",
            "content_list": str(content_list_path),
        }
        (cache_dir / "meta.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        blocks_file.write_text(
            json.dumps([block.model_dump() for block in blocks], ensure_ascii=False),
            encoding="utf-8",
        )

        logger.info(
            "Разобрано %s: блоков=%s, время=%.1f с (%.2f блок/с)",
            pdf_path.name,
            len(blocks),
            elapsed,
            len(blocks) / elapsed if elapsed > 0 else 0.0,
        )
        return blocks

    def images_dir_for(self, pdf_path: Path) -> Path | None:
        meta_path = self._doc_cache_dir(Path(pdf_path)) / "meta.json"
        if not meta_path.is_file():
            return None
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        images_dir = str(meta.get("images_dir") or "")
        return Path(images_dir) if images_dir else None
