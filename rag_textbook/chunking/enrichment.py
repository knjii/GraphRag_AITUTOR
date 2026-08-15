"""Обогащение блоков описанием от модели зрения.

Три изменения относительно прежней реализации, каждое имеет цену в часах индексации
или в качестве поиска.

1. **Только ``image`` и ``chart``.** Раньше модель зрения вызывалась ещё на таблицах
   и формулах, хотя MinerU уже отдаёт для них HTML и LaTeX. Это была двойная работа,
   а её результат затирал исходное представление.
2. **Кэш по содержимому.** Ключ включает путь к картинке, подпись и версию промпта,
   поэтому повторный прогон и переиндексация не платят за описания второй раз.
3. **Параллельные вызовы.** Прежний цикл был строго последовательным; на учебнике
   с сотнями иллюстраций это часы разницы.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.config import ChunkingSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Block, normalize_text
from rag_textbook.utils.cache import ArtifactCache

logger = get_logger("chunking.enrichment")

PROMPTS: dict[str, str] = {
    "image": (
        "Опиши изображение из учебника кратко и по фактам (до 60 слов).\n"
        "Если на изображении есть обозначения, оси, подписи — перечисли их.\n"
        "Не начинай со слов «на изображении».\n"
        "{context}"
    ),
    "chart": (
        "Опиши график или диаграмму из учебника (до 60 слов).\n"
        "Обязательно укажи: что отложено по осям, характер зависимости, ключевые точки.\n"
        "Не интерпретируй сверх изображённого.\n"
        "{context}"
    ),
}


class BlockEnricher:
    """Добавляет ``enrichment`` блокам-иллюстрациям."""

    def __init__(
        self,
        settings: ChunkingSettings,
        llm: LLMClient,
        cache: ArtifactCache | None = None,
    ) -> None:
        self.settings = settings
        self.llm = llm
        self.cache = cache

    # ------------------------------------------------------------------ помощь

    def _context_for(self, blocks: Sequence[Block], index: int) -> str:
        """Соседний текст помогает модели понять, о чём иллюстрация."""
        window = self.settings.context_window
        if window <= 0:
            return ""

        left: list[str] = []
        position = index - 1
        while position >= 0 and sum(len(part) for part in left) < window:
            if blocks[position].type in {"text", "title"} and blocks[position].text:
                left.append(blocks[position].text)
            position -= 1

        right: list[str] = []
        position = index + 1
        while position < len(blocks) and sum(len(part) for part in right) < window:
            if blocks[position].type in {"text", "title"} and blocks[position].text:
                right.append(blocks[position].text)
            position += 1

        parts: list[str] = []
        if left:
            parts.append("Текст слева: " + " ".join(reversed(left))[-window:])
        if right:
            parts.append("Текст справа: " + " ".join(right)[:window])
        block = blocks[index]
        if block.caption:
            parts.append(f"Подпись: {block.caption}")
        return "\n".join(parts)

    def _resolve_image(self, block: Block, images_dir: Path | None) -> str | None:
        if not block.img_path or images_dir is None:
            return None
        candidate = images_dir / Path(block.img_path).name
        return str(candidate) if candidate.is_file() else None

    def _describe(self, block: Block, context: str, image_path: str | None) -> str:
        template = PROMPTS.get(block.type, PROMPTS["image"])
        prompt = template.format(context=context).strip()
        message = ChatMessage(
            role="user",
            content=prompt,
            images=[image_path] if image_path else [],
        )
        try:
            answer = self.llm.chat(
                [message],
                purpose="vision",
                max_tokens=160,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            # Описание — дополнение. Его отсутствие не должно валить индексацию:
            # исходная подпись и текст блока остаются в индексе.
            logger.warning("Не удалось описать блок %s: %s", block.index, exc)
            return ""
        return normalize_text(answer)

    # ---------------------------------------------------------------- публично

    def enrich(self, blocks: list[Block], images_dir: Path | None = None) -> dict[str, int]:
        stats = {"candidates": 0, "from_cache": 0, "generated": 0, "failed": 0, "skipped": 0}
        if not self.settings.enrich_enabled:
            stats["skipped"] = len(blocks)
            return stats

        allowed = set(self.settings.enrich_types)
        targets = [
            index
            for index, block in enumerate(blocks)
            if block.type in allowed and not block.enrichment
        ]
        stats["candidates"] = len(targets)
        if not targets:
            return stats

        version = self.settings.enrich_prompt_version
        keys = {index: blocks[index].enrichment_key(version) for index in targets}

        cached: dict[str, object] = {}
        if self.cache is not None and self.settings.enrich_cache_enabled:
            cached = self.cache.get_many(keys.values())

        pending: list[int] = []
        for index in targets:
            hit = cached.get(keys[index])
            if isinstance(hit, str):
                blocks[index].enrichment = hit
                stats["from_cache"] += 1
            else:
                pending.append(index)

        if not pending:
            logger.info("Обогащение: все %s объектов взяты из кэша", stats["from_cache"])
            return stats

        logger.info(
            "Обогащение: %s объектов (из кэша %s, к генерации %s, параллелизм %s)",
            stats["candidates"],
            stats["from_cache"],
            len(pending),
            self.settings.enrich_max_concurrency,
        )

        def work(index: int) -> tuple[int, str]:
            block = blocks[index]
            context = self._context_for(blocks, index)
            image_path = self._resolve_image(block, images_dir)
            return index, self._describe(block, context, image_path)

        produced: dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=self.settings.enrich_max_concurrency) as pool:
            for index, description in pool.map(work, pending):
                if description:
                    blocks[index].enrichment = description
                    produced[keys[index]] = description
                    stats["generated"] += 1
                else:
                    stats["failed"] += 1

        if produced and self.cache is not None and self.settings.enrich_cache_enabled:
            self.cache.set_many(produced)

        logger.info(
            "Обогащение завершено: сгенерировано %s, из кэша %s, ошибок %s",
            stats["generated"],
            stats["from_cache"],
            stats["failed"],
        )
        return stats
