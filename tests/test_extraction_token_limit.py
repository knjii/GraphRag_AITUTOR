"""Предел на ответ извлечения связей.

Все 28 отказов извлечения оказались одним и тем же: `invalid_json` после трёх
попыток, причём каждый ответ **начинался корректным JSON** и обрывался.
Валидное начало при невалидном целом — подпись обрезки по лимиту токенов,
а не капризов модели.

Прежний предел был зашит числом 768 и совпадал с размером «красивого» JSON
на 12 сущностей и 12 связей: модель печатала с отступами и переносами и
упиралась в лимит ровно на последних связях. Повторы тут бессильны —
обрежется и повтор, что и наблюдалось: три попытки, три обрыва.
"""

from __future__ import annotations

from rag_textbook.config import GraphSettings
from rag_textbook.graph.extractor import EntityExtractor
from rag_textbook.models import Chunk


class _RecordingLLM:
    """Запоминает, с каким пределом её позвали."""

    def __init__(self) -> None:
        self.settings = None
        self.max_tokens: int | None = None

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.max_tokens = kwargs.get("max_tokens")
        return '{"entities": [{"name": "матрица"}], "relations": []}'


def _chunk() -> Chunk:
    return Chunk(
        id="doc:00001",
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=1,
        text="Определитель матрицы равен произведению собственных значений.",
    )


def test_limit_is_taken_from_settings():
    llm = _RecordingLLM()
    extractor = EntityExtractor(
        GraphSettings(extraction_cache_enabled=False, extraction_max_tokens=4096),
        llm=llm,
    )

    extractor.extract(_chunk(), "модель-для-теста")

    assert llm.max_tokens == 4096, "предел перестал зависеть от настройки"


def test_default_limit_fits_the_worst_case():
    """Предел по умолчанию обязан вмещать самый громоздкий допустимый ответ.

    Худший случай — предельное число сущностей и связей, напечатанное
    с отступами. Если предел его не вмещает, отказы вернутся, а повторы
    их не спасут.
    """
    import json

    settings = GraphSettings()
    entities = [{"name": f"довольно длинное имя сущности {i}"} for i in range(settings.max_entities_per_chunk)]
    relations = [
        {
            "source": f"довольно длинное имя сущности {i}",
            "target": f"довольно длинное имя сущности {i + 1}",
            "relation": "используется_в",
        }
        for i in range(settings.max_relations_per_chunk)
    ]
    pretty = json.dumps({"entities": entities, "relations": relations}, ensure_ascii=False, indent=2)
    # Грубая оценка: для русского текста около 2.5 символа на токен.
    estimated = len(pretty) / 2.5

    assert settings.extraction_max_tokens > estimated * 1.2, (
        f"предел {settings.extraction_max_tokens} не вмещает худший случай "
        f"(~{estimated:.0f} токенов) с запасом"
    )
