"""Разнообразие выдачи.

Основание для этой стадии: все 34 промаха на многошаговых вопросах устроены
одинаково — найден один эталонный фрагмент из двух. Отбор по одной лишь
релевантности не отличает «ещё один фрагмент про то же» от «фрагмента про
вторую половину вопроса», и второй нужный фрагмент уходит под отсечку.

Проверяется главное: режим ``off`` обязан ничего не менять (иначе включение
флага по умолчанию сдвинуло бы все прежние замеры), а режим ``reserve``
обязан не трогать голову выдачи — в этом весь смысл его осторожности.
"""

from __future__ import annotations

from rag_textbook.config import Settings
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.retrieval.diversity import apply_diversity


def _settings(**retrieval) -> Settings:
    settings = Settings(_env_file=None)
    if retrieval:
        settings.retrieval = settings.retrieval.model_copy(update=retrieval)
    return settings


def _item(identifier: str, text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(
            id=identifier,
            doc_id="doc",
            doc_name="Учебник",
            source_path="учебник.pdf",
            ordinal=0,
            text=text,
        ),
        score=1.0,
    )


def _items() -> list[ScoredChunk]:
    """Четыре фрагмента про разложение и один — про совсем другое.

    Так выглядит промах на связывающем вопросе: голова выдачи занята
    однотипными соседями, а нужный второй фрагмент — последний.
    """
    return [
        _item("a1", "Сингулярное разложение раскладывает матрицу на три множителя."),
        _item("a2", "Сингулярное разложение матрицы применяют для сжатия данных."),
        _item("a3", "Сингулярное разложение матрицы связано с собственными числами."),
        _item("a4", "Сингулярное разложение матрицы вычисляют численными методами."),
        _item("b1", "Гауссово распределение задаётся средним и ковариацией."),
    ]


def test_off_mode_changes_nothing():
    items = _items()

    result = apply_diversity(items, _settings(diversity_mode="off"), top_k=3)

    assert [item.chunk.id for item in result] == [item.chunk.id for item in items]


def test_reserve_promotes_dissimilar_chunk_into_output():
    items = _items()

    result = apply_diversity(
        items,
        _settings(diversity_mode="reserve", diversity_reserve_slots=1),
        top_k=3,
    )

    assert "b1" in [item.chunk.id for item in result[:3]], (
        "непохожий фрагмент не попал в выдачу — режим не сработал"
    )


def test_reserve_keeps_the_head_untouched():
    """Осторожность режима в том, что первые места он не трогает.

    Иначе он ухудшил бы одношаговые вопросы, где однотипные фрагменты
    как раз и нужны.
    """
    items = _items()

    result = apply_diversity(
        items,
        _settings(diversity_mode="reserve", diversity_reserve_slots=1),
        top_k=3,
    )

    assert [item.chunk.id for item in result[:2]] == ["a1", "a2"]


def test_mmr_puts_dissimilar_chunk_second():
    items = _items()

    result = apply_diversity(
        items,
        _settings(diversity_mode="mmr", diversity_lambda=0.3),
        top_k=3,
    )

    assert result[0].chunk.id == "a1", "первое место всегда за самым релевантным"
    assert result[1].chunk.id == "b1", "при низком λ вторым идёт наименее похожий"


def test_mmr_with_lambda_one_preserves_order():
    """При λ=1 новизна не учитывается вовсе — порядок обязан остаться прежним."""
    items = _items()

    result = apply_diversity(items, _settings(diversity_mode="mmr", diversity_lambda=1.0), top_k=3)

    assert [item.chunk.id for item in result] == [item.chunk.id for item in items]


def test_nothing_is_lost():
    """Переупорядочивание не должно терять кандидатов: отсечка бывает шире top_k."""
    items = _items()

    for mode in ("mmr", "reserve"):
        result = apply_diversity(items, _settings(diversity_mode=mode), top_k=3)
        assert len(result) == len(items), f"режим {mode} потерял кандидатов"
        assert {item.chunk.id for item in result} == {item.chunk.id for item in items}


def test_empty_input_is_safe():
    assert apply_diversity([], _settings(diversity_mode="mmr"), top_k=3) == []


def test_fewer_items_than_top_k_is_safe():
    items = _items()[:2]

    for mode in ("mmr", "reserve"):
        result = apply_diversity(items, _settings(diversity_mode=mode), top_k=8)
        assert len(result) == 2
