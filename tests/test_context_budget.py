"""Сборка контекста: бюджет символов, порядок фрагментов, промпт.

Измерено 2026-08-19: до ответа дословно доходит 0.05 формул эталонного
фрагмента. Одна из причин найдена арифметикой — бюджет делился поровну,
«половина окна / число фрагментов», и при выдаче 16 давал 768 символов
на фрагмент при медианной длине чанка 1204. В промпт не попадало 43% формул.

Здесь проверяется, что этого больше не происходит: формульным фрагментам
достаётся больше места, неиспользованный остаток возвращается в общий котёл,
а усечение перестало быть молчаливым.
"""

from __future__ import annotations

from rag_textbook.config import PromptSettings
from rag_textbook.generation.answering import (
    allocate_budget,
    build_context_block,
    order_for_attention,
)
from rag_textbook.models import Chunk, ScoredChunk


def _scored(
    identifier: str, length: int, *, formula: bool = False, table: bool = False
) -> ScoredChunk:
    chunk = Chunk(
        id=identifier,
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=0,
        text="я" * length,
        has_formula=formula,
        has_table=table,
    )
    return ScoredChunk(chunk=chunk, score=1.0)


# ------------------------------------------------------------- бюджет

def test_formula_chunk_gets_more_room_than_plain_text():
    """У формулы нет середины, которую можно опустить без потери смысла."""
    chunks = [_scored("формула", 4000, formula=True), _scored("текст", 4000)]

    formula_budget, text_budget = allocate_budget(chunks, 4000, formula_share=1.6)

    assert formula_budget > text_budget


def test_short_chunk_returns_what_it_does_not_need():
    """Иначе короткий фрагмент занимал бы место, которое ему не нужно,
    а длинный — резался бы, хотя запас есть."""
    chunks = [_scored("короткий", 100), _scored("длинный", 5000)]

    short, long = allocate_budget(chunks, 3000)

    assert short == 100, "бюджет обрезан по фактической длине"
    assert long > 3000 // 2, "остаток ушёл тому, кому места не хватает"


def test_budget_is_never_absurdly_small():
    chunks = [_scored(str(i), 2000) for i in range(50)]

    budgets = allocate_budget(chunks, 1000)

    assert min(budgets) >= 200


def test_empty_input_gives_empty_budget():
    assert allocate_budget([], 1000) == []


def test_whole_chunk_fits_when_window_is_large_enough():
    """Проверка на числах настоящего корпуса: медианный чанк 1204 символа,
    при выдаче 8 и окне 16k он обязан проходить целиком."""
    chunks = [_scored(str(i), 1204) for i in range(8)]

    budgets = allocate_budget(chunks, int(16384 * 0.5 * 3))

    assert all(budget >= 1204 for budget in budgets)


# -------------------------------------------------------------- порядок

def test_edges_order_puts_the_best_at_both_ends():
    chunks = [_scored(str(i), 500) for i in range(6)]

    ordered = order_for_attention(chunks, "edges")

    positions = [item.chunk.id for item in ordered]
    assert positions[0] == "0", "лучшее — первым"
    assert positions[-1] == "1", "второе по качеству — последним"
    assert positions[len(positions) // 2] in {"4", "5"}, "слабое — в середине"


def test_relevance_order_is_unchanged():
    chunks = [_scored(str(i), 500) for i in range(6)]

    assert order_for_attention(chunks, "relevance") == chunks


def test_short_context_is_not_reordered():
    """На трёх фрагментах середины нет, а перестановка сбила бы нумерацию."""
    chunks = [_scored(str(i), 500) for i in range(3)]

    assert order_for_attention(chunks, "edges") == chunks


# --------------------------------------------------------- сборка блока

def test_per_chunk_budgets_are_applied():
    chunks = [_scored("а", 1000), _scored("б", 1000)]

    block = build_context_block(chunks, [300, 900])

    first, second = block.split("\n\n")
    assert len(first) < len(second)


def test_single_number_still_works():
    """Прежний вызов с одним числом обязан работать: он остался в коде."""
    chunks = [_scored("а", 1000)]

    assert build_context_block(chunks, 500)


# ---------------------------------------------------------------- промпт

def test_prompt_demands_verbatim_formulas():
    """Прежняя формулировка «формулы приводи в LaTeX» говорила про формат
    записи, но не запрещала пересказ, — а теряется именно на пересказе."""
    prompt = PromptSettings(_env_file=None).qa_system.lower()

    assert "дословно" in prompt
    assert "latex" in prompt
    assert "пересказ" in prompt


def test_prompt_fingerprint_changes_with_the_prompt():
    """Отпечаток попадает в файлы метрик: замер обязан нести с собой то,
    чем он сделан. Промпт подменяется переменной окружения, и однажды это
    уже сбило разбор причин."""
    default = PromptSettings(_env_file=None)
    changed = PromptSettings(_env_file=None, QA_SYSTEM_PROMPT="другой промпт")

    assert default.fingerprint() != changed.fingerprint()
    assert default.fingerprint().startswith(default.prompt_version)
