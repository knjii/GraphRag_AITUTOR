"""Удвоенная разметка выносных формул «$$$$ … $$$$».

Замер 2026-09-17: 762 из 1151 фрагмента индекса несут формулы, обёрнутые
дважды. MinerU отдаёт формулу уже с «$$», а ``Block.to_indexable_text``
оборачивал её ещё раз. Модель видела это в контексте, а награда R6
извлекала каждую десятую эталонную формулу с лишним долларом.
"""

from __future__ import annotations

from rag_textbook.config import Settings
from rag_textbook.generation.answering import build_answer_messages, build_context_block
from rag_textbook.models import Block, Chunk, ScoredChunk, strip_math_delimiters
from rag_textbook.utils.text import normalize_math_delimiters


def _chunk(text: str) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(id="d:1", doc_id="d", doc_name="Учебник", source_path="u.pdf", ordinal=0, text=text),
        score=1.0,
    )


def test_equation_from_mineru_is_wrapped_once():
    block = Block(type="equation", text="$$\na _ { i 1 } x _ { 1 } \\tag{2.2}\n$$")
    assert block.to_indexable_text(strip_delimiters=True) == "$$a _ { i 1 } x _ { 1 } \\tag{2.2}$$"
    # Без флага — прежнее поведение: индекс MML воспроизводим.
    assert block.to_indexable_text().startswith("$$$$")


def test_equation_without_delimiters_still_wrapped():
    assert Block(type="equation", latex="x^2").to_indexable_text() == "$$x^2$$"


def test_bracket_delimiters_are_stripped():
    assert strip_math_delimiters("\\[ x^2 \\]") == "x^2"


def test_inline_dollars_inside_are_kept():
    assert strip_math_delimiters("a = $b$ + c") == "a = $b$ + c"


def test_normalize_collapses_quadruple_dollars():
    text = "всего $$$$ a x \\tag{2.2} $$$$ единиц"
    assert normalize_math_delimiters(text) == "всего $$ a x \\tag{2.2} $$ единиц"


def test_context_block_normalizes_only_when_asked():
    chunks = [_chunk("всего $$$$ a x $$$$ единиц")]
    assert "$$$$" in build_context_block(chunks, 500)
    assert "$$$$" not in build_context_block(chunks, 500, normalize_math=True)


def test_flag_reaches_the_prompt_and_fingerprint(monkeypatch):
    monkeypatch.setenv("RAG_ENV_FILE", "tests-no-such-env-file")
    monkeypatch.setenv("CONTEXT_NORMALIZE_MATH_DELIMITERS", "true")
    settings = Settings()
    _, messages = build_answer_messages(settings, "?", [_chunk("всего $$$$ a x $$$$ единиц")])
    assert "$$$$" not in messages[0].content
    assert settings.prompts.fingerprint().endswith("+math")
