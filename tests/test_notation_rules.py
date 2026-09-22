"""Обозначения по правилам (К3) и их область в структурном графе."""

from __future__ import annotations

from rag_textbook.graph.notation import contains_symbol, find_notations
from rag_textbook.graph.structural import build_structural


def _pairs(text: str) -> list[tuple[str, str]]:
    return [(item.symbol, item.meaning) for item in find_notations(text)]


def test_where_chain_with_formulas_and_plain_symbols() -> None:
    text = "Здесь $$y = f(x)$$ где $x$ — вектор признаков, $y$ — метка, а θ — параметр модели."
    assert _pairs(text) == [("x", "вектор признаков"), ("y", "метка"), ("θ", "параметр модели")]


def test_denote_takes_meaning_before_the_verb() -> None:
    text = "Множество всех матриц размера m на n обозначим через $\\mathbb { R } ^ { m \\times n }$."
    assert _pairs(text) == [("\\mathbb { R } ^ { m \\times n }", "всех матриц размера m на n")]


def test_denotes_after_symbol_and_this_form() -> None:
    assert _pairs("Здесь $C ^ { 0 }$ обозначает набор непрерывных функций.") == [
        ("C ^ { 0 }", "набор непрерывных функций")
    ]
    assert _pairs("где y это метка объекта.") == [("y", "метка объекта")]


def test_cyrillic_word_is_not_a_symbol() -> None:
    assert _pairs("где приближение — это оценка") == []
    assert _pairs("Далее обозначим. Затем") == []


def test_symbol_match_is_by_tokens_not_substring() -> None:
    assert contains_symbol("Имеем $\\gamma + x _ { 1 }$.", "x _ { 1 }")
    assert not contains_symbol("Имеем $\\gamma$.", "a")
    assert contains_symbol("Имеем $a + b$.", "a")


def test_notation_scope_is_the_section() -> None:
    chunks = [
        {"id": "c1", "doc_id": "b", "doc_name": "К", "ordinal": 1, "headers": ["1.1. Матрицы"],
         "text": "Пусть $A x = b$, где $A$ — матрица системы."},
        {"id": "c2", "doc_id": "b", "doc_name": "К", "ordinal": 2, "headers": ["1.1. Матрицы"],
         "text": "Ранг $A$ не превосходит числа строк."},
        {"id": "c3", "doc_id": "b", "doc_name": "К", "ordinal": 3, "headers": ["1.2. Другое"],
         "text": "Здесь $A$ означает событие."},
    ]
    plain, _ = build_structural(chunks)
    assert not any(row["kind"] == "notation" for row in plain.entities.values())

    graph, report = build_structural(chunks, notation=True)
    notation_ids = [eid for eid, row in graph.entities.items() if row["kind"] == "notation"]
    assert len(notation_ids) == 1
    eid = notation_ids[0]
    assert graph.mentions["c1"][eid][1] == "defines"
    assert graph.mentions["c2"][eid][1] == "uses"
    # Другой раздел — другая область: там $A$ может значить другое.
    assert eid not in graph.mentions.get("c3", {})
    assert report["books"]["b"]["notation"] == {"nodes": 1, "uses": 1}

    # Глобальный вариант К3: та же пара, область — вся книга.
    wide, _ = build_structural(chunks, notation=True, notation_scope="book")
    assert eid in wide.mentions["c3"]
