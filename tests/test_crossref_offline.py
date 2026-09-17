"""Разбор перекрёстных ссылок учебника (R8).

Правила различают место определения и место упоминания. Если различение
сломается, граф ссылок молча выродится в «все со всеми внутри главы» —
ровно та ошибка, которую мы уже ловили у графа сущностей.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "crossref_offline", ROOT / "scripts" / "crossref_offline.py"
)
crossref = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(crossref)


def chunk(ordinal: int, text: str, *, headers=(), figure=False) -> dict:
    return {
        "id": f"doc:{ordinal:05d}",
        "doc_id": "doc",
        "ordinal": ordinal,
        "text": text,
        "headers": list(headers),
        "has_figure": figure,
    }


def test_tag_is_an_anchor_and_parentheses_are_a_reference():
    chunks = [
        chunk(0, r"Запишем $$ A x = b \tag{2.32} $$ для системы."),
        chunk(1, "Подставив (2.32) в исходное выражение, получаем результат."),
    ]
    graph = crossref.build(chunks)
    assert graph["edges"]["doc:00001"] == {"doc:00000"}
    assert graph["resolved"]["equation"] == 1
    # Номер внутри \tag сам по себе ссылкой не считается.
    assert graph["seen"]["equation"] == 1


def test_reference_inside_the_same_chunk_gives_no_edge():
    chunks = [chunk(0, r"$$ A x = b \tag{2.32} $$ Из (2.32) видно, что решение единственно.")]
    graph = crossref.build(chunks)
    assert graph["edges"] == {}
    assert graph["self_only"]["equation"] == 1


def test_statement_anchor_needs_a_name_in_brackets():
    chunks = [
        chunk(0, "Определение 2.10 (векторное подпространство). Пусть V — пространство."),
        chunk(1, "Как показано в определении 2.10, множество замкнуто."),
    ]
    graph = crossref.build(chunks)
    assert graph["edges"]["doc:00001"] == {"doc:00000"}


def test_section_reference_leads_to_the_first_chunk_of_the_section():
    chunks = [
        chunk(0, "текст", headers=["8.2. РАЗЛОЖЕНИЕ"]),
        chunk(1, "продолжение", headers=["8.2. РАЗЛОЖЕНИЕ"]),
        chunk(2, "Подробнее в разделе 8.2 показано, как это делается."),
    ]
    graph = crossref.build(chunks)
    # Ребро одно: к началу раздела, а не ко всем его фрагментам.
    assert graph["edges"]["doc:00002"] == {"doc:00000"}


def test_figure_caption_is_distinguished_from_a_mention():
    chunks = [
        chunk(0, "Рис. 1.1. Схема задач машинного обучения.", figure=True),
        chunk(1, "Четыре задачи показаны на рис. 1.1, и каждая разобрана ниже."),
    ]
    graph = crossref.build(chunks)
    assert graph["edges"]["doc:00001"] == {"doc:00000"}


def test_unresolved_reference_is_counted_but_creates_nothing():
    chunks = [chunk(0, "Как следует из (9.99), ряд сходится.")]
    graph = crossref.build(chunks)
    assert graph["seen"]["equation"] == 1 and graph["resolved"]["equation"] == 0
    assert graph["edges"] == {}


def test_distance_counts_hops():
    edges = {"a": {"b"}, "b": {"a", "c"}, "c": {"b"}}
    assert crossref.distance(edges, "a", "b", 2) == 1
    assert crossref.distance(edges, "a", "c", 2) == 2
    assert crossref.distance(edges, "a", "c", 1) is None


def test_gold_pairs_skips_unknown_and_single_chunk_questions(tmp_path: Path):
    path = tmp_path / "goldset.json"
    path.write_text(
        json.dumps(
            {
                "questions": [
                    {"id": "q1", "gold_chunk_ids": ["doc:00000", "doc:00001"]},
                    {"id": "q2", "gold_chunk_ids": ["doc:00000"]},
                    {"id": "q3", "gold_chunk_ids": ["doc:00000", "нет такого"]},
                    {"id": "q4", "gold_chunk_ids": ["doc:00000", "doc:00000"]},
                ]
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    pairs = crossref.gold_pairs(path, {"doc:00000", "doc:00001"})
    assert [p[0] for p in pairs] == ["q1"]
