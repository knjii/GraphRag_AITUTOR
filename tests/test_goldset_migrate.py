"""Перенос эталона на новую нарезку (перенарезка MML, 2026-09-22)."""

from __future__ import annotations

import pytest

from rag_textbook.evaluation.migrate import chunks_fingerprint, migrate_goldset
from rag_textbook.models import GoldQuestion

A = "Матрица называется обратимой, если существует обратная матрица. " * 3
B = "Определитель произведения равен произведению определителей $$ \\det(AB) = \\det A \\det B $$. " * 3
C = "Собственный вектор умножается на число при действии оператора. " * 3


def _chunk(cid: str, text: str) -> dict:
    return {"id": cid, "doc_id": "d", "text": text, "text_hash": str(hash(text))}


def _question(qid: str, gold: list[str], question: str, answer: str) -> GoldQuestion:
    return GoldQuestion(id=qid, question=question, gold_chunk_ids=gold, answer=answer)


def test_split_chunk_goes_to_the_part_with_the_answer() -> None:
    old = {"d:00000": _chunk("d:00000", A + B), "d:00001": _chunk("d:00001", C)}
    # Новая нарезка сняла $$ и разрезала первый фрагмент надвое.
    new = [_chunk("d:00000", A), _chunk("d:00001", B.replace("$$", "")), _chunk("d:00002", C)]
    q = _question("q1", ["d:00000"], "Чему равен определитель произведения?", "Произведению определителей")
    migrated, report = migrate_goldset([q], old, new)
    assert migrated[0].gold_chunk_ids == ["d:00001"]
    assert not report.ambiguous and not report.lost


def test_pair_keeps_two_distinct_chunks_and_collision_drops_question() -> None:
    old = {"d:00000": _chunk("d:00000", A), "d:00001": _chunk("d:00001", C)}
    new = [_chunk("d:00005", A), _chunk("d:00006", C)]
    pair = _question("q2", ["d:00000", "d:00001"], "Обратимая матрица и собственный вектор", "")
    migrated, _ = migrate_goldset([pair], old, new)
    assert migrated[0].gold_chunk_ids == ["d:00005", "d:00006"]

    merged = [_chunk("d:00007", A + C)]
    migrated, report = migrate_goldset([pair], old, merged)
    assert migrated == [] and report.collided == ["q2"]


def test_missing_text_loses_the_question() -> None:
    old = {"d:00000": _chunk("d:00000", A)}
    migrated, report = migrate_goldset(
        [_question("q3", ["d:00000"], "Что такое обратимая матрица?", "")], old, [_chunk("d:00000", C)]
    )
    assert migrated == [] and report.lost == ["q3"]


def test_fix_overrides_choice_and_must_exist() -> None:
    old = {"d:00000": _chunk("d:00000", A + B)}
    new = [_chunk("d:00000", A), _chunk("d:00001", B)]
    q = _question("q4", ["d:00000"], "Определитель произведения", "")
    migrated, _ = migrate_goldset([q], old, new, {"q4": {"d:00000": "d:00000"}})
    assert migrated[0].gold_chunk_ids == ["d:00000"]
    with pytest.raises(ValueError):
        migrate_goldset([q], old, new, {"q4": {"d:00000": "d:00099"}})


def test_fingerprint_depends_on_order_and_text() -> None:
    rows = [_chunk("d:00000", A), _chunk("d:00001", B)]
    assert chunks_fingerprint(rows) == chunks_fingerprint([dict(r) for r in rows])
    assert chunks_fingerprint(rows) != chunks_fingerprint(rows[::-1])
    assert chunks_fingerprint(rows) != chunks_fingerprint([rows[0], _chunk("d:00001", C)])
