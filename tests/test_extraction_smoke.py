"""Проба извлечения v4 до полного прогона графа (шаг G0 дня 2)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from rag_textbook.models import Chunk, Entity, ExtractionResult

_SPEC = importlib.util.spec_from_file_location(
    "extraction_smoke", Path(__file__).resolve().parents[1] / "scripts" / "extraction_smoke.py"
)
smoke = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(smoke)


def _chunk(i: int) -> Chunk:
    return Chunk(
        id=f"d:{i:05d}", doc_id="d", doc_name="Книга", source_path="x.pdf", ordinal=i, text="т"
    )


def _entity(name: str, role: str) -> Entity:
    return Entity(id=name, name=name, canonical=name, role=role)


def test_roles_in_range_pass() -> None:
    results = [
        (_chunk(0), ExtractionResult(entities=[_entity("а", "defines"), _entity("б", "uses")])),
        (_chunk(1), ExtractionResult(entities=[_entity("в", "uses")])),
    ]
    summary = smoke.summarize(results)
    assert summary["годен"] and summary["доля фрагментов с defines"] == 0.5


def test_missing_roles_and_fallbacks_fail() -> None:
    results = [
        (_chunk(0), ExtractionResult(entities=[_entity("а", "")])),
        (_chunk(1), ExtractionResult(status="rule_fallback", entities=[_entity("б", "")])),
    ]
    problems = " ".join(smoke.summarize(results)["проблемы"])
    assert "откатов" in problems and "роль" in problems and "defines" in problems
