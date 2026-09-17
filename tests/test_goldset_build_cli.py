"""`goldset build`: исключение тестовой книги и дозапись без приёмки."""

from __future__ import annotations

from types import SimpleNamespace

from typer.testing import CliRunner

from rag_textbook.cli import main as cli
from rag_textbook.evaluation.goldset import exclude_documents, load_goldset, save_goldset
from rag_textbook.models import Chunk, GoldQuestion


def _chunk(doc_id: str, doc_name: str, ordinal: int) -> Chunk:
    return Chunk(id=f"{doc_id}:{ordinal}", doc_id=doc_id, doc_name=doc_name,
                 source_path=f"{doc_id}.pdf", ordinal=ordinal, text="текст")


CHUNKS = [
    _chunk("0690bb81b7e3c831", "Дайзенрот. Математика в машинном обучении", 0),
    _chunk("axler", "en-la-axler", 0),
    _chunk("chernova", "ru-prob-chernova", 0),
]


def test_exclude_by_id_or_name_part():
    kept = exclude_documents(CHUNKS, ["0690bb81b7e3c831", "AXLER"])
    assert [c.doc_id for c in kept] == ["chernova"]


class _Builder:
    seen: list[Chunk] = []

    def __init__(self, *_, **__):
        self.failures = {}

    def build(self, chunks, **_):
        _Builder.seen = list(chunks)
        return [
            GoldQuestion(id=f"new-{c.doc_id}", question="?", answer="!", gold_chunk_ids=[c.id],
                         gold_doc_ids=[c.doc_id], question_type="single_chunk")
            for c in chunks
        ]


def _invoke(monkeypatch, tmp_path, *extra):
    context = SimpleNamespace(
        vector_store=SimpleNamespace(iter_chunks=lambda: iter(CHUNKS)),
        llm=None, graph_store=None, close=lambda: None,
    )
    monkeypatch.setenv("RAG_ENV_FILE", "tests-no-such-env-file")
    monkeypatch.setattr(cli, "build_context", lambda settings: context)
    monkeypatch.setattr(cli, "GoldsetBuilder", _Builder)
    target = tmp_path / "train.json"
    result = CliRunner().invoke(cli.app, ["goldset", "build", "--output", str(target), *extra])
    return result, target


def test_training_questions_skip_the_test_book(monkeypatch, tmp_path):
    result, target = _invoke(monkeypatch, tmp_path, "--exclude-doc", "Дайзенрот")
    assert result.exit_code == 0, result.output
    assert {c.doc_id for c in _Builder.seen} == {"axler", "chernova"}
    assert len(load_goldset(target)) == 2


def test_append_without_verify_keeps_existing(monkeypatch, tmp_path):
    """Прежде ветка else относилась к --verify и затирала набор."""
    old = GoldQuestion(id="old", question="?", answer="!", gold_chunk_ids=["x:1"],
                       gold_doc_ids=["x"], question_type="single_chunk")
    save_goldset([old], tmp_path / "train.json")
    result, target = _invoke(monkeypatch, tmp_path, "--append", "--seed", "7")
    assert result.exit_code == 0, result.output
    assert "old" in {q.id for q in load_goldset(target)}
