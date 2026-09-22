"""Офлайн-среда RL: эпизоды по слепку и функция награды для TRL."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_textbook.generation.answering import build_answer_messages
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.rl.env import (
    Example,
    build_examples,
    completion_text,
    load_jsonl,
    make_reward_function,
    save_jsonl,
    split_by_docs,
)


def _chunk(cid: str, doc: str, text: str) -> Chunk:
    return Chunk(id=cid, doc_id=doc, doc_name=f"{doc}.pdf", source_path=f"{doc}.pdf",
                 ordinal=int(cid.split(":")[1]), text=text, pages=[1])


def _goldset(tmp_path: Path) -> Path:
    path = tmp_path / "goldset.json"
    path.write_text(json.dumps({"questions": [{
        "id": "q1", "question": "Что такое скалярное произведение?",
        "question_type": "formula_table", "gold_chunk_ids": ["d1:1"],
        "gold_doc_ids": ["d1"], "answer": "…",
    }]}, ensure_ascii=False), encoding="utf-8")
    return path


@pytest.fixture()
def chunks() -> dict[str, Chunk]:
    return {
        "d1:1": _chunk("d1:1", "d1", "Скалярное произведение: $$\\langle x, y \\rangle = x^{T} A y$$."),
        "d1:2": _chunk("d1:2", "d1", "Норма задаётся через скалярное произведение."),
    }


def test_prompt_matches_service(settings, chunks, tmp_path):
    row = {"question_id": "q1", "question": "Что такое скалярное произведение?",
           "question_type": "formula_table", "final": ["d1:2", "d1:1"]}
    [example] = build_examples(settings, [row], chunks, _goldset(tmp_path))
    _, messages = build_answer_messages(
        settings, row["question"],
        [ScoredChunk(chunk=chunks[c], score=1.0) for c in row["final"]],
    )
    assert example.messages == [{"role": m.role, "content": m.content} for m in messages]
    # Контекст награды — фрагменты без инструкций промпта.
    assert messages[0].content.endswith(example.context)
    assert example.context.startswith("[1]")
    assert settings.prompts.qa_system not in example.context
    assert example.gold_in_context
    assert "x^{T} A y" in example.reference


def test_gold_outside_context_is_flagged(settings, chunks, tmp_path):
    row = {"question_id": "q1", "question": "?", "question_type": "formula_table", "final": ["d1:2"]}
    [example] = build_examples(settings, [row], chunks, _goldset(tmp_path))
    assert not example.gold_in_context


def test_reference_is_limited_to_visible_gold(settings, chunks, tmp_path):
    """Связывающий вопрос: второй эталонный фрагмент в контекст не попал."""
    path = tmp_path / "goldset.json"
    path.write_text(json.dumps({"questions": [{
        "id": "q1", "question": "?", "question_type": "graph_linked",
        "gold_chunk_ids": ["d1:1", "d1:2"], "gold_doc_ids": ["d1"], "answer": "…",
    }]}, ensure_ascii=False), encoding="utf-8")
    row = {"question_id": "q1", "question": "?", "question_type": "graph_linked", "final": ["d1:1"]}
    [example] = build_examples(settings, [row], chunks, path)
    assert example.gold_in_context
    assert example.reference == chunks["d1:1"].text


def test_unknown_question_is_an_error(settings, chunks, tmp_path):
    row = {"question_id": "zzz", "question": "?", "question_type": "x", "final": ["d1:1"]}
    with pytest.raises(ValueError):
        build_examples(settings, [row], chunks, _goldset(tmp_path))


def test_split_keeps_test_books_out_of_training():
    def ex(qid, docs):
        return Example(qid, "t", "?", [], "", "", True, docs)

    train, test = split_by_docs([ex("a", ["d1"]), ex("b", ["d2"]), ex("c", ["d1", "d2"])], {"d1"})
    assert [e.question_id for e in train] == ["b"]
    assert [e.question_id for e in test] == ["a", "c"]


def test_jsonl_round_trip(tmp_path):
    example = Example("q", "t", "вопрос", [{"role": "user", "content": "вопрос"}], "к", "э", False, ["d"])
    path = tmp_path / "set.jsonl"
    assert save_jsonl([example], path) == 1
    assert load_jsonl(path) == [example]


def test_completion_text_accepts_both_trl_formats():
    assert completion_text("ответ") == "ответ"
    assert completion_text([{"role": "assistant", "content": "ответ"}]) == "ответ"


def test_reward_function_uses_dataset_columns():
    reward = make_reward_function()
    context = "Контекст: $$\\langle x, y \\rangle = x^{T} A y$$ — скалярное произведение."
    values = reward(
        completions=[
            [{"role": "assistant", "content": "Это $$\\langle x, y \\rangle = x^{T} A y$$, скалярное произведение [1]."}],
            [{"role": "assistant", "content": ""}],
        ],
        context=[context, context],
        reference=[context, context],
        gold_in_context=[True, True],
        prompts=["?", "?"],
    )
    assert values[0] > 0 > values[1]



def test_unfit_questions_are_kept_out_of_training():
    """Задача 019: импортированный набор обходил проверки утечек."""
    from rag_textbook.evaluation.verdicts import QuestionVerdict, VerdictSet
    from rag_textbook.rl.env import drop_unfit

    def ex(qid, question):
        return Example(qid, "t", question, [], "", "", True, ["d"])

    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("bad", "unanswerable"))
    verdicts.add(QuestionVerdict("good", "ok"))
    kept, reasons = drop_unfit(
        [
            ex("good", "Как определяется ортогональная матрица?"),
            ex("bad", "Что такое ранг матрицы?"),
            ex("leak", "Что сказано в данном отрывке о базисе?"),
            ex("num", "Как записана формула (8.24)?"),
        ],
        verdicts,
    )
    assert [e.question_id for e in kept] == ["good"]
    assert reasons == {"вердикт:unanswerable": 1, "отсылка к тексту": 1,
                       "номер формулы или раздела": 1}
