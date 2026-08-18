"""Вердикты ручной проверки эталонного набора.

Главное свойство, ради которого вердикты вынесены в отдельный файл: они
применяются к набору по идентификатору вопроса, поэтому переживают
расширение набора. Проверка идёт на ноутбуке по копии на 163 вопроса,
а на сервере набор уже 388 — и станет больше. Если бы отметка жила внутри
набора, каждое расширение стирало бы ручную работу.
"""

from __future__ import annotations

from pathlib import Path

from rag_textbook.evaluation.verdicts import (
    QuestionVerdict,
    VerdictSet,
    apply_verdicts,
    summarize,
)
from rag_textbook.models import GoldQuestion


def _question(number: int, question_type: str = "graph_linked") -> GoldQuestion:
    return GoldQuestion(
        id=f"q-{number}",
        question=f"Вопрос {number} о матричных разложениях?",
        gold_chunk_ids=[f"doc:{number:05d}"],
        question_type=question_type,
        expected_hops=2 if question_type == "graph_linked" else 1,
    )


def test_usable_verdicts_mark_question_verified():
    questions = [_question(1), _question(2)]
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "ok"))
    verdicts.add(QuestionVerdict("q-2", "single_hop_enough"))

    updated, counts = apply_verdicts(questions, verdicts)

    assert [item.verified for item in updated] == [True, True]
    assert counts == {"ok": 1, "single_hop_enough": 1}


def test_broken_questions_are_not_marked_verified():
    """Негодный вопрос помечается вердиктом, но проверенным не считается."""
    questions = [_question(1)]
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "unanswerable", note="фрагмент не отвечает"))

    updated, _ = apply_verdicts(questions, verdicts)

    assert updated[0].verified is False
    assert "unanswerable" in updated[0].notes
    assert "фрагмент не отвечает" in updated[0].notes


def test_questions_without_verdict_are_untouched():
    """Набор больше проверенной выборки: молча помечать непроверенное нельзя."""
    questions = [_question(1), _question(2)]
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "ok"))

    updated, counts = apply_verdicts(questions, verdicts)

    assert updated[1].verified is False
    assert updated[1].notes == ""
    assert sum(counts.values()) == 1


def test_verdicts_survive_goldset_growth():
    """Порядок операций не должен иметь значения.

    Расширить набор и потом применить вердикты — то же самое, что применить
    и потом расширить. Ради этого свойства вердикты и вынесены наружу.
    """
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "ok"))

    small = [_question(1)]
    grown = [_question(1), _question(2), _question(3)]

    applied_then_grown, _ = apply_verdicts(small, verdicts)
    applied_then_grown = applied_then_grown + [_question(2), _question(3)]
    grown_then_applied, _ = apply_verdicts(grown, verdicts)

    assert [item.verified for item in applied_then_grown] == [
        item.verified for item in grown_then_applied
    ]


def test_existing_note_is_preserved():
    question = _question(1).model_copy(update={"notes": "от генератора"})
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "ok"))

    updated, _ = apply_verdicts([question], verdicts)

    assert updated[0].notes.startswith("от генератора")
    assert "проверка: ok" in updated[0].notes


def test_summary_splits_by_question_type():
    questions = [
        _question(1, "graph_linked"),
        _question(2, "graph_linked"),
        _question(3, "single_chunk"),
    ]
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "single_hop_enough"))
    verdicts.add(QuestionVerdict("q-2", "ok"))
    verdicts.add(QuestionVerdict("q-3", "ok"))

    summary = summarize(questions, verdicts)

    assert summary["graph_linked"] == {"single_hop_enough": 1, "ok": 1}
    assert summary["single_chunk"] == {"ok": 1}


def test_round_trip_through_file(tmp_path: Path):
    path = tmp_path / "verdicts.json"
    verdicts = VerdictSet()
    verdicts.add(QuestionVerdict("q-1", "leaky", note="ссылка на номер рисунка"))
    verdicts.save(path)

    restored = VerdictSet.load(path)

    assert len(restored) == 1
    assert restored.verdicts["q-1"].verdict == "leaky"
    assert restored.verdicts["q-1"].note == "ссылка на номер рисунка"


def test_missing_file_loads_as_empty(tmp_path: Path):
    assert len(VerdictSet.load(tmp_path / "нет-такого.json")) == 0


def test_recorded_verdicts_file_is_valid():
    """Файл с реальными вердиктами обязан читаться и быть непустым.

    Проверка ручная и дорогая; сломанный файл обесценил бы её незаметно.
    """
    path = Path("evaluation/goldsets/verdicts.json")
    if not path.exists():
        return
    verdicts = VerdictSet.load(path)
    assert len(verdicts) > 0
    allowed = {"ok", "single_hop_enough", "unanswerable", "ambiguous", "leaky"}
    assert {item.verdict for item in verdicts.verdicts.values()} <= allowed
