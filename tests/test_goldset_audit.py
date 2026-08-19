"""Аудит эталонного набора и приёмка при сборке.

Проверяются два разных рубежа. Первый — аудит: он находит изъяны в уже
существующем наборе. Второй — приёмка при сборке: она не даёт этим изъянам
попасть в набор заново. Второй рубеж важнее первого: измерено, что 17.7%
связывающих вопросов набора опираются на страницу упражнений или оглавление,
и все они прошли через отбор пар, который таких фрагментов не различал.
"""

from __future__ import annotations

import pytest

from rag_textbook.evaluation.audit import (
    audit_questions,
    find_near_duplicates,
    summarize_audit,
)
from rag_textbook.evaluation.goldset import (
    GoldsetBuilder,
    classify_chunk,
    references_numbering,
)
from rag_textbook.models import Chunk, GoldQuestion


def _chunk(identifier: str, text: str, *, headers: list[str] | None = None, ordinal: int = 0) -> Chunk:
    return Chunk(
        id=identifier,
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=ordinal,
        text=text,
        headers=headers or [],
    )


def _question(
    identifier: str,
    text: str,
    *,
    answer: str = "Развёрнутый эталонный ответ по существу.",
    chunk_ids: list[str] | None = None,
    question_type: str = "single_chunk",
) -> GoldQuestion:
    return GoldQuestion(
        id=identifier,
        question=text,
        answer=answer,
        gold_chunk_ids=chunk_ids or ["c1"],
        gold_doc_ids=["doc"],
        question_type=question_type,
    )


# --------------------------------------------------------- ссылки на номера

@pytest.mark.parametrize(
    "text",
    [
        "Каким образом вычисляется вектор в методе степенных итераций согласно формуле (10.52)?",
        "Какую проблему демонстрирует график на рис. 8.8?",
        "Какое условие выпуклости указано в разделах 7.3.1 и 7.3.2?",
        "Какую роль играют матричные разложения в контексте главы 4?",
    ],
)
def test_numbered_references_are_detected(text: str):
    """Вопрос про номер формулы неразрешим поиском: номер живёт рядом с самой
    формулой, а человек спросит про предмет, а не про нумерацию."""
    assert references_numbering(text)


def test_ordinary_question_is_not_flagged():
    assert not references_numbering("Какое свойство отличает ортогональную матрицу?")


# ------------------------------------------------------- негодные фрагменты

def test_table_of_contents_is_recognized():
    text = "\n".join(f"Глава {index} ....... {index * 7}" for index in range(1, 12))
    assert classify_chunk(_chunk("c1", text)) == "оглавление"


def test_exercise_page_is_recognized():
    chunk = _chunk("c1", "1. Докажите, что...", headers=["УПРАЖНЕНИЯ 3.4"])
    assert classify_chunk(chunk) == "упражнения"


def test_substantive_chunk_passes():
    chunk = _chunk("c1", "Ортогональная матрица — это матрица, обратная к которой равна транспонированной.")
    assert classify_chunk(chunk) == "содержательный"


# -------------------------------------------------------------- аудит целиком

def test_audit_marks_structural_gold_chunk():
    chunks = {
        "c1": _chunk("c1", "Содержательный текст про матрицы и их разложения."),
        "c2": _chunk("c2", "1. Докажите...", headers=["УПРАЖНЕНИЯ"]),
    }
    question = _question("q1", "Что такое сингулярное разложение?", chunk_ids=["c1", "c2"])

    audit = audit_questions([question], chunks)[0]

    assert "structural_chunk" in audit.defects
    assert audit.chunk_kinds == ["содержательный", "упражнения"]
    assert not audit.usable


def test_audit_reports_missing_chunk():
    question = _question("q1", "Что такое ядро линейного отображения?", chunk_ids=["нет-такого"])

    audit = audit_questions([question], {})[0]

    assert "missing_chunk" in audit.defects


def test_audit_flags_thin_answer():
    question = _question("q1", "Что такое ранг матрицы?", answer="Число")

    assert "thin_answer" in audit_questions([question], None)[0].defects


def test_near_duplicates_keep_the_earlier_question():
    first = _question("q1", "Какое свойство отличает ортогональную матрицу от произвольной?")
    second = _question("q2", "Какое свойство отличает ортогональную матрицу от произвольной матрицы?")
    third = _question("q3", "Как определяется ранг матрицы через линейную независимость строк?")

    duplicates = find_near_duplicates([first, second, third])

    assert duplicates == {"q2": "q1"}


def test_summary_splits_defects_by_type():
    good = _question("q1", "Что такое ядро линейного отображения?")
    bad = _question("q2", "Что показано на рис. 4.2?", question_type="graph_linked")

    summary = summarize_audit(audit_questions([good, bad], None))

    assert summary["годных"] == 1
    assert summary["изъяны"]["numbered_reference"] == 1
    assert summary["по типам"]["graph_linked"]["numbered_reference"] == 1


# ------------------------------------------------------ приёмка при сборке

class _StubLLM:
    """Модель, отвечающая заготовленным JSON. Сборка не должна ходить в сеть."""

    def __init__(self) -> None:
        self.calls = 0

    def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        return (
            '{"question": "Какое свойство отличает ортогональную матрицу?",'
            ' "answer": "Обратная к ней равна транспонированной."}'
        )


def _corpus() -> list[Chunk]:
    body = (
        "Ортогональная матрица определяется равенством произведения матрицы "
        "на транспонированную единичной матрице, что задаёт сохранение длин "
        "векторов и углов между ними при линейном отображении пространства. "
    ) * 3
    return [
        _chunk("плотный", body, ordinal=1),
        _chunk("упражнения", body, headers=["УПРАЖНЕНИЯ 2.1"], ordinal=2),
    ]


def test_exercise_pages_are_not_selected_as_gold():
    """Без этого отбора страница упражнений попадает в эталон и вопрос
    становится неотвечаемым: лексика общая, содержания нет."""
    builder = GoldsetBuilder(_StubLLM(), seed=1)

    produced = builder.build(_corpus(), single_count=2, multihop_count=0)

    selected = {chunk_id for item in produced for chunk_id in item.gold_chunk_ids}
    assert "упражнения" not in selected
    assert selected == {"плотный"}


def test_verifier_rejects_question_and_records_reason():
    builder = GoldsetBuilder(_StubLLM(), seed=1)

    produced = builder.build(
        _corpus(), single_count=2, multihop_count=0, verifier=lambda _: "single_hop_enough"
    )

    assert produced == []
    assert builder.failures["вердикт:single_hop_enough"] == 1


def test_accepted_question_is_marked_verified():
    builder = GoldsetBuilder(_StubLLM(), seed=1)

    produced = builder.build(_corpus(), single_count=2, multihop_count=0, verifier=lambda _: "ok")

    assert [item.verified for item in produced] == [True]
    assert "абляция" in produced[0].notes
