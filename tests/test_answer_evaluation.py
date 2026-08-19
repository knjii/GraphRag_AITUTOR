"""Оценка ответов.

Проверяются прежде всего две объективные величины — сохранность формул
и доля выдумки, — потому что именно их можно предъявлять как есть. Судейские
оценки считает та же модель, что и отвечает, и потому годятся они лишь
для сравнения конфигураций между собой.

Отдельно проверяется отказ отвечать: он бывает и удачей, и провалом,
и метрика обязана их различать.
"""

from __future__ import annotations

from rag_textbook.evaluation.answers import (
    AnswerOutcome,
    evaluate_answer,
    is_refusal,
    latex_overlap,
    summarize_answers,
    unsupported_share,
)
from rag_textbook.models import Answer, Chunk, GoldQuestion, ScoredChunk


def _chunk(text: str, identifier: str = "c1") -> Chunk:
    return Chunk(
        id=identifier,
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=1,
        text=text,
    )


def _answer(text: str, contexts: list[str]) -> Answer:
    return Answer(
        question="вопрос",
        answer=text,
        contexts=[
            ScoredChunk(chunk=_chunk(item, f"c{index}"), score=1.0)
            for index, item in enumerate(contexts)
        ],
    )


# ------------------------------------------------------------- формулы

def test_latex_is_counted_when_preserved():
    reference = r"Определение: $$A A^{\mathrm{T}} = I = A^{\mathrm{T}} A$$ для ортогональных."
    answer = r"Ортогональная матрица удовлетворяет $$A A^{\mathrm{T}} = I = A^{\mathrm{T}} A$$."

    expected, found = latex_overlap(reference, answer)

    assert expected == 1
    assert found == 1


def test_latex_is_counted_as_lost_when_paraphrased():
    """Пересказ формулы прозой — это потеря, а не эквивалент.

    Ровно эта подмена уже случалась в чанкере: LaTeX заменялся описанием
    от модели зрения, и для учебника математики это было критично.
    """
    reference = r"Определение: $$A A^{\mathrm{T}} = I = A^{\mathrm{T}} A$$"
    answer = "Произведение матрицы на транспонированную даёт единичную матрицу."

    expected, found = latex_overlap(reference, answer)

    assert expected == 1
    assert found == 0


def test_spacing_inside_latex_does_not_break_match():
    """Пробелы внутри формулы расставляются как придётся и смысла не меняют."""
    reference = r"$$A A ^ { \mathrm { T } } = I$$"
    answer = r"получаем $$AA^{\mathrm{T}}=I$$"

    expected, found = latex_overlap(reference, answer)

    assert (expected, found) == (1, 1)


def test_short_fragments_are_ignored():
    """Однобуквенные обозначения совпадают случайно и метрику бы засорили."""
    reference = "величина $x$ и $y$"
    answer = "ответ про $x$"

    assert latex_overlap(reference, answer) == (0, 0)


# -------------------------------------------------------------- выдумка

def test_answer_from_context_has_low_invention():
    context = (
        "Сингулярное разложение раскладывает матрицу на три множителя "
        "и применяется для понижения размерности данных."
    )
    answer = "Сингулярное разложение раскладывает матрицу на три множителя."

    assert unsupported_share(answer, context) == 0.0


def test_invented_answer_is_detected():
    context = "Сингулярное разложение раскладывает матрицу на три множителя."
    answer = (
        "Метод опорных векторов ищет разделяющую гиперплоскость "
        "с максимальным зазором между классами."
    )

    assert unsupported_share(answer, context) > 0.8


def test_short_answer_is_not_penalized():
    """Ответ короче окна оценить нельзя — и выдумкой его считать нельзя тоже."""
    assert unsupported_share("Переобучение", "любой контекст") == 0.0


# --------------------------------------------------------------- отказ

def test_refusal_is_recognized():
    assert is_refusal("В предоставленном контексте нет данных для ответа.")
    assert not is_refusal("Определитель равен произведению собственных значений.")


def test_refusal_is_visible_in_summary():
    """Доля отказов обязана быть видна отдельно: при непустом контексте
    отказ — это провал, и в средней «верности» он растворился бы."""
    outcomes = [
        AnswerOutcome(question_id="q1", question_type="single_chunk", refused=True),
        AnswerOutcome(question_id="q2", question_type="single_chunk", refused=False),
    ]

    summary = summarize_answers(outcomes)

    assert summary["всего"]["отказов"] == 0.5


# ------------------------------------------------------------ в сборе

def test_evaluation_without_judge_still_produces_objective_metrics():
    """Без модели-судьи объективные величины обязаны считаться.

    Иначе замер оказался бы возможен только на арендованном сервере,
    а половина его смысла — в том, что часть метрик от модели не зависит.
    """
    question = GoldQuestion(
        id="q1",
        question="Какое свойство у ортогональной матрицы?",
        gold_chunk_ids=["c1"],
        answer="Обратная равна транспонированной.",
        question_type="formula_table",
    )
    produced = _answer(
        r"Для неё $$A ^ { - 1 } = A ^ { \mathrm { T } }$$, то есть обратная равна транспонированной.",
        [r"Ортогональная матрица: $$A ^ { - 1 } = A ^ { \mathrm { T } }$$"],
    )

    outcome = evaluate_answer(
        question,
        produced,
        reference_text=r"Ортогональная матрица: $$A ^ { - 1 } = A ^ { \mathrm { T } }$$",
        llm=None,
    )

    assert outcome.correctness is None, "судьи не было — оценки быть не должно"
    assert outcome.latex_recall == 1.0
    assert outcome.refused is False
    assert 0.0 <= outcome.unsupported <= 1.0


def test_summary_splits_by_type():
    outcomes = [
        AnswerOutcome(
            question_id="q1", question_type="graph_linked", correctness=1, groundedness=2
        ),
        AnswerOutcome(
            question_id="q2", question_type="formula_table", correctness=2, groundedness=2
        ),
    ]

    summary = summarize_answers(outcomes)

    assert summary["по типам"]["graph_linked"]["верность"] == 1.0
    assert summary["по типам"]["formula_table"]["верность"] == 2.0
    assert summary["всего"]["верность"] == 1.5


def test_judge_failure_does_not_lose_objective_metrics():
    """Сбой судьи не должен обесценивать прогон: объективное считается всё равно."""

    class _BrokenJudge:
        settings = None

        def chat(self, *args, **kwargs):  # noqa: ANN002, ANN003
            raise RuntimeError("судья недоступен")

    question = GoldQuestion(
        id="q1", question="вопрос", gold_chunk_ids=["c1"], question_type="single_chunk"
    )
    produced = _answer("Ответ по контексту.", ["Ответ по контексту."])

    outcome = evaluate_answer(question, produced, reference_text="", llm=_BrokenJudge())

    assert outcome.correctness is None
    assert outcome.unsupported == 0.0


# ------------------------------------------- выбор файла с фрагментами

def test_answers_refuse_a_foreign_corpus(tmp_path, monkeypatch):
    """Замер обязан остановиться, когда фрагментов эталонного набора нет.

    Так уже случилось: рядом с учебником лежали 609 файлов публичного
    набора, выборка «первым попавшимся файлом» вернула новостную статью,
    эталонные фрагменты не нашлись — и сохранность формул, главная
    объективная величина замера, молча исчезла из сводки. Прогон при этом
    выглядел удачным.
    """
    import json as _json

    from typer.testing import CliRunner

    from rag_textbook.cli.main import app

    parsed = tmp_path / "parsed"
    parsed.mkdir()
    # Чужой корпус: идентификаторы документов не совпадают с эталонными.
    (parsed / "чужой_chunks.json").write_text(
        _json.dumps(
            [
                {
                    "id": "чужой:00000",
                    "doc_id": "чужой",
                    "doc_name": "новость",
                    "source_path": "x",
                    "ordinal": 0,
                    "text": "Посторонний текст.",
                }
            ]
        ),
        encoding="utf-8",
    )
    goldset = tmp_path / "goldset.json"
    goldset.write_text(
        _json.dumps(
            {
                "version": 1,
                "count": 1,
                "questions": [
                    {
                        "id": "q1",
                        "question": "вопрос",
                        "answer": "ответ по существу",
                        "gold_chunk_ids": ["учебник:00001"],
                        "gold_doc_ids": ["учебник"],
                        "question_type": "formula_table",
                    }
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("PARSED_DIR", str(parsed))

    result = CliRunner().invoke(
        app, ["eval", "answers", "--goldset", str(goldset), "--no-judge"]
    )

    assert result.exit_code == 1
    assert "Сохранность формул" in result.output or "формул" in result.output
