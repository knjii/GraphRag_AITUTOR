"""Дозапись в эталонный набор.

Расширять набор пересборкой нельзя: сборка перезаписывает файл целиком,
и все прежние прогоны становятся несравнимыми. Между тем сравнимость —
единственное, ради чего набор и существует: абсолютные цифры по нему
не заявляются, пока он не вычитан человеком.

Отдельно проверяется сохранение отметки `verified`: её ставят вручную,
и потерять её при дозаписи значит потерять всю ручную работу.
"""

from __future__ import annotations

from rag_textbook.evaluation.goldset import merge_goldsets
from rag_textbook.models import GoldQuestion


def _question(index: int, *, text: str | None = None, verified: bool = False) -> GoldQuestion:
    return GoldQuestion(
        id=f"q-{index}",
        question=text or f"Вопрос номер {index} про матричные разложения?",
        gold_chunk_ids=[f"doc:{index:05d}"],
        gold_doc_ids=["doc"],
        answer=f"Ответ {index}",
        question_type="single_chunk",
        expected_hops=1,
        verified=verified,
    )


def test_existing_questions_survive_untouched():
    existing = [_question(1, verified=True), _question(2)]
    added = [_question(3), _question(4)]

    merged, appended = merge_goldsets(existing, added)

    assert appended == 2
    assert [item.id for item in merged] == ["q-1", "q-2", "q-3", "q-4"]
    assert merged[0].verified is True, "ручная отметка потеряна"


def test_repeated_id_is_not_added_twice():
    existing = [_question(1), _question(2)]

    merged, appended = merge_goldsets(existing, [_question(2), _question(3)])

    assert appended == 1
    assert [item.id for item in merged] == ["q-1", "q-2", "q-3"]


def test_same_question_from_another_chunk_is_caught():
    """Идентификатор считается от текста и фрагмента, поэтому один и тот же
    вопрос от другого фрагмента получил бы новый идентификатор."""
    existing = [_question(1, text="Что такое сингулярное разложение?")]
    duplicate = _question(99, text="Что   такое   Сингулярное Разложение?")

    merged, appended = merge_goldsets(existing, [duplicate])

    assert appended == 0
    assert len(merged) == 1


def test_appending_to_empty_set_works():
    merged, appended = merge_goldsets([], [_question(1), _question(2)])

    assert appended == 2
    assert len(merged) == 2
