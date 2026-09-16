"""Генерация по замороженному контексту.

Нужна для сравнения генераторов: если контекст берётся из слепка, все
модели отвечают по одному и тому же материалу, и разница относится
к генератору, а не к тому, что кому досталось при поиске. Заодно поиск
во время такого замера не выполняется вовсе — его службы не занимают
память карты, а её на 12B и так не хватает.
"""

from __future__ import annotations

from rag_textbook.config import Settings
from rag_textbook.evaluation.answers import run_answer_evaluation
from rag_textbook.generation.answering import AnswerGenerator
from rag_textbook.models import Chunk, GoldQuestion


class _LLM:
    """Отвечает пересказом того, что увидела: так видно, что ей подали."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        self.seen.append(messages[0].content)
        return "Ответ по контексту."


class _Retrieval:
    """Поиск, который обязан НЕ вызываться."""

    def __init__(self) -> None:
        self.calls = 0

    def retrieve(self, *args, **kwargs):  # noqa: ANN002, ANN003
        self.calls += 1
        raise AssertionError("поиск не должен выполняться при замороженном контексте")


class _Context:
    def __init__(self, generator: AnswerGenerator, llm: _LLM) -> None:
        self.generator = generator
        self.llm = llm


def _chunk(identifier: str, text: str) -> Chunk:
    return Chunk(
        id=identifier,
        doc_id="doc",
        doc_name="Учебник",
        source_path="учебник.pdf",
        ordinal=0,
        text=text,
    )


def _setup() -> tuple[_Context, dict[str, Chunk], _Retrieval]:
    settings = Settings(_env_file=None)
    llm = _LLM()
    retrieval = _Retrieval()
    generator = AnswerGenerator(settings, retrieval, llm)  # type: ignore[arg-type]
    chunks = {
        "c1": _chunk("c1", "Определитель равен произведению собственных значений."),
        "c2": _chunk("c2", "Собственные значения — корни характеристического многочлена."),
    }
    return _Context(generator, llm), chunks, retrieval


def _question() -> GoldQuestion:
    return GoldQuestion(
        id="q1",
        question="Чему равен определитель?",
        answer="Произведению собственных значений.",
        gold_chunk_ids=["c1"],
        gold_doc_ids=["doc"],
        question_type="single_chunk",
    )


def test_frozen_context_skips_retrieval_entirely():
    context, chunks, retrieval = _setup()

    summary, outcomes = run_answer_evaluation(
        context,
        [_question()],
        chunks=chunks,
        judge=False,
        max_workers=1,
        frozen_contexts={"q1": ["c1", "c2"]},
    )

    assert retrieval.calls == 0, "поиск выполнялся, значит сравнение генераторов нечестное"
    assert outcomes[0].context_size == 2
    assert summary["всего"]["вопросов"] == 1


def test_frozen_context_puts_exactly_the_listed_chunks_into_the_prompt():
    context, chunks, _ = _setup()

    run_answer_evaluation(
        context,
        [_question()],
        chunks=chunks,
        judge=False,
        max_workers=1,
        frozen_contexts={"q1": ["c2"]},
    )

    prompt = context.llm.seen[0]
    assert "характеристического многочлена" in prompt
    assert "произведению собственных значений" not in prompt.lower()


def test_missing_chunks_do_not_crash_the_run():
    """Слепок может ссылаться на фрагмент, которого нет в выгрузке.
    Замер обязан продолжиться, а не упасть на одном вопросе."""
    context, chunks, _ = _setup()

    _, outcomes = run_answer_evaluation(
        context,
        [_question()],
        chunks=chunks,
        judge=False,
        max_workers=1,
        frozen_contexts={"q1": ["нет-такого"]},
    )

    assert outcomes[0].context_size == 0


def test_ordinary_run_still_uses_retrieval():
    """Прежний путь обязан сохраниться: без слепка отвечаем как раньше."""
    context, chunks, retrieval = _setup()

    try:
        run_answer_evaluation(
            context, [_question()], chunks=chunks, judge=False, max_workers=1
        )
    except AssertionError:
        pass  # заглушка поиска намеренно падает — значит, он был вызван
    assert retrieval.calls == 1
