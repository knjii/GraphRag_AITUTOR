"""Размышление модели не должно доходить ни до пользователя, ни до метрик.

Разбор сохранённых ответов прогона 2026-08-19 показал, что до пользователя
доходил поток рассуждений: 46% ответов содержали тег think, 60% начинались
с английского «The user is asking», а в конфигурации с выдачей 16 таких
ответов было 100%. Ещё 38.9% ответов другого прогона — сообщение об отказе
инференса: лимита в 768 токенов не хватало, размышление съедало его целиком
и content приходил пустым.

Метрики при этом выглядели правдоподобно — верность 1.116, сохранность
формул 0.05 — и вели разбор причин в сторону поиска и размера модели.
Поэтому здесь два рубежа: вырезание блока размышления в клиенте
и признак «размышление вместо ответа» в оценке.
"""

from __future__ import annotations

from rag_textbook.clients.llm import strip_reasoning
from rag_textbook.config import LLMSettings
from rag_textbook.evaluation.answers import (
    AnswerOutcome,
    latin_share,
    looks_like_reasoning,
    summarize_answers,
)

# ------------------------------------------------- вырезание в клиенте

def test_closed_reasoning_block_is_removed():
    text = "<think>Сначала посмотрю контекст…</think>Определитель равен нулю."

    assert strip_reasoning(text) == "Определитель равен нулю."


def test_unclosed_reasoning_means_there_is_no_answer():
    """Незакрытый блок означает, что лимит токенов кончился посреди мысли.

    Отдавать обрывок чужих рассуждений хуже, чем честно вернуть пустоту:
    вызывающий код увидит отказ и не примет размышление за ответ.
    """
    assert strip_reasoning("<think>Мне нужно найти формулу и…") == ""


def test_plain_answer_survives_untouched():
    assert strip_reasoning("Определитель равен нулю.") == "Определитель равен нулю."


def test_multiline_and_mixed_case_are_handled():
    text = "<THINK>\nдлинное\nрассуждение\n</THINK>\n\nОтвет."

    assert strip_reasoning(text) == "Ответ."


def test_empty_input_is_safe():
    assert strip_reasoning("") == ""
    assert strip_reasoning(None) == ""  # type: ignore[arg-type]


# ------------------------------------------------- настройка размышления

def test_chat_reasoning_is_off_by_default():
    """Пустое значение означало «решает модель», и модель решала размышлять.

    При лимите ответа 768 токенов это давало пустой content в 39% случаев.
    """
    assert LLMSettings(_env_file=None).reasoning_effort_for("chat") == "none"


def test_chat_reasoning_can_be_enabled_explicitly():
    settings = LLMSettings(_env_file=None, LLM_CHAT_REASONING_EFFORT="medium")

    assert settings.reasoning_effort_for("chat") == "medium"
    assert settings.reasoning_effort_for("utility") == "none"


# ------------------------------------------------------- признаки в оценке

def test_english_reasoning_opener_is_detected():
    assert looks_like_reasoning("The user is asking about the determinant.")
    assert looks_like_reasoning("Looking at the context: [1] discusses…")
    assert looks_like_reasoning("<think>рассуждение</think>")


def test_normal_answer_is_not_flagged():
    assert not looks_like_reasoning("Определитель равен произведению собственных значений.")
    assert not looks_like_reasoning("")


def test_language_share_separates_russian_from_english():
    assert latin_share("The user asks about matrices") > 0.9
    assert latin_share("Определитель равен нулю") == 0.0
    # Формулы латиницу содержат законно, и мера обязана это переживать.
    assert latin_share(r"Ортогональная матрица: $A^{\mathrm{T}} A = I$") < 0.5


def test_summary_shows_both_new_signals():
    """Они обязаны быть видны в сводке. Не будь их, вчерашний прогон
    снова выглядел бы удачным."""
    outcomes = [
        AnswerOutcome(
            question_id="q1",
            question_type="formula_table",
            reasoning_leak=True,
            latin_share=0.9,
        ),
        AnswerOutcome(question_id="q2", question_type="formula_table"),
    ]

    summary = summarize_answers(outcomes)

    assert summary["всего"]["размышление вместо ответа"] == 0.5
    assert summary["всего"]["ответ не по-русски"] == 0.5
