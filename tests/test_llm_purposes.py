"""Назначение вызова языковой модели и режим размышления.

Регрессия на дефект, который проявился дважды и оба раза молча: рассуждающая
модель тратит весь лимит токенов на цепочку рассуждений и возвращает пустой
``content``. Сначала так исчезли все связи графа, затем — все вопросы
эталонного набора (140 обращений, ноль результатов). Общая причина одна:
служебный вызов ходил с назначением ``chat``, для которого размышление
намеренно не выключается.
"""

from __future__ import annotations

import pytest

from rag_textbook.clients.llm import ChatMessage, OpenAICompatibleLLMClient
from rag_textbook.config import LLMSettings

# Назначения, где ответ разбирает код, а не читает человек. Для всех них
# размышление должно быть выключено.
SERVICE_PURPOSES = ("utility", "vision", "extraction", "judge")


def _settings(**overrides: object) -> LLMSettings:
    return LLMSettings(_env_file=None, **overrides)  # type: ignore[arg-type]


@pytest.mark.parametrize("purpose", SERVICE_PURPOSES)
def test_service_purposes_disable_reasoning(purpose: str) -> None:
    settings = _settings()
    assert settings.reasoning_effort_for(purpose) == "none"


def test_chat_purpose_is_governed_separately() -> None:
    """Ответ пользователю — единственное назначение с собственной настройкой."""
    settings = _settings(LLM_CHAT_REASONING_EFFORT="medium")
    assert settings.reasoning_effort_for("chat") == "medium"
    assert settings.reasoning_effort_for("utility") == "none"


def test_utility_uses_the_chat_model() -> None:
    """Служебные текстовые вызовы идут той же моделью, что и ответы."""
    settings = _settings(LLM_MODEL="qwen3.5:4b", LLM_EXTRACTION_MODEL="other:1b")
    assert settings.model_for("utility") == "qwen3.5:4b"
    assert settings.model_for("chat") == "qwen3.5:4b"
    assert settings.model_for("extraction") == "other:1b"


@pytest.mark.parametrize("purpose", SERVICE_PURPOSES)
def test_payload_carries_reasoning_effort(purpose: str) -> None:
    """Параметр должен реально попадать в запрос, а не только в настройки."""
    client = OpenAICompatibleLLMClient(_settings())
    payload = client._payload(
        [ChatMessage(role="user", content="привет")],
        purpose=purpose,
        json_schema=None,
        max_tokens=64,
        temperature=0.0,
    )
    assert payload["reasoning_effort"] == "none"


def test_empty_effort_is_not_sent() -> None:
    """Движки, не знающие параметра, не должны получать его вовсе."""
    client = OpenAICompatibleLLMClient(_settings(LLM_CHAT_REASONING_EFFORT=""))
    payload = client._payload(
        [ChatMessage(role="user", content="привет")],
        purpose="chat",
        json_schema=None,
        max_tokens=64,
        temperature=0.0,
    )
    assert "reasoning_effort" not in payload
