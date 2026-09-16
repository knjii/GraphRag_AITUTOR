"""Назначение вызова языковой модели: режим размышления, модель и адрес.

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
    """По умолчанию служебные вызовы идут той же моделью, что и ответы.

    Разделить их можно (см. ниже), но пока разделение не задано,
    поведение обязано остаться прежним.
    """
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
    """Движки, не знающие параметра, не должны получать его вовсе.

    Прежде для этого хватало пустого значения. Теперь размышление в ответе
    выключено по умолчанию (`none`), а пустая строка в окружении означает
    «не задано» — поэтому отказ от параметра выражается словом `off`.
    """
    client = OpenAICompatibleLLMClient(_settings(LLM_CHAT_REASONING_EFFORT="off"))
    payload = client._payload(
        [ChatMessage(role="user", content="привет")],
        purpose="chat",
        json_schema=None,
        max_tokens=64,
        temperature=0.0,
    )
    assert "reasoning_effort" not in payload


# --------------------------------------------------------------- имена

def test_all_purposes_fall_back_to_the_single_model():
    """Пустая настройка означает «там же, где всё остальное».

    Это важнее, чем кажется: конфигурация сервера не перечисляет назначения,
    и молчаливая подстановка чужой модели сломала бы прогон незаметно.
    """
    settings = _settings(LLM_MODEL="базовая")

    for purpose in ("chat", "utility", "extraction", "judge"):
        assert settings.model_for(purpose) == "базовая"


def test_chat_and_utility_are_separable():
    settings = _settings(
        LLM_MODEL="базовая", LLM_CHAT_MODEL="крупная", LLM_UTILITY_MODEL="мелкая"
    )

    assert settings.model_for("chat") == "крупная"
    assert settings.model_for("utility") == "мелкая"
    # Извлечение графа за собой не тянется: стадия отдельная и дорогая.
    assert settings.model_for("extraction") == "базовая"


def test_judge_can_differ_from_the_generator():
    """Судья обязан быть из другого семейства, иначе он одобряет сам себя."""
    settings = _settings(LLM_MODEL="qwen", LLM_JUDGE_MODEL="gemma")

    assert settings.model_for("judge") == "gemma"
    assert settings.model_for("chat") == "qwen"


# --------------------------------------------------------------- адреса

def test_base_url_defaults_everywhere():
    settings = _settings(LLM_BASE_URL="http://сервер:8001/v1")

    for purpose in ("chat", "utility", "vision", "extraction", "judge"):
        assert settings.base_url_for(purpose) == "http://сервер:8001/v1"


def test_utility_can_live_on_another_server():
    settings = _settings(
        LLM_BASE_URL="http://крупная:8001/v1",
        LLM_UTILITY_BASE_URL="http://мелкая:11434/v1",
    )

    assert settings.base_url_for("chat") == "http://крупная:8001/v1"
    assert settings.base_url_for("utility") == "http://мелкая:11434/v1"


def test_trailing_slash_does_not_produce_a_double_slash():
    settings = _settings(LLM_UTILITY_BASE_URL="http://мелкая:11434/v1/")

    assert settings.base_url_for("utility") == "http://мелкая:11434/v1"


# ------------------------------------------------- запрос уходит по адресу

def test_request_goes_to_the_endpoint_of_its_purpose(monkeypatch: pytest.MonkeyPatch):
    """Проверка сквозная: мало объявить адрес, запрос обязан на него уйти.

    Первая версия клиента брала адрес один раз в конструкторе, и настройка
    была бы молчаливым украшением — служебные вызовы всё равно уходили бы
    на крупную модель.
    """
    settings = _settings(
        LLM_BASE_URL="http://крупная:8001/v1",
        LLM_UTILITY_BASE_URL="http://мелкая:11434/v1",
        LLM_CHAT_MODEL="крупная-модель",
        LLM_UTILITY_MODEL="мелкая-модель",
        LLM_MAX_RETRIES=0,
    )
    client = OpenAICompatibleLLMClient(settings)
    seen: list[tuple[str, str]] = []

    class _Response:
        status_code = 200

        @staticmethod
        def json() -> dict:
            return {"choices": [{"message": {"content": "ответ"}}]}

    class _Client:
        def post(self, url: str, json: dict) -> _Response:  # noqa: A002
            seen.append((url, json["model"]))
            return _Response()

    monkeypatch.setattr(client, "_sync_client", lambda: _Client())

    client.chat([ChatMessage(role="user", content="вопрос")], purpose="chat")
    client.chat([ChatMessage(role="user", content="служебное")], purpose="utility")

    assert seen == [
        ("http://крупная:8001/v1/chat/completions", "крупная-модель"),
        ("http://мелкая:11434/v1/chat/completions", "мелкая-модель"),
    ]


def test_client_still_works_without_any_split():
    """Прежнее поведение обязано сохраниться: у сервера ничего не задано."""
    settings = _settings(LLM_BASE_URL="http://один:8001/v1", LLM_MODEL="одна")
    client = OpenAICompatibleLLMClient(settings)

    assert isinstance(client, OpenAICompatibleLLMClient)
    assert settings.base_url_for("chat") == settings.base_url_for("utility")
    assert settings.model_for("chat") == settings.model_for("utility") == "одна"
