"""Клиент генеративной модели через OpenAI-совместимый ``/v1/chat/completions``.

Заменяет прежний ``ChatOllamaCompat`` — подкласс устаревшего
``langchain_community.ChatOllama`` на ~300 строк с сырыми ``requests``,
собственными ретраями и ручной обработкой поля ``think``.

Выигрыш: переключение Ollama → vLLM для пилота становится сменой ``LLM_BASE_URL``,
а не правкой кода; ограничение параллелизма и таймауты заданы в одном месте.
"""

from __future__ import annotations

import asyncio
import base64
import json
import re
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal, Protocol

import httpx
from pydantic import BaseModel

from rag_textbook.config import LLMSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.utils.retry import retry_async, retry_sync

logger = get_logger("clients.llm")

Role = Literal["system", "user", "assistant"]


class ChatMessage(BaseModel):
    role: Role
    content: str
    # Локальные пути к изображениям; кодируются в data-url при отправке.
    images: list[str] = []


class LLMClient(Protocol):
    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str: ...


def _image_to_data_url(path: str) -> str | None:
    file_path = Path(path)
    if not file_path.is_file():
        logger.warning("Изображение не найдено, пропускаю: %s", path)
        return None
    suffix = file_path.suffix.lower().lstrip(".") or "jpeg"
    mime = "jpeg" if suffix in {"jpg", "jpeg"} else suffix
    payload = base64.b64encode(file_path.read_bytes()).decode("ascii")
    return f"data:image/{mime};base64,{payload}"


def _to_openai_messages(messages: Sequence[ChatMessage]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for message in messages:
        if not message.images:
            result.append({"role": message.role, "content": message.content})
            continue
        parts: list[dict[str, Any]] = [{"type": "text", "text": message.content}]
        for image_path in message.images:
            data_url = _image_to_data_url(image_path)
            if data_url:
                parts.append({"type": "image_url", "image_url": {"url": data_url}})
        result.append({"role": message.role, "content": parts})
    return result


# Блок размышления рассуждающей модели. Qwen3.5 в режиме размышления
# возвращает его прямо в content, а не отдельным полем, и без вырезания
# он доходит до пользователя. Измерено 2026-08-19: 46% ответов содержали
# тег think, 60% начинались с рассуждения по-английски. Метрики при этом
# считались по потоку мыслей, а не по ответу.
_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_UNCLOSED_THINK = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> str:
    """Убирает цепочку рассуждений из ответа.

    Закрытый блок вырезается целиком. Незакрытый означает, что лимит токенов
    кончился посреди размышления и ответа в тексте нет вовсе — тогда
    возвращается пустая строка, и вызывающий код честно увидит отказ,
    а не обрывок чужих мыслей.
    """
    cleaned = _THINK_BLOCK.sub("", text or "").strip()
    if _UNCLOSED_THINK.match(cleaned):
        return ""
    return cleaned


class OpenAICompatibleLLMClient:
    def __init__(self, settings: LLMSettings) -> None:
        self.settings = settings
        self._base_url = settings.base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {settings.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        }
        self._client: httpx.Client | None = None
        self._aclient: httpx.AsyncClient | None = None
        # Ограничение параллелизма: локальный сервер инференса — общий ресурс,
        # без семафора десяток одновременных запросов просто выстроится в очередь
        # с непредсказуемыми таймаутами.
        self._semaphore = threading.Semaphore(settings.max_concurrency)
        self._asemaphore: asyncio.Semaphore | None = None

    def _sync_client(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                timeout=self.settings.timeout_seconds, headers=self._headers
            )
        return self._client

    def _async_client(self) -> httpx.AsyncClient:
        if self._aclient is None:
            limits = httpx.Limits(max_connections=self.settings.max_concurrency * 2)
            self._aclient = httpx.AsyncClient(
                timeout=self.settings.timeout_seconds, headers=self._headers, limits=limits
            )
        return self._aclient

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
            self._client = None

    async def aclose(self) -> None:
        if self._aclient is not None:
            await self._aclient.aclose()
            self._aclient = None

    def _payload(
        self,
        messages: Sequence[ChatMessage],
        purpose: str,
        json_schema: dict[str, Any] | None,
        max_tokens: int | None,
        temperature: float | None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model_for(purpose),  # type: ignore[arg-type]
            "messages": _to_openai_messages(messages),
            "temperature": (
                self.settings.temperature if temperature is None else float(temperature)
            ),
            "max_tokens": int(max_tokens or self.settings.max_tokens),
            "stream": False,
        }
        # Отключение размышления для служебных вызовов. Без него рассуждающая
        # модель тратит весь лимит токенов на цепочку рассуждений и возвращает
        # пустой content — извлечение графа получало invalid_json на каждом
        # чанке. Проверено на Ollama: `think` и `chat_template_kwargs` через
        # OpenAI-совместимый путь игнорируются, работает именно этот параметр.
        effort = self.settings.reasoning_effort_for(purpose)  # type: ignore[arg-type]
        if effort:
            payload["reasoning_effort"] = effort
        if json_schema is not None:
            # Строгий структурированный вывод. Прежняя реализация полагалась на
            # «просьбу вернуть JSON» и разбирала результат тремя эвристиками подряд.
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "extraction", "schema": json_schema, "strict": False},
            }
        return payload

    @staticmethod
    def _extract_content(data: dict[str, Any]) -> str:
        choices = data.get("choices") or []
        if not choices:
            return ""
        message = choices[0].get("message") or {}
        content = message.get("content")
        if isinstance(content, list):
            # Некоторые серверы отдают контент частями.
            return strip_reasoning(
                "".join(part.get("text", "") for part in content if isinstance(part, dict))
            )
        return strip_reasoning(str(content or ""))

    def describe_model(self, purpose: str = "chat") -> dict[str, Any]:
        """Что за модель отвечает по этому адресу и с какой длиной контекста.

        Нужно из-за расхождения, которое иначе проявляется поздно и невнятно.
        Бюджет символов считаем мы, по ``LLM_CONTEXT_WINDOW``; отвергает
        запрос движок, по своей ``--context-length``. Если движок короче,
        длинные промпты падают ошибкой 400 где-то посреди замера, а причина
        выглядит как поломка модели.

        Возвращает пустой словарь, если сервер не отдаёт эти сведения:
        отсутствие проверки не должно быть отказом.
        """
        try:
            response = self._sync_client().get(
                f"{self.settings.base_url_for(purpose)}/models"  # type: ignore[arg-type]
            )
            if response.status_code >= 400:
                return {}
            entries = response.json().get("data") or []
        except Exception:  # noqa: BLE001
            return {}
        if not entries:
            return {}
        first = entries[0]
        # Имена полей разнятся по движкам: sglang отдаёт max_model_len,
        # vllm — его же, ollama не отдаёт ничего.
        length = first.get("max_model_len") or first.get("context_length")
        return {"model": first.get("id", ""), "max_model_len": length}

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = self._payload(messages, purpose, json_schema, max_tokens, temperature)

        def call() -> str:
            with self._semaphore:
                response = self._sync_client().post(
                    f"{self.settings.base_url_for(purpose)}/chat/completions",  # type: ignore[arg-type]
                    json=payload,
                )
            if response.status_code >= 400:
                raise RuntimeError(f"LLM вернул {response.status_code}: {response.text[:500]}")
            return self._extract_content(response.json())

        return retry_sync(
            call, description=f"llm:{purpose}", attempts=self.settings.max_retries + 1
        )

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = self._payload(messages, purpose, json_schema, max_tokens, temperature)
        if self._asemaphore is None:
            self._asemaphore = asyncio.Semaphore(self.settings.max_concurrency)

        async def call() -> str:
            assert self._asemaphore is not None
            async with self._asemaphore:
                response = await self._async_client().post(
                    f"{self.settings.base_url_for(purpose)}/chat/completions",  # type: ignore[arg-type]
                    json=payload,
                )
            if response.status_code >= 400:
                raise RuntimeError(f"LLM вернул {response.status_code}: {response.text[:500]}")
            return self._extract_content(response.json())

        return await retry_async(
            call, description=f"llm:{purpose}", attempts=self.settings.max_retries + 1
        )

    async def acomplete_raw(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> dict[str, str]:
        """Сырой текст и причина остановки — для RL-замеров.

        ``achat`` вычищает размышление и не сообщает об обрыве; награде
        нужно и то и другое: утечка размышления и обрыв на пределе токенов
        закрываются её воротами.
        """
        payload = self._payload(messages, purpose, None, max_tokens, temperature)
        if self._asemaphore is None:
            self._asemaphore = asyncio.Semaphore(self.settings.max_concurrency)

        async def call() -> dict[str, str]:
            assert self._asemaphore is not None
            async with self._asemaphore:
                response = await self._async_client().post(
                    f"{self.settings.base_url_for(purpose)}/chat/completions",  # type: ignore[arg-type]
                    json=payload,
                )
            if response.status_code >= 400:
                raise RuntimeError(f"LLM вернул {response.status_code}: {response.text[:500]}")
            choice = (response.json().get("choices") or [{}])[0]
            content = (choice.get("message") or {}).get("content")
            if isinstance(content, list):
                content = "".join(
                    part.get("text", "") for part in content if isinstance(part, dict)
                )
            return {
                "answer": str(content or ""),
                "finish_reason": str(choice.get("finish_reason") or ""),
            }

        return await retry_async(
            call, description=f"llm:{purpose}:raw", attempts=self.settings.max_retries + 1
        )


class FakeLLMClient:
    """Детерминированная заглушка для тестов.

    Возвращает заранее заданные ответы либо простое эхо. Позволяет прогонять
    весь конвейер ответа без сервера инференса.
    """

    def __init__(self, responses: Sequence[str] | None = None) -> None:
        self._responses = list(responses or [])
        self._calls: list[list[ChatMessage]] = []

    @property
    def calls(self) -> list[list[ChatMessage]]:
        return self._calls

    def _next(self, messages: Sequence[ChatMessage], json_schema: dict | None) -> str:
        self._calls.append(list(messages))
        if self._responses:
            return self._responses.pop(0)
        if json_schema is not None:
            return json.dumps({"entities": [], "relations": []}, ensure_ascii=False)
        user = next(
            (message.content for message in reversed(messages) if message.role == "user"), ""
        )
        return f"[fake-answer] {user[:200]}"

    def chat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._next(messages, json_schema)

    async def achat(
        self,
        messages: Sequence[ChatMessage],
        *,
        purpose: str = "chat",
        json_schema: dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        return self._next(messages, json_schema)

    def close(self) -> None:
        return None

    async def aclose(self) -> None:
        return None


def build_llm_client(settings: LLMSettings) -> LLMClient:
    if settings.provider == "fake":
        return FakeLLMClient()
    return OpenAICompatibleLLMClient(settings)
