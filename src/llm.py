from __future__ import annotations

import time
from typing import Any, Optional

import aiohttp
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
import requests

from settings import Settings

try:
    from langchain_community.chat_models import ChatOllama as _CommunityChatOllama  # type: ignore
except Exception as exc:  # pragma: no cover
    raise RuntimeError("langchain_community ChatOllama is required") from exc


def _is_auto_no_think_model(model_name: str) -> bool:
    name = str(model_name).strip().lower()
    return name.startswith("qwen3") or name.startswith("deepseek-r1")


def _resolve_request_timeout_seconds(settings: Settings) -> int:
    configured = int(getattr(settings, "ollama_request_timeout_seconds", 0) or 0)
    if configured > 0:
        return configured
    # deepseek-r1 is noticeably slower on local hardware, use safer timeout in auto mode.
    return 600 if str(settings.ollama_model).strip().lower().startswith("deepseek-r1") else 180


def _parse_think_flag(value: str, model_name: str) -> Optional[bool]:
    raw = str(value or "auto").strip().lower()
    if raw in {"1", "true", "yes", "on"}:
        return True
    if raw in {"0", "false", "no", "off"}:
        return False
    if _is_auto_no_think_model(model_name):
        # qwen3*/deepseek-r1* often put text in `thinking`; disable by default in query path.
        return False
    return None


class ChatOllamaCompat(_CommunityChatOllama):
    """ChatOllama wrapper that supports top-level `think` in Ollama /api/chat."""

    think: Optional[bool] = None
    num_batch: Optional[int] = None
    fallback_model: Optional[str] = None

    def _build_ollama_options(self, stop=None, **kwargs: Any) -> dict:
        options = {
            "num_ctx": self.num_ctx,
            "num_gpu": self.num_gpu,
            "num_batch": self.num_batch,
            "num_thread": self.num_thread,
            "num_predict": self.num_predict,
            "repeat_penalty": self.repeat_penalty,
            "temperature": self.temperature,
            "stop": self.stop if self.stop is not None else stop,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }
        for key, value in kwargs.items():
            if key in {"stop", "think"}:
                continue
            options[key] = value
        return {k: v for k, v in options.items() if v is not None}

    def _resolve_think(self, kwargs: dict) -> Optional[bool]:
        if "think" in kwargs and kwargs["think"] is not None:
            return bool(kwargs["think"])
        return self.think

    @staticmethod
    def _is_read_timeout(exc: Exception) -> bool:
        if isinstance(exc, requests.exceptions.ReadTimeout):
            return True
        msg = str(exc).lower()
        return "read timed out" in msg or "timed out" in msg

    def _maybe_release_embedding_runner(self) -> None:
        cfg = Settings()
        embed_model = str(cfg.ollama_embed_model or "").strip()
        if not embed_model:
            return
        try:
            requests.post(
                f"{self.base_url}/api/chat",
                json={"model": embed_model, "messages": [], "keep_alive": 0, "stream": False},
                timeout=10,
            )
        except Exception:
            pass

    @staticmethod
    def _is_ollama_503(exc: Exception) -> bool:
        return "status code: 503" in str(exc).lower()

    def _chat_once(
        self,
        *,
        ollama_messages: list[dict],
        think_value: Optional[bool],
        options: dict,
        model_override: Optional[str] = None,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": model_override or self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }
        if self.format:
            payload["format"] = self.format
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if think_value is not None:
            payload["think"] = think_value

        resp = requests.post(
            f"{self.base_url}/api/chat",
            json=payload,
            timeout=self.timeout or 180,
        )
        if resp.status_code != 200:
            raise ValueError(
                f"Ollama call failed with status code: {resp.status_code}. Details: {resp.text}"
            )
        return resp.json()

    @staticmethod
    def _is_ggml_assert_failure(exc: Exception) -> bool:
        text = str(exc).lower()
        return "status code: 500" in text and "ggml_assert" in text

    async def _achat_once(
        self,
        *,
        ollama_messages: list[dict],
        think_value: Optional[bool],
        options: dict,
    ) -> dict:
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": ollama_messages,
            "stream": False,
            "options": options,
        }
        if self.format:
            payload["format"] = self.format
        if self.keep_alive is not None:
            payload["keep_alive"] = self.keep_alive
        if think_value is not None:
            payload["think"] = think_value

        timeout = aiohttp.ClientTimeout(total=self.timeout or 180)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(f"{self.base_url}/api/chat", json=payload) as resp:
                if resp.status != 200:
                    details = await resp.text()
                    raise ValueError(
                        f"Ollama call failed with status code: {resp.status}. Details: {details}"
                    )
                return await resp.json()

    def _generate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        think_value = self._resolve_think(kwargs)
        options = self._build_ollama_options(stop=stop, **kwargs)
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        response = None
        for attempt in range(2):
            try:
                response = self._chat_once(
                    ollama_messages=ollama_messages,
                    think_value=think_value,
                    options=options,
                )
                break
            except Exception as exc:
                if attempt == 0 and self._is_ollama_503(exc):
                    self._maybe_release_embedding_runner()
                    time.sleep(0.8)
                    continue
                if attempt == 0 and self._is_read_timeout(exc):
                    # one safe retry for slow local models
                    time.sleep(0.8)
                    continue
                if (
                    attempt == 0
                    and self._is_ggml_assert_failure(exc)
                    and self.fallback_model
                    and self.fallback_model != self.model
                ):
                    # qwen2.5vl can fail on some text-only prompts/local VRAM regimes.
                    # Retry once with explicit fallback text model to keep query path alive.
                    response = self._chat_once(
                        ollama_messages=ollama_messages,
                        think_value=False,
                        options=options,
                        model_override=self.fallback_model,
                    )
                    break
                raise
        if response is None:  # pragma: no cover
            raise RuntimeError("Ollama response is unexpectedly None")
        message = response.get("message", {}) if isinstance(response, dict) else {}
        content = str(message.get("content", "") or "")
        if not content.strip() and think_value is not False:
            # if answer accidentally ended in `thinking`, retry once with think disabled
            fallback = self._chat_once(
                ollama_messages=ollama_messages,
                think_value=False,
                options=options,
            )
            fallback_message = fallback.get("message", {}) if isinstance(fallback, dict) else {}
            fallback_content = str(fallback_message.get("content", "") or "")
            if fallback_content.strip():
                response = fallback
                content = fallback_content
                think_value = False
        chat_generation = ChatGeneration(
            message=AIMessage(content=content),
            generation_info={
                "done_reason": response.get("done_reason") if isinstance(response, dict) else None,
                "think": think_value,
            },
        )
        return ChatResult(generations=[chat_generation])

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop=None,
        run_manager=None,
        **kwargs: Any,
    ) -> ChatResult:
        think_value = self._resolve_think(kwargs)
        options = self._build_ollama_options(stop=stop, **kwargs)
        ollama_messages = self._convert_messages_to_ollama_messages(messages)
        response = None
        for attempt in range(2):
            try:
                response = await self._achat_once(
                    ollama_messages=ollama_messages,
                    think_value=think_value,
                    options=options,
                )
                break
            except Exception as exc:
                if attempt == 0 and self._is_ollama_503(exc):
                    self._maybe_release_embedding_runner()
                    time.sleep(0.8)
                    continue
                if attempt == 0 and self._is_read_timeout(exc):
                    # one safe retry for slow local models
                    time.sleep(0.8)
                    continue
                raise
        if response is None:  # pragma: no cover
            raise RuntimeError("Ollama async response is unexpectedly None")
        message = response.get("message", {}) if isinstance(response, dict) else {}
        content = str(message.get("content", "") or "")
        if not content.strip() and think_value is not False:
            fallback = await self._achat_once(
                ollama_messages=ollama_messages,
                think_value=False,
                options=options,
            )
            fallback_message = fallback.get("message", {}) if isinstance(fallback, dict) else {}
            fallback_content = str(fallback_message.get("content", "") or "")
            if fallback_content.strip():
                response = fallback
                content = fallback_content
                think_value = False
        chat_generation = ChatGeneration(
            message=AIMessage(content=content),
            generation_info={
                "done_reason": response.get("done_reason") if isinstance(response, dict) else None,
                "think": think_value,
            },
        )
        return ChatResult(generations=[chat_generation])


def get_chat_model(settings: Settings) -> ChatOllamaCompat:
    think_value = _parse_think_flag(settings.ollama_think, settings.ollama_model)
    return ChatOllamaCompat(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.ollama_num_ctx,
        top_k=settings.ollama_top_k,
        num_gpu=settings.ollama_num_gpu,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
        timeout=_resolve_request_timeout_seconds(settings),
        think=think_value,
        fallback_model=(settings.ollama_fallback_model or None),
    )
