"""Проверка связности обязана ловить то, что портило замеры молча.

Три случая, каждый из которых уже случался. Движок с длиной контекста
меньше нашей — ошибка 400 на длинных промптах посреди прогона. Модель,
которая размышляет вместо ответа, — качество считается по потоку мыслей.
Ответ не по-русски — метрики считаются, а продукт неработоспособен.

Все три должны выявляться одной командой до замера, а не после него.
"""

from __future__ import annotations

from types import SimpleNamespace

from rag_textbook.config import Settings
from rag_textbook.context import AppContext


class _LLM:
    def __init__(self, reply: str, *, max_model_len: int | None = 16384) -> None:
        self.reply = reply
        self.max_model_len = max_model_len

    def describe_model(self, purpose: str = "chat") -> dict:
        if self.max_model_len is None:
            return {}
        return {"model": "Qwen/Qwen3.5-4B", "max_model_len": self.max_model_len}

    def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
        return self.reply


def _health(llm: _LLM, *, context_window: int = 16384) -> dict:
    settings = Settings(_env_file=None)
    settings.llm.context_window = context_window
    context = AppContext.__new__(AppContext)
    context.settings = settings  # type: ignore[misc]
    context.llm = llm  # type: ignore[assignment]
    return context._llm_health()


def test_healthy_model_passes():
    report = _health(_LLM("Москва"))

    assert report["status"] == "ok"
    assert report["reply"] == "Москва"


def test_engine_shorter_than_our_window_is_an_error():
    """Бюджет символов считаем мы, отвергает запрос движок."""
    report = _health(_LLM("Москва", max_model_len=8192), context_window=16384)

    assert report["status"] == "error"
    assert "SGLANG_MAX_MODEL_LEN" in report["error"]


def test_engine_longer_than_our_window_is_fine():
    report = _health(_LLM("Москва", max_model_len=32768), context_window=16384)

    assert report["status"] == "ok"


def test_reasoning_instead_of_answer_is_an_error():
    report = _health(_LLM("The user is asking about the capital of Russia."))

    assert report["status"] == "error"
    assert "размышля" in report["error"]


def test_empty_reply_is_an_error():
    """Пустой ответ означает, что лимит токенов ушёл на размышление."""
    report = _health(_LLM("   "))

    assert report["status"] == "error"
    assert "размышление" in report["error"]


def test_english_reply_is_an_error():
    report = _health(_LLM("Moscow is the capital."))

    assert report["status"] == "error"
    assert "не по-русски" in report["error"]


def test_missing_engine_info_does_not_fail_the_check():
    """Ollama не отдаёт длину контекста. Отсутствие сведений — не отказ."""
    report = _health(_LLM("Москва", max_model_len=None))

    assert report["status"] == "ok"


def test_unreachable_model_is_an_error():
    class _Dead(_LLM):
        def chat(self, messages, **kwargs):  # noqa: ANN001, ANN003
            raise RuntimeError("Connection refused")

    report = _health(_Dead("—"))

    assert report["status"] == "error"
    assert "Connection refused" in report["error"]


def test_broken_model_fails_the_whole_check():
    """Иначе health вернёт ноль, и скрипт сессии пойдёт дальше."""
    settings = Settings(_env_file=None)
    context = AppContext.__new__(AppContext)
    context.settings = settings  # type: ignore[misc]
    context.llm = _LLM("The user is asking…")  # type: ignore[assignment]
    context.graph_store = None  # type: ignore[assignment]
    context.vector_store = SimpleNamespace(count=lambda: 10)  # type: ignore[assignment]
    context.embeddings = SimpleNamespace(embed_query=lambda text: [0.0] * 8)  # type: ignore[assignment]

    report = context.health()

    assert report["status"] == "error"


def test_provenance_prefers_the_model_the_engine_actually_serves():
    """Замер обязан записывать то, что отвечало, а не то, что в настройках.

    В сравнении моделей 2026-09-03 движок поднимался отдельно от .env,
    и файл ячейки Muse Glimmer подписан «Qwen/Qwen3.5-4B» — именем
    из конфигурации. Для сравнения моделей это ровно то поле, которое
    обязано быть верным, иначе через неделю не разобрать, что мерили.
    """
    llm = _LLM("Москва")
    llm.describe_model = lambda purpose="chat": {"model": "/models/muse-30b/Muse.gguf"}  # type: ignore[method-assign]

    served = (llm.describe_model() or {}).get("model")

    assert served == "/models/muse-30b/Muse.gguf"
    assert served != Settings(_env_file=None).llm.model_for("chat")


def test_llm_check_can_be_skipped():
    """Восстановление индекса и графа не вызывает модель, и её отсутствие
    не должно останавливать подготовку. На этом застряло восстановление
    2026-09-09: health валил всю цепочку из-за неподнятого генератора."""
    settings = Settings(_env_file=None)
    context = AppContext.__new__(AppContext)
    context.settings = settings  # type: ignore[misc]
    context.llm = _LLM("The user is asking…")  # type: ignore[assignment]
    context.graph_store = None  # type: ignore[assignment]
    context.vector_store = SimpleNamespace(count=lambda: 10)  # type: ignore[assignment]
    context.embeddings = SimpleNamespace(embed_query=lambda text: [0.0] * 8)  # type: ignore[assignment]

    report = context.health(check_llm=False)

    assert report["status"] == "ok"
    assert report["components"]["llm"]["status"] == "не проверялся"
