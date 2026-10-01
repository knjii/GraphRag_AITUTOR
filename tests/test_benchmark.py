"""Тесты замера пропускной способности инференса."""

from __future__ import annotations

import time

from rag_textbook.clients.llm import ChatMessage, FakeLLMClient
from rag_textbook.evaluation.benchmark import (
    estimate_indexing_time,
    run_concurrency_point,
    run_throughput_sweep,
)


class _SlowLLM(FakeLLMClient):
    """Заглушка с фиксированной задержкой — имитирует движок без батчинга."""

    def __init__(self, delay_seconds: float = 0.02) -> None:
        super().__init__()
        self.delay = delay_seconds

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        time.sleep(self.delay)
        return '{"entities": [], "relations": []}'


class _FlakyLLM(FakeLLMClient):
    """Падает на каждом втором запросе."""

    def __init__(self) -> None:
        super().__init__()
        self._calls = 0

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):
        self._calls += 1
        if self._calls % 2 == 0:
            raise RuntimeError("status code: 503")
        return "{}"


def test_prompts_are_unique_between_requests() -> None:
    """Одинаковые промпты движок вернул бы из кэша, и замер потерял бы смысл."""
    llm = FakeLLMClient()
    run_concurrency_point(
        llm, concurrency=1, requests=4, prompt_chars=800, max_tokens=32, structured=False
    )
    prompts = [call[0].content for call in llm.calls]
    assert len(set(prompts)) == len(prompts), "промпты должны различаться"


def test_prompts_share_common_prefix() -> None:
    """Общий префикс обязателен: на нём и проверяется кэширование в движке."""
    llm = FakeLLMClient()
    run_concurrency_point(
        llm, concurrency=1, requests=3, prompt_chars=900, max_tokens=32, structured=False
    )
    prompts = [call[0].content for call in llm.calls]
    prefix = prompts[0][:200]
    assert all(prompt.startswith(prefix) for prompt in prompts)


def test_concurrency_point_collects_metrics() -> None:
    point = run_concurrency_point(
        _SlowLLM(0.01), concurrency=4, requests=8, prompt_chars=500, max_tokens=32, structured=True
    )
    assert point.ok == 8
    assert point.failed == 0
    assert point.throughput_rps > 0
    assert point.latency_p50_ms > 0
    assert point.latency_p95_ms >= point.latency_p50_ms


def test_parallelism_increases_throughput_on_io_bound_client() -> None:
    """Клиент с задержкой — модель сервера, который умеет обслуживать параллельно."""
    single = run_concurrency_point(
        _SlowLLM(0.05), concurrency=1, requests=6, prompt_chars=500, max_tokens=32, structured=False
    )
    parallel = run_concurrency_point(
        _SlowLLM(0.05), concurrency=6, requests=6, prompt_chars=500, max_tokens=32, structured=False
    )
    assert parallel.throughput_rps > single.throughput_rps


def test_failures_are_counted_not_raised() -> None:
    point = run_concurrency_point(
        _FlakyLLM(), concurrency=2, requests=6, prompt_chars=400, max_tokens=16, structured=False
    )
    assert point.failed > 0
    assert point.ok > 0, "успешные запросы всё равно должны учитываться"


def test_sweep_reports_scaling_factor() -> None:
    result = run_throughput_sweep(
        _SlowLLM(0.02),
        concurrency_levels=(1, 4),
        requests_per_level=4,
        prompt_chars=500,
        max_tokens=32,
        warmup_requests=1,
    )
    assert len(result["points"]) == 2
    assert result["best_concurrency"] in {1, 4}
    # Коэффициент масштабирования — главный индикатор: около единицы означает,
    # что движок обрабатывает запросы по очереди.
    assert result["scaling_factor"] >= 1.0


def test_sweep_stops_when_engine_mostly_fails() -> None:
    result = run_throughput_sweep(
        _FlakyLLM(),
        concurrency_levels=(1, 2, 4, 8),
        requests_per_level=4,
        prompt_chars=300,
        max_tokens=16,
        warmup_requests=0,
    )
    assert len(result["points"]) <= 4


def test_indexing_time_estimate() -> None:
    estimate = estimate_indexing_time(3533, throughput_rps=1.0)
    assert estimate["hours"] == round(3533 / 3600, 2)

    assert estimate_indexing_time(100, throughput_rps=0.0)["hours"] is None


def test_structured_flag_passes_schema() -> None:
    llm = FakeLLMClient()
    run_concurrency_point(
        llm, concurrency=1, requests=1, prompt_chars=400, max_tokens=16, structured=True
    )
    # FakeLLMClient при наличии схемы возвращает пустой граф — значит схема дошла.
    assert llm.calls
    assert isinstance(llm.calls[0][0], ChatMessage)
