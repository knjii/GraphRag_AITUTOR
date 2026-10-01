"""Журнал ответов модели: повтор сборки не обращается к модели за готовым."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from rag_textbook.clients.llm import ChatMessage
from rag_textbook.clients.llm_journal import JournaledLLM


class _Counting:
    def __init__(self) -> None:
        self.calls = 0
        self.lock = threading.Lock()

    def chat(
        self, messages, *, purpose="chat", json_schema=None, max_tokens=None, temperature=None
    ):  # noqa: ANN001, ANN201
        with self.lock:
            self.calls += 1
        return f"ответ на {messages[-1].content} при t={temperature}"


def _ask(llm, text: str, temperature: float = 0.3) -> str:  # noqa: ANN001
    return llm.chat(
        [ChatMessage(role="user", content=text)], purpose="utility", temperature=temperature
    )


def test_second_run_reads_answers_from_journal(tmp_path) -> None:
    path = tmp_path / "j.jsonl"
    first = JournaledLLM(_Counting(), path, model="m")
    answer = _ask(first, "вопрос")
    first.close()

    inner = _Counting()
    second = JournaledLLM(inner, path, model="m")
    assert _ask(second, "вопрос") == answer
    assert inner.calls == 0 and second.hits == 1


def test_key_depends_on_params_and_model(tmp_path) -> None:
    path = tmp_path / "j.jsonl"
    llm = JournaledLLM(_Counting(), path, model="m")
    _ask(llm, "вопрос")
    llm.close()

    inner = _Counting()
    again = JournaledLLM(inner, path, model="m")
    _ask(again, "вопрос", temperature=0.0)
    again.close()
    other = JournaledLLM(inner, path, model="другая")
    _ask(other, "вопрос")
    assert inner.calls == 2


def test_torn_last_line_is_skipped(tmp_path) -> None:
    path = tmp_path / "j.jsonl"
    llm = JournaledLLM(_Counting(), path, model="m")
    _ask(llm, "первый")
    llm.close()
    with path.open("a", encoding="utf-8") as handle:
        handle.write('{"key": "обрыв')
    inner = _Counting()
    resumed = JournaledLLM(inner, path, model="m")
    _ask(resumed, "первый")
    assert inner.calls == 0


def test_parallel_writes_keep_every_answer(tmp_path) -> None:
    path = tmp_path / "j.jsonl"
    llm = JournaledLLM(_Counting(), path, model="m")
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda i: _ask(llm, f"q{i}"), range(50)))
    llm.close()
    inner = _Counting()
    resumed = JournaledLLM(inner, path, model="m")
    for i in range(50):
        _ask(resumed, f"q{i}")
    assert inner.calls == 0
