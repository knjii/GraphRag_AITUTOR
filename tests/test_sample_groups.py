"""Группы генераций: сырой текст, обрыв, сигнал для GRPO, слепой лист."""

from __future__ import annotations

import asyncio
import importlib.util
import json
from pathlib import Path

import httpx

from rag_textbook.clients.llm import OpenAICompatibleLLMClient
from rag_textbook.rl.env import Example

_spec = importlib.util.spec_from_file_location(
    "sample_groups", Path(__file__).resolve().parents[1] / "scripts" / "sample_groups.py"
)
assert _spec is not None and _spec.loader is not None
sample_groups = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sample_groups)

GOLD = "Скалярное произведение задаётся формулой $$\\langle x, y \\rangle = x^{T} A y$$ для векторов."
EXAMPLE = Example(
    question_id="q1", question_type="formula_table", question="Как задаётся скалярное произведение?",
    messages=[{"role": "system", "content": "Контекст"}, {"role": "user", "content": "?"}],
    context=GOLD, reference=GOLD, gold_in_context=True, doc_ids=["d"],
)


def _client(settings, replies: list[dict]) -> OpenAICompatibleLLMClient:
    queue = list(replies)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [queue.pop(0)]})

    client = OpenAICompatibleLLMClient(settings.llm.model_copy(update={"max_retries": 0}))
    client._aclient = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return client


def test_raw_completion_keeps_reasoning_and_finish_reason(settings):
    client = _client(settings, [{"message": {"content": "<think>x</think>Ответ"}, "finish_reason": "length"}])
    result = asyncio.run(sample_groups.sample_one(client, EXAMPLE, temperature=0.8, max_tokens=16))
    assert result == {"answer": "<think>x</think>Ответ", "finish_reason": "length"}


def test_truncated_generation_hits_the_gate():
    good = {"answer": "Формула: $$\\langle x, y \\rangle = x^{T} A y$$ для векторов.", "finish_reason": "stop"}
    assert sample_groups.score(EXAMPLE, good)["reward"] > 0
    assert sample_groups.score(EXAMPLE, {**good, "finish_reason": "length"})["gate"]


def test_summary_counts_groups_without_signal():
    rows = [
        {"question_id": "a", "reward": -1.0, "gate": "x", "finish_reason": "length"},
        {"question_id": "a", "reward": -1.0, "gate": "x", "finish_reason": "length"},
        {"question_id": "b", "reward": 0.2, "gate": "", "finish_reason": "stop"},
        {"question_id": "b", "reward": 0.9, "gate": "", "finish_reason": "stop"},
    ]
    summary = sample_groups.summarize(rows)
    assert summary["доля групп без разброса"] == 0.5
    assert summary["оборвано пределом"] == 2


def test_sheet_hides_reward(tmp_path):
    rows = [
        {"question_id": "q1", "sample": i, "answer": f"ответ {i}", "finish_reason": "stop",
         "reward": i / 10, "gate": "", "parts": {}}
        for i in range(3)
    ]
    sample_groups.write_sheet(rows, {"q1": EXAMPLE}, tmp_path / "g", questions=5, seed=1)
    sheet = (tmp_path / "g-sheet.md").read_text(encoding="utf-8")
    key = json.loads((tmp_path / "g-key.json").read_text(encoding="utf-8"))
    assert "reward" not in sheet and "0.2" not in sheet
    assert sorted(item["sample"] for item in key) == [0, 1, 2]


_agreement_spec = importlib.util.spec_from_file_location(
    "reward_agreement", Path(__file__).resolve().parents[1] / "scripts" / "reward_agreement.py"
)
assert _agreement_spec is not None and _agreement_spec.loader is not None
reward_agreement = importlib.util.module_from_spec(_agreement_spec)
_agreement_spec.loader.exec_module(reward_agreement)


def test_concordance_counts_only_pairs_with_different_grades():
    groups = [[(3, 1.0), (1, 0.0), (3, 0.5)], [(2, 0.3), (2, 0.9)]]
    value, pairs = reward_agreement.concordance(groups)
    assert pairs == 2
    assert value == 1.0


def test_reward_ties_count_half():
    value, pairs = reward_agreement.concordance([[(3, 0.4), (0, 0.4)]])
    assert (value, pairs) == (0.5, 1)
