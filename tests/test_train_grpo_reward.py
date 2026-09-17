"""Договор награды с TRL: как тренер её вызывает, так она и должна работать.

Проверяется на ноутбуке, потому что альтернатива — узнать о несовпадении
подписи на первом шаге оплаченной карты. TRL зовёт функцию награды
ключевыми аргументами: `prompts`, `completions`, `completion_ids`, все
столбцы набора и служебные поля, состав которых меняется от версии
к версии.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("train_grpo", ROOT / "scripts" / "train_grpo.py")
train_grpo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(train_grpo)

CONTEXT = (
    "Контекст: $$\\langle x, y \\rangle = x^{T} A y$$ — скалярное "
    "произведение, заданное матрицей $A$."
)
GOOD = (
    "Скалярное произведение задаётся как $$\\langle x, y \\rangle = x^{T} A y$$, "
    "где $A$ — симметричная положительно определённая матрица [1]."
)


def call(reward, texts, **extra):
    """Вызов в том виде, в каком его делает GRPOTrainer."""
    size = len(texts)
    payload = {
        "prompts": ["вопрос"] * size,
        "completions": [[{"role": "assistant", "content": t}] for t in texts],
        "context": [CONTEXT] * size,
        "reference": [CONTEXT] * size,
        "gold_in_context": [True] * size,
        "question_id": [f"q{i}" for i in range(size)],
        "question": ["Как задаётся скалярное произведение?"] * size,
        # Служебные поля разных версий TRL: функция обязана их пережить.
        "trainer_state": object(),
        "step": 3,
    }
    payload.update(extra)
    return reward(**payload)


def build(tmp_path: Path, kind: str = "main", *, sample_every: int = 1):
    return train_grpo.build_reward(
        kind,
        seed=1,
        max_completion_tokens=768,
        samples_path=tmp_path / "samples.jsonl",
        sample_every=sample_every,
    )


def test_unknown_keyword_arguments_do_not_break_the_reward(tmp_path: Path):
    values = call(build(tmp_path), [GOOD, ""])
    assert len(values) == 2
    assert values[0] > values[1]


def test_truncated_completion_is_penalised(tmp_path: Path):
    reward = build(tmp_path)
    whole = call(reward, [GOOD], completion_ids=[[0] * 100])
    cut = call(reward, [GOOD], completion_ids=[[0] * 768])
    assert cut[0] < whole[0]


def test_missing_completion_ids_is_allowed(tmp_path: Path):
    # В TRL 0.24 столбца нет; обрыв ловится воротами длины, не падением.
    assert len(call(build(tmp_path), [GOOD])) == 1


def test_samples_are_written_with_the_reward_breakdown(tmp_path: Path):
    call(build(tmp_path), [GOOD, ""])
    lines = (tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    record = json.loads(lines[0])
    assert record["call"] == 1 and "main" in record and "answer" in record
    # Основная награда пишется и при контрольных прогонах — по ней видно,
    # растёт ли формульная часть при случайном сигнале.
    assert "formula" in json.dumps(record["main"], ensure_ascii=False)


def test_samples_are_written_once_per_window(tmp_path: Path):
    reward = build(tmp_path, sample_every=10)
    for _ in range(10):
        call(reward, [GOOD])
    assert len((tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()) == 1


def test_control_rewards_ignore_answer_quality(tmp_path: Path):
    reward = build(tmp_path, "random")
    # Случайная награда — бросок монеты: одинаковые ответы получают одно
    # число, разные — разные хотя бы иногда, иначе преимущество внутри
    # группы всегда нулевое и контроль ничего не проверяет.
    seen = set()
    for _ in range(12):
        seen.update(call(reward, [GOOD, ""]))
    assert seen == {0.0, 1.0}
    assert call(reward, [GOOD])[0] == call(reward, [GOOD])[0]
    format_values = call(build(tmp_path, "format"), [GOOD, ""])
    assert format_values[0] > format_values[1]


def test_control_reward_still_records_the_main_one(tmp_path: Path):
    call(build(tmp_path, "random"), [GOOD])
    record = json.loads((tmp_path / "samples.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert record["main"]["total"] != record["reward"]


def test_unknown_reward_kind_is_refused(tmp_path: Path):
    with pytest.raises(ValueError):
        call(build(tmp_path, "какая-то"), [GOOD])


def test_column_length_mismatch_is_loud(tmp_path: Path):
    """Молчаливое усечение сдвинуло бы награду на чужой контекст."""
    with pytest.raises(ValueError):
        call(build(tmp_path), [GOOD, ""], context=[CONTEXT])
