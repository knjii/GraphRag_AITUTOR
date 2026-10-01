"""Процессор мультимодальной модели заменяется его текстовым токенизатором (день 5, P1)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "train_grpo_tok", Path(__file__).resolve().parents[1] / "scripts" / "train_grpo.py"
)
tg = importlib.util.module_from_spec(_SPEC)
sys.modules["train_grpo_tok"] = tg
_SPEC.loader.exec_module(tg)


def test_processor_gives_inner_tokenizer():
    inner = SimpleNamespace(eos_token_id=1, pad_token_id=2,
                            convert_tokens_to_ids=lambda t: {"<|im_end|>": 3}.get(t, 0), unk_token_id=0)
    processor = SimpleNamespace(tokenizer=inner)
    assert tg.text_tokenizer(processor) is inner
    assert tg.stop_token_ids(tg.text_tokenizer(processor)) == {1, 2, 3}


def test_plain_tokenizer_is_kept():
    plain = SimpleNamespace(eos_token_id=1)
    assert tg.text_tokenizer(plain) is plain


def test_micro_batching_keeps_prompts_per_step():
    assert tg.micro_batching(4, 4, 0) == {"per_device_train_batch_size": 4, "gradient_accumulation_steps": 4}
    assert tg.micro_batching(4, 4, 1) == {"per_device_train_batch_size": 1, "gradient_accumulation_steps": 16,
                                          "generation_batch_size": 4}
    assert tg.micro_batching(2, 2, 1)["generation_batch_size"] == 2


def test_rope_deltas_from_generation_group_do_not_empty_micro_batch():
    torch = pytest.importorskip("torch")

    class Inner(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.rope_deltas = None

        def compute_3d_position_ids(self):
            return None

        def forward(self, x):
            # как transformers 5.2: дельты растягиваются на пакет через B // G
            delta = self.rope_deltas.repeat_interleave(x.shape[0] // self.rope_deltas.shape[0], dim=0)
            return delta.shape[0]

    class Outer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = Inner()

        def forward(self, x):
            return self.model(x)

    model = Outer()
    assert tg.pin_text_rope_deltas(model) == 1
    model.model.rope_deltas = torch.zeros(4, 1)  # записала генерация группы из 4
    assert model(torch.zeros(1, 3)) == 1
    model.model.rope_deltas = torch.ones(4, 1)  # ненулевые (картинки) не трогаем
    assert model(torch.zeros(4, 3)) == 4


def test_split_dev_is_stable_and_disjoint():
    rows = [{"question_id": f"q{i}"} for i in range(50)]
    train, dev = tg.split_dev(rows, 8)
    train2, dev2 = tg.split_dev(list(reversed(rows)), 8)
    assert len(dev) == 8 and len(train) == 42
    assert {r["question_id"] for r in dev} == {r["question_id"] for r in dev2}
    assert not {r["question_id"] for r in dev} & {r["question_id"] for r in train}
    assert tg.split_dev(rows, 0) == (rows, [])


def test_validation_summary_counts_formula_only_where_expected():
    scored = [
        {"reward": 1.0, "gate": "", "latex_expected": 2, "latex_found": 1, "truncated": False, "chars": 100},
        {"reward": 0.0, "gate": "", "latex_expected": 1, "latex_found": 0, "truncated": False, "chars": 200},
        {"reward": -1.0, "gate": "length", "latex_expected": 0, "latex_found": 0, "truncated": True, "chars": 300},
    ]
    summary = tg.summarize_validation(scored)
    assert summary["n"] == 3 and summary["with_latex"] == 2
    assert summary["any_formula"] == 0.5
    assert summary["formula_recall"] == 0.25
    assert summary["gate_fail"] == round(1 / 3, 4) and summary["truncated"] == round(1 / 3, 4)


def test_score_answer_uses_same_formula_metric_as_eval():
    row = {"question_id": "q", "context": "Формула $a^2+b^2=c^2$ верна.", "reference": "Формула $a^2+b^2=c^2$ верна.",
           "gold_in_context": True, "question": "Какая формула?"}
    good = tg.score_answer(row, "По теореме Пифагора $a^2+b^2=c^2$ [1].", truncated=False)
    bad = tg.score_answer(row, "Не знаю формулы.", truncated=False)
    assert good["latex_expected"] == 1 and good["latex_found"] == 1
    assert bad["latex_found"] == 0
