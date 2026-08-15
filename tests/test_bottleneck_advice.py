"""Совет по узкому месту должен зависеть от стадии.

Разбор прогона по всей книге выдал для стадии разбора совет «поднимите
LLM_MAX_CONCURRENCY и сравните с bench». MinerU к языковой модели не
обращается вовсе, и следование такому совету не дало бы ничего — а вводящий
в заблуждение вывод инструмента анализа хуже отсутствующего.
"""

from __future__ import annotations

import pytest

from rag_textbook.observability.analyze import StageVerdict, _classify


def _idle_stage(stage: str) -> StageVerdict:
    """Стадия, где карта простаивает: ресурсы свободны, а время идёт."""
    return StageVerdict(
        stage=stage,
        samples=100,
        duration_seconds=300.0,
        gpu_util_mean=21.0,
        gpu_idle_share_pct=51.0,
        cpu_util_mean=7.0,
    )


@pytest.mark.parametrize("stage", ["parse", "embed", "chunk"])
def test_non_llm_stages_do_not_get_inference_advice(stage: str) -> None:
    bottleneck, _, recommendation = _classify(_idle_stage(stage))
    assert bottleneck == "waiting"
    assert "LLM_MAX_CONCURRENCY" not in recommendation
    assert "bench" not in recommendation


@pytest.mark.parametrize("stage", ["enrich", "graph"])
def test_llm_stages_get_inference_advice(stage: str) -> None:
    bottleneck, _, recommendation = _classify(_idle_stage(stage))
    assert bottleneck == "waiting"
    assert "LLM_MAX_CONCURRENCY" in recommendation


def test_parse_advice_names_the_real_cause() -> None:
    """Совет должен называть причину, а не общие слова."""
    _, _, recommendation = _classify(_idle_stage("parse"))
    assert "MinerU" in recommendation


def test_unknown_stage_gets_neutral_advice() -> None:
    """Незнакомая стадия не должна получать чужой диагноз."""
    _, _, recommendation = _classify(_idle_stage("новая_стадия"))
    assert "LLM_MAX_CONCURRENCY" not in recommendation
    assert recommendation.strip()
