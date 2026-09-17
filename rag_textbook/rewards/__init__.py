"""Проверяемые награды для RL-дообучения генератора (R6, docs/RESEARCH-2026-09.md)."""

from rag_textbook.rewards.composite import (
    RewardBreakdown,
    RewardConfig,
    compute_reward,
    format_reward,
    random_reward,
)
from rag_textbook.rewards.formula import (
    FormulaScore,
    canonical_tokens,
    extract_math,
    score_formulas,
)

__all__ = [
    "FormulaScore",
    "RewardBreakdown",
    "RewardConfig",
    "canonical_tokens",
    "compute_reward",
    "extract_math",
    "format_reward",
    "random_reward",
    "score_formulas",
]
