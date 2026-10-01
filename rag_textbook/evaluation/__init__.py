"""Оценка качества: метрики поиска, сборка эталонного набора, A/B-прогоны."""

from rag_textbook.evaluation.goldset import GoldsetBuilder, load_goldset, save_goldset
from rag_textbook.evaluation.metrics import RetrievalMetrics, evaluate_retrieval

__all__ = [
    "GoldsetBuilder",
    "RetrievalMetrics",
    "evaluate_retrieval",
    "load_goldset",
    "save_goldset",
]
