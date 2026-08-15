"""Роутер: решает, нужен ли графовый канал для этого вопроса.

Прежде графовый канал включался на каждый запрос. Это плохо по двум причинам:
он стоит времени (обращения к Neo4j плюс эмбеддинги), и на простых фактических
вопросах он в лучшем случае бесполезен, а в худшем разбавляет выдачу.

Эвристика дешёвая и объяснимая; при необходимости её можно заменить на LLM-роутер,
но начинать стоит с прозрачного правила, чтобы понимать, что именно происходит.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.config import RetrievalSettings
from rag_textbook.logging_setup import get_logger

logger = get_logger("retrieval.router")

# Формулировки, требующие связи между разными местами текста.
_MULTIHOP_MARKERS: tuple[str, ...] = (
    "как связан",
    "как связана",
    "как связаны",
    "чем отличается",
    "чем отличаются",
    "в чем разница",
    "в чём разница",
    "сравни",
    "почему",
    "как влияет",
    "как используется",
    "связь между",
    "зависит ли",
    "следует ли из",
    "выводится ли",
    "какая связь",
    "how are",
    "difference between",
    "relationship between",
    "compare",
)

# Вопросы-определения обычно закрываются одним фрагментом.
_SINGLE_HOP_MARKERS: tuple[str, ...] = (
    "что такое",
    "дай определение",
    "определение",
    "как называется",
    "what is",
    "define",
)

_CONJUNCTION_RE = re.compile(r"\bи\b|\bа также\b|\bпри этом\b", re.IGNORECASE)


@dataclass
class RouteDecision:
    use_graph: bool
    reason: str
    confidence: float = 0.0


class QueryRouter:
    def __init__(self, settings: RetrievalSettings, llm: LLMClient | None = None) -> None:
        self.settings = settings
        self.llm = llm

    def _heuristic(self, question: str) -> RouteDecision:
        lowered = question.lower().replace("ё", "е")

        for marker in _MULTIHOP_MARKERS:
            if marker in lowered:
                return RouteDecision(True, f"маркер многохоповости: «{marker}»", 0.8)

        for marker in _SINGLE_HOP_MARKERS:
            if lowered.startswith(marker) or f" {marker}" in lowered[:40]:
                return RouteDecision(False, f"вопрос-определение: «{marker}»", 0.7)

        # Длинный вопрос с сочинительной связью часто требует свести два факта.
        words = len(lowered.split())
        if words >= 12 and _CONJUNCTION_RE.search(lowered):
            return RouteDecision(True, "длинный составной вопрос", 0.5)

        return RouteDecision(False, "простой фактический вопрос", 0.4)

    def _llm_route(self, question: str) -> RouteDecision:
        if self.llm is None:
            return self._heuristic(question)
        prompt = (
            "Определи, требует ли вопрос связывания информации из разных разделов учебника.\n"
            "Ответь одним словом: ДА или НЕТ.\n\n"
            f"Вопрос: {question}"
        )
        try:
            answer = self.llm.chat(
                [ChatMessage(role="user", content=prompt)],
                purpose="chat",
                max_tokens=8,
                temperature=0.0,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("LLM-роутер недоступен, откатываюсь к эвристике: %s", exc)
            return self._heuristic(question)
        normalized = answer.strip().lower()
        use_graph = normalized.startswith("да") or normalized.startswith("yes")
        return RouteDecision(use_graph, f"решение LLM: «{normalized[:20]}»", 0.6)

    def route(self, question: str) -> RouteDecision:
        mode = self.settings.router_mode
        if not self.settings.router_enabled or mode == "always":
            return RouteDecision(True, "роутер выключен: граф используется всегда", 1.0)
        if mode == "never":
            return RouteDecision(False, "граф отключён настройкой", 1.0)
        if mode == "llm":
            return self._llm_route(question)
        return self._heuristic(question)
