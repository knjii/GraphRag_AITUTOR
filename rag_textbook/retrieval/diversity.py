"""Разнообразие выдачи.

Основание. Все 34 промаха на многошаговых вопросах устроены одинаково: система
нашла один эталонный фрагмент из двух. Это не случайность выборки, а признак
механизма: фрагменты, похожие на первый, получают близкие баллы и занимают
места в выдаче, а второй нужный фрагмент по построению *другой* по теме —
пара отбиралась как связанная типизированной связью и почти не пересекающаяся
по словам, — и оказывается ниже отсечки.

Отбор по одной лишь релевантности такое поведение и предписывает: он
не различает «ещё один фрагмент про то же» и «фрагмент про вторую половину
вопроса». Здесь добавляется различение.

Два режима, потому что они по-разному рискуют:

``mmr``     классический баланс релевантности и новизны по всей выдаче. Даёт
            больше свободы, но может выбить из выдачи верные однотипные
            фрагменты — а на одношаговых вопросах именно они и нужны.

``reserve`` трогает только хвост выдачи: первые места остаются за порядком
            реранкера, а последние ``reserve_slots`` отдаются фрагментам,
            наименее похожим на первый. Осторожнее: ухудшить одношаговые
            вопросы он почти не может, потому что не трогает их голову.

Ничего из этого не включено по умолчанию. Порог хабов 40 уже был подтверждён
офлайн и оказался вредным на сервере; порядок теперь обратный — сначала замер
на сервере, потом значение по умолчанию.
"""

from __future__ import annotations

from collections.abc import Sequence

from rag_textbook.config import Settings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import ScoredChunk
from rag_textbook.utils.text import content_terms

logger = get_logger("retrieval.diversity")


def _terms(item: ScoredChunk) -> set[str]:
    return set(content_terms(item.chunk.text))


def _similarity(left: set[str], right: set[str]) -> float:
    """Мера Жаккара по значимым словам.

    Векторы фрагментов дали бы более тонкое сравнение, но требовали бы
    обращения к хранилищу на каждом запросе. Дедупликация в конвейере уже
    сравнивает фрагменты по множеству лемм, и здесь используется тот же
    приём — чтобы две соседние стадии не расходились в понимании похожести.
    """
    if not left or not right:
        return 0.0
    intersection = len(left & right)
    if not intersection:
        return 0.0
    return intersection / len(left | right)


def _mmr(items: Sequence[ScoredChunk], top_k: int, lambda_: float) -> list[ScoredChunk]:
    if not items:
        return []
    terms = {index: _terms(item) for index, item in enumerate(items)}
    # Место в списке после реранкера — это и есть релевантность: сами баллы
    # у каналов и реранкера в разных шкалах, а ранг сопоставим всегда.
    relevance = {index: 1.0 - index / max(len(items), 1) for index in range(len(items))}

    chosen: list[int] = [0]
    remaining = set(range(1, len(items)))
    while remaining and len(chosen) < top_k:
        best_index, best_value = None, float("-inf")
        for index in remaining:
            penalty = max(_similarity(terms[index], terms[picked]) for picked in chosen)
            value = lambda_ * relevance[index] - (1.0 - lambda_) * penalty
            if value > best_value:
                best_index, best_value = index, value
        if best_index is None:
            break
        chosen.append(best_index)
        remaining.discard(best_index)

    ordered = [items[index] for index in chosen]
    # Остаток сохраняет прежний порядок: отсечка может быть шире top_k.
    ordered.extend(items[index] for index in range(len(items)) if index not in set(chosen))
    return ordered


def _reserve(items: Sequence[ScoredChunk], top_k: int, slots: int) -> list[ScoredChunk]:
    """Отдаёт последние места фрагментам, непохожим на первый."""
    if len(items) <= top_k or slots <= 0:
        return list(items)

    slots = min(slots, top_k - 1)
    if slots <= 0:
        return list(items)

    head = list(items[: top_k - slots])
    tail = list(items[top_k - slots :])
    anchor = _terms(items[0])

    # Наименее похожие на первый фрагмент — вперёд, остальное следом.
    tail.sort(key=lambda item: _similarity(_terms(item), anchor))
    return head + tail


def apply_diversity(
    items: Sequence[ScoredChunk], settings: Settings, *, top_k: int
) -> list[ScoredChunk]:
    """Переупорядочивает выдачу согласно выбранному режиму разнообразия."""
    mode = settings.retrieval.diversity_mode
    if mode == "off" or not items:
        return list(items)

    if mode == "mmr":
        result = _mmr(items, top_k, settings.retrieval.diversity_lambda)
    elif mode == "reserve":
        result = _reserve(items, top_k, settings.retrieval.diversity_reserve_slots)
    else:  # pragma: no cover - значение ограничено типом настройки
        return list(items)

    logger.debug("Разнообразие: режим %s, кандидатов %s", mode, len(items))
    return result
