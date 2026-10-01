"""Награда Context-Picker (arXiv 2512.14465, формула стадии II), этап 2.

    R = Cov(S, S_g) − γ·(|S| − |S_g|) / L,   если ответ разобран и |S| ≤ L;
    R = 0,                                   если разобран и |S| > L;
    R = −1,                                  если не разобран (нет строки выбора,
                                             обрыв по пределу токенов);
    L = |S_g| + red.

Стадия I — та же формула с широким запасом ``red`` и, по их описанию,
упором на полноту («relaxed redundancy tolerance»): здесь γ = 0. Стадия II
сужает запас и включает штраф за лишнее.

Числа γ, red₁, red₂ в статье не названы — значения по умолчанию наши
и подлежат подбору.

**Отступление от статьи:** штраф берётся от превышения, max(0, |S|−|S_g|).
Буквально формула даёт недобору прибавку γ·(|S_g|−|S|)/L, и проверка всухую
(2026-09-29) показала эксплойт: на стадии II пустой выбор получал +0.25 —
больше, чем всё окно (0). Недобор и так теряет 1/|S_g| полноты за каждый
пропущенный фрагмент.

Эталон ``S_g`` — опорные фрагменты набора (у MuSiQue — is_supporting,
у нашего — gold_chunk_ids). Минимальность эталона статья добывает
исключением по одному (LOO) через судью; здесь вместо этого разметка
набора: у MuSiQue опорные абзацы и есть минимальное множество по построению.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass

from rag_textbook.retrieval.set_selection import parse_selection


@dataclass(frozen=True)
class PickerRewardConfig:
    red: int = 2
    gamma: float = 0.5


STAGE_ONE = PickerRewardConfig(red=5, gamma=0.0)
STAGE_TWO = PickerRewardConfig(red=2, gamma=0.5)


@dataclass(frozen=True)
class PickerScore:
    total: float
    valid: bool
    coverage: float
    size: int
    gold: int

    def as_dict(self) -> dict[str, float | int | bool]:
        return {
            "total": round(self.total, 4), "valid": self.valid,
            "coverage": round(self.coverage, 4), "size": self.size, "gold": self.gold,
        }


def picker_reward(
    text: str,
    gold: Collection[int],
    window: int,
    *,
    config: PickerRewardConfig = STAGE_TWO,
    truncated: bool = False,
) -> PickerScore:
    """``gold`` — номера эталонных фрагментов в окне, с нуля."""
    gold_set = set(gold)
    picked = None if truncated else parse_selection(text, window)
    if picked is None or not gold_set:
        # Пустая строка выбора — законный ответ «ничего», он получает
        # нулевое покрытие; −1 только за отсутствие ответа по формату.
        return PickerScore(-1.0, False, 0.0, 0, len(gold_set))
    chosen = set(picked)
    coverage = len(chosen & gold_set) / len(gold_set)
    limit = len(gold_set) + config.red
    if len(chosen) > limit:
        return PickerScore(0.0, True, coverage, len(chosen), len(gold_set))
    total = coverage - config.gamma * max(0, len(chosen) - len(gold_set)) / limit
    return PickerScore(total, True, coverage, len(chosen), len(gold_set))
