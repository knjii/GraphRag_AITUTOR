"""Вердикты ручной проверки эталонного набора.

Зачем отдельный файл, а не отметка прямо в наборе. Проверка идёт на ноутбуке
по локальной копии набора, а набор растёт на сервере: 163 вопроса превратились
в 388, и превратятся ещё раз. Если хранить результат проверки внутри набора,
он теряется при каждом расширении — либо приходится сливать два файла, каждый
из которых считает себя главным.

Вердикты хранятся отдельно и применяются к любому набору по идентификатору
вопроса. Тогда порядок операций перестаёт иметь значение: расширить набор
и потом применить вердикты — то же самое, что применить и потом расширить.

Отдельно фиксируется вердикт ``single_hop_enough``: он означает, что вопрос
размечен как связывающий, но отвечается одним фрагментом. Это не брак разметки
в обычном смысле — вопрос осмысленный, — но для измерения вклада графа такой
вопрос бесполезен, а в среднем по типу он занижает результат.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from rag_textbook.logging_setup import get_logger
from rag_textbook.models import GoldQuestion

logger = get_logger("evaluation.verdicts")

Verdict = Literal["ok", "single_hop_enough", "unanswerable", "ambiguous", "leaky"]

# Что означает каждый вердикт:
#   ok                 — вопрос корректен, эталонные фрагменты отвечают на него;
#   single_hop_enough  — размечен связывающим, но хватает одного фрагмента;
#   unanswerable       — эталонные фрагменты на вопрос не отвечают;
#   ambiguous          — вопрос допускает несколько прочтений;
#   leaky              — в вопросе есть отсылка к тексту («в данном фрагменте»).
USABLE: frozenset[str] = frozenset({"ok", "single_hop_enough"})


@dataclass
class QuestionVerdict:
    question_id: str
    verdict: Verdict
    note: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {"question_id": self.question_id, "verdict": self.verdict, "note": self.note}


@dataclass
class VerdictSet:
    verdicts: dict[str, QuestionVerdict] = field(default_factory=dict)

    def add(self, verdict: QuestionVerdict) -> None:
        self.verdicts[verdict.question_id] = verdict

    def __len__(self) -> int:
        return len(self.verdicts)

    @classmethod
    def load(cls, path: Path) -> VerdictSet:
        if not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        items = raw.get("verdicts", raw) if isinstance(raw, dict) else raw
        result = cls()
        for item in items:
            result.add(
                QuestionVerdict(
                    question_id=str(item["question_id"]),
                    verdict=item.get("verdict", "ok"),
                    note=item.get("note", ""),
                )
            )
        return result

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "verdicts": [item.as_dict() for item in self.verdicts.values()],
        }
        Path(path).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def apply_verdicts(
    questions: Sequence[GoldQuestion], verdicts: VerdictSet
) -> tuple[list[GoldQuestion], dict[str, int]]:
    """Проставляет ``verified`` и записывает вердикт в ``notes``.

    Проверенным считается вопрос с вердиктом из ``USABLE``: он годен для замера.
    Негодные помечаются тоже — с вердиктом в примечании, — но ``verified``
    у них остаётся ложным, чтобы они не попали в выборку, которую мы называем
    проверенной.

    Вопросы, для которых вердикта нет, возвращаются нетронутыми: набор больше
    проверенной выборки, и молча помечать непроверенное нельзя.
    """
    counts: dict[str, int] = {}
    result: list[GoldQuestion] = []
    for question in questions:
        verdict = verdicts.verdicts.get(question.id)
        if verdict is None:
            result.append(question)
            continue
        counts[verdict.verdict] = counts.get(verdict.verdict, 0) + 1
        updated = question.model_copy(
            update={
                "verified": verdict.verdict in USABLE,
                "notes": _merge_note(question.notes, verdict),
            }
        )
        result.append(updated)

    logger.info("Применено вердиктов: %s из %s вопросов", sum(counts.values()), len(questions))
    return result, counts


def _merge_note(existing: str, verdict: QuestionVerdict) -> str:
    mark = f"проверка: {verdict.verdict}"
    if verdict.note:
        mark = f"{mark} — {verdict.note}"
    if not existing:
        return mark
    # Прежнее примечание сохраняется: оно могло прийти от генератора.
    return f"{existing}; {mark}"


def summarize(
    questions: Sequence[GoldQuestion], verdicts: VerdictSet
) -> dict[str, dict[str, int]]:
    """Считает вердикты в разбивке по типам вопросов.

    Главное число, ради которого всё затевалось, — доля ``single_hop_enough``
    среди связывающих: она показывает, какую часть разрыва по многошаговым
    вопросам объясняет разметка, а не система.
    """
    by_type: dict[str, dict[str, int]] = {}
    for question in questions:
        verdict = verdicts.verdicts.get(question.id)
        if verdict is None:
            continue
        bucket = by_type.setdefault(question.question_type, {})
        bucket[verdict.verdict] = bucket.get(verdict.verdict, 0) + 1
    return by_type
