"""Проверка эталонного набора без участия модели.

Набор сгенерирован моделью, и часть его изъянов измерима арифметикой —
дешевле и надёжнее, чем спрашивать модель о её же работе. Здесь собраны
именно такие проверки: каждая либо срабатывает, либо нет, и результат
воспроизводится на любой машине без GPU.

Что проверяется и почему именно это.

``numbered_reference``  вопрос ссылается на номер формулы, рисунка, раздела:
                        «согласно формуле (10.52)». Ответить на такой вопрос
                        поиском нельзя — номер не назван ни в одном фрагменте
                        отдельно от самой формулы, а человек его не задаст.
``textual_anchor``      вопрос привязан к тексту словами «в приведённом
                        фрагменте»: осмыслен только рядом с источником.
``structural_chunk``    эталонный фрагмент — оглавление или страница
                        упражнений. Содержания в нём нет, ответить по нему
                        нельзя, и вопрос с таким фрагментом в паре на деле
                        одношаговый.
``missing_chunk``       эталонного фрагмента нет в корпусе: вопрос
                        засчитывается промахом при любом поиске.
``thin_answer``         эталонный ответ пуст или короче трёх слов: сравнивать
                        с ним нечего.
``near_duplicate``      вопрос почти повторяет другой. На дубликатах метрика
                        считает один и тот же случай дважды и завышает вес
                        темы, которой повезло попасть в набор трижды.

Чего здесь **нет** намеренно: проверки «нужны ли для ответа оба фрагмента».
Она требует суждения о смысле, арифметикой не берётся — для неё есть
абляционная проверка в :mod:`rag_textbook.evaluation.ablation`, и работает
она на сервере.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_textbook.evaluation.goldset import (
    classify_chunk,
    looks_leaky,
    references_numbering,
)
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, GoldQuestion
from rag_textbook.utils.text import content_terms

logger = get_logger("evaluation.audit")

DUPLICATE_THRESHOLD = 0.8


@dataclass
class QuestionAudit:
    """Изъяны одного вопроса. Пустой список изъянов — вопрос годен."""

    question_id: str
    question_type: str
    defects: list[str] = field(default_factory=list)
    chunk_kinds: list[str] = field(default_factory=list)
    duplicate_of: str | None = None

    @property
    def usable(self) -> bool:
        return not self.defects

    def as_dict(self) -> dict[str, Any]:
        return {
            "question_id": self.question_id,
            "question_type": self.question_type,
            "defects": list(self.defects),
            "chunk_kinds": list(self.chunk_kinds),
            "duplicate_of": self.duplicate_of,
        }


def _question_key(question: GoldQuestion) -> frozenset[str]:
    return frozenset(content_terms(question.question, limit=40))


def find_near_duplicates(
    questions: Sequence[GoldQuestion], threshold: float = DUPLICATE_THRESHOLD
) -> dict[str, str]:
    """Возвращает отображение «повтор -> первый вопрос той же формулировки».

    Первым считается тот, что раньше в наборе: так при дозаписи новые вопросы
    отбрасываются, а прежние сохраняются, и прогоны остаются сравнимыми.
    """
    keys = [(item.id, _question_key(item)) for item in questions]
    duplicates: dict[str, str] = {}
    for index, (identifier, key) in enumerate(keys):
        if not key or identifier in duplicates:
            continue
        for other_id, other_key in keys[index + 1 :]:
            if not other_key or other_id in duplicates:
                continue
            union = key | other_key
            if len(key & other_key) / len(union) >= threshold:
                duplicates[other_id] = identifier
    return duplicates


def audit_questions(
    questions: Sequence[GoldQuestion],
    chunks: dict[str, Chunk] | None = None,
    *,
    threshold: float = DUPLICATE_THRESHOLD,
) -> list[QuestionAudit]:
    """Проверяет весь набор. Без ``chunks`` проверки по фрагментам пропускаются."""
    duplicates = find_near_duplicates(questions, threshold)
    results: list[QuestionAudit] = []
    for question in questions:
        audit = QuestionAudit(question_id=question.id, question_type=question.question_type)
        if references_numbering(question.question):
            audit.defects.append("numbered_reference")
        if looks_leaky(question.question):
            audit.defects.append("textual_anchor")
        if len((question.answer or "").split()) < 3:
            audit.defects.append("thin_answer")
        if question.id in duplicates:
            audit.defects.append("near_duplicate")
            audit.duplicate_of = duplicates[question.id]
        if chunks is not None:
            for chunk_id in question.gold_chunk_ids:
                chunk = chunks.get(chunk_id)
                if chunk is None:
                    audit.chunk_kinds.append("нет в корпусе")
                    if "missing_chunk" not in audit.defects:
                        audit.defects.append("missing_chunk")
                    continue
                kind = classify_chunk(chunk)
                audit.chunk_kinds.append(kind)
                if kind != "содержательный" and "structural_chunk" not in audit.defects:
                    audit.defects.append("structural_chunk")
        results.append(audit)
    return results


def summarize_audit(audits: Iterable[QuestionAudit]) -> dict[str, Any]:
    """Сводка с разбивкой по типам вопросов.

    Разбивка обязательна: изъяны распределены неравномерно, и среднее по
    набору уже однажды скрыло, что негодные фрагменты сосредоточены
    в связывающих вопросах.
    """
    items = list(audits)
    if not items:
        return {"всего": 0}
    defects: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = {}
    totals: Counter[str] = Counter()
    for item in items:
        totals[item.question_type] += 1
        bucket = by_type.setdefault(item.question_type, Counter())
        if item.usable:
            bucket["годен"] += 1
        for defect in item.defects:
            defects[defect] += 1
            bucket[defect] += 1
    usable = sum(1 for item in items if item.usable)
    return {
        "всего": len(items),
        "годных": usable,
        "доля годных": round(usable / len(items), 4),
        "изъяны": dict(defects.most_common()),
        "по типам": {
            name: {"вопросов": totals[name], **dict(counter.most_common())}
            for name, counter in sorted(by_type.items())
        },
    }
