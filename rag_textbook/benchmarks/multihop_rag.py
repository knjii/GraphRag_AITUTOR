"""Переходник к MultiHop-RAG.

Почему именно этот набор. Наш спор идёт о многошаговых вопросах: граф нужен
затем, чтобы находить второй фрагмент, до которого похожесть не достаёт.
MultiHop-RAG устроен ровно вокруг этого — каждый вопрос требует свидетельств
из нескольких документов, и **каждое свидетельство размечено дословной
цитатой**. Значит, по нему считается не только качество ответа, но и полнота
поиска, а именно её мы и меряем весь проект. Числа по нему опубликованы,
поэтому наши можно поставить рядом.

Чего этот набор не покажет. Корпус — новостной и английский, а продукт
про русский учебник математики. Перенос выводов с одного на другой
незаконен: это внешняя точка сравнения, а не замена собственному набору.
Ровно так же, как OmniDocBench измеряет разбор, но ничего не говорит
о поиске.

Устройство набора:

``corpus.json``       документы: заголовок, источник, дата, тело;
``MultiHopRAG.json``  вопросы: текст, ответ, тип, список свидетельств,
                      каждое — с дословной цитатой ``fact``.

Главная работа переходника — перевести цитату в идентификатор нашего чанка.
Цитата дословна, поэтому основной путь — поиск подстроки после нормализации
пробелов. Когда он не срабатывает (чанкер мог разрезать документ ровно
посреди цитаты), берётся чанк с наибольшим пересечением по словам, и только
при уверенном пересечении. Несопоставленные свидетельства не выбрасываются
молча, а считаются: их доля — прямая мера того, насколько результату можно
верить.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from rag_textbook.benchmarks.text_corpus import TextDocument, stable_doc_id
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Chunk, GoldQuestion, content_hash

logger = get_logger("benchmarks.multihop_rag")

# Вопросы без ответа в корпусе. В нашем измерителе им нет места: recall
# по пустому эталону не определён. Считаем их отдельно.
NULL_QUERY = "null_query"

# Доля общих слов, ниже которой сопоставление цитаты с чанком признаётся
# ненадёжным. Порог высокий намеренно: ложное сопоставление портит эталон
# тише, чем пропуск, и потому опаснее.
_MIN_OVERLAP = 0.6

# Исходный тип вопроса хранится в примечании с этой приметой: по нему
# восстанавливается разбивка набора, не совпадающая с нашей.
_TYPE_PREFIX = "MultiHop-RAG, тип набора: "

_SPACES = re.compile(r"\s+")
_WORDS = re.compile(r"\w+", re.UNICODE)


def _normalize(text: str) -> str:
    return _SPACES.sub(" ", (text or "").strip().lower())


def _words(text: str) -> set[str]:
    return set(_WORDS.findall((text or "").lower()))


@dataclass
class MappingReport:
    """Насколько полно свидетельства легли на наши чанки."""

    evidence_total: int = 0
    matched_exact: int = 0
    matched_overlap: int = 0
    unmatched: int = 0
    questions_total: int = 0
    questions_kept: int = 0
    dropped_null: int = 0
    per_type: Counter[str] = field(default_factory=Counter)

    @property
    def coverage(self) -> float:
        if not self.evidence_total:
            return 0.0
        return (self.matched_exact + self.matched_overlap) / self.evidence_total

    def as_dict(self) -> dict[str, Any]:
        return {
            "свидетельств": self.evidence_total,
            "сопоставлено дословно": self.matched_exact,
            "сопоставлено по пересечению": self.matched_overlap,
            "не сопоставлено": self.unmatched,
            "покрытие": round(self.coverage, 4),
            "вопросов в наборе": self.questions_total,
            "вопросов оставлено": self.questions_kept,
            "вопросов без ответа отброшено": self.dropped_null,
            "по типам": dict(self.per_type),
        }


def load_corpus(path: Path) -> list[TextDocument]:
    """Читает ``corpus.json`` набора.

    Идентификатор документа считается от заголовка и источника, а не берётся
    порядковым: набор пересобирается, порядок может измениться, а эталонные
    фрагменты обязаны находиться и после пересборки.
    """
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    documents: list[TextDocument] = []
    for item in raw:
        title = str(item.get("title") or "")
        source = str(item.get("source") or "")
        body = str(item.get("body") or "")
        if not body.strip():
            continue
        documents.append(
            TextDocument(
                doc_id=stable_doc_id("multihop-rag", title, source),
                title=title,
                text=body,
                metadata={
                    "source": source,
                    "published_at": item.get("published_at", ""),
                    "category": item.get("category", ""),
                    "url": item.get("url", ""),
                },
            )
        )
    logger.info("MultiHop-RAG: документов %s", len(documents))
    return documents


def _locate(fact: str, candidates: Sequence[Chunk]) -> tuple[str | None, str]:
    """Находит чанк, содержащий цитату. Возвращает пару «id, способ»."""
    needle = _normalize(fact)
    if not needle:
        return None, "пусто"
    for chunk in candidates:
        if needle in _normalize(chunk.text):
            return chunk.id, "дословно"

    fact_words = _words(fact)
    if not fact_words:
        return None, "пусто"
    best_id, best_share = None, 0.0
    for chunk in candidates:
        share = len(fact_words & _words(chunk.text)) / len(fact_words)
        if share > best_share:
            best_id, best_share = chunk.id, share
    if best_id is not None and best_share >= _MIN_OVERLAP:
        return best_id, "по пересечению"
    return None, "не найдено"


def build_goldset(
    questions_path: Path,
    chunks: Sequence[Chunk],
    *,
    limit: int = 0,
    keep_partial: bool = False,
) -> tuple[list[GoldQuestion], MappingReport]:
    """Переводит вопросы набора в наш формат.

    ``keep_partial`` оставляет вопрос, у которого сопоставились не все
    свидетельства. По умолчанию такие вопросы отбрасываются: неполный эталон
    завышает промахи и делает наше число несопоставимым с опубликованными.
    """
    raw = json.loads(Path(questions_path).read_text(encoding="utf-8"))
    by_doc: dict[str, list[Chunk]] = {}
    for chunk in chunks:
        by_doc.setdefault(chunk.doc_id, []).append(chunk)

    report = MappingReport(questions_total=len(raw))
    produced: list[GoldQuestion] = []

    for item in raw:
        question_type = str(item.get("question_type") or "")
        if question_type == NULL_QUERY:
            report.dropped_null += 1
            continue

        gold_ids: list[str] = []
        complete = True
        for evidence in item.get("evidence_list") or []:
            report.evidence_total += 1
            doc_id = stable_doc_id(
                "multihop-rag",
                str(evidence.get("title") or ""),
                str(evidence.get("source") or ""),
            )
            chunk_id, how = _locate(str(evidence.get("fact") or ""), by_doc.get(doc_id, []))
            if chunk_id is None:
                report.unmatched += 1
                complete = False
                continue
            report.matched_exact += how == "дословно"
            report.matched_overlap += how == "по пересечению"
            if chunk_id not in gold_ids:
                gold_ids.append(chunk_id)

        if not gold_ids or (not complete and not keep_partial):
            continue

        query = str(item.get("query") or "")
        produced.append(
            GoldQuestion(
                id=content_hash("multihop-rag", query)[:16],
                question=query,
                answer=str(item.get("answer") or ""),
                gold_chunk_ids=gold_ids,
                gold_doc_ids=sorted({chunk_id.split(":")[0] for chunk_id in gold_ids}),
                # Наш словарь типов и словарь набора не совпадают, а ломать
                # доменную модель под чужую разметку нельзя: по нашему типу
                # считается разбивка всех прежних замеров. Поэтому тип
                # переводится в наш, а исходный сохраняется в примечании —
                # по нему считается разбивка, сопоставимая с опубликованной.
                question_type="multi_hop" if len(gold_ids) > 1 else "single_chunk",
                expected_hops=len(gold_ids),
                generator_model="MultiHop-RAG",
                verified=True,
                notes=f"{_TYPE_PREFIX}{question_type}",
            )
        )
        report.per_type[question_type] += 1
        if limit and len(produced) >= limit:
            break

    report.questions_kept = len(produced)
    logger.info(
        "MultiHop-RAG: вопросов %s из %s, покрытие свидетельств %.1f%%",
        report.questions_kept,
        report.questions_total,
        report.coverage * 100,
    )
    return produced, report


def original_type(question: GoldQuestion) -> str:
    """Тип вопроса в терминах самого набора.

    Нужен для разбивки, сопоставимой с опубликованной: у нас типы свои,
    и прямое сравнение по ним было бы сравнением разных величин.
    """
    notes = question.notes or ""
    return notes.removeprefix(_TYPE_PREFIX) if notes.startswith(_TYPE_PREFIX) else "неизвестен"
