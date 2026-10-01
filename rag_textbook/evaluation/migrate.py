"""Перенос эталонного набора на новую нарезку того же документа.

Номера фрагментов эталона — это ``doc_id:ordinal`` старой нарезки. После
перенарезки (``CHUNKER_RESPECT_FORMULAS``) границы и число фрагментов
другие, и смещения ``char_start`` тоже сдвинуты: новый чанкер снимает
лишнюю разметку ``$$``, то есть меняет сам текст документа. Поэтому перенос
идёт по тексту, а не по смещениям.

Каждый старый эталонный фрагмент переходит ровно в **один** новый.
Иначе метрика поменялась бы молча: recall считает долю эталонных
фрагментов, и лишний эталон делает вопрос труднее. Кандидаты — новые
фрагменты, накрывающие заметную часть старого. Из них выбирается тот,
в котором есть опора ответа: слова вопроса и ответа, встречающиеся
в старом фрагменте. Если опора разрезана между двумя кандидатами и ни
один не держит её целиком, перенос помечается неоднозначным и выносится
на ручной просмотр.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

SHINGLE = 8
# Какая доля старого фрагмента должна лечь в новый, чтобы тот стал кандидатом.
MIN_COVER = 0.2
# Опора считается целой, если в новом фрагменте есть такая доля слов
# вопроса и ответа из старого. Тогда выбор между кандидатами безразличен:
# ответ лежит в любом из них.
SUPPORT_OK = 0.8
# Иначе опора разрезана, и второй кандидат делает перенос неоднозначным,
# если накрывает не меньше этой доли старого и уступает первому по опоре
# меньше, чем на AMBIGUOUS_MARGIN.
AMBIGUOUS_COVER = 0.35
AMBIGUOUS_MARGIN = 0.15

_WORD = re.compile(r"[0-9A-Za-zА-Яа-яЁё]+")


def normalize(text: str) -> str:
    """Текст без разметки формул и пробелов: так сравнимы обе нарезки."""
    return re.sub(r"\s+", "", text.replace("$", "")).lower()


def shingles(text: str, size: int = SHINGLE) -> set[str]:
    norm = normalize(text)
    if len(norm) <= size:
        return {norm} if norm else set()
    return {norm[i : i + size] for i in range(len(norm) - size + 1)}


def words(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text) if len(w) > 2}


@dataclass
class ChunkMapping:
    old_id: str
    new_id: str | None
    cover: float
    support: float
    candidates: list[tuple[str, float, float]] = field(default_factory=list)
    ambiguous: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "old": self.old_id,
            "new": self.new_id,
            "cover": round(self.cover, 3),
            "support": round(self.support, 3),
            "ambiguous": self.ambiguous,
            "candidates": [
                {"id": cid, "cover": round(cov, 3), "support": round(sup, 3)}
                for cid, cov, sup in self.candidates
            ],
        }


class ChunkIndex:
    """Шинглы новых фрагментов с обратным индексом для быстрого поиска."""

    def __init__(self, chunks: Iterable[Mapping[str, Any]]) -> None:
        self.texts: dict[str, str] = {}
        self.shingles: dict[str, set[str]] = {}
        self.inverted: dict[str, set[str]] = {}
        for chunk in chunks:
            cid = str(chunk["id"])
            self.texts[cid] = str(chunk["text"])
            grams = shingles(self.texts[cid])
            self.shingles[cid] = grams
            for gram in grams:
                self.inverted.setdefault(gram, set()).add(cid)

    def cover(self, text: str) -> dict[str, float]:
        """Доля шинглов ``text``, лежащих в каждом новом фрагменте."""
        grams = shingles(text)
        if not grams:
            return {}
        counts: dict[str, int] = {}
        for gram in grams:
            for cid in self.inverted.get(gram, ()):
                counts[cid] = counts.get(cid, 0) + 1
        return {cid: count / len(grams) for cid, count in counts.items()}


def map_chunk(
    old_id: str,
    old_text: str,
    index: ChunkIndex,
    evidence: str,
) -> ChunkMapping:
    """Один старый фрагмент → один новый, с опорой на слова вопроса и ответа."""
    covers = index.cover(old_text)
    candidates = [(cid, cov) for cid, cov in covers.items() if cov >= MIN_COVER]
    if not candidates:
        best = max(covers.items(), key=lambda item: item[1], default=(None, 0.0))
        return ChunkMapping(old_id, None, best[1], 0.0)

    # Опора — слова вопроса и ответа, которые есть в старом фрагменте:
    # только они говорят, какая его часть отвечает на вопрос.
    anchor = words(evidence) & words(old_text)
    scored: list[tuple[str, float, float]] = []
    for cid, cov in candidates:
        support = len(anchor & words(index.texts[cid])) / len(anchor) if anchor else 0.0
        scored.append((cid, cov, support))
    scored.sort(key=lambda item: (item[2], item[1]), reverse=True)
    best_id, best_cover, best_support = scored[0]
    ambiguous = best_support < SUPPORT_OK and any(
        cov >= AMBIGUOUS_COVER and best_support - sup < AMBIGUOUS_MARGIN
        for _, cov, sup in scored[1:]
    )
    return ChunkMapping(old_id, best_id, best_cover, best_support, scored, ambiguous)


@dataclass
class MigrationReport:
    mappings: dict[str, list[ChunkMapping]]
    lost: list[str]
    collided: list[str]
    ambiguous: list[str]

    def summary(self) -> dict[str, Any]:
        flat = [m for items in self.mappings.values() for m in items]
        covers = sorted(m.cover for m in flat if m.new_id)
        return {
            "вопросов": len(self.mappings),
            "эталонных фрагментов": len(flat),
            "перенесено": sum(1 for m in flat if m.new_id),
            "потеряно вопросов": len(self.lost),
            "два эталона слились в один": len(self.collided),
            "неоднозначно": len(self.ambiguous),
            "медиана охвата": covers[len(covers) // 2] if covers else 0.0,
        }


def migrate_goldset(
    questions: Sequence[Any],
    old_chunks: Mapping[str, Mapping[str, Any]],
    new_chunks: Sequence[Mapping[str, Any]],
    fixes: Mapping[str, Mapping[str, str]] | None = None,
) -> tuple[list[Any], MigrationReport]:
    """Возвращает вопросы с новыми номерами фрагментов и отчёт.

    ``fixes`` — ручные поправки ``{вопрос: {старый номер: новый номер}}``
    по листу просмотра. Задаются от старого номера, поэтому переживают
    повторный перенос; поправленный перенос неоднозначным не считается.

    Вопросы, у которых хоть один эталон не перенёсся, в набор не входят:
    вопрос с половиной эталона мерил бы уже другое. Вопросы, где два
    старых эталона легли в один новый фрагмент, тоже выбывают — пара
    перестала быть парой, а тип вопроса ещё обещает два шага.
    """
    by_doc: dict[str, list[Mapping[str, Any]]] = {}
    for chunk in new_chunks:
        by_doc.setdefault(str(chunk["doc_id"]), []).append(chunk)
    indexes = {doc: ChunkIndex(chunks) for doc, chunks in by_doc.items()}

    migrated: list[Any] = []
    mappings: dict[str, list[ChunkMapping]] = {}
    lost: list[str] = []
    collided: list[str] = []
    ambiguous: list[str] = []
    for question in questions:
        evidence = f"{question.question} {question.answer}"
        items: list[ChunkMapping] = []
        for old_id in question.gold_chunk_ids:
            old = old_chunks.get(old_id)
            doc = str(old["doc_id"]) if old else old_id.split(":", 1)[0]
            if old is None or doc not in indexes:
                items.append(ChunkMapping(old_id, None, 0.0, 0.0))
                continue
            mapping = map_chunk(old_id, str(old["text"]), indexes[doc], evidence)
            fixed = (fixes or {}).get(question.id, {}).get(old_id)
            if fixed is not None:
                if fixed not in indexes[doc].texts:
                    raise ValueError(f"поправка {question.id}: нет фрагмента {fixed}")
                mapping.new_id = fixed
                mapping.ambiguous = False
            items.append(mapping)
        mappings[question.id] = items
        new_ids = [m.new_id for m in items]
        if any(new_id is None for new_id in new_ids):
            lost.append(question.id)
            continue
        if len(set(new_ids)) < len(new_ids):
            collided.append(question.id)
            continue
        if any(m.ambiguous for m in items):
            ambiguous.append(question.id)
        migrated.append(question.model_copy(update={"gold_chunk_ids": new_ids}))
    return migrated, MigrationReport(mappings, lost, collided, ambiguous)


def chunks_fingerprint(chunks: Sequence[Mapping[str, Any]]) -> str:
    """Отпечаток нарезки: номера и хэши текста по порядку.

    Перенос делается на ноутбуке, а индекс строится на сервере. Совпадение
    отпечатков доказывает, что сервер нарезал ровно то, на что перенесён
    эталон; иначе номера фрагментов эталона указывают в чужие тексты.
    """
    import hashlib

    lines = [f"{chunk['id']}|{chunk['text_hash']}" for chunk in chunks]
    return hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest()


def review_sheet(
    questions: Sequence[Any],
    report: MigrationReport,
    old_chunks: Mapping[str, Mapping[str, Any]],
    new_chunks: Mapping[str, Mapping[str, Any]],
    excerpt: int = 500,
) -> str:
    """Лист ручного просмотра неоднозначных переносов (Markdown)."""
    by_id = {question.id: question for question in questions}
    lines = [
        "# Перенос эталона: неоднозначные случаи",
        "",
        "Опора ответа разрезана между двумя новыми фрагментами. Выбран первый",
        "кандидат; если верен другой — впишите его номер в `fix` файла",
        '`migration-fixes.json` вида `{"<вопрос>": {"<старый>": "<новый>"}}`.',
        "",
    ]
    for number, qid in enumerate(report.ambiguous, start=1):
        question = by_id[qid]
        lines += [
            f"## {number}. `{qid}` ({question.question_type})",
            "",
            f"**Вопрос.** {question.question}",
            "",
            f"**Ответ.** {question.answer}",
            "",
        ]
        for mapping in report.mappings[qid]:
            if not mapping.ambiguous:
                continue
            old_text = str(old_chunks[mapping.old_id]["text"])[:excerpt]
            lines += [f"Старый `{mapping.old_id}`:", "", f"> {old_text}", ""]
            for cid, cover, support in mapping.candidates[:3]:
                mark = "выбран" if cid == mapping.new_id else "кандидат"
                text = str(new_chunks[cid]["text"])[:excerpt]
                lines += [
                    f"- {mark} `{cid}` — охват {cover:.2f}, опора {support:.2f}",
                    "",
                    f"  > {text}",
                    "",
                ]
    return "\n".join(lines)
