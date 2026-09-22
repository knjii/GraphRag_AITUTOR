"""Офлайн-среда RL для генератора: вопрос и контекст из слепка поиска.

Поиск во время обучения не выполняется. Контекст каждого вопроса — ровно
те фрагменты, что отобрал конвейер при снятии слепка, в том же порядке
и с тем же бюджетом символов, что и в сервисе (``build_answer_messages``).
Поэтому шаг среды — это одна генерация и расчёт награды за миллисекунды,
а обученная политика мерится на том же входе, на котором будет работать.

Для обучения нужны вопросы с **других** книг: 388 вопросов эталонного
набора остаются тестом. Фильтр по документам — :func:`split_by_docs`.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rag_textbook.config import Settings
from rag_textbook.evaluation.goldset import load_goldset, looks_leaky, references_numbering
from rag_textbook.evaluation.verdicts import USABLE, VerdictSet
from rag_textbook.generation.answering import build_answer_messages
from rag_textbook.models import Chunk, ScoredChunk
from rag_textbook.rewards.composite import RewardConfig, compute_reward


@dataclass
class Example:
    """Один эпизод: промпт и всё, что нужно для награды."""

    question_id: str
    question_type: str
    question: str
    messages: list[dict[str, str]]
    context: str
    reference: str
    gold_in_context: bool
    doc_ids: list[str]

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_chunks(parsed_dir: Path) -> dict[str, Chunk]:
    """Фрагменты из выгрузки разбора — тот же источник, что и в замерах."""
    chunks: dict[str, Chunk] = {}
    for path in sorted(Path(parsed_dir).rglob("*_chunks.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        items = payload["chunks"] if isinstance(payload, dict) else payload
        for item in items:
            chunks[item["id"]] = Chunk(**item)
    return chunks


def load_trace(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("kind") != "trace-header":
            rows.append(record)
    return rows


def build_examples(
    settings: Settings,
    trace_rows: Iterable[dict[str, Any]],
    chunks: dict[str, Chunk],
    goldset_path: Path,
) -> list[Example]:
    """Эпизоды по слепку. Вопрос без эталона или без фрагментов — ошибка, а не пропуск."""
    gold = {q.id: q for q in load_goldset(goldset_path)}
    examples: list[Example] = []
    missing: list[str] = []
    for row in trace_rows:
        question = gold.get(row["question_id"])
        if question is None:
            raise ValueError(f"Вопроса {row['question_id']} нет в эталонном наборе")
        picked = [ScoredChunk(chunk=chunks[cid], score=1.0) for cid in row["final"] if cid in chunks]
        if len(picked) < len(row["final"]):
            missing.append(row["question_id"])
        if not picked:
            continue
        ordered, messages = build_answer_messages(settings, row["question"], picked)
        # Контекст для награды берётся из промпта: награда должна видеть
        # ровно то, что видела модель, включая усечение. Но без инструкций —
        # в промпте v4 есть пример формулы, и её перенос в ответ считался бы
        # переносом из контекста.
        system = messages[0].content
        prefix = f"{settings.prompts.qa_system}\n\nКонтекст:\n"
        context = system[len(prefix):] if system.startswith(prefix) else system
        shown = {item.chunk.id for item in ordered}
        # Эталон — только те эталонные фрагменты, что попали в контекст:
        # у связывающего вопроса с одним видимым фрагментом из двух награда
        # иначе требовала бы формул, которых модель не видела.
        visible_gold = [cid for cid in question.gold_chunk_ids if cid in shown]
        reference = "\n\n".join(
            chunks[cid].text
            for cid in (visible_gold or question.gold_chunk_ids)
            if cid in chunks
        )
        examples.append(Example(
            question_id=row["question_id"],
            question_type=row.get("question_type", question.question_type),
            question=row["question"],
            messages=[{"role": m.role, "content": m.content} for m in messages],
            context=context,
            reference=reference,
            gold_in_context=bool(visible_gold),
            doc_ids=sorted({item.chunk.doc_id for item in ordered}),
        ))
    if len(missing) > len(examples) // 2:
        raise ValueError(
            f"У {len(missing)} вопросов фрагменты не найдены в выгрузке разбора — "
            "проверьте PARSED_DIR"
        )
    return examples


def split_by_docs(
    examples: Sequence[Example], test_doc_ids: set[str]
) -> tuple[list[Example], list[Example]]:
    """Обучение — только эпизоды, не касающиеся тестовых книг.

    Эпизод с контекстом хотя бы из одной тестовой книги уходит в тест:
    иначе политика видела бы при обучении текст, по которому её потом
    проверяют.
    """
    train, test = [], []
    for example in examples:
        (test if set(example.doc_ids) & test_doc_ids else train).append(example)
    return train, test


def drop_unfit(
    examples: Iterable[Example], verdicts: VerdictSet | None = None
) -> tuple[list[Example], Counter[str]]:
    """Эпизоды, годные для обучения, и счётчик причин отбраковки.

    Генератор вопросов проверяет утечки при создании, но импортированный
    набор этих проверок не проходил (задача 019): вопрос с отсылкой к тексту
    («в данном фрагменте») или к номеру формулы учит политику зависеть
    от конкретного источника. Отклонённые вручную — тоже вон.

    Применяется к обучению, а не к тесту: тестовый набор фиксирован,
    и база с обученной моделью меряются на одних и тех же вопросах.
    """
    kept: list[Example] = []
    reasons: Counter[str] = Counter()
    for example in examples:
        verdict = verdicts.verdicts.get(example.question_id) if verdicts else None
        if verdict is not None and verdict.verdict not in USABLE:
            reasons[f"вердикт:{verdict.verdict}"] += 1
        elif looks_leaky(example.question):
            reasons["отсылка к тексту"] += 1
        elif references_numbering(example.question):
            reasons["номер формулы или раздела"] += 1
        else:
            kept.append(example)
    return kept, reasons


def save_jsonl(examples: Iterable[Example], path: Path) -> int:
    count = 0
    with Path(path).open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(example.as_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def load_jsonl(path: Path) -> list[Example]:
    return [
        Example(**json.loads(line))
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def completion_text(completion: Any) -> str:
    """Текст генерации в любом формате TRL: строка или список сообщений."""
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content", ""))
    return str(completion or "")


def make_reward_function(config: RewardConfig | None = None):
    """Функция награды в сигнатуре ``GRPOTrainer`` (TRL).

    TRL передаёт колонки набора данных именованными аргументами —
    отсюда ``context``, ``reference``, ``gold_in_context`` и ``question``.
    Флаг обрыва TRL не передаёт; обрыв ловится по пределу длины в воротах
    и отдельно — по ``completion_ids`` в скрипте обучения.
    """

    def reward(
        completions, context, reference, gold_in_context, question=None, **_: Any
    ) -> list[float]:
        questions = question or [""] * len(completions)
        return [
            compute_reward(
                completion_text(item),
                context=ctx,
                reference=ref,
                question=asked,
                gold_in_context=bool(gold),
                config=config,
            ).total
            for item, ctx, ref, gold, asked in zip(
                completions, context, reference, gold_in_context, questions, strict=True
            )
        ]

    reward.__name__ = "formula_faithful_reward"
    return reward
