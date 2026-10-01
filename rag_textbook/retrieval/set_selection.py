"""Отбор множества фрагментов моделью: SetR и Context-Picker (этап 2).

Реранкер оценивает фрагменты поодиночке, и второй фрагмент связывающей пары,
который сам на вопрос не отвечает, уходит за отсечку. Здесь модель видит
верхние кандидаты разом и выбирает **множество**, которое вместе покрывает
вопрос. Выбранное ставится первым, остаток пула — за ним в порядке
реранкера, так что recall@k сравним с обычной выдачей, а размер и полнота
самого множества меряются отдельно (фрагменты помечены каналом ``selected``).

``setr``
    SetR, Lee et al., ACL 2025, arXiv 2507.06838. Подсказка ``selection_IRI``
    перенесена дословно из их ``generate_data.py``: перечислить потребности
    вопроса, найти под каждую фрагменты, выбрать покрывающие. Их модель —
    Llama-3.1-8B после SFT на ответах GPT-4o; весов нет, поэтому здесь
    та же подсказка на нашей модели без обучения.
``picker``
    Context-Picker, arXiv 2512.14465: минимальное достаточное множество.
    Формат вывода тот же, что у SetR, чтобы обученный отборщик (GRPO,
    награда — покрытие эталона, затем штраф за лишнее) подключался той же
    строкой настроек.

Ответ разбирается по строке ``### Final Selection: [4] [2]``. Не разобрался —
порядок реранкера без изменений и статус ``fallback``: деградация видна
в сводке, а не выдаёт себя за результат метода.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.config import RetrievalSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import ScoredChunk

logger = get_logger("retrieval.set_selection")

SELECTED_CHANNEL = "selected"

# SetR, generate_data.py: selection_sys_prompt и selection_IRI_prompt.
SETR_SYSTEM = (
    "You are RankLLM, an intelligent assistant that can rank and select passages "
    "based on their relevancy to the query."
)
SETR_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. Select the passages based on their relevance to the search query: {question}.

{context}


Search Query: {question}


Please follow the steps below:
Step 1. Please list up the information requirements to answer the query.
Step 2. for each requirement in Step 1, find the passages that has the information of the requirement.
Step 3. Choose the passages that mostly covers clear and diverse informations to answer the query. Number of passages is unlimited. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""

# Context-Picker публикует награду, а не подсказку; формулировка наша,
# по их определению: минимальное множество, достаточное для ответа.
PICKER_SYSTEM = (
    "You are an evidence picker for a question answering system. You select the "
    "smallest set of passages that is sufficient to answer the question."
)
PICKER_PROMPT = """I will provide you with {num} passages, each indicated by a numerical identifier []. The question: {question}

{context}


Question: {question}


Think briefly about which facts the answer needs and where each fact is stated. Some questions need several passages that complement each other: a passage that does not answer the question alone may still be required. Then output the minimal sufficient set: every passage needed for a complete answer, and no redundant passages. The format of final output should be '### Final Selection: [] []', e.g., ### Final Selection: [4] [2]."""

PROMPTS = {"setr": (SETR_SYSTEM, SETR_PROMPT), "picker": (PICKER_SYSTEM, PICKER_PROMPT)}
MODES = tuple(PROMPTS)

_FINAL = re.compile(r"#+[ 	]*Final\s+Selection[ 	]*:?", re.IGNORECASE)
_INDEX = re.compile(r"\[(\d+)\]")


@dataclass
class SetSelection:
    """Что выбрала модель; ``status`` — ok, empty или fallback."""

    ordered: list[ScoredChunk]
    chosen: list[str] = field(default_factory=list)
    status: str = "ok"


def format_passages(items: Sequence[ScoredChunk], max_chars: int) -> str:
    # Как в SetR: переводы строк сжимаются, идентификатор — с единицы.
    lines = []
    for index, item in enumerate(items, start=1):
        text = re.sub(r"\n+", " ", item.chunk.text.strip())[:max_chars]
        lines.append(f"[{index}] {text}")
    return "\n\n\n".join(lines)


def build_messages(mode: str, question: str, items: Sequence[ScoredChunk], max_chars: int) -> list[ChatMessage]:
    system, template = PROMPTS[mode]
    user = template.format(
        num=len(items), question=question, context=format_passages(items, max_chars)
    )
    return [ChatMessage(role="system", content=system), ChatMessage(role="user", content=user)]


def parse_selection(raw: str, size: int) -> list[int] | None:
    """Номера из строки итогового выбора, с нуля, без повторов и выходов за пул.

    ``None`` — строки выбора нет вовсе (ответ оборван или не по формату);
    пустой список — строка есть, но годных номеров в ней нет.
    """
    # Берётся последняя метка: в рассуждении модель может цитировать формат.
    matches = list(_FINAL.finditer(raw or ""))
    if not matches:
        return None
    match = matches[-1]
    # Только первая непустая строка после метки: дальше модель иногда
    # продолжает текст, а номера иногда переносит на следующую строку.
    lines = [line for line in (raw or "")[match.end():].splitlines() if line.strip()]
    tail = lines[0] if lines else ""
    picked: list[int] = []
    for number in _INDEX.findall(tail):
        index = int(number) - 1
        if 0 <= index < size and index not in picked:
            picked.append(index)
    return picked


def mark_selected(item: ScoredChunk) -> ScoredChunk:
    return item.model_copy(update={"channels": [*item.channels, SELECTED_CHANNEL]})


def select_set(
    items: Sequence[ScoredChunk],
    question: str,
    llm: LLMClient,
    settings: RetrievalSettings,
    top_k: int,
    *,
    mode: str | None = None,
) -> SetSelection:
    """Модель выбирает множество из верхних ``selection_llm_pool`` кандидатов.

    Выбранное идёт первым (не больше ``top_k``), затем остаток пула в прежнем
    порядке. Состав пула не меняется.
    """
    mode = mode or settings.selection_mode
    if mode not in PROMPTS:
        raise ValueError(f"Нет подсказки отбора для режима {mode}")
    if len(items) <= 1:
        return SetSelection(ordered=list(items), status="empty")
    window = list(items[: settings.selection_llm_pool])
    messages = build_messages(mode, question, window, settings.selection_llm_chars)
    try:
        raw = llm.chat(
            messages,
            purpose="utility",
            max_tokens=settings.selection_llm_max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Отбор %s: вызов модели не удался: %s", mode, exc)
        return SetSelection(ordered=list(items), status="fallback")

    picked = parse_selection(raw, len(window))
    if picked is None:
        logger.warning("Отбор %s: нет строки Final Selection: %s", mode, str(raw)[-160:])
        return SetSelection(ordered=list(items), status="fallback")
    if not picked:
        return SetSelection(ordered=list(items), status="empty")

    picked = picked[:top_k]
    chosen_set = set(picked)
    head = [mark_selected(window[index]) for index in picked]
    rest = [item for position, item in enumerate(items) if position not in chosen_set]
    return SetSelection(
        ordered=head + rest,
        chosen=[item.chunk.id for item in head],
        status="ok",
    )


def selected_ids(items: Sequence[ScoredChunk]) -> list[str]:
    """Выбранное моделью множество по меткам канала, в порядке выдачи."""
    return [item.chunk.id for item in items if SELECTED_CHANNEL in item.channels]
