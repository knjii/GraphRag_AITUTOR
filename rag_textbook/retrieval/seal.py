"""SEAL-RAG: «заменяй, а не расширяй» (arXiv 2512.10787), этап 2.

Когда выдачу расширяют ради второго шага, отвлекающие фрагменты вытесняют
нужные — у нас это цена top_k 16. SEAL держит число мест постоянным
и меняет состав: модель ведёт реестр того, что уже подтверждено и чего
не хватает, по каждому пробелу уходит точечный микрозапрос, найденное
встаёт на место худшего неопорного фрагмента.

Круг: **извлечь → оценить → найти → заменить**, не больше ``seal_max_loops``.

* Извлечь и оценить — один вызов модели: факты с номерами фрагментов,
  недостающее с микрозапросом, признак достаточности. Это реестр SEAL
  (у них — сущности, связи и уточнения; здесь факты, без отдельной схемы).
* Найти — по каждому пробелу поиск тем же конвейером (векторный + BM25 +
  граф), кандидаты ранжируются реранкером против микрозапроса. В отличие
  от setr/picker, SEAL приносит фрагменты **не из пула**: единственный из
  методов этапа 2, который чинит доступ, а не только отбор.
* Заменить — вытесняются фрагменты, на которые реестр не опирается, начиная
  с худшего балла реранкера против вопроса. Опорные не трогаются никогда.

Отличия от статьи названы честно: у них GPT-4 и Pinecone, служебная
полезность из четырёх слагаемых (покрытие пробела, подтверждение, новизна,
избыточность); здесь покрытие пробела — балл реранкера против микрозапроса,
новизна — «нет в текущей выдаче», избыточность — дедупликация по id.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.clients.llm_json import loads_llm_json
from rag_textbook.config import RetrievalSettings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import ScoredChunk
from rag_textbook.retrieval.set_selection import format_passages

logger = get_logger("retrieval.seal")

SEAL_CHANNEL = "seal"

SEAL_SYSTEM = (
    "You assemble evidence for a question answering system. You keep a ledger of the "
    "facts the answer needs: which are supported by the passages and which are missing."
)
SEAL_PROMPT = """Question: {question}

Passages, each indicated by a numerical identifier []:

{context}


Build the evidence ledger for the question.
1. "found": facts needed for the answer that the passages state, each with the identifiers of the passages that state it.
2. "missing": facts needed for the answer that no passage states. For each give a short, precise search query that would find it (name the concept, entity or quantity explicitly; do not repeat the whole question). At most {max_gaps}.
3. "sufficient": true if the found facts are enough for a complete answer.

Answer strictly in JSON:
{{"found": [{{"fact": "...", "passages": [1, 3]}}], "missing": [{{"need": "...", "query": "..."}}], "sufficient": false}}"""

SEAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "found": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "fact": {"type": "string"},
                    "passages": {"type": "array", "items": {"type": "integer"}},
                },
                "required": ["fact", "passages"],
            },
        },
        "missing": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"need": {"type": "string"}, "query": {"type": "string"}},
                "required": ["need", "query"],
            },
        },
        "sufficient": {"type": "boolean"},
    },
    "required": ["found", "missing", "sufficient"],
}

Search = Callable[[str], list[ScoredChunk]]
Score = Callable[[str, Sequence[str]], list[float]]


@dataclass
class Ledger:
    supporting: set[int] = field(default_factory=set)
    gaps: list[str] = field(default_factory=list)
    sufficient: bool = False


@dataclass
class SealResult:
    final: list[ScoredChunk]
    added: list[str] = field(default_factory=list)
    loops: int = 0
    gaps: list[str] = field(default_factory=list)
    # ok — дошёл до достаточности или исчерпал пробелы; loops — упёрся
    # в предел кругов; fallback — модель не дала разбираемого реестра.
    status: str = "ok"


def parse_ledger(raw: str, size: int, max_gaps: int) -> Ledger | None:
    try:
        payload = loads_llm_json(_strip_fence(raw))
    except (ValueError, TypeError):
        return None
    if not isinstance(payload, dict):
        return None
    ledger = Ledger(sufficient=bool(payload.get("sufficient")))
    for fact in payload.get("found") or []:
        if not isinstance(fact, dict):
            continue
        for number in fact.get("passages") or []:
            try:
                index = int(number) - 1
            except (TypeError, ValueError):
                continue
            if 0 <= index < size:
                ledger.supporting.add(index)
    for gap in payload.get("missing") or []:
        query = str(gap.get("query") or "").strip() if isinstance(gap, dict) else ""
        if query and query not in ledger.gaps:
            ledger.gaps.append(query)
    ledger.gaps = ledger.gaps[:max_gaps]
    return ledger


def _strip_fence(raw: str) -> str:
    text = (raw or "").strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0]
    return text.strip()


def assess(
    question: str,
    evidence: Sequence[ScoredChunk],
    llm: LLMClient,
    settings: RetrievalSettings,
) -> Ledger | None:
    user = SEAL_PROMPT.format(
        question=question,
        context=format_passages(evidence, settings.selection_llm_chars),
        max_gaps=settings.seal_max_gaps,
    )
    try:
        raw = llm.chat(
            [ChatMessage(role="system", content=SEAL_SYSTEM), ChatMessage(role="user", content=user)],
            purpose="utility",
            json_schema=SEAL_SCHEMA,
            max_tokens=settings.selection_llm_max_tokens,
            temperature=0.0,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("SEAL: вызов модели не удался: %s", exc)
        return None
    ledger = parse_ledger(raw, len(evidence), settings.seal_max_gaps)
    if ledger is None:
        logger.warning("SEAL: реестр не разобран: %s", str(raw)[:160])
    return ledger


def _question_scores(question: str, items: Sequence[ScoredChunk], score: Score) -> list[float]:
    """Балл против вопроса: сохранённый реранкером или досчитанный."""
    missing = [index for index, item in enumerate(items) if item.rerank_score is None]
    values = [item.rerank_score if item.rerank_score is not None else 0.0 for item in items]
    if missing:
        fresh = score(question, [items[index].chunk.text for index in missing])
        for index, value in zip(missing, fresh, strict=True):
            values[index] = value
    return [float(value) for value in values]


def run(
    question: str,
    initial: Sequence[ScoredChunk],
    *,
    search: Search,
    score: Score,
    llm: LLMClient,
    settings: RetrievalSettings,
    top_k: int,
) -> SealResult:
    """Круги SEAL поверх выдачи ``initial``; число мест не растёт."""
    evidence = list(initial[:top_k])
    result = SealResult(final=evidence)
    if not evidence:
        return result
    for loop in range(settings.seal_max_loops):
        ledger = assess(question, evidence, llm, settings)
        if ledger is None:
            result.status = "fallback" if loop == 0 else result.status
            break
        result.loops = loop + 1
        if ledger.sufficient or not ledger.gaps:
            result.status = "ok"
            break
        result.gaps.extend(ledger.gaps)

        present = {item.chunk.id for item in evidence}
        incoming: list[ScoredChunk] = []
        for query in ledger.gaps:
            found = [item for item in search(query) if item.chunk.id not in present]
            if not found:
                continue
            gap_scores = score(query, [item.chunk.text for item in found])
            ranked = sorted(zip(found, gap_scores, strict=True), key=lambda pair: -pair[1])
            for item, _ in ranked[: settings.seal_candidates_per_gap]:
                if item.chunk.id in present:
                    continue
                present.add(item.chunk.id)
                incoming.append(item)
        if not incoming:
            result.status = "ok"
            break

        # Вытесняются неопорные, худшие против вопроса — первыми.
        own = _question_scores(question, evidence, score)
        evictable = sorted(
            (index for index in range(len(evidence)) if index not in ledger.supporting),
            key=lambda index: (own[index], -index),
        )
        slots = evictable[: len(incoming)]
        if not slots:
            # Всё опорное: SEAL не расширяет выдачу, пробел остаётся пробелом.
            result.status = "ok"
            break
        replaced = set(slots)
        added = [
            item.model_copy(update={"channels": [*item.channels, SEAL_CHANNEL]})
            for item in incoming[: len(slots)]
        ]
        result.added.extend(item.chunk.id for item in added)
        # Порядок: опорные, затем пришедшие, затем прочие оставшиеся.
        supporting = [evidence[i] for i in range(len(evidence)) if i in ledger.supporting]
        others = [
            evidence[i]
            for i in range(len(evidence))
            if i not in ledger.supporting and i not in replaced
        ]
        evidence = supporting + added + others
        result.status = "loops"
    result.final = evidence
    return result
