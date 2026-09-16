"""Генерация ответа по найденному контексту.

Что изменено относительно прежней цепочки:

* контекст нумеруется и снабжается ссылкой с **номером страницы**, поэтому ответ
  можно проверить по учебнику; раньше страница не доходила до метаданных вовсе,
  и источники выводились как «p.?»;
* история диалога передаётся в модель отдельно от контекста и уже после
  переписывания вопроса, а не вместо него;
* промпт требует признать нехватку контекста явно — на учебном материале
  выдуманный ответ хуже отказа.
"""

from __future__ import annotations

import re
import time
from collections.abc import Sequence

from rag_textbook.clients.llm import ChatMessage, LLMClient
from rag_textbook.config import Settings
from rag_textbook.logging_setup import get_logger
from rag_textbook.models import Answer, Citation, ScoredChunk
from rag_textbook.retrieval.pipeline import RetrievalPipeline, RetrievalResult
from rag_textbook.utils.text import truncate

logger = get_logger("generation.answering")

_CITATION_RE = re.compile(r"\[(\d+)\]")

NO_CONTEXT_MESSAGE = (
    "В доступных материалах нет данных, чтобы ответить на этот вопрос. "
    "Уточните формулировку или проверьте, загружен ли нужный учебник."
)


def allocate_budget(
    chunks, total_chars: int, *, formula_share: float = 1.6
) -> list[int]:
    """Делит бюджет символов между фрагментами.

    Прежде бюджет делился поровну: «половина окна / число фрагментов».
    При выдаче 16 это давало 768 символов на фрагмент, а 99.8% наших чанков
    длиннее — и **43% формул отрезалось** до того, как модель их увидит
    (измерено 2026-08-19).

    Поровну — не значит справедливо. Фрагмент с формулой теряет от обрезки
    больше, чем связный текст: у формулы нет середины, которую можно
    опустить без потери смысла. Такие фрагменты получают долю
    ``formula_share`` от обычной.

    Остаток от фрагментов, которым бюджет не нужен целиком, раздаётся
    остальным: иначе короткий фрагмент занимал бы место, которое ему
    не требуется.
    """
    if not chunks:
        return []
    weights = [
        formula_share if (item.chunk.has_formula or item.chunk.has_table) else 1.0
        for item in chunks
    ]
    total_weight = sum(weights) or 1.0
    budgets = [max(200, int(total_chars * weight / total_weight)) for weight in weights]

    spare = 0
    for index, item in enumerate(chunks):
        length = len(item.chunk.text)
        if length < budgets[index]:
            spare += budgets[index] - length
            budgets[index] = length
    if spare:
        needy = [i for i, item in enumerate(chunks) if len(item.chunk.text) > budgets[i]]
        for index in needy:
            budgets[index] += spare // len(needy)
    return budgets


def order_for_attention(chunks, mode: str) -> list:
    """Порядок фрагментов в контексте.

    ``relevance`` — как отобрано, лучшее первым.
    ``edges`` — лучшее по краям, слабое в середине: известно, что модель
    хуже использует середину длинного контекста, а при выдаче 16 середина —
    это десяток фрагментов.

    Нумерация цитат считается **после** переупорядочивания, поэтому ссылка
    [3] указывает на третий фрагмент в том виде, в каком его увидела модель.
    """
    items = list(chunks)
    if mode != "edges" or len(items) < 4:
        return items
    head, tail = [], []
    for index, item in enumerate(items):
        (head if index % 2 == 0 else tail).append(item)
    return head + tail[::-1]


def build_context_block(chunks, max_chars_per_chunk) -> str:
    """Собирает пронумерованный контекст.

    Номер фрагмента — это и есть якорь цитаты: модель ссылается на [1], [2],
    а мы разворачиваем их в «учебник, с. N».

    ``max_chars_per_chunk`` принимает и одно число (прежнее поведение),
    и список бюджетов по фрагментам.
    """
    if isinstance(max_chars_per_chunk, int):
        budgets = [max_chars_per_chunk] * len(chunks)
    else:
        budgets = list(max_chars_per_chunk)
    blocks: list[str] = []
    for index, item in enumerate(chunks, start=1):
        header = item.chunk.citation_label()
        if item.chunk.headers:
            header = f"{header} — {item.chunk.headers[-1]}"
        body = truncate(item.chunk.text, budgets[index - 1])
        blocks.append(f"[{index}] {header}\n{body}")
    return "\n\n".join(blocks)


def extract_citations(answer_text: str, chunks: Sequence[ScoredChunk]) -> list[Citation]:
    """Достаёт из ответа использованные ссылки.

    Возвращаем только реально упомянутые фрагменты: список «всех найденных»
    создаёт ложное впечатление, будто модель опиралась на всё сразу.
    """
    used: list[int] = []
    for match in _CITATION_RE.finditer(answer_text or ""):
        number = int(match.group(1))
        if 1 <= number <= len(chunks) and number not in used:
            used.append(number)

    citations: list[Citation] = []
    for number in used:
        item = chunks[number - 1]
        citations.append(
            Citation(
                index=number,
                doc_name=item.chunk.doc_name,
                pages=item.chunk.pages,
                chunk_id=item.chunk.id,
                label=item.chunk.citation_label(),
                from_graph=item.from_graph,
            )
        )
    return citations


class AnswerGenerator:
    def __init__(
        self,
        settings: Settings,
        retrieval: RetrievalPipeline,
        llm: LLMClient,
    ) -> None:
        self.settings = settings
        self.retrieval = retrieval
        self.llm = llm

    def _context_budget(self) -> int:
        """Сколько всего символов отдано под контекст.

        Грубая, но честная оценка: примерно 3 символа на токен для русского.
        Доля окна вынесена в настройку, потому что от неё напрямую зависит,
        доедут ли формулы: при выдаче 16 и окне 8192 на фрагмент выходило
        768 символов, и 43% формул отрезалось.
        """
        share = self.settings.prompts.context_window_share
        return max(800, int(self.settings.llm.context_window * share * 3))

    def _max_chars_per_chunk(self) -> int:
        """Прежняя равная дележка. Оставлена для совместимости вызовов."""
        return max(
            400, self._context_budget() // max(1, self.settings.retrieval.top_k)
        )

    def answer(
        self,
        question: str,
        history: Sequence[ChatMessage] | None = None,
    ) -> Answer:
        started = time.perf_counter()
        retrieval: RetrievalResult = self.retrieval.retrieve(question, history or [])

        if not retrieval.chunks:
            return Answer(
                question=question,
                rewritten_question=retrieval.rewritten_question,
                answer=NO_CONTEXT_MESSAGE,
                contexts=[],
                used_graph=False,
                timings_ms={
                    **retrieval.timings_ms,
                    "total": round((time.perf_counter() - started) * 1000, 1),
                },
            )

        ordered, text, generation_ms = self.answer_from_context(
            question, retrieval.chunks, history=history
        )
        citations = extract_citations(text, ordered)
        timings = {
            **retrieval.timings_ms,
            "generation": round(generation_ms, 1),
            "total": round((time.perf_counter() - started) * 1000, 1),
        }
        return Answer(
            question=question,
            rewritten_question=retrieval.rewritten_question,
            answer=text,
            citations=citations,
            contexts=ordered,
            used_graph=bool(retrieval.route and retrieval.route.use_graph),
            timings_ms=timings,
        )

    def answer_from_context(
        self,
        question: str,
        chunks: Sequence[ScoredChunk],
        *,
        history: Sequence[ChatMessage] | None = None,
    ) -> tuple[list[ScoredChunk], str, float]:
        """Отвечает по готовому контексту, минуя поиск.

        Нужно для сравнения генераторов: если контекст берётся из слепка,
        все модели отвечают по одному и тому же материалу, и разница
        относится к генератору, а не к тому, что кому досталось.
        Заодно поиск во время такого замера не нужен вовсе, и его службы
        не занимают память карты.

        Возвращает тройку «фрагменты в том порядке, в каком их увидела
        модель; текст ответа; время генерации в миллисекундах».
        """
        # Порядок и бюджет считаются здесь, а не в поиске: это свойства
        # промпта, а не выдачи. Нумерация цитат идёт после упорядочивания,
        # поэтому [3] всегда указывает на третий фрагмент в том виде,
        # в каком его увидела модель.
        ordered = order_for_attention(
            chunks, self.settings.prompts.context_order
        )
        budgets = allocate_budget(
            ordered,
            self._context_budget(),
            formula_share=self.settings.prompts.formula_budget_share,
        )
        truncated = sum(
            1 for item, budget in zip(ordered, budgets, strict=True)
            if len(item.chunk.text) > budget
        )
        if truncated:
            # Молчаливое усечение однажды стоило 43% формул. Пусть будет видно.
            logger.info(
                "Контекст: усечено фрагментов %s из %s (бюджет %s символов)",
                truncated,
                len(ordered),
                self._context_budget(),
            )
        context_block = build_context_block(ordered, budgets)
        messages: list[ChatMessage] = [
            ChatMessage(
                role="system",
                content=f"{self.settings.prompts.qa_system}\n\nКонтекст:\n{context_block}",
            )
        ]
        if history:
            messages.extend(history[-self.settings.retrieval.max_history_turns * 2 :])
        messages.append(ChatMessage(role="user", content=question))

        stage = time.perf_counter()
        try:
            text = self.llm.chat(messages, purpose="chat")
        except Exception as exc:  # noqa: BLE001
            logger.error("Генерация ответа не удалась: %s", exc)
            text = (
                "Не удалось получить ответ модели. "
                "Проверьте доступность сервера инференса и повторите запрос."
            )
        generation_ms = (time.perf_counter() - stage) * 1000
        return list(ordered), text, generation_ms
