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


def build_context_block(chunks: Sequence[ScoredChunk], max_chars_per_chunk: int) -> str:
    """Собирает пронумерованный контекст.

    Номер фрагмента — это и есть якорь цитаты: модель ссылается на [1], [2],
    а мы разворачиваем их в «учебник, с. N».
    """
    blocks: list[str] = []
    for index, item in enumerate(chunks, start=1):
        header = item.chunk.citation_label()
        if item.chunk.headers:
            header = f"{header} — {item.chunk.headers[-1]}"
        body = truncate(item.chunk.text, max_chars_per_chunk)
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

    def _max_chars_per_chunk(self) -> int:
        """Делит окно контекста между фрагментами.

        Грубая, но честная оценка: примерно 3 символа на токен для русского,
        половина окна отдана под контекст, остальное — под промпт и ответ.
        """
        budget_tokens = self.settings.llm.context_window // 2
        budget_chars = budget_tokens * 3
        return max(400, budget_chars // max(1, self.settings.retrieval.top_k))

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

        context_block = build_context_block(retrieval.chunks, self._max_chars_per_chunk())
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

        citations = extract_citations(text, retrieval.chunks)
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
            contexts=retrieval.chunks,
            used_graph=bool(retrieval.route and retrieval.route.use_graph),
            timings_ms=timings,
        )
