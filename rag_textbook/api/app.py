"""FastAPI-сервис.

Появление сервиса закрывает главную проблему прежней архитектуры: там на каждый
вопрос поднимался процесс, который заново собирал цепочку, заново строил BM25-индекс
из всего корпуса в оперативной памяти и заново открывал соединения. При десятке
одновременных пользователей это неработоспособно.

Здесь ресурсы создаются один раз на процесс, число одновременных запросов ограничено,
а история диалога привязана к пользователю.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Annotated, Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from pydantic import BaseModel, Field

from rag_textbook.config import Settings
from rag_textbook.context import AppContext, build_context
from rag_textbook.logging_setup import get_logger

logger = get_logger("api")


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    session_id: str = Field(default="default", max_length=64)
    stateless: bool = False


class CitationOut(BaseModel):
    index: int
    label: str
    doc_name: str
    pages: list[int]
    from_graph: bool


class ContextOut(BaseModel):
    index: int
    chunk_id: str
    label: str
    score: float
    channels: list[str]
    preview: str


class AskResponse(BaseModel):
    answer: str
    question: str
    rewritten_question: str
    citations: list[CitationOut]
    contexts: list[ContextOut]
    used_graph: bool
    graph_share: float
    timings_ms: dict[str, float]


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings()
    context = build_context(settings)
    app.state.context = context
    # Семафор ограничивает нагрузку на локальный сервер инференса: без него
    # запросы выстраиваются в неуправляемую очередь и упираются в таймауты.
    app.state.semaphore = asyncio.Semaphore(settings.service.max_concurrent_requests)
    logger.info("Сервис запущен")
    try:
        yield
    finally:
        context.close()
        logger.info("Сервис остановлен")


app = FastAPI(
    title="RAG-ассистент по учебной литературе",
    version="0.2.0",
    lifespan=lifespan,
)


def get_context(request: Request) -> AppContext:
    context = getattr(request.app.state, "context", None)
    if context is None:  # pragma: no cover
        raise HTTPException(status_code=503, detail="Сервис ещё не готов")
    return context


def get_user_id(
    x_user_id: Annotated[str | None, Header(alias="X-User-Id")] = None,
) -> str:
    """Идентификатор пользователя.

    Заголовок обязателен: прежняя схема позволяла прочитать чужую переписку,
    просто угадав идентификатор сессии. Полноценную аутентификацию ставим
    перед сервисом на этапе внедрения; здесь фиксируем, что пользователь есть.
    """
    if not x_user_id or not x_user_id.strip():
        raise HTTPException(status_code=401, detail="Требуется заголовок X-User-Id")
    return x_user_id.strip()[:64]


@app.get("/health")
async def health(context: Annotated[AppContext, Depends(get_context)]) -> dict[str, Any]:
    report = await asyncio.to_thread(context.health)
    if report["status"] == "error":
        raise HTTPException(status_code=503, detail=report)
    return report


@app.post("/ask", response_model=AskResponse)
async def ask(
    payload: AskRequest,
    request: Request,
    context: Annotated[AppContext, Depends(get_context)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> AskResponse:
    settings = context.settings
    semaphore: asyncio.Semaphore = request.app.state.semaphore

    try:
        await asyncio.wait_for(semaphore.acquire(), timeout=5.0)
    except TimeoutError as exc:
        raise HTTPException(status_code=429, detail="Сервис перегружен, повторите позже") from exc

    try:
        history = (
            []
            if payload.stateless
            else await asyncio.to_thread(
                context.history.recent,
                user_id,
                payload.session_id,
                settings.retrieval.max_history_turns,
            )
        )
        answer = await asyncio.wait_for(
            asyncio.to_thread(context.generator.answer, payload.question, history),
            timeout=settings.service.request_timeout_seconds,
        )
    except TimeoutError as exc:
        raise HTTPException(status_code=504, detail="Превышено время ожидания ответа") from exc
    finally:
        semaphore.release()

    if not payload.stateless:
        await asyncio.to_thread(
            context.history.append, user_id, payload.session_id, "user", payload.question
        )
        await asyncio.to_thread(
            context.history.append, user_id, payload.session_id, "assistant", answer.answer
        )

    graph_share = (
        sum(1 for item in answer.contexts if item.from_graph) / len(answer.contexts)
        if answer.contexts
        else 0.0
    )
    return AskResponse(
        answer=answer.answer,
        question=answer.question,
        rewritten_question=answer.rewritten_question,
        citations=[
            CitationOut(
                index=citation.index,
                label=citation.label,
                doc_name=citation.doc_name,
                pages=citation.pages,
                from_graph=citation.from_graph,
            )
            for citation in answer.citations
        ],
        contexts=[
            ContextOut(
                index=index,
                chunk_id=item.chunk.id,
                label=item.chunk.citation_label(),
                score=round(item.score, 6),
                channels=item.channels,
                preview=item.chunk.text[:300],
            )
            for index, item in enumerate(answer.contexts, start=1)
        ],
        used_graph=answer.used_graph,
        graph_share=round(graph_share, 3),
        timings_ms=answer.timings_ms,
    )


@app.delete("/sessions/{session_id}")
async def clear_session(
    session_id: str,
    context: Annotated[AppContext, Depends(get_context)],
    user_id: Annotated[str, Depends(get_user_id)],
) -> dict[str, int]:
    removed = await asyncio.to_thread(context.history.clear, user_id, session_id)
    return {"removed": removed}


def run() -> None:  # pragma: no cover
    import uvicorn

    settings = Settings()
    uvicorn.run(
        "rag_textbook.api.app:app",
        host=settings.service.host,
        port=settings.service.port,
        log_level=settings.log_level.lower(),
    )
