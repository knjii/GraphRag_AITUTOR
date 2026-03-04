import argparse
import json
import os

from dotenv import load_dotenv
from opentelemetry import trace
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from phoenix.otel import register

from chat_history import append_chat_turn, build_chat_history_messages, clear_chat_history
from rag_chain import build_rag_chain
from settings import Settings
from utils import chroma_has_data


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question to the local RAG index.")
    parser.add_argument("question", type=str, help="Вопрос к локальным материалам")
    parser.add_argument(
        "--session-id",
        type=str,
        default=None,
        help="Conversation session id for chat history storage.",
    )
    parser.add_argument(
        "--max-history-turns",
        type=int,
        default=None,
        help="How many latest turns to send to LLM/retrieval.",
    )
    parser.add_argument(
        "--stateless",
        action="store_true",
        help="Disable chat history usage for this request.",
    )
    parser.add_argument(
        "--clear-history",
        action="store_true",
        help="Clear saved history for the selected session before asking.",
    )
    args = parser.parse_args()

    tracer_provider = register(
        project_name="ai_tutor_tracing",
        endpoint="http://127.0.0.1:4317",
        protocol="grpc",
        auto_instrument=True,
    )
    tracer = trace.get_tracer(__name__, tracer_provider=tracer_provider)

    settings = Settings()
    if not chroma_has_data(settings):
        raise SystemExit("Chroma индекс не найден. Сначала запустите `python ingest.py`.")

    session_id = (args.session_id or settings.chat_session_id or "default").strip() or "default"
    history_enabled = bool(settings.conversational_rag_enabled) and not bool(args.stateless)
    history_limit = (
        int(settings.chat_history_max_turns)
        if args.max_history_turns is None
        else max(0, int(args.max_history_turns))
    )

    if args.clear_history:
        clear_chat_history(settings, session_id)

    chat_history = (
        build_chat_history_messages(settings, session_id, max_turns=history_limit)
        if history_enabled
        else []
    )
    history_turns_sent = len(chat_history) // 2

    chain = build_rag_chain(settings)
    emb_path = str(settings.embedding_model_path)

    metadata = {
        "llm.model_name": settings.ollama_model,
        "embed.model_path": os.path.basename(emb_path),
        "llm.temperature": float(settings.temperature),
        "llm.n_ctx": int(settings.ollama_num_ctx),
        "retrieval.mode": str(settings.retriever_mode),
        "retrieval.top_k": int(settings.top_k),
        "retrieval.hybrid_sparse_k": int(settings.hybrid_sparse_k),
        "retrieval.hybrid_dense_weight": float(settings.hybrid_dense_weight),
        "retrieval.hybrid_sparse_weight": float(settings.hybrid_sparse_weight),
        "rag.question": args.question,
        "rag.session_id": session_id,
        "rag.chat_history_enabled": bool(history_enabled),
        "rag.chat_history_turns_sent": int(history_turns_sent),
        "rag.chat_history_max_turns": int(history_limit),
    }

    with tracer.start_as_current_span("rag_query") as span:
        span.set_attribute(
            SpanAttributes.OPENINFERENCE_SPAN_KIND,
            OpenInferenceSpanKindValues.CHAIN.value,
        )

        for key, value in metadata.items():
            span.set_attribute(f"metadata.{key}", value)

        result = chain.invoke({"input": args.question, "chat_history": chat_history})

        answer = str(result.get("answer") or "").strip()
        sources = result.get("context", []) or result.get("source_documents", [])

        span.set_attribute(SpanAttributes.INPUT_VALUE, args.question)
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

        for i, doc in enumerate(sources):
            span.set_attribute(
                f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}",
                doc.page_content,
            )
            span.set_attribute(
                f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}",
                json.dumps(doc.metadata, ensure_ascii=False),
            )
            span.set_attribute(
                f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_ID}",
                doc.metadata.get("source", str(i)),
            )

    if history_enabled:
        append_chat_turn(
            settings,
            session_id,
            user_query=args.question,
            assistant_answer=answer,
            metadata={
                "sources": len(sources),
                "retrieval_mode": settings.retriever_mode,
            },
        )

    print("\n=== Ответ модели ===")
    print(answer)


if __name__ == "__main__":
    load_dotenv()
    main()
