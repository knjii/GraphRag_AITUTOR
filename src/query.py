import argparse
from dotenv import load_dotenv
import json
import os

from phoenix.otel import register
from opentelemetry import trace
from openinference.semconv.trace import SpanAttributes, OpenInferenceSpanKindValues, DocumentAttributes

from rag_chain import build_rag_chain
from settings import Settings
from utils import chroma_has_data
from vectorstore import load_vector_store

def main():
    parser = argparse.ArgumentParser(description="Ask a question to the local RAG index.")
    parser.add_argument("question", type=str, help="Вопрос к локальным материалам")
    args = parser.parse_args()

    tracer_provider = register(
        project_name="ai_tutor_tracing",
        endpoint="http://127.0.0.1:4317",
        protocol="grpc",
        auto_instrument=True 
    )
    tracer = trace.get_tracer(__name__, tracer_provider=tracer_provider)

    settings = Settings()
    if not chroma_has_data(settings):
        raise SystemExit("Chroma индекс не найден. Сначала запустите `python ingest.py`.")
    
    chain = build_rag_chain(settings)
    store = load_vector_store(settings)

    emb_path = str(settings.embedding_model_path)

    metadata = {
        "llm.model_name": settings.ollama_model,
        "embed.model_path": os.path.basename(emb_path),
        "llm.temperature": float(settings.temperature),
        "llm.n_ctx": int(settings.ollama_num_ctx),
        "retrieval.top_k": int(settings.top_k),
        "rag.question": args.question
    }
    with tracer.start_as_current_span("rag_query") as span:
            span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, OpenInferenceSpanKindValues.CHAIN.value)
            
            for key, value in metadata.items():
                span.set_attribute(f"metadata.{key}", value)

            result = chain.invoke({"input": args.question, "chat_history": []})

            answer = result.get("answer", "").strip()
            sources = result.get("context", []) or result.get("source_documents", [])

            span.set_attribute(SpanAttributes.INPUT_VALUE, args.question)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)

            for i, doc in enumerate(sources):
                span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_CONTENT}", 
                    doc.page_content
                )
                span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_METADATA}", 
                    json.dumps(doc.metadata, ensure_ascii=False)
                )
                span.set_attribute(
                    f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{i}.{DocumentAttributes.DOCUMENT_ID}", 
                    doc.metadata.get("source", str(i))
                )

            print("\n=== Ответ модели ===")
            print(answer)


if __name__ == "__main__":
    load_dotenv()
    main()
