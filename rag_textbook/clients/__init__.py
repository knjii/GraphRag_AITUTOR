"""Клиенты внешних сервисов инференса.

Всё общение идёт через OpenAI-совместимый протокол, поэтому Ollama на этапе
разработки и vLLM на этапе пилота отличаются только значением base_url.
Прежний код содержал ~300 строк подкласса поверх устаревшего ``ChatOllama``
с сырыми ``requests`` — эта абстракция его заменяет.
"""

from rag_textbook.clients.embeddings import EmbeddingClient, build_embedding_client
from rag_textbook.clients.llm import ChatMessage, LLMClient, build_llm_client
from rag_textbook.clients.reranker import RerankerClient, build_reranker_client

__all__ = [
    "ChatMessage",
    "EmbeddingClient",
    "LLMClient",
    "RerankerClient",
    "build_embedding_client",
    "build_llm_client",
    "build_reranker_client",
]
