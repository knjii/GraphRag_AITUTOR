# from langchain_community.chat_models import ChatOllama
try:
    from langchain_ollama import ChatOllama  # type: ignore
except Exception:
    from langchain_community.chat_models import ChatOllama  # type: ignore
from settings import Settings


def get_chat_model(settings: Settings) -> ChatOllama:
    return ChatOllama(
        model=settings.ollama_model,
        base_url=settings.ollama_base_url,
        temperature=settings.temperature,
        num_ctx=settings.ollama_num_ctx,
        top_k=settings.ollama_top_k,
        num_gpu=settings.ollama_num_gpu,
        num_batch=settings.ollama_num_batch,
        num_predict=settings.ollama_num_predict,
    )
