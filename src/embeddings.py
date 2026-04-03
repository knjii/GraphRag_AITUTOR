try:
    from langchain_ollama import OllamaEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import OllamaEmbeddings  # type: ignore

try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from settings import Settings


def get_embeddings_model(settings: Settings, force_cpu: bool = False):
    """Return embeddings model based on configuration."""
    backend = str(settings.embedding_backend or "ollama").strip().lower()

    if backend == "ollama":
        # Prefer explicit Ollama runtime params to avoid silent CPU fallback.
        base_kwargs = {
            "model": settings.ollama_embed_model,
            "base_url": settings.ollama_base_url,
        }
        tuned_kwargs = {
            "num_ctx": int(settings.ollama_num_ctx),
            "num_gpu": int(settings.ollama_num_gpu),
            "num_thread": int(settings.n_threads),
        }
        try:
            return OllamaEmbeddings(**base_kwargs, **tuned_kwargs)
        except Exception:
            # Keep backward compatibility for wrappers that don't accept tuned kwargs.
            return OllamaEmbeddings(**base_kwargs)

    model_kwargs = {}
    if force_cpu:
        model_kwargs["device"] = "cpu"
    elif settings.embed_device in {"cpu", "cuda"}:
        model_kwargs["device"] = settings.embed_device

    return HuggingFaceEmbeddings(
        model_name=settings.embedding_model_name,
        model_kwargs=model_kwargs,
        encode_kwargs={"batch_size": max(1, int(settings.embed_batch_size))},
    )
