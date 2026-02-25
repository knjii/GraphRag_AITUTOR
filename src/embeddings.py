try:
    from langchain_huggingface import HuggingFaceEmbeddings  # type: ignore
except Exception:
    from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore
from settings import Settings


def get_embeddings_model(settings: Settings):
    """Return embeddings model based on configuration."""
    return HuggingFaceEmbeddings(model_name=settings.embedding_model_name)

