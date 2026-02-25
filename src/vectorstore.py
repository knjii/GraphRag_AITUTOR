from pathlib import Path
from typing import List, Optional
import shutil

from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma

from embeddings import get_embeddings_model
from settings import Settings
from utils import chroma_has_data, get_logger

logger = get_logger("vectorstore")


def _clear_chroma_dir(chroma_path: Path) -> None:
    if chroma_path.exists():
        logger.info("Clearing existing Chroma directory: %s", chroma_path)
        shutil.rmtree(chroma_path, ignore_errors=True)
    chroma_path.mkdir(parents=True, exist_ok=True)


def persist_documents(documents: List[Document], settings: Settings, force: bool = False) -> None:
    """Persist documents into Chroma. If force is True, deletes existing index first."""
    chroma_path = Path(settings.chroma_dir)
    embeddings = get_embeddings_model(settings)

    if force:
        _clear_chroma_dir(chroma_path)

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(chroma_path),
    )


def load_vector_store(settings: Settings) -> Optional[Chroma]:
    """Load an existing Chroma store if available."""
    if not chroma_has_data(settings):
        logger.warning("No existing Chroma data at %s", settings.chroma_dir)
        return None

    embeddings = get_embeddings_model(settings)
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=embeddings)
