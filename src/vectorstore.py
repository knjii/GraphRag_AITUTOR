from pathlib import Path
from typing import List, Optional
import shutil
import time

from langchain_core.documents import Document
# from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
import torch

from embeddings import get_embeddings_model
from settings import Settings
from utils import chat_with_ollama, chroma_has_data, get_logger

logger = get_logger("vectorstore")


def _clear_chroma_dir(chroma_path: Path) -> None:
    if chroma_path.exists():
        logger.info("Clearing existing Chroma directory: %s", chroma_path)
        shutil.rmtree(chroma_path, ignore_errors=True)
    chroma_path.mkdir(parents=True, exist_ok=True)


def _is_cuda_embedding_failure(exc: Exception) -> bool:
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "cuda error",
            "cublas_status_execution_failed",
            "out of memory",
            "cuda out of memory",
        )
    )


def _get_vram_info_mb() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))


def _ensure_embedding_memory(settings: Settings) -> bool:
    """
    Ensure enough free VRAM for embeddings.
    Returns True when CPU fallback should be used for embeddings.
    """
    if settings.embed_device == "cpu":
        logger.info("Embeddings forced to CPU by EMBED_DEVICE=cpu.")
        return True

    if settings.embed_device == "cuda":
        logger.info("Embeddings forced to CUDA by EMBED_DEVICE=cuda.")
        return False

    if settings.embed_probe_llm_before_index:
        try:
            _ = chat_with_ollama(
                message="Ping before embedding index run.",
                settings=settings,
                options={"num_predict": 8, "num_ctx": 256},
            )
        except Exception as exc:
            logger.warning("LLM warmup probe failed (non-fatal): %s", exc)

    vram_before = _get_vram_info_mb()
    if vram_before is None:
        return False
    free_before, total_vram = vram_before

    min_free_abs = max(256, int(settings.embed_min_free_vram_mb))
    min_free_ratio = max(0.0, min(float(settings.embed_min_free_vram_ratio), 1.0))
    min_free = max(min_free_abs, int(total_vram * min_free_ratio))
    if free_before >= min_free:
        logger.info(
            "Free VRAM before embedding: %s MB (total=%s MB, threshold=%s MB)",
            free_before,
            total_vram,
            min_free,
        )
        return False

    logger.warning(
        "Low free VRAM before embedding: %s MB (target >= %s MB). Unloading LLM...",
        free_before,
        min_free,
    )
    try:
        chat_with_ollama(turn_off=True, settings=settings)
    except Exception as exc:
        logger.warning("Failed to unload LLM (non-fatal): %s", exc)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        try:
            torch.cuda.ipc_collect()
        except Exception:
            pass

    time.sleep(1.0)
    vram_after = _get_vram_info_mb()
    if vram_after is None:
        return False
    free_after, _ = vram_after

    logger.info("Free VRAM after LLM unload: %s MB", free_after)
    if free_after >= min_free:
        return False

    if settings.embed_force_cpu_on_low_vram:
        logger.warning(
            "VRAM still low (%s MB < %s MB). Falling back to CPU embeddings.",
            free_after,
            min_free,
        )
        return True

    return False


def _add_documents_adaptive(store: Chroma, docs: List[Document]) -> None:
    if not docs:
        return
    try:
        store.add_documents(docs)
        return
    except RuntimeError as exc:
        if not _is_cuda_embedding_failure(exc) or len(docs) == 1:
            raise
        mid = len(docs) // 2
        logger.warning(
            "CUDA embedding failure on batch of %s docs. Retrying as %s + %s.",
            len(docs),
            mid,
            len(docs) - mid,
        )
        _add_documents_adaptive(store, docs[:mid])
        _add_documents_adaptive(store, docs[mid:])


def persist_documents(documents: List[Document], settings: Settings, force: bool = False) -> None:
    """Persist documents into Chroma. If force is True, deletes existing index first."""
    chroma_path = Path(settings.chroma_dir)
    use_cpu_embeddings = _ensure_embedding_memory(settings)
    embeddings = get_embeddings_model(settings, force_cpu=use_cpu_embeddings)
    add_batch_size = max(1, int(settings.chroma_add_batch_size))

    if force:
        _clear_chroma_dir(chroma_path)

    store = Chroma(
        persist_directory=str(chroma_path),
        embedding_function=embeddings,
    )
    for start in range(0, len(documents), add_batch_size):
        end = min(start + add_batch_size, len(documents))
        batch = documents[start:end]
        _add_documents_adaptive(store, batch)
        logger.info("Chroma add_documents progress: %s/%s", end, len(documents))


def load_vector_store(settings: Settings) -> Optional[Chroma]:
    """Load an existing Chroma store if available."""
    if not chroma_has_data(settings):
        logger.warning("No existing Chroma data at %s", settings.chroma_dir)
        return None

    embeddings = get_embeddings_model(settings)
    return Chroma(persist_directory=str(settings.chroma_dir), embedding_function=embeddings)
