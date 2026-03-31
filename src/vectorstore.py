from pathlib import Path
from typing import List, Optional
import os
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


def _is_sqlite_disk_io_error(exc: Exception) -> bool:
    text = str(exc).lower()
    return "disk i/o error" in text or "(code: 2570)" in text


def _cleanup_sqlite_recovery_files(chroma_path: Path) -> None:
    """
    Clean up SQLite sidecar files that may remain after interrupted runs.
    This can unblock Chroma initialization without touching the main DB file.
    """
    for name in ("chroma.sqlite3-journal", "chroma.sqlite3-wal", "chroma.sqlite3-shm"):
        p = chroma_path / name
        if not p.exists():
            continue
        try:
            os.remove(p)
            logger.warning("Removed stale SQLite sidecar file: %s", p)
        except Exception:
            # In restrictive environments delete can fail; truncation is enough for journal recovery.
            try:
                with open(p, "wb"):
                    pass
                logger.warning("Truncated stale SQLite sidecar file: %s", p)
            except Exception as cleanup_exc:
                logger.warning("Failed to clean SQLite sidecar %s: %s", p, cleanup_exc)


def _get_vram_info_mb() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))


def _wait_for_vram_recovery(
    settings: Settings,
    *,
    min_free_mb: int,
    initial_free_mb: int,
) -> tuple[int, bool]:
    """
    Wait for delayed VRAM release after LLM unload.
    Returns (latest_free_mb, reached_threshold).
    """
    max_wait = max(0, int(settings.embed_post_unload_wait_seconds))
    poll = max(1, int(settings.embed_post_unload_poll_seconds))
    if max_wait <= 0:
        return initial_free_mb, initial_free_mb >= min_free_mb

    logger.info(
        "Waiting up to %ss for VRAM recovery after LLM unload (poll=%ss, threshold=%s MB).",
        max_wait,
        poll,
        min_free_mb,
    )
    latest_free = initial_free_mb
    waited = 0
    while waited < max_wait:
        sleep_s = min(poll, max_wait - waited)
        time.sleep(sleep_s)
        waited += sleep_s
        vram_now = _get_vram_info_mb()
        if vram_now is None:
            return latest_free, False
        latest_free, _ = vram_now
        logger.info(
            "VRAM recheck after unload: %s MB free (%ss/%ss).",
            latest_free,
            waited,
            max_wait,
        )
        if latest_free >= min_free_mb:
            logger.info(
                "VRAM recovery reached threshold: %s MB >= %s MB.",
                latest_free,
                min_free_mb,
            )
            return latest_free, True
    return latest_free, latest_free >= min_free_mb


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

    vram_after = _get_vram_info_mb()
    if vram_after is None:
        return False
    free_after, _ = vram_after

    logger.info("Free VRAM right after LLM unload: %s MB", free_after)
    free_after, recovered = _wait_for_vram_recovery(
        settings,
        min_free_mb=min_free,
        initial_free_mb=free_after,
    )
    if recovered:
        return False

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


def _open_chroma_store(
    chroma_path: Path,
    embeddings,
    *,
    retry_on_disk_io: bool = True,
) -> Chroma:
    try:
        return Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
        )
    except Exception as exc:
        if not retry_on_disk_io or not _is_sqlite_disk_io_error(exc):
            raise
        logger.warning(
            "Chroma open failed with SQLite disk I/O error. Attempting one-time recovery."
        )
        _cleanup_sqlite_recovery_files(chroma_path)
        return Chroma(
            persist_directory=str(chroma_path),
            embedding_function=embeddings,
        )


def persist_documents(documents: List[Document], settings: Settings, force: bool = False) -> None:
    """Persist documents into Chroma. If force is True, deletes existing index first."""
    chroma_path = Path(settings.chroma_dir)
    use_cpu_embeddings = _ensure_embedding_memory(settings)
    embeddings = get_embeddings_model(settings, force_cpu=use_cpu_embeddings)
    add_batch_size = max(1, int(settings.chroma_add_batch_size))

    if force:
        _clear_chroma_dir(chroma_path)

    store = _open_chroma_store(chroma_path, embeddings, retry_on_disk_io=True)
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

    chroma_path = Path(settings.chroma_dir)
    embeddings = get_embeddings_model(settings)
    return _open_chroma_store(chroma_path, embeddings, retry_on_disk_io=True)
