from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunker import LayoutAwareChunker
from neo4j_client import write_passages_to_graph
from pdf_parser import PdfParser
from settings import Settings
from utils import ensure_directories, get_logger
from vectorstore import persist_documents

logger = get_logger("ingest")


def _now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _source_key(source_type: str, path: Path) -> str:
    return f"{source_type}:{path.resolve().as_posix()}"


def _default_checkpoint_path(settings: Settings) -> Path:
    path = Path(settings.ingest_checkpoint_file)
    if path.is_absolute():
        return path
    return Path(path)


def _load_checkpoint(path: Path) -> Dict:
    if not path.exists():
        return {"version": 1, "updated_at": _now_utc_iso(), "sources": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {"version": 1, "updated_at": _now_utc_iso(), "sources": {}}
    if not isinstance(data, dict):
        data = {}
    data.setdefault("version", 1)
    data.setdefault("updated_at", _now_utc_iso())
    data.setdefault("sources", {})
    if not isinstance(data["sources"], dict):
        data["sources"] = {}
    return data


def _save_checkpoint(path: Path, checkpoint: Dict) -> None:
    checkpoint["updated_at"] = _now_utc_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(checkpoint, ensure_ascii=False, indent=2), encoding="utf-8")


def _mark_checkpoint(
    checkpoint: Dict,
    source_type: str,
    source_path: Path,
    status: str,
    chunks: int,
    error: str = "",
) -> None:
    key = _source_key(source_type, source_path)
    checkpoint["sources"][key] = {
        "status": status,
        "chunks": int(chunks),
        "updated_at": _now_utc_iso(),
        "error": str(error or ""),
    }


def _load_pdf_chunks_for_file(
    parser: PdfParser,
    chunker: LayoutAwareChunker,
    settings: Settings,
    pdf_path: Path,
) -> List[Document]:
    logger.info("Parsing PDF: %s", pdf_path)
    if settings.mineru_parse_in_subprocess:
        blocks = _parse_pdf_in_subprocess(settings, pdf_path)
    else:
        blocks = parser.parse_doc([pdf_path])
    chunker.images_root = Path(settings.markdown_dir) / pdf_path.stem / "auto"
    chunks = chunker.chunk(blocks, doc_name=pdf_path.name, doc_format="pdf")

    docs: List[Document] = []
    for idx, ch in enumerate(chunks):
        meta = _sanitize_metadata(dict(ch))
        meta.setdefault("source", str(pdf_path))
        chunk_id = str(meta.get("chunk_id", "") or "").strip()
        if not chunk_id:
            meta["chunk_id"] = f"{pdf_path.stem}:{idx}"
        docs.append(Document(page_content=ch["text"], metadata=meta))

    logger.info("Chunks from PDF %s: %s", pdf_path.name, len(docs))
    return docs


def _worker_parse_pdf(pdf_path_str: str, parser_kwargs: dict, queue) -> None:
    try:
        parser = PdfParser(**parser_kwargs)
        blocks = parser.parse_doc([Path(pdf_path_str)])
        queue.put({"ok": True, "blocks": blocks})
    except Exception as exc:  # pragma: no cover - subprocess path
        queue.put(
            {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )


def _get_vram_info_mb() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))
    except Exception:
        return None


def _wait_for_vram_release_after_subprocess(settings: Settings) -> None:
    max_wait = max(0, int(settings.mineru_gpu_release_wait_seconds))
    poll = max(1, int(settings.mineru_gpu_release_poll_seconds))
    target_free = max(0, int(settings.mineru_gpu_release_target_free_vram_mb))
    if max_wait <= 0:
        return

    logger.info(
        "MinerU subprocess exited. Waiting up to %ss for GPU release (poll=%ss, target_free=%s MB).",
        max_wait,
        poll,
        target_free,
    )
    waited = 0
    while waited <= max_wait:
        vram = _get_vram_info_mb()
        if vram is None:
            logger.info("GPU release check: CUDA unavailable, skipping.")
            return
        free_mb, total_mb = vram
        logger.info(
            "GPU release check: free_vram=%s/%s MB (%ss/%ss).",
            free_mb,
            total_mb,
            waited,
            max_wait,
        )
        if target_free > 0 and free_mb >= target_free:
            logger.info("GPU release target reached: %s MB >= %s MB.", free_mb, target_free)
            return
        if waited >= max_wait:
            logger.warning(
                "GPU release wait timeout reached (%ss). Continuing pipeline.",
                max_wait,
            )
            return
        sleep_s = min(poll, max_wait - waited)
        time.sleep(sleep_s)
        waited += sleep_s


def _parse_pdf_in_subprocess(settings: Settings, pdf_path: Path) -> List[dict]:
    parser_kwargs = {
        "output_dir": settings.markdown_dir,
        "post_release_wait_seconds": 0,
        "post_release_poll_seconds": 1,
        "release_check_enabled": False,
    }
    ctx = mp.get_context("spawn")
    queue = ctx.Queue(maxsize=1)
    proc = ctx.Process(
        target=_worker_parse_pdf,
        args=(str(pdf_path), parser_kwargs, queue),
    )
    start_ts = time.time()
    proc.start()

    timeout_s = max(0, int(settings.mineru_parse_subprocess_timeout_seconds))
    proc.join(timeout=timeout_s if timeout_s > 0 else None)
    if proc.is_alive():
        proc.terminate()
        proc.join(10)
        raise TimeoutError(
            f"MinerU parse subprocess timeout for {pdf_path.name} after {timeout_s}s"
        )

    elapsed = time.time() - start_ts
    logger.info(
        "MinerU parse subprocess finished for %s (exit_code=%s, elapsed=%.1fs).",
        pdf_path.name,
        proc.exitcode,
        elapsed,
    )

    if queue.empty():
        raise RuntimeError(
            f"MinerU parse subprocess returned no data for {pdf_path.name} (exit_code={proc.exitcode})."
        )
    result = queue.get()
    if not result.get("ok"):
        err = str(result.get("error", "unknown error"))
        tb = str(result.get("traceback", ""))
        raise RuntimeError(f"MinerU parse subprocess failed: {err}\n{tb}")

    _wait_for_vram_release_after_subprocess(settings)
    return list(result.get("blocks") or [])


def _load_markdown_chunks_for_file(
    splitter: RecursiveCharacterTextSplitter,
    txt_path: Path,
) -> List[Document]:
    loader = TextLoader(str(txt_path), encoding="utf-8")
    raw_docs = loader.load()
    chunks = splitter.split_documents(raw_docs)
    for idx, doc in enumerate(chunks):
        chunk_id = str(doc.metadata.get("chunk_id", "") or "").strip()
        if not chunk_id:
            doc.metadata["chunk_id"] = f"{txt_path.stem}:{idx}"
    logger.info("Chunks from markdown/text %s: %s", txt_path.name, len(chunks))
    return chunks


def _iter_sources(settings: Settings) -> List[Tuple[str, Path]]:
    sources: List[Tuple[str, Path]] = []
    sources.extend(("pdf", p) for p in sorted(Path(settings.pdf_dir).rglob("*.pdf")))
    sources.extend(
        ("markdown", p)
        for p in sorted(Path(settings.markdown_dir).rglob("*"))
        if p.suffix.lower() in {".md", ".txt"}
    )
    return sources


def prepare_index(
    settings: Settings,
    force: bool = False,
    checkpoint_enabled: bool = False,
    checkpoint_file: str | None = None,
) -> None:
    """Build and persist Chroma index with optional source-level checkpointing."""
    ensure_directories(settings)

    checkpoint_path = Path(checkpoint_file) if checkpoint_file else _default_checkpoint_path(settings)
    checkpoint = _load_checkpoint(checkpoint_path) if checkpoint_enabled else {"sources": {}}

    if force and checkpoint_enabled:
        checkpoint = {"version": 1, "updated_at": _now_utc_iso(), "sources": {}}
        _save_checkpoint(checkpoint_path, checkpoint)

    parser = PdfParser(
        output_dir=settings.markdown_dir,
        post_release_wait_seconds=settings.mineru_post_release_wait_seconds,
        post_release_poll_seconds=settings.mineru_post_release_poll_seconds,
        release_check_enabled=settings.mineru_release_check_enabled,
    )
    chunker = LayoutAwareChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        use_llm=bool(settings.chunker_use_llm),
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    sources = _iter_sources(settings)
    if not sources:
        logger.warning("No documents found in %s or %s", settings.pdf_dir, settings.markdown_dir)
        return

    first_persist = True
    skipped = 0
    processed = 0
    failures: List[str] = []

    for source_type, source_path in sources:
        key = _source_key(source_type, source_path)
        state = checkpoint.get("sources", {}).get(key, {}) if checkpoint_enabled else {}

        if checkpoint_enabled and not force and state.get("status") == "done":
            skipped += 1
            logger.info("Skip (checkpoint done): %s", source_path)
            continue

        try:
            if source_type == "pdf":
                docs = _load_pdf_chunks_for_file(parser, chunker, settings, source_path)
            else:
                docs = _load_markdown_chunks_for_file(splitter, source_path)

            if docs:
                persist_documents(docs, settings, force=(force and first_persist))
                first_persist = False
                try:
                    write_passages_to_graph(docs, settings)
                except Exception as exc:
                    if settings.graph_fail_on_error:
                        raise
                    logger.warning("Neo4j graph write failed (non-fatal): %s", exc)
            _mark_checkpoint(checkpoint, source_type, source_path, "done", len(docs))
            processed += 1
        except Exception as exc:
            error_text = str(exc)
            logger.exception("Failed source: %s (%s)", source_path, error_text)
            _mark_checkpoint(checkpoint, source_type, source_path, "failed", 0, error=error_text)
            failures.append(f"{source_path}: {error_text}")
        finally:
            if checkpoint_enabled:
                _save_checkpoint(checkpoint_path, checkpoint)

    logger.info(
        "Indexing finished. processed=%s skipped=%s failed=%s checkpoint=%s",
        processed,
        skipped,
        len(failures),
        checkpoint_path if checkpoint_enabled else "disabled",
    )

    if failures:
        raise RuntimeError(
            "Some sources failed during indexing (processed data preserved). "
            f"First failure: {failures[0]}"
        )


def _sanitize_metadata(meta: dict) -> dict:
    """Convert complex metadata values to JSON strings for Chroma compatibility."""
    sanitized = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build/update vector index with optional checkpoint.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Recreate index from scratch before first persisted source.",
    )
    parser.add_argument(
        "--checkpoint",
        action="store_true",
        default=False,
        help="Enable source-level checkpointing (default: disabled).",
    )
    parser.add_argument(
        "--checkpoint-file",
        type=str,
        default=None,
        help="Custom checkpoint file path (default: <CHROMA_DIR>/ingest_checkpoint.json).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    load_dotenv()
    cfg = Settings()
    args = _parse_args()
    prepare_index(
        cfg,
        force=bool(args.force),
        checkpoint_enabled=bool(args.checkpoint),
        checkpoint_file=args.checkpoint_file,
    )
