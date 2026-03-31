from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import pickle
import shutil
import tempfile
import threading
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


def _worker_parse_pdf(
    pdf_path_str: str,
    parser_kwargs: dict,
    result_file: str,
    status_file: str,
    heartbeat_file: str,
    heartbeat_interval_seconds: int,
) -> None:
    hb_interval = max(1, int(heartbeat_interval_seconds))
    hb_stop = threading.Event()

    def _heartbeat_loop() -> None:
        hb_path = Path(heartbeat_file)
        hb_path.parent.mkdir(parents=True, exist_ok=True)
        while not hb_stop.is_set():
            try:
                hb_path.write_text(str(time.time()), encoding="utf-8")
            except Exception:
                pass
            hb_stop.wait(hb_interval)

    hb_thread = threading.Thread(target=_heartbeat_loop, daemon=True)
    hb_thread.start()
    try:
        parser = PdfParser(**parser_kwargs)
        blocks = parser.parse_doc([Path(pdf_path_str)])
        with open(result_file, "wb") as f:
            pickle.dump(blocks, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump({"ok": True}, f, ensure_ascii=False)
    except Exception as exc:  # pragma: no cover - subprocess path
        with open(status_file, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "ok": False,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                },
                f,
                ensure_ascii=False,
            )
    finally:
        hb_stop.set()
        try:
            hb_thread.join(timeout=1)
        except Exception:
            pass


def _get_vram_info_mb() -> tuple[int, int] | None:
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        return int(free_bytes // (1024 * 1024)), int(total_bytes // (1024 * 1024))
    except Exception:
        return None


def _force_cuda_cleanup() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        pass


def _terminate_then_kill_process(proc: mp.Process, pdf_name: str, reason: str) -> None:
    logger.warning(
        "Stopping MinerU subprocess for %s: %s. Sending terminate().",
        pdf_name,
        reason,
    )
    try:
        proc.terminate()
    except Exception:
        pass
    proc.join(15)
    if proc.is_alive():
        logger.warning(
            "MinerU subprocess still alive for %s after terminate. Sending kill().",
            pdf_name,
        )
        try:
            proc.kill()
        except Exception:
            pass
        proc.join(5)


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
    _force_cuda_cleanup()
    waited = 0
    while waited <= max_wait:
        _force_cuda_cleanup()
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
    temp_dir = Path(
        tempfile.mkdtemp(prefix="mineru_parse_", dir=str(Path(settings.chroma_dir).resolve()))
    )
    result_file = temp_dir / "blocks.pkl"
    status_file = temp_dir / "status.json"
    heartbeat_file = temp_dir / "heartbeat.txt"
    proc = ctx.Process(
        target=_worker_parse_pdf,
        args=(
            str(pdf_path),
            parser_kwargs,
            str(result_file),
            str(status_file),
            str(heartbeat_file),
            int(settings.mineru_parse_heartbeat_interval_seconds),
        ),
    )
    start_ts = time.time()
    try:
        proc.start()

        hard_timeout_s = int(settings.mineru_parse_subprocess_timeout_seconds)
        stall_timeout_s = max(0, int(settings.mineru_parse_stall_timeout_seconds))
        poll_s = max(1, int(settings.mineru_parse_wait_poll_seconds))
        timeout_label = "disabled" if hard_timeout_s <= 0 else f"{hard_timeout_s}s"
        logger.info(
            "MinerU parse wait mode for %s: hard_timeout=%s, stall_timeout=%ss, poll=%ss.",
            pdf_path.name,
            timeout_label,
            stall_timeout_s,
            poll_s,
        )

        while proc.is_alive():
            proc.join(timeout=poll_s)
            if not proc.is_alive():
                break
            elapsed = int(time.time() - start_ts)

            if hard_timeout_s > 0 and elapsed >= hard_timeout_s:
                _terminate_then_kill_process(
                    proc,
                    pdf_path.name,
                    f"hard timeout reached ({hard_timeout_s}s)",
                )
                if proc.is_alive():
                    raise TimeoutError(
                        f"MinerU subprocess did not stop for {pdf_path.name} after hard timeout terminate/kill (pid={proc.pid})"
                    )
                raise TimeoutError(
                    f"MinerU parse subprocess hard timeout for {pdf_path.name} after {hard_timeout_s}s (pid={proc.pid})"
                )

            if stall_timeout_s > 0:
                if heartbeat_file.exists():
                    hb_age = int(time.time() - heartbeat_file.stat().st_mtime)
                    if hb_age >= stall_timeout_s:
                        _terminate_then_kill_process(
                            proc,
                            pdf_path.name,
                            f"heartbeat stale for {hb_age}s",
                        )
                        if proc.is_alive():
                            raise TimeoutError(
                                f"MinerU subprocess did not stop for {pdf_path.name} after stall terminate/kill (pid={proc.pid})"
                            )
                        raise TimeoutError(
                            f"MinerU parse subprocess stalled for {pdf_path.name}: heartbeat stale {hb_age}s (pid={proc.pid})"
                        )
                elif elapsed >= stall_timeout_s:
                    _terminate_then_kill_process(
                        proc,
                        pdf_path.name,
                        f"no heartbeat file for {elapsed}s",
                    )
                    if proc.is_alive():
                        raise TimeoutError(
                            f"MinerU subprocess did not stop for {pdf_path.name} after no-heartbeat terminate/kill (pid={proc.pid})"
                        )
                    raise TimeoutError(
                        f"MinerU parse subprocess stalled for {pdf_path.name}: no heartbeat in {elapsed}s (pid={proc.pid})"
                    )

        elapsed = time.time() - start_ts
        logger.info(
            "MinerU parse subprocess finished for %s (exit_code=%s, elapsed=%.1fs).",
            pdf_path.name,
            proc.exitcode,
            elapsed,
        )

        if not status_file.exists():
            raise RuntimeError(
                f"MinerU parse subprocess returned no status for {pdf_path.name} (exit_code={proc.exitcode})."
            )
        status = json.loads(status_file.read_text(encoding="utf-8"))
        if not status.get("ok"):
            err = str(status.get("error", "unknown error"))
            tb = str(status.get("traceback", ""))
            raise RuntimeError(f"MinerU parse subprocess failed: {err}\n{tb}")
        if not result_file.exists():
            raise RuntimeError(
                f"MinerU parse subprocess returned no result file for {pdf_path.name}."
            )
        with open(result_file, "rb") as f:
            blocks = pickle.load(f)

        _force_cuda_cleanup()
        _wait_for_vram_release_after_subprocess(settings)
        return list(blocks or [])
    finally:
        if proc.is_alive():
            try:
                proc.kill()
            except Exception:
                pass
            proc.join(2)
        try:
            shutil.rmtree(temp_dir, ignore_errors=True)
        except Exception:
            pass


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


def _resolve_mineru_config_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute():
        return path
    return Path.home() / path


_MINERU_REQUIRED_PIPELINE_REL_PATHS = [
    "models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth",
    "models/Layout/PP-DocLayoutV2",
    "models/MFR/unimernet_hf_small_2503",
    "models/TabRec/SlanetPlus/slanet-plus.onnx",
]


def _is_valid_mineru_pipeline_root(root: Path) -> bool:
    if not root.exists() or not root.is_dir():
        return False
    for rel in _MINERU_REQUIRED_PIPELINE_REL_PATHS:
        if not (root / rel).exists():
            return False
    return True


def _iter_mineru_pipeline_candidates(settings: Settings) -> List[Path]:
    candidates: List[Path] = []
    configured = str(settings.mineru_local_pipeline_models_dir or "").strip()
    if configured:
        candidates.append(Path(configured))

    home = Path.home()
    candidates.append(home / ".cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1.0")
    candidates.append(home / ".cache/modelscope/hub/models/OpenDataLab/PDF-Extract-Kit-1___0")

    hf_snapshots_dir = home / ".cache/huggingface/hub/models--opendatalab--PDF-Extract-Kit-1.0/snapshots"
    if hf_snapshots_dir.exists():
        snapshots = sorted(
            [p for p in hf_snapshots_dir.iterdir() if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(snapshots)

    unique: List[Path] = []
    seen = set()
    for p in candidates:
        key = str(p.resolve()) if p.exists() else str(p)
        if key in seen:
            continue
        seen.add(key)
        unique.append(p)
    return unique


def _resolve_valid_mineru_pipeline_root(settings: Settings) -> Path:
    checked: List[str] = []
    for candidate in _iter_mineru_pipeline_candidates(settings):
        checked.append(str(candidate))
        if _is_valid_mineru_pipeline_root(candidate):
            return candidate
    checked_preview = "; ".join(checked[:8])
    raise RuntimeError(
        "No valid local MinerU pipeline model root found. "
        "Checked paths: "
        f"{checked_preview}. "
        "Expected at least: "
        "models/OCR/paddleocr_torch/ch_PP-OCRv4_rec_server_doc_infer.pth"
    )


def _ensure_mineru_local_models_config(settings: Settings) -> None:
    """
    Ensure MinerU local model config exists when MINERU_MODEL_SOURCE=local.
    """
    os.environ["MINERU_MODEL_SOURCE"] = settings.mineru_model_source
    config_path = _resolve_mineru_config_path(settings.mineru_tools_config_json)
    os.environ["MINERU_TOOLS_CONFIG_JSON"] = str(config_path)

    if settings.mineru_model_source != "local":
        return

    pipeline_root = _resolve_valid_mineru_pipeline_root(settings)
    pipeline_dir = str(pipeline_root).replace("\\", "/")
    configured = str(settings.mineru_local_pipeline_models_dir or "").strip()
    if configured and Path(configured).resolve() != pipeline_root.resolve():
        logger.warning(
            "Configured MINERU_LOCAL_PIPELINE_MODELS_DIR is invalid. Using detected fallback: %s",
            pipeline_dir,
        )

    cfg: dict = {}
    if config_path.exists():
        try:
            cfg = json.loads(config_path.read_text(encoding="utf-8"))
            if not isinstance(cfg, dict):
                cfg = {}
        except Exception:
            cfg = {}

    models_dir = cfg.get("models-dir")
    if not isinstance(models_dir, dict):
        models_dir = {}
    models_dir["pipeline"] = pipeline_dir
    vlm_dir = str(settings.mineru_local_vlm_models_dir or "").strip()
    if vlm_dir:
        models_dir["vlm"] = vlm_dir
    cfg["models-dir"] = models_dir

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("MinerU local config prepared: %s", config_path)


def prepare_index(
    settings: Settings,
    force: bool = False,
    checkpoint_enabled: bool = False,
    checkpoint_file: str | None = None,
) -> None:
    """Build and persist Chroma index with optional source-level checkpointing."""
    ensure_directories(settings)
    _ensure_mineru_local_models_config(settings)

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
