from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from chunker import LayoutAwareChunker
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
    blocks = parser.parse_doc([pdf_path])
    chunker.images_root = Path(settings.markdown_dir) / pdf_path.stem / "auto"
    chunks = chunker.chunk(blocks, doc_name=pdf_path.name, doc_format="pdf")

    docs: List[Document] = []
    for ch in chunks:
        meta = _sanitize_metadata(dict(ch))
        meta.setdefault("source", str(pdf_path))
        docs.append(Document(page_content=ch["text"], metadata=meta))

    logger.info("Chunks from PDF %s: %s", pdf_path.name, len(docs))
    return docs


def _load_markdown_chunks_for_file(
    splitter: RecursiveCharacterTextSplitter,
    txt_path: Path,
) -> List[Document]:
    loader = TextLoader(str(txt_path), encoding="utf-8")
    raw_docs = loader.load()
    chunks = splitter.split_documents(raw_docs)
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

    parser = PdfParser(output_dir=settings.markdown_dir)
    chunker = LayoutAwareChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
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
