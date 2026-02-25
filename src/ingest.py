from pathlib import Path
from typing import List
from dotenv import load_dotenv

import json

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader

from chunker import LayoutAwareChunker
from pdf_parser import PdfParser
from settings import Settings
from utils import get_logger, ensure_directories
from vectorstore import persist_documents

logger = get_logger("ingest")

def load_markdown_documents(settings: Settings) -> List[Document]:
    """Load text/markdown documents from configured folder."""
    docs: List[Document] = []
    for txt_path in Path(settings.markdown_dir).rglob("*"):
        if txt_path.suffix.lower() not in {".md", ".txt"}:
            continue
        loader = TextLoader(str(txt_path), encoding="utf-8")
        docs.extend(loader.load())
        logger.info("Loaded text/markdown: %s", txt_path)
    return docs


def load_pdf_chunks(settings: Settings) -> List[Document]:
    """Parse PDFs with layout-aware parser and chunker."""
    docs: List[Document] = []
    parser = PdfParser(output_dir=settings.markdown_dir)
    chunker = LayoutAwareChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    for pdf_path in Path(settings.pdf_dir).rglob("*.pdf"):
        logger.info("Parsing PDF: %s", pdf_path)
        blocks = parser.parse_doc([pdf_path])
        images_root = Path(settings.markdown_dir) / Path(pdf_path).stem / "auto"
        chunker.images_root = images_root
        chunks = chunker.chunk(blocks, doc_name=pdf_path.name, doc_format="pdf")
        for ch in chunks:
            meta = _sanitize_metadata(dict(ch))
            meta.setdefault("source", str(pdf_path))
            docs.append(Document(page_content=ch["text"], metadata=meta))
        logger.info("Chunks from PDF %s: %s", pdf_path.name, len(chunks))
    return docs


def split_documents(documents: List[Document], settings: Settings) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap
    )
    return splitter.split_documents(documents)


def prepare_index(settings: Settings, force: bool = False) -> None:
    """Build and persist the Chroma index. Use force=True to rebuild."""
    ensure_directories(settings)

    pdf_docs = load_pdf_chunks(settings)
    md_docs = load_markdown_documents(settings)
    documents = pdf_docs + md_docs
    if not documents:
        logger.warning("No documents found in %s or %s", settings.pdf_dir, settings.markdown_dir)
        return

    md_chunks = split_documents(md_docs, settings) if md_docs else []
    final_docs = pdf_docs + md_chunks
    persist_documents(final_docs, settings, force=force)
    logger.info("Index updated at %s", settings.chroma_dir)


def _sanitize_metadata(meta: dict) -> dict:
    """Convert complex metadata values to JSON strings for Chroma compatibility."""
    sanitized = {}
    for key, value in meta.items():
        if isinstance(value, (str, int, float, bool)) or value is None:
            sanitized[key] = value
        else:
            sanitized[key] = json.dumps(value, ensure_ascii=False)
    return sanitized


if __name__ == "__main__":
    load_dotenv()
    cfg = Settings()
    prepare_index(cfg, force=False)
