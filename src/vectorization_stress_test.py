from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from settings import Settings
from utils import get_logger
from vectorstore import persist_documents

logger = get_logger("vectorization_stress_test")


def _load_tripled_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # One original + two copies.
    return "\n\n".join([text, text, text])


def _build_docs(text: str) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=300)
    chunks = splitter.split_text(text)
    docs: list[Document] = []
    for idx, chunk in enumerate(chunks):
        docs.append(
            Document(
                page_content=chunk,
                metadata={"source": "vectorization_stress_test", "chunk_id": idx},
            )
        )
    return docs


def main() -> None:
    parser = argparse.ArgumentParser(description="Stress test for embedding/vectorization path.")
    parser.add_argument(
        "--input",
        type=str,
        default=(
            r"C:\python\rag_textbook\output\Dayzenrot_Feyzal_On_Matematika_v_mashinnom_"
            r"obuchen_241126_230954\auto\Dayzenrot_Feyzal_On_Matematika_v_mashinnom_"
            r"obuchen_241126_230954.md"
        ),
        help="Path to source markdown file.",
    )
    parser.add_argument(
        "--chroma-dir",
        type=str,
        default="chroma_db_stress",
        help="Separate Chroma directory for this stress test.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Clear target Chroma directory before adding stress docs.",
    )
    args = parser.parse_args()

    source_path = Path(args.input)
    if not source_path.exists():
        raise FileNotFoundError(f"Input file not found: {source_path}")

    settings = Settings()
    settings.chroma_dir = Path(args.chroma_dir)

    tripled_text = _load_tripled_text(source_path)
    docs = _build_docs(tripled_text)
    logger.info("Stress test chunks prepared: %s", len(docs))

    persist_documents(docs, settings, force=bool(args.force))
    logger.info("Stress test indexing finished. Chroma dir: %s", settings.chroma_dir)


if __name__ == "__main__":
    load_dotenv()
    main()
