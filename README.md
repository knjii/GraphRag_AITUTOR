# rag_textbook

Local RAG project with:
- LangChain + ChromaDB
- Ollama for generation/evaluation models
- DeepEval metrics for RAG evaluation
- Arize Phoenix tracing (OpenInference spans)

## Project Structure

```text
rag_textbook/
|-- src/
|   |-- ingest.py
|   |-- query.py
|   |-- deepeval_eval.py
|   |-- rag_chain.py
|   |-- chunker.py
|   |-- embeddings.py
|   |-- llm.py
|   |-- vectorstore.py
|   |-- settings.py
|   `-- utils.py
|-- documents/
|   |-- pdf_docs/          # knowledge base PDFs (ignored in git)
|   `-- markdown_docs/     # knowledge base MD/TXT (ignored in git)
|-- deepeval_artifacts/
|   `-- rag_eval_inputs.json
|-- .env.example
|-- requirements.txt
`-- README.md
```

## Environment Setup

```bash
conda create -n rag_test python=3.11 -y
conda activate rag_test
pip install -r requirements.txt
```

Create local env config:

```bash
cp .env.example .env
```

PowerShell alternative:

```powershell
Copy-Item .env.example .env
```

Then update paths and model names in `.env` for your machine.

## Knowledge Base Organization

Put source files here:
- `documents/pdf_docs/` for PDF files
- `documents/markdown_docs/` for Markdown and text files

These folders are intentionally excluded from git, so each user keeps their own dataset locally.

## Build/Refresh Vector Index

Run from repository root:

```bash
python src/ingest.py
```

This reads files from `PDF_DIR` / `MARKDOWN_DIR` and writes Chroma index to `CHROMA_DIR`.

## Query the RAG System

```bash
python src/query.py "Your question here"
```

The query command runs the same RAG chain and emits tracing spans to Phoenix.

## RAG Evaluation (DeepEval)

Default dataset:
- `deepeval_artifacts/rag_eval_inputs.json`

Smoke test:

```bash
python src/deepeval_eval.py --max-rows 1 --output-prefix deepeval_smoke
```

Full run:

```bash
python src/deepeval_eval.py --dataset deepeval_artifacts/rag_eval_inputs.json --output-prefix deepeval
```

Used metrics:
- `AnswerRelevancyMetric`
- `FaithfulnessMetric`
- `ContextualPrecisionMetric`

Outputs are saved to `deepeval_artifacts/` using the selected prefix.

## Tracing (Arize Phoenix)

Set in `.env`:
- `PHOENIX_ENDPOINT` (default: `http://127.0.0.1:4317`)
- `PHOENIX_PROTOCOL` (default: `grpc`)
- `PHOENIX_PROJECT_NAME` (default: `rag_eval`)

Make sure Phoenix collector is running before query/evaluation.

## Notes

- This repository uses DeepEval for evaluation. Legacy Ragas demo content is excluded from git.
- Local models, vector DB, and generated artifacts are ignored via `.gitignore`.
