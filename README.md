# rag_textbook

Local RAG project with:
- LangChain + ChromaDB
- Hybrid retrieval (dense embeddings + BM25 sparse fusion)
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
|   |-- retriever.py
|   |-- chat_history.py
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

Checkpoint mode (resume by source file after failures) is optional in `ingest.py`:

```bash
python src/ingest.py --checkpoint
python src/ingest.py --checkpoint-file chroma_db/ingest_checkpoint.json
python src/ingest.py --force --checkpoint
```

Default checkpoint path can be configured via `INGEST_CHECKPOINT_FILE` in `.env`.

Optional graph write during indexing (safe off by default):
- `NEO4J_ENABLED=1`
- `GRAPH_WRITE_ENABLED=1`
- `GRAPH_RELATIONS_ENABLED=1` (write `MENTIONS`/`CO_OCCURS`, set `0` to write only passages)
- `GRAPH_ENTITY_MAX_PER_PASSAGE=20` (caps extracted entity terms per chunk)
- `GRAPH_COOCCURS_MAX_PER_PASSAGE=200` (caps co-occurrence pairs per chunk)
- `NEO4J_URI`, `NEO4J_USER`, `NEO4J_PASSWORD`, `NEO4J_DATABASE`
- `GRAPH_FAIL_ON_ERROR=0` (keep ingest running even if graph write fails)

When enabled, `ingest.py` upserts `(:Passage)` nodes to Neo4j alongside Chroma indexing.
It links `(:Source)-[:HAS_PASSAGE]->(:Passage)` and sequential `(:Passage)-[:NEXT]->(:Passage)` per source.
If `GRAPH_RELATIONS_ENABLED=1`, it also creates `(:Passage)-[:MENTIONS]->(:Entity)` and `(:Entity)-[:CO_OCCURS {source_id, weight}]->(:Entity)`.

For large corpora / GPU stability during embedding, tune:
- `EMBEDDINGS_BACKEND=ollama|sentence` (default: `ollama` for offline local embeddings)
- `OLLAMA_EMBED_MODEL` (local Ollama embedding model, default `qwen3-embedding:0.6b`)
- `EMBED_BATCH_SIZE` (batch size inside sentence-transformers encode)
- `CHROMA_ADD_BATCH_SIZE` (how many chunks are sent per `add_documents` call)
- `EMBED_DEVICE=auto|cpu|cuda` (embedding device policy)
- `EMBED_MIN_FREE_VRAM_MB` (minimum free VRAM target before embedding)
- `EMBED_MIN_FREE_VRAM_RATIO` (required free VRAM ratio, default `0.7`)
- `EMBED_POST_UNLOAD_WAIT_SECONDS=180` (wait window for delayed VRAM release after unloading LLM)
- `EMBED_POST_UNLOAD_POLL_SECONDS=5` (VRAM recheck interval during wait window)
- `EMBED_FORCE_CPU_ON_LOW_VRAM=1` (fallback to CPU when VRAM remains low)
- `EMBED_PROBE_LLM_BEFORE_INDEX=1` (run LLM ping before VRAM check/unload)
- `MINERU_PARSE_IN_SUBPROCESS=1` (force MinerU PDF parse to run in a child process; process exit force-releases CUDA context)
- `MINERU_PARSE_SUBPROCESS_TIMEOUT_SECONDS=0` (optional hard timeout for child parse, `0` disables timeout)
- `MINERU_GPU_RELEASE_WAIT_SECONDS=60` (max wait after child exit to verify GPU release)
- `MINERU_GPU_RELEASE_POLL_SECONDS=5` (poll interval for post-exit GPU checks)
- `MINERU_GPU_RELEASE_TARGET_FREE_VRAM_MB=3000` (target free VRAM threshold during post-exit checks)
- `MINERU_POST_RELEASE_WAIT_SECONDS=120` (mandatory pause between MinerU parse and next stage)
- `MINERU_POST_RELEASE_POLL_SECONDS=5` (poll interval for MinerU unload checks)
- `MINERU_RELEASE_CHECK_ENABLED=1` (verify MinerU singleton caches are unloaded during barrier)

If CUDA errors appear on indexing, reduce both values first (e.g. `EMBED_BATCH_SIZE=16`, `CHROMA_ADD_BATCH_SIZE=64`).

For fully offline embedding flow with Ollama, pull embedding model once:

```bash
ollama pull qwen3-embedding:0.6b
```

Stress test for vectorization path (tripled markdown + split 1200/300):

```bash
python src/vectorization_stress_test.py --force
```

## Retrieval Configuration

Hybrid retrieval is enabled by default and does not change CLI commands.

Set in `.env`:
- `CONVERSATIONAL_RAG_ENABLED=1` (set `0` for stateless mode by default)
- `OLLAMA_THINK=auto|0|1` (`auto` disables think for `qwen3*` and `deepseek-r1*` on query generation to avoid empty outputs)
- `OLLAMA_REQUEST_TIMEOUT_SECONDS=0|N` (`0` = model-aware auto timeout, uses 600s for `deepseek-r1*`, 180s otherwise)
- `CHAT_HISTORY_DIR=chat_history` (JSONL storage for sessions)
- `CHAT_SESSION_ID=default` (default conversation id)
- `CHAT_HISTORY_MAX_TURNS=6` (how many latest turns are sent to LLM)
- `RETRIEVER_MODE=hybrid` (`dense` for dense-only mode)
- `TOP_K=4` (final number of retrieved chunks)
- `HYBRID_SPARSE_K=8` (BM25 candidate pool size)
- `HYBRID_DENSE_WEIGHT=0.6`
- `HYBRID_SPARSE_WEIGHT=0.4`
- `HYBRID_RRF_K=60` (reciprocal-rank-fusion smoothing)
- `GRAPH_RAG_ENABLED=0` / `GRAPH_RETRIEVER_ENABLED=0` (reserved feature flags, no behavior change while disabled)

## Query the RAG System

```bash
python src/query.py "Your question here"
```

Conversation controls (same command, optional flags):

```bash
python src/query.py "How does gradient descent work?" --session-id ml_course
python src/query.py "How to implement it in Python?" --session-id ml_course
python src/query.py "One-off question" --stateless
python src/query.py "Reset session and ask again" --session-id ml_course --clear-history
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

If your eval model is `qwen3:*` or `deepseek-r1:*` and metrics fail with `Invalid JSON ... input_value=''`, use non-thinking eval mode for structured metrics:
- `OLLAMA_EVAL_THINK=0` (or `auto`, which sets `think=0` for `qwen3:*` and `deepseek-r1:*`)
- `OLLAMA_EVAL_JSON_RETRY_ATTEMPTS=1` (one extra structured retry in think mode)
- `OLLAMA_EVAL_RETRY_NUM_PREDICT_MULTIPLIER=1.5` (increases `num_predict` on retry)
- `OLLAMA_EVAL_MAX_NUM_PREDICT=512` (caps growth to avoid timeout spikes)
- `OLLAMA_EVAL_STRUCTURED_RECOVERY=1` (runs one JSON-normalization pass from content)
- `OLLAMA_EVAL_STRUCTURED_RECOVERY_INPUT_CHARS=6000` (limits recovery prompt size)
- optional last resort: `OLLAMA_EVAL_JSON_RETRY_WITHOUT_THINK=1`

## Tracing (Arize Phoenix)

Set in `.env`:
- `PHOENIX_ENDPOINT` (default: `http://127.0.0.1:4317`)
- `PHOENIX_PROTOCOL` (default: `grpc`)
- `PHOENIX_PROJECT_NAME` (default: `rag_eval`)

Make sure Phoenix collector is running before query/evaluation.

## Notes

- This repository uses DeepEval for evaluation. Legacy Ragas demo content is excluded from git.
- Local models, vector DB, and generated artifacts are ignored via `.gitignore`.
