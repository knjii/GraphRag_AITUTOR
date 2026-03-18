import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load local environment variables before reading them below.
load_dotenv()


@dataclass
class Settings:
    """Central configuration for the RAG pipeline."""

    pdf_dir: Path = Path(os.getenv("PDF_DIR", "textbook"))
    markdown_dir: Path = Path(os.getenv("MARKDOWN_DIR", "markdown_docs"))
    chroma_dir: Path = Path(os.getenv("CHROMA_DIR", "chroma_db"))
    ingest_checkpoint_file: Path = Path(
        os.getenv("INGEST_CHECKPOINT_FILE", "chroma_db/ingest_checkpoint.json")
    )
    eval_pdf_dir: Path = Path(
        os.getenv("EVAL_PDF_DIR", r"C:\python\rag_textbook\documents\pdf_docs\test")
    )
    deepeval_artifacts_dir: Path = Path(os.getenv("DEEPEVAL_ARTIFACTS_DIR", "deepeval_artifacts"))

    # Local model paths
    llm_model_path: Path = Path(
        os.getenv("LLM_MODEL_PATH", r"C:\python\rag_textbook\models\LLM\model-q4_K.gguf")
    )
    embedding_model_path: Path = Path(
        os.getenv(
            "EMBED_MODEL_PATH",
            r"C:\python\rag_textbook\models\Embeddings\Qwen3-Embedding-0.6B-f16.gguf",
        )
    )

    # Embeddings backend metadata
    embedding_backend: str = os.getenv("EMBEDDINGS_BACKEND", "ollama")
    embedding_model_name: str = os.getenv("EMBED_MODEL_NAME", "Qwen/Qwen3-Embedding-0.6B")
    ollama_embed_model: str = os.getenv("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
    local_embedding_model_path: Path = Path(
        os.getenv("LOCAL_EMBED_PATH", r"C:\python\rag_textbook\models\Embeddings")
    )

    # Generic runtime params
    n_ctx: int = int(os.getenv("N_CTX", "4096"))
    embed_ctx: int = int(os.getenv("EMBED_CTX", "256"))  # used only when backend=llamacpp
    n_threads: int = int(os.getenv("N_THREADS", "4"))
    n_batch: int = int(os.getenv("N_BATCH", "1"))
    n_gpu_layers: int = int(os.getenv("N_GPU_LAYERS", "35"))
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    chroma_add_batch_size: int = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "256"))
    embed_device: str = os.getenv("EMBED_DEVICE", "auto").strip().lower()
    embed_min_free_vram_mb: int = int(os.getenv("EMBED_MIN_FREE_VRAM_MB", "2048"))
    embed_min_free_vram_ratio: float = float(os.getenv("EMBED_MIN_FREE_VRAM_RATIO", "0.7"))
    embed_force_cpu_on_low_vram: bool = os.getenv("EMBED_FORCE_CPU_ON_LOW_VRAM", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    embed_probe_llm_before_index: bool = os.getenv("EMBED_PROBE_LLM_BEFORE_INDEX", "1").lower() not in {
        "0",
        "false",
        "no",
    }

    # Ollama runtime params (for chat LLM)
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    ollama_eval_model: str = os.getenv(
        "OLLAMA_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    )
    ollama_top_k: int = int(os.getenv("OLLAMA_TOP_K", "4"))
    ollama_num_ctx: int = int(os.getenv("OLLAMA_NUM_CTX", "4096"))
    ollama_num_gpu: int = int(os.getenv("OLLAMA_NUM_GPU", "30"))
    ollama_num_batch: int = int(os.getenv("OLLAMA_NUM_BATCH", "1"))
    ollama_num_predict: int = int(os.getenv("OLLAMA_NUM_PREDICT", "256"))
    ollama_request_timeout_seconds: int = int(os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "0"))
    ollama_think: str = os.getenv("OLLAMA_THINK", "auto").strip().lower()
    ollama_eval_num_predict: int = int(os.getenv("OLLAMA_EVAL_NUM_PREDICT", "256"))
    ollama_eval_temperature: float = float(os.getenv("OLLAMA_EVAL_TEMPERATURE", "0.0"))
    ollama_eval_think: str = os.getenv("OLLAMA_EVAL_THINK", "auto").strip().lower()
    ollama_eval_json_retry_without_think: bool = os.getenv(
        "OLLAMA_EVAL_JSON_RETRY_WITHOUT_THINK", "0"
    ).lower() not in {"0", "false", "no"}
    ollama_eval_json_retry_attempts: int = int(os.getenv("OLLAMA_EVAL_JSON_RETRY_ATTEMPTS", "2"))
    ollama_eval_retry_num_predict_multiplier: float = float(
        os.getenv("OLLAMA_EVAL_RETRY_NUM_PREDICT_MULTIPLIER", "2.0")
    )
    ollama_eval_max_num_predict: int = int(os.getenv("OLLAMA_EVAL_MAX_NUM_PREDICT", "512"))
    ollama_eval_structured_recovery: bool = os.getenv(
        "OLLAMA_EVAL_STRUCTURED_RECOVERY", "1"
    ).lower() not in {"0", "false", "no"}
    ollama_eval_structured_recovery_input_chars: int = int(
        os.getenv("OLLAMA_EVAL_STRUCTURED_RECOVERY_INPUT_CHARS", "6000")
    )

    # Eval runtime controls
    ragas_timeout: int = int(os.getenv("RAGAS_TIMEOUT", "600"))
    ragas_max_retries: int = int(os.getenv("RAGAS_MAX_RETRIES", "3"))
    ragas_max_wait: int = int(os.getenv("RAGAS_MAX_WAIT", "180"))
    ragas_max_workers: int = int(os.getenv("RAGAS_MAX_WORKERS", "1"))
    ragas_debug: bool = os.getenv("RAGAS_DEBUG", "1").lower() not in {"0", "false", "no"}
    ragas_log_samples: int = int(os.getenv("RAGAS_LOG_SAMPLES", "3"))
    ragas_separate_metrics: bool = os.getenv("RAGAS_SEPARATE_METRICS", "1").lower() not in {
        "0",
        "false",
        "no",
    }

    chunk_size: int = int(os.getenv("CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "400"))
    top_k: int = int(os.getenv("TOP_K", "4"))
    conversational_rag_enabled: bool = os.getenv("CONVERSATIONAL_RAG_ENABLED", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    chat_history_dir: Path = Path(os.getenv("CHAT_HISTORY_DIR", "chat_history"))
    chat_session_id: str = str(os.getenv("CHAT_SESSION_ID", "default"))
    chat_history_max_turns: int = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "6"))
    retriever_mode: str = os.getenv("RETRIEVER_MODE", "hybrid")
    hybrid_sparse_k: int = int(os.getenv("HYBRID_SPARSE_K", "8"))
    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
    hybrid_sparse_weight: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.4"))
    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", "60"))
    chunker_use_llm: bool = os.getenv("CHUNKER_USE_LLM", "1").lower() not in {"0", "false", "no"}

    contextualize_q_system_prompt: str = str(
        os.getenv(
            "CONTEXTUALIZE_Q_SYSTEM_PROMPT",
            "Use chat history to clarify the current user question.",
        )
    )
    contextualize_q_system_prompt_version: str = str(
        os.getenv("CONTEXTUALIZE_Q_SYSTEM_PROMPT_version", "Undefined")
    )
    qa_system_prompt: str = str(
        os.getenv(
            "QA_SYSTEM_PROMPT",
            (
                "You are a helpful assistant for educational content. "
                "Use provided context and answer clearly. "
                "If context is insufficient, say it explicitly.\n\n{context}"
            ),
        )
    )
    qa_system_prompt_version: str = str(os.getenv("QA_SYSTEM_PROMPT_version", "Undefined"))

    phoenix_project_name: str = os.getenv("PHOENIX_PROJECT_NAME", "rag_eval")
    phoenix_endpoint: str = os.getenv(
        "PHOENIX_ENDPOINT",
        os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://127.0.0.1:4317"),
    )
    phoenix_protocol: str = os.getenv("PHOENIX_PROTOCOL", "grpc")
