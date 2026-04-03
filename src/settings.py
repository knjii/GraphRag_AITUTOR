import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

# Load local environment variables before reading them below.
load_dotenv()


def _env_bool(name: str, default: str = "0") -> bool:
    return os.getenv(name, default).lower() not in {"0", "false", "no"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or str(raw).strip() == "":
        return int(default)
    try:
        return int(str(raw).strip().strip("'\""))
    except Exception:
        return int(default)


def _env_int_alias(primary: str, aliases: list[str], default: int) -> int:
    if os.getenv(primary) not in {None, ""}:
        return _env_int(primary, default)
    for alias in aliases:
        if os.getenv(alias) not in {None, ""}:
            return _env_int(alias, default)
    return int(default)


def _env_choice(name: str, default: str, allowed: set[str]) -> str:
    value = str(os.getenv(name, default) or "").strip().lower()
    if value in allowed:
        return value
    return default


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

    # Generic runtime params / legacy aliases
    # Canonical keys are OLLAMA_NUM_*. N_* are supported as backward-compatible aliases.
    n_ctx: int = _env_int_alias("OLLAMA_NUM_CTX", ["N_CTX"], 4096)
    embed_ctx: int = int(os.getenv("EMBED_CTX", "256"))  # used only when backend=llamacpp
    n_threads: int = _env_int("N_THREADS", 4)
    n_batch: int = _env_int_alias("OLLAMA_NUM_BATCH", ["N_BATCH"], 1)
    n_gpu_layers: int = _env_int_alias("OLLAMA_NUM_GPU", ["N_GPU_LAYERS"], 30)
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.1"))
    embed_batch_size: int = int(os.getenv("EMBED_BATCH_SIZE", "64"))
    chroma_add_batch_size: int = int(os.getenv("CHROMA_ADD_BATCH_SIZE", "256"))
    embed_device: str = os.getenv("EMBED_DEVICE", "auto").strip().lower()
    embed_min_free_vram_mb: int = int(os.getenv("EMBED_MIN_FREE_VRAM_MB", "2048"))
    embed_min_free_vram_ratio: float = float(os.getenv("EMBED_MIN_FREE_VRAM_RATIO", "0.7"))
    embed_post_unload_wait_seconds: int = int(os.getenv("EMBED_POST_UNLOAD_WAIT_SECONDS", "180"))
    embed_post_unload_poll_seconds: int = int(os.getenv("EMBED_POST_UNLOAD_POLL_SECONDS", "5"))
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
    ollama_fallback_model: str = os.getenv("OLLAMA_FALLBACK_MODEL", "").strip()
    ollama_eval_model: str = os.getenv(
        "OLLAMA_EVAL_MODEL", os.getenv("OLLAMA_MODEL", "qwen2.5vl:3b")
    )
    ollama_top_k: int = _env_int("OLLAMA_TOP_K", 4)
    ollama_num_ctx: int = _env_int_alias("OLLAMA_NUM_CTX", ["N_CTX"], 4096)
    ollama_num_gpu: int = _env_int_alias("OLLAMA_NUM_GPU", ["N_GPU_LAYERS"], 30)
    ollama_num_batch: int = _env_int_alias("OLLAMA_NUM_BATCH", ["N_BATCH"], 1)
    ollama_num_predict: int = _env_int("OLLAMA_NUM_PREDICT", 256)
    ollama_request_timeout_seconds: int = _env_int("OLLAMA_REQUEST_TIMEOUT_SECONDS", 0)
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
    conversational_rag_enabled: bool = _env_bool("CONVERSATIONAL_RAG_ENABLED", "1")
    chat_history_dir: Path = Path(os.getenv("CHAT_HISTORY_DIR", "chat_history"))
    chat_session_id: str = str(os.getenv("CHAT_SESSION_ID", "default"))
    chat_history_max_turns: int = int(os.getenv("CHAT_HISTORY_MAX_TURNS", "6"))
    retriever_mode: str = os.getenv("RETRIEVER_MODE", "hybrid")
    hybrid_sparse_k: int = int(os.getenv("HYBRID_SPARSE_K", "8"))
    hybrid_dense_weight: float = float(os.getenv("HYBRID_DENSE_WEIGHT", "0.6"))
    hybrid_sparse_weight: float = float(os.getenv("HYBRID_SPARSE_WEIGHT", "0.4"))
    hybrid_rrf_k: int = int(os.getenv("HYBRID_RRF_K", "60"))
    chunker_use_llm: bool = _env_bool("CHUNKER_USE_LLM", "1")
    mineru_model_source: str = os.getenv("MINERU_MODEL_SOURCE", "huggingface").strip().lower()
    mineru_tools_config_json: str = os.getenv("MINERU_TOOLS_CONFIG_JSON", "mineru.json")
    mineru_local_pipeline_models_dir: str = os.getenv("MINERU_LOCAL_PIPELINE_MODELS_DIR", "").strip()
    mineru_local_vlm_models_dir: str = os.getenv("MINERU_LOCAL_VLM_MODELS_DIR", "").strip()
    mineru_parse_in_subprocess: bool = _env_bool("MINERU_PARSE_IN_SUBPROCESS", "1")
    mineru_parse_subprocess_timeout_seconds: int = int(
        os.getenv("MINERU_PARSE_SUBPROCESS_TIMEOUT_SECONDS", "0")
    )
    mineru_parse_stall_timeout_seconds: int = int(
        os.getenv("MINERU_PARSE_STALL_TIMEOUT_SECONDS", "300")
    )
    mineru_parse_heartbeat_interval_seconds: int = int(
        os.getenv("MINERU_PARSE_HEARTBEAT_INTERVAL_SECONDS", "5")
    )
    mineru_parse_wait_poll_seconds: int = int(os.getenv("MINERU_PARSE_WAIT_POLL_SECONDS", "5"))
    mineru_gpu_release_wait_seconds: int = int(os.getenv("MINERU_GPU_RELEASE_WAIT_SECONDS", "60"))
    mineru_gpu_release_poll_seconds: int = int(os.getenv("MINERU_GPU_RELEASE_POLL_SECONDS", "5"))
    mineru_gpu_release_target_free_vram_mb: int = int(
        os.getenv("MINERU_GPU_RELEASE_TARGET_FREE_VRAM_MB", "3000")
    )
    mineru_post_release_wait_seconds: int = int(os.getenv("MINERU_POST_RELEASE_WAIT_SECONDS", "120"))
    mineru_post_release_poll_seconds: int = int(os.getenv("MINERU_POST_RELEASE_POLL_SECONDS", "5"))
    mineru_release_check_enabled: bool = _env_bool("MINERU_RELEASE_CHECK_ENABLED", "1")
    neo4j_enabled: bool = _env_bool("NEO4J_ENABLED", "0")
    neo4j_uri: str = os.getenv("NEO4J_URI", "bolt://localhost:7687")
    neo4j_user: str = os.getenv("NEO4J_USER", "neo4j")
    neo4j_password: str = os.getenv("NEO4J_PASSWORD", "")
    neo4j_database: str = os.getenv("NEO4J_DATABASE", "neo4j")
    graph_rag_enabled: bool = _env_bool("GRAPH_RAG_ENABLED", "0")
    graph_write_enabled: bool = _env_bool("GRAPH_WRITE_ENABLED", "0")
    graph_relations_enabled: bool = _env_bool("GRAPH_RELATIONS_ENABLED", "1")
    graph_entity_extractor: str = _env_choice("GRAPH_ENTITY_EXTRACTOR", "rule", {"rule", "llm"})
    graph_llm_model: str = os.getenv("GRAPH_LLM_MODEL", "").strip()
    graph_llm_temperature: float = float(os.getenv("GRAPH_LLM_TEMPERATURE", "0.0"))
    graph_llm_num_predict: int = int(os.getenv("GRAPH_LLM_NUM_PREDICT", "384"))
    graph_llm_input_max_chars: int = int(os.getenv("GRAPH_LLM_INPUT_MAX_CHARS", "4000"))
    graph_llm_max_relations_per_passage: int = int(
        os.getenv("GRAPH_LLM_MAX_RELATIONS_PER_PASSAGE", "20")
    )
    graph_llm_progress_every: int = int(os.getenv("GRAPH_LLM_PROGRESS_EVERY", "25"))
    graph_llm_fallback_to_rule: bool = _env_bool("GRAPH_LLM_FALLBACK_TO_RULE", "1")
    graph_entity_min_token_len: int = int(os.getenv("GRAPH_ENTITY_MIN_TOKEN_LEN", "3"))
    graph_entity_use_bigrams: bool = os.getenv("GRAPH_ENTITY_USE_BIGRAMS", "1").lower() not in {
        "0",
        "false",
        "no",
    }
    graph_entity_max_bigrams_per_passage: int = int(
        os.getenv("GRAPH_ENTITY_MAX_BIGRAMS_PER_PASSAGE", "12")
    )
    graph_entity_max_per_passage: int = int(os.getenv("GRAPH_ENTITY_MAX_PER_PASSAGE", "20"))
    graph_cooccurs_max_per_passage: int = int(os.getenv("GRAPH_COOCCURS_MAX_PER_PASSAGE", "200"))
    graph_cooccurs_provenance_limit: int = int(
        os.getenv("GRAPH_COOCCURS_PROVENANCE_LIMIT", "20")
    )
    graph_retriever_enabled: bool = _env_bool("GRAPH_RETRIEVER_ENABLED", "0")
    graph_retriever_hops: int = int(os.getenv("GRAPH_RETRIEVER_HOPS", "1"))
    graph_retriever_entity_limit: int = int(os.getenv("GRAPH_RETRIEVER_ENTITY_LIMIT", "30"))
    graph_retriever_passage_limit: int = int(os.getenv("GRAPH_RETRIEVER_PASSAGE_LIMIT", "30"))
    graph_retriever_weight: float = float(os.getenv("GRAPH_RETRIEVER_WEIGHT", "0.35"))
    graph_fail_on_error: bool = _env_bool("GRAPH_FAIL_ON_ERROR", "0")

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
