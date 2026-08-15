"""Конфигурация приложения.

Отличия от прежней реализации на ``dataclass``:

* значения читаются в момент создания ``Settings()``, а не один раз при импорте модуля,
  поэтому A/B-прогоны, меняющие окружение, работают предсказуемо;
* пустая строка в переменной окружения больше не означает «включено»
  (в прежнем ``_env_bool`` ``NEO4J_ENABLED=`` давало ``True``);
* пароль Neo4j хранится в ``SecretStr`` и не попадает в трейсы и логи;
* значения валидируются, а не молча заменяются на дефолт при опечатке.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field, SecretStr, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Поля составных типов (tuple/list/dict) pydantic-settings по умолчанию пытается
# разобрать как JSON ещё ДО вызова наших валидаторов. Для значения вида
# `RELATES` или `1,3,5` разбор падает с SettingsError, и валидатор, который умеет
# делить строку по запятой, до работы не доходит. NoDecode отключает этот
# предварительный разбор: значение приходит в валидатор строкой, как и задумано.
CommaSeparatedStr = Annotated[tuple[str, ...], NoDecode]
CommaSeparatedInt = Annotated[tuple[int, ...], NoDecode]

# Путь к файлу переменных окружения. Переопределяется через RAG_ENV_FILE —
# это нужно тестам: иначе их результат зависит от того, лежит ли рядом рабочий
# .env, и на сервере с развёрнутым окружением они падают, а на машине без
# него проходят. Тест, зависящий от окружения, ничего не гарантирует.
_ENV_FILE = os.environ.get("RAG_ENV_FILE", ".env")


class _Base(BaseSettings):
    # Имена переменных окружения задаются явными алиасами на каждом поле,
    # поэтому env_prefix не используется: так видно точное имя переменной.
    # populate_by_name нужен, чтобы тесты и A/B-прогоны могли передавать
    # значения по имени поля, а не только по алиасу.
    model_config = SettingsConfigDict(
        env_file=_ENV_FILE,
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        populate_by_name=True,
        # У нас есть поля model / model_source; пространство имён model_ в pydantic
        # защищено по умолчанию и иначе засыпает предупреждениями.
        protected_namespaces=(),
    )

    @model_validator(mode="before")
    @classmethod
    def _drop_empty_values(cls, data: object) -> object:
        """Пустая переменная окружения означает «не задано».

        Прежний ``_env_bool`` на пустой строке возвращал ``True``, из-за чего
        ``NEO4J_ENABLED=`` незаметно включал графовый слой. Падать с ошибкой
        валидации тоже неудобно: закомментированная наполовину строка в ``.env``
        роняла бы приложение. Поэтому пустое значение просто игнорируется,
        и применяется значение по умолчанию.
        """
        if isinstance(data, dict):
            return {
                key: value
                for key, value in data.items()
                if not (isinstance(value, str) and value.strip() == "")
            }
        return data


class PathsSettings(_Base):
    """Каталоги данных.

    ``pdf_dir`` и ``parsed_dir`` намеренно разведены: в прежней версии MinerU писал
    результат в тот же каталог, который сканировался как источник markdown,
    что рано или поздно приводило к повторной индексации одного документа.
    """

    pdf_dir: Path = Field(default=Path("documents/pdf_docs"), alias="PDF_DIR")
    markdown_dir: Path = Field(default=Path("documents/markdown_docs"), alias="MARKDOWN_DIR")
    parsed_dir: Path = Field(default=Path("artifacts/parsed"), alias="PARSED_DIR")
    cache_dir: Path = Field(default=Path("artifacts/cache"), alias="CACHE_DIR")
    manifest_dir: Path = Field(default=Path("artifacts/manifests"), alias="MANIFEST_DIR")
    metrics_dir: Path = Field(default=Path("artifacts/metrics"), alias="METRICS_DIR")
    goldset_dir: Path = Field(default=Path("evaluation/goldsets"), alias="GOLDSET_DIR")
    state_dir: Path = Field(default=Path("artifacts/state"), alias="STATE_DIR")

    def ensure(self) -> None:
        for path in (
            self.pdf_dir,
            self.markdown_dir,
            self.parsed_dir,
            self.cache_dir,
            self.manifest_dir,
            self.metrics_dir,
            self.goldset_dir,
            self.state_dir,
        ):
            Path(path).mkdir(parents=True, exist_ok=True)


class ParsingSettings(_Base):
    """MinerU.

    ``backend`` вынесен в конфиг: в MinerU 3.x ``hybrid`` заметно быстрее ``pipeline``
    при почти той же точности, а ``vlm`` точнее на таблицах. Раньше backend был зашит в код.
    """

    backend: Literal["pipeline", "hybrid", "vlm"] = Field(
        default="pipeline", alias="MINERU_BACKEND"
    )
    # Не "ru": MinerU такого значения не знает и падает с кодом 2. Русскому
    # соответствует набор моделей распознавания east_slavic (есть ещё более
    # широкий cyrillic, но он менее точен на русском).
    lang: str = Field(default="east_slavic", alias="MINERU_LANG")
    method: Literal["auto", "txt", "ocr"] = Field(default="auto", alias="MINERU_METHOD")
    # Диапазон страниц. Нужен для дешёвых пробных прогонов: разбор — самая
    # дорогая стадия, и ограничивать объём осмысленно именно здесь, а не ниже
    # по конвейеру. Отрицательное значение page_end означает «до конца».
    # Диапазон входит в ключ кэша разбора, иначе сокращённый прогон подменил бы
    # собой полный и сравнение прогонов стало бы недействительным.
    page_start: int = Field(default=0, ge=0, alias="MINERU_PAGE_START")
    page_end: int = Field(default=-1, ge=-1, alias="MINERU_PAGE_END")
    formula_enable: bool = Field(default=True, alias="MINERU_FORMULA_ENABLE")
    table_enable: bool = Field(default=True, alias="MINERU_TABLE_ENABLE")
    model_source: Literal["huggingface", "modelscope", "local"] = Field(
        default="huggingface", alias="MINERU_MODEL_SOURCE"
    )
    local_pipeline_models_dir: str = Field(default="", alias="MINERU_LOCAL_PIPELINE_MODELS_DIR")
    tools_config_json: str = Field(default="mineru.json", alias="MINERU_TOOLS_CONFIG_JSON")
    parse_in_subprocess: bool = Field(default=True, alias="MINERU_PARSE_IN_SUBPROCESS")
    stall_timeout_seconds: int = Field(default=900, ge=0, alias="MINERU_STALL_TIMEOUT_SECONDS")
    heartbeat_interval_seconds: int = Field(
        default=5, ge=1, alias="MINERU_HEARTBEAT_INTERVAL_SECONDS"
    )
    # Верхняя граница ожидания выгрузки моделей. Выход происходит по факту освобождения,
    # а не по истечении таймера, как было раньше.
    gpu_release_max_wait_seconds: int = Field(
        default=60, ge=0, alias="MINERU_GPU_RELEASE_MAX_WAIT_SECONDS"
    )
    gpu_release_poll_seconds: int = Field(default=2, ge=1, alias="MINERU_GPU_RELEASE_POLL_SECONDS")


class ChunkingSettings(_Base):
    chunk_size: int = Field(default=1200, ge=200, alias="CHUNK_SIZE")
    # 40% перекрытия в прежней конфигурации порождало почти-дубликаты в top_k.
    chunk_overlap: int = Field(default=180, ge=0, alias="CHUNK_OVERLAP")
    context_window: int = Field(default=250, ge=0, alias="CHUNKER_CONTEXT_WINDOW")
    sticky_headers: bool = Field(default=True, alias="CHUNKER_STICKY_HEADERS")

    # Обогащение спец-объектов моделью зрения.
    enrich_enabled: bool = Field(default=True, alias="CHUNKER_ENRICH_ENABLED")
    # Формулы и таблицы MinerU уже отдаёт структурно (LaTeX / HTML), звать VLM на них —
    # это и потеря качества, и лишние часы индексации.
    enrich_types: CommaSeparatedStr = Field(
        default=("image", "chart"), alias="CHUNKER_ENRICH_TYPES"
    )
    enrich_cache_enabled: bool = Field(default=True, alias="CHUNKER_ENRICH_CACHE_ENABLED")
    enrich_max_concurrency: int = Field(
        default=2, ge=1, le=16, alias="CHUNKER_ENRICH_MAX_CONCURRENCY"
    )
    enrich_prompt_version: str = Field(default="v2", alias="CHUNKER_ENRICH_PROMPT_VERSION")

    @field_validator("enrich_types", mode="before")
    @classmethod
    def _split_types(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip() for part in value.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _check_overlap(self) -> ChunkingSettings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP должен быть меньше CHUNK_SIZE")
        return self


class IndexingSettings(_Base):
    """Режим выполнения конвейера индексации и страховки по памяти.

    ``mode`` определяет порядок обхода:

    * ``document`` — по документу целиком: разбор, обогащение, векторы, граф,
      затем следующий документ. Минимальный расход оперативной памяти,
      но каждая стадия работает маленькими порциями;
    * ``stage`` — по стадиям: сначала разбираются все документы, потом все
      обогащаются и так далее. Батчи получаются крупными, а на карте в каждый
      момент работает один потребитель — это и безопаснее, и утилизация выше.
    """

    mode: Literal["document", "stage"] = Field(default="stage", alias="INDEXING_MODE")
    # Окно накопления чанков перед отправкой в эмбеддер в режиме стадий.
    # Ограничивает потребление оперативной памяти при большом корпусе.
    embed_window: int = Field(default=1024, ge=32, alias="INDEXING_EMBED_WINDOW")
    # Разбор и чанкинг упираются в процессор, а не в карту, поэтому их
    # имеет смысл выполнять несколькими процессами.
    cpu_workers: int = Field(default=4, ge=1, le=64, alias="INDEXING_CPU_WORKERS")

    # Страховка от нехватки видеопамяти.
    vram_guard_enabled: bool = Field(default=True, alias="INDEXING_VRAM_GUARD_ENABLED")
    min_free_vram_mib: int = Field(default=1536, ge=0, alias="INDEXING_MIN_FREE_VRAM_MIB")
    vram_wait_seconds: float = Field(default=120.0, ge=0, alias="INDEXING_VRAM_WAIT_SECONDS")
    vram_poll_seconds: float = Field(default=2.0, gt=0, alias="INDEXING_VRAM_POLL_SECONDS")
    # Сколько памяти требуется стадии разбора: MinerU поднимает несколько моделей.
    parse_required_vram_mib: int = Field(
        default=7000, ge=0, alias="INDEXING_PARSE_REQUIRED_VRAM_MIB"
    )
    # Оценка расхода памяти на один элемент батча эмбеддингов.
    # Используется, чтобы уменьшить батч заранее, а не после сбоя.
    embed_per_item_mib: float = Field(default=0.0, ge=0, alias="INDEXING_EMBED_PER_ITEM_MIB")


class EmbeddingSettings(_Base):
    """Клиент эмбеддингов.

    ``base_url`` указывает на OpenAI-совместимый сервер (Infinity, vLLM, TEI).
    Это снимает зависимость от Ollama на самом горячем участке индексации.
    """

    provider: Literal["openai_compatible", "ollama", "fake"] = Field(
        default="openai_compatible", alias="EMBEDDING_PROVIDER"
    )
    # Без префикса /v1: Infinity отдаёт OpenAI-совместимые пути в корне
    # (/embeddings, /rerank, /models). С /v1 получаем 404.
    base_url: str = Field(default="http://127.0.0.1:7997", alias="EMBEDDING_BASE_URL")
    api_key: SecretStr = Field(default=SecretStr("not-needed"), alias="EMBEDDING_API_KEY")
    # bge-m3, а не Qwen3-Embedding: последняя построена на архитектуре qwen3,
    # которой не знает transformers ни в одном выпущенном образе Infinity
    # (проверено на 0.0.76 и 0.0.77 — сервер не стартует вовсе). bge-m3
    # построена на XLM-RoBERTa, многоязычна по замыслу и происходит из одного
    # семейства с нашим реранкером bge-reranker-v2-m3.
    model: str = Field(default="BAAI/bge-m3", alias="EMBEDDING_MODEL")
    dimensions: int = Field(default=1024, gt=0, alias="EMBEDDING_DIMENSIONS")
    batch_size: int = Field(default=64, ge=1, le=512, alias="EMBEDDING_BATCH_SIZE")
    timeout_seconds: float = Field(default=120.0, gt=0, alias="EMBEDDING_TIMEOUT_SECONDS")
    max_chars: int = Field(default=8000, ge=200, alias="EMBEDDING_MAX_CHARS")
    cache_enabled: bool = Field(default=True, alias="EMBEDDING_CACHE_ENABLED")
    # У bge-m3 обучение симметричное: запрос и документ кодируются одинаково,
    # инструктивный префикс не помогает, а сдвигает представление запроса.
    # Для асимметричных моделей (Qwen3-Embedding, e5) префикс задаётся здесь.
    query_prefix: str = Field(default="", alias="EMBEDDING_QUERY_PREFIX")
    document_prefix: str = Field(default="", alias="EMBEDDING_DOCUMENT_PREFIX")


class RerankerSettings(_Base):
    enabled: bool = Field(default=True, alias="RERANKER_ENABLED")
    provider: Literal["infinity", "fake", "none"] = Field(
        default="infinity", alias="RERANKER_PROVIDER"
    )
    base_url: str = Field(default="http://127.0.0.1:7997", alias="RERANKER_BASE_URL")
    model: str = Field(default="BAAI/bge-reranker-v2-m3", alias="RERANKER_MODEL")
    top_n: int = Field(default=8, ge=1, alias="RERANKER_TOP_N")
    candidates: int = Field(default=30, ge=1, alias="RERANKER_CANDIDATES")
    timeout_seconds: float = Field(default=60.0, gt=0, alias="RERANKER_TIMEOUT_SECONDS")
    max_chars: int = Field(default=4000, ge=200, alias="RERANKER_MAX_CHARS")


class LLMSettings(_Base):
    """Генеративная модель через OpenAI-совместимый API.

    И Ollama, и vLLM отдают ``/v1``, поэтому переключение между ними —
    смена ``LLM_BASE_URL``, без правки кода.
    """

    provider: Literal["openai_compatible", "fake"] = Field(
        default="openai_compatible", alias="LLM_PROVIDER"
    )
    base_url: str = Field(default="http://127.0.0.1:11434/v1", alias="LLM_BASE_URL")
    api_key: SecretStr = Field(default=SecretStr("not-needed"), alias="LLM_API_KEY")
    model: str = Field(default="qwen3.5:4b", alias="LLM_MODEL")
    vision_model: str = Field(default="qwen2.5vl:3b", alias="LLM_VISION_MODEL")
    extraction_model: str = Field(default="", alias="LLM_EXTRACTION_MODEL")
    judge_model: str = Field(default="", alias="LLM_JUDGE_MODEL")
    temperature: float = Field(default=0.1, ge=0.0, le=2.0, alias="LLM_TEMPERATURE")
    max_tokens: int = Field(default=768, ge=16, alias="LLM_MAX_TOKENS")
    context_window: int = Field(default=8192, ge=512, alias="LLM_CONTEXT_WINDOW")
    timeout_seconds: float = Field(default=300.0, gt=0, alias="LLM_TIMEOUT_SECONDS")
    max_retries: int = Field(default=3, ge=0, le=10, alias="LLM_MAX_RETRIES")
    max_concurrency: int = Field(default=4, ge=1, le=64, alias="LLM_MAX_CONCURRENCY")
    # Глубина размышления для служебных вызовов (извлечение графа, описание
    # иллюстраций). Рассуждающие модели вроде Qwen3.5 по умолчанию тратят на
    # цепочку размышлений сотни токенов и упираются в max_tokens ДО того, как
    # выдадут ответ: поле content приходит пустым, а извлечение получает
    # invalid_json. Замер на qwen3.5:4b: 1962 токена с размышлением против 116
    # без него при одинаковом результате.
    # Пустая строка означает «не передавать параметр» — для движков, которые
    # его не понимают.
    reasoning_effort: str = Field(default="none", alias="LLM_REASONING_EFFORT")
    # Для ответов пользователю размышление может быть полезно, поэтому оно
    # управляется отдельно и по умолчанию остаётся на усмотрение движка.
    chat_reasoning_effort: str = Field(default="", alias="LLM_CHAT_REASONING_EFFORT")

    def reasoning_effort_for(
        self, purpose: Literal["chat", "vision", "extraction", "judge"]
    ) -> str:
        return self.chat_reasoning_effort if purpose == "chat" else self.reasoning_effort

    def model_for(self, purpose: Literal["chat", "vision", "extraction", "judge"]) -> str:
        if purpose == "vision":
            return self.vision_model or self.model
        if purpose == "extraction":
            return self.extraction_model or self.model
        if purpose == "judge":
            return self.judge_model or self.model
        return self.model


class VectorStoreSettings(_Base):
    """Qdrant.

    Заменяет Chroma: локальный SQLite-файл Chroma не выдерживает конкурентного доступа,
    из-за чего в прежнем коде появился обработчик ``disk I/O error``.
    Заодно Qdrant закрывает лексический канал серверными sparse-векторами.
    """

    provider: Literal["qdrant", "memory"] = Field(default="qdrant", alias="QDRANT_PROVIDER")
    url: str = Field(default="http://127.0.0.1:6333", alias="QDRANT_URL")
    api_key: SecretStr | None = Field(default=None, alias="QDRANT_API_KEY")
    collection: str = Field(default="textbook_chunks", alias="QDRANT_COLLECTION")
    timeout_seconds: float = Field(default=60.0, gt=0, alias="QDRANT_TIMEOUT_SECONDS")
    upsert_batch_size: int = Field(default=128, ge=1, alias="QDRANT_UPSERT_BATCH_SIZE")
    # BM25 со стеммингом и стоп-словами русского языка — то, чего не было
    # у прежнего BM25Retriever с токенизацией по пробелам.
    sparse_enabled: bool = Field(default=True, alias="QDRANT_SPARSE_ENABLED")
    sparse_model: str = Field(default="Qdrant/bm25", alias="QDRANT_SPARSE_MODEL")
    sparse_language: str = Field(default="russian", alias="QDRANT_SPARSE_LANGUAGE")
    hnsw_m: int = Field(default=16, ge=4, alias="QDRANT_HNSW_M")
    hnsw_ef_construct: int = Field(default=128, ge=16, alias="QDRANT_HNSW_EF_CONSTRUCT")


class GraphSettings(_Base):
    """Neo4j и поведение графового канала.

    Ключевые изменения против прежней версии:

    * ``CO_OCCURS`` больше не используется для многохопового расширения — при 280 тысячах
      таких рёбер обход ``*1..2`` доставал почти весь граф;
    * рёбра фильтруются по PMI, а степень узла ограничена, поэтому клики из частотных
      терминов не строятся;
    * поиск seed-сущностей идёт через full-text индекс, а не ``CONTAINS`` с полным сканом.
    """

    enabled: bool = Field(default=True, alias="GRAPH_ENABLED")
    uri: str = Field(default="bolt://127.0.0.1:7687", alias="NEO4J_URI")
    user: str = Field(default="neo4j", alias="NEO4J_USER")
    password: SecretStr = Field(default=SecretStr(""), alias="NEO4J_PASSWORD")
    database: str = Field(default="neo4j", alias="NEO4J_DATABASE")
    write_batch_size: int = Field(default=2000, ge=100, alias="GRAPH_WRITE_BATCH_SIZE")
    fail_on_error: bool = Field(default=False, alias="GRAPH_FAIL_ON_ERROR")

    # Построение графа
    extractor: Literal["llm", "rule"] = Field(default="llm", alias="GRAPH_EXTRACTOR")
    extraction_cache_enabled: bool = Field(default=True, alias="GRAPH_EXTRACTION_CACHE_ENABLED")
    extraction_prompt_version: str = Field(default="v3", alias="GRAPH_EXTRACTION_PROMPT_VERSION")
    extraction_max_chars: int = Field(default=3500, ge=500, alias="GRAPH_EXTRACTION_MAX_CHARS")
    max_entities_per_chunk: int = Field(default=12, ge=1, alias="GRAPH_MAX_ENTITIES_PER_CHUNK")
    max_relations_per_chunk: int = Field(default=12, ge=1, alias="GRAPH_MAX_RELATIONS_PER_CHUNK")
    lemmatize_entities: bool = Field(default=True, alias="GRAPH_LEMMATIZE_ENTITIES")
    min_entity_length: int = Field(default=3, ge=1, alias="GRAPH_MIN_ENTITY_LENGTH")

    # Фильтрация шума
    cooccurrence_enabled: bool = Field(default=True, alias="GRAPH_COOCCURRENCE_ENABLED")
    cooccurrence_min_pmi: float = Field(default=1.0, alias="GRAPH_COOCCURRENCE_MIN_PMI")
    cooccurrence_min_count: int = Field(default=2, ge=1, alias="GRAPH_COOCCURRENCE_MIN_COUNT")
    # 0 отключает удаление узлов-хабов; иначе — максимальная допустимая степень.
    max_entity_degree: int = Field(default=64, ge=0, alias="GRAPH_MAX_ENTITY_DEGREE")

    @field_validator("max_entity_degree")
    @classmethod
    def _degree_is_meaningful(cls, value: int) -> int:
        if 0 < value < 4:
            raise ValueError(
                "GRAPH_MAX_ENTITY_DEGREE ниже 4 удалит почти все сущности; "
                "используйте 0, чтобы отключить обрезку"
            )
        return value

    # Извлечение
    retrieval_enabled: bool = Field(default=True, alias="GRAPH_RETRIEVAL_ENABLED")
    expansion_hops: int = Field(default=1, ge=1, le=3, alias="GRAPH_EXPANSION_HOPS")
    expansion_rel_types: CommaSeparatedStr = Field(
        default=("RELATES",), alias="GRAPH_EXPANSION_REL_TYPES"
    )
    seed_entity_limit: int = Field(default=20, ge=1, alias="GRAPH_SEED_ENTITY_LIMIT")
    passage_limit: int = Field(default=30, ge=1, alias="GRAPH_PASSAGE_LIMIT")
    weight: float = Field(default=0.4, ge=0.0, le=1.0, alias="GRAPH_WEIGHT")

    @field_validator("expansion_rel_types", mode="before")
    @classmethod
    def _split_rel_types(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(part.strip().upper() for part in value.split(",") if part.strip())
        return value

    @model_validator(mode="after")
    def _warn_on_cooccurs_expansion(self) -> GraphSettings:
        if "CO_OCCURS" in self.expansion_rel_types and self.expansion_hops > 1:
            raise ValueError(
                "Многохоповое расширение по CO_OCCURS запрещено: на плотном графе "
                "оно достаёт почти весь граф. Используйте RELATES либо hops=1."
            )
        return self


class RetrievalSettings(_Base):
    top_k: int = Field(default=8, ge=1, le=50, alias="RETRIEVAL_TOP_K")
    dense_candidates: int = Field(default=40, ge=1, alias="RETRIEVAL_DENSE_CANDIDATES")
    sparse_candidates: int = Field(default=40, ge=1, alias="RETRIEVAL_SPARSE_CANDIDATES")
    fusion: Literal["rrf", "dbsf"] = Field(default="rrf", alias="RETRIEVAL_FUSION")
    rrf_k: int = Field(default=60, ge=1, alias="RETRIEVAL_RRF_K")
    dedup_enabled: bool = Field(default=True, alias="RETRIEVAL_DEDUP_ENABLED")
    dedup_similarity: float = Field(
        default=0.92, ge=0.0, le=1.0, alias="RETRIEVAL_DEDUP_SIMILARITY"
    )

    # Роутер: графовый канал стоит денег и времени, поэтому включается не на каждый вопрос.
    router_enabled: bool = Field(default=True, alias="RETRIEVAL_ROUTER_ENABLED")
    router_mode: Literal["heuristic", "llm", "always", "never"] = Field(
        default="heuristic", alias="RETRIEVAL_ROUTER_MODE"
    )
    min_graph_docs: int = Field(default=0, ge=0, alias="RETRIEVAL_MIN_GRAPH_DOCS")

    # Переписывание вопроса по истории. Без него запрос вида «а как это на Python?»
    # уходил в поиск буквально — этого в прежней версии не было вовсе.
    query_rewrite_enabled: bool = Field(default=True, alias="RETRIEVAL_QUERY_REWRITE_ENABLED")
    max_history_turns: int = Field(default=3, ge=0, alias="RETRIEVAL_MAX_HISTORY_TURNS")


class EvalSettings(_Base):
    k_values: CommaSeparatedInt = Field(default=(1, 3, 5, 8, 10), alias="EVAL_K_VALUES")
    goldset_path: Path = Field(
        default=Path("evaluation/goldsets/goldset.json"), alias="EVAL_GOLDSET_PATH"
    )
    max_concurrency: int = Field(default=4, ge=1, alias="EVAL_MAX_CONCURRENCY")
    questions_per_chunk: int = Field(default=1, ge=1, le=5, alias="EVAL_QUESTIONS_PER_CHUNK")

    @field_validator("k_values", mode="before")
    @classmethod
    def _split_k(cls, value: object) -> object:
        if isinstance(value, str):
            return tuple(int(part) for part in value.split(",") if part.strip())
        return value


class TracingSettings(_Base):
    enabled: bool = Field(default=False, alias="PHOENIX_ENABLED")
    endpoint: str = Field(default="http://127.0.0.1:4317", alias="PHOENIX_ENDPOINT")
    protocol: Literal["grpc", "http/protobuf"] = Field(default="grpc", alias="PHOENIX_PROTOCOL")
    project_name: str = Field(default="rag_textbook", alias="PHOENIX_PROJECT_NAME")


class ServiceSettings(_Base):
    host: str = Field(default="127.0.0.1", alias="SERVICE_HOST")
    port: int = Field(default=8000, ge=1, le=65535, alias="SERVICE_PORT")
    request_timeout_seconds: float = Field(
        default=120.0, gt=0, alias="SERVICE_REQUEST_TIMEOUT_SECONDS"
    )
    max_concurrent_requests: int = Field(default=16, ge=1, alias="SERVICE_MAX_CONCURRENT_REQUESTS")
    history_db_path: Path = Field(
        default=Path("artifacts/state/chat_history.sqlite3"), alias="SERVICE_HISTORY_DB_PATH"
    )
    history_enabled: bool = Field(default=True, alias="SERVICE_HISTORY_ENABLED")


class PromptSettings(_Base):
    qa_system: str = Field(
        default=(
            "Ты — помощник для студентов, разбирающих учебные материалы по математике "
            "и техническим дисциплинам.\n"
            "Отвечай только по предоставленному контексту. Если контекста недостаточно, "
            "прямо скажи об этом, а не додумывай.\n"
            "Формулы приводи в LaTeX. После каждого утверждения ставь ссылку на источник "
            "в формате [номер], соответствующий номеру фрагмента контекста."
        ),
        alias="QA_SYSTEM_PROMPT",
    )
    query_rewrite_system: str = Field(
        default=(
            "Ты переформулируешь вопрос пользователя так, чтобы он был понятен без истории диалога.\n"
            "Подставь предмет обсуждения вместо местоимений и отсылок.\n"
            "Верни только переформулированный вопрос, без пояснений."
        ),
        alias="CONTEXTUALIZE_Q_SYSTEM_PROMPT",
    )
    prompt_version: str = Field(default="v2", alias="PROMPT_VERSION")


class Settings(BaseSettings):
    """Корневая конфигурация. Создавайте через ``Settings()`` или ``get_settings()``."""

    model_config = SettingsConfigDict(env_file=_ENV_FILE, env_file_encoding="utf-8", extra="ignore")

    paths: PathsSettings = Field(default_factory=PathsSettings)
    parsing: ParsingSettings = Field(default_factory=ParsingSettings)
    chunking: ChunkingSettings = Field(default_factory=ChunkingSettings)
    indexing: IndexingSettings = Field(default_factory=IndexingSettings)
    embedding: EmbeddingSettings = Field(default_factory=EmbeddingSettings)
    reranker: RerankerSettings = Field(default_factory=RerankerSettings)
    llm: LLMSettings = Field(default_factory=LLMSettings)
    vector_store: VectorStoreSettings = Field(default_factory=VectorStoreSettings)
    graph: GraphSettings = Field(default_factory=GraphSettings)
    retrieval: RetrievalSettings = Field(default_factory=RetrievalSettings)
    evaluation: EvalSettings = Field(default_factory=EvalSettings)
    tracing: TracingSettings = Field(default_factory=TracingSettings)
    service: ServiceSettings = Field(default_factory=ServiceSettings)
    prompts: PromptSettings = Field(default_factory=PromptSettings)

    log_level: str = Field(default="INFO", alias="LOG_LEVEL")
    log_json: bool = Field(default=False, alias="LOG_JSON")

    @model_validator(mode="after")
    def _cross_checks(self) -> Settings:
        if self.graph.enabled and not self.graph.password.get_secret_value():
            # Не падаем: без пароля просто отключаем графовый слой, чтобы
            # локальный запуск и тесты работали без Neo4j.
            object.__setattr__(self.graph, "enabled", False)
            object.__setattr__(self.graph, "retrieval_enabled", False)
        if self.reranker.candidates < self.retrieval.top_k:
            object.__setattr__(self.reranker, "candidates", self.retrieval.top_k)
        return self


def get_settings(**overrides: object) -> Settings:
    """Создаёт конфигурацию, читая окружение прямо сейчас.

    ``overrides`` удобно использовать в тестах и A/B-прогонах.
    """

    return Settings(**overrides)  # type: ignore[arg-type]
