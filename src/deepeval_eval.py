from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence

from dotenv import load_dotenv
from langchain_core.documents import Document

from deepeval import evaluate
from deepeval.evaluate.configs import AsyncConfig, ErrorConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    FaithfulnessMetric,
)
from deepeval.models import OllamaModel
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel

from rag_chain import build_rag_chain
from settings import Settings
from utils import chroma_has_data, get_logger
from vectorstore import load_vector_store

logger = get_logger("deepeval_eval")


from opentelemetry import trace
from openinference.semconv.trace import (
    DocumentAttributes,
    OpenInferenceSpanKindValues,
    SpanAttributes,
)
from phoenix.otel import register as phoenix_register


class EvaluationPreconditionError(RuntimeError):
    """Raised when evaluation cannot start due to missing prerequisites."""


@dataclass
class RunnerConfig:
    dataset_path: Path
    max_rows: int | None
    max_contexts: int
    max_context_chars: int
    output_prefix: str
    eval_batch_size: int
    deepeval_timeout_seconds: int | None
    ignore_eval_errors: bool


def _is_auto_no_think_model(model_name: str) -> bool:
    name = str(model_name).strip().lower()
    return name.startswith("qwen3") or name.startswith("deepseek-r1")


class CompatibleOllamaModel(OllamaModel):
    """Ollama wrapper for robust structured outputs in think mode."""

    def __init__(
        self,
        *args,
        json_retry_without_think: bool = False,
        json_retry_attempts: int = 2,
        retry_num_predict_multiplier: float = 2.0,
        max_num_predict: int = 512,
        structured_recovery: bool = True,
        structured_recovery_input_chars: int = 6000,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.json_retry_without_think = bool(json_retry_without_think)
        self.json_retry_attempts = max(0, int(json_retry_attempts))
        self.retry_num_predict_multiplier = max(1.0, float(retry_num_predict_multiplier))
        self.max_num_predict = max(1, int(max_num_predict))
        self.structured_recovery = bool(structured_recovery)
        self.structured_recovery_input_chars = max(500, int(structured_recovery_input_chars))

    def _build_options(self, think_override=None, extra_options: Dict[str, Any] | None = None):
        options = {
            **{"temperature": self.temperature},
            **self.generation_kwargs,
        }
        if extra_options:
            options.update(extra_options)
        think_value = options.pop("think", None)
        if think_override is not None:
            think_value = bool(think_override)
        return options, think_value

    def _call_chat(
        self,
        chat_model,
        messages,
        schema: BaseModel | None,
        think_override=None,
        extra_options: Dict[str, Any] | None = None,
    ):
        options, think_value = self._build_options(
            think_override=think_override, extra_options=extra_options
        )
        return chat_model.chat(
            model=self.name,
            messages=messages,
            think=think_value,
            format=schema.model_json_schema() if schema else None,
            options=options,
        )

    async def _a_call_chat(
        self,
        chat_model,
        messages,
        schema: BaseModel | None,
        think_override=None,
        extra_options: Dict[str, Any] | None = None,
    ):
        options, think_value = self._build_options(
            think_override=think_override, extra_options=extra_options
        )
        return await chat_model.chat(
            model=self.name,
            messages=messages,
            think=think_value,
            format=schema.model_json_schema() if schema else None,
            options=options,
        )

    @staticmethod
    def _extract_parts(response) -> tuple[str, str]:
        message = getattr(response, "message", None)
        content = getattr(message, "content", None) if message is not None else None
        thinking = getattr(message, "thinking", None) if message is not None else None
        return str(content or ""), str(thinking or "")

    @staticmethod
    def _extract_json_candidate(text: str) -> str | None:
        if not text:
            return None
        stripped = text.strip()
        if stripped.startswith("```") and stripped.endswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                fenced = "\n".join(lines[1:-1]).strip()
                if fenced:
                    return fenced
        decoder = json.JSONDecoder()
        for idx, ch in enumerate(text):
            if ch not in "{[":
                continue
            try:
                obj, _ = decoder.raw_decode(text[idx:])
            except Exception:
                continue
            if isinstance(obj, (dict, list)):
                return json.dumps(obj, ensure_ascii=False)
        return None

    @staticmethod
    def _sanitize_invalid_json_escapes(text: str) -> str:
        """Repair invalid escapes inside JSON string literals without corrupting valid ones."""
        if not text:
            return text
        out: List[str] = []
        i = 0
        in_string = False
        text_len = len(text)
        hex_digits = set("0123456789abcdefABCDEF")

        while i < text_len:
            ch = text[i]

            if not in_string:
                out.append(ch)
                if ch == '"':
                    in_string = True
                i += 1
                continue

            # in_string == True
            if ch == '"':
                in_string = False
                out.append(ch)
                i += 1
                continue

            if ch != "\\":
                out.append(ch)
                i += 1
                continue

            # Backslash handling inside JSON string:
            # preserve valid escapes, normalize invalid ones to escaped backslash.
            if i + 1 >= text_len:
                out.append("\\\\")
                i += 1
                continue

            nxt = text[i + 1]
            if nxt in {'"', "\\", "/", "b", "f", "n", "r", "t"}:
                out.append("\\")
                out.append(nxt)
                i += 2
                continue

            if nxt == "u" and i + 5 < text_len:
                seq = text[i + 2 : i + 6]
                if all(c in hex_digits for c in seq):
                    out.append("\\")
                    out.append("u")
                    out.extend(seq)
                    i += 6
                    continue

            out.append("\\\\")
            i += 1

        return "".join(out)

    @classmethod
    def _validate_schema_from_text(cls, schema: BaseModel, text: str):
        if not text:
            return None
        candidates_with_source: List[tuple[str, str]] = [("raw", text)]
        extracted = cls._extract_json_candidate(text)
        if extracted and extracted != text:
            candidates_with_source.append(("extracted", extracted))
        for source, candidate in list(candidates_with_source):
            sanitized = cls._sanitize_invalid_json_escapes(candidate)
            if sanitized != candidate and all(
                existing != sanitized for _, existing in candidates_with_source
            ):
                candidates_with_source.append((f"{source}:sanitized", sanitized))

        first_error: Exception | None = None
        for source, candidate in candidates_with_source:
            try:
                parsed = schema.model_validate_json(candidate)
                if source.endswith(":sanitized"):
                    logger.warning(
                        "Schema validation recovered via JSON escape sanitizer: schema=%s source=%s",
                        getattr(schema, "__name__", str(schema)),
                        source,
                    )
                return parsed
            except Exception as exc_json:
                if first_error is None:
                    first_error = exc_json
                try:
                    parsed = schema.model_validate(json.loads(candidate))
                    if source.endswith(":sanitized"):
                        logger.warning(
                            "Schema validation recovered via JSON escape sanitizer+json.loads: schema=%s source=%s",
                            getattr(schema, "__name__", str(schema)),
                            source,
                        )
                    return parsed
                except Exception as exc_obj:
                    if first_error is None:
                        first_error = exc_obj
                    continue
        if first_error is not None:
            logger.warning(
                "Schema validation detail: schema=%s err_type=%s err=%s",
                getattr(schema, "__name__", str(schema)),
                type(first_error).__name__,
                str(first_error),
            )
        return None

    def _predict_override_for_attempt(self, attempt_idx: int) -> Dict[str, Any] | None:
        if attempt_idx <= 0:
            return None
        base_predict = int(self.generation_kwargs.get("num_predict") or 0)
        if base_predict <= 0:
            return None
        grown = int(round(base_predict * (self.retry_num_predict_multiplier ** attempt_idx)))
        if grown <= base_predict:
            grown = base_predict * (attempt_idx + 1)
        return {"num_predict": min(self.max_num_predict, max(base_predict, grown))}

    @staticmethod
    def _preview(text: str, limit: int = 800) -> str:
        value = str(text or "").replace("\r", "\\r").replace("\n", "\\n")
        if len(value) <= limit:
            return value
        return value[:limit] + "...<truncated>"

    @staticmethod
    def _build_recovery_prompt(schema: BaseModel, raw_text: str) -> str:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        return (
            "You are a strict JSON formatter.\n"
            "Return only valid JSON matching the provided schema.\n"
            "No markdown, no explanation.\n\n"
            f"SCHEMA:\n{schema_json}\n\n"
            f"TEXT:\n{raw_text}"
        )

    def _recover_structured_sync(
        self,
        chat_model,
        schema: BaseModel,
        raw_text: str,
        *,
        source_kind: str = "content",
    ):
        prompt = self._build_recovery_prompt(schema, raw_text[: self.structured_recovery_input_chars])
        messages = [{"role": "user", "content": prompt}]
        recovery_num_predict = min(
            self.max_num_predict,
            max(128, int(self.generation_kwargs.get("num_predict") or 256)),
        )
        response = self._call_chat(
            chat_model,
            messages,
            schema=schema,
            think_override=False,
            extra_options={"num_predict": recovery_num_predict},
        )
        content, thinking = self._extract_parts(response)
        logger.warning(
            "Structured recovery (sync): schema=%s source=%s content_len=%s thinking_len=%s content_preview=%s",
            getattr(schema, "__name__", str(schema)),
            source_kind,
            len(content),
            len(thinking),
            self._preview(content),
        )
        parsed = self._validate_schema_from_text(schema, content)
        if parsed is None and thinking:
            parsed = self._validate_schema_from_text(schema, thinking)
            if parsed is not None:
                logger.warning(
                    "Structured recovery (sync): schema=%s parsed from thinking fallback.",
                    getattr(schema, "__name__", str(schema)),
                )
        return parsed

    async def _recover_structured_async(
        self,
        chat_model,
        schema: BaseModel,
        raw_text: str,
        *,
        source_kind: str = "content",
    ):
        prompt = self._build_recovery_prompt(schema, raw_text[: self.structured_recovery_input_chars])
        messages = [{"role": "user", "content": prompt}]
        recovery_num_predict = min(
            self.max_num_predict,
            max(128, int(self.generation_kwargs.get("num_predict") or 256)),
        )
        response = await self._a_call_chat(
            chat_model,
            messages,
            schema=schema,
            think_override=False,
            extra_options={"num_predict": recovery_num_predict},
        )
        content, thinking = self._extract_parts(response)
        logger.warning(
            "Structured recovery (async): schema=%s source=%s content_len=%s thinking_len=%s content_preview=%s",
            getattr(schema, "__name__", str(schema)),
            source_kind,
            len(content),
            len(thinking),
            self._preview(content),
        )
        parsed = self._validate_schema_from_text(schema, content)
        if parsed is None and thinking:
            parsed = self._validate_schema_from_text(schema, thinking)
            if parsed is not None:
                logger.warning(
                    "Structured recovery (async): schema=%s parsed from thinking fallback.",
                    getattr(schema, "__name__", str(schema)),
                )
        return parsed

    def generate(self, prompt: str, schema: BaseModel | None = None):
        chat_model = self.load_model()
        messages = [{"role": "user", "content": prompt}]
        if not schema:
            response = self._call_chat(chat_model, messages, schema=None)
            content, _ = self._extract_parts(response)
            return content, 0

        attempts_total = self.json_retry_attempts + 1
        last_error: Exception | None = None
        last_content = ""
        last_thinking = ""
        for attempt_idx in range(attempts_total):
            attempt_options = self._predict_override_for_attempt(attempt_idx)
            response = self._call_chat(
                chat_model,
                messages,
                schema=schema,
                extra_options=attempt_options,
            )
            content, thinking = self._extract_parts(response)
            last_content = content
            last_thinking = thinking
            parsed = self._validate_schema_from_text(schema, content)
            if parsed is not None:
                return parsed, 0
            logger.warning(
                "Structured parse failed (sync): schema=%s attempt=%s/%s think=%s options=%s content_len=%s thinking_len=%s content_preview=%s",
                getattr(schema, "__name__", str(schema)),
                attempt_idx + 1,
                attempts_total,
                bool(self.generation_kwargs.get("think")),
                attempt_options or {},
                len(content),
                len(thinking),
                self._preview(content),
            )
            last_error = ValueError(
                "Structured output parse failed (content did not contain valid schema JSON)."
            )

        if self.structured_recovery and (last_content or last_thinking):
            source_kind = "content" if last_content else "thinking"
            raw_text = last_content if last_content else last_thinking
            recovered = self._recover_structured_sync(
                chat_model,
                schema=schema,
                raw_text=raw_text,
                source_kind=source_kind,
            )
            if recovered is not None:
                return recovered, 0

        if self.json_retry_without_think and bool(self.generation_kwargs.get("think")):
            response = self._call_chat(chat_model, messages, schema=schema, think_override=False)
            content, _ = self._extract_parts(response)
            parsed = self._validate_schema_from_text(schema, content)
            if parsed is not None:
                return parsed, 0

        raise last_error or ValueError("Structured output parse failed.")

    async def a_generate(self, prompt: str, schema: BaseModel | None = None):
        chat_model = self.load_model(async_mode=True)
        messages = [{"role": "user", "content": prompt}]
        if not schema:
            response = await self._a_call_chat(chat_model, messages, schema=None)
            content, _ = self._extract_parts(response)
            return content, 0

        attempts_total = self.json_retry_attempts + 1
        last_error: Exception | None = None
        last_content = ""
        last_thinking = ""
        for attempt_idx in range(attempts_total):
            attempt_options = self._predict_override_for_attempt(attempt_idx)
            response = await self._a_call_chat(
                chat_model,
                messages,
                schema=schema,
                extra_options=attempt_options,
            )
            content, thinking = self._extract_parts(response)
            last_content = content
            last_thinking = thinking
            parsed = self._validate_schema_from_text(schema, content)
            if parsed is not None:
                return parsed, 0
            logger.warning(
                "Structured parse failed (async): schema=%s attempt=%s/%s think=%s options=%s content_len=%s thinking_len=%s content_preview=%s",
                getattr(schema, "__name__", str(schema)),
                attempt_idx + 1,
                attempts_total,
                bool(self.generation_kwargs.get("think")),
                attempt_options or {},
                len(content),
                len(thinking),
                self._preview(content),
            )
            last_error = ValueError(
                "Structured output parse failed (content did not contain valid schema JSON)."
            )

        if self.structured_recovery and (last_content or last_thinking):
            source_kind = "content" if last_content else "thinking"
            raw_text = last_content if last_content else last_thinking
            recovered = await self._recover_structured_async(
                chat_model,
                schema=schema,
                raw_text=raw_text,
                source_kind=source_kind,
            )
            if recovered is not None:
                return recovered, 0

        if self.json_retry_without_think and bool(self.generation_kwargs.get("think")):
            response = await self._a_call_chat(
                chat_model,
                messages,
                schema=schema,
                think_override=False,
            )
            content, _ = self._extract_parts(response)
            parsed = self._validate_schema_from_text(schema, content)
            if parsed is not None:
                return parsed, 0

        raise last_error or ValueError("Structured output parse failed.")


class DeepEvalEvaluationRunner:
    def __init__(self, settings: Settings, config: RunnerConfig) -> None:
        self.settings = settings
        self.config = config
        self.artifacts_dir = Path(settings.deepeval_artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self._configure_console_encoding()
        self._ensure_no_proxy_for_local_ollama()
        self._configure_deepeval_timeouts()
        self.tracer = self._init_tracer()

    def run(self) -> Dict[str, Any]:
        start_time = time.monotonic()
        dataset_rows = self._load_dataset_rows(self.config.dataset_path)
        if self.config.max_rows is not None:
            dataset_rows = dataset_rows[: max(1, int(self.config.max_rows))]
        logger.info("Loaded dataset rows: %s", len(dataset_rows))

        rag_records = self.collect_rag_answers(dataset_rows)
        logger.info("Collected live RAG records: %s", len(rag_records))
        test_cases = self.build_test_cases(rag_records)
        logger.info("Prepared DeepEval test cases: %s", len(test_cases))

        evaluation_result = self.evaluate_test_cases(test_cases)
        summary, detailed = self._summarize_results(evaluation_result, rag_records)

        outputs = self._write_outputs(summary, detailed, rag_records)
        elapsed = round(time.monotonic() - start_time, 1)
        logger.info("DeepEval run complete. Elapsed: %ss", elapsed)

        return {
            "status": "ok",
            "elapsed_seconds": elapsed,
            "rows": len(rag_records),
            "summary": summary,
            "outputs": outputs,
        }

    def collect_rag_answers(self, dataset_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not chroma_has_data(self.settings):
            raise EvaluationPreconditionError(
                "Chroma index not found. Run `python src/ingest.py` first."
            )

        store = load_vector_store(self.settings)
        if store is None:
            raise EvaluationPreconditionError(
                "Vector store was not loaded. Rebuild the index and retry."
            )

        chain = build_rag_chain(self.settings)
        records: List[Dict[str, Any]] = []

        for row_id, row in enumerate(dataset_rows, start=1):
            question = str(row.get("question") or "").strip()
            if not question:
                logger.warning("Skipping empty question at row %s", row_id)
                continue

            ground_truth = str(row.get("ground_truth") or "").strip()
            logger.info("RAG row %s/%s: %s", row_id, len(dataset_rows), question)
            row_start = time.monotonic()

            with self.tracer.start_as_current_span("deepeval_rag_query") as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.CHAIN.value,
                )
                span.set_attribute(SpanAttributes.INPUT_VALUE, question)
                span.set_attribute("metadata.eval.row_id", row_id)
                span.set_attribute("metadata.eval.ground_truth", ground_truth)
                span.set_attribute("metadata.retrieval.mode", str(self.settings.retriever_mode))
                span.set_attribute("metadata.rag.chat_history_enabled", False)
                span.set_attribute("metadata.retrieval.top_k", int(self.settings.top_k))
                span.set_attribute(
                    "metadata.retrieval.hybrid_sparse_k", int(self.settings.hybrid_sparse_k)
                )

                result = chain.invoke({"input": question, "chat_history": []})
                answer = str(result.get("answer") or "").strip()

                source_docs = self._extract_source_documents(result)
                contexts = self._prepare_contexts(source_docs)

                span.set_attribute(SpanAttributes.OUTPUT_VALUE, answer)
                for idx, doc in enumerate(source_docs):
                    span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{idx}.{DocumentAttributes.DOCUMENT_CONTENT}",
                        doc.page_content,
                    )
                    span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{idx}.{DocumentAttributes.DOCUMENT_METADATA}",
                        json.dumps(doc.metadata, ensure_ascii=False),
                    )
                    span.set_attribute(
                        f"{SpanAttributes.RETRIEVAL_DOCUMENTS}.{idx}.{DocumentAttributes.DOCUMENT_ID}",
                        str(doc.metadata.get("source", idx)),
                    )

            record = {
                "row_id": row_id,
                "question": question,
                "ground_truth": ground_truth,
                "answer": answer,
                "contexts": contexts,
                "source_contexts": [str(item) for item in row.get("contexts", [])],
                "source_answer": str(row.get("answer") or ""),
            }
            records.append(record)
            logger.info(
                "RAG row %s done (contexts=%s, elapsed=%.1fs)",
                row_id,
                len(contexts),
                time.monotonic() - row_start,
            )

        if not records:
            raise EvaluationPreconditionError("No valid rows were collected from dataset.")

        return records

    def build_test_cases(self, rag_records: Sequence[Dict[str, Any]]) -> List[LLMTestCase]:
        test_cases: List[LLMTestCase] = []
        for record in rag_records:
            test_cases.append(
                LLMTestCase(
                    input=record["question"],
                    actual_output=record["answer"],
                    expected_output=record["ground_truth"],
                    retrieval_context=record["contexts"],
                )
            )
        return test_cases

    def evaluate_test_cases(self, test_cases: Sequence[LLMTestCase]):
        generation_kwargs = {
            "num_predict": int(self.settings.ollama_eval_num_predict),
            "num_ctx": int(self.settings.ollama_num_ctx),
            "top_k": int(self.settings.ollama_top_k),
        }
        eval_think = str(self.settings.ollama_eval_think or "auto").strip().lower()
        if eval_think in {"0", "false", "no", "off"}:
            generation_kwargs["think"] = False
        elif eval_think in {"1", "true", "yes", "on"}:
            generation_kwargs["think"] = True
        elif _is_auto_no_think_model(self.settings.ollama_eval_model):
            # qwen3*/deepseek-r1* are usually more stable in structured eval with think disabled.
            generation_kwargs["think"] = False

        eval_model = CompatibleOllamaModel(
            model=self.settings.ollama_eval_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.ollama_eval_temperature,
            generation_kwargs=generation_kwargs,
            json_retry_without_think=self.settings.ollama_eval_json_retry_without_think,
            json_retry_attempts=self.settings.ollama_eval_json_retry_attempts,
            retry_num_predict_multiplier=self.settings.ollama_eval_retry_num_predict_multiplier,
            max_num_predict=self.settings.ollama_eval_max_num_predict,
            structured_recovery=self.settings.ollama_eval_structured_recovery,
            structured_recovery_input_chars=self.settings.ollama_eval_structured_recovery_input_chars,
        )

        metrics = self._build_metrics(eval_model)
        batch_size = max(1, int(self.config.eval_batch_size))
        total_cases = len(test_cases)
        total_batches = (total_cases + batch_size - 1) // batch_size
        aggregated_results: List[Any] = []

        logger.info(
            "Running DeepEval metrics: %s",
            ", ".join(metric.__class__.__name__ for metric in metrics),
        )
        logger.info(
            "DeepEval config: model=%s base_url=%s batch_size=%s ignore_errors=%s eval_think=%s json_retry_without_think=%s json_retry_attempts=%s max_num_predict=%s structured_recovery=%s",
            self.settings.ollama_eval_model,
            self.settings.ollama_base_url,
            batch_size,
            self.config.ignore_eval_errors,
            generation_kwargs.get("think", "default"),
            bool(self.settings.ollama_eval_json_retry_without_think),
            int(self.settings.ollama_eval_json_retry_attempts),
            int(self.settings.ollama_eval_max_num_predict),
            bool(self.settings.ollama_eval_structured_recovery),
        )

        for batch_start in range(0, total_cases, batch_size):
            batch_end = min(batch_start + batch_size, total_cases)
            batch_index = batch_start // batch_size + 1
            batch_cases = list(test_cases[batch_start:batch_end])
            logger.info(
                "DeepEval batch %s/%s: cases %s-%s",
                batch_index,
                total_batches,
                batch_start + 1,
                batch_end,
            )
            batch_started = time.monotonic()

            with self.tracer.start_as_current_span("deepeval_metrics_batch") as span:
                span.set_attribute(
                    SpanAttributes.OPENINFERENCE_SPAN_KIND,
                    OpenInferenceSpanKindValues.CHAIN.value,
                )
                span.set_attribute("metadata.eval.model", self.settings.ollama_eval_model)
                span.set_attribute("metadata.eval.batch_index", batch_index)
                span.set_attribute("metadata.eval.batch_size", len(batch_cases))
                span.set_attribute("metadata.eval.total_cases", total_cases)
                span.set_attribute(
                    "metadata.eval.metrics",
                    ",".join(metric.__class__.__name__ for metric in metrics),
                )

                try:
                    batch_result = evaluate(
                        batch_cases,
                        metrics=self._build_metrics(eval_model),
                        async_config=AsyncConfig(run_async=False, max_concurrent=1),
                        error_config=ErrorConfig(
                            ignore_errors=self.config.ignore_eval_errors,
                            skip_on_missing_params=False,
                        ),
                    )
                    batch_test_results = list(getattr(batch_result, "test_results", []))
                except Exception as exc:
                    logger.exception("DeepEval batch %s failed: %s", batch_index, exc)
                    if not self.config.ignore_eval_errors:
                        raise
                    batch_test_results = [
                        self._build_failed_test_result(str(exc)) for _ in batch_cases
                    ]

            if len(batch_test_results) < len(batch_cases):
                missing_count = len(batch_cases) - len(batch_test_results)
                logger.warning(
                    "DeepEval batch %s returned fewer results than cases (%s/%s). Filling missing with failures.",
                    batch_index,
                    len(batch_test_results),
                    len(batch_cases),
                )
                batch_test_results.extend(
                    [
                        self._build_failed_test_result(
                            "missing DeepEval result for case"
                        )
                        for _ in range(missing_count)
                    ]
                )
            elif len(batch_test_results) > len(batch_cases):
                batch_test_results = batch_test_results[: len(batch_cases)]

            aggregated_results.extend(batch_test_results)
            logger.info(
                "DeepEval batch %s/%s done (elapsed=%.1fs)",
                batch_index,
                total_batches,
                time.monotonic() - batch_started,
            )

        return SimpleNamespace(test_results=aggregated_results)

    def _build_metrics(self, eval_model: OllamaModel) -> List[Any]:
        return [
            AnswerRelevancyMetric(
                model=eval_model,
                threshold=0.01,
                include_reason=True,
                async_mode=False,
            ),
            FaithfulnessMetric(
                model=eval_model,
                threshold=0.01,
                include_reason=True,
                async_mode=False,
            ),
            ContextualPrecisionMetric(
                model=eval_model,
                threshold=0.01,
                include_reason=True,
                async_mode=False,
            ),
        ]

    @staticmethod
    def _build_failed_test_result(error_message: str) -> Any:
        metric_names = (
            "Answer Relevancy",
            "Faithfulness",
            "Contextual Precision",
        )
        return SimpleNamespace(
            metrics_data=[
                SimpleNamespace(
                    name=metric_name,
                    score=None,
                    threshold=0.01,
                    success=False,
                    reason=f"evaluation failed: {error_message}",
                )
                for metric_name in metric_names
            ]
        )

    def _summarize_results(
        self,
        evaluation_result: Any,
        rag_records: Sequence[Dict[str, Any]],
    ) -> tuple[Dict[str, Dict[str, Any]], List[Dict[str, Any]]]:
        detailed: List[Dict[str, Any]] = []
        by_metric: Dict[str, Dict[str, Any]] = {}

        test_results = list(getattr(evaluation_result, "test_results", []))
        for idx, test_result in enumerate(test_results):
            record = rag_records[idx] if idx < len(rag_records) else {}
            metrics_data = list(getattr(test_result, "metrics_data", []))
            case_metrics: List[Dict[str, Any]] = []

            for item in metrics_data:
                metric_name = str(getattr(item, "name", "unknown"))
                score = getattr(item, "score", None)
                threshold = getattr(item, "threshold", None)
                success = getattr(item, "success", None)
                reason = getattr(item, "reason", None)

                case_metrics.append(
                    {
                        "name": metric_name,
                        "score": score,
                        "threshold": threshold,
                        "success": success,
                        "reason": reason,
                    }
                )

                agg = by_metric.setdefault(
                    metric_name,
                    {
                        "scores": [],
                        "success_total": 0,
                        "count": 0,
                        "threshold": threshold,
                    },
                )
                if isinstance(score, (float, int)):
                    agg["scores"].append(float(score))
                agg["count"] += 1
                if success is True:
                    agg["success_total"] += 1

            detailed.append(
                {
                    "row_id": record.get("row_id", idx + 1),
                    "question": record.get("question", ""),
                    "answer": record.get("answer", ""),
                    "ground_truth": record.get("ground_truth", ""),
                    "contexts_count": len(record.get("contexts", [])),
                    "metrics": case_metrics,
                }
            )

        summary: Dict[str, Dict[str, Any]] = {}
        for name, agg in by_metric.items():
            scores = agg["scores"]
            count = int(agg["count"])
            success_total = int(agg["success_total"])
            summary[name] = {
                "mean_score": round(statistics.fmean(scores), 4) if scores else None,
                "pass_rate": round(success_total / count, 4) if count else None,
                "count": count,
                "threshold": agg["threshold"],
            }

        return summary, detailed

    def _write_outputs(
        self,
        summary: Dict[str, Dict[str, Any]],
        detailed: List[Dict[str, Any]],
        rag_records: Sequence[Dict[str, Any]],
    ) -> Dict[str, str]:
        prefix = self.config.output_prefix.strip() or "deepeval"

        rag_inputs_path = self.artifacts_dir / f"{prefix}_rag_eval_inputs_live.json"
        scores_path = self.artifacts_dir / f"{prefix}_scores.json"
        detailed_path = self.artifacts_dir / f"{prefix}_scores_detailed.json"

        self._write_json_array(list(rag_records), rag_inputs_path)
        self._write_json_array(summary, scores_path)
        self._write_json_array(detailed, detailed_path)

        logger.info("Saved live RAG inputs: %s", rag_inputs_path)
        logger.info("Saved metric summary: %s", scores_path)
        logger.info("Saved detailed metrics: %s", detailed_path)

        return {
            "rag_inputs_live": str(rag_inputs_path),
            "scores": str(scores_path),
            "scores_detailed": str(detailed_path),
        }

    def _init_tracer(self):
        try:
            provider = phoenix_register(
                project_name=self.settings.phoenix_project_name,
                endpoint=self.settings.phoenix_endpoint,
                protocol=self.settings.phoenix_protocol,
                auto_instrument=True,
            )
        except Exception as exc:
            raise RuntimeError(
                "Phoenix tracing initialization failed. "
                "Verify arize-phoenix and collector availability."
            ) from exc

        logger.info("Tracing via phoenix.otel.register")
        logger.info(
            "Phoenix config: endpoint=%s protocol=%s project=%s",
            self.settings.phoenix_endpoint,
            self.settings.phoenix_protocol,
            self.settings.phoenix_project_name,
        )
        return trace.get_tracer(__name__, tracer_provider=provider)

    def _prepare_contexts(self, source_docs: Sequence[Document]) -> List[str]:
        contexts: List[str] = []
        for doc in source_docs[: self.config.max_contexts]:
            text = str(doc.page_content or "").strip()
            if not text:
                continue
            contexts.append(text[: self.config.max_context_chars])
        return contexts

    @staticmethod
    def _extract_source_documents(result: Dict[str, Any]) -> List[Document]:
        candidates = result.get("context") or result.get("source_documents") or []
        docs: List[Document] = []
        for item in candidates:
            if isinstance(item, Document):
                docs.append(item)
            elif isinstance(item, str):
                docs.append(Document(page_content=item, metadata={"source": "string_context"}))
            elif isinstance(item, dict):
                docs.append(
                    Document(
                        page_content=str(item.get("page_content") or item.get("text") or ""),
                        metadata=item.get("metadata", {}),
                    )
                )
        return docs

    @staticmethod
    def _write_json_array(data: Any, path: Path) -> None:
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def _load_dataset_rows(dataset_path: Path) -> List[Dict[str, Any]]:
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {dataset_path}")

        raw_text = dataset_path.read_text(encoding="utf-8-sig")
        rows: List[Dict[str, Any]]
        try:
            parsed = json.loads(raw_text)
            if isinstance(parsed, dict):
                rows = [parsed]
            elif isinstance(parsed, list):
                rows = [row for row in parsed if isinstance(row, dict)]
            else:
                raise ValueError("Unsupported dataset format. Use JSON object/list or JSONL.")
        except json.JSONDecodeError:
            rows = []
            for line_number, line in enumerate(raw_text.splitlines(), start=1):
                value = line.strip()
                if not value:
                    continue
                try:
                    parsed_line = json.loads(value)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSONL at line {line_number} in {dataset_path}: {exc}"
                    ) from exc
                if isinstance(parsed_line, dict):
                    rows.append(parsed_line)

        if not rows:
            raise ValueError(f"Dataset is empty after parsing: {dataset_path}")

        required_fields = ("question", "ground_truth")
        invalid_rows = [
            idx + 1
            for idx, row in enumerate(rows)
            if any(not str(row.get(field) or "").strip() for field in required_fields)
        ]
        if invalid_rows:
            raise ValueError(
                "Rows with missing required fields question/ground_truth: "
                + ", ".join(map(str, invalid_rows[:20]))
            )

        return rows

    def _configure_deepeval_timeouts(self) -> None:
        # DeepEval default timeout is often too high/low for local models; set explicitly.
        timeout_seconds = self.config.deepeval_timeout_seconds
        if timeout_seconds is None:
            timeout_value = "None"
        else:
            timeout_value = str(max(30, int(timeout_seconds)))
        os.environ["DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"] = timeout_value
        logger.info(
            "DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE=%s",
            os.environ.get("DEEPEVAL_PER_ATTEMPT_TIMEOUT_SECONDS_OVERRIDE"),
        )

    def _ensure_no_proxy_for_local_ollama(self) -> None:
        # Avoid localhost proxy interception in DeepEval/Ollama httpx client.
        default_no_proxy = "127.0.0.1,localhost"
        os.environ.setdefault("NO_PROXY", default_no_proxy)
        os.environ.setdefault("no_proxy", os.environ["NO_PROXY"])

        logger.info("NO_PROXY=%s", os.environ.get("NO_PROXY"))

    @staticmethod
    def _configure_console_encoding() -> None:
        # DeepEval uses rich output with unicode symbols; force utf-8 on Windows shells.
        for stream_name in ("stdout", "stderr"):
            stream = getattr(sys, stream_name, None)
            reconfigure = getattr(stream, "reconfigure", None)
            if callable(reconfigure):
                try:
                    reconfigure(encoding="utf-8", errors="replace")
                except Exception:
                    continue



def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DeepEval metrics against live RAG outputs from a test dataset."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="deepeval_artifacts/rag_eval_inputs.json",
        help="Path to test dataset (JSON/JSONL).",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional row limit for a quick run.",
    )
    parser.add_argument(
        "--max-contexts",
        type=int,
        default=2,
        help="How many retrieved contexts to keep per case.",
    )
    parser.add_argument(
        "--max-context-chars",
        type=int,
        default=1400,
        help="Max chars per retrieved context.",
    )
    parser.add_argument(
        "--output-prefix",
        type=str,
        default="deepeval",
        help="Prefix for output files in deepeval_artifacts.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=1,
        help="How many test cases to send to one DeepEval call.",
    )
    parser.add_argument(
        "--deepeval-timeout-seconds",
        type=int,
        default=300,
        help="Per-attempt timeout for DeepEval in seconds; <=0 disables timeout.",
    )
    parser.add_argument(
        "--ignore-eval-errors",
        dest="ignore_eval_errors",
        action="store_true",
        default=True,
        help="Keep run alive when one batch fails/timeouts (default: true).",
    )
    parser.add_argument(
        "--fail-on-eval-error",
        dest="ignore_eval_errors",
        action="store_false",
        help="Stop immediately when one DeepEval batch fails.",
    )
    return parser.parse_args()



def _resolve_dataset_path(raw_path: str) -> Path:
    path = Path(raw_path)
    if path.exists():
        return path

    project_root = Path(__file__).resolve().parents[1]
    candidate = project_root / raw_path
    if candidate.exists():
        return candidate

    return path



def main() -> None:
    load_dotenv()
    args = _parse_args()
    settings = Settings()

    config = RunnerConfig(
        dataset_path=_resolve_dataset_path(args.dataset),
        max_rows=args.max_rows,
        max_contexts=max(1, int(args.max_contexts)),
        max_context_chars=max(100, int(args.max_context_chars)),
        output_prefix=args.output_prefix,
        eval_batch_size=max(1, int(args.eval_batch_size)),
        deepeval_timeout_seconds=(
            None
            if int(args.deepeval_timeout_seconds) <= 0
            else int(args.deepeval_timeout_seconds)
        ),
        ignore_eval_errors=bool(args.ignore_eval_errors),
    )

    runner = DeepEvalEvaluationRunner(settings, config)
    result = runner.run()

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        logger.exception("DeepEval run failed: %s", exc)
        print(
            json.dumps(
                {
                    "status": "failed",
                    "error": str(exc),
                },
                ensure_ascii=False,
                indent=2,
            ),
            file=sys.stderr,
        )
        sys.exit(1)





