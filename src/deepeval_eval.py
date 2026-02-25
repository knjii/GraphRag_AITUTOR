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
        eval_model = OllamaModel(
            model=self.settings.ollama_eval_model,
            base_url=self.settings.ollama_base_url,
            temperature=self.settings.ollama_eval_temperature,
            generation_kwargs={
                "num_predict": int(self.settings.ollama_eval_num_predict),
                "num_ctx": int(self.settings.ollama_num_ctx),
                "top_k": int(self.settings.ollama_top_k),
            },
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
            "DeepEval config: model=%s base_url=%s batch_size=%s ignore_errors=%s",
            self.settings.ollama_eval_model,
            self.settings.ollama_base_url,
            batch_size,
            self.config.ignore_eval_errors,
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
                threshold=0.5,
                include_reason=True,
                async_mode=False,
            ),
            FaithfulnessMetric(
                model=eval_model,
                threshold=0.5,
                include_reason=True,
                async_mode=False,
            ),
            ContextualPrecisionMetric(
                model=eval_model,
                threshold=0.5,
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
                    threshold=0.5,
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






