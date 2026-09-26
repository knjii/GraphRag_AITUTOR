"""Пакетная оценка сохранённых ответов без поиска и генератора.

Допуск судьи заранее: согласие пар within >= 0.75, нижняя граница
95% бутстрэп-интервала строго > 0.5. Сравнивается correctness (верность).
within содержит 60 ответов, across — 50; Спирмен выводится по across и всем.
--dry-run использует постоянную подделку: проверяет восстановление текстов,
но не допускает модель к использованию. Контекст — final из того же --trace,
с которым получены ответы, тексты — --chunks (JSON или каталог разбора).

Пример: python scripts/judge_saved.py answers_run.json --goldset capture/goldset.json
--trace capture/session-0819/trace-always.jsonl --chunks capture --output-dir runs/judged
Для сверки вместо файлов ответов: --calibrate [--dry-run].
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from collections import defaultdict
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import fields
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.clients.llm import LLMClient, OpenAICompatibleLLMClient  # noqa: E402
from rag_textbook.config import LLMSettings  # noqa: E402
from rag_textbook.evaluation.answers import (  # noqa: E402
    JUDGE_PROMPT,
    AnswerOutcome,
    judge_answer,
    summarize_answers,
)
from rag_textbook.evaluation.goldset import load_goldset  # noqa: E402
from rag_textbook.evaluation.trace import TraceSet  # noqa: E402
from rag_textbook.models import GoldQuestion  # noqa: E402
from scripts.reward_agreement import THRESHOLD, bootstrap, concordance  # noqa: E402


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_contexts(
    questions: Sequence[GoldQuestion], trace: Path, chunks: Path,
) -> dict[str, str]:
    wanted = {doc for question in questions for doc in question.gold_doc_ids}
    paths = [chunks] if chunks.is_file() else [
        path for path in sorted(chunks.glob("*_chunks.json"))
        if not wanted or path.stem.removesuffix("_chunks") in wanted
    ]
    corpus = {row["id"]: row["text"] for path in paths for row in read_json(path)}
    traces = {item.question_id: item for item in TraceSet.load(trace).traces}
    contexts = {}
    for question in questions:
        if question.id not in traces:
            raise ValueError(f"Нет вопроса в слепке: {question.id}")
        ids = traces[question.id].final
        missing = [cid for cid in ids if cid not in corpus]
        if missing:
            raise ValueError(f"Нет фрагментов для {question.id}: {missing}")
        contexts[question.id] = "\n\n".join(corpus[cid] for cid in ids)
    return contexts


class DryJudge:
    """Постоянная оценка проверяет тракт без доступа к ручным баллам."""

    def describe_model(self, *, purpose: str) -> dict[str, str]:
        return {"model": "dry-run-constant-judge"}

    def chat(self, messages: Sequence[Any], **kwargs: Any) -> str:
        return '{"correctness": 1, "groundedness": 1, "reason": "сухой прогон"}'


def score_rows(
    rows: Sequence[dict[str, Any]], questions: dict[str, GoldQuestion],
    contexts: dict[str, str], llm: LLMClient, workers: int,
) -> tuple[list[dict[str, Any]], int]:
    # Проверяем вход целиком до первого обращения к модели.
    for row in rows:
        qid = row["question_id"]
        if qid not in questions or qid not in contexts:
            raise ValueError(f"Нет вопроса или контекста: {qid}")
        if not isinstance(row.get("answer"), str):
            raise ValueError(f"Нет текста ответа: {qid}")

    def one(row: dict[str, Any]) -> dict[str, Any]:
        question = questions[row["question_id"]]
        verdict = judge_answer(llm, question=question.question, answer=row["answer"],
                               context=contexts[question.id], reference=question.answer)
        valid = all(type(verdict.get(key)) is int and 0 <= verdict[key] <= 2
                    for key in ("correctness", "groundedness"))
        valid = valid and isinstance(verdict.get("reason", ""), str)
        result = dict(row)
        result.update(correctness=verdict["correctness"] if valid else None,
                      groundedness=verdict["groundedness"] if valid else None,
                      judge_reason=verdict.get("reason", "")[:300] if valid else "")
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        scored = list(pool.map(one, rows))
    return scored, sum(row["correctness"] is None for row in scored)


def provenance(llm: LLMClient, sources: Sequence[Path]) -> dict[str, Any]:
    describe = getattr(llm, "describe_model", None)
    description = describe(purpose="judge") if callable(describe) else {}
    settings = getattr(llm, "settings", None)
    configured = settings.model_for("judge") if settings is not None else None
    return {"judge_model": description.get("model") or configured,
            "judge_description": description,
            "input_sha256": {str(path): sha256(path) for path in sources},
            "judge_prompt_sha256": hashlib.sha256(JUDGE_PROMPT.encode("utf-8")).hexdigest()}


def spearman(pairs: Sequence[tuple[float, float]]) -> float | None:
    def ranks(values: Sequence[float]) -> list[float]:
        ordered = sorted(values)
        positions: dict[float, list[int]] = defaultdict(list)
        for index, value in enumerate(ordered):
            positions[value].append(index)
        return [statistics.fmean(positions[value]) for value in values]

    if len(pairs) < 2:
        return None
    left, right = zip(*pairs, strict=True)
    if len(set(left)) < 2 or len(set(right)) < 2:
        return None
    return statistics.correlation(ranks(left), ranks(right))


def calibration_metrics(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    across = []
    all_pairs = []
    for row in rows:
        if row["correctness"] is None:
            continue
        pair = (float(row["human_grade"]), float(row["correctness"]))
        all_pairs.append(pair)
        if row["group"] == "within":
            groups[row["question_id"]].append(pair)
        else:
            across.append(pair)
    # Группы без сравнимых пар не несут информации для этой меры.
    informative = [group for group in groups.values() if concordance([group])[1]]
    value, pairs = concordance(informative)
    low, high = bootstrap(informative) if informative else (float("nan"), float("nan"))
    return {"within_concordance": value if math.isfinite(value) else None,
            "within_pairs": pairs,
            "within_ci95": [v if math.isfinite(v) else None for v in (low, high)],
            "spearman_across": spearman(across), "spearman_all": spearman(all_pairs),
            "valid_within": sum(row["group"] == "within" and row["correctness"] is not None
                                for row in rows),
            "valid_across": len(across),
            "accepted": value >= THRESHOLD and low > 0.5}


def restore_calibration(checks: Path, answers_dir: Path) -> tuple[list[dict[str, Any]], list[Path]]:
    rows = []
    sources = []
    answers = {}
    for group in ("within", "across"):
        key_path, grade_path = checks / f"{group}-key.json", checks / f"{group}-grades.json"
        sources.extend([key_path, grade_path])
        grades = read_json(grade_path)
        for item in read_json(key_path):
            model, qid = item["model"], item["qid"]
            if model not in answers:
                path = answers_dir / f"answers_model-{model}-w16384.json"
                sources.append(path)
                answers[model] = {r["question_id"]: r for r in read_json(path)["outcomes"]}
            label = str(item.get("id", item.get("n")))
            grade = grades[label]
            if type(grade) is not int or not 0 <= grade <= 3:
                raise ValueError(f"Невалидная ручная оценка: {group}/{label}")
            answer = answers[model][qid]["answer"]
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"Пустой текст ответа: {model}/{qid}")
            rows.append({"question_id": qid, "answer": answer, "model": model,
                         "group": group, "human_grade": grade, "human_id": label})
    return rows, sources


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Не перезаписываем ни входы, ни результат предыдущего прогона.
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)


def main(argv: Sequence[str] | None = None, *, llm: LLMClient | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("answers", type=Path, nargs="*")
    parser.add_argument("--goldset", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--chunks", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--checks", type=Path, default=ROOT / "evaluation/reward_checks/2026-09-17")
    parser.add_argument("--answers-dir", type=Path, default=ROOT / "capture/session-0903")
    args = parser.parse_args(argv)
    if args.workers < 1 or bool(args.answers) == args.calibrate:
        parser.error("Нужны --workers >= 1 и либо файлы ответов, либо --calibrate")
    if args.dry_run and not args.calibrate:
        parser.error("--dry-run разрешён только с --calibrate")
    try:
        questions = {q.id: q for q in load_goldset(args.goldset)}
        jobs = []
        if args.calibrate:
            rows, sources = restore_calibration(args.checks, args.answers_dir)
            jobs.append((args.output_dir / "calibration.json", {}, rows, sources))
        else:
            for path in args.answers:
                data = read_json(path)
                jobs.append((args.output_dir / f"{path.stem}_judged.json",
                             data, data["outcomes"], [path]))
        targets = [job[0].resolve() for job in jobs]
        inputs = {path.resolve() for job in jobs for path in job[3]}
        inputs.update(path.resolve() for path in (args.goldset, args.trace, args.chunks))
        if len(set(targets)) != len(targets) or any(p.exists() or p in inputs for p in targets):
            raise ValueError("Выходные пути совпадают или уже существуют")
        qids = {row["question_id"] for job in jobs for row in job[2]}
        contexts = load_contexts([questions[qid] for qid in sorted(qids)], args.trace, args.chunks)
        if llm is None:
            llm = DryJudge() if args.dry_run else OpenAICompatibleLLMClient(LLMSettings(_env_file=None))
        failed = False
        for target, data, rows, sources in jobs:
            if not rows:
                raise ValueError("Нет ответов для оценки")
            scored, invalid = score_rows(rows, questions, contexts, llm, args.workers)
            fraction = invalid / len(rows)
            meta = provenance(llm, sources)
            meta.update(goldset_sha256=sha256(args.goldset), trace_sha256=sha256(args.trace))
            result = dict(data)
            result.update(outcomes=scored, judge_provenance=meta,
                          invalid_judge_fraction=fraction)
            if args.calibrate:
                metrics = calibration_metrics(scored)
                metrics["accepted"] &= fraction <= 0.05 and not args.dry_run
                result.update(calibration=metrics, dry_run=args.dry_run,
                              restored_answers=len(rows), restored_questions=len(qids))
                print(f"Восстановлено ответов: {len(rows)}, вопросов: {len(qids)}")
                print(json.dumps(metrics, ensure_ascii=False, allow_nan=False))
                failed |= not metrics["accepted"] and not args.dry_run
            else:
                names = {field.name for field in fields(AnswerOutcome)}
                outcomes = [AnswerOutcome(**{k: v for k, v in row.items() if k in names})
                            for row in scored]
                summary = summarize_answers(outcomes)
                if "чем сделано" in data.get("summary", {}):
                    summary["чем сделано"] = data["summary"]["чем сделано"]
                result["summary"] = summary
            write_json(target, result)
            print(f"{target}: пустых/невалидных оценок {invalid}/{len(rows)} ({fraction:.2%})")
            failed |= fraction > 0.05
        return int(failed)
    except (ValueError, KeyError, OSError, TypeError) as error:
        print(f"Ошибка входных данных: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
