"""Проверка награды R6 на уже сохранённых ответах моделей.

Если награда осмысленна, она должна ранжировать модели так же, как
независимая метрика «хотя бы одна формула»: 9B > 27B > Gemma 12B > 4B
(замер 2026-09-09) — и не должна награждать мусор (пустые ответы Muse
с пределом 768 токенов).

    python scripts/rl_score_saved.py capture/session-0903/answers_model-*.json

Контекст восстанавливается по слепку с настройками замера: промпт v4,
окно 16384, порядок relevance, доля формул 1.6, доля окна 0.5.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.evaluation.answers import latex_overlap  # noqa: E402
from rag_textbook.rewards.composite import compute_reward  # noqa: E402


def looks_truncated(answer: str) -> bool:
    """Ответ оборван пределом токенов (в сохранённых ячейках причины нет).

    При обучении обрыв известен точно (``completion_ids``) и закрывается
    воротами; здесь он угадывается по последнему знаку: законченный ответ
    кончается точкой, скобкой ссылки или формулой.
    """
    text = (answer or "").rstrip()
    return bool(text) and text[-1] not in ".!?)]}$»…:;*|"


def configure_env() -> None:
    os.environ.setdefault("RAG_ENV_FILE", "tests-no-such-env-file")
    os.environ["QA_SYSTEM_PROMPT"] = (ROOT / "deploy/prompts/qa-v4.txt").read_text(encoding="utf-8")
    os.environ["LLM_CONTEXT_WINDOW"] = "16384"
    os.environ["CONTEXT_ORDER"] = "relevance"
    os.environ["CONTEXT_FORMULA_BUDGET_SHARE"] = "1.6"
    os.environ["CONTEXT_WINDOW_SHARE"] = "0.5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells", nargs="+", type=Path)
    parser.add_argument("--trace", type=Path, default=ROOT / "capture/session-0819/trace-always.jsonl")
    parser.add_argument("--json", type=Path, help="куда сохранить сводку")
    args = parser.parse_args()

    configure_env()
    from rag_textbook.config import Settings
    from rag_textbook.rl.env import build_examples, load_chunks, load_trace

    settings = Settings()
    chunks = load_chunks(settings.paths.parsed_dir)
    examples = {
        e.question_id: e
        for e in build_examples(
            settings, load_trace(args.trace), chunks,
            ROOT / "evaluation/goldsets/goldset.json",
        )
    }
    print(f"эпизодов: {len(examples)}, эталон в контексте: "
          f"{sum(e.gold_in_context for e in examples.values())}")

    summary = {}
    for cell in args.cells:
        data = json.loads(cell.read_text(encoding="utf-8"))
        rewards, v1_hit, v1_expected, v2_hit, v2_expected = [], 0, 0, 0, 0
        gates: Counter[str] = Counter()
        by_type: dict[str, list[float]] = defaultdict(list)
        foreign = 0
        parts_sum: Counter[str] = Counter()
        passed = 0
        for outcome in data["outcomes"]:
            example = examples.get(outcome["question_id"])
            if example is None:
                continue
            answer = outcome.get("answer", "")
            result = compute_reward(
                answer, context=example.context, reference=example.reference,
                question=example.question, gold_in_context=example.gold_in_context,
                truncated=looks_truncated(answer),
            )
            rewards.append(result.total)
            by_type[example.question_type].append(result.total)
            gates[result.gate or "прошёл"] += 1
            if not result.gate:
                passed += 1
                parts_sum.update(result.parts)
            expected, found = latex_overlap(example.reference, answer)
            v1_expected += bool(expected)
            v1_hit += bool(expected and found)
            if result.formula and result.formula.expected:
                v2_expected += 1
                v2_hit += bool(result.formula.carried)
                foreign += result.formula.foreign
        n = len(rewards)
        row = {
            "ответов": n,
            "награда": round(statistics.fmean(rewards), 4) if rewards else None,
            "хотя бы одна формула v1": round(v1_hit / v1_expected, 4) if v1_expected else None,
            "хотя бы одна формула v2 (из вопросов с формулами)":
                round(v2_hit / v2_expected, 4) if v2_expected else None,
            "формул не из контекста": foreign,
            "части награды (среднее по прошедшим)": {k: round(v / passed, 4) for k, v in sorted(parts_sum.items())} if passed else {},
            "ворота": dict(gates),
            "по типам": {k: round(statistics.fmean(v), 4) for k, v in sorted(by_type.items())},
        }
        summary[cell.name] = row
        print(f"\n{cell.name}")
        for key, value in row.items():
            print(f"   {key}: {value}")

    if args.json:
        args.json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
