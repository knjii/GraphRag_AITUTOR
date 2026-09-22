"""Пересчёт ручной сверки награды текущей версией награды.

Ключи сверки хранят награду на момент оценки. После любой правки награды
её надо пересчитать на тех же ответах и сравнить с теми же ручными
оценками — иначе правка, закрывшая эксплойт, может молча сломать порядок
на настоящих ответах. Так и было в задаче 019: первая версия правок
уронила согласие 0.816 → 0.673, ниже порога приёмки.

    python scripts/reward_recheck.py --episodes artifacts/rl/mml-test.jsonl

Ответы берутся из ``capture/session-0903/answers_model-<модель>-w16384.json``
(их оценивали вслепую), ключи и оценки — из ``--checks``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from reward_agreement import THRESHOLD, bootstrap, concordance  # noqa: E402
from rl_score_saved import looks_truncated  # noqa: E402

from rag_textbook.rewards.composite import compute_reward  # noqa: E402
from rag_textbook.rl.env import load_jsonl  # noqa: E402


def rescore(
    key: list[dict], grades: dict, episodes: dict, answers: dict
) -> tuple[dict, dict, list]:
    """Группы «оценка, награда» было/стало и список изменившихся ответов."""
    before: dict = {}
    after: dict = {}
    changed = []
    for row in key:
        label = str(row.get("id", row.get("n")))
        example = episodes[row["qid"]]
        answer = answers[row["model"]][row["qid"]]
        result = compute_reward(
            answer,
            context=example.context,
            reference=example.reference,
            question=example.question,
            gold_in_context=example.gold_in_context,
            truncated=looks_truncated(answer),
        )
        grade = float(grades[label])
        group = row.get("q", 0)
        before.setdefault(group, []).append((grade, float(row["reward"])))
        after.setdefault(group, []).append((grade, result.total))
        if abs(result.total - float(row["reward"])) > 1e-6:
            changed.append((label, grade, row["reward"], result.total, result.gate))
    return before, after, changed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=Path, required=True)
    parser.add_argument("--checks", type=Path, default=ROOT / "evaluation/reward_checks/2026-09-17")
    parser.add_argument("--answers", type=Path, default=ROOT / "capture/session-0903")
    args = parser.parse_args()

    episodes = {e.question_id: e for e in load_jsonl(args.episodes)}
    key = json.loads((args.checks / "within-key.json").read_text(encoding="utf-8"))
    grades = json.loads((args.checks / "within-grades.json").read_text(encoding="utf-8"))
    answers = {}
    for model in sorted({row["model"] for row in key}):
        data = json.loads(
            (args.answers / f"answers_model-{model}-w16384.json").read_text(encoding="utf-8")
        )
        answers[model] = {o["question_id"]: o.get("answer", "") for o in data["outcomes"]}

    before, after, changed = rescore(key, grades, episodes, answers)
    print(f"ответов {len(key)}, награда изменилась у {len(changed)}")
    for label, grade, old, new, gate in changed:
        print(f"  {label}: оценка {grade:g}, награда {old:+.3f} → {new:+.3f} {gate}")
    for name, groups in (("было", before), ("стало", after)):
        value, pairs = concordance(list(groups.values()))
        low, high = bootstrap(list(groups.values()))
        print(f"{name}: согласие {value:.3f} на {pairs} парах, 95% [{low:.3f}; {high:.3f}]")
    # Решает текущая награда — последняя строка цикла.
    accepted = value >= THRESHOLD and low >= 0.5
    print(f"текущая награда {'принята' if accepted else 'НЕ принята'} (порог {THRESHOLD})")
    return 0 if accepted else 1


if __name__ == "__main__":
    raise SystemExit(main())
