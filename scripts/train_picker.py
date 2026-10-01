"""GRPO-обучение отборщика Context-Picker (arXiv 2512.14465), этап 2.

Модель видит вопрос и окно кандидатов (подсказка picker из
``rag_textbook.retrieval.set_selection``) и отвечает строкой
``### Final Selection: [i] [j]``. Награда — ``rag_textbook.rewards.picker``:
стадия I до шага ``--stage2-from`` (полнота, широкий запас, без штрафа),
дальше стадия II (штраф за лишнее, узкий запас) — порядок статьи.

Эпизоды — scripts/build_picker_episodes.py. Загрузка модели, микробатчи,
проверка настроек TRL и запись конфигурации — общие с train_grpo.py: две
копии разошлись бы, и отборщик учился бы иначе, чем генератор, незаметно.

Перед арендой — всегда ``--dry-run`` (без GPU): награда на заглушках обязана
развести эталонный выбор, всё окно, пустой и неформатный ответ.

    python scripts/train_picker.py --dataset artifacts/picker/musique-train.jsonl \\
        --model Qwen/Qwen3.5-4B --steps 300 --stage2-from 150 --val-every 25 --out runs/picker
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import train_grpo as base  # noqa: E402

from rag_textbook.rewards.picker import (  # noqa: E402
    STAGE_ONE,
    STAGE_TWO,
    PickerRewardConfig,
    picker_reward,
)
from rag_textbook.rl.env import completion_text  # noqa: E402


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def stage_config(step: int, stage2_from: int, *, red1: int, red2: int, gamma: float) -> PickerRewardConfig:
    if step < stage2_from:
        return PickerRewardConfig(red=red1, gamma=STAGE_ONE.gamma)
    return PickerRewardConfig(red=red2, gamma=gamma)


def build_reward(args, *, stop_ids: set[int] | None, state: dict[str, int],
                 samples_path: Path | None = None, metrics_path: Path | None = None):
    calls = {"n": 0}

    def reward(completions, gold, window, question_id=None, completion_ids=None, **kwargs):
        calls["n"] += 1
        trainer_state = kwargs.get("trainer_state")
        step = int(getattr(trainer_state, "global_step", state.get("step", 0)) or 0)
        config = stage_config(step, args.stage2_from, red1=args.red1, red2=args.red2, gamma=args.gamma)
        texts = [completion_text(item) for item in completions]
        truncated = (
            [base.is_truncated(ids, args.max_completion_length, stop_ids) for ids in completion_ids]
            if completion_ids is not None else [False] * len(texts)
        )
        keys = question_id or [str(i) for i in range(len(texts))]
        scores = [
            picker_reward(text, g, len(w), config=config, truncated=cut)
            for text, g, w, cut in zip(texts, gold, window, truncated, strict=True)
        ]
        values = [s.total for s in scores]
        if metrics_path:
            n = max(1, len(scores))
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({
                    "call": calls["n"], "step": step, "stage": 1 if step < args.stage2_from else 2,
                    "reward": round(sum(values) / n, 4),
                    "coverage": round(sum(s.coverage for s in scores) / n, 4),
                    "size": round(sum(s.size for s in scores) / n, 2),
                    "gold": round(sum(s.gold for s in scores) / n, 2),
                    "invalid": round(sum(1 for s in scores if not s.valid) / n, 4),
                    "truncated": round(sum(truncated) / n, 4),
                }, ensure_ascii=False) + "\n")
        if samples_path and (calls["n"] - 1) % max(1, args.sample_every) == 0:
            with samples_path.open("a", encoding="utf-8") as handle:
                for key, text, score in list(zip(keys, texts, scores, strict=True))[:4]:
                    handle.write(json.dumps({"call": calls["n"], "step": step, "question_id": key,
                                             **score.as_dict(), "answer": text[-1500:]},
                                            ensure_ascii=False) + "\n")
        return values

    reward.__name__ = "reward_picker"
    return reward


def summarize(scored: list[dict[str, Any]]) -> dict[str, Any]:
    if not scored:
        return {"n": 0}
    n = len(scored)
    return {
        "n": n,
        "reward": round(sum(s["total"] for s in scored) / n, 4),
        "coverage": round(sum(s["coverage"] for s in scored) / n, 4),
        # Главное для многошаговых: все эталонные в выбранном.
        "all_gold": round(sum(1 for s in scored if s["coverage"] >= 1.0) / n, 4),
        "size": round(sum(s["size"] for s in scored) / n, 2),
        "gold": round(sum(s["gold"] for s in scored) / n, 2),
        "invalid": round(sum(1 for s in scored if not s["valid"]) / n, 4),
    }


def build_validator(model: Any, tokenizer: Any, rows: list[dict[str, Any]], *, out: Path, args,
                    stop_ids: set[int] | None):
    import torch

    def validate(step: int) -> dict[str, Any]:
        started = time.perf_counter()
        was_training = model.training
        fast = None
        if args.backend == "unsloth":
            from unsloth import FastLanguageModel as fast
            fast.for_inference(model)
        model.eval()
        scored = []
        side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            for start in range(0, len(rows), args.val_batch):
                chunk = rows[start:start + args.val_batch]
                enc = tokenizer([r["prompt"] for r in chunk], return_tensors="pt", padding=True,
                                add_special_tokens=False).to(model.device)
                with torch.no_grad():
                    ids = model.generate(**enc, max_new_tokens=args.max_completion_length,
                                         do_sample=False, pad_token_id=tokenizer.pad_token_id)
                for row, seq in zip(chunk, ids[:, enc["input_ids"].shape[1]:].tolist(), strict=True):
                    while seq and seq[-1] == tokenizer.pad_token_id and tokenizer.pad_token_id not in (stop_ids or ()):
                        seq.pop()
                    text = tokenizer.decode(seq, skip_special_tokens=True)
                    cut = base.is_truncated(seq, args.max_completion_length, stop_ids)
                    # Валидация — всегда по стадии II: одно мерило на весь прогон.
                    score = picker_reward(text, row["gold"], len(row["window"]),
                                          config=PickerRewardConfig(red=args.red2, gamma=args.gamma),
                                          truncated=cut)
                    scored.append({"question_id": row["question_id"], **score.as_dict(),
                                   "answer": text[-1500:]})
        finally:
            tokenizer.padding_side = side
            if fast is not None:
                fast.for_training(model)
            if was_training:
                model.train()
        record = {"step": step, "dev": summarize(scored),
                  "seconds": round(time.perf_counter() - started, 1)}
        with (out / "validation-dev.jsonl").open("a", encoding="utf-8") as handle:
            for s in scored:
                handle.write(json.dumps({"step": step, **s}, ensure_ascii=False) + "\n")
        with (out / "validation.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        print(f"валидация, шаг {step}: {json.dumps(record, ensure_ascii=False)}", flush=True)
        return record

    return validate


def dry_run(args) -> int:
    episodes = load_jsonl(args.dataset)
    if not episodes:
        print("набор пуст", file=sys.stderr)
        return 1
    batch = episodes[: min(16, len(episodes))]
    state = {"step": 0}
    fn = build_reward(args, stop_ids=None, state=state)

    def line(indices) -> str:
        return "Нужны факты … ### Final Selection: " + " ".join(f"[{i + 1}]" for i in indices)

    stubs = {
        "эталон": [line(e["gold"]) for e in batch],
        "эталон+1": [line(e["gold"] + [i for i in range(len(e["window"])) if i not in e["gold"]][:1])
                     for e in batch],
        "половина": [line(e["gold"][: max(1, len(e["gold"]) // 2)]) for e in batch],
        "всё окно": [line(range(len(e["window"]))) for e in batch],
        "пусто": ["### Final Selection:" for _ in batch],
        "без строки": ["Ответ: второй абзац." for _ in batch],
    }
    for stage, step in (("I", 0), ("II", args.stage2_from)):
        state["step"] = step
        print(f"стадия {stage}:")
        for name, texts in stubs.items():
            values = fn(completions=texts, gold=[e["gold"] for e in batch],
                        window=[e["window"] for e in batch])
            print(f"  {name:10s} средняя награда {sum(values) / len(values):+.3f}")
    sizes = sorted(len(e["window"]) for e in episodes)
    golds = sorted(len(e["gold"]) for e in episodes)
    chars = sorted(len(e["messages"][-1]["content"]) for e in episodes)
    print(f"эпизодов {len(episodes)}; окно медиана {sizes[len(sizes) // 2]}, эталонных медиана "
          f"{golds[len(golds) // 2]}; подсказка медиана {chars[len(chars) // 2]} знаков, "
          f"максимум {chars[-1]}")
    return 0


def train(args) -> int:
    if args.backend == "unsloth":
        import unsloth  # noqa: F401
    from datasets import Dataset
    from transformers import TrainerCallback
    from trl import GRPOConfig, GRPOTrainer

    episodes = load_jsonl(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model, tokenizer, peft_config = base.load_model(args)
    stop_ids = base.stop_token_ids(tokenizer)
    state = {"step": 0}
    reward = build_reward(args, stop_ids=stop_ids, state=state,
                          samples_path=out / "samples.jsonl", metrics_path=out / "train-rewards.jsonl")
    tokenizer.truncation_side = "left"
    max_prompt = args.max_seq_length - args.max_completion_length
    rows, too_long = [], []
    for e in episodes:
        # Размышление гасится шаблоном: рассуждение отборщика — видимый текст
        # перед строкой выбора, как в подсказке SetR/picker.
        prompt = tokenizer.apply_chat_template(
            e["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > max_prompt:
            too_long.append(e["question_id"])
            continue
        rows.append({"prompt": prompt, "gold": e["gold"], "window": e["window"],
                     "question_id": e["question_id"]})
    print(f"эпизодов {len(rows)}, отброшено длиннее {max_prompt} токенов: {len(too_long)}")
    rows, dev_rows = base.split_dev(rows, args.val_dev if args.val_every else 0)
    dataset = Dataset.from_list(rows)
    settings = {
        "output_dir": str(out),
        "max_steps": args.steps,
        "learning_rate": args.lr,
        **base.micro_batching(args.num_generations, args.grad_accum, args.micro_batch),
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        "max_prompt_length": max_prompt,
        "temperature": args.temperature,
        "beta": args.beta,
        "loss_type": "dr_grpo",
        "scale_rewards": False,
        "logging_steps": 1,
        "save_steps": args.save_steps or max(10, args.steps // 5),
        "bf16": True,
        "report_to": [],
        "seed": args.seed,
        "use_vllm": False,
    }
    config = GRPOConfig(**base._supported(GRPOConfig, settings))
    base._write_run_config(out / "run-config.json", config)
    (out / "picker-config.json").write_text(json.dumps({
        "stage2_from": args.stage2_from, "red1": args.red1, "red2": args.red2, "gamma": args.gamma,
        "stage_one_default": vars(STAGE_ONE), "stage_two_default": vars(STAGE_TWO),
        "dataset": str(args.dataset), "episodes": len(rows), "dev": len(dev_rows),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    class StepTracker(TrainerCallback):
        # Запасной путь, если TRL не передаёт trainer_state в награду.
        def on_step_begin(self, args_, trainer_state, control, **kwargs):
            state["step"] = trainer_state.global_step

    callbacks: list[Any] = [StepTracker()]
    if dev_rows:
        validate = build_validator(model, tokenizer, dev_rows, out=out, args=args, stop_ids=stop_ids)

        class Validation(TrainerCallback):
            def on_train_begin(self, args_, trainer_state, control, **kwargs):
                validate(0)

            def on_step_end(self, args_, trainer_state, control, **kwargs):
                step = trainer_state.global_step
                if step % args.val_every == 0 or step == args.steps:
                    validate(step)

        callbacks.append(Validation())
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=[reward],
        args=config, train_dataset=dataset, peft_config=peft_config, callbacks=callbacks,
    )
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    done = int(getattr(trainer.state, "global_step", 0) or args.steps)
    summary = {"steps": done, "seconds": round(elapsed, 1),
               "seconds_per_step": round(elapsed / max(1, done), 2), "model": args.model,
               "load_in_4bit": args.load_in_4bit, "episodes": len(rows)}
    try:
        import torch

        summary["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    except Exception:  # noqa: BLE001
        pass
    (out / "probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    trainer.save_model(str(out / "adapter"))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--out", default="runs/picker")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--stage2-from", type=int, default=150, help="шаг перехода к стадии II")
    parser.add_argument("--red1", type=int, default=STAGE_ONE.red)
    parser.add_argument("--red2", type=int, default=STAGE_TWO.red)
    parser.add_argument("--gamma", type=float, default=STAGE_TWO.gamma)
    parser.add_argument("--num-generations", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260929)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--backend", choices=("unsloth", "hf"), default="unsloth")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--micro-batch", type=int, default=0)
    parser.add_argument("--save-steps", type=int, default=0)
    parser.add_argument("--val-every", type=int, default=0)
    parser.add_argument("--val-dev", type=int, default=64)
    parser.add_argument("--val-batch", type=int, default=8)
    parser.add_argument("--probe", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.load_in_4bit and args.backend != "unsloth":
        parser.error("--load-in-4bit поддержан только с --backend unsloth")
    if args.micro_batch and (args.num_generations * args.grad_accum) % args.micro_batch:
        parser.error("--micro-batch должен делить num_generations × grad_accum")
    if args.probe:
        args.steps = min(args.steps, 20)
        args.stage2_from = min(args.stage2_from, 10)
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
