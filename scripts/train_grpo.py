"""GRPO-дообучение генератора с наградой R6 (docs/RESEARCH-2026-09.md, раздел 4).

Запускается на арендованной карте. Проба спринта 2 — сначала это:

    python scripts/train_grpo.py --dataset rl-train.jsonl --model Qwen/Qwen3.5-4B \\
        --steps 20 --num-generations 4 --out runs/probe --probe

``--probe`` печатает пиковую память и секунды на шаг — ради этих двух чисел
и арендуется день. Полный прогон и контроли:

    --reward main      основная награда
    --reward random    контроль: случайная награда
    --reward format    контроль: только ворота формата

``--dry-run`` работает без GPU и без torch: проверяет набор и награду
на заглушках ответов. Его надо гонять перед арендой.

Правило проекта: читать генерации до замера. Каждые ``--sample-every`` шагов
в ``<out>/samples.jsonl`` пишутся ответы с разбивкой награды.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rag_textbook.rewards.composite import (  # noqa: E402
    RewardConfig,
    compute_reward,
    format_reward,
    random_reward,
)
from rag_textbook.rl.env import completion_text, load_jsonl  # noqa: E402


def build_reward(kind: str, *, seed: int, max_completion_tokens: int, samples_path: Path | None,
                 sample_every: int):
    config = RewardConfig()
    state = {"calls": 0}

    def reward(completions, context, reference, gold_in_context, question_id=None,
               question=None, completion_ids=None, **_):
        state["calls"] += 1
        texts = [completion_text(item) for item in completions]
        # Обрыв по пределу токенов виден только по длине сгенерированного.
        # Новые версии TRL передают completion_ids; если нет — ловят ворота длины.
        truncated = (
            [len(ids) >= max_completion_tokens for ids in completion_ids]
            if completion_ids is not None else [False] * len(texts)
        )
        keys = question_id or [str(i) for i in range(len(texts))]
        questions = question or [""] * len(texts)
        values, records = [], []
        for text, ctx, ref, gold, cut, key, asked in zip(
            texts, context, reference, gold_in_context, truncated, keys, questions, strict=True
        ):
            main = compute_reward(text, context=ctx, reference=ref, question=asked,
                                  gold_in_context=bool(gold), truncated=cut, config=config)
            if kind == "main":
                value = main.total
            elif kind == "random":
                value = random_reward(text, seed=seed, key=f"{key}:{state['calls']}")
            elif kind == "format":
                value = format_reward(text, config=config)
            else:
                raise ValueError(f"неизвестная награда {kind}")
            values.append(value)
            # Основная награда пишется и в контрольных прогонах: так видно,
            # растёт ли формульная часть при случайном сигнале.
            records.append({"question_id": key, "reward": value, "main": main.as_dict(),
                            "answer": text[:1500]})
        # Считаем от первого вызова, а не по остатку 1: при
        # --sample-every 1 условие «% 1 == 1» не выполнялось никогда,
        # и генерации молча не писались — как раз в том прогоне,
        # где их просили писать каждый раз.
        if samples_path and (state["calls"] - 1) % max(1, sample_every) == 0:
            with samples_path.open("a", encoding="utf-8") as handle:
                for record in records[:4]:
                    handle.write(json.dumps({"call": state["calls"], **record}, ensure_ascii=False) + "\n")
        return values

    reward.__name__ = f"reward_{kind}"
    return reward


def dry_run(args) -> int:
    examples = load_jsonl(args.dataset)
    if not examples:
        print("набор пуст", file=sys.stderr)
        return 1
    fn = build_reward(args.reward, seed=args.seed, max_completion_tokens=args.max_completion_length,
                      samples_path=None, sample_every=1)
    batch = examples[: min(8, len(examples))]
    # Заглушки: эталонный текст (должен получить высокую награду), пустой
    # ответ (штраф) и английский ответ (штраф ворот).
    stubs = {
        "эталон": [e.reference[:1200] for e in batch],
        "пусто": ["" for _ in batch],
        "англ": ["The answer is in the context." for _ in batch],
    }
    for name, texts in stubs.items():
        values = fn(completions=texts, context=[e.context for e in batch],
                    reference=[e.reference for e in batch],
                    gold_in_context=[e.gold_in_context for e in batch],
                    question_id=[e.question_id for e in batch],
                    question=[e.question for e in batch])
        print(f"{name:7s} средняя награда {sum(values) / len(values):+.3f}")
    lengths = sorted(len(m["content"]) for e in examples for m in e.messages[:1])
    print(f"эпизодов {len(examples)}, системный промпт: медиана {lengths[len(lengths) // 2]} знаков, "
          f"максимум {lengths[-1]}")
    return 0


def train(args) -> int:
    # Тяжёлые зависимости — только на карте. Unsloth — до TRL и Transformers:
    # его патчи применяются при импорте (так в официальном Qwen3.5 GRPO
    # notebook; задача 014).
    if args.backend == "unsloth":
        import unsloth  # noqa: F401
    from datasets import Dataset
    from trl import GRPOConfig, GRPOTrainer

    examples = load_jsonl(args.dataset)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reward = build_reward(args.reward, seed=args.seed, max_completion_tokens=args.max_completion_length,
                          samples_path=out / "samples.jsonl", sample_every=args.sample_every)

    model, tokenizer, peft_config = load_model(args)
    # TRL 0.24.0 не меняет сторону обрезки переданного токенизатора;
    # длинные эпизоды отбрасываются ниже, это страховка контракта.
    tokenizer.truncation_side = "left"
    max_prompt = args.max_seq_length - args.max_completion_length
    rows, too_long = [], []
    for e in examples:
        # Промпт собирается здесь, а не в TRL: размышление Qwen3.5 гасится
        # только аргументом шаблона чата. Иначе каждый ответ начинался бы
        # с размышления, ворота давали бы −1 всей группе и преимущество
        # было бы нулевым — обучение шло бы вхолостую.
        prompt = tokenizer.apply_chat_template(
            e.messages, tokenize=False, add_generation_prompt=True, enable_thinking=False,
        )
        if len(tokenizer(prompt, add_special_tokens=False)["input_ids"]) > max_prompt:
            too_long.append(e.question_id)
            continue
        rows.append({"prompt": prompt, "context": e.context, "reference": e.reference,
                     "gold_in_context": e.gold_in_context, "question_id": e.question_id,
                     "question": e.question})
    # Длинный промпт не обрезается молча: обрезка слева съела бы контекст,
    # а награда считалась бы по полному — модель наказывалась бы за то,
    # чего не видела.
    print(f"эпизодов {len(rows)}, отброшено длиннее {max_prompt} токенов: {len(too_long)}")
    dataset = Dataset.from_list(rows)
    settings = {
        "output_dir": str(out),
        "max_steps": args.steps,
        "learning_rate": args.lr,
        "per_device_train_batch_size": args.num_generations,
        "gradient_accumulation_steps": args.grad_accum,
        "num_generations": args.num_generations,
        "max_completion_length": args.max_completion_length,
        # По умолчанию TRL обрезает промпт (до 1.x — 512 токенов);
        # длина уже проверена при сборке набора. В TRL 1.13 поля нет вовсе.
        "max_prompt_length": max_prompt,
        "temperature": args.temperature,
        "beta": args.beta,
        # Dr. GRPO: без нормировки на длину ответа и на разброс группы —
        # иначе обучение поощряет длинные неверные ответы (2503.20783).
        "loss_type": "dr_grpo",
        "scale_rewards": False,
        "logging_steps": 1,
        "save_steps": max(10, args.steps // 5),
        "bf16": True,
        "report_to": [],
        "seed": args.seed,
        "use_vllm": args.vllm,
        # colocate — движок делит карту с обучением (доля памяти и выгрузка
        # весов на шаге оптимизатора); server — движок поднят отдельно
        # (`trl vllm-serve`), обычно на второй карте, и делить память не надо.
        "vllm_mode": args.vllm_mode,
        "vllm_gpu_memory_utilization": args.vllm_memory,
        "vllm_enable_sleep_mode": args.vllm_mode == "colocate",
    }
    config = GRPOConfig(**_supported(GRPOConfig, settings))
    trainer = GRPOTrainer(
        model=model, processing_class=tokenizer, reward_funcs=[reward],
        args=config, train_dataset=dataset, peft_config=peft_config,
    )
    started = time.perf_counter()
    trainer.train()
    elapsed = time.perf_counter() - started
    prompt_tokens = sorted(len(tokenizer(r["prompt"], add_special_tokens=False)["input_ids"])
                           for r in rows[: min(64, len(rows))])
    summary = {
        "steps": args.steps, "seconds": round(elapsed, 1),
        "seconds_per_step": round(elapsed / max(1, args.steps), 2),
        # Условия пробы: без них две пробы нельзя ни сравнить между собой,
        # ни вычесть одну из другой ради доли генерации.
        "stack": "vllm" if args.vllm else args.backend,
        "backend": args.backend, "vllm": args.vllm,
        "vllm_mode": args.vllm_mode if args.vllm else None,
        "num_generations": args.num_generations,
        "grad_accum": args.grad_accum,
        "max_completion_length": args.max_completion_length,
        "max_seq_length": args.max_seq_length,
        "prompt_tokens_median": prompt_tokens[len(prompt_tokens) // 2] if prompt_tokens else 0,
        "episodes": len(rows),
    }
    try:
        import torch

        summary["peak_memory_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        summary["reserved_memory_gib"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        props = torch.cuda.get_device_properties(0)
        summary["device"] = props.name
        summary["device_memory_gib"] = round(props.total_memory / 2**30, 2)
        summary["devices"] = torch.cuda.device_count()
    except Exception:  # noqa: BLE001
        pass
    (out / "probe.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False))
    trainer.save_model(str(out / "adapter"))
    return 0


def _supported(config_class, settings: dict[str, Any]) -> dict[str, Any]:
    """Оставляет только те поля, что есть в установленной версии TRL.

    Версии расходятся сильнее, чем кажется: `max_prompt_length` в TRL 1.13
    удалён, а `vllm_mode` и `vllm_enable_sleep_mode` появились только в 1.x
    (задача 014, 017). Передать лишнее — упасть на конструкторе уже
    на оплаченной карте.
    """
    known = {field.name for field in dataclasses.fields(config_class)}
    dropped = sorted(set(settings) - known)
    if dropped:
        print(f"настройки, которых нет в этой версии TRL: {', '.join(dropped)}")
    return {name: value for name, value in settings.items() if name in known}


def load_model(args):
    """Unsloth, если установлен; иначе transformers + PEFT."""
    target = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    if args.backend == "unsloth":
        # Без перехвата ImportError: сломанный стек не должен молча
        # превращаться в другой режим обучения (задача 014).
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            args.model, max_seq_length=args.max_seq_length, load_in_4bit=False,
            fast_inference=False, max_lora_rank=args.lora_rank,
        )
        model = FastLanguageModel.get_peft_model(
            model, r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=target,
            use_gradient_checkpointing="unsloth", random_state=args.seed,
        )
        return model, tokenizer, None

    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype="bfloat16")
    model.gradient_checkpointing_enable()
    peft = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_rank, target_modules=target,
                      task_type="CAUSAL_LM")
    return model, tokenizer, peft


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen3.5-4B")
    parser.add_argument("--reward", choices=("main", "random", "format"), default="main")
    parser.add_argument("--out", default="runs/grpo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=768)
    parser.add_argument("--max-seq-length", type=int, default=12288)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--sample-every", type=int, default=10)
    parser.add_argument("--backend", choices=("unsloth", "hf"), default="unsloth",
                        help="загрузка модели: Unsloth или transformers+PEFT (без тихого отката)")
    parser.add_argument("--vllm", action="store_true",
                        help="генерация через vLLM в совмещённом режиме; только с --backend hf "
                             "(с Unsloth совместимого набора версий нет, задача 014)")
    parser.add_argument("--vllm-mode", choices=("colocate", "server"), default="colocate",
                        help="colocate — движок на той же карте; server — отдельный процесс "
                             "`trl vllm-serve`, обычно на второй карте")
    parser.add_argument("--vllm-memory", type=float, default=0.3,
                        help="доля памяти карты под движок генерации (совмещённый режим)")
    parser.add_argument("--probe", action="store_true", help="короткий прогон ради памяти и скорости")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.vllm and args.backend == "unsloth":
        parser.error("--vllm с Unsloth: совместимого набора версий нет (tasks/014-report.md). "
                     "Связка с vLLM ставится из deploy/requirements-rl-vllm.txt "
                     "и запускается с --backend hf")
    if args.vllm_mode == "server" and not args.vllm:
        parser.error("--vllm-mode server имеет смысл только с --vllm")
    if args.probe:
        args.steps = min(args.steps, 20)
    return dry_run(args) if args.dry_run else train(args)


if __name__ == "__main__":
    raise SystemExit(main())
