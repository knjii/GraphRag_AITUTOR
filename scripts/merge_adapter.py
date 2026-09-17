"""Вплавить LoRA-адаптер в веса модели, чтобы её можно было подать в сервис.

После GRPO у нас остаётся `runs/<прогон>/adapter` — несколько десятков
мегабайт поверх базовой модели. Ни SGLang в нашей сборке, ни клиент
сервиса про адаптеры не знают: в `.env` задаётся один путь к весам.
Поэтому перед замером адаптер вплавляется в копию базовой модели, и
сервис поднимается на ней как на обычных весах.

    python scripts/merge_adapter.py --adapter runs/grpo-main/adapter \\
        --out runs/grpo-main/merged

Проверка, без которой сливать нельзя: после вплавления веса обязаны
отличаться от базовых. Пустой или несовместимый адаптер иначе дал бы
«обученную» модель, неотличимую от базовой, а замер показал бы честный
ноль прироста — и мы искали бы причину в награде.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def base_model_of(adapter: Path) -> str:
    """Базовая модель записана самим PEFT — не задаём её руками."""
    config = adapter / "adapter_config.json"
    if not config.exists():
        raise FileNotFoundError(
            f"{adapter} — не каталог адаптера: нет adapter_config.json"
        )
    data = json.loads(config.read_text(encoding="utf-8"))
    name = data.get("base_model_name_or_path")
    if not name:
        raise ValueError(f"{config}: не указана базовая модель")
    return str(name)


def changed_parameters(before, after) -> int:
    """Сколько из проверенных тензоров изменились после вплавления."""
    import torch

    return sum(
        0 if torch.equal(old, new) else 1
        for (_, old), (_, new) in zip(before, after, strict=True)
    )


def sample_weights(model, limit: int = 12):
    """Срез весов проекций внимания: именно их трогает наш LoRA."""
    picked = []
    for name, tensor in model.named_parameters():
        if any(part in name for part in ("q_proj", "v_proj")) and "lora" not in name:
            picked.append((name, tensor.detach().clone().float().cpu()))
        if len(picked) >= limit:
            break
    return picked


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=None,
                        help="базовая модель, если в adapter_config.json не та")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args(argv)

    base = args.base or base_model_of(args.adapter)
    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} не пуст — сначала уберите его", file=sys.stderr)
        return 1

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"базовая модель: {base}")
    model = AutoModelForCausalLM.from_pretrained(base, dtype=getattr(torch, args.dtype))
    before = sample_weights(model)
    model = PeftModel.from_pretrained(model, str(args.adapter))
    model = model.merge_and_unload()
    after = sample_weights(model)

    changed = changed_parameters(before, after)
    print(f"изменившихся тензоров из проверенных: {changed} из {len(before)}")
    if changed == 0:
        print("адаптер не изменил ни одного веса — сливать нечего", file=sys.stderr)
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out))
    AutoTokenizer.from_pretrained(base).save_pretrained(str(args.out))
    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"записано: {args.out} ({size / 2**30:.1f} ГиБ)")
    print("поднять в сервисе: bash deploy/model-swap.sh /runs/"
          f"{args.out.name} — каталог runs смонтирован в контейнер")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
