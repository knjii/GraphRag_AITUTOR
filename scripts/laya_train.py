"""Дообучение laya-multilingual на вопросах L1 и L2 (docs/HYPOTHESES.md, серия L).

Рецепт — авторский, из ноутбука Laya
(github.com/NandhaKishorM/laya, notebooks/laya_finetune_typed_decisions_2xT4_kaggle.ipynb,
Apache-2.0): RLCD — градиент политики с правильным правилом оценки
(``laya.common.proper_reward``) и групповой базой как в GRPO, плюс мягкая
перекрёстная энтропия; после обучения — одна температура на тип вопроса,
подогнанная на части, которая в обучение не входила. Отличия от ноутбука:
одна карта вместо DDP на двух, наши данные вместо typed-decisions,
гиперпараметры те же.

Температура подгоняется на отложенной части train, а не на 500 отложенных
вопросах: те служат проверкой классификатора (AUC — условие действительности
замера), и подгонка на них сделала бы проверку нечестной.

    python scripts/laya_train.py --data artifacts/laya/data --out artifacts/laya/model
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer

from laya.agent import _fix_tokenizer_config
from laya.common import QTYPES, build_model, build_sequence, clamp_temperature, proper_reward

BASE = "convaiinnovations/laya-multilingual"

EPOCHS = 3
MICRO_BATCH = 16
GRAD_ACCUM = 4
GROUP_SIZE = 4
LR_ENCODER = 2.5e-5
LR_HEAD = 1.0e-4
SIGMA_START = 0.4
SIGMA_END = 0.1
CALIB_MAX = 1000


def internal(question: dict) -> dict:
    return {"t": question["type"], "ins": question["instructions"], "crit": question.get("criteria", {})}


def build_items(rows: list[dict], tok, cfg: dict) -> tuple[list[dict], dict]:
    items, truncated = [], 0
    for row in rows:
        (task, question), = row["questions"].items()
        p_true = float(row["answers"][task]["noul"])
        target = [1.0 - p_true, p_true]
        ids, markers, stats = build_sequence(
            tok, row["state"], internal(question), cfg["max_len"], cfg["head_max_len"],
            return_truncation_stats=True,
        )
        if len(markers) != 2:
            continue
        truncated += int(stats["truncated"])
        items.append({
            "ids": ids, "markers": markers, "qtype": QTYPES["noul"],
            "target": target, "label": int(p_true > 0.5), "task": task,
        })
    return items, {"rows": len(rows), "items": len(items), "truncated": truncated}


def collate(items: list[dict], pad_id: int) -> dict:
    n, width = len(items), max(len(it["ids"]) for it in items)
    kmax = max(len(it["markers"]) for it in items)
    ids = torch.full((n, width), pad_id, dtype=torch.long)
    att = torch.zeros((n, width), dtype=torch.long)
    mpos = torch.zeros((n, kmax), dtype=torch.long)
    mmask = torch.zeros((n, kmax), dtype=torch.bool)
    target = torch.zeros((n, kmax), dtype=torch.float32)
    for i, it in enumerate(items):
        ids[i, : len(it["ids"])] = torch.tensor(it["ids"])
        att[i, : len(it["ids"])] = 1
        k = len(it["markers"])
        mpos[i, :k] = torch.tensor(it["markers"])
        mmask[i, :k] = True
        target[i, :k] = torch.tensor(it["target"], dtype=torch.float32)
    return {
        "input_ids": ids, "attention_mask": att, "marker_pos": mpos, "marker_mask": mmask,
        "target": target, "qtype": torch.tensor([it["qtype"] for it in items]),
    }


def fit_temperature(pairs: list[tuple[list[float], list[float]]]) -> float:
    """Одна температура по перекрёстной энтропии, как в ноутбуке авторов."""
    if len(pairs) < 10:
        return 1.0
    z = torch.tensor([p for p, _ in pairs], dtype=torch.float32)
    t = torch.tensor([q for _, q in pairs], dtype=torch.float32)
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = -(t * torch.log_softmax(z / log_t.exp(), -1)).sum(-1).mean()
        loss.backward()
        return loss

    opt.step(closure)
    # Пределы — те же, что применит laya.load: иначе подогнанное молча обрежется.
    return clamp_temperature(float(log_t.exp().item()))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=BASE)
    parser.add_argument("--limit", type=int, default=0, help="сухой прогон: столько строк")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = snapshot_download(args.base)
    _fix_tokenizer_config(model_dir)
    tok = AutoTokenizer.from_pretrained(os.path.join(model_dir, "tokenizer"))
    cfg = json.loads(Path(model_dir, "rl_agent_config.json").read_text(encoding="utf-8"))

    rows = [json.loads(line) for line in (args.data / "train.jsonl").open(encoding="utf-8")]
    if args.limit:
        rows = rows[: args.limit]
    items, stats = build_items(rows, tok, cfg)
    print(f"строк {stats['rows']}, примеров {stats['items']}, обрезано состояний {stats['truncated']}",
          flush=True)

    order = list(range(len(items)))
    random.Random(20261001).shuffle(order)
    n_calib = min(CALIB_MAX, len(items) // 10)
    calib = [items[i] for i in sorted(order[:n_calib])]
    train = [items[i] for i in sorted(order[n_calib:])]

    model = build_model(cfg, encoder_dir=os.path.join(model_dir, "encoder"))
    model.load_state_dict(load_file(os.path.join(model_dir, "model.safetensors")), strict=True)
    if device.type == "cuda":
        model.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.head_checkpointing = True
    model.to(device)
    model.train()

    enc_params = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = torch.optim.AdamW(
        [{"params": enc_params, "lr": LR_ENCODER}, {"params": head_params, "lr": LR_HEAD}],
        weight_decay=0.01,
    )
    total = max(1, math.ceil(len(train) / (MICRO_BATCH * GRAD_ACCUM)) * args.epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total, eta_min=1e-6)
    use_amp = device.type == "cuda"
    # RTX 3090 умеет bf16 — без масштабирования потерь; fp16 и GradScaler —
    # как в ноутбуке авторов (T4), только если bf16 нет.
    amp_dtype = torch.bfloat16 if use_amp and torch.cuda.is_bf16_supported() else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp and amp_dtype == torch.float16)
    print(f"точность: {amp_dtype if use_amp else 'float32'}", flush=True)
    print(f"обучение: {len(train)} примеров, {n_calib} на температуру, эпох {args.epochs}, "
          f"шагов {total}", flush=True)

    started = time.time()
    log = []
    for epoch in range(args.epochs):
        random.Random(42 + epoch).shuffle(train)
        sigma = SIGMA_START + (SIGMA_END - SIGMA_START) * epoch / max(1, args.epochs - 1)
        optimizer.zero_grad(set_to_none=True)
        running, batches = 0.0, 0
        for start in range(0, len(train), MICRO_BATCH):
            batch = collate(train[start : start + MICRO_BATCH], tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, act = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            logits = logits.float()
            mask = batch["marker_mask"].to(device)
            k = mask.sum(-1, keepdim=True).float()
            target = batch["target"].to(device)
            eps = torch.randn((GROUP_SIZE,) + logits.shape, device=device) * sigma * mask
            eps = (eps - eps.sum(-1, keepdim=True) / k) * mask
            z = logits.detach().unsqueeze(0) + eps
            q = torch.softmax(z.masked_fill(~mask, -1e4), -1)
            with torch.no_grad():
                reward = proper_reward(q, target.unsqueeze(0), batch["qtype"].to(device), mask,
                                       w_sph=0.75, w_rps=1.0)
                adv = reward - reward.mean(0, keepdim=True)
                adv = adv / (adv.std() + 1e-6)
            logp = -(((z - logits.unsqueeze(0)) ** 2) * mask).sum(-1) / (2 * sigma ** 2)
            loss_rl = -(adv * logp).mean()
            loss_ce = -(target * torch.log_softmax(logits.masked_fill(~mask, -1e4), -1)).sum(-1).mean()
            loss = (loss_rl + loss_ce) / GRAD_ACCUM + 0.0 * act.sum()
            scaler.scale(loss).backward()
            batches += 1
            running += loss_ce.item()
            if not math.isfinite(running):
                print(f"СТОП: потери не число на эпохе {epoch + 1}, шаг {batches}", flush=True)
                return 5
            if batches % GRAD_ACCUM == 0 or start + MICRO_BATCH >= len(train):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            if batches % 100 == 0:
                print(f"  эпоха {epoch + 1} шаг {batches}: CE {running / batches:.4f}, "
                      f"награда {reward.mean().item():.3f}, {time.time() - started:.0f} с", flush=True)
        log.append({"epoch": epoch + 1, "ce": running / max(1, batches),
                    "seconds": round(time.time() - started, 1)})
        print(f"== эпоха {epoch + 1}: CE {running / max(1, batches):.4f}, "
              f"{time.time() - started:.0f} с", flush=True)

    model.eval()
    pairs = []
    with torch.no_grad():
        for start in range(0, len(calib), 32):
            chunk = calib[start : start + 32]
            batch = collate(chunk, tok.pad_token_id)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits, _ = model(
                    batch["input_ids"].to(device), batch["attention_mask"].to(device),
                    batch["marker_pos"].to(device), batch["marker_mask"].to(device),
                    batch["qtype"].to(device),
                )
            values = logits.float().cpu().tolist()
            for row, it in zip(values, chunk):
                pairs.append((row[:2], it["target"]))
    temperature = fit_temperature(pairs)
    print(f"температура noul: {temperature:.3f}", flush=True)

    args.out.mkdir(parents=True, exist_ok=True)
    state = {k: v.half().contiguous().cpu() for k, v in model.state_dict().items()}
    save_file(state, str(args.out / "model.safetensors"))
    shutil.copytree(os.path.join(model_dir, "encoder"), args.out / "encoder", dirs_exist_ok=True)
    tok.save_pretrained(str(args.out / "tokenizer"))
    cfg["fine_tuned"] = True
    cfg["model_name"] = "laya-multilingual-rag-L"
    temps = list(cfg.get("temperature") or [1.0, 1.0, 1.0])
    temps[QTYPES["noul"]] = temperature
    cfg["temperature"] = temps
    cfg.pop("temperature_by_options", None)
    (args.out / "rl_agent_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    (args.out / "train_log.json").write_text(json.dumps(
        {"base": args.base, "stats": stats, "epochs": log, "temperature": temperature,
         "calib": n_calib, "train": len(train)}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"сохранено: {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
