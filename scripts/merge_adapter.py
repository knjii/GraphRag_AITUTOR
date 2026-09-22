"""Вплавить LoRA-адаптер в веса модели, чтобы её можно было подать в сервис.

После GRPO у нас остаётся `runs/<прогон>/adapter` — несколько десятков
мегабайт поверх базовой модели. Ни SGLang в нашей сборке, ни клиент
сервиса про адаптеры не знают: в `.env` задаётся один путь к весам.
Поэтому перед замером адаптер вплавляется в копию базовой модели, и
сервис поднимается на ней как на обычных весах.

    python scripts/merge_adapter.py --adapter runs/grpo-main/adapter \\
        --out runs/grpo-main/merged
    python scripts/merge_adapter.py --adapter runs/grpo-main/adapter \\
        --out runs/grpo-main/merged --is-current   # 0 — вплавлен этот адаптер
    python scripts/merge_adapter.py --identity --base Qwen/Qwen3.5-4B \\
        --out runs/base-packed                      # та же упаковка без адаптера

Проверка, без которой сливать нельзя: после вплавления веса обязаны
отличаться от базовых. Пустой или несовместимый адаптер иначе дал бы
«обученную» модель, неотличимую от базовой, а замер показал бы честный
ноль прироста — и мы искали бы причину в награде.

После задачи 021 проверка идёт по именам, а не по порядку: сверяются
все тензоры модулей из `target_modules` адаптера (по отпечатку содержимого,
без копий весов в памяти), изменения обязаны быть конечными, `dtype` берётся
из конфигурации базы, токенизатор — из каталога адаптера (с ним шло
обучение). Запись атомарна: `<out>.partial` → `<out>` с меткой `merge.json`,
где записаны отпечаток входов и список файлов с размерами.

Упаковка (задача 023). Обучение идёт на текстовой модели
(`AutoModelForCausalLM`), а она сохраняется как `Qwen3_5ForCausalLM` —
этот класс SGLang 0.5.17 не поднимает: в реестре только
`Qwen3_5ForConditionalGeneration`, а запасной путь через Transformers
отвергается. Поэтому вплавленные текстовые веса раскладываются в формат
базы: те же файлы весов и ключи (`model.language_model.*`, `lm_head.*`),
визуальная часть и конфигурация — из базы без изменений. Режим
`--identity` пакует саму базу: шаг B0 меряет её через тот же путь,
и поломка упаковки видна до обучения, а не после трёх прогонов.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
from collections.abc import Callable, Iterable
from pathlib import Path

MARKER = "merge.json"
INDEX = "model.safetensors.index.json"
# Версия формата результата входит в отпечаток: текстовый экспорт,
# сделанный до упаковки, не должен сойти за актуальный.
FORMAT = "packed-like-base-v1"
# Файлы, которые определяют результат вплавления: веса адаптера и
# токенизатор, с которым шло обучение (он копируется в merged).
_ADAPTER_FILES = ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin")
_TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja",
                    "special_tokens_map.json")
# Настройки PEFT, при которых выборка целевых весов шире фактической:
# is_target их не повторяет, поэтому они отвергаются явно (задача 022).
_UNSUPPORTED = ("layers_to_transform", "layers_pattern", "exclude_modules", "modules_to_save")
_TEXT_PREFIX = "model.language_model."


def base_model_of(adapter: Path) -> str:
    """Базовая модель записана самим PEFT — не задаём её руками."""
    data = adapter_config(adapter)
    name = data.get("base_model_name_or_path")
    if not name:
        raise ValueError(f"{adapter / 'adapter_config.json'}: не указана базовая модель")
    return str(name)


def adapter_config(adapter: Path) -> dict:
    config = adapter / "adapter_config.json"
    if not config.exists():
        raise FileNotFoundError(f"{adapter} — не каталог адаптера: нет adapter_config.json")
    return json.loads(config.read_text(encoding="utf-8"))


def adapter_digest(adapter: Path | None, base: str = "") -> str:
    """Отпечаток входов вплавления: формат результата, имя базы, веса и
    конфигурация адаптера и токенизатор рядом с ним. Без адаптера
    (`--identity`) — только формат и база."""
    digest = hashlib.sha256(f"{FORMAT}\0{base}".encode())
    if adapter is None:
        digest.update(b"\0identity")
        return digest.hexdigest()
    weights = 0
    for name in (*_ADAPTER_FILES, *_TOKENIZER_FILES):
        path = adapter / name
        if not path.is_file():
            continue
        weights += name in _ADAPTER_FILES
        digest.update(name.encode())
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    if weights < 2:
        raise FileNotFoundError(f"{adapter}: нет весов адаптера рядом с adapter_config.json")
    return digest.hexdigest()


def manifest(directory: Path) -> dict[str, int]:
    """Файлы результата и их размеры — без самой метки."""
    return {path.relative_to(directory).as_posix(): path.stat().st_size
            for path in sorted(directory.rglob("*")) if path.is_file() and path.name != MARKER}


def is_current(adapter: Path | None, out: Path, base: str = "") -> bool:
    """Лежит ли в `out` целое вплавление именно этих входов.

    Одной метки мало (задача 022): каталог с меткой, но без весов, или
    с недописанным файлом прошёл бы проверку. Поэтому сверяются и
    список файлов с размерами, и наличие конфигурации и весов.
    """
    marker = out / MARKER
    if not marker.is_file():
        return False
    try:
        data = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    files = data.get("files") or {}
    if "config.json" not in files or not any(name.endswith(".safetensors") for name in files):
        return False
    if manifest(out) != files:
        return False
    return data.get("adapter_sha256") == adapter_digest(adapter, base)


def targets_of(config: dict) -> list[str] | str:
    """Модули, которые трогает LoRA, — из самого адаптера, не догадкой."""
    targets = config.get("target_modules")
    if not targets:
        raise ValueError("adapter_config.json: не указаны target_modules")
    if targets == "all-linear":
        raise ValueError("target_modules='all-linear' не поддержан: перечислите модули явно")
    extra = [key for key in _UNSUPPORTED if config.get(key)]
    if extra:
        raise ValueError(f"adapter_config.json: не поддержано {extra} — проверка весов "
                         "считала бы не ту выборку")
    return targets if isinstance(targets, str) else sorted(targets)


def is_target(parameter: str, targets: list[str] | str) -> bool:
    """Вес целевого модуля: `model.layers.0.self_attn.q_proj.weight`.

    Правило PEFT (`check_target_module_exists`): строка — регулярное
    выражение на всё имя модуля; элемент списка совпадает с именем целиком
    или с его окончанием по границе звена (`self_attn.q_proj`).
    """
    if "lora_" in parameter or not parameter.endswith(".weight"):
        return False
    module = parameter[: -len(".weight")]
    if isinstance(targets, str):
        return re.fullmatch(targets, module) is not None
    return any(module == target or module.endswith("." + target) for target in targets)


def tensor_digest(tensor) -> str:
    """Отпечаток содержимого тензора без копии в памяти карты и без float32."""
    import torch

    data = tensor.detach().contiguous().cpu().view(torch.uint8).numpy()
    return hashlib.blake2b(data.tobytes(), digest_size=16).hexdigest()


def fingerprint(model, targets) -> dict[str, str]:
    return {name: tensor_digest(tensor)
            for name, tensor in model.named_parameters() if is_target(name, targets)}


def compare(before: dict[str, str], after: dict[str, str]) -> tuple[int, list[str]]:
    """Сколько целевых тензоров изменилось и каких имён не хватает.

    Сопоставление по имени: у PeftModel после merge_and_unload порядок
    и префиксы могли бы разойтись, а парность по порядку сравнила бы
    разные тензоры и насчитала «изменение» там, где его нет.
    """
    missing = sorted(set(before) ^ set(after))
    changed = sum(1 for name in before.keys() & after.keys() if before[name] != after[name])
    return changed, missing


def non_finite(model, targets) -> list[str]:
    import torch

    return [name for name, tensor in model.named_parameters()
            if is_target(name, targets) and not bool(torch.isfinite(tensor).all())]


def serve_path(out: Path) -> str:
    """Путь в контейнере: каталог runs смонтирован как /runs."""
    parts = out.resolve().parts
    if "runs" in parts:
        index = len(parts) - 1 - parts[::-1].index("runs")
        return "/runs/" + "/".join(parts[index + 1:])
    return str(out)


def saved_dtype(config: dict) -> str | None:
    """dtype из config.json: верхний уровень или text_config."""
    for level in (config, config.get("text_config") or {}):
        value = level.get("dtype") or level.get("torch_dtype")
        if value:
            return str(value).replace("torch.", "")
    return None


def check_saved(config: dict, architectures: list[str] | None, dtype: str) -> list[str]:
    """Расхождения сохранённой конфигурации с ожидаемой.

    Ожидается архитектура базы (результат упакован в её формат — только
    его поднимает SGLang, задача 023) и dtype, в котором реально шло
    вплавление, а не объявленный где-то ещё.
    """
    problems = []
    if config.get("architectures") != architectures:
        problems.append(f"architectures {config.get('architectures')} ≠ {architectures}")
    if saved_dtype(config) != dtype:
        problems.append(f"dtype {saved_dtype(config)} ≠ {dtype}")
    return problems


# ---------------------------------------------------------------- упаковка


def text_key_for(base_key: str) -> str | None:
    """Ключ текстовой модели для ключа базы; None — брать из базы как есть."""
    if base_key.startswith(_TEXT_PREFIX):
        return "model." + base_key[len(_TEXT_PREFIX):]
    if base_key.startswith("lm_head."):
        return base_key
    return None


def plan_pack(base_keys: Iterable[str], text_keys: Iterable[str]) -> dict[str, str | None]:
    """Ключ базы → ключ вплавленной текстовой модели (или None — из базы).

    Каждый текстовый вес обязан найти место, и каждое место — вес:
    потерянный слой молча остался бы базовым. Исключение — `lm_head`,
    которого нет в файлах базы: эмбеддинги связаны (у Qwen3.5-4B так),
    и голова восстанавливается из них при загрузке.
    """
    plan = {key: text_key_for(key) for key in base_keys}
    wanted = {source for source in plan.values() if source}
    if not any(key.startswith(_TEXT_PREFIX) for key in plan):
        raise ValueError(f"в весах базы нет {_TEXT_PREFIX}* — не во что упаковывать")
    text = set(text_keys)
    head_in_base = any(key.startswith("lm_head.") for key in plan)
    missing = sorted(wanted - text)
    unused = sorted(text - wanted - (set() if head_in_base else {"lm_head.weight"}))
    if missing or unused:
        raise ValueError(f"ключи не сходятся: нет в модели {missing[:5]}, "
                         f"лишние {unused[:5]} (всего {len(missing)} и {len(unused)})")
    return plan


def base_shards(base_dir: Path) -> dict[str, list[str]] | None:
    """Файл весов → его ключи по индексу базы; None — один файл без индекса."""
    index = base_dir / INDEX
    if not index.is_file():
        return None
    shards: dict[str, list[str]] = {}
    for key, shard in json.loads(index.read_text(encoding="utf-8"))["weight_map"].items():
        shards.setdefault(shard, []).append(key)
    return shards


def pack_like_base(state: dict, base_dir: Path, out: Path, *, targets,
                   read_shard: Callable[[Path], dict],
                   write_shard: Callable[[dict, Path], None]) -> int:
    """Разложить вплавленные веса по файлам и ключам базы. Возвращает число
    заменённых тензоров.

    Заменяются только веса целевых модулей LoRA — других вплавление
    не меняет. Остальное берётся из базы байт в байт: в Qwen3.5-4B 48
    тензоров float32 (`linear_attn.A_log`, `linear_attn.norm`), а модель
    в памяти целиком bf16 — копия из памяти потеряла бы точность, сверка
    типа отвергла бы исправное вплавление (задача 024, заголовки файлов
    весов с хаба). Без адаптера (`targets=None`) не заменяется ничего,
    но полнота ключей и формы проверяются так же.

    Файлы обрабатываются по одному — в памяти не больше одного файла базы
    сверх самой модели.
    """
    shards = base_shards(base_dir)
    if shards is None:
        only = read_shard(base_dir / "model.safetensors")
        shards = {"model.safetensors": list(only)}
    plan = plan_pack([key for keys in shards.values() for key in keys], state)
    expected = sum(1 for source in plan.values()
                   if source and targets is not None and is_target(source, targets))
    replaced = 0
    for shard, keys in sorted(shards.items()):
        base = read_shard(base_dir / shard)
        if set(base) != set(keys):
            raise ValueError(f"{shard}: ключи файла не совпали с индексом базы")
        tensors = {}
        for key in keys:
            source = plan[key]
            if source is None:
                tensors[key] = base[key]
                continue
            tensor = state[source]
            if tuple(tensor.shape) != tuple(base[key].shape):
                raise ValueError(f"{key}: форма {tuple(tensor.shape)} против базы "
                                 f"{tuple(base[key].shape)}")
            if targets is None or not is_target(source, targets):
                tensors[key] = base[key]
                continue
            if tensor.dtype != base[key].dtype:
                raise ValueError(f"{key}: {tensor.dtype} против базы {base[key].dtype}")
            tensors[key] = tensor.detach().contiguous().cpu()
            replaced += 1
        write_shard(tensors, out / shard)
    if replaced != expected:
        raise ValueError(f"заменено {replaced} целевых весов из {expected}")
    return replaced


def copy_side_files(base_dir: Path, out: Path) -> None:
    """Всё, кроме весов, — как у базы: конфигурация, индекс файлов (формы и
    типы те же, значит и размеры), препроцессоры, токенизатор (его потом
    перезапишет токенизатор обучения)."""
    for path in sorted(base_dir.iterdir()):
        if path.is_file() and path.suffix != ".safetensors" and path.name != MARKER:
            shutil.copy2(path, out / path.name)  # снимок кэша HF — ссылки, копируется файл


def local_base(base: str) -> Path:
    if Path(base).is_dir():
        return Path(base)
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(base, allow_patterns=["*.json", "*.safetensors", "*.jinja",
                                                        "*.txt", "*.model"]))


def write_model(model, base_dir: Path, base_config: dict, out: Path, targets) -> str:
    """Сохранить в формате базы. Возвращает способ, записанный в метку."""
    if base_config.get("architectures") == [type(model).__name__]:
        model.save_pretrained(str(out))
        return "save_pretrained"
    from safetensors.torch import load_file, save_file

    replaced = pack_like_base(
        model.state_dict(), base_dir, out, targets=targets,
        read_shard=lambda path: load_file(str(path)),
        write_shard=lambda tensors, path: save_file(tensors, str(path), metadata={"format": "pt"}),
    )
    copy_side_files(base_dir, out)
    print(f"упаковано в формат базы: заменено {replaced} целевых тензоров")
    return "packed"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--identity", action="store_true",
                        help="без адаптера: упаковать саму базу тем же путём (шаг B0)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--base", default=None,
                        help="базовая модель, если в adapter_config.json не та")
    parser.add_argument("--dtype", default="auto",
                        help="auto — как в конфигурации базы (иначе сервис поднимет другую модель)")
    parser.add_argument("--min-changed", type=float, default=0.5,
                        help="доля целевых тензоров, которые обязаны измениться")
    parser.add_argument("--is-current", action="store_true",
                        help="только проверить метку: 0 — в --out вплавлены эти входы")
    args = parser.parse_args(argv)
    if args.identity == (args.adapter is not None):
        parser.error("нужен ровно один из --adapter и --identity")
    if args.identity and not args.base:
        parser.error("--identity требует --base")
    adapter = None if args.identity else args.adapter

    base = args.base or base_model_of(adapter)
    if args.is_current:
        return 0 if is_current(adapter, args.out, base) else 1

    if args.out.exists() and any(args.out.iterdir()):
        print(f"{args.out} не пуст — сначала уберите его", file=sys.stderr)
        return 1
    targets = targets_of(adapter_config(adapter)) if adapter else None
    digest = adapter_digest(adapter, base)
    partial = args.out.with_name(args.out.name + ".partial")
    if partial.exists():
        # Остаток прерванного вплавления: его никто не принимал за модель.
        shutil.rmtree(partial)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    base_dir = local_base(base)
    base_config = json.loads((base_dir / "config.json").read_text(encoding="utf-8"))
    print(f"базовая модель: {base} ({base_dir}); целевые модули: {targets or '—'}")
    dtype = "auto" if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(str(base_dir), dtype=dtype)
    loaded_dtype = str(next(model.parameters()).dtype).replace("torch.", "")
    declared = saved_dtype(base_config)
    if args.dtype == "auto" and declared and loaded_dtype != declared:
        print(f"база объявляет {declared}, а загрузилась в {loaded_dtype}", file=sys.stderr)
        return 1

    changed = checked = 0
    if adapter is not None:
        from peft import PeftModel

        before = fingerprint(model, targets)
        if not before:
            print(f"в базе нет ни одного тензора из target_modules {targets}", file=sys.stderr)
            return 1
        model = PeftModel.from_pretrained(model, str(adapter))
        model = model.merge_and_unload()
        after = fingerprint(model, targets)
        changed, missing = compare(before, after)
        checked = len(before)
        print(f"изменившихся целевых тензоров: {changed} из {checked}")
        if missing:
            print(f"после вплавления не совпали имена тензоров: {missing[:5]}", file=sys.stderr)
            return 1
        if changed == 0 or changed < args.min_changed * checked:
            print("адаптер почти ничего не изменил — сливать нечего "
                  f"(порог {args.min_changed:.0%})", file=sys.stderr)
            return 1
        broken = non_finite(model, targets)
        if broken:
            print(f"после вплавления есть NaN/inf: {broken[:5]}", file=sys.stderr)
            return 1

    # Токенизатор — тот, с которым шло обучение (Trainer кладёт его рядом
    # с адаптером). Шаблон чата сверяется с базой: другой шаблон — другие
    # промпты при замере.
    source = adapter if adapter and (adapter / "tokenizer_config.json").exists() else base_dir
    tokenizer = AutoTokenizer.from_pretrained(str(source))
    same_template = tokenizer.chat_template == AutoTokenizer.from_pretrained(str(base_dir)).chat_template
    if not same_template:
        print("ВНИМАНИЕ: шаблон чата адаптера отличается от базового", file=sys.stderr)

    partial.mkdir(parents=True)
    how = write_model(model, base_dir, base_config, partial, targets)
    tokenizer.save_pretrained(str(partial))
    saved = json.loads((partial / "config.json").read_text(encoding="utf-8"))
    problems = check_saved(saved, base_config.get("architectures"), loaded_dtype)
    if problems:
        print(f"сохранённая конфигурация не та, что у базы: {problems}", file=sys.stderr)
        return 1
    (partial / MARKER).write_text(json.dumps({
        "format": FORMAT, "written_by": how,
        "adapter": str(adapter) if adapter else "identity", "adapter_sha256": digest,
        "base": base, "target_modules": targets, "changed": changed, "checked": checked,
        "architectures": saved.get("architectures"), "dtype": loaded_dtype,
        "tokenizer_from": str(source), "chat_template_as_base": same_template,
        "files": manifest(partial),
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.out.exists():
        args.out.rmdir()  # пустой — проверено выше
    os.replace(partial, args.out)

    size = sum(p.stat().st_size for p in args.out.rglob("*") if p.is_file())
    print(f"записано: {args.out} ({size / 2**30:.1f} ГиБ)")
    print(f"поднять в сервисе: bash deploy/model-swap.sh {serve_path(args.out)}"
          " — каталог runs смонтирован в контейнер")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
