"""Вплавление адаптера: проверки, которые можно сделать без весов.

Полный прогон требует torch и базовой модели, поэтому здесь закреплено
то, что решается до загрузки: откуда берётся базовая модель, что
считается «адаптер ничего не изменил» и почему нельзя писать в непустой
каталог.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "merge_adapter", ROOT / "scripts" / "merge_adapter.py"
)
merge_adapter = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(merge_adapter)


def adapter_dir(tmp_path: Path, **config) -> Path:
    path = tmp_path / "adapter"
    path.mkdir(parents=True)
    (path / "adapter_config.json").write_text(
        json.dumps(config, ensure_ascii=False), encoding="utf-8"
    )
    return path


def test_base_model_comes_from_the_adapter(tmp_path: Path):
    path = adapter_dir(tmp_path, base_model_name_or_path="Qwen/Qwen3.5-4B")
    assert merge_adapter.base_model_of(path) == "Qwen/Qwen3.5-4B"


def test_directory_without_adapter_config_is_refused(tmp_path: Path):
    (tmp_path / "runs").mkdir()
    with pytest.raises(FileNotFoundError):
        merge_adapter.base_model_of(tmp_path / "runs")


def test_adapter_without_base_model_is_refused(tmp_path: Path):
    with pytest.raises(ValueError):
        merge_adapter.base_model_of(adapter_dir(tmp_path, r=16))


def test_non_empty_output_is_refused(tmp_path: Path, capsys):
    path = adapter_dir(tmp_path, base_model_name_or_path="Qwen/Qwen3.5-4B")
    out = tmp_path / "merged"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")
    # Молчаливая дозапись в чужой каталог дала бы смесь двух моделей.
    assert merge_adapter.main(["--adapter", str(path), "--out", str(out)]) == 1
    assert "не пуст" in capsys.readouterr().err



# ---- задача 021: проверка по именам, метка, путь в контейнере


def full_adapter(tmp_path: Path, weights: bytes = b"w1", **config) -> Path:
    config.setdefault("base_model_name_or_path", "Qwen/Qwen3.5-4B")
    config.setdefault("target_modules", ["q_proj", "v_proj"])
    path = adapter_dir(tmp_path, **config)
    (path / "adapter_model.safetensors").write_bytes(weights)
    return path


def test_targets_are_read_from_the_adapter(tmp_path: Path):
    config = merge_adapter.adapter_config(full_adapter(tmp_path, target_modules=["v_proj", "o_proj"]))
    assert merge_adapter.targets_of(config) == ["o_proj", "v_proj"]
    with pytest.raises(ValueError):
        merge_adapter.targets_of({"r": 16})


def test_target_matching_follows_peft_rules():
    targets = ["q_proj", "v_proj"]
    assert merge_adapter.is_target("model.layers.0.self_attn.q_proj.weight", targets)
    # Слои LoRA, смещения и чужие проекции не проверяются.
    assert not merge_adapter.is_target("model.layers.0.self_attn.q_proj.lora_A.default.weight", targets)
    assert not merge_adapter.is_target("model.layers.0.self_attn.q_proj.bias", targets)
    assert not merge_adapter.is_target("model.layers.0.self_attn.k_proj.weight", targets)
    assert not merge_adapter.is_target("model.layers.0.mlp.up_proj_q_proj.weight", ["q_proj"])
    # Строка — регулярное выражение на полное имя модуля.
    regex = r".*\.(q|v)_proj"
    assert merge_adapter.is_target("model.layers.3.self_attn.v_proj.weight", regex)
    assert not merge_adapter.is_target("model.layers.3.self_attn.o_proj.weight", regex)


def test_compare_pairs_by_name_not_order():
    before = {"a": "1", "b": "2", "c": "3"}
    # Тот же набор в другом порядке — ничего не изменилось.
    assert merge_adapter.compare(before, {"c": "3", "b": "2", "a": "1"}) == (0, [])
    assert merge_adapter.compare(before, {"c": "3", "b": "9", "a": "1"}) == (1, [])
    changed, missing = merge_adapter.compare(before, {"a": "1", "b": "2", "d": "3"})
    assert missing == ["c", "d"]


def test_digest_depends_on_weights_and_config(tmp_path: Path):
    first = full_adapter(tmp_path / "1", b"w1")
    second = full_adapter(tmp_path / "2", b"w2")
    third = full_adapter(tmp_path / "3", b"w1", r=32)
    assert merge_adapter.adapter_digest(first) == merge_adapter.adapter_digest(full_adapter(tmp_path / "4", b"w1"))
    assert merge_adapter.adapter_digest(first) != merge_adapter.adapter_digest(second)
    assert merge_adapter.adapter_digest(first) != merge_adapter.adapter_digest(third)


def test_adapter_without_weights_has_no_digest(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        merge_adapter.adapter_digest(adapter_dir(tmp_path, base_model_name_or_path="x"))


BASE = "Qwen/Qwen3.5-4B"


def merged_dir(out: Path, adapter: Path, **marker) -> None:
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text("{}", encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"weights")
    data = {"adapter_sha256": merge_adapter.adapter_digest(adapter, BASE),
            "files": merge_adapter.manifest(out)}
    data.update(marker)
    (out / merge_adapter.MARKER).write_text(json.dumps(data), encoding="utf-8")


def current(adapter: Path, out: Path) -> int:
    return merge_adapter.main(["--adapter", str(adapter), "--out", str(out), "--is-current"])


def test_merged_dir_is_current_only_with_matching_marker(tmp_path: Path):
    adapter = full_adapter(tmp_path / "run", b"new")
    out = tmp_path / "run" / "merged"
    out.mkdir()
    # Каталог без метки — остаток прерванного или старого вплавления.
    assert current(adapter, out) == 1
    merged_dir(out, adapter, adapter_sha256="старый")
    assert current(adapter, out) == 1
    merged_dir(out, adapter)
    assert current(adapter, out) == 0
    # Переобучили — метка больше не подходит.
    (adapter / "adapter_model.safetensors").write_bytes(b"newer")
    assert current(adapter, out) == 1


def test_marker_without_model_files_is_not_current(tmp_path: Path):
    # Задача 022: одна метка без весов проходила проверку.
    adapter = full_adapter(tmp_path / "run")
    out = tmp_path / "run" / "merged"
    out.mkdir(parents=True)
    (out / merge_adapter.MARKER).write_text(json.dumps({
        "adapter_sha256": merge_adapter.adapter_digest(adapter, BASE), "files": {}}),
        encoding="utf-8")
    assert current(adapter, out) == 1


def test_truncated_or_missing_file_is_not_current(tmp_path: Path):
    adapter = full_adapter(tmp_path / "run")
    out = tmp_path / "run" / "merged"
    merged_dir(out, adapter)
    (out / "model.safetensors").write_bytes(b"w")
    assert current(adapter, out) == 1
    merged_dir(out, adapter)
    (out / "config.json").unlink()
    assert current(adapter, out) == 1


def test_changed_tokenizer_or_base_is_not_current(tmp_path: Path):
    adapter = full_adapter(tmp_path / "run")
    (adapter / "tokenizer_config.json").write_text('{"a": 1}', encoding="utf-8")
    out = tmp_path / "run" / "merged"
    merged_dir(out, adapter)
    assert current(adapter, out) == 0
    (adapter / "tokenizer_config.json").write_text('{"a": 2}', encoding="utf-8")
    assert current(adapter, out) == 1
    merged_dir(out, adapter)
    args = ["--adapter", str(adapter), "--out", str(out), "--is-current", "--base", "другая"]
    assert merge_adapter.main(args) == 1


def test_dotted_targets_match_like_peft():
    name = "model.layers.0.self_attn.q_proj.weight"
    assert merge_adapter.is_target(name, ["self_attn.q_proj"])
    assert merge_adapter.is_target(name, ["model.layers.0.self_attn.q_proj"])
    assert not merge_adapter.is_target(name, ["attn.q_proj"])


@pytest.mark.parametrize("extra", [
    {"target_modules": "all-linear"},
    {"layers_to_transform": [0]},
    {"exclude_modules": ["q_proj"]},
    {"modules_to_save": ["lm_head"]},
])
def test_unsupported_peft_settings_are_refused(extra):
    config = {"target_modules": ["q_proj"], **extra}
    with pytest.raises(ValueError):
        merge_adapter.targets_of(config)


def test_saved_config_is_checked_against_the_base_format():
    # Результат упакован в формат базы (задача 023): ожидается её
    # архитектура и dtype, в котором реально шло вплавление; dtype может
    # лежать в text_config.
    arch = ["Qwen3_5ForConditionalGeneration"]
    full = {"architectures": arch, "text_config": {"dtype": "bfloat16"}}
    assert merge_adapter.check_saved(full, arch, "bfloat16") == []
    assert len(merge_adapter.check_saved(full, arch, "float32")) == 1
    text = {"architectures": ["Qwen3_5ForCausalLM"], "dtype": "bfloat16"}
    assert len(merge_adapter.check_saved(text, arch, "bfloat16")) == 1


# ---- упаковка в формат базы (задача 023)


class FakeTensor:
    def __init__(self, shape, dtype="bf16", tag="base"):
        self.shape, self.dtype, self.tag = tuple(shape), dtype, tag

    def detach(self):
        return self

    def contiguous(self):
        return self

    def cpu(self):
        return self


def fake_base(tmp_path: Path, shards: dict[str, dict[str, FakeTensor]], index: bool = True,
              **config) -> tuple[Path, dict]:
    base = tmp_path / "base"
    base.mkdir()
    config.setdefault("architectures", ["Qwen3_5ForConditionalGeneration"])
    (base / "config.json").write_text(json.dumps(config), encoding="utf-8")
    (base / "preprocessor_config.json").write_text("{}", encoding="utf-8")
    if index:
        weight_map = {key: shard for shard, keys in shards.items() for key in keys}
        (base / merge_adapter.INDEX).write_text(json.dumps({"weight_map": weight_map}),
                                                encoding="utf-8")
    store = {}
    for shard, tensors in shards.items():
        (base / shard).write_bytes(b"x")
        store[str(base / shard)] = tensors
    return base, store


TARGETS = ["q_proj", "v_proj"]


def run_pack(tmp_path, base, store, state, targets=TARGETS):
    out = tmp_path / "out"
    out.mkdir()
    written = {}
    replaced = merge_adapter.pack_like_base(
        state, base, out, targets=targets,
        read_shard=lambda path: store[str(path)],
        write_shard=lambda tensors, path: written.__setitem__(path.name, tensors),
    )
    return replaced, written, out


# Раскладка как у Qwen3.5-4B (заголовки файлов весов с хаба, задача 024):
# два файла, lm_head нет (эмбеддинги связаны), visual и mtp рядом с
# текстом, A_log и linear_attn.norm — float32.
SHARDS = {
    "model.safetensors-00001-of-00002.safetensors": {
        "model.language_model.embed_tokens.weight": FakeTensor((10, 4)),
        "model.language_model.layers.3.self_attn.q_proj.weight": FakeTensor((4, 4)),
        "model.language_model.layers.3.mlp.up_proj.weight": FakeTensor((8, 4)),
    },
    "model.safetensors-00002-of-00002.safetensors": {
        "model.language_model.norm.weight": FakeTensor((4,)),
        "model.language_model.layers.0.linear_attn.A_log": FakeTensor((2,), dtype="fp32"),
        "model.visual.blocks.0.attn.qkv.weight": FakeTensor((6, 4)),
        "mtp.layers.0.self_attn.q_proj.weight": FakeTensor((4, 4)),
    },
}


def merged_state(**override):
    # Модель в памяти целиком bf16, lm_head связан с эмбеддингами.
    state = {
        "model.embed_tokens.weight": FakeTensor((10, 4), tag="merged"),
        "model.layers.3.self_attn.q_proj.weight": FakeTensor((4, 4), tag="merged"),
        "model.layers.3.mlp.up_proj.weight": FakeTensor((8, 4), tag="merged"),
        "model.norm.weight": FakeTensor((4,), tag="merged"),
        "model.layers.0.linear_attn.A_log": FakeTensor((2,), tag="merged"),
        "lm_head.weight": FakeTensor((10, 4), tag="merged"),
    }
    state.update(override)
    return state


def all_keys(shards=SHARDS):
    return [key for tensors in shards.values() for key in tensors]


def test_text_keys_map_into_the_base_layout():
    assert merge_adapter.text_key_for("model.language_model.layers.3.mlp.up_proj.weight") \
        == "model.layers.3.mlp.up_proj.weight"
    assert merge_adapter.text_key_for("lm_head.weight") == "lm_head.weight"
    assert merge_adapter.text_key_for("model.visual.merger.weight") is None
    assert merge_adapter.text_key_for("mtp.fc.weight") is None


def test_pack_replaces_only_trained_weights(tmp_path: Path):
    base, store = fake_base(tmp_path, SHARDS)
    replaced, written, _ = run_pack(tmp_path, base, store, merged_state())
    assert replaced == 1  # только q_proj: up_proj не в целевых
    assert set(written) == set(SHARDS)
    for shard, tensors in SHARDS.items():
        assert set(written[shard]) == set(tensors)
    first = written["model.safetensors-00001-of-00002.safetensors"]
    assert first["model.language_model.layers.3.self_attn.q_proj.weight"].tag == "merged"
    assert first["model.language_model.layers.3.mlp.up_proj.weight"].tag == "base"
    second = written["model.safetensors-00002-of-00002.safetensors"]
    # float32 из базы не проходит через bf16 модели в памяти.
    a_log = second["model.language_model.layers.0.linear_attn.A_log"]
    assert a_log.tag == "base" and a_log.dtype == "fp32"
    # Визуальная часть и MTP — из базы, даже с тем же именем модуля.
    assert second["model.visual.blocks.0.attn.qkv.weight"].tag == "base"
    assert second["mtp.layers.0.self_attn.q_proj.weight"].tag == "base"


def test_identity_pack_replaces_nothing_but_checks_keys(tmp_path: Path):
    base, store = fake_base(tmp_path, SHARDS)
    replaced, written, _ = run_pack(tmp_path, base, store, merged_state(), targets=None)
    assert replaced == 0
    assert all(t.tag == "base" for tensors in written.values() for t in tensors.values())
    state = merged_state()
    del state["model.norm.weight"]
    other = tmp_path / "second"
    other.mkdir()
    with pytest.raises(ValueError, match="не сходятся"):
        run_pack(other, base, store, state, targets=None)


def test_pack_without_index_uses_the_single_file(tmp_path: Path):
    single = {"model.safetensors": {k: t for tensors in SHARDS.values() for k, t in tensors.items()}}
    base, store = fake_base(tmp_path, single, index=False)
    replaced, written, _ = run_pack(tmp_path, base, store, merged_state())
    assert replaced == 1 and set(written) == {"model.safetensors"}


def test_pack_refuses_lost_or_extra_layers(tmp_path: Path):
    base, store = fake_base(tmp_path, SHARDS)
    state = merged_state()
    del state["model.layers.3.self_attn.q_proj.weight"]
    with pytest.raises(ValueError, match="не сходятся"):
        run_pack(tmp_path, base, store, state)
    extra = merged_state(**{"model.layers.4.self_attn.q_proj.weight": FakeTensor((4, 4))})
    with pytest.raises(ValueError, match="не сходятся"):
        merge_adapter.plan_pack(all_keys(), extra)


def test_pack_refuses_shape_mismatch_even_for_copied_weights(tmp_path: Path):
    base, store = fake_base(tmp_path, SHARDS)
    wrong = merged_state(**{"model.norm.weight": FakeTensor((5,), tag="merged")})
    with pytest.raises(ValueError, match="model.language_model.norm.weight"):
        run_pack(tmp_path, base, store, wrong)


def test_pack_refuses_dtype_mismatch_of_a_trained_weight(tmp_path: Path):
    base, store = fake_base(tmp_path, SHARDS)
    fp32 = merged_state(**{"model.layers.3.self_attn.q_proj.weight": FakeTensor((4, 4), dtype="fp32")})
    with pytest.raises(ValueError, match="fp32"):
        run_pack(tmp_path, base, store, fp32)


def test_head_must_be_written_when_base_has_one(tmp_path: Path):
    # lm_head без места допустим, только если в базе его нет (связанные
    # эмбеддинги); при голове в базе она обязана найтись в модели.
    with_head = all_keys() + ["lm_head.weight"]
    plan = merge_adapter.plan_pack(with_head, merged_state())
    assert plan["lm_head.weight"] == "lm_head.weight"
    state = merged_state()
    del state["lm_head.weight"]
    with pytest.raises(ValueError, match="не сходятся"):
        merge_adapter.plan_pack(with_head, state)
    assert "lm_head.weight" not in merge_adapter.plan_pack(all_keys(), merged_state())


def test_text_only_base_has_nothing_to_pack():
    with pytest.raises(ValueError, match="не во что"):
        merge_adapter.plan_pack(["model.layers.0.q.weight"], ["model.layers.0.q.weight"])


def test_side_files_come_from_the_base(tmp_path: Path):
    base, _ = fake_base(tmp_path, SHARDS)
    out = tmp_path / "out"
    out.mkdir()
    merge_adapter.copy_side_files(base, out)
    names = {path.name for path in out.iterdir()}
    assert {"config.json", "preprocessor_config.json", merge_adapter.INDEX} <= names
    assert not any(name.endswith(".safetensors") for name in names)


def test_identity_marker_is_separate_from_adapters(tmp_path: Path):
    out = tmp_path / "runs" / "base-packed"
    out.mkdir(parents=True)
    (out / "config.json").write_text("{}", encoding="utf-8")
    (out / "model.safetensors").write_bytes(b"w")
    (out / merge_adapter.MARKER).write_text(json.dumps({
        "adapter_sha256": merge_adapter.adapter_digest(None, BASE),
        "files": merge_adapter.manifest(out)}), encoding="utf-8")
    args = ["--identity", "--base", BASE, "--out", str(out), "--is-current"]
    assert merge_adapter.main(args) == 0
    assert merge_adapter.main(["--identity", "--base", "другая", "--out", str(out),
                               "--is-current"]) == 1


def test_identity_and_adapter_are_exclusive(tmp_path: Path):
    adapter = full_adapter(tmp_path)
    with pytest.raises(SystemExit):
        merge_adapter.main(["--identity", "--adapter", str(adapter), "--base", BASE,
                            "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        merge_adapter.main(["--identity", "--out", str(tmp_path / "o")])
    with pytest.raises(SystemExit):
        merge_adapter.main(["--out", str(tmp_path / "o")])


def test_serve_path_keeps_the_run_directory(tmp_path: Path):
    out = tmp_path / "rag_textbook" / "runs" / "grpo-main" / "merged"
    # Прежняя подсказка давала /runs/merged — такого каталога в контейнере нет.
    assert merge_adapter.serve_path(out) == "/runs/grpo-main/merged"


def test_tensor_digest_sees_bf16_changes():
    torch = pytest.importorskip("torch")
    weight = torch.zeros(4, 4, dtype=torch.bfloat16)
    changed = weight.clone()
    changed[3, 3] = 1e-3
    assert merge_adapter.tensor_digest(weight) == merge_adapter.tensor_digest(weight.clone())
    assert merge_adapter.tensor_digest(weight) != merge_adapter.tensor_digest(changed)
