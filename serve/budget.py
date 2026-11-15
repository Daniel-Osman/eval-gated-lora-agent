"""Weight and KV budgets read from real files, not from the model name.

Two questions this answers before anyone books a GPU:

* how many parameters sit in each family of the checkpoint (quantizable vs.
  retained in bf16), so a 4-bit artifact's size is a sum of measured shapes;
* how much KV cache and recurrent state one token and one sequence cost on this
  hybrid architecture, so "does it fit 24 GB" is arithmetic on revealed numbers.

The scan reads safetensors **headers only** (mmap), so a 22 GB checkpoint costs
a few seconds and no download.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

# Bitsandbytes stores two 4-bit values per uint8. Every other dtype stores one
# value per element.
PACKED_DTYPE = "U8"

FAMILIES = (
    ("vision", re.compile(r"visual|vision")),
    ("mtp", re.compile(r"\.mtp|mtp_")),
    ("embed_tokens", re.compile(r"embed_tokens")),
    ("lm_head", re.compile(r"lm_head")),
    ("linear_attn", re.compile(r"linear_attn")),
    ("conv1d", re.compile(r"conv1d")),
    ("full_attn", re.compile(r"self_attn")),
    ("mlp", re.compile(r"mlp")),
)


class BudgetError(RuntimeError):
    """A budget that cannot be computed from what is on disk."""


def family_of(name: str) -> str:
    for label, pattern in FAMILIES:
        if pattern.search(name):
            return label
    return "other"


def _shards(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    index = path / "model.safetensors.index.json"
    if index.is_file():
        names = sorted(set(json.loads(index.read_text(encoding="utf-8"))["weight_map"].values()))
        return [path / name for name in names]
    shards = sorted(path.glob("*.safetensors"))
    if not shards:
        raise BudgetError(f"no safetensors under {path}")
    return shards


def scan(path: str | Path) -> dict:
    """Parameter counts per family, from tensor shapes in the file headers.

    The safetensors header is an 8-byte little-endian length followed by JSON, so
    this needs neither torch nor a framework loader: a 22 GB checkpoint costs one
    short read per shard.
    """
    path = Path(path)
    per_family: dict[str, dict] = {}
    quantized: set[str] = set()
    for shard in _shards(path):
        header = _header(shard)
        for key, meta in header.items():
            if key == "__metadata__":
                continue
            # Companion tensors (absmax, quant_map, nested_*) describe the
            # quantization of a `weight`; they are not parameters.
            if not key.endswith(".weight"):
                continue
            shape = meta["shape"]
            dtype = str(meta["dtype"]).upper()
            is_packed = dtype == PACKED_DTYPE
            params = 2 * _prod(shape) if is_packed else _prod(shape)
            family = family_of(key)
            row = per_family.setdefault(family, {"params": 0, "tensors": 0, "packed": 0})
            row["params"] += params
            row["tensors"] += 1
            if is_packed:
                row["packed"] += 1
                quantized.add(family)
    total = sum(row["params"] for row in per_family.values())
    if not total:
        raise BudgetError(f"{path}: no `.weight` tensors found")
    return {
        "path": str(path),
        "shards": [shard.name for shard in _shards(path)],
        "bytes": sum(shard.stat().st_size for shard in _shards(path)),
        "total_params": total,
        "families": {
            name: {
                "params": row["params"],
                "tensors": row["tensors"],
                "quantized": name in quantized,
            }
            for name, row in sorted(per_family.items(), key=lambda item: -item[1]["params"])
        },
    }


def _header(path: Path) -> dict:
    with open(path, "rb") as handle:
        length = int.from_bytes(handle.read(8), "little")
        if not 0 < length < 100 * 2**20:
            raise BudgetError(f"{path}: {length} is not a safetensors header length")
        return json.loads(handle.read(length))


def project(scanned: dict, retained: tuple[str, ...], bits: float = 4.125) -> dict:
    """Bytes for a weight-only quantization, split into quantized and retained.

    `bits` is effective bits per quantized parameter: 4 for the values plus the
    group scale, which is one 16-bit value per group of 128 — 16/128 = 0.125
    bits, so 4.125 in total. An earlier draft said 0.5 bits for the scale, which
    is the ratio for a group of 32.
    """
    quant_params = 0
    retained_params = 0
    retained_detail = {}
    for family, row in scanned["families"].items():
        if family in retained:
            retained_params += row["params"]
            retained_detail[family] = row["params"]
        else:
            quant_params += row["params"]
    quant_bytes = quant_params * bits / 8
    retained_bytes = retained_params * 2
    return {
        "bits": bits,
        "scale_bits_per_param": 16 / 128,
        "quantized_params": quant_params,
        "retained_params": retained_params,
        "retained_families": retained_detail,
        "weights_gib": (quant_bytes + retained_bytes) / 2**30,
        "quantized_gib": quant_bytes / 2**30,
        "retained_gib": retained_bytes / 2**30,
    }


def kv_budget(config: dict) -> dict:
    """Per-token KV, and the per-sequence state a hybrid model allocates.

    The KV figure is only the full-attention half. The linear-attention half
    keeps a fixed recurrent state and a convolution state per sequence, and the
    config says the state is float32 (`mamba_ssm_dtype`), not bf16 — which is
    the difference between 72 MiB and roughly 151 MiB per sequence.
    """
    text = config.get("text_config", config)
    layers = text["num_hidden_layers"]
    interval = text.get("full_attention_interval") or 1
    full_layers = sum(1 for kind in text.get("layer_types", []) if kind == "full_attention")
    if not full_layers:
        full_layers = layers // interval
    linear_layers = layers - full_layers
    kv_heads = text["num_key_value_heads"]
    head_dim = text["head_dim"]
    dtype_bytes = 2
    per_token_per_layer = 2 * kv_heads * head_dim * dtype_bytes
    per_token = full_layers * per_token_per_layer

    # Recurrent state: value heads x key dim x value dim, one per linear layer,
    # in the dtype the config names for the scan.
    state_dtype = str(text.get("mamba_ssm_dtype", "float32")).lower()
    state_bytes_per_element = {"float32": 4, "bfloat16": 2, "float16": 2}.get(state_dtype, 4)
    value_heads = text.get("linear_num_value_heads", 0)
    key_head_dim = text.get("linear_key_head_dim", 0)
    value_head_dim = text.get("linear_value_head_dim", 0)
    state = linear_layers * value_heads * key_head_dim * value_head_dim * state_bytes_per_element

    # Convolution state: the causal conv keeps `kernel - 1` taps of every input
    # channel, per linear layer.
    conv_channels = (
        2 * text.get("linear_num_key_heads", 0) * key_head_dim
        + value_heads * value_head_dim
    )
    kernel = text.get("linear_conv_kernel_dim", 0)
    conv = linear_layers * conv_channels * max(kernel - 1, 0) * state_bytes_per_element

    total = state + conv
    return {
        "layers": layers,
        "full_attention_layers": full_layers,
        "linear_attention_layers": linear_layers,
        "kv_heads": kv_heads,
        "head_dim": head_dim,
        "kv_bytes_per_token": per_token,
        "kv_kib_per_token": per_token / 1024,
        "state_dtype": state_dtype,
        "state_bytes_per_sequence": state,
        "state_mib_per_sequence": state / 2**20,
        "conv_bytes_per_sequence": conv,
        "conv_mib_per_sequence": conv / 2**20,
        "sequence_state_mib": total / 2**20,
        "kv_gib_per_1000_tokens": per_token * 1000 / 2**30,
        "batch8_state_gib": 8 * total / 2**30,
        "batch256_state_gib": 256 * total / 2**30,
    }


def _prod(shape) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
