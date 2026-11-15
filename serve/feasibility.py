"""Stage 6 slice 1 — can the locked serving target serve this adapter at all?

Measurement only. No deploy, no endpoint, no demo, no commit. Each paid step
prints its worst-case dollar bound at list rates before it starts, runs with
`retries=0`, and reports what it measured; a step that fails names the route that
failed, never the target.

The locked target is Modal vLLM on an A10 24 GB (`SPEC.md` §2). The trained
artifact is a bitsandbytes 4-bit QLoRA adapter, which vLLM cannot read, so the
path is merge → requantize → load.

    modal run serve/feasibility.py::scan_cli      # CPU  2 cores /   8 GiB
    modal run serve/feasibility.py::merge_cli     # CPU  8 cores / 112 GiB
    modal run serve/feasibility.py::quant_cli     # A100 80 GB
    modal run serve/feasibility.py::fit_cli       # A10 24 GB
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

from serve import budget as budget_mod

REPO_ROOT = Path(__file__).resolve().parents[1]

VOLUME_NAME = "eval-gated-lora-agent"
VOLUME_MOUNT = "/vol"
BASE_SNAPSHOT = (
    f"{VOLUME_MOUNT}/hf/hub/models--unsloth--Qwen3.8-27B-unsloth-bnb-4bit"
    "/snapshots/8aa5f05d26b7205477066e1449e0af13f762a299"
)
RUN_ID = "full-20261114T201900Z"
ADAPTER_DIR = f"{VOLUME_MOUNT}/runs/{RUN_ID}/adapter"
WORK_DIR = f"{VOLUME_MOUNT}/serve/slice1"
MERGED_DIR = f"{WORK_DIR}/merged-bf16"
QUANT_DIR = f"{WORK_DIR}/awq-w4a16"

DATASET_LOCAL = REPO_ROOT / "data" / "sft" / "sft.jsonl"
DATASET_REMOTE = "/root/sft.jsonl"
BENCH_LOCAL = REPO_ROOT / "bench"
BENCH_REMOTE = "/root/bench"
REPO_REMOTE = "/root"  # shopagent.bench takes the repo root, then bench/cases under it

EXPECTED_ADAPTER_PARAMS = 116_727_808
EXPECTED_ADAPTER_TENSORS = 992  # 496 A/B pairs

# Modal list rates, read from modal.com/pricing on 2026-11-15.
GPU_USD_PER_SECOND = {"A100-80GB": 0.000694, "A100-40GB": 0.000583, "A10": 0.000306}
GPU_USD_PER_HOUR = {"A100-80GB": 2.4984, "A100-40GB": 2.0988, "A10": 1.1016}
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222

SCAN_CPU, SCAN_MIB, SCAN_TIMEOUT = 2.0, 8 * 1024, 600
MERGE_CPU, MERGE_MIB, MERGE_TIMEOUT = 8.0, 112 * 1024, 1500
QUANT_GPU, QUANT_CPU, QUANT_MIB, QUANT_TIMEOUT = "A100-80GB", 8.0, 96 * 1024, 1800
FIT_GPU, FIT_CPU, FIT_MIB, FIT_TIMEOUT = "A10", 4.0, 32 * 1024, 1200

# Families kept out of the 4-bit weights. Everything else is quantized — the MLP
# and both attention stacks, linear attention included — because the A10 budget
# needs it. `conv1d` (a Conv1d, not a Linear) stays bf16 inside that family.
RETAINED_FAMILIES = ("vision", "embed_tokens", "lm_head", "mtp")
AWQ_IGNORE = [
    "lm_head",
    "re:.*embed_tokens.*",
    "re:.*visual\\..*",
    "re:.*\\.mtp\\..*",
    "re:.*conv1d.*",
]

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
app = modal.App("eval-gated-lora-agent-serve-feasibility")


def common_env() -> dict:
    return {
        "HF_HOME": f"{VOLUME_MOUNT}/hf",
        "PYTHONUNBUFFERED": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }


scan_image = modal.Image.debian_slim(python_version="3.12").add_local_python_source("serve")

merge_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
        "peft==0.21.3",
        "kernels>=0.11.1",
        extra_options="--index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple",
    )
    .env(common_env())
    .add_local_python_source("serve")
)

quant_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "torch==2.14.1",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "datasets==4.8.5",
        "torchvision",
        "llmcompressor",
    )
    .env(common_env())
    .add_local_python_source("serve")
    .add_local_file(str(DATASET_LOCAL), DATASET_REMOTE)
)

serve_image = (
    # vLLM 0.33.0 JIT-compiles a kernel for this hybrid architecture at engine
    # init and refuses to start without nvcc. A devel image is the reliable way
    # to have one; `flashinfer-cubin`, `ninja` and build-essential were not
    # enough, which is itself the finding.
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install(
        "vllm",
        "huggingface_hub",
        "torchvision",
        "pillow",
        "ninja",
    )
    .apt_install("build-essential")
    .env(common_env())
    .add_local_python_source("serve", "shopagent")
    .add_local_dir(str(BENCH_LOCAL), BENCH_REMOTE)
    .add_local_file(str(DATASET_LOCAL), DATASET_REMOTE)
)


def bound(gpu: str | None, cores: float, mib: int, timeout: float) -> dict:
    """Worst case at the timeout, at list rates. Printed before every paid run."""
    gpu_usd = GPU_USD_PER_SECOND[gpu] * timeout if gpu else 0.0
    cpu_usd = cores * CPU_USD_PER_CORE_SECOND * timeout
    memory_usd = (mib / 1024) * MEMORY_USD_PER_GIB_SECOND * timeout
    total = gpu_usd + cpu_usd + memory_usd
    return {
        "gpu": gpu,
        "timeout_seconds": timeout,
        "worst_case_usd": round(total, 4),
        "gpu_usd": round(gpu_usd, 4),
        "cpu_usd": round(cpu_usd, 4),
        "memory_usd": round(memory_usd, 4),
        "hourly_usd": round(total / (timeout / 3600), 4),
    }


def spent(gpu: str | None, cores: float, mib: int, seconds: float) -> dict:
    """What the wall time cost at the same list rates. An estimate, never a bill."""
    gpu_usd = GPU_USD_PER_SECOND[gpu] * seconds if gpu else 0.0
    cpu_usd = cores * CPU_USD_PER_CORE_SECOND * seconds
    memory_usd = (mib / 1024) * MEMORY_USD_PER_GIB_SECOND * seconds
    return {
        "seconds": round(seconds, 1),
        "gpu_usd": round(gpu_usd, 4),
        "cpu_usd": round(cpu_usd, 4),
        "memory_usd": round(memory_usd, 4),
        "total_usd": round(gpu_usd + cpu_usd + memory_usd, 4),
        "kind": "estimated from wall time at list rates, excludes Volume commit",
    }


def peak_rss_gib() -> float:
    # `resource` is Unix-only, and the Modal client imports this file on Windows
    # to read the app definition.
    try:
        import resource
    except ModuleNotFoundError:
        return float("nan")
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20


def announce(name: str, row: dict) -> None:
    print(
        f"{name} bound: worst case ${row['worst_case_usd']:.4f} at ${row['hourly_usd']:.4f}/hr "
        f"(gpu ${row['gpu_usd']:.4f} + cpu ${row['cpu_usd']:.4f} + memory ${row['memory_usd']:.4f}), "
        f"timeout {row['timeout_seconds']:.0f}s, retries 0",
        flush=True,
    )


# --------------------------------------------------------------------------
# 1. scan — the weight and KV budget, from headers, before any GPU is booked
# --------------------------------------------------------------------------


@app.function(image=scan_image, cpu=SCAN_CPU, memory=SCAN_MIB, timeout=SCAN_TIMEOUT, volumes={VOLUME_MOUNT: volume}, retries=0)
def scan() -> dict:
    announce("scan", bound(None, SCAN_CPU, SCAN_MIB, SCAN_TIMEOUT))
    started = time.time()
    scanned = budget_mod.scan(BASE_SNAPSHOT)
    config = json.loads((Path(BASE_SNAPSHOT) / "config.json").read_text(encoding="utf-8"))
    print(f"checkpoint: {scanned['total_params']:,} parameters in {len(scanned['shards'])} shard(s)")
    for family, row in scanned["families"].items():
        print(
            f"  {family:<14} {row['params']:>13,} params  "
            f"{'packed 4-bit' if row['quantized'] else 'bf16':<12} {row['tensors']:>4} tensors"
        )
    projection = budget_mod.project(scanned, retained=RETAINED_FAMILIES)
    print(
        f"projection: {projection['weights_gib']:.2f} GiB at {projection['bits']} bits/param — "
        f"quantized {projection['quantized_gib']:.2f} GiB ({projection['quantized_params']:,} params), "
        f"retained {projection['retained_gib']:.2f} GiB ({', '.join(projection['retained_families'])})"
    )
    alternative = budget_mod.project(scanned, retained=RETAINED_FAMILIES + ("linear_attn",))
    print(
        f"projection if linear attention were kept in bf16: {alternative['weights_gib']:.2f} GiB "
        f"(+{alternative['weights_gib'] - projection['weights_gib']:.2f} GiB) — the A10 comparison"
    )
    kv = budget_mod.kv_budget(config)
    print(
        f"kv: {kv['full_attention_layers']}/{kv['layers']} full-attention layers, {kv['kv_heads']} kv heads, "
        f"head_dim {kv['head_dim']} → {kv['kv_kib_per_token']:.0f} KiB/token"
    )
    print(
        f"state: {kv['state_dtype']} recurrent state {kv['state_mib_per_sequence']:.1f} MiB + conv state "
        f"{kv['conv_mib_per_sequence']:.2f} MiB = {kv['sequence_state_mib']:.1f} MiB per sequence; "
        f"batch 8 costs {kv['batch8_state_gib']:.2f} GiB, batch 256 costs {kv['batch256_state_gib']:.1f} GiB"
    )
    usable = 24 * 0.92
    headroom = usable - projection["weights_gib"]
    tokens = headroom / (kv["kv_bytes_per_token"] / 2**30)
    print(
        f"a10: {usable:.2f} GiB usable at 0.92 of 24 GB; weights {projection['weights_gib']:.2f} GiB leave "
        f"{headroom:.2f} GiB before graphs and workspace → about {tokens:,.0f} kv tokens"
    )
    result = {
        "scanned": scanned,
        "projection": projection,
        "projection_linear_attn_retained": alternative,
        "kv": kv,
        "a10_headroom_gib": round(headroom, 3),
        "a10_kv_tokens": round(tokens),
        "cost": spent(None, SCAN_CPU, SCAN_MIB, time.time() - started),
    }
    print(f"scan: {json.dumps(result['cost'])}", flush=True)
    return result


# --------------------------------------------------------------------------
# 2. merge — dequantize explicitly, merge, prove one layer, save 16-bit
# --------------------------------------------------------------------------


class MergeError(RuntimeError):
    """A merge that must stop. Always names what failed."""


def set_submodule(root, dotted: str, module) -> None:
    parent_path, _, leaf = dotted.rpartition(".")
    parent = root.get_submodule(parent_path) if parent_path else root
    setattr(parent, leaf, module)


def one_layer_proof() -> dict:
    """Prove the dequantize op and its per-value cost before the whole model.

    A synthetic bf16 weight goes through the same quantize → dequantize path the
    checkpoint will, on this container's bitsandbytes build. If the op is
    CUDA-only, this raises here, not forty minutes into a 54 GB load.
    """
    import bitsandbytes as bnb
    import torch

    values = 1024 * 5120
    weight = torch.randn(1024, 5120, dtype=torch.bfloat16)
    started = time.time()
    packed, state = bnb.functional.quantize_4bit(
        weight, blocksize=64, quant_type="nf4", compress_statistics=True, quant_storage=torch.uint8
    )
    dense = bnb.functional.dequantize_4bit(packed, state).to(torch.bfloat16)
    elapsed = time.time() - started
    return {
        "values": values,
        "seconds": round(elapsed, 4),
        "seconds_per_billion_values": round(elapsed / (values / 1e9), 2),
        "mean_abs_error": round((dense.float() - weight.float()).abs().mean().item(), 6),
        "packed_bytes": packed.numel(),
        "bnb": bnb.__version__,
        "torch": torch.__version__,
    }


def bnb_dequantize(packed, state):
    """One 4-bit tensor to bf16, through bitsandbytes' own kernel."""
    import bitsandbytes as bnb
    import torch

    return bnb.functional.dequantize_4bit(packed, state).to(torch.bfloat16)


def dequantize_model(model) -> dict:
    """Turn every 4-bit linear into a bf16 linear, explicitly, freeing as it goes.

    PEFT's `merge_and_unload` cannot do this on a bnb checkpoint: it would add a
    delta into packed uint8 weights. So the dequantize is its own step.
    """
    import bitsandbytes as bnb
    import torch
    from bitsandbytes.nn import Linear4bit

    started = time.time()
    replaced = 0
    for name, module in list(model.named_modules()):
        if not isinstance(module, Linear4bit):
            continue
        weight = module.weight
        state = getattr(weight, "quant_state", None)
        if state is None:
            raise MergeError(f"dequantize: {name} has no quant_state; the checkpoint did not load as 4-bit")
        dense = bnb.functional.dequantize_4bit(weight.data, state).to(torch.bfloat16)
        replacement = torch.nn.Linear(
            module.in_features,
            module.out_features,
            bias=module.bias is not None,
            dtype=torch.bfloat16,
            device=weight.device,
        )
        with torch.no_grad():
            replacement.weight.copy_(dense)
            if module.bias is not None:
                replacement.bias.copy_(module.bias.data.to(torch.bfloat16))
        set_submodule(model, name, replacement)
        del module, weight, state, dense
        replaced += 1
    # Embeddings, lm_head and the vision tower were never 4-bit; the bnb loader
    # keeps them in the checkpoint's fp16 whatever dtype is requested, and
    # `model.to(dtype)` refuses on a model that still carries a quantizer. Cast
    # them one parameter at a time.
    cast = 0
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if parameter.is_floating_point() and parameter.dtype != torch.bfloat16:
                parameter.data = parameter.data.to(torch.bfloat16)
                cast += 1
    return {
        "layers_replaced": replaced,
        "parameters_cast": cast,
        "seconds": round(time.time() - started, 1),
    }


def remaining_4bit(model) -> int:
    from bitsandbytes.nn import Linear4bit

    return sum(1 for module in model.modules() if isinstance(module, Linear4bit))


def adapter_tensor_proof(adapter_dir: str) -> dict:
    """The adapter on disk is the run's own weights, counted from shapes."""
    from safetensors.torch import load_file

    path = Path(adapter_dir) / "adapter_model.safetensors"
    saved = load_file(str(path))
    lora = {key: tensor for key, tensor in saved.items() if "lora_" in key}
    params = sum(tensor.numel() for tensor in lora.values())
    if params != EXPECTED_ADAPTER_PARAMS:
        raise MergeError(f"adapter: {params:,} lora parameters, expected the trained {EXPECTED_ADAPTER_PARAMS:,}")
    if len(lora) != EXPECTED_ADAPTER_TENSORS:
        raise MergeError(f"adapter: {len(lora)} lora tensors, expected {EXPECTED_ADAPTER_TENSORS}")
    return {"lora_tensors": len(lora), "lora_params": params, "file_bytes": path.stat().st_size}


def module_name_variants(module_name: str) -> list[str]:
    """One module, named as the loaded wrapper does and as the adapter file does."""
    out = [module_name]
    if ".language_model.layers." in module_name:
        out.append(module_name.replace(".language_model.layers.", ".layers."))
    elif ".layers." in module_name:
        out.append(module_name.replace(".layers.", ".language_model.layers.", 1))
    return out


def adapter_module_index(peft_model) -> dict[str, tuple]:
    """Every LoRA-wrapped module in the live model, keyed by its plain name."""
    found = {}
    for name, module in peft_model.named_modules():
        if hasattr(module, "base_layer") and hasattr(module, "lora_A") and hasattr(module, "lora_B"):
            found[name] = module
    return found


def layer_index(module_name: str) -> int:
    match = re.search(r"\.layers\.(\d+)\.", module_name)
    return int(match.group(1)) if match else -1


def resolve_file_key(saved: dict, module_name: str, half: str) -> str | None:
    """The adapter file's key for one half of a module, named either way."""
    for candidate in module_name_variants(module_name):
        key = f"{candidate}.lora_{half}.weight"
        if key in saved:
            return key
    return None


def pick_targets(peft_model, adapter_file: str, count: int = 5) -> list[dict]:
    """Pick modules spread across the depth and prove their factors came from disk.

    One module is a spot check; a partial merge elsewhere in the stack would pass
    it. So this samples across the depth, checks `lora_A` and `lora_B` for each
    against the exact file key, and then requires that **every** adapter tensor in
    the file maps to a module in the model — a key that no module consumes is a
    factor that was silently dropped.
    """
    import torch
    from safetensors.torch import load_file

    saved = load_file(adapter_file)
    modules = adapter_module_index(peft_model)
    if not modules:
        raise MergeError("merge_proof: no module carries both a base layer and a LoRA pair")

    by_layer: dict[int, list[str]] = {}
    for name in modules:
        by_layer.setdefault(layer_index(name), []).append(name)
    depths = sorted(index for index in by_layer if index >= 0)
    if not depths:
        raise MergeError("merge_proof: no adapted module sits under a numbered layer")
    # First layer, last layer, and evenly spaced layers between them.
    wanted = {depths[0], depths[-1]}
    step = max(len(depths) // count, 1)
    wanted.update(depths[::step][:count])
    chosen: list[str] = []
    for depth in sorted(wanted):
        # Prefer a full-attention projection where the layer has one: those are
        # the layers this run dequantized from 4-bit.
        names = sorted(by_layer[depth])
        preferred = [name for name in names if name.endswith(".self_attn.k_proj")] or names
        chosen.append(preferred[0])
        if len(chosen) >= count:
            break

    targets = []
    for name in chosen:
        module = modules[name]
        adapters = list(module.lora_A.keys())
        if not adapters:
            raise MergeError(f"merge_proof: {name} has an empty lora_A dict")
        target = {"module": name, "adapter": adapters[0], "linear": module.base_layer, "layers": []}
        for half, holder in (("A", module.lora_A), ("B", module.lora_B)):
            tensor = holder[adapters[0]].weight
            key = resolve_file_key(saved, name, half)
            if key is None:
                raise MergeError(f"merge_proof: no file key for {name} lora_{half}")
            loaded = tensor.detach().to(torch.float32)
            reference = saved[key].to(torch.float32)
            magnitude = reference.norm().item()
            relative_l2 = (reference - loaded).norm().item() / magnitude if magnitude else float("inf")
            if relative_l2 > 0.05:
                raise MergeError(
                    f"merge_proof: loaded lora_{half} of {name} differs from {key} at relative L2 "
                    f"{relative_l2:.3e}; the adapter did not load"
                )
            target[half] = tensor
            target["layers"].append(
                {"half": half, "file_key": key, "relative_l2": relative_l2, "dtype": str(saved[key].dtype)}
            )
        target["base"] = module.base_layer.weight.detach().float().clone()
        targets.append(target)

    # Key accounting: every factor on disk must belong to a module that exists.
    consumed = 0
    for key in saved:
        if ".lora_A.weight" not in key and ".lora_B.weight" not in key:
            continue
        module_name = key.rsplit(".lora_", 1)[0]
        if any(
            candidate in modules for candidate in module_name_variants(module_name)
        ) or module_name in modules:
            consumed += 1
    total = sum(1 for key in saved if ".lora_A.weight" in key or ".lora_B.weight" in key)
    print(f"merge_proof: {consumed}/{total} adapter tensors map to a module in the model", flush=True)
    if consumed != total:
        raise MergeError(f"merge_proof: {total - consumed} adapter tensors map to no module")
    return targets


def merged_delta_check(target: dict, scaling: float) -> dict:
    """The four checks bf16 storage supports, on one merged module.

    A delta of ~one bf16 ulp is rounded by the storage dtype, so an element-wise
    equality is impossible by construction and the mean delta sits below the
    rounding noise of its own sum. What is checkable: the weight moved, the
    delta's norm matches `B @ A`, the two correlate, and no element strays past
    the weight's bf16 resolution.
    """
    import torch

    actual = target["linear"].weight.detach().float() - target["base"]
    expected = (target["B"].float() @ target["A"].float()) * scaling
    eps = torch.finfo(torch.bfloat16).eps
    ceiling = 4 * eps * target["base"].abs().max().item()
    moved = actual.abs().max().item()
    expected_norm = expected.norm().item()
    norm_ratio = actual.norm().item() / expected_norm if expected_norm else float("inf")
    max_error = (actual - expected).abs().max().item()
    cosine = torch.nn.functional.cosine_similarity(
        actual.flatten().unsqueeze(0), expected.flatten().unsqueeze(0)
    ).item()
    numbers = {
        "module": target["module"],
        "moved": moved,
        "norm_ratio": norm_ratio,
        "cosine": cosine,
        "max_element_error": max_error,
        "bf16_ceiling": ceiling,
    }
    print(
        f"merge_proof: {target['module']} moved {moved:.3e}, delta norm ratio {norm_ratio:.3f}, "
        f"cosine {cosine:.3f}, max element error {max_error:.3e} against bf16 ceiling {ceiling:.3e}",
        flush=True,
    )
    if moved == 0:
        raise MergeError(f"merge_proof: {target['module']} did not move; the adapter was not merged")
    if not 0.8 <= norm_ratio <= 1.25:
        raise MergeError(
            f"merge_proof: {target['module']} delta norm is {norm_ratio:.3f} of B@A's; the merge is not B@A"
        )
    if cosine < 0.5:
        raise MergeError(f"merge_proof: {target['module']} delta correlates with B@A at only {cosine:.3f}")
    if max_error > ceiling:
        raise MergeError(
            f"merge_proof: {target['module']} element error {max_error:.3e} exceeds bf16 resolution {ceiling:.3e}"
        )
    return numbers


def unwrapped_state_dict(model) -> dict:
    """The merged weights under the names the base model would have produced.

    The base model is still inside PEFT's wrapper when the merge is written, so
    its state dict carries `.base_layer` in every adapted path and still holds the
    LoRA factors. Both are dropped: the factors are already added into the base
    layer, and a checkpoint that kept them would ask a server to add them twice.
    """
    clean = {}
    dropped = 0
    for key, tensor in model.state_dict().items():
        if ".lora_A." in key or ".lora_B." in key:
            dropped += 1
            continue
        name = key.replace(".base_layer.", ".")
        for prefix in ("base_model.model.", "base_model."):
            while name.startswith(prefix):
                name = name[len(prefix) :]
        clean[name] = tensor
    if dropped != EXPECTED_ADAPTER_TENSORS:
        raise MergeError(f"merge: dropped {dropped} lora tensors, expected {EXPECTED_ADAPTER_TENSORS}")
    if any("lora" in name.lower() or "base_layer" in name for name in clean):
        raise MergeError("merge: the state dict still carries wrapper names")
    return clean


def merged_weights(peft_model, scaling: float):
    """Add each adapter's `B @ A` into its base layer, in place.

    Returns the unwrapped base model, with the count of merged modules attached
    as an attribute for the log. This is the arithmetic `merge_and_unload` does,
    written out because that call does not survive this architecture.
    """
    import torch

    count = 0
    with torch.no_grad():
        for _, module in peft_model.named_modules():
            if not (
                hasattr(module, "base_layer")
                and hasattr(module, "lora_A")
                and hasattr(module, "lora_B")
            ):
                continue
            for adapter in module.lora_A.keys():
                a = module.lora_A[adapter].weight
                b = module.lora_B[adapter].weight
                delta = (b.to(torch.float32) @ a.to(torch.float32)).mul(scaling)
                module.base_layer.weight.data += delta.to(module.base_layer.weight.dtype)
                count += 1
    base = peft_model.get_base_model()
    base._merged_modules = count
    return base


def save_state_dict(state: dict, out: Path, max_bytes: int) -> None:
    """Write a state dict as safetensors shards plus the index vLLM reads."""
    import torch
    from safetensors.torch import save_file

    shards: list[dict] = []
    current: dict = {}
    size = 0
    seen: set[tuple[int, tuple[int, ...], str]] = set()
    for key, tensor in state.items():
        tensor = tensor.detach().to("cpu").contiguous()
        signature = (tensor.data_ptr(), tuple(tensor.shape), str(tensor.dtype))
        if signature in seen:
            raise MergeError(f"save_state_dict: {key} shares storage with an earlier tensor")
        seen.add(signature)
        nbytes = tensor.numel() * tensor.element_size()
        if current and size + nbytes > max_bytes:
            shards.append(current)
            current, size = {}, 0
        current[key] = tensor
        size += nbytes
    if current:
        shards.append(current)
    weight_map = {}
    for index, shard in enumerate(shards):
        name = f"model-{index + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file(shard, str(out / name), metadata={"format": "pt"})
        for key in shard:
            weight_map[key] = name
    (out / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": sum(v.numel() * v.element_size() for v in state.values())}, "weight_map": weight_map}, indent=2)
        + "\n",
        encoding="utf-8",
    )


@app.function(
    image=merge_image,
    cpu=MERGE_CPU,
    memory=MERGE_MIB,
    timeout=MERGE_TIMEOUT,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def merge() -> dict:
    """Dequantize the bnb base to bf16, merge the adapter, save 16-bit shards."""
    import torch
    import transformers
    from peft import PeftModel
    from transformers import AutoTokenizer

    announce("merge", bound(None, MERGE_CPU, MERGE_MIB, MERGE_TIMEOUT))
    started = time.time()

    proof = one_layer_proof()
    print(f"one_layer_proof: {json.dumps(proof)}", flush=True)

    scanned = budget_mod.scan(BASE_SNAPSHOT)
    params = scanned["total_params"]
    packed_gib = scanned["bytes"] / 2**30
    dense_gib = params * 2 / 2**30
    print(
        f"memory_model: packed ~{packed_gib:.1f} GiB + bf16 {dense_gib:.1f} GiB = ~{packed_gib + dense_gib:.1f} GiB peak "
        f"against a {MERGE_MIB // 1024} GiB request",
        flush=True,
    )
    predicted = proof["seconds_per_billion_values"] * (params / 1e9)
    print(f"memory_model: dequant predicted ~{predicted:.0f}s for {params:,} parameters", flush=True)

    adapter = adapter_tensor_proof(ADAPTER_DIR)
    print(f"adapter: {json.dumps(adapter)}", flush=True)

    load_started = time.time()
    # Load the architecture the adapter was saved against, not the text-only
    # class. `AutoModelForCausalLM` gives `model.layers.*`, while the adapter's
    # tensors are keyed `model.language_model.layers.*`; PEFT then matches no key
    # and silently keeps its fresh initialisation — a 116,727,808-parameter
    # adapter that is not the trained one. `tests`-style proof below refuses it.
    from transformers import AutoConfig

    architecture = AutoConfig.from_pretrained(BASE_SNAPSHOT).architectures[0]
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise MergeError(f"load: transformers {transformers.__version__} has no {architecture}")
    print(f"load: {architecture}", flush=True)
    model = model_class.from_pretrained(
        BASE_SNAPSHOT, device_map="cpu", low_cpu_mem_usage=True, dtype=torch.bfloat16
    )
    print(f"load: 4-bit base in {time.time() - load_started:.1f}s, peak {peak_rss_gib():.1f} GiB", flush=True)
    if remaining_4bit(model) == 0:
        raise MergeError("load: no 4-bit linear found; the checkpoint did not load through bitsandbytes")
    print(f"model: first modules {[name for name, _ in model.named_modules()][:4]}", flush=True)

    dequant = dequantize_model(model)
    print(
        f"dequantize: {dequant['layers_replaced']} linears to bf16, {dequant['parameters_cast']} other "
        f"parameters cast, {dequant['seconds']}s, peak {peak_rss_gib():.1f} GiB",
        flush=True,
    )
    if remaining_4bit(model):
        raise MergeError(f"dequantize: {remaining_4bit(model)} 4-bit linears remain")
    dtypes = {str(parameter.dtype) for parameter in model.parameters()}
    if dtypes != {"torch.bfloat16"}:
        raise MergeError(f"dequantize: model carries {sorted(dtypes)}, expected bfloat16 only")

    peft_model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    attached = sum(parameter.numel() for name, parameter in peft_model.named_parameters() if "lora_" in name)
    print(f"adapter: attached, {attached:,} lora parameters", flush=True)
    if attached != EXPECTED_ADAPTER_PARAMS:
        raise MergeError(f"adapter: {attached:,} attached, expected {EXPECTED_ADAPTER_PARAMS:,}")

    config = json.loads((Path(ADAPTER_DIR) / "adapter_config.json").read_text(encoding="utf-8"))
    alpha, rank = config["lora_alpha"], config["r"]
    print(f"merge: scaling alpha {alpha} / r {rank} = {alpha / rank}", flush=True)

    targets = pick_targets(peft_model, str(Path(ADAPTER_DIR) / "adapter_model.safetensors"))
    print(
        f"merge_proof: {len(targets)} modules sampled across depth: "
        f"{[target['module'].split('layers.')[-1].split('.')[0] for target in targets]}",
        flush=True,
    )
    # `merge_and_unload` raises KeyError(slice(None, None, -1)) on this model, so
    # the merge is done here: PEFT gave the exact name mapping and the tensors,
    # and the base layers are the same objects the base model holds.
    merge_started = time.time()
    merged = merged_weights(peft_model, alpha / rank)
    merged.eval()
    print(
        f"merge: {merged._merged_modules} modules merged in {time.time() - merge_started:.1f}s, "
        f"peak {peak_rss_gib():.1f} GiB",
        flush=True,
    )

    checks = [merged_delta_check(target, alpha / rank) for target in targets]
    print(f"merge_proof: {len(checks)} of {len(checks)} sampled modules match B@A", flush=True)

    state = unwrapped_state_dict(merged)
    print(f"merge: state dict {len(state)} tensors, e.g. {sorted(state)[:3]}", flush=True)
    if remaining_4bit(merged):
        raise MergeError(f"merge: {remaining_4bit(merged)} 4-bit linears remain")

    out = Path(MERGED_DIR)
    out.mkdir(parents=True, exist_ok=True)
    save_started = time.time()
    # The loader applies its own checkpoint conversion on the way in, so
    # `save_pretrained` tries to reverse it and raises NotImplementedError for
    # this architecture. `_weight_conversions` is what it walks: empty it, and if
    # the save still refuses, write the shards directly.
    save_note = "sharded state dict"
    save_state_dict(state, out, max_bytes=8 * 2**30)
    merged.config.quantization_config = None
    merged.config.save_pretrained(str(out))
    AutoTokenizer.from_pretrained(BASE_SNAPSHOT).save_pretrained(str(out))
    for extra in ("chat_template.jinja", "processor_config.json", "preprocessor_config.json", "generation_config.json"):
        source = Path(BASE_SNAPSHOT) / extra
        if source.is_file() and not (out / extra).is_file():
            (out / extra).write_bytes(source.read_bytes())
    print(f"save: {save_note}", flush=True)
    files = sorted(path.name for path in out.iterdir() if path.is_file())
    written = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    print(f"save: {written / 2**30:.2f} GiB in {len(files)} files, {time.time() - save_started:.1f}s → {out}", flush=True)
    volume.commit()

    elapsed = time.time() - started
    cost = spent(None, MERGE_CPU, MERGE_MIB, elapsed)
    print(f"merge: wall {elapsed:.0f}s, peak {peak_rss_gib():.1f} GiB, {json.dumps(cost)}", flush=True)
    return {
        "merged_dir": str(out),
        "files": files,
        "bytes": written,
        "gib": round(written / 2**30, 3),
        "peak_rss_gib": round(peak_rss_gib(), 1),
        "one_layer_proof": proof,
        "merge_proof_modules": len(checks),
        "merge_proof": checks,
        "adapter": adapter,
        "total_params": params,
        "wall_seconds": round(elapsed, 1),
        "cost": cost,
        "bound": bound(None, MERGE_CPU, MERGE_MIB, MERGE_TIMEOUT),
    }


def plain_name(adapter_key: str) -> str:
    """An adapter key minus PEFT wrappers and the tensor suffix."""
    name = adapter_key
    for prefix in ("base_model.model.", "base_model."):
        while name.startswith(prefix):
            name = name[len(prefix) :]
    if name.endswith(".weight"):
        name = name[: -len(".weight")]
    return name


@app.function(
    image=merge_image,
    cpu=8.0,
    memory=64 * 1024,
    timeout=900,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def verify_merge(sample: int = 8) -> dict:
    """Check the merged checkpoint on the Volume against the base and the adapter.

    Re-merging to prove the merge is expensive; the artifact is already stored, so
    this reads it. For modules spread across the depth it dequantizes the base
    weight for that module only, reads the merged tensor out of the shard, and
    requires `merged - base == (B @ A) * alpha / r` at bf16 resolution. It also
    requires every adapter tensor in the file to map to a weight that exists in
    the merged checkpoint, so a factor dropped anywhere in the stack fails.
    """
    import re

    import torch
    import transformers
    from safetensors import safe_open
    from safetensors.torch import load_file
    from transformers import AutoConfig

    announce("verify_merge", bound(None, 8.0, 64 * 1024, 900))
    started = time.time()
    adapter_file = str(Path(ADAPTER_DIR) / "adapter_model.safetensors")
    saved = load_file(adapter_file)
    config = json.loads((Path(ADAPTER_DIR) / "adapter_config.json").read_text(encoding="utf-8"))
    scaling = config["lora_alpha"] / config["r"]

    index = json.loads((Path(MERGED_DIR) / "model.safetensors.index.json").read_text(encoding="utf-8"))
    weight_map = index["weight_map"]
    print(f"merged: {len(weight_map)} tensors across {len(set(weight_map.values()))} shards", flush=True)
    handles: dict[str, object] = {}

    def merged_tensor(name: str):
        shard = weight_map[name]
        if shard not in handles:
            handles[shard] = safe_open(str(Path(MERGED_DIR) / shard), framework="pt")
        return handles[shard].get_tensor(name)

    architecture = AutoConfig.from_pretrained(BASE_SNAPSHOT).architectures[0]
    model = getattr(transformers, architecture).from_pretrained(
        BASE_SNAPSHOT, device_map="cpu", low_cpu_mem_usage=True, dtype=torch.bfloat16
    )
    from bitsandbytes.nn import Linear4bit

    def base_module(name: str):
        for prefix in ("", "model.", "model.language_model.", "model.model."):
            try:
                return model.get_submodule(prefix + name)
            except AttributeError:
                continue
        raise MergeError(f"verify_merge: no base module for {name}")

    # plain module name -> the key prefix the file uses, which keeps PEFT's
    # `base_model.model.` wrapper. Both are needed: the model answers to the plain
    # path, the file to its own.
    modules: dict[str, str] = {}
    for key in saved:
        if ".lora_A.weight" in key:
            prefix = key[: -len(".lora_A.weight")]
            modules.setdefault(plain_name(prefix), prefix)
    # Sample only layers that carry a full-attention projection: those are the
    # weights this run dequantized from 4-bit, so the check covers dequant plus
    # merge. Depth coverage is over the 16 full-attention layers, and the
    # presence check below still covers every adapted module in the stack.
    depths = sorted(
        {
            int(m.group(1))
            for name in modules
            if ".self_attn." in name and (m := re.search(r"\.layers\.(\d+)\.", name))
        }
    )
    if not depths:
        depths = sorted(
            {int(m.group(1)) for name in modules if (m := re.search(r"\.layers\.(\d+)\.", name))}
        )
    if not depths:
        raise MergeError("verify_merge: no numbered layer in the adapter keys")
    wanted = {depths[0], depths[-1]}
    wanted.update(depths[:: max(len(depths) // sample, 1)][:sample])

    checks = []
    for depth in sorted(wanted):
        names = [name for name in modules if f".layers.{depth}." in name]
        name = next((cand for cand in sorted(names) if cand.endswith(".self_attn.k_proj")), sorted(names)[0])
        prefix = modules[name]
        module = base_module(name)
        if not isinstance(module, Linear4bit):
            continue
        packed = module.weight
        state = getattr(packed, "quant_state", None)
        if state is None:
            raise MergeError(f"verify_merge: {name} is not 4-bit in the base")
        base_weight = bnb_dequantize(packed.data, state).float()
        merged_name = next(
            (candidate + ".weight" for candidate in module_name_variants(name) if candidate + ".weight" in weight_map),
            None,
        )
        if merged_name is None:
            raise MergeError(f"verify_merge: no merged tensor for {name}")
        merged_weight = merged_tensor(merged_name).float()
        a = saved[f"{prefix}.lora_A.weight"].float()
        b = saved[f"{prefix}.lora_B.weight"].float()
        shim = {"module": name, "linear": type("L", (), {"weight": merged_weight})(), "base": base_weight, "A": a, "B": b}
        checks.append(merged_delta_check(shim, scaling))

    missing = [name for name in modules if not any(c + ".weight" in weight_map for c in module_name_variants(name))]
    print(f"verify_merge: {len(checks)} modules checked, {len(missing)} adapter modules absent from the checkpoint", flush=True)
    if missing:
        raise MergeError(f"verify_merge: {len(missing)} adapted modules are missing from the merged checkpoint")
    elapsed = time.time() - started
    cost = spent(None, 8.0, 64 * 1024, elapsed)
    print(f"verify_merge: wall {elapsed:.0f}s, {json.dumps(cost)}", flush=True)
    return {
        "modules_checked": len(checks),
        "checks": checks,
        "adapter_modules": len(modules),
        "merged_tensors": len(weight_map),
        "cost": cost,
    }


# --------------------------------------------------------------------------
# 3. quantize — W4A16 AWQ through llm-compressor, calibration from the SFT text
# --------------------------------------------------------------------------


@app.function(
    image=quant_image,
    gpu=QUANT_GPU,
    cpu=QUANT_CPU,
    memory=QUANT_MIB,
    timeout=QUANT_TIMEOUT,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def quantize() -> dict:
    """Quantize the merged 16-bit model to a vLLM-readable 4-bit artifact."""
    import inspect

    import torch
    import transformers
    from transformers import AutoTokenizer

    announce("quantize", bound(QUANT_GPU, QUANT_CPU, QUANT_MIB, QUANT_TIMEOUT))
    started = time.time()
    import llmcompressor
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    print(f"llm-compressor {llmcompressor.__version__}", flush=True)
    print(f"oneshot signature: {inspect.signature(oneshot)}", flush=True)
    print(f"AWQModifier signature: {inspect.signature(AWQModifier.__init__)}", flush=True)

    merged = budget_mod.scan(MERGED_DIR)
    print(f"merged: {merged['total_params']:,} parameters in {len(merged['shards'])} shard(s)", flush=True)

    from transformers import AutoConfig

    tokenizer = AutoTokenizer.from_pretrained(MERGED_DIR)
    from transformers import AutoProcessor

    # A multimodal checkpoint needs its processor for a calibration dataset; the
    # tokenizer alone is not enough for `oneshot`.
    processor = AutoProcessor.from_pretrained(MERGED_DIR)
    print(f"quantize: processor {type(processor).__name__}", flush=True)
    architecture = AutoConfig.from_pretrained(MERGED_DIR).architectures[0]
    model_class = getattr(transformers, architecture, None)
    if model_class is None:
        raise RuntimeError(f"quantize: transformers {transformers.__version__} has no {architecture}")
    print(f"quantize: loading {architecture}", flush=True)
    model = model_class.from_pretrained(MERGED_DIR, dtype=torch.bfloat16, device_map="cuda")
    gib = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters()) / 2**30
    print(f"model: on gpu, {gib:.2f} GiB of weights", flush=True)

    rows = [json.loads(line) for line in Path(DATASET_REMOTE).read_text(encoding="utf-8").splitlines() if line.strip()]
    texts = [row["text"] for row in rows]
    print(f"calibration: {len(texts)} SFT records available, using 32 at up to 1024 tokens", flush=True)

    from datasets import Dataset

    dataset = Dataset.from_list([{"text": text} for text in texts])
    recipe = AWQModifier(targets=["Linear"], scheme="W4A16", ignore=AWQ_IGNORE)
    print(f"ignore: {AWQ_IGNORE}", flush=True)

    quant_started = time.time()
    oneshot(
        model=model,
        dataset=dataset,
        processor=processor,
        recipe=recipe,
        max_seq_length=1024,
        num_calibration_samples=32,
        output_dir=QUANT_DIR,
    )
    print(f"quantize: done in {time.time() - quant_started:.0f}s", flush=True)
    tokenizer.save_pretrained(QUANT_DIR)
    volume.commit()

    out = Path(QUANT_DIR)
    files = sorted(path.name for path in out.iterdir() if path.is_file())
    written = sum(path.stat().st_size for path in out.iterdir() if path.is_file())
    print(f"artifact: {written / 2**30:.2f} GiB in {len(files)} files → {out}", flush=True)
    elapsed = time.time() - started
    cost = spent(QUANT_GPU, QUANT_CPU, QUANT_MIB, elapsed)
    print(f"quantize: wall {elapsed:.0f}s, {json.dumps(cost)}", flush=True)
    return {
        "quant_dir": str(out),
        "files": files,
        "bytes": written,
        "gib": round(written / 2**30, 3),
        "merged_params": merged["total_params"],
        "ignore": AWQ_IGNORE,
        "llmcompressor": llmcompressor.__version__,
        "wall_seconds": round(elapsed, 1),
        "cost": cost,
        "bound": bound(QUANT_GPU, QUANT_CPU, QUANT_MIB, QUANT_TIMEOUT),
    }


# --------------------------------------------------------------------------
# 4. fit — one fresh vLLM process on an A10, VRAM, cold start, speed, suite
# --------------------------------------------------------------------------


@app.function(
    image=serve_image,
    gpu=FIT_GPU,
    cpu=FIT_CPU,
    memory=FIT_MIB,
    timeout=FIT_TIMEOUT,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def fit() -> dict:
    """Run the probe in a fresh process and relay exactly what it measured."""
    import subprocess
    import sys

    announce("fit", bound(FIT_GPU, FIT_CPU, FIT_MIB, FIT_TIMEOUT))
    started = time.time()
    completed = subprocess.run(
        [sys.executable, "-m", "serve.probe", QUANT_DIR, DATASET_REMOTE, REPO_REMOTE],
        capture_output=True,
        text=True,
        timeout=FIT_TIMEOUT - 120,
    )
    print(completed.stdout, flush=True)
    if completed.stderr:
        print(f"probe stderr:\n{completed.stderr}", flush=True)
    payload = None
    for line in completed.stdout.splitlines():
        if line.startswith("PROBE_JSON "):
            payload = json.loads(line[len("PROBE_JSON ") :])
    elapsed = time.time() - started
    cost = spent(FIT_GPU, FIT_CPU, FIT_MIB, elapsed)
    print(f"fit: wall {elapsed:.0f}s, exit {completed.returncode}, {json.dumps(cost)}", flush=True)
    result = {
        "exit_code": completed.returncode,
        "probe": payload,
        "cost": cost,
        "bound": bound(FIT_GPU, FIT_CPU, FIT_MIB, FIT_TIMEOUT),
    }
    # A failed probe is a failed run. The payload is printed above so the reason
    # survives in the log, but the caller must not see a green exit.
    if completed.returncode != 0:
        raise RuntimeError(
            f"fit: probe exited {completed.returncode}; see probe stdout/stderr above "
            f"(payload present: {payload is not None})"
        )
    return result


# --------------------------------------------------------------------------
# Local entrypoints: the bound is printed on this side, before anything is paid
# --------------------------------------------------------------------------


@app.function(image=serve_image, cpu=2.0, memory=8 * 1024, timeout=900, retries=0)
def versions_serve() -> dict:
    """Force the serve image to build and say what it resolved, before a GPU is booked."""
    import subprocess
    import sys

    import torch
    import transformers
    import vllm

    from vllm.model_executor.models.registry import ModelRegistry

    arch = "Qwen3_5ForConditionalGeneration"
    import pathlib

    package = pathlib.Path(vllm.__file__).parent
    hits = [str(path.relative_to(package)) for path in package.rglob("qwen3_5*.py")][:6]
    try:
        from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

        resolved = f"importable: {Qwen3_5ForConditionalGeneration.__name__}"
    except Exception as exc:  # noqa: BLE001 - the answer is the point
        resolved = f"{type(exc).__name__}: {exc}"[:300]
    print(f"torch {torch.__version__}, transformers {transformers.__version__}, vllm {vllm.__version__}", flush=True)
    print(f"arch {arch}: {resolved}", flush=True)
    print(f"qwen3_5 modules in the wheel: {hits}", flush=True)
    print(f"python {sys.version.split()[0]}", flush=True)
    return json.dumps(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "arch": arch,
            "resolved": resolved,
            "qwen3_5_modules": hits,
            "python": sys.version.split()[0],
        }
    )


@app.function(image=quant_image, cpu=2.0, memory=8 * 1024, timeout=900, retries=0)
def versions_quant() -> dict:
    """Force the quantize image to build and print the API this version actually has."""
    import inspect

    import torch
    import transformers

    import llmcompressor
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    print(f"torch {torch.__version__}, transformers {transformers.__version__}, llmcompressor {llmcompressor.__version__}", flush=True)
    print(f"oneshot{inspect.signature(oneshot)}", flush=True)
    print(f"AWQModifier{inspect.signature(AWQModifier.__init__)}", flush=True)
    try:
        modifier = AWQModifier(targets=["Linear"], scheme="W4A16", ignore=["lm_head"])
        print(f"constructed: {modifier}", flush=True)
        ok = True
    except Exception as exc:  # noqa: BLE001 - the answer is the point
        print(f"construct failed: {type(exc).__name__}: {exc}", flush=True)
        ok = False
    return json.dumps(
        {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "llmcompressor": llmcompressor.__version__,
            "oneshot": str(inspect.signature(oneshot)),
            "AWQModifier": str(inspect.signature(AWQModifier.__init__)),
            "constructs": ok,
        }
    )


@app.local_entrypoint()
def versions_cli():
    announce("versions_serve", {"worst_case_usd": 0.01, "hourly_usd": 0.05, "gpu_usd": 0.0, "cpu_usd": 0.01, "memory_usd": 0.0, "timeout_seconds": 900})
    print("serve", versions_serve.remote())
    print("quant", versions_quant.remote())


@app.local_entrypoint()
def verify_cli():
    announce("verify_merge", bound(None, 8.0, 64 * 1024, 900))
    print(json.dumps(verify_merge.remote(), indent=2))


@app.local_entrypoint()
def scan_cli():
    announce("scan", bound(None, SCAN_CPU, SCAN_MIB, SCAN_TIMEOUT))
    print(json.dumps(scan.remote(), indent=2))


@app.local_entrypoint()
def merge_cli():
    announce("merge", bound(None, MERGE_CPU, MERGE_MIB, MERGE_TIMEOUT))
    print(json.dumps(merge.remote(), indent=2))


@app.local_entrypoint()
def quant_cli():
    announce("quantize", bound(QUANT_GPU, QUANT_CPU, QUANT_MIB, QUANT_TIMEOUT))
    print(json.dumps(quantize.remote(), indent=2))


@app.local_entrypoint()
def fit_cli():
    announce("fit", bound(FIT_GPU, FIT_CPU, FIT_MIB, FIT_TIMEOUT))
    print(json.dumps(fit.remote(), indent=2))
