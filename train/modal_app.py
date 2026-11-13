"""Train a 4-bit QLoRA adapter for Qwen3.8-27B on the shopagent SFT records.

Input: `data/sft/sft.jsonl` from `python -m shopagent.sft` — each row carries its
rendered chat text, its `input_ids`, and `labels` that keep only the assistant
tokens and mask everything else with -100. The file's sha256 is pinned here; a
byte-level change stops the run before a GPU is used.

This module trains, and nothing else — plus one load-and-shape check that proves
the adapter it wrote can be loaded and driven. It does not build the corpus, does
not score the benchmark, and does not touch `shopagent/`. The gate, the frozen
benchmark and any quality claim are other stages.

What it does:

* loads `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at a pinned revision, 4-bit, as a
  language model: vision layers are frozen and the first forward carries text
  only;
* attaches LoRA to the attention and MLP projections with Unsloth gradient
  checkpointing, and refuses to train if either class of projection is missing
  from the adapter, or if any vision parameter became trainable;
* runs its own loop over pre-tokenised records, because the supervision mask
  lives in `labels` and no trainer needs to re-render the chat;
* adds the assistant `<|im_end|>` stop token to every supervised span in memory.
  The producer masks it, so EOS would never be in the loss; the fix belongs
  here and the file on disk is never rewritten;
* caches weights and tokenizer on a Modal Volume, writes the adapter and
  `run.json` there, and logs the loss every 10 optimizer steps.

Thinking is off by construction: nothing is re-rendered here, so the tokens are
exactly the ones `shopagent.sft` rendered with `enable_thinking=False`, and the
run refuses a row whose think block is not empty.

Cost is bounded twice — a hard optimizer-step cap and a function timeout. The
local entrypoint prints the worst case before anything is allocated.

Smoke: `python -m modal run train/modal_app.py --smoke`

Full run:  `python -m modal run train/modal_app.py`

Check:   `python -m modal run train/modal_app.py --check` — loads the adapter of
`CHECK_RUN_ID` in a fresh container and prints two generations. It is a
load-and-shape check, not a quality claim: both cases are training-split and
seen, so agreement proves the weights and the prompt format load, nothing more.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_REVISION = "8aa5f05d26b7205477066e1449e0af13f762a299"
UNSLOTH_VERSION = "2026.11.4"
# Every current unsloth release caps transformers at <=5.17.0 and torch<2.15.0,
# so 5.19.0 cannot resolve. 5.17.0 does carry the qwen3_5 architecture. The
# dataset manifest recorded transformers 5.19.0 at render time, so the tokenizer
# bytes, not the library version, are what this run holds fixed.
TRANSFORMERS_VERSION = "5.17.0"
TOKENIZER_JSON_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"
TORCH_VERSION = "2.14.1"

DATASET_LOCAL = REPO_ROOT / "data" / "sft" / "sft.jsonl"
DATASET_REMOTE = "/root/sft.jsonl"
DATASET_SHA256 = "5cab4f46a0ade4f8db2535773c8bff015c1ef99885d02c65bbaef0d9134850fb"

VOLUME_NAME = "eval-gated-lora-agent"
VOLUME_MOUNT = "/vol"
RUNS_DIR = f"{VOLUME_MOUNT}/runs"
HF_HOME = f"{VOLUME_MOUNT}/hf"

# Shape of the run.
MAX_SEQ_LEN = 2560  # longest record is 2540 tokens; nothing is truncated
BATCH_SIZE = 1
GRAD_ACCUM = 4  # 160 train rows / 1 / 4 = 40 optimizer steps per epoch, 120 over 3 epochs
EPOCHS = 3
LOG_EVERY = 10  # loss log stride in optimizer steps; the first and last five logs are the windows
LEARNING_RATE = 1e-4
WARMUP_STEPS = 10
MAX_GRAD_NORM = 1.0
SEED = 20261113

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.0

# Hard limits. Neither is a target.
STEP_CAP = 200
SMOKE_STEPS = 20
TIMEOUT_SECONDS = 4800  # 80 minutes
GPU = "A100-40GB"
CPU_CORES = 4.0
MEMORY_MIB = 32 * 1024

# Load-and-shape check: one trained run, two frozen cases, no training.
CHECK_RUN_ID = "full-20261113T180308Z"
CHECK_ADAPTER_REMOTE = f"{RUNS_DIR}/{CHECK_RUN_ID}/adapter"
CHECK_CASE_COMPOSED = "N-105"  # refund ORD-1059 for 2100 cents; train split; no holdout row names it
CHECK_CASE_FREE_TEXT = "N-31"  # "What is the status of ORD-1013?"; train split; no holdout row names it
CHECK_MAX_NEW_TOKENS = 256
CHECK_TIMEOUT_SECONDS = 1200
CHECK_TRAINABLE_PARAMS = 116_727_808  # the count run.json records for the trained adapter
CHECK_BANNER = (
    "load-and-shape check, not a quality claim: every case here is training-split and was seen, "
    "so agreement proves the weights and the prompt format load, nothing more"
)

# Modal list prices, https://modal.com/pricing (read 2026-11-13).
GPU_USD_PER_SECOND = 0.000583  # A100 40 GB
GPU_USD_PER_HOUR = 2.10
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222

ESTIMATE_BASIS = (
    "estimated from wall time at list rates: GPU, CPU and memory only; "
    "excludes adapter save and Volume commit, so the metered bill is higher"
)

IGNORE_INDEX = -100
THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ASSISTANT_TURN = "<|im_start|>assistant"
IM_END = "<|im_end|>"

RECORD_KEYS = ("id", "input_ids", "labels", "split", "text")
ATTENTION_TARGET = re.compile(r"(q_proj|k_proj|v_proj|o_proj|in_proj|out_proj|qkv|attn)")
MLP_TARGET = re.compile(r"(gate_proj|up_proj|down_proj)")
VISION_NAME = re.compile(r"(visual|vision|multi_modal_projector)")

EOS_POLICY = (
    "Every supervised span has its assistant <|im_end|> stop token appended in memory. "
    "The producer masks that token, so EOS would never be in the loss; the file on disk is not rewritten."
)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        f"torch=={TORCH_VERSION}",
        "torchvision",
        f"transformers=={TRANSFORMERS_VERSION}",
        f"unsloth=={UNSLOTH_VERSION}",
        "datasets==4.8.5",
        "accelerate==1.15.0",
        "bitsandbytes==0.50.2",
    )
    .env(
        {
            "HF_HOME": HF_HOME,
            "HF_HUB_CACHE": f"{HF_HOME}/hub",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "PYTHONUNBUFFERED": "1",
        }
    )
    .add_local_file(str(DATASET_LOCAL), DATASET_REMOTE)
    # The check runs the project's own composed-plan validator, not a copy of it.
    .add_local_python_source("shopagent")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
app = modal.App("eval-gated-lora-agent-train", image=image)


class RunError(RuntimeError):
    """A check that must stop the run. Always names what failed."""


def planned_steps(rows: int) -> int:
    per_epoch = math.ceil(rows / GRAD_ACCUM) if rows else 0
    return per_epoch * EPOCHS


def step_limit_for(smoke: bool, train_rows: int) -> int:
    if smoke:
        return SMOKE_STEPS
    return min(STEP_CAP, planned_steps(train_rows))


def worst_case_usd(seconds: float | None = None) -> dict:
    """Estimated cost of one function: GPU, CPU and memory over timeout or elapsed time.

    An estimate from list rates, never a bill. It excludes adapter save and
    Volume commit, so the metered figure is higher; read that from the Modal
    dashboard.
    """
    seconds = TIMEOUT_SECONDS if seconds is None else seconds
    gpu = seconds * GPU_USD_PER_SECOND
    cpu = seconds * CPU_CORES * CPU_USD_PER_CORE_SECOND
    memory = seconds * (MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND
    return {
        "kind": "estimated",
        "basis": ESTIMATE_BASIS,
        "seconds": seconds,
        "gpu_usd": round(gpu, 4),
        "cpu_usd": round(cpu, 4),
        "memory_usd": round(memory, 4),
        "total_usd": round(gpu + cpu + memory, 4),
    }


def sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tokenizer_json_sha256() -> str | None:
    """Digest of the pinned `tokenizer.json` in the cache the model reads."""
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    return sha256_file(path) if path.is_file() else None


def load_records(path: str, expected_sha: str) -> tuple[list[dict], dict]:
    """Read the SFT rows, refusing any byte-level change to the pinned digest."""
    payload = Path(path).read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    if digest != expected_sha:
        raise RunError(f"dataset_digest: {digest} does not match the pinned {expected_sha}")
    rows = []
    for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise RunError(f"dataset_unreadable: line {lineno}: {exc}") from exc
    if not rows:
        raise RunError("dataset_empty")
    return rows, {"sha256": digest, "rows": len(rows), "bytes": len(payload)}


def validate_records(rows: list[dict]) -> dict:
    """Refuse a record the loop cannot train, and prove thinking is off."""
    for row in rows:
        missing = [key for key in RECORD_KEYS if key not in row]
        if missing:
            raise RunError(f"record_shape: {row.get('id')} is missing {missing}")
        ids, labels = row["input_ids"], row["labels"]
        if len(ids) != len(labels):
            raise RunError(f"record_shape: {row['id']} has {len(ids)} ids and {len(labels)} labels")
        if not ids:
            raise RunError(f"record_empty: {row['id']}")
        if len(ids) > MAX_SEQ_LEN:
            raise RunError(
                f"record_too_long: {row['id']} is {len(ids)} tokens, above max_seq_length {MAX_SEQ_LEN}; "
                "truncating would cut the reply off"
            )
        if not any(label != IGNORE_INDEX for label in labels):
            raise RunError(f"record_unmasked: {row['id']} supervises no token")
        if row["split"] not in ("train", "val"):
            raise RunError(f"record_split: {row['id']} has split {row['split']!r}")

    for row in rows:
        text = row["text"]
        for match in re.finditer(re.escape(THINK_OPEN), text):
            close = text.find(THINK_CLOSE, match.end())
            if close < 0:
                raise RunError(f"thinking_unclosed: {row['id']}")
            if text[match.end() : close].strip():
                raise RunError(
                    f"thinking_not_off: {row['id']} carries reasoning text in its think block; "
                    "this run is language-only and expected enable_thinking=False"
                )

    lengths = [len(row["input_ids"]) for row in rows]
    return {
        "rows": len(rows),
        "train_rows": sum(1 for row in rows if row["split"] == "train"),
        "val_rows": sum(1 for row in rows if row["split"] == "val"),
        "tokens_total": sum(lengths),
        "max_tokens": max(lengths),
        "supervised_source": sum(sum(1 for label in row["labels"] if label != IGNORE_INDEX) for row in rows),
        "supervised_source_train": sum(
            sum(1 for label in row["labels"] if label != IGNORE_INDEX) for row in rows if row["split"] == "train"
        ),
    }


def extend_eos(rows: list[dict], im_end_id: int) -> dict:
    """Supervise the assistant stop token on every span, in memory only.

    The producer masks `<|im_end|>`, so without this the adapter never learns to
    stop. Each kept run must be followed by the stop token; anything else is a
    refusal, not a guess.
    """
    runs = 0
    added = 0
    for row in rows:
        ids, labels = row["input_ids"], list(row["labels"])
        turns = row["text"].count(ASSISTANT_TURN)
        row_runs = 0
        index = 0
        while index < len(labels):
            if labels[index] == IGNORE_INDEX:
                index += 1
                continue
            end = index
            while end < len(labels) and labels[end] != IGNORE_INDEX:
                end += 1
            row_runs += 1
            stop = None
            for position in range(end, min(end + 3, len(ids))):
                if ids[position] == im_end_id:
                    stop = position
                    break
            if stop is None:
                raise RunError(f"eos_missing: {row['id']} has a supervised span with no {IM_END} within 3 tokens")
            for position in range(end, stop + 1):
                labels[position] = ids[position]
                added += 1
            index = stop + 1
        if row_runs != turns:
            raise RunError(
                f"eos_turns: {row['id']} has {row_runs} supervised spans but {turns} assistant turns in its render"
            )
        row["labels"] = labels
        runs += row_runs
    return {"runs": runs, "tokens_added": added}


def adapter_target_classes(trainable) -> dict:
    """Classify the adapted modules from the tensors that carry a LoRA weight.

    Unsloth configures PEFT with a regex, not a list, so the config says nothing
    we can count. The trainable tensors do: `...q_proj.lora_A.default.weight`
    names the module that was actually adapted.
    """
    modules = sorted({name.split(".lora_")[0] for name, _ in trainable if ".lora_" in name})
    if not modules:
        raise RunError("adapter_targets: no LoRA tensor among the trainable parameters")
    leaves = sorted({name.rsplit(".", 1)[-1] for name in modules})
    attention = [leaf for leaf in leaves if ATTENTION_TARGET.fullmatch(leaf)]
    mlp = [leaf for leaf in leaves if MLP_TARGET.fullmatch(leaf)]
    if not attention:
        raise RunError(f"adapter_targets: no attention projection in {leaves}")
    if not mlp:
        raise RunError(f"adapter_targets: no MLP projection in {leaves}")
    return {"modules": len(modules), "attention": attention, "mlp": mlp}


def adapter_checksums(model) -> dict:
    """Per-tensor absolute sum of every trainable weight: a change detector, not a copy."""
    import torch

    return {
        name: float(torch.abs(param.detach().float()).sum().item())
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def package_versions() -> dict:
    from importlib import metadata

    versions = {}
    for name in ("transformers", "unsloth", "unsloth_zoo", "torch", "bitsandbytes", "peft", "datasets", "accelerate"):
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "absent"
    return versions


def loss_schedule(total_steps: int):
    """Warm up for WARMUP_STEPS, then decay linearly to zero over the run."""

    def lr_lambda(completed: int) -> float:
        if completed < WARMUP_STEPS:
            return (completed + 1) / WARMUP_STEPS
        tail = max(1, total_steps - WARMUP_STEPS)
        return max(0.0, (total_steps - completed) / tail)

    return lr_lambda


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def train(smoke: bool = False) -> dict:
    """One training run. `smoke=True` takes 20 optimizer steps and keeps no adapter."""
    import torch
    from unsloth import FastModel

    started = time.time()
    bound = worst_case_usd()
    scope = "smoke" if smoke else "full"

    rows, file_facts = load_records(DATASET_REMOTE, DATASET_SHA256)
    facts = validate_records(rows)
    train_rows = [row for row in rows if row["split"] == "train"]
    limit = step_limit_for(smoke, len(train_rows))
    planned = limit if smoke else planned_steps(len(train_rows))

    print(
        f"data: {file_facts['rows']} records, sha256 {file_facts['sha256'][:16]}, "
        f"{facts['train_rows']} train / {facts['val_rows']} val, {facts['tokens_total']} tokens, "
        f"longest {facts['max_tokens']} (max_seq_length {MAX_SEQ_LEN})",
        flush=True,
    )
    print(f"scope: {scope}; hard limit {limit} optimizer steps; timeout {TIMEOUT_SECONDS}s", flush=True)
    print(
        f"worst case at the timeout: ${bound['total_usd']:.2f} "
        f"(gpu ${bound['gpu_usd']:.3f} + cpu ${bound['cpu_usd']:.3f} + memory ${bound['memory_usd']:.3f})",
        flush=True,
    )

    versions = package_versions()
    if versions["transformers"] != TRANSFORMERS_VERSION:
        raise RunError(f"transformers_version: expected {TRANSFORMERS_VERSION}, found {versions['transformers']}")
    print(f"packages: {json.dumps(versions, sort_keys=True)}", flush=True)

    # FastModel returns a processor for a vision-language checkpoint. The stop
    # token must come from the tokenizer inside it, never from a guess.
    load_started = time.time()
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    missing = [name for name in ("convert_tokens_to_ids", "encode") if not hasattr(tokenizer, name)]
    if missing:
        raise RunError(
            f"tokenizer_missing: FastModel returned {type(processor).__name__}, and its tokenizer "
            f"({type(tokenizer).__name__}) has no {' or '.join(missing)}"
        )
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    if not isinstance(im_end_id, int) or im_end_id == tokenizer.unk_token_id:
        raise RunError(f"eos_id: {IM_END} is not a single known token id ({im_end_id!r})")
    if tokenizer.decode([im_end_id]) != IM_END:
        raise RunError(f"eos_id: id {im_end_id} decodes to {tokenizer.decode([im_end_id])!r}, not {IM_END}")
    tokenizer_sha = tokenizer_json_sha256()
    if tokenizer_sha != TOKENIZER_JSON_SHA256:
        raise RunError(f"tokenizer_bytes: tokenizer.json is {tokenizer_sha}, expected {TOKENIZER_JSON_SHA256}")
    print(f"tokenizer: {MODEL_ID}@{MODEL_REVISION[:8]} tokenizer.json sha256 {tokenizer_sha[:16]}", flush=True)

    eos = extend_eos(rows, im_end_id)
    supervised = facts["supervised_source"] + eos["tokens_added"]
    supervised_train = sum(sum(1 for label in row["labels"] if label != IGNORE_INDEX) for row in train_rows)
    print(
        f"supervision: {facts['supervised_source']} tokens from the stored masks + {eos['tokens_added']} "
        f"{IM_END} tokens across {eos['runs']} assistant spans = {supervised} supervised tokens; "
        f"train split {supervised_train}",
        flush=True,
    )

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,
        finetune_language_layers=True,
        finetune_attention_modules=True,
        finetune_mlp_modules=True,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )

    trainable = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not trainable:
        raise RunError("adapter_missing: no trainable parameter after get_peft_model")
    vision = [name for name, _ in trainable if VISION_NAME.search(name)]
    if vision:
        raise RunError(f"vision_trainable: {len(vision)} vision parameters are trainable, first {vision[0]}")
    targets = adapter_target_classes(trainable)
    lora_b = [name for name, _ in trainable if "lora_B" in name]
    checkpointing = bool(getattr(model, "is_gradient_checkpointing", False))
    if not checkpointing:
        raise RunError("gradient_checkpointing: the model reports it is off")
    trainable_params = sum(param.numel() for _, param in trainable)
    print(
        f"adapter: r={LORA_R} alpha={LORA_ALPHA} dropout={LORA_DROPOUT}, {len(lora_b)} lora_B tensors, "
        f"{trainable_params} trainable parameters; {targets['modules']} adapted modules, "
        f"attention targets {targets['attention']}; mlp targets {targets['mlp']}",
        flush=True,
    )
    print(f"model: loaded in {time.time() - load_started:.1f}s, gradient checkpointing {checkpointing}", flush=True)

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda")
    params = [param for _, param in trainable]
    optimizer = torch.optim.AdamW(params, lr=LEARNING_RATE, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, loss_schedule(planned))
    before = adapter_checksums(model)
    model.train()

    rng = random.Random(SEED)
    order = list(range(len(train_rows)))
    train_started = time.time()
    updates = 0
    micro_losses: list[float] = []
    logged: list[tuple[int, float]] = []
    grad_norms: list[float] = []

    for epoch in range(EPOCHS):
        rng.shuffle(order)
        print(f"epoch {epoch + 1}/{EPOCHS}: {len(order)} records, {len(order) // GRAD_ACCUM} steps", flush=True)
        for start in range(0, len(order), GRAD_ACCUM):
            batch = [train_rows[index] for index in order[start : start + GRAD_ACCUM]]
            for row in batch:
                input_ids = torch.tensor([row["input_ids"]], dtype=torch.long, device=device)
                labels = torch.tensor([row["labels"]], dtype=torch.long, device=device)
                output = model(input_ids=input_ids, labels=labels, use_cache=False)
                loss = output.loss
                if loss is None or not math.isfinite(loss.item()):
                    raise RunError(f"loss_not_finite: step {updates + 1} on {row['id']} gave {loss}")
                (loss / len(batch)).backward()
                micro_losses.append(loss.item())
                del output, loss
            grad_norm = torch.nn.utils.clip_grad_norm_(params, MAX_GRAD_NORM)
            if not math.isfinite(float(grad_norm)):
                raise RunError(f"grad_not_finite: step {updates + 1} gave grad norm {grad_norm}")
            grad_norms.append(float(grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates == 1 or updates % LOG_EVERY == 0 or updates == limit:
                window = micro_losses[-5 * GRAD_ACCUM :]
                mean = sum(window) / len(window)
                logged.append((updates, mean))
                elapsed = time.time() - train_started
                print(
                    f"step {updates}/{limit} loss {mean:.4f} lr {scheduler.get_last_lr()[0]:.2e} "
                    f"grad_norm {float(grad_norm):.3f} elapsed {elapsed:.0f}s "
                    f"{updates / elapsed if elapsed else 0:.2f} steps/s",
                    flush=True,
                )
            if updates >= limit:
                break
        if updates >= limit:
            break

    optimizer_seconds = time.time() - train_started
    wall_seconds = time.time() - started

    if updates != limit:
        raise RunError(f"steps: completed {updates} optimizer updates, expected {limit}")
    if len(logged) < 2:
        raise RunError(f"log_short: only {len(logged)} loss logs, cannot show movement")
    movement = max(abs(b - a) for (_, a), (_, b) in zip(logged, logged[1:]))
    if movement <= 1e-4:
        raise RunError(f"loss_flat: largest change between logged losses is {movement:.2e}")
    after = adapter_checksums(model)
    changed = [name for name, value in before.items() if abs(after[name] - value) > 0.0]
    b_changed = [name for name in changed if "lora_B" in name]
    if not b_changed:
        raise RunError(f"adapter_unchanged: {len(changed)} trainable tensors changed but no lora_B did")

    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    first_mean = sum(value for _, value in logged[:5]) / min(5, len(logged))
    last_mean = sum(value for _, value in logged[-5:]) / min(5, len(logged))
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{scope}-{stamp}"
    run_dir = Path(RUNS_DIR) / run_id

    result = {
        "run_id": run_id,
        "scope": scope,
        "status": "ok",
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "load_in_4bit": True,
            "vision_layers": "frozen",
            "tokenizer_json_sha256": tokenizer_sha,
        },
        "dataset": {
            **file_facts,
            **facts,
            "eos_supervision": {**eos, "policy": EOS_POLICY},
            "supervised_total": supervised,
            "supervised_train": supervised_train,
        },
        "config": {
            "max_seq_length": MAX_SEQ_LEN,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation_steps": GRAD_ACCUM,
            "epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "lr_schedule": f"linear warmup {WARMUP_STEPS} steps then linear decay",
            "optimizer": "AdamW fp32, betas (0.9, 0.999), eps 1e-8, weight_decay 0",
            "max_grad_norm": MAX_GRAD_NORM,
            "seed": SEED,
            "lora": {
                "r": LORA_R,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "bias": "none",
                "gradient_checkpointing": "unsloth",
                "target_modules": targets,
                "trainable_parameters": trainable_params,
                "lora_b_tensors": len(lora_b),
            },
            "enable_thinking": False,
            "step_cap": STEP_CAP,
            "smoke_steps": SMOKE_STEPS,
            "timeout_seconds": TIMEOUT_SECONDS,
        },
        "training": {
            "optimizer_steps": updates,
            "micro_batches": updates * GRAD_ACCUM,
            "planned_steps": planned,
            "loss_first_log": logged[0][1],
            "loss_last_log": logged[-1][1],
            "loss_first5_mean": round(first_mean, 4),
            "loss_last5_mean": round(last_mean, 4),
            "loss_largest_step_change": round(movement, 6),
            "loss_curve": [{"step": step, "loss": round(value, 4)} for step, value in logged],
            "grad_norm_max": round(max(grad_norms), 4),
            "grad_norm_mean": round(sum(grad_norms) / len(grad_norms), 4),
            "adapter_tensors_changed": len(changed),
        },
        "time": {
            "wall_seconds": round(wall_seconds, 1),
            "optimizer_seconds": round(optimizer_seconds, 1),
            "seconds_per_step": round(optimizer_seconds / updates, 2),
        },
        "hardware": {
            "gpu_requested": GPU,
            "gpu_name": torch.cuda.get_device_name(0),
            "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
            "peak_memory_gb": round(peak_gb, 2),
            "cpu_cores": CPU_CORES,
            "memory_mib": MEMORY_MIB,
        },
        "cost": {
            "kind": "estimated",
            "basis": ESTIMATE_BASIS,
            "gpu_usd_per_hour": GPU_USD_PER_HOUR,
            "gpu_usd": round(wall_seconds * GPU_USD_PER_SECOND, 4),
            "cpu_usd": round(wall_seconds * CPU_CORES * CPU_USD_PER_CORE_SECOND, 4),
            "memory_usd": round(wall_seconds * (MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND, 4),
            "total_usd": round(
                wall_seconds
                * (GPU_USD_PER_SECOND + CPU_CORES * CPU_USD_PER_CORE_SECOND + (MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND),
                4,
            ),
            "worst_case_usd": bound["total_usd"],
        },
        "packages": versions,
        "notes": {
            "loss_rule": "labels are the stored mask plus the assistant stop token; the model shifts them, -100 is ignored",
            "scope_note": (
                "smoke: 20 steps prove the stack builds and steps; adapter save and reload are not covered"
                if smoke
                else "full: 3 epochs over the training split; the adapter is saved, but load and eval gate are later slices"
            ),
            "log_stride": f"loss logged at step 1, then every {LOG_EVERY} optimizer steps, plus the last step",
            "transformers_delta": (
                "dataset manifest rendered under transformers 5.19.0; unsloth caps it at 5.17.0, "
                "so the tokenizer is fixed by its tokenizer.json sha256 instead"
            ),
        },
    }

    if not smoke:
        adapter_dir = run_dir / "adapter"
        adapter_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        result["adapter"] = {"dir": str(adapter_dir), "saved": True}
        print(f"adapter: saved to {adapter_dir}", flush=True)
    else:
        result["adapter"] = {"dir": None, "saved": False, "why": "smoke discards the adapter"}

    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "run.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    volume.commit()
    print(f"run.json: {run_dir / 'run.json'}", flush=True)
    print(
        f"done: {updates} steps, {wall_seconds:.0f}s wall, {optimizer_seconds / updates:.2f}s/step, "
        f"peak {peak_gb:.1f} GB, estimated cost ${result['cost']['total_usd']:.2f} "
        f"(compute only; excludes save and Volume commit)",
        flush=True,
    )
    return result


def check_prompt(text: str, case_id: str, tokenizer, input_ids: list[int]) -> str:
    """The exact prompt training saw before the case's last assistant content.

    The SFT record keeps the rendered chat, so this is a byte slice of it, not a
    re-render that a template change could move. The slice must tokenize to the
    record's own id prefix, or it is not the training prompt.
    """
    marker = ASSISTANT_TURN + "\n"
    start = text.rindex(marker)
    empty_think = THINK_CLOSE + "\n\n"
    stop = text.find(empty_think, start)
    if stop < 0:
        raise RunError(f"check_prompt: {case_id} has no empty {THINK_OPEN} block after its assistant marker")
    prompt = text[: stop + len(empty_think)]
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids != input_ids[: len(ids)]:
        raise RunError(f"check_prompt: {case_id} prompt does not tokenize to its training id prefix")
    return prompt


def _adapter_key(name: str) -> str:
    """PEFT's loaded name and the name it saved, without a key map."""
    while name.startswith("base_model."):
        name = name[len("base_model.") :]
    return name.replace(".default.", ".")


def adapter_proof(model, adapter_dir: str) -> dict:
    """Three guards that the adapter is loaded and is the run's own weights."""
    import torch
    from safetensors.torch import load_file

    names = sorted(model.peft_config)
    active = [name for name in (getattr(model, "active_adapters", None) or [])]
    if not names or not active:
        raise RunError(f"adapter_inactive: peft_config names {names}, active_adapters {active}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable != CHECK_TRAINABLE_PARAMS:
        raise RunError(
            f"adapter_params: {trainable} trainable parameters, expected the trained {CHECK_TRAINABLE_PARAMS}"
        )
    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    by_key = {_adapter_key(key): key for key in saved}
    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        key = _adapter_key(name)
        if key not in by_key:
            raise RunError(f"adapter_file: no saved tensor matches {name} (looked for {key})")
        loaded = param.detach().to("cpu", torch.float32)
        stored = saved[by_key[key]].detach().to("cpu", torch.float32)
        if not torch.equal(loaded, stored):
            raise RunError(f"adapter_weights: {name} is not bit-identical to {by_key[key]} in adapter_model.safetensors")
        return {
            "adapter_names": names,
            "active_adapters": active,
            "trainable_parameters": trainable,
            "tensor": name,
            "file_key": by_key[key],
            "dtype": str(param.dtype),
            "shape": list(param.shape),
            "torch_equal": True,
        }
    raise RunError("adapter_file: the loaded model has no lora_B parameter")


def _strip_stop_token(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(stop):
            return text[: -len(stop)].rstrip()
    return text


def parse_plan(text: str) -> tuple[dict | None, str]:
    """The output must be one JSON object at the root, and nothing else.

    One object, or one fence whose whole content is that object. A list of tool
    calls is a valid free-text plan and an invalid composed plan, so a dict inside
    a list is refused rather than extracted. Prose outside a fence is refused, and
    the fence must wrap the entire output. Keys are judged by the project's own
    validator, so a prose key is refused here too.
    """
    from shopagent.compose import ComposeError, _check_plan

    cleaned = _strip_stop_token(text.strip())
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S)
    if fenced:
        cleaned = _strip_stop_token(fenced.group(1).strip())
    elif "```" in cleaned:
        return None, "text outside the fence: a fence must wrap the whole output"
    if not cleaned:
        return None, "empty output"
    try:
        payload, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"not one JSON value at the root: {exc}"
    if not isinstance(payload, dict):
        return None, f"plan is {type(payload).__name__}, not an object at the root"
    trailing = cleaned[end:].strip()
    if trailing:
        return None, f"trailing text after the plan: {trailing[:60]!r}"
    try:
        _check_plan(payload)
    except ComposeError as exc:
        return None, f"composed-plan validator refused: {exc}"
    return payload, "ok"


def generate(model, tokenizer, prompt: str, eos_id: int) -> str:
    """Greedy decode from the training prompt; returns the new text only."""
    import torch

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=CHECK_MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    return tokenizer.decode(output[0, inputs["input_ids"].shape[1] :], skip_special_tokens=False)


@app.function(
    gpu=GPU,
    timeout=CHECK_TIMEOUT_SECONDS,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def check_adapter(cases: dict) -> dict:
    """Load the trained adapter in a fresh container and drive it on two cases."""
    import torch
    from peft import PeftModel
    from unsloth import FastModel

    started = time.time()
    records, file_facts = load_records(DATASET_REMOTE, DATASET_SHA256)
    by_case = {row["case"]: row for row in records if row["case"] in cases}
    missing = sorted(set(cases) - set(by_case))
    if missing:
        raise RunError(f"check_cases: {missing} are not in the SFT dataset")
    print(f"data: {file_facts['rows']} records, sha256 {file_facts['sha256'][:16]}", flush=True)

    load_started = time.time()
    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    model = PeftModel.from_pretrained(model, CHECK_ADAPTER_REMOTE, is_trainable=True)
    model.eval()
    model.config.use_cache = True
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    print(f"model: base and adapter loaded in {time.time() - load_started:.1f}s", flush=True)

    proof = adapter_proof(model, CHECK_ADAPTER_REMOTE)
    print(
        f"adapter: {CHECK_ADAPTER_REMOTE}; peft names {proof['adapter_names']}, "
        f"active {proof['active_adapters']}",
        flush=True,
    )
    print(
        f"adapter: {proof['trainable_parameters']} trainable parameters "
        f"(trained count {CHECK_TRAINABLE_PARAMS})",
        flush=True,
    )
    print(
        f"adapter: {proof['tensor']} ({proof['dtype']} {proof['shape']}) is torch.equal to "
        f"{proof['file_key']} in adapter_model.safetensors",
        flush=True,
    )

    results = {}
    for case_id, meta in cases.items():
        prompt = check_prompt(by_case[case_id]["text"], case_id, tokenizer, by_case[case_id]["input_ids"])
        print(f"\n==== {case_id} {meta['kind']} prompt, {len(prompt)} chars ====", flush=True)
        print(prompt, flush=True)
        print(f"==== end {case_id} prompt ====", flush=True)
        raw = generate(model, tokenizer, prompt, im_end_id)
        print(f"==== {case_id} raw output ====", flush=True)
        print(raw, flush=True)
        print(f"==== end {case_id} output ====", flush=True)
        entry = {"kind": meta["kind"], "prompt_chars": len(prompt), "output": raw}
        if meta["kind"] == "composed":
            plan, verdict = parse_plan(raw)
            expected = meta["expected_plan"]
            print(f"parse: {verdict}", flush=True)
            if plan is None:
                entry.update({"plan": None, "parse": verdict, "expected": expected, "matches": False})
            else:
                fields = {key: plan.get(key) == expected.get(key) for key in sorted(set(plan) | set(expected))}
                print(f"plan: {json.dumps(plan, sort_keys=True)}", flush=True)
                print(f"required_calls: {json.dumps(expected, sort_keys=True)}", flush=True)
                print(f"matches required_calls: {plan == expected} {fields}", flush=True)
                entry.update(
                    {"plan": plan, "parse": verdict, "expected": expected, "matches": plan == expected, "fields": fields}
                )
        else:
            print(f"{case_id}: free text, printed and not scored", flush=True)
        results[case_id] = entry

    elapsed = time.time() - started
    cost = worst_case_usd(elapsed)
    hardware = {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
    }
    print(
        f"\ngpu: {hardware['gpu_name']} {hardware['gpu_total_gb']} GB; wall {elapsed:.0f}s; "
        f"estimated cost ${cost['total_usd']:.2f} "
        f"(gpu ${cost['gpu_usd']:.3f} + cpu ${cost['cpu_usd']:.3f} + memory ${cost['memory_usd']:.3f}; "
        f"compute only, excludes Volume commit)",
        flush=True,
    )
    print(f"check: {CHECK_BANNER}", flush=True)
    return {
        "run_id": CHECK_RUN_ID,
        "adapter": proof,
        "cases": results,
        "hardware": hardware,
        "time": {"wall_seconds": round(elapsed, 1)},
        "cost": cost,
        "banner": CHECK_BANNER,
    }


def check_payload() -> dict:
    """The two frozen cases, read here so the image needs no copy of `bench/`."""
    from shopagent.bench import load_all_cases
    from shopagent.cli import plan_from_case

    wanted = {CHECK_CASE_COMPOSED: "composed", CHECK_CASE_FREE_TEXT: "free_text"}
    payload = {}
    for case in load_all_cases():
        kind = wanted.get(case["id"])
        if kind is None:
            continue
        entry = {"kind": kind, "ticket": case["ticket"], "required_calls": case["required_calls"]}
        if kind == "composed":
            entry["expected_plan"] = plan_from_case(case)
        payload[case["id"]] = entry
    missing = sorted(set(wanted) - set(payload))
    if missing:
        raise RunError(f"check_cases: {missing} are not in bench/cases")
    return payload


@app.local_entrypoint()
def main(smoke: bool = False, check: bool = False):
    """Print the cost bound here, then hand the run to a GPU."""
    if check:
        payload = check_payload()
        bound = worst_case_usd(CHECK_TIMEOUT_SECONDS)
        print(
            f"check {CHECK_RUN_ID}: adapter {CHECK_ADAPTER_REMOTE}, gpu {GPU}, "
            f"timeout {CHECK_TIMEOUT_SECONDS}s; worst case ${bound['total_usd']:.2f}"
        )
        print(f"cases: {json.dumps(payload, sort_keys=True)}")
        print(json.dumps(check_adapter.remote(payload), indent=2, sort_keys=True))
        return
    bound = worst_case_usd()
    lines = [line for line in DATASET_LOCAL.read_text(encoding="utf-8").splitlines() if line.strip()]
    train_rows = sum(1 for line in lines if json.loads(line)["split"] == "train")
    steps = SMOKE_STEPS if smoke else min(STEP_CAP, planned_steps(train_rows))
    print(f"scope {'smoke' if smoke else 'full'}: {steps} optimizer steps, cap {STEP_CAP}, timeout {TIMEOUT_SECONDS}s")
    print(
        f"worst case at the timeout: ${bound['total_usd']:.2f} "
        f"(gpu ${bound['gpu_usd']:.3f} + cpu ${bound['cpu_usd']:.3f} + memory ${bound['memory_usd']:.3f})"
    )
    result = train.remote(smoke=smoke)
    print(json.dumps(result, indent=2, sort_keys=True))
