"""Bench a student model on the frozen benchmark: one attempt per case, Phase A only.

Input: the frozen case files under `bench/cases`, verified by `verify-frozen`
before anything leaves this machine. Output: one row per case in
`/vol/bench/<NAME>/rows.jsonl` and a manifest beside it, copied to
`out/<NAME>.jsonl` and `out/<NAME>.manifest.json` by the local entrypoint.

What this measures, and what it does not:

* One attempt per case, never a retry. Temperature is zero, so a second sample
  of the same prompt is the same sample; a retry would spend GPU and read as a
  recovery. A draft the gate refuses is recorded refused.
* `run_gated` is not used, so nothing is retried and `human_queue` is never
  exercised. This is a gate measurement, not a full ticket lifecycle.
* Phase A only. Phase B is not called, no judge is injected, no network is
  reached and no key exists in this image.
* The user turn for every case is the frozen `case["ticket"]`. Training used
  the teacher's paraphrase of that ticket as its user turn, so this is a
  distribution difference and the manifest says so.

Prompt rendering is the training renderer, proved per case: the generation
prompt is compared byte for byte against the slice of the full chat render that
`shopagent.sft` supervised. A template or flag drift fails the case, not the
score.

The local side owns the frozen evidence; the GPU side needs no bench file, only
the case dicts it is handed. That keeps the remote container free of a second
copy of the benchmark. The bench window is the training window (`2816`), and an
adapter run must name a `run.json` whose dataset sha256 and `max_seq_length`
match the SFT file and the window here; a mismatch stops before generation.

Smoke: `python -m modal run train/bench_app.py --student base --limit 8 --out bench-smoke-base`

Full:  `python -m modal run train/bench_app.py --student base --out bench-base`

Adapter: `python -m modal run train/bench_app.py --student adapter --run-id full-20261113T205627Z --out bench-adapter`
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ID = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
MODEL_REVISION = "8aa5f05d26b7205477066e1449e0af13f762a299"
UNSLOTH_VERSION = "2026.11.4"
TRANSFORMERS_VERSION = "5.17.0"  # the pin the adapter was trained under
TORCH_VERSION = "2.14.1"
TOKENIZER_JSON_SHA256 = "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523"

VOLUME_NAME = "eval-gated-lora-agent"
VOLUME_MOUNT = "/vol"
RUNS_DIR = f"{VOLUME_MOUNT}/runs"
BENCH_DIR = f"{VOLUME_MOUNT}/bench"
HF_HOME = f"{VOLUME_MOUNT}/hf"

ADAPTER_RUN_ID = "full-20261113T205627Z"
STUDENTS = ("base", "adapter")

# The window the student was trained at, not a bench preference. The longest
# training record is 2757 tokens; a bench conversation that runs past this is a
# refusal or a shortened reply, never a silent cut of the input.
BENCH_MAX_SEQ_LEN = 2816
MAX_NEW_TOKENS_PLAN = 256
MAX_NEW_TOKENS_REPLY = 320
# Below this the window has no room for an answer, so the case cannot be run.
MIN_NEW_TOKENS = 16

GPU = "A100-40GB"
CPU_CORES = 4.0
MEMORY_MIB = 32 * 1024
TIMEOUT_SECONDS = 7200  # 284 cases; the bound is printed before anything is allocated
FLUSH_EVERY = 25  # rows committed to the Volume this often, so a crash keeps its work

# Modal list prices, https://modal.com/pricing (same read as train/modal_app.py).
GPU_USD_PER_SECOND = 0.000583  # A100 40 GB
CPU_USD_PER_CORE_SECOND = 0.0000131
MEMORY_USD_PER_GIB_SECOND = 0.00000222
ESTIMATE_BASIS = (
    "estimated from wall time at list rates: GPU, CPU and memory only; "
    "excludes Volume commit, so the metered bill is higher"
)

ENABLE_THINKING = False
ASSISTANT_TURN = "<|im_start|>assistant"
IM_END = "<|im_end|>"
THINK_CLOSE = "</think>"
STOP_TOKENS = ("<|im_end|>", "<|endoftext|>")
PROBE = '{"probe": true}'

# The user turn is the frozen ticket, which is not what training read. Every
# report of these numbers carries this line.
TICKET_NOTE = (
    "the frozen case ticket is the user turn; training used the teacher's paraphrase of "
    "that ticket, so the student sees wording it was not trained on"
)
LIMITS = (
    "one attempt per case; temperature 0 makes a retry identical, so no retry is taken",
    "run_gated is not used: nothing is retried and human_queue is never exercised",
    "Phase A only: no Phase B, no judge, no network, no key",
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
    .add_local_python_source("shopagent")
)

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True, version=2)
app = modal.App("eval-gated-lora-agent-bench", image=image)


class FatalError(RuntimeError):
    """A fault that would make every later case meaningless: stop the run."""


def worst_case_usd(seconds: float | None = None) -> dict:
    seconds = TIMEOUT_SECONDS if seconds is None else seconds
    gpu = seconds * GPU_USD_PER_SECOND
    cpu = seconds * CPU_CORES * CPU_USD_PER_CORE_SECOND
    memory = seconds * (MEMORY_MIB / 1024) * MEMORY_USD_PER_GIB_SECOND
    return {
        "kind": "estimated",
        "basis": ESTIMATE_BASIS,
        "seconds": round(seconds, 1),
        "gpu_usd": round(gpu, 4),
        "cpu_usd": round(cpu, 4),
        "memory_usd": round(memory, 4),
        "total_usd": round(gpu + cpu + memory, 4),
    }


def cost_for(elapsed: float) -> dict:
    cost = worst_case_usd(elapsed)
    cost["gpu_usd_per_hour"] = 2.10
    return cost


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# -- case selection -------------------------------------------------------


def case_pool() -> dict[str, list[dict]]:
    """Every frozen case file, in the project's own file order, with its cases.

    `shopagent.bench.ALL_CASE_FILES` is the list `freeze.py` hashes, so the bench
    runs exactly the bytes `verify-frozen` approved.
    """
    from shopagent.bench import ALL_CASE_FILES, CASES_DIR

    pool: dict[str, list[dict]] = {}
    for name in ALL_CASE_FILES:
        path = CASES_DIR / name
        if not path.exists():
            continue
        cases: list[dict] = []
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                cases.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise FatalError(f"{name}:{lineno}: invalid JSON: {exc}") from exc
        if cases:
            pool[name] = cases
    return pool


def is_composed(case: dict) -> bool:
    """The path rule `shopagent.corpus.generate` applies, called, not copied."""
    from shopagent.corpus import composed_unsupported

    return bool(case.get("composed")) and composed_unsupported(case) is None


def _even_indices(n: int, k: int) -> list[int]:
    """`k` indices spread evenly over `n`, first and last included."""
    if k <= 0:
        return []
    if k >= n:
        return list(range(n))
    if k == 1:
        return [0]
    return [round(i * (n - 1) / (k - 1)) for i in range(k)]


def per_file_quota(sizes: dict[str, int], limit: int) -> dict[str, int]:
    """How many cases each file gives.

    One per file before any file gets a second, then the remainder by size
    (largest remainder, file order breaking ties). A file never gives more cases
    than it holds.
    """
    names = [name for name in sizes if sizes[name] > 0]
    if not names:
        return {}
    if limit <= 0 or limit >= sum(sizes.values()):
        return dict(sizes)
    if limit <= len(names):
        # Fewer picks than files: one case from evenly spaced files.
        step = len(names) / limit
        chosen = {names[int(index * step)] for index in range(limit)}
        return {name: (1 if name in chosen else 0) for name in names}

    remaining = limit - len(names)
    total = sum(sizes[name] for name in names)
    exact = {name: remaining * sizes[name] / total for name in names}
    extra = {name: int(exact[name]) for name in names}
    left = remaining - sum(extra.values())
    ranked = sorted(names, key=lambda name: (-(exact[name] - extra[name]), names.index(name)))
    for name in ranked[:left]:
        extra[name] += 1

    counts = {name: min(1 + extra[name], sizes[name]) for name in names}
    spill = limit - sum(counts.values())
    while spill > 0:
        room = [name for name in names if counts[name] < sizes[name]]
        if not room:
            break
        pick = max(room, key=lambda name: (sizes[name] - counts[name], -names.index(name)))
        counts[pick] += 1
        spill -= 1
    return counts


def _ensure_composed(pool: dict[str, list[dict]], picks: list[tuple[str, dict]]) -> list[tuple[str, dict]]:
    """Put one composed case in a partial sample when the pool holds one.

    Nothing moves when the sample already carries a composed case.
    """
    if any(is_composed(case) for _, case in picks):
        return picks
    candidates = [
        (name, index, case)
        for name, cases in pool.items()
        for index, case in enumerate(cases)
        if is_composed(case)
    ]
    if not candidates:
        return picks
    for name, index, case in candidates:
        positions = [position for position, (pick, _) in enumerate(picks) if pick == name]
        if not positions:
            continue
        order = {row["id"]: position for position, row in enumerate(pool[name])}
        nearest = min(positions, key=lambda position: abs(order[picks[position][1]["id"]] - index))
        picks[nearest] = (name, case)
        return picks
    # No pick from a composed-bearing file: exchange the last one.
    picks[-1] = (candidates[0][0], candidates[0][2])
    return picks


def select_cases(pool: dict[str, list[dict]], limit: int) -> list[tuple[str, dict]]:
    """A stride across the concatenated pool, grouped by file.

    `--limit 8` over six files takes roughly one case per file, so normal,
    adversarial and paraphrase all appear, and one composed case rides along
    when the pool holds one.
    """
    sizes = {name: len(cases) for name, cases in pool.items()}
    quota = per_file_quota(sizes, limit)
    picks: list[tuple[str, dict]] = []
    for name, cases in pool.items():
        take = quota.get(name, 0)
        if not take:
            continue
        picks.extend((name, cases[index]) for index in _even_indices(len(cases), take))
    if 0 < limit < sum(sizes.values()):
        picks = _ensure_composed(pool, picks)
        file_order = {name: position for position, name in enumerate(pool)}
        order = {name: {case["id"]: index for index, case in enumerate(cases)} for name, cases in pool.items()}
        picks.sort(key=lambda item: (file_order[item[0]], order[item[0]][item[1]["id"]]))
    return picks


def build_payload(student: str, limit: int, run_id: str, dataset_sha256: str) -> dict:
    """The frozen selection, the digests it was taken from, and the dataset pin."""
    from shopagent.freeze import read

    record = read()
    if record is None:
        raise FatalError("bench/FROZEN.json is missing; run `python -m shopagent.cli freeze`")
    pool = case_pool()
    picks = select_cases(pool, limit)
    if not picks:
        raise FatalError("case selection is empty")
    return {
        "student": student,
        "adapter_run_id": run_id if student == "adapter" else None,
        "dataset": {"sha256": dataset_sha256, "file": "data/sft/sft.jsonl"},
        "cases": [case for _, case in picks],
        "selection": {
            "limit": limit,
            "files": [name for name, _ in picks],
            "per_file": dict(sorted(Counter(name for name, _ in picks).items())),
            "order": [case["id"] for _, case in picks],
            "composed": sum(1 for _, case in picks if is_composed(case)),
            "composed_available": sum(1 for cases in pool.values() for case in cases if is_composed(case)),
            "pool": {name: len(cases) for name, cases in pool.items()},
            "rule": (
                "--limit is a stride across the concatenated pool, grouped by file: one case per "
                "file first, then the remainder by file size, evenly spaced within each file, with "
                "one composed case kept in a partial sample when the pool holds one"
            ),
        },
        "benchmark": {
            "verified_with": "python -m shopagent.cli verify-frozen",
            "verified": True,
            "cases_digest": record["cases_digest"],
            "original_cases_digest": record["original_cases_digest"],
            "count": record["count"],
            "files": record["files"],
        },
    }


# -- model side -----------------------------------------------------------


def tokenizer_json_sha256() -> str | None:
    from huggingface_hub import try_to_load_from_cache

    cached = try_to_load_from_cache(MODEL_ID, "tokenizer.json", revision=MODEL_REVISION)
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def load_model(student: str, adapter_run_id: str | None):
    """The pinned base model, plus the run's adapter for `student == adapter`."""
    from unsloth import FastModel

    model, processor = FastModel.from_pretrained(
        model_name=MODEL_ID,
        revision=MODEL_REVISION,
        max_seq_length=BENCH_MAX_SEQ_LEN,
        load_in_4bit=True,
        full_finetuning=False,
        offload_embedding=True,
    )
    tokenizer = getattr(processor, "tokenizer", processor)
    if not hasattr(tokenizer, "apply_chat_template"):
        raise FatalError(f"tokenizer_missing: {type(tokenizer).__name__} has no chat template")
    proof = None
    if student == "adapter":
        from peft import PeftModel

        if not adapter_run_id:
            raise FatalError("adapter_run_id: --student adapter needs --run-id")
        adapter_dir = f"{RUNS_DIR}/{adapter_run_id}/adapter"
        if not Path(adapter_dir).is_dir():
            raise FatalError(f"adapter_missing: {adapter_dir} is not on the volume")
        # is_trainable=True is the path train/modal_app.py proved: it leaves the
        # LoRA tensors with requires_grad set, which is how the proof below reads
        # them. Inference runs under inference_mode, so nothing accumulates grad.
        model = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
        proof = adapter_proof(model, adapter_dir)
    model.eval()
    model.config.use_cache = True
    im_end_id = tokenizer.convert_tokens_to_ids(IM_END)
    if not isinstance(im_end_id, int) or tokenizer.decode([im_end_id]) != IM_END:
        raise FatalError(f"eos_id: {IM_END} does not resolve to one token id")
    return model, tokenizer, im_end_id, proof


def _adapter_key(name: str) -> str:
    while name.startswith("base_model."):
        name = name[len("base_model.") :]
    return name.replace(".default.", ".")


def adapter_proof(model, adapter_dir: str) -> dict:
    """The adapter is loaded, active, and holds the run's own weights.

    It checks the two things a bench depends on: the adapter is attached and
    driving, and one of its tensors is bit-identical to the file on the volume.
    The trained-parameter count is train/modal_app.py's check, not this one's.
    """
    import torch
    from safetensors.torch import load_file

    names = sorted(model.peft_config)
    active = list(getattr(model, "active_adapters", None) or [])
    if not names or not active:
        raise FatalError(f"adapter_inactive: peft_config names {names}, active_adapters {active}")
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    if trainable <= 0:
        raise FatalError("adapter_inactive: no trainable parameter, so the adapter cannot be driving")
    saved = load_file(str(Path(adapter_dir) / "adapter_model.safetensors"))
    by_key = {_adapter_key(key): key for key in saved}
    for name, param in model.named_parameters():
        if "lora_B" not in name:
            continue
        key = _adapter_key(name)
        if key not in by_key:
            raise FatalError(f"adapter_file: no saved tensor matches {name}")
        loaded = param.detach().to("cpu", torch.float32)
        stored = saved[by_key[key]].detach().to("cpu", torch.float32)
        if not torch.equal(loaded, stored):
            raise FatalError(f"adapter_weights: {name} is not bit-identical to {by_key[key]}")
        return {
            "dir": adapter_dir,
            "adapter_names": names,
            "active_adapters": active,
            "trainable_parameters": trainable,
            "tensor": name,
            "file_key": by_key[key],
            "torch_equal": True,
        }
    raise FatalError("adapter_file: the loaded model has no lora_B parameter")


def training_prompt(tokenizer, messages: list[dict]) -> str:
    """The prompt training rendered before the assistant turn it teaches.

    `messages` ends with the turn before the assistant content being generated.
    The generation prompt is compared against a byte slice of the full chat
    render — the same slice `train/modal_app.py` takes from an SFT record — so a
    template change or a moved flag fails loudly instead of changing the input.
    """
    head = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=ENABLE_THINKING
    )
    full = tokenizer.apply_chat_template(
        messages + [{"role": "assistant", "content": PROBE}],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=ENABLE_THINKING,
    )
    start = full.rindex(ASSISTANT_TURN + "\n")
    stop = full.find(THINK_CLOSE + "\n\n", start)
    if stop < 0:
        raise FatalError("prompt_render: the assistant turn carries no empty think block")
    prefix = full[: stop + len(THINK_CLOSE + "\n\n")]
    if prefix != head:
        raise FatalError(
            f"prompt_render: the generation prompt is {len(head)} chars and the training slice is "
            f"{len(prefix)}; the bench would not be reading what training read"
        )
    return head


def generate(model, tokenizer, prompt: str, eos_id: int, max_new_tokens: int) -> dict:
    """Greedy decode inside the training window; returns the new text and its budget.

    The window is fixed at `BENCH_MAX_SEQ_LEN`. A prompt that already fills it is
    a refusal. A prompt that leaves less than `max_new_tokens` gets what is left,
    and the row records that the reply budget was shortened rather than the
    input: the model's answer is the only thing a window may shorten.
    """
    import torch

    inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt").to("cuda")
    prompt_tokens = int(inputs["input_ids"].shape[1])
    room = BENCH_MAX_SEQ_LEN - prompt_tokens
    if room < MIN_NEW_TOKENS:
        raise FatalError(
            f"prompt_too_long: {prompt_tokens} tokens leaves {room} of {BENCH_MAX_SEQ_LEN}; "
            "truncating the input would change the question"
        )
    allowance = min(max_new_tokens, room)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=allowance,
            do_sample=False,
            eos_token_id=eos_id,
            pad_token_id=eos_id,
        )
    new_ids = output[0, prompt_tokens:]
    return {
        "text": tokenizer.decode(new_ids, skip_special_tokens=False),
        "prompt_tokens": prompt_tokens,
        "new_tokens": int(new_ids.shape[0]),
        "max_new": allowance,
        "reduced_budget": allowance < max_new_tokens,
        "hit_cap": int(new_ids.shape[0]) >= allowance,
    }


def strip_stop(text: str) -> str:
    cleaned = text.strip()
    for stop in STOP_TOKENS:
        if cleaned.endswith(stop):
            return cleaned[: -len(stop)].strip()
    return cleaned


def _unfence(text: str) -> str:
    """Drop one fence that wraps the whole output; leave any other text alone."""
    if not text.startswith("```"):
        return text
    body = text.strip("`")
    if body.startswith("json"):
        body = body[4:]
    return body.strip()


def parse_free_text_plan(raw: str) -> tuple[list[dict] | None, str]:
    """The trained shape: one JSON list of {name, args}, nothing around it."""
    from shopagent.corpus import InvalidPlan, _check_plan_list

    cleaned = _unfence(strip_stop(raw))
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"plan is not JSON: {exc}"
    try:
        return _check_plan_list(payload), "ok"
    except InvalidPlan as exc:
        return None, f"plan is {type(payload).__name__} and unusable: {exc}"


def parse_composed_plan(raw: str) -> tuple[dict | None, str]:
    """The trained shape: one JSON object, no prose sold as a fence, no list."""
    from shopagent.compose import ComposeError, _check_plan

    cleaned = _unfence(strip_stop(raw))
    try:
        payload, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"plan is not one JSON value at the root: {exc}"
    trailing = cleaned[end:].strip()
    if trailing:
        return None, f"trailing text after the plan: {trailing[:60]!r}"
    if not isinstance(payload, dict):
        return None, f"plan is {type(payload).__name__}, not an object at the root"
    try:
        _check_plan(payload)
    except ComposeError as exc:
        return None, f"composed-plan validator refused: {exc}"
    return payload, "ok"


def composed_match(case: dict, plan: dict) -> tuple[bool, str]:
    """The project's own check: action, ids and amount against `required_calls`."""
    from shopagent.corpus import InvalidPlan, PlanMismatch, _check_composed_plan

    try:
        action = _check_composed_plan(case, plan)
    except InvalidPlan as exc:
        return False, f"invalid plan: {exc}"
    except PlanMismatch as exc:
        return False, f"plan mismatch: {exc}"
    return True, f"action {action} matches required_calls"


def tool_calls_since(conn, seq: int) -> list[dict]:
    """Every tool call since `seq`, with the full envelope it returned."""
    from shopagent.db import audit_after

    return [
        {"tool": row["tool"], "args": json.loads(row["args_json"]), "result": json.loads(row["result_json"])}
        for row in audit_after(conn, seq)
    ]


def run_case(model, tokenizer, eos_id: int, case: dict, file_name: str) -> dict:
    """One case, one attempt. A bad draft is recorded, never raised."""
    from shopagent.bench import _check_reply, match_plan, prepare_case
    from shopagent.compose import ComposeError, compose_reply
    from shopagent.corpus import _message_row
    from shopagent.db import audit_after, last_audit_seq
    from shopagent.gate import evaluate
    from shopagent.referee import check_expect
    from shopagent.sft import MODE_PROMPTS

    started = time.time()
    path = "composed" if is_composed(case) else "free_text"
    row: dict = {
        "id": case["id"],
        "kind": case["kind"],
        "file": file_name,
        "path": path,
        "attempts": 1,
        "status": "ok",
        "reason": "",
        "plan": None,
        "plan_parse": "",
        "plan_errors": [],
        "reply": None,
        "phase_a": None,
        "state_errors": [],
        "tool_calls": [],
        "prompts": {},
        "generation": {},
        "raw_plan": "",
        "raw_reply": "",
        "tokens": {"plan_prompt": 0, "plan_new": 0, "reply_prompt": 0, "reply_new": 0, "total": 0},
    }

    opening = [
        {"role": "system", "content": MODE_PROMPTS[path]},
        {"role": "user", "content": case["ticket"]},
    ]
    plan_prompt = training_prompt(tokenizer, opening)
    row["prompts"]["plan"] = {"chars": len(plan_prompt), "sha256": sha256_text(plan_prompt)}
    plan_call = generate(model, tokenizer, plan_prompt, eos_id, MAX_NEW_TOKENS_PLAN)
    raw_plan = plan_call["text"]
    row["raw_plan"] = raw_plan
    row["tokens"].update(plan_prompt=plan_call["prompt_tokens"], plan_new=plan_call["new_tokens"])
    row["generation"] = {
        "plan": {
            "max_new": plan_call["max_new"],
            "reduced_budget": plan_call["reduced_budget"],
            "hit_cap": plan_call["hit_cap"],
        }
    }

    plan, plan_parse = parse_composed_plan(raw_plan) if path == "composed" else parse_free_text_plan(raw_plan)
    row["plan_parse"] = plan_parse
    if plan is None:
        row.update(status="rejected", reason=f"plan: {plan_parse}")
        return _finish(row, started)
    row["plan"] = plan

    conn, toolkit, baseline = prepare_case(case)
    try:
        seq = last_audit_seq(conn)
        if path == "composed":
            matched, note = composed_match(case, plan)
            if not matched:
                row["plan_errors"].append(note)
            try:
                reply = compose_reply(conn, ticket_id=case["id"], plan=plan)
            except ComposeError as exc:
                row.update(status="rejected", reason=f"compose: {exc}", tool_calls=tool_calls_since(conn, seq))
                return _finish(row, started)
            row["tool_calls"] = tool_calls_since(conn, seq)
        else:
            results: list[dict] = []
            for call in plan:
                args = call.get("args", {})
                results.append(
                    {"tool": call["name"], "args": args, "result": toolkit.call(call["name"], **args)}
                )
            row["plan_errors"] = match_plan(case, [dict(item) for item in audit_after(conn, seq)])
            row["tool_calls"] = tool_calls_since(conn, seq)

            # The conversation training supervised: system, ticket, plan, the
            # real tool results, then the reply turn. `sft.convert_row` rewrites
            # the system turn to `MODE_PROMPTS[path]` before rendering, so this
            # must too; the trailing assistant turn is the one being generated.
            messages, _target = _message_row(
                case=case, path="free_text", ticket=case["ticket"], plan=plan, reply="", results=results
            )
            messages = [dict(message) for message in messages]
            messages[0]["content"] = MODE_PROMPTS["free_text"]
            reply_prompt = training_prompt(tokenizer, messages[:-1])
            row["prompts"]["reply"] = {"chars": len(reply_prompt), "sha256": sha256_text(reply_prompt)}
            reply_call = generate(model, tokenizer, reply_prompt, eos_id, MAX_NEW_TOKENS_REPLY)
            row["raw_reply"] = reply_call["text"]
            row["tokens"].update(reply_prompt=reply_call["prompt_tokens"], reply_new=reply_call["new_tokens"])
            row["generation"]["reply"] = {
                "max_new": reply_call["max_new"],
                "reduced_budget": reply_call["reduced_budget"],
                "hit_cap": reply_call["hit_cap"],
            }
            reply = strip_stop(reply_call["text"])

        row["reply"] = reply
        if not isinstance(reply, str) or not reply.strip():
            row.update(status="rejected", reason="empty reply")
            return _finish(row, started)

        if path == "composed":
            # A composed sentence is rendered from the row the action wrote, so
            # `required_facts` — which describe a free-text reply — are not
            # checked here. `shopagent.corpus` states and applies the same rule.
            row["reply_facts_ok"] = None
            row["reply_phrases_ok"] = not any(
                phrase.lower() in reply.lower() for phrase in case.get("forbidden_phrases", [])
            )
        else:
            facts_ok, phrases_ok, _notes = _check_reply(case, reply)
            row["reply_facts_ok"] = facts_ok
            row["reply_phrases_ok"] = phrases_ok
        row["state_errors"] = check_expect(conn, case["state_expect"])

        decision = evaluate(
            conn,
            ticket_id=case["id"],
            reply=reply,
            baseline=baseline,
            ticket_text=case["ticket"],
        )
        row["phase_a"] = {
            "verdict": decision.verdict,
            "codes": list(decision.codes),
            "reasons": list(decision.reasons),
            "phase": decision.phase,
        }
        return _finish(row, started)
    finally:
        conn.close()


def _finish(row: dict, started: float) -> dict:
    tokens = row["tokens"]
    tokens["total"] = tokens["plan_prompt"] + tokens["plan_new"] + tokens["reply_prompt"] + tokens["reply_new"]
    row["seconds"] = round(time.time() - started, 1)
    return row


# -- the run --------------------------------------------------------------


def dataset_pin(local_sha256: str, adapter_run_id: str | None) -> dict:
    """Which dataset the student was trained from, and at which window.

    The local entrypoint hashes `data/sft/sft.jsonl`; an adapter run must carry
    that same digest and the same `max_seq_length` in its `run.json`, or the bench
    would be scoring weights trained on other bytes.
    """
    pin = {
        "sft_sha256": local_sha256,
        "bench_max_seq_length": BENCH_MAX_SEQ_LEN,
        "run_id": adapter_run_id,
        "trained": bool(adapter_run_id),
    }
    if not adapter_run_id:
        pin["note"] = "base weights have no training data; the digest names the SFT file the pair is measured against"
        return pin
    run_json = Path(RUNS_DIR) / adapter_run_id / "run.json"
    if not run_json.is_file():
        raise FatalError(f"run_json_missing: {run_json}")
    record = json.loads(run_json.read_text(encoding="utf-8"))
    trained = record["dataset"]["sha256"]
    trained_len = int(record["config"]["max_seq_length"])
    if trained != local_sha256:
        raise FatalError(
            f"dataset_digest: {adapter_run_id} trained on {trained[:16]} and the local SFT is {local_sha256[:16]}"
        )
    if trained_len != BENCH_MAX_SEQ_LEN:
        raise FatalError(
            f"seq_length: {adapter_run_id} trained at {trained_len} and the bench window is {BENCH_MAX_SEQ_LEN}"
        )
    pin.update(
        {
            "run_steps": record["training"]["optimizer_steps"],
            "run_status": record["status"],
            "trainable_parameters": record["config"]["lora"]["trainable_parameters"],
        }
    )
    return pin


def _write_rows(out_dir: Path, rows: list[dict]) -> Path:
    path = out_dir / "rows.jsonl"
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return path


def _tally(rows: list[dict], key: str) -> dict:
    """Approved / rejected / escalated / no-draft per group, in a fixed order."""
    out: dict[str, dict] = {}
    for row in rows:
        group = row.get(key) or "unknown"
        bucket = out.setdefault(
            group, {"APPROVED": 0, "REJECTED": 0, "ESCALATED": 0, "NO_DRAFT": 0}
        )
        bucket[(row.get("phase_a") or {}).get("verdict", "NO_DRAFT")] += 1
    return {group: out[group] for group in sorted(out)}


@app.function(
    gpu=GPU,
    timeout=TIMEOUT_SECONDS,
    cpu=CPU_CORES,
    memory=MEMORY_MIB,
    volumes={VOLUME_MOUNT: volume},
    retries=0,
)
def bench(payload: dict) -> dict:
    """One attempt per case, Phase A, then the rows and the manifest on the Volume."""
    import torch
    from shopagent.sft import FREE_TEXT_SYSTEM_PROMPT, MODE_PROMPTS
    from shopagent.teacher import SYSTEM_PROMPT

    started = time.time()
    student = payload["student"]
    name = payload["out"]
    if student not in STUDENTS:
        raise FatalError(f"student: expected one of {STUDENTS}, got {student!r}")
    cases = payload["cases"]
    bound = worst_case_usd()
    print(
        f"bench {name}: student {student}, {len(cases)} cases, gpu {GPU}, timeout {TIMEOUT_SECONDS}s; "
        f"worst case ${bound['total_usd']:.2f}",
        flush=True,
    )
    print(f"selection: {json.dumps(payload['selection']['per_file'], sort_keys=True)}", flush=True)
    print(f"benchmark: cases_digest {payload['benchmark']['cases_digest'][:16]}, verified locally", flush=True)
    print(f"dataset: sft.jsonl sha256 {payload['dataset']['sha256'][:16]}, window {BENCH_MAX_SEQ_LEN}", flush=True)
    print(f"distribution: {TICKET_NOTE}", flush=True)

    pin = dataset_pin(payload["dataset"]["sha256"], payload.get("adapter_run_id"))
    print(f"dataset pin: {json.dumps(pin, sort_keys=True)}", flush=True)

    tokenizer_sha = tokenizer_json_sha256()
    if tokenizer_sha != TOKENIZER_JSON_SHA256:
        raise FatalError(f"tokenizer_bytes: tokenizer.json is {tokenizer_sha}, expected {TOKENIZER_JSON_SHA256}")
    load_started = time.time()
    model, tokenizer, eos_id, proof = load_model(student, payload.get("adapter_run_id"))
    print(f"model: loaded in {time.time() - load_started:.1f}s, eos id {eos_id}", flush=True)
    if proof:
        print(
            f"adapter: {proof['dir']}, active {proof['active_adapters']}, "
            f"{proof['trainable_parameters']} trainable parameters; {proof['tensor']} is torch.equal "
            "to its saved tensor",
            flush=True,
        )

    files = payload["selection"]["files"]
    out_dir = Path(BENCH_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    for index, case in enumerate(cases):
        try:
            row = run_case(model, tokenizer, eos_id, case, files[index])
        except FatalError:
            raise
        except Exception as exc:  # one broken case must not lose the rest
            row = {
                "id": case.get("id"),
                "kind": case.get("kind"),
                "file": files[index],
                "path": "error",
                "attempts": 1,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "phase_a": None,
            }
        rows.append(row)
        decision = row.get("phase_a") or {}
        print(
            f"[{index + 1}/{len(cases)}] {row['id']} {row['path']} "
            f"{decision.get('verdict', row['status'])} {decision.get('codes', [])} "
            f"{row.get('seconds', 0)}s",
            flush=True,
        )
        if len(rows) % FLUSH_EVERY == 0:
            _write_rows(out_dir, rows)
            volume.commit()
            print(f"flush: {len(rows)} rows committed", flush=True)

    wall = time.time() - started
    peak_gb = torch.cuda.max_memory_allocated() / 2**30
    hardware = {
        "gpu_requested": GPU,
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "peak_memory_gb": round(peak_gb, 2),
        "cpu_cores": CPU_CORES,
        "memory_mib": MEMORY_MIB,
    }
    verdicts = Counter((row.get("phase_a") or {}).get("verdict", "NO_DRAFT") for row in rows)
    tokens = {
        key: sum(int(row.get("tokens", {}).get(key, 0)) for row in rows)
        for key in ("plan_prompt", "plan_new", "reply_prompt", "reply_new", "total")
    }
    cost = cost_for(wall)

    manifest = {
        "name": name,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "student": {
            "role": student,
            "adapter_run_id": payload.get("adapter_run_id"),
            "adapter": proof,
        },
        "model": {
            "id": MODEL_ID,
            "revision": MODEL_REVISION,
            "load_in_4bit": True,
            "enable_thinking": ENABLE_THINKING,
            "max_seq_length": BENCH_MAX_SEQ_LEN,
            "tokenizer_json_sha256": tokenizer_sha,
        },
        "benchmark": payload["benchmark"],
        "dataset": {**pin, "file": payload["dataset"]["file"]},
        "cases": {
            "count": len(rows),
            "per_file": payload["selection"]["per_file"],
            "composed": payload["selection"]["composed"],
            "composed_available": payload["selection"]["composed_available"],
            "pool": payload["selection"]["pool"],
            "limit": payload["selection"]["limit"],
            "ids": payload["selection"]["order"],
            "selection_rule": payload["selection"]["rule"],
        },
        "config": {
            "attempts_per_case": 1,
            "decoding": "greedy, do_sample False (temperature 0)",
            "max_new_tokens_plan": MAX_NEW_TOKENS_PLAN,
            "max_new_tokens_reply": MAX_NEW_TOKENS_REPLY,
            "paths": {
                "composed": "case carries composed and corpus.composed_unsupported() is None",
                "free_text": "everything else",
            },
            "prompts": {
                "free_text": {
                    "source": "shopagent.sft.FREE_TEXT_SYSTEM_PROMPT (teacher prompt + contract)",
                    "sha256": sha256_text(FREE_TEXT_SYSTEM_PROMPT),
                },
                "composed": {
                    "source": "shopagent.sft.COMPOSED_SYSTEM_PROMPT",
                    "sha256": sha256_text(MODE_PROMPTS["composed"]),
                },
                "teacher_prompt_sha256": sha256_text(SYSTEM_PROMPT),
            },
            "render_rule": (
                "the generation prompt is compared byte for byte against the slice of the full chat "
                "render that shopagent.sft supervised; a mismatch fails the case"
            ),
            "user_turn": "the frozen case ticket",
            "phase_b": "off; no judge, no network, no key",
            "store": "one fresh seeded sqlite database per case, case faults injected",
            "gate": "shopagent.gate.evaluate, Phase A, commit False",
        },
        "distribution": {"note": TICKET_NOTE, "training_user_turn": "the teacher's paraphrase"},
        "limits": list(LIMITS),
        "results": {
            "phase_a": {
                verdict: verdicts.get(verdict, 0)
                for verdict in ("APPROVED", "REJECTED", "ESCALATED", "NO_DRAFT")
            },
            "errors": sum(1 for row in rows if row["status"] == "error"),
            "rejected_before_gate": sum(1 for row in rows if row["status"] == "rejected"),
            "plan_errors": sum(1 for row in rows if row.get("plan_errors")),
            "reply_budget_reduced": sum(
                1 for row in rows if (row.get("generation") or {}).get("reply", {}).get("reduced_budget")
            ),
            "seconds_per_case": round(wall / len(rows), 1) if rows else 0,
            "by_path": _tally(rows, "path"),
            "by_file": _tally(rows, "file"),
        },
        "tokens": tokens,
        "time": {"wall_seconds": round(wall, 1)},
        "hardware": hardware,
        "cost": cost,
    }

    out_dir = Path(BENCH_DIR) / name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_rows(out_dir, rows)
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    volume.commit()

    print(f"phase A: {manifest['results']['phase_a']}", flush=True)
    print(
        f"phase A by path: {json.dumps(manifest['results']['by_path'], sort_keys=True)}", flush=True
    )
    print(
        f"plan errors {manifest['results']['plan_errors']}, no draft "
        f"{manifest['results']['rejected_before_gate']}, errors {manifest['results']['errors']}",
        flush=True,
    )
    print(
        f"gpu: {hardware['gpu_name']} {hardware['gpu_total_gb']} GB, peak {peak_gb:.1f} GB; "
        f"wall {wall:.0f}s; estimated ${cost['total_usd']:.2f} "
        "(compute only, excludes Volume commit)",
        flush=True,
    )
    print(f"rows: {out_dir / 'rows.jsonl'}", flush=True)
    return {"rows": rows, "manifest": manifest}


def _frozen_record() -> dict:
    """`verify-frozen` gate: refuse to bench a benchmark whose bytes moved."""
    from shopagent.freeze import read, verify

    ok, problems = verify()
    if not ok:
        raise FatalError("benchmark is not frozen: " + "; ".join(problems[:3]))
    record = read()
    if record is None:
        raise FatalError("bench/FROZEN.json is missing")
    return record


@app.local_entrypoint()
def main(student: str = "base", limit: int = 0, out: str = "bench", run_id: str = ADAPTER_RUN_ID):
    """Verify the frozen benchmark, print the bound, then hand the run to a GPU."""
    if student not in STUDENTS:
        raise SystemExit(f"--student must be one of {STUDENTS}")
    if not out:
        raise SystemExit("--out needs a name")
    record = _frozen_record()
    dataset = REPO_ROOT / "data" / "sft" / "sft.jsonl"
    if not dataset.is_file():
        raise SystemExit(f"no dataset at {dataset}; `python -m shopagent.sft` writes it")
    dataset_sha256 = hashlib.sha256(dataset.read_bytes()).hexdigest()
    payload = build_payload(student, limit, run_id, dataset_sha256)
    payload["out"] = out
    selection = payload["selection"]
    bound = worst_case_usd()
    print(
        f"bench {out}: student {student}"
        + (f" (run {run_id})" if student == "adapter" else "")
        + f", {len(payload['cases'])} of {record['count']} cases, gpu {GPU}, timeout {TIMEOUT_SECONDS}s"
    )
    print(
        f"frozen: cases_digest {record['cases_digest'][:16]}, original "
        f"{record['original_cases_digest'][:16]}; verify-frozen OK"
    )
    print(f"dataset: sft.jsonl sha256 {dataset_sha256[:16]}, bench window {BENCH_MAX_SEQ_LEN}")
    print(f"per file: {json.dumps(selection['per_file'], sort_keys=True)}; composed {selection['composed']}")
    print(f"worst case at the timeout: ${bound['total_usd']:.2f}")

    result = bench.remote(payload)
    rows = result["rows"]
    manifest = result["manifest"]

    local = REPO_ROOT / "out"
    local.mkdir(exist_ok=True)
    rows_path = local / f"{out}.jsonl"
    manifest_path = local / f"{out}.manifest.json"
    rows_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print()
    for row in rows:
        decision = row.get("phase_a") or {}
        print(
            f"  {row['id']:<6} {row['path']:<9} {decision.get('verdict', row['status']):<9} "
            f"{row.get('seconds', 0):>5}s"
        )
    print(f"rows: {rows_path}")
    print(f"manifest: {manifest_path}")
    print(f"volume: {BENCH_DIR}/{out}/rows.jsonl")
