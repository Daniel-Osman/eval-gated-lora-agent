"""Stage 6 slice 2 — the deployed endpoint: one request runs the whole loop.

A composed preset id, or one free-text ticket, in; the plan, every tool call with
its result, Phase A's verdict and codes, the reply, and timings out. The plan runs
against a **fresh seeded mock store**, in memory, discarded with the request.

Three things this file is deliberately careful about:

* **One gate attempt.** `shopagent.gate.run_gated` is not used: its retry loop
  re-proposes and re-runs the action on the same store, and a composed refund
  retried that way would refund twice. `evaluate` runs once; a rejection is
  returned rejected.
* **Phase B stays off.** The judge needs `OPENCODE_API_KEY`, and that key does not
  go into a remote sandbox. The payload says so on every response.
* **No untracked input.** The composed presets come from the frozen
  `bench/cases/normal-ext.jsonl` (tracked), not from `data/sft/sft.jsonl`
  (git-ignored). See `TRAINING_SEEN` for which ids the composed training set held.

Endpoint-level refusals carry `SERVE_` codes so they cannot be mistaken for the
gate's own vocabulary (`UNVERIFIED_CLAIM`, `STATE_INVARIANT`, ...).
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
BENCH_LOCAL = REPO_ROOT / "bench"
BENCH_REMOTE = "/root/bench"
PRESET_FILE = ("cases", "normal-ext.jsonl")

VOLUME_NAME = "eval-gated-lora-agent"
VOLUME_MOUNT = "/vol"
MODEL_DIR = f"{VOLUME_MOUNT}/serve/slice1/awq-w4a16"

# The measured operating point, from docs/SERVING.md. 256 sequences OOMs on the
# A10; 8 is the only width that was tested and loaded.
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 8
GPU_MEMORY_UTILIZATION = 0.92
SCALEDOWN_SECONDS = 60
FUNCTION_TIMEOUT = 900

# Caps. A ticket over the char cap is a 400; a prompt that plus its reply would
# not fit the context is a 400 too. Generation counts are fixed per mode: the
# composed path generates once, the free-text path twice, never more.
MAX_TICKET_CHARS = 1200
MAX_NEW_TOKENS_COMPOSED = 64
MAX_NEW_TOKENS_PLAN = 128
MAX_NEW_TOKENS_REPLY = 256
MAX_TOOL_CALLS = 6

# One fixed sentence for a tool that raised. `repr(exc)` would name the tool's own
# method and arguments in a page a stranger reads, so the detail stops at the log.
TOOL_FAILED_REASON = "a tool call was rejected; the plan stopped there"

# The composed rows the student trained on, 13 train + 1 validation (N-104), as
# recorded in docs/SERVING.md's composed-suite table. A preset outside this set
# is a frozen bench scenario the model never saw; it is labelled as such rather
# than claimed as seen.
TRAINING_SEEN = frozenset(
    {
        "N-82",
        "N-88",
        "N-89",
        "N-97",
        "N-99",
        "N-100",
        "N-101",
        "N-102",
        "N-103",
        "N-104",
        "N-105",
        "N-106",
        "N-107",
        "N-108",
    }
)

PHASE_B = {
    "enabled": False,
    "reason": (
        "Phase B is the DeepSeek judge behind the subscription key. That key does not go "
        "into a remote sandbox, so the deployed app runs Phase A only."
    ),
}

PROVENANCE = (
    "Presets are frozen composed bench scenarios from bench/cases/normal-ext.jsonl. "
    "The ticket is the bench's wording; the SFT records paraphrased it, so this is the "
    "same scenario, not the exact training prompt. Reproducing a preset shows the serving "
    "path preserved behaviour; it is not a generalisation claim."
)

# Built once per container from the tokenizer, appended where the assistant turn
# begins. It is the shape every rendered record ends with: the empty think block
# then two newlines.
ASSISTANT_PREFIX = "<|im_start|>assistant\n thinking\n\n</think>\n\n"

MODULE_STARTED = time.time()

app = modal.App("eval-gated-lora-agent")

volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=False, version=2)

health_image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]")
    .add_local_python_source("serve", "shopagent")
    .add_local_dir(str(BENCH_LOCAL), BENCH_REMOTE)
)

agent_image = (
    # vLLM 0.33.0 JIT-compiles a kernel for this hybrid architecture at engine
    # init and refuses to start without nvcc; only a devel base image was enough
    # (docs/SERVING.md). Versions pinned to what was measured in out/serve-versions.log.
    modal.Image.from_registry("nvidia/cuda:13.0.1-devel-ubuntu24.04", add_python="3.12")
    .pip_install(
        "vllm==0.33.0",
        "transformers==5.20.2",
        "fastapi[standard]",
        "huggingface_hub",
        "torchvision",
        "pillow",
        "ninja",
    )
    .apt_install("build-essential")
    .env(
        {
            "PYTHONUNBUFFERED": "1",
            "TOKENIZERS_PARALLELISM": "false",
            # Read-only Volume, local model path: nothing to download, and nothing
            # to write outside the container.
            "HF_HOME": "/root/.cache/hf",
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
        }
    )
    .add_local_python_source("serve", "shopagent")
    .add_local_dir(str(BENCH_LOCAL), BENCH_REMOTE)
)


# --------------------------------------------------------------------------
# request contract — pure, no model, no store
# --------------------------------------------------------------------------


class RequestError(ValueError):
    """Bad input. Becomes a 400 whose text names the cap and nothing internal."""


def load_presets(bench_root: str | Path) -> dict[str, dict]:
    """The composed presets: frozen bench cases with `composed: true`.

    The file is tracked and frozen (`bench/FROZEN.json`); reading it is the only
    source of preset tickets, so no git-ignored artifact decides what the demo
    can run.
    """
    path = Path(bench_root).joinpath(*PRESET_FILE)
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as exc:
        raise RequestError(f"preset file is missing: {path.name}") from exc
    found: dict[str, dict] = {}
    for line in lines:
        if not line.strip():
            continue
        case = json.loads(line)
        if not case.get("composed"):
            continue
        ticket = case.get("ticket")
        if not isinstance(ticket, str) or not ticket.strip():
            raise RequestError(f"preset {case.get('id')} carries no ticket")
        found[case["id"]] = {
            "id": case["id"],
            "title": case.get("title", ""),
            "kind": case.get("kind", ""),
            "ticket": ticket,
            "training_seen": case["id"] in TRAINING_SEEN,
            "case": case,
        }
    if not found:
        raise RequestError("the preset file holds no composed case")
    return found


def preset_view(preset: dict) -> dict:
    """What is safe to show: no required calls, facts or expected answers."""
    return {
        "id": preset["id"],
        "title": preset["title"],
        "kind": preset["kind"],
        "training_seen": preset["training_seen"],
        "ticket": preset["ticket"],
    }


def validate_request(payload: object, presets: dict[str, dict]) -> dict:
    """Exactly one of `scenario` (composed) or `ticket` (free text)."""
    if not isinstance(payload, dict):
        raise RequestError("body must be a JSON object")
    scenario = payload.get("scenario")
    ticket = payload.get("ticket")
    if (scenario is None) == (ticket is None):
        raise RequestError(
            'send exactly one of "scenario" (a composed preset id) or "ticket" (free text)'
        )
    if scenario is not None:
        if not isinstance(scenario, str) or scenario not in presets:
            ids = ", ".join(sorted(presets))
            raise RequestError(f"unknown scenario; the composed presets are: {ids}")
        return {"mode": "composed", "preset": presets[scenario]}
    if not isinstance(ticket, str) or not ticket.strip():
        raise RequestError('"ticket" must be a non-empty string')
    if len(ticket) > MAX_TICKET_CHARS:
        raise RequestError(
            f"ticket is {len(ticket)} characters; the cap is {MAX_TICKET_CHARS}. "
            "Shorten the message and send it again."
        )
    return {"mode": "free_text", "ticket": ticket}


def parse_plan_list(text: str) -> tuple[list | None, str]:
    """A free-text plan: one JSON list of {"name", "args"} objects, nothing else.

    The same fence and trailing-text rules as the composed parser
    (`serve.suite.parse_plan`), because both prompts demand one JSON value.
    """
    from shopagent.teacher import ALLOWED_PLAN_TOOLS
    from serve import suite

    cleaned = suite._strip_stop_token(text.strip())
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S)
    if fenced:
        cleaned = suite._strip_stop_token(fenced.group(1).strip())
    elif "```" in cleaned:
        return None, "text outside the fence: a fence must wrap the whole output"
    if not cleaned:
        return None, "empty output"
    try:
        payload, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"not one JSON value at the root: {exc}"
    if not isinstance(payload, list):
        return None, f"plan is {type(payload).__name__}, not a list at the root"
    trailing = cleaned[end:].strip()
    if trailing:
        return None, f"trailing text after the plan: {trailing[:60]!r}"
    allowed = set(ALLOWED_PLAN_TOOLS)
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            return None, f"call {index} is {type(entry).__name__}, not an object"
        unknown = sorted(set(entry) - {"name", "args"})
        if unknown:
            return None, f"call {index} carries unknown keys {unknown}"
        name = entry.get("name")
        if not isinstance(name, str) or name not in allowed:
            return None, f"call {index} names {name!r}, which is not a plan tool"
        args = entry.get("args", {})
        if not isinstance(args, dict):
            return None, f"call {index} args is {type(args).__name__}, not an object"
    return payload, "ok"


def composed_messages(ticket: str) -> list[dict]:
    from shopagent.sft import COMPOSED_SYSTEM_PROMPT

    return [
        {"role": "system", "content": COMPOSED_SYSTEM_PROMPT},
        {"role": "user", "content": ticket},
    ]


def free_text_messages(ticket: str) -> list[dict]:
    from shopagent.sft import FREE_TEXT_SYSTEM_PROMPT

    return [
        {"role": "system", "content": FREE_TEXT_SYSTEM_PROMPT},
        {"role": "user", "content": ticket},
    ]


def tool_messages(calls: list[dict]) -> list[dict]:
    """The training shape: one tool turn per call, the full envelope as JSON.

    `shopagent.corpus._message_row` builds exactly this, with `sort_keys=True`;
    the chat template renders it as the `<tool_response>` turn the model saw.
    """
    return [
        {
            "role": "tool",
            "content": json.dumps(
                {"args": call["args"], "result": call["result"], "tool": call["tool"]},
                sort_keys=True,
            ),
        }
        for call in calls
    ]


def render_prompt(tokenizer, messages: list[dict]) -> str:
    """The chat template's rendering plus the assistant turn it does not add.

    The template closes the last turn with `<|im_end|>` and a trailing newline —
    the records show `...<|im_end|>\n<|im_start|>assistant`. The prefix is
    concatenated onto that text unchanged, which is the byte shape training saw.
    """
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=False
    )
    from serve import suite

    if not text.rstrip("\n").endswith(suite.IM_END):
        raise RuntimeError(f"chat template did not close the last turn: {text[-40:]!r}")
    return text + ASSISTANT_PREFIX


def check_context(prompt_tokens: int, max_new_tokens: int) -> None:
    if prompt_tokens + max_new_tokens > MAX_MODEL_LEN:
        raise RequestError(
            f"prompt is {prompt_tokens} tokens and the answer needs up to {max_new_tokens}; "
            f"max_model_len is {MAX_MODEL_LEN}"
        )


# --------------------------------------------------------------------------
# the loop — a fake tokenizer and a fake generate are enough to test all of it
# --------------------------------------------------------------------------


def read_calls(conn, since: int) -> list[dict]:
    """Every tool call one attempt wrote, with its full result envelope."""
    from shopagent.gate import _calls_since

    return _calls_since(conn, since)


def _verdict(decision, committed: bool) -> dict:
    payload = decision.as_dict()
    payload["reply_committed"] = bool(committed)
    return payload


def _refused(codes: list[str], reasons: list[str]) -> dict:
    return {
        "verdict": "REJECTED",
        "codes": codes,
        "reasons": reasons,
        "phase": "A",
        "provenance": [],
        "reply_committed": False,
    }


def run_request(
    selection: dict,
    *,
    tokenizer,
    generate,
    clock=time.perf_counter,
) -> dict:
    """One request, one fresh store. `generate(ids, max_new) -> {"text","output_tokens"}`."""
    from shopagent.gate import evaluate
    from shopagent.referee import snapshot
    from shopagent.seed import fresh
    from shopagent.db import last_audit_seq
    from serve import suite

    started = clock()
    mode = selection["mode"]
    timings: dict = {"generations": 0}
    plan = None
    calls: list[dict] = []
    reply = ""
    verdict: dict | None = None

    if mode == "composed":
        preset = selection["preset"]
        ticket = preset["ticket"]
        prompt = render_prompt(tokenizer, composed_messages(ticket))
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        timings["prompt_tokens"] = len(ids)
        check_context(len(ids), MAX_NEW_TOKENS_COMPOSED)

        born = clock()
        sent = generate(ids, MAX_NEW_TOKENS_COMPOSED)
        timings["plan_seconds"] = round(clock() - born, 3)
        timings["generations"] = 1
        timings["output_tokens"] = {"plan": sent.get("output_tokens")}

        plan, why = suite.parse_plan(sent["text"])
        if plan is None:
            verdict = _refused(["SERVE_PLAN_UNPARSED"], [why])
        else:
            conn = fresh()
            baseline = snapshot(conn)
            since = last_audit_seq(conn)
            from shopagent.compose import ComposeError, compose_reply

            born = clock()
            try:
                reply = compose_reply(conn, ticket_id=preset["id"], plan=plan)
            except ComposeError as exc:
                timings["action_seconds"] = round(clock() - born, 3)
                calls = read_calls(conn, since)
                verdict = _refused(["SERVE_COMPOSE_FAILED"], [str(exc)])
            else:
                timings["action_seconds"] = round(clock() - born, 3)
                calls = read_calls(conn, since)
                born = clock()
                decision = evaluate(
                    conn,
                    ticket_id=preset["id"],
                    reply=reply,
                    baseline=baseline,
                    phase_b="off",
                    calls=calls,
                    ticket_text=ticket,
                    commit=True,
                )
                timings["gate_seconds"] = round(clock() - born, 3)
                verdict = _verdict(decision, decision.approved)
    else:
        ticket = selection["ticket"]
        prompt = render_prompt(tokenizer, free_text_messages(ticket))
        ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        timings["prompt_tokens"] = len(ids)
        check_context(len(ids), MAX_NEW_TOKENS_PLAN)

        born = clock()
        sent = generate(ids, MAX_NEW_TOKENS_PLAN)
        timings["plan_seconds"] = round(clock() - born, 3)
        timings["generations"] = 1
        timings["output_tokens"] = {"plan": sent.get("output_tokens")}

        plan, why = parse_plan_list(sent["text"])
        if plan is None:
            verdict = _refused(["SERVE_PLAN_UNPARSED"], [why])
        elif len(plan) > MAX_TOOL_CALLS:
            verdict = _refused(
                ["SERVE_PLAN_TOO_LONG"],
                [f"plan asks for {len(plan)} calls; the cap is {MAX_TOOL_CALLS}"],
            )
        else:
            from shopagent.tools import ToolKit

            conn = fresh()
            baseline = snapshot(conn)
            since = last_audit_seq(conn)
            toolkit = ToolKit(conn, ticket_id="FREE-TEXT")
            born = clock()
            tool_failed = False
            for entry in plan:
                try:
                    toolkit.call(entry["name"], **entry.get("args", {}))
                except Exception as exc:
                    # A raising tool stops the plan: the rest of it was written
                    # against state this call did not produce. The visitor gets a
                    # fixed sentence; the log gets the type name and nothing about
                    # the tool's own code.
                    print(f"tool call stopped the plan: {type(exc).__name__}", flush=True)
                    tool_failed = True
                    break
            timings["tools_seconds"] = round(clock() - born, 3)
            calls = read_calls(conn, since)

            if tool_failed:
                verdict = _refused(["SERVE_TOOL_FAILED"], [TOOL_FAILED_REASON])
            else:
                messages = free_text_messages(ticket) + [
                    {"role": "assistant", "content": json.dumps(plan, sort_keys=True)}
                ] + tool_messages(calls)
                followup = render_prompt(tokenizer, messages)
                followup_ids = tokenizer(followup, add_special_tokens=False)["input_ids"]
                if len(followup_ids) + MAX_NEW_TOKENS_REPLY > MAX_MODEL_LEN:
                    verdict = _refused(
                        ["SERVE_CONTEXT_OVERFLOW"],
                        [
                            f"the tool results grew the prompt to {len(followup_ids)} tokens; "
                            f"max_model_len is {MAX_MODEL_LEN}"
                        ],
                    )
                else:
                    born = clock()
                    sent = generate(followup_ids, MAX_NEW_TOKENS_REPLY)
                    timings["reply_seconds"] = round(clock() - born, 3)
                    timings["generations"] = 2
                    timings["output_tokens"]["reply"] = sent.get("output_tokens")
                    reply = sent["text"]
                    born = clock()
                    decision = evaluate(
                        conn,
                        ticket_id="FREE-TEXT",
                        reply=reply,
                        baseline=baseline,
                        phase_b="off",
                        calls=calls,
                        ticket_text=ticket,
                        commit=True,
                    )
                    timings["gate_seconds"] = round(clock() - born, 3)
                    verdict = _verdict(decision, decision.approved)

    timings["total_seconds"] = round(clock() - started, 3)

    response = {
        "mode": mode,
        "phase_b": PHASE_B,
        "plan": plan,
        "calls": calls,
        "verdict": verdict,
        "reply": reply,
        "timings": timings,
        "provenance": PROVENANCE,
        "store": "in-memory seeded mock shop, discarded with the request",
    }
    if mode == "composed":
        preset = selection["preset"]
        matched, why = suite.composed_verdict(preset["case"], plan)
        response["scenario"] = {**preset_view(preset), "prompt_tokens": None}
        response["scenario"]["matches_required_calls"] = matched
        response["scenario"]["match_reason"] = why
        response["unproven"] = [
            (
                "presets are frozen training-seen bench scenarios; reproducing them is a "
                "compatibility check, not a quality claim"
            )
            if preset["training_seen"]
            else (
                "this preset is a frozen bench case the adapter never trained on; a match "
                "is a serving-compatibility result, not generalisation"
            ),
        ]
    else:
        response["ticket"] = selection["ticket"]
        response["unproven"] = [
            "free-text quality on unseen tickets is unmeasured: SPEC.md section 7's held-out "
            "set has no student run (NOW.md, Open)",
            "this is a demo of the gate, not a claim that the model's prose is correct",
        ]
    return response


# --------------------------------------------------------------------------
# container-side model plumbing
# --------------------------------------------------------------------------

_PRESETS: dict | None = None
_TOKENIZER = None
_ENGINE = {"llm": None, "init_seconds": None, "loaded_at": None}
_REQUESTS = {"count": 0}


def _presets() -> dict:
    global _PRESETS
    if _PRESETS is None:
        _PRESETS = load_presets(BENCH_REMOTE)
    return _PRESETS


def _engine_info() -> dict:
    return {
        "artifact": MODEL_DIR,
        "max_model_len": MAX_MODEL_LEN,
        "max_num_seqs": MAX_NUM_SEQS,
        "gpu_memory_utilization": GPU_MEMORY_UTILIZATION,
        "init_seconds": _ENGINE["init_seconds"],
        "loaded_at": _ENGINE["loaded_at"],
    }


def _versions() -> dict:
    import sys

    import torch
    import transformers
    import vllm

    return {
        "vllm": vllm.__version__,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "python": sys.version.split()[0],
    }


@app.cls(
    image=agent_image,
    gpu="A10",
    cpu=4.0,
    memory=32 * 1024,
    timeout=FUNCTION_TIMEOUT,
    startup_timeout=FUNCTION_TIMEOUT,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_SECONDS,
    # No `retries`: Modal refuses a retry setting on a Web Function at all
    # (`InvalidError: Web Functions do not support retries`), so the endpoint is
    # single-shot by construction. A retry would re-run a mutating plan anyway.
    volumes={VOLUME_MOUNT: volume.read_only()},
)
class Agent:
    """The model lives as long as the container: one A10, one request at a time."""

    @modal.enter()
    def load(self):
        """Build the engine once per container, in the main thread."""
        from transformers import AutoTokenizer
        from vllm import LLM

        global _TOKENIZER

        started = time.time()
        _TOKENIZER = AutoTokenizer.from_pretrained(MODEL_DIR)
        _ENGINE["llm"] = LLM(
            model=MODEL_DIR,
            max_model_len=MAX_MODEL_LEN,
            max_num_seqs=MAX_NUM_SEQS,
            gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
            dtype="bfloat16",
            enforce_eager=False,
            disable_log_stats=True,
        )
        _ENGINE["init_seconds"] = round(time.time() - started, 1)
        _ENGINE["loaded_at"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        print(f"engine: ready in {_ENGINE['init_seconds']}s, presets {len(_presets())}", flush=True)

    def _generate(self, ids: list[int], max_new_tokens: int) -> dict:
        from serve import suite
        from vllm import SamplingParams

        params = SamplingParams(max_tokens=max_new_tokens, temperature=0.0, top_p=1.0)
        output = _ENGINE["llm"].generate([{"prompt_token_ids": ids}], params, use_tqdm=False)[0]
        choice = output.outputs[0]
        return {
            "text": suite._strip_stop_token(choice.text),
            "output_tokens": len(choice.token_ids),
        }

    def _answer(self, item: dict) -> dict:
        """Validate and run one request. `RequestError` is the only refusal."""
        selection = validate_request(item, _presets())

        _REQUESTS["count"] += 1
        result = run_request(selection, tokenizer=_TOKENIZER, generate=self._generate)

        result["engine"] = {
            **_engine_info(),
            "request_number": _REQUESTS["count"],
            "versions": _versions(),
            "container_uptime_seconds": round(time.time() - MODULE_STARTED, 1),
        }
        return result

    @modal.fastapi_endpoint(method="POST", docs=False)
    def run(self, item: dict):
        from fastapi import HTTPException

        try:
            return self._answer(item)
        except RequestError as exc:
            raise HTTPException(status_code=400, detail=str(exc))

    @modal.method()
    def serve(self, item: dict) -> str:
        """In-process door for the demo app: the HTTP body, plus its status, as JSON text.

        A refusal is a value, not a raise: an `HTTPException` raised in a
        container reaches the caller as traceback text, not as a 400.

        Text, not a dict: the caller is a GPU-free image without torch, so
        unpickling the agent's own objects there is a deserialization failure
        waiting to happen. `json.dumps` also refuses anything that is not plain
        JSON, so the wire format cannot drift from the contract.
        """
        try:
            envelope = {"status_code": 200, "body": self._answer(item)}
        except RequestError as exc:
            envelope = {"status_code": 400, "body": {"detail": str(exc)}}
        return json.dumps(envelope, ensure_ascii=False)


def health_payload(bench_root: str | Path) -> dict:
    try:
        presets = load_presets(bench_root)
    except RequestError as exc:
        return {"ok": False, "engine": "not loaded", "error": str(exc)}
    return {
        "ok": True,
        "engine": "not loaded",
        "model": "not touched by this endpoint",
        "phase_b": "off",
        "presets": [preset_view(preset) for preset in sorted(presets.values(), key=lambda p: p["id"])],
    }


@app.function(
    image=health_image,
    cpu=0.25,
    memory=256,
    timeout=30,
    min_containers=0,
    max_containers=1,
    scaledown_window=SCALEDOWN_SECONDS,
)
@modal.fastapi_endpoint(method="GET", docs=False)
def health():
    """Cheap, model-free: no GPU, no Volume, no weights touched."""
    return health_payload(BENCH_REMOTE)
