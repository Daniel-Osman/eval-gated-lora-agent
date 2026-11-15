"""One fresh process on the GPU it is given: does the artifact fit, and how fast.

`serve/feasibility.py::fit` runs this as a subprocess so "cold start" means what
it says: an interpreter that has never held this model constructs the vLLM engine
and the time to first answer is measured from that. It excludes container start,
image pull and scale-from-zero scheduling, and the log says so.

Emits one `PROBE_JSON {...}` line and prints everything else as it goes.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
import time

PROCESS_START = time.time()

MAX_NEW_TOKENS = 256
# Tightest first: the gate needs 2,757 tokens of prompt plus a 256-token reply.
# The hybrid model also holds 72 MiB of recurrent state per sequence, so the
# batch width is part of "does it fit" and appears as its own rung.
ATTEMPTS = (
    {"max_model_len": 4096, "enforce_eager": False},
    {"max_model_len": 4096, "enforce_eager": False, "max_num_seqs": 8},
    {"max_model_len": 2048, "enforce_eager": True, "max_num_seqs": 4},
)


class LogCapture(logging.Handler):
    """Keep vLLM's own lines, so KV cache size is quoted from the engine."""

    def __init__(self) -> None:
        super().__init__()
        self.lines: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.lines.append(record.getMessage())
        except Exception:  # noqa: BLE001 - a log line must never kill a measurement
            pass

    def matching(self, needle: str) -> list[str]:
        return [line for line in self.lines if needle in line]


def gpu_facts() -> dict:
    import torch

    query = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader"],
        capture_output=True,
        text=True,
    )
    return {
        "torch_cuda_name": torch.cuda.get_device_name(0),
        "torch_total_gib": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2),
        "nvidia_smi": query.stdout.strip() or query.stderr.strip(),
    }


def vram_used() -> dict:
    import torch

    free, total = torch.cuda.mem_get_info()
    return {
        "used_gib": round((total - free) / 2**30, 2),
        "total_gib": round(total / 2**30, 2),
        "free_gib": round(free / 2**30, 2),
    }


def load_engine(model_dir: str, attempt: dict) -> tuple[object, dict]:
    """Construct the engine once. Every failure is reported with its text."""
    from vllm import LLM

    torch_peak = {"max_memory_allocated_gib": None}
    started = time.time()
    try:
        engine_kwargs = {
            "model": model_dir,
            "max_model_len": attempt["max_model_len"],
            "gpu_memory_utilization": 0.92,
            "dtype": "bfloat16",
            "enforce_eager": attempt["enforce_eager"],
            "disable_log_stats": True,
        }
        if "max_num_seqs" in attempt:
            engine_kwargs["max_num_seqs"] = attempt["max_num_seqs"]
        llm = LLM(**engine_kwargs)
    except Exception as exc:  # noqa: BLE001 - the failure is the measurement
        import torch

        torch_peak["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        return None, {
            "attempt": attempt,
            "ok": False,
            "error_type": type(exc).__name__,
            "error": str(exc).strip().splitlines()[0][:400] if str(exc).strip() else repr(exc),
            "seconds": round(time.time() - started, 1),
            "vram": vram_used() if torch.cuda.is_available() else None,
        }
    import torch

    torch_peak["max_memory_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return llm, {
        "attempt": attempt,
        "ok": True,
        "seconds": round(time.time() - started, 1),
        "torch_max_memory_allocated_gib": torch_peak["max_memory_allocated_gib"],
        "vram": vram_used(),
    }


def main() -> int:
    model_dir, sft_path, bench_root = sys.argv[1], sys.argv[2], sys.argv[3]
    from pathlib import Path

    from transformers import AutoTokenizer

    from serve import suite

    capture = LogCapture()
    logging.getLogger("vllm").addHandler(capture)
    logging.getLogger("vllm").setLevel(logging.INFO)

    import torch  # noqa: F401 - imported after the handler so CUDA init is counted

    import_started = PROCESS_START
    facts = gpu_facts()
    print(f"gpu: {json.dumps(facts)}", flush=True)

    rows = [json.loads(line) for line in Path(sft_path).read_text(encoding="utf-8").splitlines() if line.strip()]
    payload = suite.check_payload(rows, Path(bench_root))
    by_case = {row["case"]: row for row in rows}
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"suite: {len(payload)} composed cases, tokenizer from {model_dir}", flush=True)

    prompts: dict[str, list[int]] = {}
    for case_id, meta in payload.items():
        prompt = suite.check_prompt(by_case[case_id]["text"], case_id, tokenizer, by_case[case_id]["input_ids"])
        prompts[case_id] = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    longest = max(len(ids) for ids in prompts.values())
    print(f"suite: prompts {min(len(ids) for ids in prompts.values())}-{longest} tokens", flush=True)

    engine = None
    attempts = []
    for attempt in ATTEMPTS:
        print(f"load: attempting {attempt}", flush=True)
        engine, record = load_engine(model_dir, attempt)
        attempts.append(record)
        print(f"load: {json.dumps({k: v for k, v in record.items() if k != 'attempt'})}", flush=True)
        if engine is not None:
            break
    if engine is None:
        print("load: every attempt failed; the artifact does not fit this GPU as configured", flush=True)
        print("PROBE_JSON " + json.dumps({"gpu": facts, "attempts": attempts, "loaded": False, "suite": None}), flush=True)
        return 2

    cold_start = time.time() - import_started
    cache_lines = capture.matching("KV cache size") or capture.matching("kv cache")
    print(f"cold_start: fresh process, engine ready in {cold_start:.1f}s", flush=True)
    for line in cache_lines:
        print(f"kv: {line.strip()}", flush=True)

    from vllm import SamplingParams

    params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, top_p=1.0)
    probe_case = "N-105" if "N-105" in prompts else sorted(prompts)[0]
    started = time.time()
    output = engine.generate([{"prompt_token_ids": prompts[probe_case]}], params, use_tqdm=False)
    elapsed = time.time() - started
    result = output[0]
    produced = len(result.outputs[0].token_ids)
    metrics = getattr(result, "metrics", None)
    timing = {
        "case": probe_case,
        "prompt_tokens": len(prompts[probe_case]),
        "output_tokens": produced,
        "seconds": round(elapsed, 2),
        "tokens_per_second": round(produced / elapsed, 2) if elapsed else None,
        "total_tokens_per_second": round((len(prompts[probe_case]) + produced) / elapsed, 2) if elapsed else None,
    }
    if metrics is not None:
        for field in ("first_token_time", "finished_time", "scheduled_time"):
            if hasattr(metrics, field):
                timing[field] = getattr(metrics, field)
        first = getattr(metrics, "first_token_time", None) or getattr(metrics, "first_token_ts", None)
        finished = getattr(metrics, "finished_time", None) or getattr(metrics, "finished_ts", None)
        if first and finished and finished > first and produced > 1:
            timing["decode_tokens_per_second"] = round((produced - 1) / (finished - first), 2)
    print(f"speed: {json.dumps(timing)}", flush=True)
    print(f"vram: {json.dumps(vram_used())}", flush=True)

    started = time.time()
    suite_params = SamplingParams(max_tokens=MAX_NEW_TOKENS, temperature=0.0, top_p=1.0)
    outputs = engine.generate([{"prompt_token_ids": prompts[case_id]} for case_id in sorted(prompts)], suite_params, use_tqdm=False)
    suite_seconds = time.time() - started
    table = []
    for case_id, result in zip(sorted(prompts), outputs):
        text = result.outputs[0].text
        meta = payload[case_id]
        plan, why = suite.parse_plan(text)
        matched, verdict = suite.composed_verdict(meta["case"], plan)
        target = suite.training_target(by_case[case_id]["text"], case_id)
        table.append(
            {
                "case": case_id,
                "parse": why,
                "parsed": plan is not None,
                "matched": matched,
                "verdict": verdict,
                "plan": plan,
                "equals_target": plan == target,
                "output_tokens": len(result.outputs[0].token_ids),
            }
        )
    print("\ncase    parse  match  equals_target", flush=True)
    for row in table:
        print(
            f"{row['case']:<7} {'ok' if row['parsed'] else 'FAIL':<6} {str(row['matched']):<6} {row['equals_target']}",
            flush=True,
        )
    matched = sum(1 for row in table if row["matched"])
    parsed = sum(1 for row in table if row["parsed"])
    print(f"suite: {parsed}/{len(table)} parse ok, {matched}/{len(table)} match required_calls", flush=True)
    for row in table:
        if not row["matched"]:
            print(f"  {row['case']} parse: {row['parse']}; verdict: {row['verdict']}", flush=True)
    # A case that does not match is a failed run, not a note in the log: the
    # parent turns a non-zero exit into a raised error.
    if matched != len(table) or parsed != len(table):
        print(
            f"suite: FAILED, {len(table) - matched} of {len(table)} cases do not match required_calls",
            flush=True,
        )
        failure = True
    else:
        failure = False

    print(
        "PROBE_JSON "
        + json.dumps(
            {
                "gpu": facts,
                "attempts": attempts,
                "loaded": True,
                "cold_start_seconds": round(cold_start, 1),
                "engine_seconds": attempts[-1]["seconds"],
                "kv_lines": [line.strip() for line in cache_lines],
                "timing": timing,
                "vram": vram_used(),
                "suite": {
                    "total": len(table),
                    "parsed": parsed,
                    "matched": matched,
                    "seconds": round(suite_seconds, 1),
                    "table": table,
                },
                "process_seconds": round(time.time() - PROCESS_START, 1),
            }
        ),
        flush=True,
    )
    return 3 if failure else 0


if __name__ == "__main__":
    raise SystemExit(main())
