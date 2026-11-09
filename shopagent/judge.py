"""Phase B judge interface.

Phase B is a separate model call (Qwen2.5-3B-Instruct behind llama.cpp). It is
stubbed: the interface, the exact JSON contract and the validator exist now so
the gate's retry loop can be wired and tested. No network call is made.

Contract (fixed for this project):

    {"verdict": "APPROVED" | "REJECTED", "reason": "..."}

Any other shape is invalid. An invalid or unavailable judge fails closed: the
gate treats it as a rejection with code `JUDGE_INVALID` / `JUDGE_UNAVAILABLE`.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

JUDGE_CONTRACT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"enum": ["APPROVED", "REJECTED"]},
        "reason": {"type": "string", "minLength": 1},
    },
}

SYSTEM_PROMPT = """You are a support reply auditor for a small online shop.
You receive: the customer ticket, the tool calls that ran, the database state
after the run, and the reply that was sent.

Approve only if every factual claim in the reply is supported by the state.
Reject if the reply claims an action that did not happen, states an amount or
status that disagrees with the state, or promises a refund or shipment that is
not confirmed.

Answer with one JSON object and nothing else:
{"verdict": "APPROVED" | "REJECTED", "reason": "one sentence"}

Rules:
- A failed or pending refund is not a refund.
- "Already refunded" is true only if the state shows a succeeded refund.
- Be strict about amounts. $80.00 is not $88.00.
"""


class JudgeUnavailable(RuntimeError):
    """Raised when Phase B is not configured or cannot be reached."""


@dataclass
class JudgeVerdict:
    verdict: str
    reason: str


def build_judge_prompt(*, ticket: str, calls: list[dict], reply: str, state: dict) -> str:
    """Render the evidence block the judge sees. Keep it small and factual."""
    call_lines = [
        f"- {call['tool']}({json.dumps(call.get('args', {}), sort_keys=True)}) -> "
        f"ok={call.get('ok')} code={call.get('code')}"
        for call in calls
    ]
    state_lines = [
        f"- {table}: {json.dumps(value, sort_keys=True)}" for table, value in state.items()
    ]
    return (
        f"TICKET:\n{ticket}\n\n"
        f"TOOL CALLS:\n{chr(10).join(call_lines) or '- none'}\n\n"
        f"STATE AFTER RUN:\n{chr(10).join(state_lines)}\n\n"
        f"REPLY SENT:\n{reply}\n"
    )


def parse_judge_output(text: str) -> JudgeVerdict:
    """Parse and validate the judge's output. Raises ValueError when invalid."""
    cleaned = text.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", cleaned, re.S)
    if fenced:
        cleaned = fenced.group(1).strip()
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge output must be a JSON object")
    if set(payload) != {"verdict", "reason"}:
        raise ValueError(f"judge output keys must be verdict and reason, got {sorted(payload)}")
    if payload["verdict"] not in ("APPROVED", "REJECTED"):
        raise ValueError(f"invalid verdict {payload['verdict']!r}")
    if not isinstance(payload["reason"], str) or not payload["reason"].strip():
        raise ValueError("reason must be a non-empty string")
    return JudgeVerdict(verdict=payload["verdict"], reason=payload["reason"].strip())


def stub_judge(**_: object) -> JudgeVerdict:
    """Phase B placeholder. Always unavailable until a local judge is wired."""
    raise JudgeUnavailable(
        "Phase B judge is not configured. Set a llama.cpp endpoint and a 3B judge "
        "model, then replace this stub."
    )
