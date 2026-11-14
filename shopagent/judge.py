"""Phase B judge: the locked DeepSeek auditor behind the gate.

The judge is `opencode-go/deepseek-v4.1-flash` over the OpenCode Go
subscription API (SPEC.md §2). The wire is not re-implemented here: a
`JudgeClient` composes a `teacher.TeacherClient`, so the gateway, the
`OPENCODE_API_KEY` / `OPENCODE_GO_API_KEY` rule, the required
`x-opencode-session` header, the 429/5xx backoff and the one-retry 4xx rule are
the same code the teacher uses.

Contract (fixed for this project):

    {"verdict": "APPROVED" | "REJECTED" | "ESCALATED", "reason": "..."}

One JSON object, nothing else: no prose, no fence, no extra key, no repeated
key. The prompt carries the execution log and the draft reply, and nothing that
hints at the expected verdict. Any off-contract answer, and any transport
failure after retries, fails closed to `ESCALATED` -- never to `APPROVED`.

Phase B widens prose coverage. It does not make a guarantee: composed mode in
`compose.py` remains the only mode with a provable no-false-claim property.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Callable

from .db import now_iso
from .teacher import (
    GATEWAY,
    MAX_ATTEMPTS,
    MODEL,
    TeacherClient,
    TeacherParseError,
    TeacherUnavailable,
)

VERDICTS = ("APPROVED", "REJECTED", "ESCALATED")
REASON_MAX_CHARS = 300

JUDGE_CONTRACT = {
    "type": "object",
    "additionalProperties": False,
    "required": ["verdict", "reason"],
    "properties": {
        "verdict": {"enum": list(VERDICTS)},
        "reason": {"type": "string", "minLength": 1, "maxLength": REASON_MAX_CHARS},
    },
}

SYSTEM_PROMPT = """You are an auditor for customer support replies in a small
online shop. You see the execution log of the tools that ran and the draft reply
written for the customer. You do not see the customer's ticket.

Every factual claim in the reply must be supported by the log: every action it
claims must appear in a tool result, and every amount, status, order id,
tracking number or address must match one.

Answer with one JSON object and nothing else:
{"verdict": "APPROVED" | "REJECTED" | "ESCALATED", "reason": "one sentence"}

Verdicts:
- APPROVED: every factual claim in the draft is supported by the log.
- REJECTED: the draft claims something the log does not show, or contradicts it.
- ESCALATED: the log and the draft do not settle the question. A human decides.

Rules:
- A failed or pending refund is not a refund.
- "Already refunded" is true only if a refund result shows succeeded.
- Be strict about amounts. $80.00 is not $88.00.
- Never approve because the reply sounds plausible. Approve only on the log.
- Keep the reason to one sentence, at most 300 characters.
"""


class JudgeUnavailable(RuntimeError):
    """The judge is not configured, cannot be reached, or refused the call."""

    def __init__(self, message: str, record: dict | None = None) -> None:
        super().__init__(message)
        self.record = record


class JudgeInvalid(ValueError):
    """The judge was reachable but its answer is off contract.

    Subclasses ValueError so the gate's fail-closed handler covers it.
    """

    def __init__(self, message: str, record: dict | None = None) -> None:
        super().__init__(message)
        self.record = record


@dataclass(frozen=True)
class JudgeVerdict:
    """One judge answer plus the provenance of the call that produced it."""

    verdict: str
    reason: str
    model: str = ""
    gateway: str = ""
    case_id: str = ""
    checked_at: str = ""


def build_judge_prompt(*, calls: list[dict], reply: str, state: dict) -> str:
    """The evidence block: execution log plus the draft reply.

    A call's full result envelope is carried, not just `ok`/`code`: a read-only
    call such as `get_order` writes no state delta, and the judge needs its
    returned facts to check a status reply.

    The ticket is not carried, and neither is any gate code, phrase list or
    expected verdict. The brief fixes the payload at log plus draft.
    """
    call_lines: list[str] = []
    for call in calls:
        tool = call.get("tool", "?")
        args = json.dumps(call.get("args", {}), sort_keys=True, default=str)
        if "result" in call:
            outcome = json.dumps(call["result"], sort_keys=True, default=str)
        else:
            outcome = f"ok={call.get('ok')} code={call.get('code')}"
        call_lines.append(f"- {tool}({args}) -> {outcome}")

    state_lines = [
        f"- {table}: {json.dumps(value, sort_keys=True, default=str)}"
        for table, value in state.items()
    ]

    return (
        "EXECUTION LOG\n"
        f"TOOL CALLS:\n{chr(10).join(call_lines) or '- none'}\n\n"
        f"STATE AFTER RUN:\n{chr(10).join(state_lines) or '- unchanged'}\n\n"
        f"DRAFT REPLY:\n{reply}\n"
    )


def check_verdict(verdict: JudgeVerdict) -> JudgeVerdict:
    """One rule set for a parsed verdict and for a typed one.

    The gate calls this on every judge answer, so a typed `MAYBE`, a blank
    reason or an over-long reason is off contract no matter how it arrived.
    """
    if not isinstance(verdict, JudgeVerdict):
        raise ValueError("judge verdict must be a JudgeVerdict")
    if not isinstance(verdict.verdict, str) or verdict.verdict not in VERDICTS:
        raise ValueError(f"verdict {verdict.verdict!r} is not one of {list(VERDICTS)}")
    reason = verdict.reason
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("reason must be a non-empty string")
    reason = reason.strip()
    if len(reason) > REASON_MAX_CHARS:
        raise ValueError(
            f"reason must be at most {REASON_MAX_CHARS} characters, got {len(reason)}"
        )
    return JudgeVerdict(
        verdict=verdict.verdict,
        reason=reason,
        model=verdict.model,
        gateway=verdict.gateway,
        case_id=verdict.case_id,
        checked_at=verdict.checked_at,
    )


def _reject_duplicate_keys(pairs: list[tuple[str, object]]) -> dict:
    """Build one object, and raise the moment a key repeats.

    `json.loads` keeps the last value of a repeated key, so
    `{"verdict": "REJECTED", "verdict": "APPROVED"}` would read as APPROVED.
    Runs for every object the parser reads, not only the outer one.
    """
    seen: dict[str, object] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"judge output repeats the key {key!r}")
        seen[key] = value
    return seen


def parse_judge_output(text: str) -> JudgeVerdict:
    """Parse and validate the judge's text. Raises ValueError when off contract.

    Strict: `json.loads` on the stripped text only. No fence, no prose, no
    prefix, no extra key, no repeated key.
    """
    cleaned = text.strip()
    if not cleaned:
        raise ValueError("judge output is empty")
    try:
        payload = json.loads(cleaned, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise ValueError(f"judge output is not JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("judge output must be one JSON object")
    if set(payload) != {"verdict", "reason"}:
        raise ValueError(
            f"judge output keys must be verdict and reason, got {sorted(payload)}"
        )
    return check_verdict(JudgeVerdict(verdict=payload["verdict"], reason=payload["reason"]))


class JudgeClient:
    """One judge call per evaluation. Wire, retries and key rules are the teacher's.

    The gate owns the retry loop and the three-attempt cap; this client owns one
    call, its transport retries within that call, and the provenance record.
    """

    def __init__(
        self,
        *,
        session_id: str,
        model: str = MODEL,
        base_url: str = GATEWAY,
        api_key: str | None = None,
        transport=None,
        sleep=None,
        timeout: float = 60.0,
        max_attempts: int = MAX_ATTEMPTS,
        record: Callable[[dict], None] | None = None,
    ) -> None:
        try:
            self._teacher = TeacherClient(
                session_id=session_id,
                model=model,
                base_url=base_url,
                api_key=api_key,
                transport=transport,
                sleep=sleep,
                timeout=timeout,
                max_attempts=max_attempts,
                max_concurrency=1,
            )
        except TeacherUnavailable as exc:
            raise JudgeUnavailable(f"judge is not configured: {exc}") from exc
        self.record = record
        self.model = self._teacher.model
        self.base_url = self._teacher.base_url

    def __repr__(self) -> str:
        return (
            f"JudgeClient(model={self.model!r}, base_url={self.base_url!r}, "
            f"session_id={self._teacher.session_id!r}, key=<redacted>)"
        )

    @property
    def requests(self) -> int:
        """Every HTTP attempt made, transport retries included."""
        return self._teacher.requests

    def __call__(
        self, *, case_id: str, calls: list[dict], reply: str, state: dict
    ) -> JudgeVerdict:
        """One judgement. Raises JudgeUnavailable or JudgeInvalid."""
        user = build_judge_prompt(calls=calls, reply=reply, state=state)
        try:
            completion = self._teacher.chat(SYSTEM_PROMPT, user)
        except TeacherUnavailable as exc:
            record = self._failure(case_id, "JUDGE_UNAVAILABLE", str(exc))
            raise JudgeUnavailable(str(exc), record=record) from exc
        except TeacherParseError as exc:
            record = self._failure(case_id, "JUDGE_INVALID", str(exc))
            raise JudgeInvalid(str(exc), record=record) from exc

        try:
            verdict = parse_judge_output(completion.text)
        except ValueError as exc:
            record = self._failure(case_id, "JUDGE_INVALID", str(exc))
            raise JudgeInvalid(str(exc), record=record) from exc

        verdict = JudgeVerdict(
            verdict=verdict.verdict,
            reason=verdict.reason,
            model=completion.model,
            gateway=self.base_url,
            case_id=case_id,
            checked_at=now_iso(),
        )
        self._emit(verdict)
        return verdict

    # -- provenance ------------------------------------------------------

    def _failure(self, case_id: str, code: str, detail: str) -> dict:
        """A failed call still leaves one record, and never one that approves."""
        return self._emit_record(
            case_id=case_id,
            verdict="ESCALATED",
            reason=f"{code}: {detail}",
            checked_at=now_iso(),
        )

    def _emit(self, verdict: JudgeVerdict) -> None:
        self._emit_record(
            case_id=verdict.case_id,
            verdict=verdict.verdict,
            reason=verdict.reason,
            checked_at=verdict.checked_at,
        )

    def _emit_record(
        self, *, case_id: str, verdict: str, reason: str, checked_at: str
    ) -> dict:
        record = {
            "model": self.model,
            "gateway": self.base_url,
            "case_id": case_id,
            "checked_at": checked_at,
            "verdict": verdict,
            "reason": reason,
        }
        if self.record is not None:
            self.record(record)
        return record
