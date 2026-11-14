"""The gate: Phase A checks database state, Phase B asks a judge.

Phase A is deterministic and evidence-based. It never reads the agent's audit
log as proof of an action: it diffs the database before and after the run, then
binds every completion claim in the reply to a concrete state change -- the
entity named, and the amount, tracking number or address value stated.

The gate also owns the commit. ``send_reply`` only validates and prepares a
draft; nothing reaches the customer-visible ``replies`` table until Phase A (and
Phase B, when enabled) approves. A rejection writes no reply. An unverifiable
claim escalates straight to ``human_queue`` with its draft.

Phase B is the DeepSeek judge (see `judge.py`); it runs only after Phase A has
found nothing, so Phase A keeps its own semantics. A rejection feeds the judge's
reason back, up to three attempts, then the ticket goes to a human queue instead
of looping. Every off-contract judge answer, and a transport failure after its
retries, escalates.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from .claims import ClaimResult, check_tokens, evaluate_claims, parse_claims
from .compose import ComposeError, compose_reply
from .db import audit_after, insert, last_audit_seq, now_iso, one, rows
from .judge import JudgeUnavailable, JudgeVerdict, check_verdict, parse_judge_output
from .referee import diff, snapshot

MAX_ATTEMPTS = 3

JUDGE_CODES = frozenset(
    {"JUDGE_UNAVAILABLE", "JUDGE_INVALID", "JUDGE_REJECTED", "JUDGE_ESCALATED"}
)

# Codes that send the draft to a human instead of failing it outright. Anything
# not listed here is a rejection: the reply is contradicted, and a retry with a
# corrected answer is possible. Every Phase B non-answer and the judge's own
# ESCALATED go to a human: no second opinion exists to retry against.
ESCALATE_CODES = frozenset(
    {
        "UNVERIFIED_CLAIM",
        "CLAIM_ENTITY_UNBOUND",
        "JUDGE_INVALID",
        "JUDGE_UNAVAILABLE",
        "JUDGE_ESCALATED",
    }
)


@dataclass
class Decision:
    verdict: str  # APPROVED | REJECTED | ESCALATED
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    phase: str = "A"
    # One record per judge call that Phase B made: model, gateway, case id,
    # checked_at, verdict, reason. A failed call leaves one too.
    provenance: list[dict] = field(default_factory=list)

    @property
    def approved(self) -> bool:
        return self.verdict == "APPROVED"

    @property
    def escalated(self) -> bool:
        return self.verdict == "ESCALATED"

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "codes": self.codes,
            "reasons": self.reasons,
            "phase": self.phase,
            "provenance": self.provenance,
        }


def _check_invariants(conn: sqlite3.Connection) -> list[str]:
    problems: list[str] = []
    over = rows(
        conn,
        "SELECT r.order_id, SUM(r.amount_cents) AS refunded, o.total_cents"
        " FROM refunds r JOIN orders o ON o.id = r.order_id"
        " WHERE r.status = 'succeeded' GROUP BY r.order_id"
        " HAVING refunded > o.total_cents",
    )
    for row in over:
        problems.append(
            f"STATE_INVARIANT: {row['order_id']} refunded {row['refunded']} > total {row['total_cents']}"
        )
    unpaid = rows(
        conn,
        "SELECT DISTINCT r.order_id FROM refunds r JOIN orders o ON o.id = r.order_id"
        " WHERE r.status = 'succeeded' AND o.status IN ('pending_payment','cancelled')",
    )
    for row in unpaid:
        problems.append(f"STATE_INVARIANT: succeeded refund on unpaid order {row['order_id']}")
    return problems


def _no_judge(**_: object) -> JudgeVerdict:
    """Phase B is on and nothing was injected. Fail closed, do not approve."""
    raise JudgeUnavailable("Phase B is enabled but no judge was injected")


def _run_phase_b(
    judge, *, case_id: str, calls: list[dict], reply: str, changes: dict
) -> tuple[list[str], list[str], list[dict]]:
    """Returns (codes, reasons, provenance). Fails closed on any judge failure.

    The prompt carries the execution log and the draft reply only: no ticket,
    no gate code, no phrase list, no expected verdict.
    """
    state = {
        table: {"added": change.get("added", []), "changed": change.get("changed", [])}
        for table, change in changes.items()
    }
    try:
        output = judge(case_id=case_id, calls=calls, reply=reply, state=state)
        verdict = output if isinstance(output, JudgeVerdict) else parse_judge_output(str(output))
        verdict = check_verdict(verdict)
    except JudgeUnavailable as exc:
        return ["JUDGE_UNAVAILABLE"], [f"JUDGE_UNAVAILABLE: {exc}"], _record(exc)
    except (ValueError, TypeError) as exc:
        return ["JUDGE_INVALID"], [f"JUDGE_INVALID: {exc}"], _record(exc)

    provenance = [
        {
            "model": verdict.model,
            "gateway": verdict.gateway,
            "case_id": verdict.case_id or case_id,
            "checked_at": verdict.checked_at,
            "verdict": verdict.verdict,
            "reason": verdict.reason,
        }
    ]
    if verdict.verdict == "REJECTED":
        return ["JUDGE_REJECTED"], [f"JUDGE_REJECTED: {verdict.reason}"], provenance
    if verdict.verdict == "ESCALATED":
        return ["JUDGE_ESCALATED"], [f"JUDGE_ESCALATED: {verdict.reason}"], provenance
    return [], [], provenance


def _record(exc: Exception) -> list[dict]:
    """The failed call's provenance, when the client attached one."""
    record = getattr(exc, "record", None)
    return [record] if isinstance(record, dict) else []


def _calls_since(conn: sqlite3.Connection, since: int) -> list[dict]:
    """The execution log one attempt wrote: tool, args and the full result.

    Read from the audit log rows, because that is where every tool call lands.
    A read-only call leaves no state delta, so its result envelope must travel
    or the judge cannot check a status reply.
    """
    calls: list[dict] = []
    for row in audit_after(conn, since):
        try:
            args = json.loads(row["args_json"])
        except (TypeError, json.JSONDecodeError):
            args = {}
        try:
            result = json.loads(row["result_json"])
        except (TypeError, json.JSONDecodeError):
            result = {}
        calls.append({"tool": row["tool"], "args": args, "result": result})
    return calls


def commit_reply(conn: sqlite3.Connection, *, ticket_id: str, reply: str) -> int:
    """Write the customer-visible reply. Only the gate calls this."""
    cur = conn.execute(
        "INSERT INTO replies (ticket_id, message, created_at) VALUES (?, ?, ?)",
        (ticket_id, reply, now_iso()),
    )
    conn.commit()
    return int(cur.lastrowid)


def evaluate(
    conn: sqlite3.Connection,
    *,
    ticket_id: str,
    reply: str,
    baseline: dict,
    phase_b: str | None = None,
    judge=None,
    calls: list[dict] | None = None,
    ticket_text: str = "",
    commit: bool = False,
) -> Decision:
    """Phase A (always) plus Phase B when enabled. `baseline` is a referee snapshot.

    `commit=True` makes the gate persist an approved reply. Rejections write
    nothing; escalations write nothing here (the retry loop puts the draft in
    `human_queue`).
    """
    if not isinstance(reply, str) or not reply.strip():
        return Decision("REJECTED", ["NO_REPLY"], ["NO_REPLY: the agent sent no reply"])

    after = snapshot(conn)
    changes = diff(baseline, after)

    claim_result: ClaimResult = evaluate_claims(conn, parse_claims(reply), changes)
    token_result: ClaimResult = check_tokens(conn, reply, changes, ticket_text=ticket_text)
    codes: list[str] = list(token_result.rejects) + list(claim_result.rejects) + list(claim_result.escalates)
    reasons: list[str] = list(token_result.reasons) + list(claim_result.reasons)

    reasons.extend(_check_invariants(conn))
    if any(reason.startswith("STATE_INVARIANT") for reason in reasons) and "STATE_INVARIANT_VIOLATION" not in codes:
        codes.append("STATE_INVARIANT_VIOLATION")

    # Phase B runs after Phase A approves, and only then: an unverifiable Phase
    # A claim still escalates immediately, and a judge answer can never turn a
    # Phase A rejection into a retry.
    phase = "A"
    provenance: list[dict] = []
    if phase_b and phase_b != "off" and not codes:
        phase = "A+B"
        judge_codes, judge_reasons, provenance = _run_phase_b(
            judge or _no_judge,
            case_id=ticket_id,
            calls=calls or [],
            reply=reply,
            changes=changes,
        )
        codes.extend(judge_codes)
        reasons.extend(judge_reasons)

    # An escalate code sent to a human, not failed outright: any other code (a
    # contradiction) outranks it. Phase-B invalid output is one of these.
    escalate_codes = set(claim_result.escalates) | ESCALATE_CODES
    if any(code not in escalate_codes for code in codes):
        verdict = "REJECTED"
    elif codes:
        verdict = "ESCALATED"
    else:
        verdict = "APPROVED"

    decision = Decision(verdict, codes, reasons, phase, provenance)
    if decision.approved and commit:
        commit_reply(conn, ticket_id=ticket_id, reply=reply)
    return decision


# -- retry / escalation ---------------------------------------------------


@dataclass
class GatedResult:
    status: str  # APPROVED | ESCALATED
    attempts: list[dict]
    reply: str


def record_decision(
    conn: sqlite3.Connection, *, ticket_id: str, attempt: int, decision: Decision, reply: str
) -> None:
    insert(
        conn,
        "gate_runs",
        ts=now_iso(),
        ticket_id=ticket_id,
        attempt=attempt,
        phase=decision.phase,
        verdict=decision.verdict,
        codes=json.dumps(decision.codes),
        reason="; ".join(decision.reasons),
        reply=reply,
    )
    conn.commit()


def _enqueue(conn: sqlite3.Connection, *, ticket_id: str, attempts: int, reason: str, draft: str) -> None:
    insert(
        conn,
        "human_queue",
        ts=now_iso(),
        ticket_id=ticket_id,
        attempts=attempts,
        last_reason=reason or "gate rejected",
        draft=draft,
        status="open",
    )
    conn.commit()


def run_gated(
    conn: sqlite3.Connection,
    *,
    ticket_id: str,
    propose,
    phase_b: str | None = None,
    judge=None,
    record: bool = True,
    composed: bool = False,
) -> GatedResult:
    """Run propose(feedback) until the gate approves, or escalate.

    A rejection is retried up to three attempts with the reason fed back. An
    escalation (a recognised completion claim Phase A cannot verify) goes to a
    human immediately: feeding "I could not verify that" back to a model only
    invites a better lie.

    With `composed=True` the model does not write prose. `propose` returns a plan
    (`{"action": ..., "order_id": ..., "amount_cents": ...}`), `compose_reply`
    runs the action and renders the sentence from the row it wrote, and the gate
    checks that derived text. A failed action can only produce a failure
    sentence.

    Each attempt starts from a fresh state snapshot: a change made by an earlier
    attempt cannot be used as evidence for a later one. The gate commits an
    approved reply; rejected attempts persist no reply. The escalated draft is
    stored in `human_queue`, never in `replies`.
    """
    attempts: list[dict] = []
    feedback: str | None = None
    reply = ""
    for attempt in range(1, MAX_ATTEMPTS + 1):
        baseline = snapshot(conn)
        since = last_audit_seq(conn)
        plan = propose(feedback)
        reply = compose_reply(conn, ticket_id=ticket_id, plan=plan) if composed else plan
        if not isinstance(reply, str) or not reply.strip():
            raise ComposeError("propose returned no reply text")
        decision = evaluate(
            conn,
            ticket_id=ticket_id,
            reply=reply,
            baseline=baseline,
            phase_b=phase_b,
            judge=judge,
            calls=_calls_since(conn, since),
            commit=True,
        )
        attempts.append(
            {
                "attempt": attempt,
                "verdict": decision.verdict,
                "codes": decision.codes,
                "reasons": decision.reasons,
                "reply": reply,
                "provenance": decision.provenance,
            }
        )
        if record:
            record_decision(conn, ticket_id=ticket_id, attempt=attempt, decision=decision, reply=reply)
        if decision.approved:
            return GatedResult("APPROVED", attempts, reply)
        if decision.escalated:
            _enqueue(
                conn,
                ticket_id=ticket_id,
                attempts=len(attempts),
                reason="; ".join(decision.reasons),
                draft=reply,
            )
            return GatedResult("ESCALATED", attempts, reply)
        feedback = "; ".join(decision.reasons)

    _enqueue(
        conn,
        ticket_id=ticket_id,
        attempts=len(attempts),
        reason=feedback or "gate rejected",
        draft=reply,
    )
    return GatedResult("ESCALATED", attempts, reply)
