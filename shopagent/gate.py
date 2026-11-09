"""The gate: Phase A checks database state, Phase B asks a judge.

Phase A is deterministic and evidence-based. It never reads the agent's audit
log as proof of an action: it diffs the database before and after the run, then
binds every completion claim in the reply to a concrete state change -- the
entity named, and the amount, tracking number or address value stated.

The gate also owns the commit. ``send_reply`` only validates and prepares a
draft; nothing reaches the customer-visible ``replies`` table until Phase A (and
Phase B, when enabled) approves. A rejection writes no reply. An unverifiable
claim escalates straight to ``human_queue`` with its draft.

Phase B is a stubbed interface today (see `judge.py`). The retry loop is real:
a rejection feeds the reason back, up to three attempts, then the ticket goes to
a human queue instead of looping.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field

from .claims import ClaimResult, check_tokens, evaluate_claims, parse_claims
from .compose import ComposeError, compose_reply
from .db import insert, now_iso, one, rows
from .judge import JudgeUnavailable, JudgeVerdict, parse_judge_output, stub_judge
from .referee import diff, snapshot

MAX_ATTEMPTS = 3

JUDGE_CODES = frozenset({"JUDGE_UNAVAILABLE", "JUDGE_INVALID", "JUDGE_REJECTED"})

# Codes that send the draft to a human instead of failing it outright. Anything
# not listed here is a rejection: the reply is contradicted, and a retry with a
# corrected answer is possible.
ESCALATE_CODES = frozenset({"UNVERIFIED_CLAIM", "CLAIM_ENTITY_UNBOUND", "JUDGE_INVALID"})


@dataclass
class Decision:
    verdict: str  # APPROVED | REJECTED | ESCALATED
    codes: list[str] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)
    phase: str = "A"

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


def _run_phase_b(
    judge, *, ticket: str, calls: list[dict], reply: str, changes: dict
) -> tuple[list[str], list[str]]:
    """Returns (codes, reasons). Fails closed on invalid or missing judge output."""
    state = {
        table: {"added": change.get("added", []), "changed": change.get("changed", [])}
        for table, change in changes.items()
    }
    prompt = {
        "ticket": ticket,
        "calls": calls,
        "reply": reply,
        "state": state,
    }
    try:
        output = judge(**prompt)
        verdict = output if isinstance(output, JudgeVerdict) else parse_judge_output(str(output))
    except JudgeUnavailable as exc:
        return ["JUDGE_UNAVAILABLE"], [f"JUDGE_UNAVAILABLE: {exc}"]
    except (ValueError, TypeError) as exc:
        return ["JUDGE_INVALID"], [f"JUDGE_INVALID: {exc}"]
    # Only the two exact enum values are a decision. "MAYBE", "" or any other
    # typed verdict is not an approval: it escalates.
    if verdict.verdict == "REJECTED":
        return ["JUDGE_REJECTED"], [f"JUDGE_REJECTED: {verdict.reason}"]
    if verdict.verdict != "APPROVED":
        return (
            ["JUDGE_INVALID"],
            [f"JUDGE_INVALID: verdict {verdict.verdict!r} is not APPROVED or REJECTED"],
        )
    return [], []


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

    phase = "A"
    if phase_b and phase_b != "off":
        phase = "A+B"
        judge_codes, judge_reasons = _run_phase_b(
            judge or stub_judge,
            ticket=ticket_text or ticket_id,
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

    decision = Decision(verdict, codes, reasons, phase)
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
            commit=True,
        )
        attempts.append(
            {
                "attempt": attempt,
                "verdict": decision.verdict,
                "codes": decision.codes,
                "reasons": decision.reasons,
                "reply": reply,
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
