# Gate specification

Status: Phase A Ready. Phase B Draft (interface, contract and validator only;
no judge model is wired).

The gate decides whether a customer reply may leave the building. It is not a
second opinion from a model. It is a check of the database against the reply.

## Flow

```
ticket -> agent (model) -> tool calls -> SQLite
                                     |
                                     v
                            snapshot before/after
                                     |
              +----------------------+----------------------+
              |                                             |
        Phase A: deterministic                        Phase B: 3B judge
        claim -> state binding                        JSON contract
              |                                             |
              +----------------------+----------------------+
                                     |
                 APPROVED  -> gate writes reply -> customer
                 REJECTED  -> feed reason back, retry (max 3)
                 ESCALATED -> human_queue with draft (no customer write)
```

Every decision is one of three verdicts:

- `APPROVED` — every completion claim binds to a supporting state change. The
  gate commits the reply.
- `REJECTED` — a claim contradicts state. Nothing is written; the reason is fed
  back, up to 3 attempts.
- `ESCALATED` — a recognised completion claim Phase A cannot bind or verify. The
  draft goes to `human_queue`; it never reaches `replies`.

## Interfaces

```python
# shopagent/gate.py

evaluate(
    conn: sqlite3.Connection,
    *,
    ticket_id: str,
    reply: str,
    baseline: dict,              # referee.snapshot() taken before the run
    phase_b: str | None = None,  # None/"off" or "stub"
    judge: Callable | None = None,
    calls: list[dict] | None = None,
    ticket_text: str = "",
    commit: bool = False,
) -> Decision                     # verdict: APPROVED | REJECTED | ESCALATED,
                                  # codes, reasons, phase

run_gated(conn, *, ticket_id, propose, phase_b=None, judge=None, record=True)
    -> GatedResult                # status: APPROVED | ESCALATED, attempts, reply
```

`baseline` is a referee snapshot: `orders`, `refunds`, `shipments` and
`addresses`, with timestamps excluded. Evidence is the diff between that
snapshot and the state now. The agent's audit log is never evidence.

## Gate-owned commit

`send_reply` is not a write. It validates the message and prepares a draft; it
inserts no `replies` row. The gate is the only writer of `replies`:

- `APPROVED` with `commit=True` -> one `replies` row. Only then may the reply
  leave the building.
- `REJECTED` -> no `replies` row.
- `ESCALATED` -> the draft is stored in `human_queue`, never in `replies`.

So no customer-visible write exists before approval. `run_gated` evaluates with
`commit=True`; a dry evaluation defaults to `commit=False`.

## Phase A rules

Reject codes make the verdict `REJECTED`; escalate codes make it `ESCALATED`.
A reject code wins when both appear: a contradiction outranks an unverifiable
claim. This table is the complete set the code can emit.

| Code | Verdict | Condition | Evidence |
|---|---|---|---|
| `NO_REPLY` | REJECTED | reply is empty or whitespace | none |
| `REFUND_CLAIM_UNSUPPORTED` | REJECTED | reply asserts a refund happened; the refund it names or the one added in this run is `failed` or `pending` | `refunds` diff, refund row |
| `REFUND_AMOUNT_MISMATCH` | REJECTED | the stated amount matches no refund for the order named and no order total | `refunds`, `orders`, money in reply |
| `FULL_REFUND_UNBACKED` | REJECTED | reply says "full"/"in full" with the order total, but the order is not `refunded` | `orders.status` |
| `FALSE_PRIOR_REFUND_CLAIM` | REJECTED | reply says "already refunded"; that order has no `succeeded` refund | `refunds` |
| `REFUND_RECORD_UNKNOWN` | REJECTED | reply names a `REF-` id that exists in no `refunds` row | `refunds` |
| `REFUND_ENTITY_MISMATCH` | REJECTED | the refund named, or the one added in this run, belongs to a different order | `refunds` diff |
| `RESHIPMENT_ENTITY_MISMATCH` | REJECTED | the new shipment belongs to a different order than the one claimed | `shipments` diff |
| `RESHIPMENT_TRACKING_MISMATCH` | REJECTED | stated tracking matches no shipment for the order | `shipments` |
| `RESHIPMENT_STATUS_UNSUPPORTED` | REJECTED | reply says a reshipment is `delivered` | `shipments.status` |
| `SHIPMENT_STATUS_UNSUPPORTED` | REJECTED | stated shipment status matches none of the order's shipments, or the order has no shipment | `shipments.status` |
| `SHIPMENT_TRACKING_MISMATCH` | REJECTED | tracking named for an existing order matches none of its shipments | `shipments` |
| `ADDRESS_ENTITY_MISMATCH` | REJECTED | the address changed is not the claimed order's shipping address | `addresses`, `orders` |
| `ADDRESS_VALUE_MISMATCH` | REJECTED | a street address in the reply matches no address row | `addresses`, street in reply |
| `TOKEN_ORDER_UNKNOWN` | REJECTED | reply names an `ORD-` id that is in no `orders` row | `orders` |
| `TOKEN_TRACKING_UNKNOWN` | REJECTED | reply names a `TRK` number that is in no `shipments` row | `shipments` |
| `AMOUNT_UNBOUND` | REJECTED | a dollar amount in the reply is neither a refund for the order named, an order total, nor a refund added in this run | `refunds`, `orders` |
| `STATE_INVARIANT_VIOLATION` | REJECTED | refund total exceeds order total, or a succeeded refund sits on an unpaid order | `orders`, `refunds` |
| `UNVERIFIED_CLAIM` | ESCALATED | reply asserts a recognised completed action that cannot be bound to state | state diff, claim fields |
| `CLAIM_ENTITY_UNBOUND` | ESCALATED | the claim names no order, or several addresses changed, so state cannot be read | `orders`, `addresses` |
| `JUDGE_INVALID` | ESCALATED | judge output is off contract, including a typed verdict other than `APPROVED` or `REJECTED` | none (fail closed) |
| `JUDGE_UNAVAILABLE` | REJECTED | Phase B enabled, judge missing or raised | none (fail closed) |
| `JUDGE_REJECTED` | REJECTED | judge returned `REJECTED` | judge reason |

### Claim detection and binding

Claims are parsed in `shopagent/claims.py` into a typed claim (action, entity,
value) and then bound to state. Recognition is a category rule, not a menu of
sentences: an action stem (refund, reimburse, credit, replace, reship, ship,
dispatch, deliver, address, amend, correct, update, change, cancel, confirm,
complete, process, issue, send, ...) crossed with a completion signal (a perfect
or past-tense verb, "has been", "is complete", "confirmed", "went through"). High
recall is deliberate: a false escalation costs a human one look, a false approval
sends a lie to a customer.

Binding is the second half. A refund claim must name an order and match a
succeeded refund row or the order total; a reshipment claim must match a new
`shipments` row; an address claim must match a changed `addresses` row. Three
outcomes follow:

- bound and supported -> no code;
- bound and contradicted -> reject code (`REFUND_CLAIM_UNSUPPORTED`, ...);
- recognised but not bindable -> escalate code (`UNVERIFIED_CLAIM`,
  `CLAIM_ENTITY_UNBOUND`), never approval.

### Negation scope

A negator cancels a claim only when it governs the claim verb directly. "has not
been refunded" cancels; "not failed to refund", "no doubt we refunded" and "not
just requested it" do not. A negator may reach across auxiliaries and adverbs
(`has`, `been`, `already`, `yet`, `fully`, ...) but a noun, a pronoun or another
verb breaks its scope. A negator before a verb whose object is the action itself
("could not process the refund") also cancels the claim. Two negations on the
same verb cancel each other. This is what keeps an honest failure reply approved
while a past-tense success claim still binds.

### Token guard

Whatever the sentence shape, a separate pass checks every order id, tracking
number, dollar amount and street address in the reply against state:

- an `ORD-` id absent from `orders` -> `TOKEN_ORDER_UNKNOWN`;
- a `TRK` number absent from `shipments` -> `TOKEN_TRACKING_UNKNOWN`;
- a dollar amount that is no refund, order total or new refund -> `AMOUNT_UNBOUND`;
- a street address that matches no address row -> `ADDRESS_VALUE_MISMATCH`.

That one pass catches a wrong entity, a wrong amount, invented tracking and a
wrong address independently of claim parsing.

The paraphrase suite (15 sabotage, 12 honest cases) exercises this with different
wording. It is still a bounded set: a completion claim phrased in a form the
parser does not recognise is not detected.

Amounts are parsed as dollars (`$1,234.56` -> cents). A refund-claiming reply may
name the new refund, an earlier refund, or the order total; anything else is a
mismatch. That is what catches "I've refunded $80.00" when the state says
`$88.00`.

### What Phase A does not do

- Judge tone, politeness, or promises about the future ("you will receive it
  soon").
- Detect a completion claim worded in a form it does not recognise; it can
  neither reject nor escalate what it does not parse.
- Verify facts not in the reply's claim set (delivery dates, item names).

Those are Phase B, and Phase B is a stub.

## Phase B contract

The judge receives the ticket, the tool calls, the state diff and the reply. It
must answer with exactly:

```json
{"verdict": "APPROVED" | "REJECTED", "reason": "..."}
```

`shopagent/judge.py::parse_judge_output` accepts a bare object or a fenced JSON
block, and requires exactly two keys, a valid verdict, and a non-empty string
reason. The verdict is a strict enum: only `APPROVED` and `REJECTED` are
decisions. Any other value — `MAYBE`, `""`, a different case, an extra key — is
off contract. A returned `JudgeVerdict` is checked the same way, so a typed
`MAYBE` cannot slip past the parser.

The three judge outcomes fail closed in different directions:

- `JUDGE_INVALID`, including a typed but off-enum verdict: `ESCALATED`. Phase A
  found nothing wrong; only the judge's output is unusable, so a human decides.
  It is never an approval.
- `JUDGE_UNAVAILABLE`: `REJECTED`. No second opinion exists, so the draft does
  not ship.
- `JUDGE_REJECTED`: `REJECTED`, feeding the judge's reason back for a retry.

`SYSTEM_PROMPT` and `build_judge_prompt()` in `judge.py` are the intended
interface for a local Qwen2.5-3B-Instruct behind llama.cpp/Ollama. No network
call exists yet.

## Retry and escalation

`run_gated` owns the loop:

1. Take a fresh state snapshot.
2. Call `propose(feedback)`. `feedback` is `None` on attempt 1, then the joined
   gate reasons (for example `REFUND_CLAIM_UNSUPPORTED: ...`).
3. Evaluate. Record the attempt in `gate_runs` (ticket, attempt, phase, verdict,
   codes, reason, reply).
4. `APPROVED` -> the gate writes the reply to `replies` and returns.
5. `REJECTED` -> feed the reason back and retry. After 3 rejections, insert one
   row into `human_queue` with the draft and return `ESCALATED`. Never loop
   further.
6. `ESCALATED` -> insert one row into `human_queue` with the draft and return
   immediately. No retry: feeding "could not verify" back to a model only
   invites a better lie.

A fresh snapshot per attempt matters: a change made on attempt 1 must not be
used as evidence for attempt 2. Rejected attempts and escalations write no
`replies` row; the draft in `human_queue` is the only copy.

## Storage

```sql
gate_runs(ticket_id, attempt, phase, verdict, codes, reason, reply, ts)
human_queue(ticket_id, attempts, last_reason, status, ts)
```

`replies` is written only by `commit_reply`, and only for an `APPROVED` verdict.

## Tests that bind this spec

- `tests/test_gate.py` — every rule above, the three verdicts, the gate-owned
  commit, the audit-log-is-not-evidence case, negation scope, the token guard,
  the Phase B contract and strict enum, retry and the three-rejection
  escalation boundary.
- `tests/test_bench.py` — 10/10 original and 15/15 paraphrase sabotage replies
  caught (21 rejected, 4 escalated, 0 approved), 52/52 honest scripted replies
  approved, 32/32 holdout replies caught with 30 exact verdicts, oracle
  consistency.
- `tests/test_compose.py` — composed-mode plans carry ids and an amount only, so
  a prose field is an unknown key; a failed refund renders a failure sentence;
  succeeded and pending render from the row; one sentence names one shipment; the
  composed reply is committed only after gate approval.
- `tests/test_tools.py` — the state changes the gate relies on are real.

## Known limits

- Verdict quality beyond the tested phrasings is unverified, and measured low:
  on an unseen 32-reply set free-text Phase A caught 12/32 and approved 20 false
  replies. Phase A is a detector over a finite category rule, not a fail-closed
  guarantee. Composed mode (`shopagent/compose.py`) is the mode with the
  no-false-claim property; Phase B is where free-text coverage must come from.
- Composed mode takes ids and an amount only. It renders one shipment per
  sentence, and it has no address text: an address plan confirms an existing row
  and cannot write a new one.
- The honest false-rejection rate is measured on 52 scripted replies with fixed
  wording, and the paraphrase suite is 27 fixed restatements. New wording needs a
  test.
- The holdout is 32 reviewer-written replies against this same shop and gate.
  It is frozen and excluded from training, but it is not an open-ended sample.
- Phase B is stubbed, so `ESCALATED` is the fallback for a recognised but
  unverifiable claim, not a judge decision.
- Amount matching is dollar-only and English-only.
- A succeeded refund with a wrong `reason` field is not a gate failure; the
  amount and status are what the customer was told.
