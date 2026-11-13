# eval-gated-lora-agent

A gate that checks a customer reply against real database and API state, never
the agent's own log. Its free-text mode is a detector, and detection does not
generalise: on an unseen, freshly written set it caught 12 of 32 false replies
and approved 20. The structural answer is composed mode, where code renders the
sentence from the state the action actually wrote; there, a failed action cannot
produce a success sentence. Phase B, the model judge, is stubbed.

A store support agent for a small shop. It may call six tools that write real
SQLite state and an append-only audit row. Before any reply reaches a customer,
a gate checks the database, not the agent's own log: if the refund failed, a
reply that says "I've refunded your order" is rejected.

**Stage: incubator, Phase 1 (harness).** `SPEC.md` holds the locked plan: 27B
student, Modal compute, DeepSeek teacher and judge. There is no trained model in
this repo yet and no training run. A first teacher corpus exists: 178 admitted
traces of 284 frozen cases, from a full pass plus a resumed pass after a brief
change, gated, manifested and leak-checked (see below). Everything else measures
the harness against a scripted fake agent, hand-written adversarial replies
across 284 frozen cases (205 bank + 79 paraphrase), a 32-case reviewer-authored
holdout that was remediation-targeted, and a third-party 32-case unseen set that
measured 12/32 caught. No number here is model performance. Bench scores after
this corpus are **seen-scenario** scores, not generalisation; the two holdouts
are the outside check.

## The rule this project is built on

Build the benchmark first, freeze it by content hash, then generate training
data. The plan's original order — 1000 sessions, then extract test cases — leaks
test cases into training and leaves nothing honest to measure against.

## Quickstart

Python 3.11+ (3.13 tested). Standard library only, no install step.

```bash
python -m shopagent.cli demo        # tools, state diff, gate, retry, composed mode
python -m shopagent.cli bench       # oracle, scripted agent, gate catch rate
python -m unittest discover -s tests -v
```

From a fresh clone those three commands are the whole setup.

## What is in the repo

```
shopagent/
  db.py          SQLite schema, connections, audit helper
  seed.py        fixed 60-order demo dataset
  billing.py     mock Stripe-shaped refunds, idempotency, deterministic failures
  faults.py      fault injection table (refund, reshipment, address)
  tools.py       six tools, policy enforced in code, every call audited
  referee.py     state snapshot, diff, declarative state expectations
  gate.py        Phase A on state evidence, retry loop, escalation
  judge.py       Phase B contract, validator, stub
  policies.py    deterministic honest fake agent (a fixture, not a model)
  compose.py     composed replies: sentence rendered from the row the action wrote
  teacher.py     OpenCode Go client, prompts, briefs, parsers, status-id source
  corpus.py      one case -> one admitted trace or one recorded reject; run, shards, manifest
  bench.py       case loading, validation, oracle, scoring, gate check, holdout
  freeze.py      content-hash freeze of the benchmark and the holdout
  leakcheck.py   exact and near-duplicate similarity, tickets and holdout replies
  cli.py         demo / bench / trace / freeze / verify-frozen / leakcheck / teacher
bench/cases/     284 cases: 134 normal + 150 adversarial
                 (205 bank + 79 paraphrase: 24 honest, 55 sabotage)
bench/holdout/review-01.jsonl  32 reviewer-authored replies, frozen, never trained on
bench/holdout/review-02.jsonl  60 replies from a different model, scored once
bench/FROZEN.json
SPEC.md          the locked plan: stack, stages, success criteria, limits
docs/GATE.md     gate contract, Phase A rules, retry semantics
docs/BENCHMARK.md case format and benchmark discipline
docs/TEACHER-DATA-PLAN.md the cleared slice plan for the teacher corpus
data/train/      shard-NNN.jsonl: admitted traces only (gitignored, README tracked)
data/raw/        raw.jsonl: every teacher response verbatim (gitignored)
data/rejects/    rejected.jsonl: one row per reject with its reason (gitignored)
data/manifest.json counts, digests, session id, template digest (gitignored)
tests/
```

## The domain

SQLite holds `customers`, `addresses`, `orders`, `order_items`, `refunds`,
`shipments`, `replies`, an append-only `audit_log`, `fault_injection`,
`gate_runs`, `human_queue` and `billing_log`.

Tools: `get_order`, `list_orders`, `refund_order`, `create_reshipment`,
`update_address`, `send_reply`.

The mock billing API returns Stripe-shaped refund objects with `status` of
`succeeded` / `failed` / `pending`, a `failure_code`, and idempotency keys.
Failures are forced by a row in `fault_injection`, so a case fails the same way
every run:

```python
faults.inject(conn, "refund", "ORD-1001", "card_declined", times=1)
```

A `processor_timeout` returns HTTP 504 and leaves a `pending` refund: the honest
hard case where neither success nor failure is known.

## Benchmark

284 cases: 134 normal, 150 adversarial. The original suite is 40 cases (30
normal, 10 adversarial); the growth bank adds 165 cases, and a paraphrase suite
adds 79 cases (24 honest, 55 sabotage) that restate the same situations in
different wording. Twenty of the normal cases are composed. Each case
carries the ticket, any injected faults, required tool calls with expected
`ok`/`code`, expected final database state, required facts in the reply, and
forbidden phrases. Adversarial cases also carry a known-bad `sabotage.reply` —
the gate's acceptance test.

Every case is checked three ways:

1. **Oracle** — execute the case's own required calls, then compare the final
   state. Proves each marking guide is executable and self-consistent.
2. **Scripted agent** — a deterministic rule-based fixture acts through the real
   tools; score required calls, final state and reply facts.
3. **Gate check** — feed each adversarial `sabotage.reply` to the gate against
   the real failed state and require `REJECTED` or `ESCALATED`, never
   `APPROVED`; feed every honest scripted reply and require `APPROVED`.

A fourth run scores the reviewer holdout. `bench/holdout/review-01.jsonl` holds
32 rows written by a reviewer, each naming a case, a reply and the verdict the
reviewer expects. The gate sees the case's real end state. `caught` (never
approved) is the safety property; `exact` (the reviewer's verdict reproduced)
is reported separately and is stricter.

**This holdout is remediation-targeted, not a general-resistance result.** Its
rows and the cases behind them were written while the gate was being repaired,
so 32/32 proves the repaired gate handles the reviewed set. It does not prove
free-text mode resists new wording. A third review measured exactly that on a
fresh, unseen 32-reply set: free-text Phase A caught 12/32 and approved 20 false
replies. That number is the honest free-text limit. Composed mode is the answer
to it; Phase B is where free-text coverage must come from.

The benchmark is frozen before any training data exists:

```bash
python -m shopagent.cli freeze          # writes bench/FROZEN.json
python -m shopagent.cli verify-frozen
python -m shopagent.cli leakcheck       # exact + near-duplicate similarity
```

`bench` refuses to score if the frozen hashes changed. `bench/FROZEN.json` holds
two case digests: the original 40-case digest
`34833a08bbe4ff761cfdc7aad311fbe468a81061d6166bf83eee342a8c0a341f`, unchanged,
and the all-cases digest
`a76a732fa9b1f0126a76a2465f00e1b9bee6eca66e3fd22fa993598621c66364` over all 284.
It also pins the holdout file at `47034a0f…c180`, 32 rows, marked
training-excluded.

## Teacher corpus

The pipeline turns one frozen case into one trainable trace, or one recorded
reject. The teacher is `deepseek-v4.1-flash` over the OpenCode Go subscription.

```
case (title, ids, amounts, fault modes, required facts)
  -> call 1: a fresh customer message + a tool plan      (teacher never sees the bench ticket)
  -> the plan runs against the real tool layer
  -> free text: call 2 with the real tool outputs -> reply
  -> composed: compose_reply renders the sentence from the row the action wrote
  -> admit only if plan matches required_calls, state matches state_expect,
     the reply carries the required facts, no forbidden phrase is present,
     and the gate returns APPROVED
```

```bash
python -m shopagent.cli teacher --dry-run 5      # sample into out/teacher-dry
python -m shopagent.cli teacher --out data --session-id teacher-<UTC>
python -m shopagent.cli teacher --out data --resume --session-id teacher-<UTC>
python -m shopagent.cli leakcheck --corpus data/train
```

Flags: `--dry-run N`, `--out DIR`, `--limit N`, `--only IDS`, `--session-id ID`,
`--resume`. Concurrency is capped at 8. `verify-frozen` runs before the first
request. A missing key exits 2 and writes nothing.

Runs of 2026-11-13, session `teacher-20261113T011445Z`. Run 1 was the cleared
plan; run 2 re-ran run 1's rejects only, after two brief changes.

| | run 1 | run 2 (resume) | corpus |
|---|---|---|---|
| attempted | 284 | 232 | 284 |
| admitted | 52 | 126 | **178** |
| rejected | 232 | 106 | 106 |
| requests | 512 | 463 | 975 |
| wall time | 899 s | 490 s | — |
| shards | 1 of 52 rows | 3 of 178 rows | 3 shards |

Rejects now: `gate:REJECTED` 71, `gate:ESCALATED` 28, `plan_mismatch` 3,
`forbidden_phrase` 2, `invalid_plan` 1, `leak_near:N-24` 1. Before the brief
change the two largest were `facts_missing` 107 and `plan_mismatch` 68; they are
0 and 3 now. Everything rejected is in `data/rejects/rejected.jsonl` with its
reason; nothing rejected reached a shard.

Composition: 164 free text (74 normal, 90 adversarial) and 14 composed (refund
6, reshipment 4, status 4, address 0). Two rows are routed from composed to free
text by the `unsupported_composed` rule: N-98 and N-110.

Two admission rules, both approved by Daniel, both visible in the brief or the
retry: the free-text brief lists the case's required facts as values the reply
must state; and a `parse_error` or `invalid_plan` is resent once with the
required shape, while a rejected verdict, a state mismatch or a missing fact is
never resent.

The brief also names the exact calls the scenario needs, and the reply brief
repeats the required facts. That is wording only: no gate code, phrase list,
verdict or reason code reaches the teacher.

`--resume` is new surface, added outside the cleared plan — Daniel asked for one
round of brief improvement and a re-run of only the rejected cases. A resumed
pass reads the corpus on disk, regenerates only the given cases, keeps every
other row, re-cuts shards and recomputes the manifest.

**The corpus mixes two brief revisions.** 52 rows were generated under revision
1, 126 under revision 2; `data/manifest.json` records the revision per run. The
52 are not regenerated. See `EVIDENCE.md` for the full record.

**The limit this creates.** The teacher was briefed on every case's ids and
amounts, so the 284 bench scenarios are now seen. A score of a model trained on
this corpus against the same cases measures recall, not generalisation. The two
holdouts are the outside check; neither was in a prompt.

**The corpus is 178 traces, not the 1,000+ sessions `SPEC.md` §3 targets.** It is
an approved slice from the frozen bank; hand-written seeds and schema-driven
expansion are still Open. See `EVIDENCE.md` for the full record.

## Gate

Every decision is one of three verdicts:

- **APPROVED** — every recognised completion claim in the reply binds to a supporting
  state change; the gate writes the reply to `replies`.
- **REJECTED** — a claim contradicts state. Nothing is written; the reason is
  fed back, up to 3 attempts.
- **ESCALATED** — a recognised completion claim that Phase A cannot verify. The
  draft goes to `human_queue`, never to `replies`.

The gate owns the commit. `send_reply` only validates and prepares a draft, so
no customer-visible write exists before approval. After 3 rejections the ticket
is escalated with its draft, not looped further.

Phase A is deterministic pattern-and-state binding. Phase B is a model judge —
`opencode-go/deepseek-v4.1-flash` over the OpenCode Go subscription API — with
this exact contract, parsed as JSON and validated:

```json
{"verdict": "APPROVED" | "REJECTED", "reason": "..."}
```

Phase B is stubbed today and fails closed when unavailable. See `docs/GATE.md`.

## Composed mode

Free-text mode asks the gate to detect a lie in prose, and detection over
unseen wording is weak: 12/32 caught, 20 false approvals on a fresh set. That
coverage is **unsolved**, and composed mode does not try to solve it. Composed
mode removes the prose channel instead. The model returns only a plan of ids and
an amount in cents:

```json
{"action": "refund", "order_id": "ORD-1001", "amount_cents": 2400}
{"action": "reshipment", "order_id": "ORD-1002", "shipment_id": "TRK2010"}
{"action": "address", "address_id": "ADR-1006"}
```

The guarantee, stated exactly: **ids in, row out, one shipment, no text input.**

- Only `order_id`, `address_id`, `shipment_id` and an integer `amount_cents` are
  accepted. Any other key — `line1`, `city`, `postcode`, `message` — is refused
  with `ComposeError`, so no string the model wrote can reach the customer text.
- The sentence is rendered from the row the action wrote, read back from the
  database. A failed action has no success sentence to render.
- One sentence names one shipment, chosen by id. A plan naming two is refused.
- An address plan names an existing row by id and nothing else: composed mode
  **does not support address text**, so it cannot change an address, only
  confirm the stored one.

`shopagent/compose.py` runs the action through the same tools, then renders:

```
failed refund : We could not process the $24.00 refund for ORD-1001; the payment provider reported card declined.
succeeded     : We refunded $24.00 for ORD-1001.
address       : The delivery address ADR-1006 is 5 Market Square, Saida, 1601.
```

`run_gated(..., composed=True)` selects the mode;
`python -m shopagent.cli trace --case A-01 --composed` and `demo` show it. The
gate still checks the derived text, as a second check rather than the first
line of defence. Tests: `tests/test_compose.py`.

This is the only mode with a provable no-false-claim property. It constrains the
sentence, not the choice: a model that picks the wrong action still causes a
wrong action, and only Phase B can widen prose coverage.

## Numbers, with the command that produced each

Run 2026-11-11 on this machine (Windows, Python 3.13.14), with the teacher rows
added 2026-11-13.

| Claim | Command | Result |
|---|---|---|
| Demo runs end to end | `python -m shopagent.cli demo` | exit 0; writes `out/demo.sqlite3` (114,688 bytes); retry rejects attempt 1 and approves attempt 2 |
| Composed mode, failed refund | same | plan `{"action": "refund", "amount_cents": 2400, "order_id": "ORD-1001"}` → `We could not process the $24.00 refund for ORD-1001; the payment provider reported card declined.` → `APPROVED (codes: none)` |
| Composed mode, real refund | same | `We refunded $24.00 for ORD-1001.` → `APPROVED (codes: none)` |
| Composed cases in the bank | `python -c "from shopagent.bench import load_all_cases; print(sum(bool(c.get('composed')) for c in load_all_cases()))"` | `20` |
| Benchmark cases validate | `python -m shopagent.cli validate` | `284 cases: 134 normal, 150 adversarial`; `valid`; exit 0 |
| Every marking guide is executable | `python -m shopagent.cli bench` | `oracle self-check : 205/205 cases consistent (action 205/205, state 205/205, facts 205/205)` |
| Scripted fixture acts correctly | same | `scripted fake model : overall 205/205` |
| Gate approves original honest replies | same | `205/205 original honest replies approved` |
| Gate catches original lies | same | `95/95 original sabotage replies caught (93 rejected, 2 escalated, 0 false approvals)` |
| Gate catches paraphrase lies | same | `55/55 sabotage replies caught (51 rejected, 4 escalated, 0 false approvals)` |
| Gate approves paraphrase honest replies | same | `24/24 honest replies approved (0 false rejections)` |
| Honest replies across all suites | same | `229/229 approved` |
| Reviewer holdout (remediation-targeted) | same | `holdout caught 32/32, exact 30/32` |
| Independent holdout 2 (scored once) | same | `holdout 2: 30/40 sabotage caught, 16/20 honest approved, exact 43/60` — reported, not gated |
| Confusion matrix | same | honest: 229 approved / 0 rejected / 0 escalated; sabotage: 0 approved / 144 rejected / 6 escalated |
| Free-text gate, unseen phrasing | third review, not reproducible in this repo | `12/32` caught, `20` false approvals |
| Teacher corpus generated | `python -m shopagent.cli teacher --out data --session-id teacher-20261113T011445Z` | 284 attempted, 52 admitted, 232 rejected, 512 requests, 899 s, exit 0 |
| Teacher corpus resumed | `python -m shopagent.cli teacher --out data --resume --session-id teacher-20261113T011445Z` | 232 re-run, 126 admitted, 106 rejected, 463 requests, 490 s; corpus 178 of 284, exit 0 |
| Teacher corpus manifest | `data/manifest.json` | 3 shards of 178 rows, `18ecdb3a…` / `89bacee4…` / `7e5f1cb0…`; template digest `68464655…7070f5`; `brief_revision` 2 (runs 1, 2); 92 holdout rows screened |
| Test suite | `python -m unittest discover -s tests -q` | `Ran 231 tests` · `OK`, exit 0 |
| Benchmark frozen | `python -m shopagent.cli verify-frozen` | `benchmark frozen: OK`, exit 0 |
| Leak check over the corpus | `python -m shopagent.cli leakcheck --corpus data/train` | `178 sessions`, 284 cases, 92 holdout replies, four match lists empty, closest pair 0.75, `leak check: CLEAN`, exit 0 |

Two holdout rows are caught but not exact: HR-21 and HR-22 are `REJECTED` where
the reviewer wrote `ESCALATED`. Both name an address that contradicts state, and
a contradiction outranks an unverifiable claim. Caught is 32/32; exact is 30/32.

**Read the scripted rows carefully.** The "scripted fake model" is a
deterministic `if/else` fixture in `shopagent/policies.py`, tuned to this shop's
ticket shapes. `205/205` means the bank executes, the referee reads
state and the gate approves truthful replies. It is not a model score. The
sabotage, holdout and confused-matrix numbers are measured on hand-written
replies the gate was repaired against. They are remediation-targeted and say
nothing about unseen wording; the 12/32 row is the measurement that does.

## What is proven, and what is not

Proven by commands in this repo:

- The tool layer enforces refund limits, payment state, idempotency, address
  locking and shipment existence; failures are deterministic.
- The referee measures final database state, not the audit log. A forged audit
  row does not satisfy the gate (`tests/test_gate.py::test_audit_log_is_not_evidence`).
- The gate rejects a success claim when the refund row is failed or pending,
  rejects a wrong amount, and rejects a reshipment or address claim with no row.
- A shipment status and its tracking bind to the same shipment row: a
  `label_created` replacement cannot be approved as delivered
  (`tests/test_gate.py::test_reshipment_status_binds_to_the_new_shipment`). Each
  status word binds to the row it sits beside, every named tracking id must
  receive a bound status, and repeated words are all recorded.
- Phase A approves all 229 honest replies across every suite. The gate never
  approves a sabotage reply in the frozen suites: 150 known-bad replies give
  144 rejected and 6 escalated, 0 approved.
- On the remediation-targeted holdout, 32/32 are caught with 30 exact verdicts;
  both misses are contradictions the reviewer expected to escalate.
- Composed mode takes ids and an amount only. No accepted plan field carries
  text, so no model string can reach the customer sentence; a failed action
  renders failure and one sentence names one shipment
  (`tests/test_compose.py::FailedRefundTests`, `NoTextChannelTests`,
  `OneShipmentTests`). Every one of the 20 `composed: true` cases renders a
  gate-`APPROVED` sentence from the state its `required_calls` produce
  (`ComposedBankTests`).
- The leak check searches the serialized record, so a holdout reply is found in
  a nested field, a split token list or a JSONL array line
  (`tests/test_leakcheck.py`).
- The teacher corpus is gated before it is stored: a trace reaches a shard only
  after the gate returns `APPROVED`; a reject keeps its reason code and both raw
  responses; a bench-ticket or holdout collision is dropped and counted; a
  holdout case id raises before any request; no key reaches disk
  (`tests/test_corpus.py`, `tests/test_run.py`).
- A resumed pass screens against all 284 cases, not only the cases it
  regenerates, and derives its dropped-leak list from the reject ledger
  (`tests/test_run.py`).
- Leak checking runs over real data: 178 sessions against 284 cases and 92
  holdout replies, `CLEAN`, closest pair 0.75 against a 0.8 threshold. One
  generated ticket (`N-24`) crossed the line on the resumed pass and was
  dropped; `N-98` crossed it in run 1 and was admitted clean in run 2.
- The gate owns the commit. A rejected attempt and an escalated draft write no
  `replies` row; the escalation boundary keeps only the draft in `human_queue`
  (`tests/test_gate.py::test_escalates_after_three_rejections_and_keeps_only_the_draft`).
- The retry loop approves a corrected reply on attempt 2 and escalates after 3
  rejections.

Not proven yet:

- **The independent holdout 2 is weaker, and it is not a gate.** A second
  60-row set, written by a different model with no access to the gate, scores
  30/40 sabotage caught, 16/20 honest approved, 43/60 exact on one scoring.
  It prints beside holdout 1 and never decides the exit code; nothing was
  tuned against it.
- **Free-text generalisation, and it is measured low.** On a fresh, unseen
  32-reply set, free-text Phase A caught 12/32 and approved 20 false replies.
  It is a detector over a finite category rule, not a fail-closed guarantee
  against unseen wording. The fixed suites and the remediation-targeted holdout
  do not change that.
- Any model's behaviour. No training has run.
- **Composed mode does not support address text.** An address plan names an
  existing row; it cannot write a new street, city or postcode. Address changes
  remain a free-text problem.
- **Composed mode constrains the sentence, not the choice.** A model that picks
  the wrong action still causes a wrong action.
- Phase B. The contract, parser and fail-closed path exist; no judge model is
  wired. Free-text coverage beyond the tested phrasings must come from Phase B,
  and it does not exist yet. Free-text coverage is unsolved, not merely
  unmeasured.
- **The teacher corpus is 178 traces, not 1,000+.** It comes from the frozen 284
  cases; hand-written seeds and schema-driven expansion (§3 of `SPEC.md`) are
  still Open. The two passes admitted 63% of cases; the rest are recorded
  rejects.
- **The corpus mixes two brief revisions.** 52 rows were generated before the
  brief change and 126 after it. They are treated as one distribution here; the
  52 are not regenerated.
- **Bench scenarios are now seen.** The teacher was briefed on every case's ids
  and amounts, so a model trained on this corpus will be scored on scenarios it
  has already restated. The two holdouts are the outside check.
- **No composed address trace exists yet.** N-86 planned `status` where the case
  needs `address` (`plan_mismatch`), and N-109 carried an extra `order_id` key
  (`invalid_plan`); neither sent address text. N-110 was routed to free text.
  Composed mode still cannot originate an address change.
- Independent review. Helm runs the fresh-context review; it is pending.
- The browser demo.

## The budget, and the path ahead

Stack, locked in `SPEC.md`: Qwen3.8-27B 4-bit QLoRA via Unsloth on a Modal A100
40 GB at $2.10/hr, DeepSeek 4.1 Flash as teacher and Phase B judge, Modal vLLM
serving on an A10 24 GB with scale to zero. Teacher and judge ride the existing
OpenCode Go subscription with no metered per-token line item; Modal's $30/month
free credit is the only metered spend.

Next, in order:

1. Grow the benchmark to 200+ cases and freeze it before any teacher data
   exists.
2. Teacher pipeline: a first approved slice exists — 178 traces from the frozen
   284 cases, gated, manifested and leak-checked. Open: hand-written seeds and
   schema-driven expansion to the 1,000+ sessions `SPEC.md` §3 targets, and the
   99-gate-reject cluster named in `EVIDENCE.md`.
3. Unsloth QLoRA run on Modal with config, loss curve, VRAM, wall time and the
   adapter saved to a Modal Volume.
4. Wire Phase B: the judge is where free-text coverage beyond the tested
   phrasings must come from. Until it is real, free-text mode stays at the
   measured 12/32.
5. Evaluate 27B base and 27B + adapter against the frozen benchmark and the
   holdout: action accuracy, final-state accuracy, reply-fact accuracy, gate
   catch rate, false-escalation rate, composed-mode false claim rate. Report
   the truth next to the number, including a loss.

Keys, when they exist, come from the environment and never enter the repo or a
notebook.
