# Teacher data plan — milestone 3 (revision 2)

Basis: `SPEC.md` §3, `data/train/README.md`, and Daniel's brief. No training, no
Modal, no commit, no gate relaxation.

## Status — built (2026-11-13)

Slices S0–S4 are built and gated: **178 of 284 cases are admitted traces**
(164 free text, 14 composed), 106 rejects with reason codes, three shards,
manifest with per-shard digests, `leakcheck` CLEAN on 178 sessions, 231 tests OK.

Two pieces are **new surface, added outside this cleared plan**:

- `teacher --resume` / `run_generation(..., resume=True)` — Daniel asked for one
  round of brief improvement followed by a re-run of only the rejected cases,
  into the same output and session. A fresh run behaves exactly as before.
- `teacher.BRIEF_REVISION` and the manifest's per-run `brief_revision` — the
  brief changed after clearance, and `template_digest` cannot see a brief change.

The first pass (52 admitted) was produced by the cleared path. Known limits,
carried to `EVIDENCE.md`: 178 is below the 1,000+ target in `SPEC.md` §3; the
corpus mixes brief revisions (52 rows rev 1, 126 rev 2); composed address has no
trace; 99 rejects are `gate:REJECTED`/`gate:ESCALATED`, left unfixed and named
Open.

## Conflict settled with Daniel

The brief names the frozen 284 cases as source material and also requires
`leakcheck --corpus data/train` to stay CLEAN. `leakcheck` fails on any record
that contains a bench ticket. Daniel chose: **new tickets, same scenarios** —
the cases set the scenario, the customer message is written fresh, collisions
are dropped and counted. `data/train/README.md` already forbids copying a bench
ticket into a teacher instruction, so the teacher never sees one.

## Flow

```
case (title, ids, amounts, fault modes, required_calls, state_expect)
  -> composed-supported check; unsupported -> free text path
  -> call 1: fresh customer ticket + tool plan           (1 request)
  -> execute the plan against the real tool layer        (no request)
  -> free text: call 2 with the tool outputs -> reply    (1 request)
  -> composed: compose_reply renders the sentence        (no request)
  -> admit only if the slice's checks all hold
  -> train row; else reject row with its reason
```

`send_reply` is never allowed in a plan. The gate owns the commit, and the reply
exists only after call 2. A plan that names it is rejected as `invalid_plan`.

## Slices

### S0 — `leakcheck` covers both holdouts
Prerequisite: the run screens against both sets, so the tool must load both
before generation exists.
- `cmd_leakcheck` loads `review-01` and `review-02` and prints 92 replies.
- `check()` keeps its signature; the caller concatenates.
- Tests: a planted exact replay and a planted near-duplicate of one reply from
  each file are both caught; the count is 92.

Accept: unit tests pass; `leakcheck --corpus data/train` prints
`holdout replies (forbidden in training): 92` and CLEAN on the empty corpus.

### S1 — `shopagent/teacher.py`: client and contract
- `TeacherClient`: base `https://opencode.ai/zen/go/v1`, `/chat/completions`,
  model `deepseek-v4.1-flash`, `Authorization: Bearer <key>`, one
  `x-opencode-session` per run, `User-Agent`.
- Key from `OPENCODE_API_KEY`; `OPENCODE_GO_API_KEY` is a fallback only
  (Daniel's brief asks for the alias; the primary name is the one SPEC records).
  Absent -> `TeacherUnavailable`. The key never enters a row, a raw file, a log
  line or `repr`.
- Retry: 429 and 5xx up to 3 attempts with backoff; 403 at most one retry;
  400/401/402/404 not retried. Concurrency capped at 8.
- `build_scenario_brief(case)`: title, order ids and amounts from
  `required_calls`, fault modes, required action. No ticket text.
- `build_reply_brief(...)`: generated ticket, plan, tool envelopes.
- `parse_stage1` / `parse_stage2`: strict JSON, fixed keys, tool allow-list.
- `template_digest()`: sha256 over both prompt templates.

Accept: fake-transport tests prove URL, headers, model, key source and fallback,
retry counts per status, redaction, concurrency cap, and that malformed output
raises `TeacherParseError` with the text preserved.

### S2a — free-text admission
`generate_free_text(client, case)`:
1. call 1 -> `{ticket, plan}`; plan tools must be in `TOOL_SPECS` minus
   `send_reply`, args must be objects.
2. Execute the plan through `ToolKit` on a fresh `prepare_case`; collect audit
   rows.
3. `match_plan` (public alias of `bench._match_calls`) -> `plan_mismatch`.
4. `check_expect` -> `state_mismatch`.
5. **call 2** with the generated ticket, the plan and the tool envelopes ->
   `reply`. No fact check can run before this point: the reply does not exist.
6. Required facts present -> `facts_missing`; forbidden phrases absent ->
   `forbidden_phrase`.
7. `gate.evaluate` -> only `APPROVED` admits; else `gate:REJECTED` /
   `gate:ESCALATED`.

Rejects keep both raw responses. A holdout id raises before any request.

Accept: one negative test per reason, each proving the row is absent from the
admitted set and present in rejects with that reason; a gate-rejected truthful-
looking lie never reaches a row; parse failure keeps raw.

### S2b — composed admission
`generate_composed(client, case)`:
1. call 1 -> `{ticket, plan}` where plan is the composed object.
2. Expected action from the mutating `required_calls` entry: `refund_order` ->
   `refund`, `create_reshipment` -> `reshipment`, `update_address` -> `address`,
   else `status`. The teacher's `plan["action"]` must equal it.
3. **Argument match, not just the action.** Compare every id and amount the
   required call names against the plan:

   | action | must equal `required_calls` args |
   |---|---|
   | refund | `order_id`, `amount_cents` |
   | reshipment | `order_id`; `shipment_id`, if present, must be the one the call produced |
   | address | `address_id` |
   | status | the `get_order` call's `order_id`; when `required_calls` is empty, the `orders` entry's `match.id` in `state_expect` |

   A missing or different value -> `plan_mismatch`. The status id source is the
   same one the scenario brief names, so the teacher and the match test read it
   from one function.
4. `compose_reply` validates the plan, runs the action and renders the
   sentence (`ComposeError` -> `invalid_plan`). Address composition renders
   only; it never calls `update_address`.
5. `check_expect`, required facts, forbidden phrases, then the gate on the
   rendered sentence.

**Supported-case rule (no expectation is ever weakened).** A composed case is
`unsupported_composed` when the plan shape cannot perform what the case
requires:
- more than one mutating `required_calls` entry (N-83: a refused 9000 refund
  then a successful 8000 one);
- an `update_address` call that succeeds and carries `line1`/`city`/`postcode`
  (N-98, N-110: composed address names an existing row and cannot write it).

An unsupported case goes down the S2a free-text path with its `required_calls`,
`state_expect` and facts unchanged, and is recorded as
`routed: unsupported_composed` in provenance. Applying the rule to the frozen
bank flags N-83, N-98 and N-110.

**Execution assertion.** Composition writes one audit row for `refund` and
`reshipment`, and zero for `status` and `address` (`render_status` and
`render_address` only read). Each test pins the exact count for its action.

Accept, one test each: successful admission for all four actions; state
mismatch; missing fact; forbidden phrase present; invalid plan shape; gate
rejection; gate escalation. Each excluded case is absent from the shard and
present in rejects with its reason. A status plan on a refund case is
`plan_mismatch`. A status case with no `required_calls` (N-99..N-102) takes its
order id from `state_expect` and is admitted. The plan expected the
locked-address cases (N-86, N-109) to render their refusal and be admitted; the
run did not: N-86 planned `status` for an address case (`plan_mismatch`) and
N-109 carried an extra `order_id` key (`invalid_plan`). N-83, N-98 and N-110
route to free text.

### S3 — run, store, manifest
- `verify-frozen` runs programmatically before the first request; a mismatch
  aborts the run.
- 8 workers, results re-sorted by case id, so layout is deterministic.
- `data/train/shard-NNN.jsonl` (64 rows each), `data/raw/raw.jsonl` (verbatim
  stage replies), `data/rejects/rejected.jsonl`, `data/manifest.json`.
- Manifest: counts, rows per shard + sha256, `cases_digest`, session id, model,
  gateway, template digest, generated_at, requests made, dropped leaks.
- Leak screen against cases and both holdouts runs before writing; a collision
  is dropped and recorded as `leak:<source>`.
- Provenance per row: model, gateway, case id, generated_at, verdict, codes,
  session id, source-ticket sha256 (never the text).

Accept: manifest arithmetic adds up and digests recompute; a sentinel key never
appears in any file under `data/`; a holdout case id cannot reach a shard; a
planted collision is dropped and counted; `.gitignore` covers `data/raw/`,
`data/rejects/` and `data/manifest.json`.

### S4 — CLI
- `python -m shopagent.cli teacher --dry-run N --out DIR --limit N --only IDS
  --session-id ID`.
- Exit 2 with a clear message when the key is absent; nothing written.

Accept: no-key path exits nonzero and writes no file; each flag parses
(`--dry-run`, `--out`, `--limit`, `--only`, `--session-id`); a stub-client test
of the printout.

### S5 — real run and record
- Dry run of 5, one approved trace pasted end to end.
- Full run over 284 cases, then `leakcheck`, `unittest`, `verify-frozen`,
  `validate`, `bench`, `git diff --check`.
- README, EVIDENCE, NOW updated with counts, reject reasons, composition,
  subscription requests, and two limits: (a) the corpus is 284 sessions, an
  approved subset of §3's 1,000+ target — seeds and expansion remain Open;
  (b) bench scenarios are now seen, so later bench scores are not generalisation
  and the holdouts are the outside check.

Accept: every command exits 0 (`leakcheck` CLEAN), EVIDENCE numbers match the
manifest, and `git status --short` shows only intended files.

## Out of scope
Training, Modal, holdout scoring, any change to `bench/`, any relaxation of the
leak check or the gate.
