# Spec — eval-gated-lora-agent

Status: **Locked 2026-11-09 by Daniel.** Stage 1 (harness) is done. No code in
this document is new work; it records decisions already made.

## 1. Objective

A store support agent that **cannot tell a customer something the database and
the API do not prove**.

The model proposes an action and a draft reply. A deterministic gate binds every
completion claim to real SQLite state and real mock-Stripe API codes before a
reply leaves the building. Composed mode goes further: the model returns ids and
an amount, and code renders the sentence from the row the action wrote.

Portfolio thesis: **fine-tuning shapes behaviour and format; a deterministic gate
guarantees truth.** The adapter makes the model a better agent. It is never the
thing that makes the agent honest.

User: a small shop's support desk. Success: an adapter that measurably beats its
own base model on a frozen benchmark, behind a gate with no false approvals,
with every number reproducible from a command.

## 2. Locked stack

| Part | Locked choice |
|---|---|
| Base / student | **Qwen3.8-27B**, 4-bit QLoRA via **Unsloth**. No 7B anywhere. |
| Training compute | **Modal Starter**, **A100 40 GB** at **$2.10/hr**; weights cached on a Modal Volume; 1-2 epochs; short sequences; `reasoning_effort` none |
| Teacher | `opencode-go/deepseek-v4.1-flash` over the **OpenCode Go** subscription's OpenAI-compatible API |
| Judge (Phase B) | `opencode-go/deepseek-v4.1-flash` over the **OpenCode Go** subscription's OpenAI-compatible API |
| Serving | **Modal vLLM** on **A10 24 GB**, scale to zero |
| Cost | Covered by the existing **OpenCode Go subscription**; no metered per-token line item. Modal's **$30/month credit** is the only metered spend |

Rules that follow from the table:

- 1-2 epochs and short sequences are a cost decision, not a quality decision.
  The adapter must be justified by measurement, not by run length.
- Weights live on the Modal Volume so a re-run does not re-download them.
- Scale to zero means the demo costs nothing when idle. Cold start is accepted.
- The only money we spend is Modal's $30/month credit. Teacher and judge calls
  ride the existing OpenCode Go subscription and carry no metered per-token line
  item. Report the Modal spend; do not hide it.
- The key comes from `OPENCODE_API_KEY` in the environment, sent as
  `Authorization: Bearer $OPENCODE_API_KEY` — never in the repo, never in a
  notebook. `OPENCODE_GO_API_KEY` does not exist on this box.
- Gateway `https://opencode.ai/zen/go/v1`, chat path `/chat/completions`,
  models list at `/models`. A request without an `x-opencode-session` header
  gets 400 `MissingSessionID`; any stable session id per run works.
- 8 concurrent requests succeeded at 1.0-1.8 s each, with no rate-limit header
  and no 429. Rate limits above 8 remain unmeasured.
- `https://opencode.ai/zen/v1` answers 402 Insufficient balance for this
  account, so the Go path is the one to use.

## 3. Date engine

```
15-20 hand-written seed tasks
      -> 1,000+ sessions (prompt, tool call, tool output, final reply)
      -> validators
      -> leak check against the frozen benchmark
      -> train.jsonl
```

- **Seeds.** 15-20 tasks written by hand: status, refund, reshipment, address,
  refusal, multi-step. Each seed names its tools, its fault and its required
  reply facts.
- **Sessions.** Schema-driven expansion over order ids, amounts and fault types,
  fixed seed per batch. Every tool call executes against the real tool layer.
  Every reply is checked against final state. Target 1,000+ validated sessions.
- **Three-way split.** `raw.jsonl` (teacher output), `rejected.jsonl` (with
  reason), `train.jsonl` (validated). Kept apart so the filter is auditable.
- **Leak check is a gate, not a report.** `python -m shopagent.cli leakcheck
  --corpus data/train` must print CLEAN against both the frozen tickets and the
  holdout replies. Any exact or near match fails the batch.
- **Never train on `bench/`.** `bench/` is read-only to the pipeline.

## 4. Benchmark

- **200+ locked cases**, frozen by content hash **before any training data is
  generated**.
- Today: 284 cases (134 normal, 150 adversarial; 205 bank + 79 paraphrase) plus
  a 32-case reviewer holdout. Both are frozen and both digests are pinned in
  `bench/FROZEN.json`.
- Growing 67 to 200+ is stage 2 work, and the freeze runs before the first
  teacher session. `bench` refuses to score a modified benchmark or holdout.
- Marking guides per case, all executable:
  - **expected calls** — tool plus expected `ok`/`code`
  - **expected DB state** — declarative expectations over the row
  - **required reply facts** — what the customer must be told
- Adversarial cases also carry a known-bad `sabotage.reply`; that is the gate's
  acceptance test.

## 5. Gate

Three verdicts, and only three: `APPROVED` / `REJECTED` / `ESCALATED`.

**Phase A — deterministic.** State delta, token binding, API codes. It reads
tables and rows, never the agent's log. A forged audit row fails the gate.

**Phase B — judge.** `opencode-go/deepseek-v4.1-flash` over the OpenCode Go
subscription API reads the execution log plus the draft reply. It returns strict
JSON with a fixed verdict enum and a reason. Invalid or
missing output fails closed. The gate has three outcomes:

| Outcome | Effect |
|---|---|
| `APPROVED` | the gate commits the reply |
| `REJECTED` | feed the reason back, retry; capped at 3 attempts |
| `ESCALATED` | draft to `human_queue`, never to the customer |

Retry stops at 3. A fourth attempt is a human queue entry, not a loop.

**The property.** Composed mode — ids in, row out, one shipment, no text input —
remains the only mode with a provable no-false-claim property. Phase B widens
prose coverage; it does not make a guarantee. The doc must keep saying that.

Closed in stage 4: `shopagent/judge.py` is wired to
`opencode-go/deepseek-v4.1-flash` through the shared client in `teacher.py`
(gateway, key rule, session header and retries are one implementation). It runs
after Phase A approves and only then. It stays off by default (`phase_b` unset)
until the measure slice turns it on. Phase B widens prose coverage; it does not
weaken the no-guarantee rule above.

## 6. Stages

| # | Stage | Status |
|---|---|---|
| 1 | **Harness** | **Done.** 171 tests OK; 284 cases and the 32-case holdout frozen; composed mode shipped; free-text limit measured at 12/32. |
| 2 | **Teacher data** | **Done.** Benchmark grown to 284 and frozen before any teacher call; 178 admitted traces of 284 cases, gated, manifested and leak-checked. §3's 1,000+ target stands unmet, and the freeze below records it as the constraint on everything after. |
| 3 | **Train** | **Done.** Qwen3.8-27B 4-bit QLoRA on a Modal A100 40 GB. Adapter `full-20261114T201900Z`: 263 written rows from 178 teacher traces, 40 % composed per epoch, 186 steps, loss 0.2051 → 0.0840 (first-five → last-five means), est. `$0.8592`, at `/vol/runs/full-20261114T201900Z/adapter` on the Volume. |
| 4 | **Measure** | **Verified; no fresh-context PASS is recorded, so unaccepted by this spec's own rule.** Both students answered all 284 frozen cases; Phase B scored every Phase A approval; the adjudicator measured the claims. Daniel closed training here on 2026-11-15: an owner's decision, not a verdict. |
| 5 | **Iterate** | **Further runs not run by decision (2026-11-15).** Three more runs on 17,868 supervised tokens would chase noise: the corpus is the constraint, not the run count, and §3's 1,000+ target is unmet. The stage-5 adjudication slice did run; it is verified and records no fresh-context PASS. Not run is not failed. |
| 6 | **Serve** | **Next.** Modal vLLM on A10 24 GB, scale to zero, plus a browser demo. |

Training stops at the freeze: no third iteration, no further corpus growth. The
comparison in §7 is against the adapter named above.

**No milestone here carries a fresh-context PASS.** Milestone 2, milestone 3,
milestone 4, milestone 4b, stage 4 and the stage-5 adjudication slice are each
verified on green commands and unaccepted by the rule below. The freeze is
Daniel's decision; it does not make any of them accepted, and no document should
say it does.

Each stage ends on a gate: green commands, quoted output, and a fresh-context
review. The author never signs its own stage.

## 7. Success criteria

- The adapter **beats base 27B** on the frozen benchmark, **or** the README says
  in plain words that it did not.
- **Zero false approvals** on a held-out adversarial set authored by someone
  else.
- **False-escalation rate reported.** A cautious gate that escalates everything
  is not a pass.
- The composed-mode no-false-claim claim still holds under the same tests.
- **Every number reproducible by a command**, printed in the README table next
  to the number.
- The **full cycle fits inside $30**.

### Verdict at the freeze (2026-11-15)

1. **Met on partly-seen data.** The adapter was approved on 233 of 284 cases
   under Phase A, base 27B on 101. The caveat is part of the verdict: the teacher
   was briefed on all 284 scenarios and traces from 178 of them trained the
   adapter, so its number is partly recall; base trained on nothing, so its 101 is
   neither recall nor a generalisation measure. Neither holdout scored either
   student.
2. **Unmeasured.** No student benchmark, no Phase B judgeship and no
   adjudication has run on either holdout. The gate's own holdout scores (32/32
   remediation-targeted, 43/60 reported and not gated) are gate results, not this
   criterion. Unmeasured is not met and not failed.
3. **Reported, and only partly measured.** Phase A escalated 39 of 284 base rows
   (13.7 %) and 18 of 284 adapter rows (6.3 %); Phase A + B left both counts
   unchanged. Which of those escalations were *false* is not measured. The only
   false-escalation evidence is the scripted honest control, 229/229 approved and
   0 escalated, and that is hand-written replies. Base's 97 tool-plan-in-reply
   rows are a plan defect, not an explanation of its 39.
4. **Met as the renderer's property.** The frozen adapter parses and matches all
   14 composed cases in the corpus (`out/modal-check5.log`), all seen; the
   no-false-claim property itself belongs to `shopagent/compose.py` and its tests,
   not to the student.
5. **Partly met.** The README table binds a command to every number this repo
   produced. Three gaps are named beside their numbers: the free-text 12/32 has no
   command in this repo and no retained artifact, both `modal run …` lines are
   `Unverified:` in the stage-4 record, and the metered `$11.10346860` is a Modal
   dashboard read that no command here prints.
6. **Met to the freeze.** Eleven Modal estimates total `$5.4430`; metered
   `$11.10346860`; billed `$0.00`, because credits cover it. Stage 6 serving has
   not run, so its spend is not in the figure: this is the cycle so far, not a
   finished cycle.

## 8. Limits to record

- **Teacher and judge are the same model.** Both are
  `opencode-go/deepseek-v4.1-flash` over the OpenCode Go subscription API. Judge
  independence is reduced and its agreement rate is partly self-confirming.
  Mitigation: a blind check where the judge scores replies produced by a
  different generator, reported separately. The README must name the shared
  model.
- **Phase B needs network and an API key at demo time.** The key comes from
  `OPENCODE_API_KEY` in the environment, never from the repo, never from a
  notebook. If the judge is unreachable the gate fails closed and the demo says
  so. Rate limits above 8 concurrent requests remain unmeasured.
- **Cost.** Teacher and judge are covered by the existing OpenCode Go
  subscription; no metered per-token line item. The only metered spend is
  Modal's $30/month credit.
- **27B gains less from LoRA than a small model** does. The adapter must be
  justified by measurement. "It trained" is not a result.

## 9. Non-goals

Real Stripe. Auth. Multi-tenant. A 7B variant. Paid hosting beyond the free
credit. RL.

## 10. Commands

```bash
python -m shopagent.cli demo          # tools, gate, retry, composed mode
python -m shopagent.cli bench         # oracle, scripted fixture, gate check, holdout
python -m shopagent.cli validate      # case files against the schema
python -m shopagent.cli freeze        # write bench/FROZEN.json
python -m shopagent.cli verify-frozen # refuse to score a changed benchmark
python -m shopagent.cli leakcheck --corpus data/train
python -m unittest discover -s tests -q
```

Stage 3-6 commands are added with those stages. Each joins the README numbers
table with its cost and its wall time.

## 11. Project structure

```
shopagent/     gate, referee, tools, judge stub, compose, bench, freeze, leakcheck, cli
bench/         cases/ + holdout/ + FROZEN.json — read-only once frozen
data/train/    seeds, raw.jsonl, rejected.jsonl, train.jsonl (gitignored)
docs/          GATE.md, BENCHMARK.md
tests/         one file per subsystem
train/         Unsloth QLoRA script and Modal app (stages 3-5)
serve/         Modal vLLM app and browser demo (stage 6)
```

## 12. Boundaries

- **Always:** run the gates before reporting a number; quote command output;
  keep `bench/` frozen and read-only; run leak check after every teacher batch;
  keep keys in the environment.
- **Ask first:** any spend beyond the Modal credit; any change to the frozen
  benchmark or holdout; any change to the gate's verdict enum; publishing or
  deploying the demo.
- **Never:** train on benchmark or holdout rows; commit a secret; name a 7B
  model as the base; claim a model result that no command produced; let Phase B
  approve its own evidence without the Phase A check.

## 13. Open items

1. Who authors the independent adversarial set, and by when. It must be someone
   other than the person who writes the gate rules.
2. The exact teacher prompt format for tool-call turns — one call per turn or
   batched. Decided in stage 2 against a validator, not by preference.
3. Cold-start time on the A10 scale-to-zero path, measured in stage 6.
