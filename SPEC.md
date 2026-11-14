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

1. **Harness - done.** 171 tests OK; 284 cases and the 32-case holdout frozen;
   composed mode shipped; free-text limit measured at 12/32.
2. **Teacher data.** Grow the benchmark to 200+ and freeze it. Then seeds,
   sessions, validators, leak check.
3. **Train.** Qwen3.8-27B 4-bit QLoRA on Modal A100 40 GB.
4. **Measure.** 27B base vs 27B + adapter on the same frozen benchmark. Wire
   Phase B to the DeepSeek judge in this stage.
5. **Iterate.** 3-4 runs inside the credit.
6. **Serve.** Modal vLLM on A10 24 GB, scale to zero, plus a browser demo.

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
