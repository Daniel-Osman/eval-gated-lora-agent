# Portfolio: an eval-gated LoRA agent

A reading copy of the project for someone judging it without running it. Every
number here is carried by `README.md`, `EVIDENCE.md` or `docs/SERVING.md`; this
file adds no measurement of its own.

## What this is

This is a 27B QLoRA adapter trained to run the support desk of a small mock shop:
it reads a ticket, calls the shop's tools, and writes a reply. Before anything
reaches the customer, a deterministic gate checks the draft against the shop's
database and refuses or escalates what the rows do not support.

## The question it answers

Can a small, cheaply trained model be trusted to talk to customers if something
other than the model decides what is sent? For the case built here the answer is
yes, and the yes is narrow. The send decision belongs to code that reads the
shop's state, never to the model's own account of what it did. In composed mode
the model returns ids and an amount, and code renders the sentence from the row
the action wrote, so no model text can reach a customer. In free-text mode the
gate checks every completion claim it recognises against state, and refuses or
escalates what state contradicts.

What "trusted" does not mean here: the model is right, free text is true, or the
gate generalises. A gate can refuse a draft; it cannot turn a wrong plan into a
right one. The demo's N-104 case is exactly that. The model refunded 2,100 cents
where 2,599 was required, the reply matched the refund row, and Phase A approved
it, because the sentence was true against the row even though the plan was wrong.
The page draws that as a wrong decision the gate approved. That is the design
working, not a hidden failure.

## What was built

**The benchmark.** 284 frozen cases, 134 normal and 150 adversarial, plus a
32-case reviewer holdout and a separate 60-case holdout scored once. Artifact:
`bench/cases/`, frozen by content hash in `bench/FROZEN.json`.

**The teacher corpus.** 178 admitted traces of the 284 cases, each admitted only
after its plan, state, facts and gate verdict checked out, with a leak check over
the result. Artifact: `data/manifest.json` and three shards under `data/train/`.
Both are gitignored and local-only, so not in this repo. A public reader checks
the frozen benchmark under `bench/`, `EVIDENCE.md` and `README.md` instead.

**The adapter.** One training run, `full-20261114T201900Z`: Qwen3.8-27B, 4-bit
QLoRA, 263 written rows from the 178 traces, 186 steps, loss 0.2051 to 0.0840.
Artifact: `/vol/runs/full-20261114T201900Z/adapter` on the Modal Volume, not in
this repo.

**The gate.** Phase A is a deterministic state check: every recognised
completion claim is bound to SQLite rows and mock-payment codes. Phase B is a
judge that reads a Phase A approval and can move it off approval. Artifacts:
`shopagent/gate.py`, `shopagent/judge.py`, `docs/GATE.md`.

**The adjudicator.** A second model, `gpt-6-luna`, reads each approved draft
against the ticket, the state and the tool log, blind to case kind, expectation
and gate code, and labels its claims supported, unsupported or incomplete.
Artifacts: `shopagent/adjudicate.py`, `out/adjudicated-base.jsonl`,
`out/adjudicated-adapter.jsonl`.

**The serving path.** The adapter is merged to bf16, quantized to W4A16, and
served on one A10 with vLLM at a 4,096-token window and `max_num_seqs 8`.
Artifacts: `serve/feasibility.py`, `serve/app.py`, artifact
`/vol/serve/slice1/awq-w4a16` at 17.31 GiB, `docs/SERVING.md`.

**The demo.** A CPU-only Modal app serves a page, proxies the agent's preset
list, refuses a bad preset before the GPU call, and forwards a run through the
Modal SDK. Artifacts: `serve/demo.py`, `serve/demo.html`, live at the URL below.

## What is measured

| Measurement | Command | Result |
|---|---|---|
| Approvals on the frozen bench, base 27B against 27B plus adapter | `python -m modal run train/bench_app.py --student base --out bench-base-284` and the same with `--student adapter --run-id full-20261114T201900Z --out bench-adapter-284` | 101 of 284 for base, 233 of 284 for the adapter. The adapter trained on traces from 178 of these cases, so its number is partly recall; base trained on nothing. |
| Unsupported claims in approved adversarial drafts | `python -m shopagent.cli adjudicate --concurrency 8` | 2 of 36 for base, 0 of 118 for the adapter, read once by one blind adjudicator. The honest control disagrees on 5 of 65 base drafts and 4 of 115 adapter drafts, raw. After two hand rulings, `N-102` a false positive and removed, `N-81` flagged unstable and still counted, the narrower reading is 4 of 65 (6.2 %) and 3 of 115 (2.6 %). So zero means none was found, not that none exists. |
| Composed schema | `python -m modal run train/modal_app.py --check` | 14 of 14 cases parse and match their required calls. All fourteen were seen in training or validation, so this is a compatibility check on the serving path, not a quality score. |
| Serving | `python -m modal run serve/feasibility.py::fit_cli`, and `POST /run` to the demo | 309 s cold through the demo with 291.4 s of engine init, 2 to 3 s warm, 19.03 of 22.06 GiB used, 22.84 output tokens per second, one long request at a time. |

Two readings of these rows matter more than the rows. First, the approval
comparison is not an accuracy score. No accuracy column exists for action, final
state or reply facts, and neither holdout was run, so nothing in it is unseen
data. Second, the zero of 118 is smaller than it sounds: it is one model's
reading of drafts on seen cases, and the same reading finds disagreements in the
honest control.

## What the demo shows

Two URLs, both live on Modal in `us-east`:

`https://daniel-osman--eval-gated-lora-agent-demo-web.us-east.modal.run` is the
page. `https://daniel-osman--eval-gated-lora-agent-health.us-east.modal.run` is
the agent's model-free preset list; the page reads it to fill its presets.

Those two URLs are the public artifacts: the served weights and the adapter
live in a private Hub repo, `Daneil960/eval-gated-qwen3.8-27b`, available on
request.

A visitor sees a single page with a mode switch, the 20 frozen composed presets
pulled from `/health`, a ticket box, and a result panel. A run fills that panel
with the plan, the tool call, the gate verdict, the reply, the timings, the
library versions, an UNPROVEN line and a Phase B off block. The first call after
idle waits about five minutes for the GPU; the page says so while it waits.

Two cases are worth clicking. **N-104** is the validation row and the button
pinned to the panel: the model refunds 2,100 cents where 2,599 was required, the
gate approves, and the page labels the result a wrong decision the gate approved,
with the reason. **N-110** is one of the six presets never trained on: the gate
rejects it, the response carries `training_seen false`, and the page says the
preset is a frozen bench case the adapter never trained on.

The shop is a single-user mock. Its database is seeded SQLite, its payment API
is a mock, and the A10 serves one long request at a time.

## What is not proven

**Free-text quality on unseen tickets.** One warm free-text call returned a
correct-looking reply on a scenario the adapter had seen. The free-text gate has
one historical observation on an unseen 32-reply sample, 12 caught and 20 false
approvals, with no command and no retained artifact in this repo.

**The held-out zero-false-approval target.** `SPEC.md` section 7 asks for zero
false approvals on a held-out adversarial set. Neither holdout has a student run,
a Phase B judgeship or an adjudication. The 0 of 118 is measured on seen cases.

**Generalisation.** Every bench case was seen by the adapter or by the teacher
that wrote its traces. A match on an unseen preset in the demo is a
serving-compatibility result. The two holdouts remain untouched by any student.

**The corpus against the plan.** 178 admitted traces sit against `SPEC.md`
section 3's 1,000+ validated sessions. Stage 5's further training runs were not
run by decision, because the corpus is the constraint and more runs would chase
noise; the stage-5 adjudication slice did run, and its record is in `EVIDENCE.md`.

**Composed mode cannot originate a change.** The model returns ids and an amount,
and code renders the sentence from the row the action wrote. A change the rows do
not already carry cannot come out of composed mode: an address change needs the
free-text path, and the quality of that path on unseen tickets is unmeasured.

**No milestone carries a fresh-context PASS.** Each milestone in this repo is
verified on green commands and none is accepted, because acceptance is the
owner's call and no reviewer has signed one. This slice is no exception: its own
review found four defects, the cache rule in `serve/demo.py` among them, before
this fix.

**Phase B in the deployed app.** The judge runs offline. The live demo answers
with Phase A only, because the judge's key does not go into a remote sandbox.

**Concurrency above one request.** The serving measurement loaded at
`max_num_seqs 8`, which vLLM reports as 1.08x concurrency at 4,096 tokens. Two
long requests cannot be in flight together, and load behaviour is unmeasured.

**Scale-from-zero latency minus container start.** The 309 s cold figure holds
291.4 s of engine init. Container start, image pull and scheduling are the
remaining 18 s or so, and no command here splits them.

## Cost

`python -m modal billing summary --for "this month" --json` was read three times
on 2026-11-16, and each read keeps its own moment. Metered **17.18067970** at the
stage-6 deploy record, before the web-layer slice. Metered **17.90068367** on the
read immediately before the last cold forward call. Metered **17.97068375** in the
reviewer's read after that call, a read whose output is not retained in this repo.
Billed stayed **0.00** (credits) on each. The counter moves with use, so no
snapshot is a bill. `EVIDENCE.md` holds the eleven-run estimate history, 5.4430
across those runs.

Trust the metered figure. It is Modal's own counter, it includes idle container
time, image builds, failed attempts and five cold starts, and it is not an
arithmetic projection. The run estimates cover compute at list rates only and
exclude the adapter save and the Volume commit, which is why they read lower.

## Reproduce it

```
python -m shopagent.cli verify-frozen
python -m modal run train/bench_app.py --student base --out bench-base-284
python -m modal run train/bench_app.py --student adapter --run-id full-20261114T201900Z --out bench-adapter-284
python -m shopagent.cli adjudicate --concurrency 8
python -m modal run train/modal_app.py --check
python -m modal run serve/feasibility.py::fit_cli
```

The first command re-checks the frozen benchmark by content hash. The next two
regenerate the approval counts, the fourth regenerates the adjudication, the
fifth the 14 of 14 composed check, and the sixth the A10 fit and throughput. The
demo's cold and warm timings come from `POST /run` on the demo URL, quoted in
`docs/SERVING.md`. The adjudicator is the one command here that needs the
OpenCode Go key; the Modal runs need a Modal login, and the suite,
`python -m unittest discover -s tests -q`, needs the pinned tokenizer and
`data/train/` as `README.md` describes. On this tree it prints `Ran 448 tests`,
`OK`.
