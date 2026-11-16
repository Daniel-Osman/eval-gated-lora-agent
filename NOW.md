# NOW - eval-gated-lora-agent

Status: **stage 6 is live and measured, a bad preset is refused at the web layer,
and the artifact is published to a private Hub repo, 2026-11-16.** The endpoint and
the browser demo are deployed and verified. Agent: `POST
https://daniel-osman--eval-gated-lora-agent-agent-run.us-east.modal.run` and `GET
https://daniel-osman--eval-gated-lora-agent-health.us-east.modal.run`, one A10 at
`max_model_len 4096`, `max_num_seqs 8`, `min_containers=0`, Phase A only. Demo:
`https://daniel-osman--eval-gated-lora-agent-demo-web.us-east.modal.run`. The demo
caches the preset ids from the model-free `/health`, refreshes once on a miss, and
refuses an unknown id with **HTTP 400**, the endpoint's own sentence, before the
GPU RPC; an unreadable `/health` fails open, so a stale list cannot turn a good
request into a 400. An unknown id answered **400 in 0.96 s** warm and **4.85 s**
with a cold health function, and neither added an `engine: ready` line, so no
model container woke. N-100 after those 400s was **319 s wall**,
`engine.init_seconds 301`, APPROVED; cold start through the demo from **0** agent
containers was **309 s wall** with **291.4 s** of engine init; the post-deploy
first request took **384 s**; warm calls answered in **2-3 s**; scale-down returned
the agent to **0** containers in about two minutes. Spend is metered, not
estimated: **18.68353523** this month after the export, **0.00** billed; the
stage-6 record read 18.16068483. Training stays frozen at `full-20261114T201900Z`
- 27B 4-bit QLoRA, 263 written rows from 178 teacher traces, 186 steps, loss
0.2051 -> 0.0840, est. `$0.8592`, on the Modal Volume, not in this repo (see
`HISTORY.md` for the superseded freeze note).

One line per stage: harness - 171 tests, 284 cases and a 32-case holdout frozen.
Teacher - 178 admitted traces of 284, gated, manifested, leak check CLEAN. Train -
three runs, the third frozen. Measure - Phase A base 101 / adapter 233 approvals;
Phase B moved 45 and 7 rows; the adjudicator found 0 unsupported claims in 118
approved adapter adversarial drafts against 2 in 36. Serve - endpoint and demo
live, 309 s idle cold, 384 s post-deploy first, 2-3 s warm, 400 in 0.96 s and
4.85 s with no GPU wake, 319 s cold forward, N-110 rejected, N-104 in the
browser. Export - private Hub repo `Daneil960/eval-gated-qwen3.8-27b`, **16
files**, 19,075,486,047 bytes, `private True`, anonymous read 401.

**No milestone here records a fresh-context PASS.** Milestones 2, 3, 4, 4b, stage 4
and the stage-5 adjudication slice are each verified on green commands and
unaccepted, by the rule that an author never signs its own stage; stage 6's deploy
and demo record, and this web-layer slice, are the same kind. Daniel closed
training on 2026-11-15, and that is an owner's decision, not a verdict.

Verified 2026-11-16: `python -m unittest discover -s tests -q` -> `Ran 448 tests`,
`OK`, exit 0 (443 before the health fix; five new). Web-layer refusal:
`POST /run {"scenario":"N-999"}` with a warm cache and warm health -> **400 in
0.96 s**; the same call with a **cold health function** -> **400 in 4.85 s**;
`modal app logs eval-gated-lora-agent` holds one container start and one
`engine: ready in 301.0s` line, both from the forward call below, and neither 400
added a line. Then `POST /run {"scenario":"N-100"}` -> **200** in **319 s**,
APPROVED, `matches_required_calls true`, `engine.init_seconds 301`. The stage-6
deploy record that preceded this slice (384 s post-deploy first request, REJECTED
N-110, 3 s warm, 309 s idle cold, scale-down to 0, N-104 in the browser) is in
`HISTORY.md` under Verified history. `git diff --check` exit 0, no output.

Verified 2026-11-16, private Hub export: `python -m modal run serve/export_hf.py
--verify-only Daneil960/eval-gated-qwen3.8-27b` -> `ok private is True`; `ok
list_repo_files agrees with model_info on 16 files`; `ok model.safetensors is
18,568,390,152 bytes`; `ok adapter/adapter_model.safetensors is 467,062,560
bytes`; `ok every one of the 15 source files is in the repo`; `ok anonymous read
refused with HTTP 401`. Repo **`Daneil960/eval-gated-qwen3.8-27b`**, **16 files,
19,075,486,047 bytes**, head `417d1ff`. The token lives only in the Modal secret
`scenesift-hf` (`HF_TOKEN`). Spend 18.17068511 before, 18.68353523 after,
**0.51285012** metered, 0.00 billed. No commit here.

## Next

1. Daniel's ship call: commit the web-layer check, and whether this single-user A10
   demo is the stage-6 delivery or the serving line moves to a larger GPU. Evidence
   and every number: `docs/SERVING.md`, `EVIDENCE.md` stage 6, `out/serve-*.json`.

## Open

**Scale-from-zero latency is not separated from container start.** The 309 s wall
holds 291.4 s of engine init; the remaining ~18 s is container start, image pull
and scheduling, and no command here splits them. `SPEC.md` §13.3 stays open.

**Free-text quality on unseen tickets is unmeasured.** One warm free-text call
returned a correct-looking reply; that is one seen scenario, not a quality score.
`SPEC.md` §7's held-out set still has no student run.

**The corpus is the constraint, and it stays short of the plan.** 178 traces
against the 1,000+ sessions `SPEC.md` §3 targets. Hand-written seeds and
schema-driven expansion are unwritten, and the shortfall is why further iteration
would chase noise rather than learn.

**The held-out metric is unmeasured.** `SPEC.md` §7's zero false approvals on a
held-out adversarial set has no student run, no Phase B judgeship and no
adjudication behind it. The gate's own holdout scores (32/32
remediation-targeted, 43/60 reported and not gated) are gate results, not this
criterion.

Also open: Phase B is off in the deployed app, so the live demo shows Phase A
only; the endpoint answers one request at a time at `max_num_seqs 8`; the accuracy
columns were never computed (action, final-state, reply-fact); 52 corpus rows
still carry brief revision 1; `train/bench_app.py`'s docstring names the old run
id `full-20261113T205627Z`; the web-layer check and this record are uncommitted.

## History

`HISTORY.md` holds what this project is, its decisions, the freeze decision, the
superseded next steps, the verified-run tail and the evidence index.
