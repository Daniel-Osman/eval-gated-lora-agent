# NOW - eval-gated-lora-agent

Status: **stage 6 is live and measured, 2026-11-16.** The endpoint and the browser
demo are deployed and verified. Agent: `POST
https://daniel-osman--eval-gated-lora-agent-agent-run.us-east.modal.run` and `GET
https://daniel-osman--eval-gated-lora-agent-health.us-east.modal.run`, one A10 at
`max_model_len 4096`, `max_num_seqs 8`, `min_containers=0`, Phase A only. Demo:
`https://daniel-osman--eval-gated-lora-agent-demo-web.us-east.modal.run`. Cold
start through the demo's `/run` from **0** agent containers was **309 s wall**
with **291.4 s** of engine init; the post-deploy first request to the same door
took **384 s**, the difference being image pull and container start; warm calls
answered in **2-3 s**; scale-down returned the agent to **0** containers in
about two minutes. Spend is metered, not
estimated: **17.18067970** this month, **0.00** billed; the deploy and demo
verification cost **$2.39591760**. Training stays frozen at
`full-20261114T201900Z` - 27B 4-bit QLoRA, 263 written rows from 178 teacher
traces, 186 steps, loss 0.2051 -> 0.0840, est. `$0.8592`, on the Modal Volume, not
in this repo (see `HISTORY.md` for the superseded freeze note).

One line per stage: harness - 171 tests, 284 cases and a 32-case holdout frozen.
Teacher - 178 admitted traces of 284, gated, manifested, leak check CLEAN. Train -
three runs, the third frozen. Measure - Phase A base 101 / adapter 233 approvals;
Phase B moved 45 and 7 rows; the adjudicator found 0 unsupported claims in 118
approved adapter adversarial drafts against 2 in 36. Serve - endpoint and demo
live, 309 s idle cold, 384 s post-deploy first request, 2-3 s warm, 400
passthrough, N-110 rejected, N-104 shown in the browser.

**No milestone here records a fresh-context PASS.** Milestones 2, 3, 4, 4b, stage 4
and the stage-5 adjudication slice are each verified on green commands and
unaccepted by the rule that an author never signs its own stage. Stage 6's deploy
and demo record is the same kind: verified, not accepted. Daniel closed training
on 2026-11-15; that is an owner's decision, not a verdict.

Verified 2026-11-16: `python -m unittest discover -s tests -q` -> `Ran 437 tests`,
`OK`, exit 0. Post-deploy first request, through the demo's `/run`:
`POST /run {"scenario":"N-110"}` -> 384 s wall, HTTP 200, REJECTED,
`matches_required_calls false`, `training_seen false`, UNPROVEN reading "this
preset is a frozen bench case the adapter never trained on". Then warm:
`POST /run {"scenario":"N-100"}` -> HTTP 200 in 3 s, APPROVED,
`matches_required_calls true`, `training_seen true`. Idle cold call:
`modal container list --json` printed 0 agent containers, then
`POST /run {"scenario":"N-100"}` -> 309 s wall, 291.4 s init, APPROVED
(`out/serve-deployed3-cold.json`); 309 s is the idle scale-from-zero
measurement, 384 s the post-deploy first request, the difference image pull and
container start. Scale-down: `modal container list --json` after
idle -> demo container only, agent containers 0. `POST /run {"scenario":"N-999"}`
-> HTTP 400 with the endpoint's own preset list. Browser: `Run N-104` filled the
panel; `out/demo-n104.png`. Spend:
`python -m modal billing summary --for "this month" --json` -> metered
17.18067970, billed 0.00. `git diff --check` exit 0, no output. Earlier verified
runs, back to 2026-11-13, sit in `HISTORY.md` under Verified history.

## Next

1. Validate `scenario` before the model container: check the id against the preset
   list in the demo app (it already proxies `/health`) or a small CPU function, so
   an unknown id costs 400, not a five-minute cold start.
2. Daniel's ship call: commit, and whether this single-user A10 demo is the stage-6
   delivery or the serving line moves to a larger GPU. Evidence and every number:
   `docs/SERVING.md`, `EVIDENCE.md` stage 6, `out/serve-*.json`.

## Open

**Validation wakes the GPU.** An unknown preset id pays a ~5-minute cold start
inside the model container before it answers 400. The page offers only known
presets; a caller can still send one.

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
id `full-20261113T205627Z`; nothing committed.

## History

`HISTORY.md` holds what this project is, its decisions, the freeze decision, the
superseded next steps, the verified-run tail and the evidence index.
