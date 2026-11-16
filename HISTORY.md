# HISTORY - eval-gated-lora-agent

Overflow from `NOW.md`: what the project is, the decisions it holds, and where
the evidence lives. `NOW.md` keeps the current status, next step and open items.

## What this is

A store support agent whose replies are gated on database state. A separator
between "the model's claim" and "what actually happened". Locked stack: a 27B
Qwen3.8 student tuned 4-bit with Unsloth on Modal, DeepSeek 4.1 Flash as teacher
and judge, served by Modal vLLM, inside the $30 Modal credit.

## Freeze (2026-11-15)

**Daniel's call: training stops here.** No further iteration, no further corpus
growth. The frozen artifact is `full-20261114T201900Z`: 27B 4-bit QLoRA, 263
written rows from 178 teacher traces, 40 % composed per epoch, 186 steps, loss
0.2051 -> 0.0840 (first-five -> last-five means; logged points 0.2769 -> 0.0975),
est. `$0.8592`, at `/vol/runs/full-20261114T201900Z/adapter` on the Modal Volume.

The reason, in the same words each document uses: three more runs on 17,868
supervised train-split tokens would chase noise. The corpus, not the run count, is
the constraint, and `SPEC.md` §3's 1,000+ target stands unmet. Stage 5's iteration
runs were therefore **not attempted**; that is a decision, not a failure, and no
document calls it one.

What the freeze closes: the search for a better adapter on this corpus, and the
iteration stage itself. What it does not close: the held-out metric, the corpus
shortfall, the accuracy columns, and the gap between the gate's guarantee and
free-text detection. Stage 6 serving is next.

Stage 4 and every milestone before it are verified, and none carries a
fresh-context PASS: milestone 2, milestone 3, milestone 4, milestone 4b, stage 4
and the stage-5 adjudication slice are all unaccepted by the rule that an author
never signs its own stage. Daniel closed training at the freeze; that decision
does not make them accepted, and every document here says so.

## Decisions held

- Benchmark before training data, frozen by content hash. `bench` refuses to
  score a modified benchmark or a modified holdout.
- The referee reads tables, never the agent's log. A forged audit row fails the
  gate (test proves it).
- The gate's adversarial suite is the acceptance test: a claim of a refund after
  a declined refund must be rejected. The reviewer holdout is the outside check
  on that suite.
- The gate stays strict on a pre-existing refund or shipment read as a new claim
  (Daniel, 2026-11-13). The 71 `gate:REJECTED` and 28 `gate:ESCALATED` teacher
  replies remain rejects; no brief softening, no re-run for them.
- Phase B is a validated JSON contract with a strict enum; invalid or missing
  output fails closed.
- Three attempts, then `human_queue`. No infinite retry.
- Composed shape is taught by prompt plus a solved 40 % rebalance, not by
  constrained decoding. One prompt per mode names the contract; the free-text
  prompt is now the teacher prompt plus an output-shape paragraph, superseding the
  earlier "unchanged" decision.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `EVIDENCE.md` § Where this stands (2026-11-15) — the closing index: proven,
  measured but caveated, unmeasured, deliberately not done.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `docs/COMPOSED-MODE.md` for the composed promise, the failure, the fix and the
  limit that travels with 14/14.
- `EVIDENCE.md` for the measured runs, spends and gate outputs.
- `out/bench-results.json` after `python -m shopagent.cli bench`.

## Stage 5 status, as it stood before the freeze

Moved out of `NOW.md` when the freeze replaced it. The stage-5 adjudication slice
ran and is measured; its review is outstanding; no further iteration run followed.

Stage 5 slice 1 **settles the false-approval metric**, measured, not accepted:
`gpt-6-luna` adjudicated every Phase A-approved draft blind to kind, expectation
and gate code. Base 101 approved: adversarial 36 -> 15 supported / 2 unsupported
/ 19 incomplete, honest 65 -> 38 / 5 / 22. Adapter 233 approved: adversarial 118
-> 116 / 0 / 2, honest 115 -> 105 / 4 / 6. 334 rows adjudicated over 335 HTTP
attempts, 0 invalid, 0 unavailable, 0 not scored; 152.7 s. The old case-kind rule
scored 154 approvals as false approvals; the adjudicator finds 2 and 0 unsupported
claims, and base's 19 incompletes are tool plans sent as replies.

Two rows were ruled by hand. `N-102` is a false positive: the state holds one
shipment, so "the latest" is true - with it base 5/65 and adapter 4/115, without
it base 4/65 (6.2 %) and adapter 3/115 (2.6 %). `N-81` is label-unstable: both
students sent the same sentence on the same log and the labels split, base
`incomplete`, adapter `unsupported`; it stays in the adapter's totals, flagged
unstable, so it carries no verdict. All eleven Modal estimates total `$5.4430`,
metered `$11.10346860`, billed `$0.00`.

## Superseded next steps (2026-11-15)

The four items that stood in `NOW.md` before the freeze, and where each went.

1. "Repeat the adjudication in a fresh session over both artifacts, so `unstable`
   becomes a measured rate, then fix the strictness `N-102` exposed." Not done.
   A second pass is listed as deliberately not done in `EVIDENCE.md`.
2. "Stage 5 or 6: run the two holdouts with Phase B on, or serve the student on
   Modal vLLM." Split: serving became stage 6; the holdouts stay untouched.
3. "Score the composed mode, not just its shape: the action choice has no number
   yet." Still open - the composed student is proven on the schema only.
4. "The two adapter adversarial `incomplete` rows (A-62, A-84) need a ruling."
   Still open, and unfixed: they are adequacy questions, not false claims.

## Verified history

Moved out of `NOW.md` to keep it inside the line budget. `NOW.md` carries the
current verification line only.

Verified 2026-11-16 (stage-6 deploy record, before the web-layer slice):
post-deploy first request through the demo's `/run`, `POST /run
{"scenario":"N-110"}` -> 384 s wall, HTTP 200, REJECTED,
`matches_required_calls false`, `training_seen false`, UNPROVEN reading "this
preset is a frozen bench case the adapter never trained on"; then warm, `POST
/run {"scenario":"N-100"}` -> HTTP 200 in 3 s, APPROVED,
`matches_required_calls true`; idle cold call, `modal container list --json`
printed 0 agent containers, then `POST /run {"scenario":"N-100"}` -> 309 s wall,
291.4 s init, APPROVED (`out/serve-deployed3-cold.json`), where 309 s is the
idle scale-from-zero measurement and 384 s the post-deploy first request, the
difference being image pull and container start; scale-down,
`modal container list --json` after idle -> demo container only, agent
containers 0; browser, `Run N-104` filled the panel, `out/demo-n104.png`; spend at
this record, `python -m modal billing summary --for "this month" --json` ->
metered 17.18067970, billed 0.00. Later reads are not this record's: 17.90068367
came before the last cold forward call, in the web-layer check after this record,
and 17.97068375 in the reviewer's read after that call, with its output not
retained in this repo. The counter moves with use, so no snapshot is a bill.

Verified 2026-11-15 (adjudication slice, before the freeze):
`python -m unittest discover -s tests -q` exit 0 (`Ran 363 tests`, `OK`, 26 of
them `tests/test_adjudicate.py`); `git diff --check` exit 0;
`python -m shopagent.cli adjudicate --concurrency 8` exit 0 -> 334 rows
adjudicated over 335 HTTP attempts (base 102, adapter 233), 0 invalid, 0
unavailable, 0 not scored, 152.7 s, writes `out/adjudicated-base.jsonl`,
`out/adjudicated-adapter.jsonl` and `out/adjudicated.manifest.json`; the counts in
`EVIDENCE.md` were read back from `out/adjudicated.manifest.json` and
`out/adjudicate-run.txt`; `verify-frozen` still OK.

Verified 2026-11-14 on this tree:
`python -m shopagent.cli judge-bench --rows out/bench-base-284.jsonl
--student base` exit 0 → 284 rows, 101 judged, 0 not-scored, 0 invalid, 0
unavailable, A+B 56 / 186 / 39, 61.4 s; the adapter twin → 233 judged, 0
not-scored, 0 invalid, 0 unavailable, A+B 226 / 38 / 18, 116.6 s; both write
`out/judged-<student>.jsonl` + manifest; replay fidelity checked over both
artifacts: 281 and 282 rows replayed, 0 envelope mismatches, 0 Phase A
mismatches; `python -m shopagent.cli validate` prints
`284 cases: 134 normal, 150 adversarial … valid`; `git diff --check` exit 0, no
output. Both bench artifacts carry `benchmark.verified: true` against
`cases_digest a76a732fa9b1f012…` and 2816-token windows.

Verified 2026-11-13 on the pre-slice tree:
`python -m shopagent.sft` exit 0 (178 source records → 263 rows, digest
`bfe5bb3e61676dd4`, `train_app_pin.matches_output: true`); 164 free-text records
hash to `0bb6d133…77d85b` (`test_free_text_records_hash_to_the_pre_fix_digest`;
the older `9519a868…87479` is recorded as superseded);
`full-20261113T205627Z` `run.json` `status: ok`, 186 of 186 steps;
`out/modal-train2.log` `supervision: 19693 tokens … 20120 supervised tokens; train
split 18260`; `out/modal-check2.log` `parse: ok`, `matches required_calls: True`;
`out/modal-check3.log` `summary: 14/14 parse ok, 14/14 match required_calls`.
Earlier: `validate` 284 valid, `bench` exit 0 (both holdouts reported, 229/229
honest approved), `leakcheck` 178 sessions CLEAN.
