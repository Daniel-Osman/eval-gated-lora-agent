# NOW - eval-gated-lora-agent

Status: stage 3 gated, not yet accepted — the fresh-context verdict is
outstanding, and a PASS is what accepts it. Composed mode is **closed on seen
cases, with its limit stated**: the student now returns the plan object, and all
14 composed cases parse and match. SFT writes 263 rows from 178 records — 98
composed instances per epoch (six rows at 7 copies, seven at 8, seed-chosen)
against 147 free text, **40.00 % composed**, 19,693 supervised tokens, digest
`bfe5bb3e61676dd4079cf2f81285ed82b9219b2659913924a6a20f7c2bd0e9cc`, 62 steps
per epoch × 3 = 186, cap 200; val keeps its 18 rows, ids and order, but is not
byte-identical to the last run — `T-N-104`'s system turn is the composed prompt
now. Run
`full-20261113T205627Z` retrained the adapter: 186 steps, loss first five
0.2042 → last five 0.0850 (points 0.2757 → 0.0994), 1174.7 s wall, 6.18 s/step,
peak 21.22 GB, **estimated $0.8299**, cap $3.39; adapter at
`/vol/runs/full-20261113T205627Z/adapter`. `out/modal-check2.log`: composed
N-105 emits `{"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059"}`,
parse ok, three fields match, adapter-active proof passed. `out/modal-check3.log`:
**14/14 composed parse and match** ($0.0695 and $0.0848). The 14 are all seen
cases: 13 were in the training split, N-104 is the val row and never trained, so
14/14 measures the schema across cases, not generalisation. The three runs
estimate **$0.9842**; the metered figure last read is **$5.99564250**, billed
**$0.00** (credits cover it). Gates green on this tree: 275 tests OK,
`git diff --check` clean. Nothing committed.

Next: stage 4 — wire the Phase B judge, then evaluate base 27B against 27B +
adapter on the frozen bench and both holdouts.

Verified 2026-11-13 on this tree: `python -m unittest discover -s tests -q` exit 0
(`Ran 275 tests`, `OK`); `python -m shopagent.sft` exit 0 (178 source records →
263 rows, digest `bfe5bb3e61676dd4`, `train_app_pin.matches_output: true`);
164 free-text records hash to `9519a868…87479`, the pre-fix value
(`test_free_text_records_hash_to_the_pre_fix_digest`);
`full-20261113T205627Z` `run.json` `status: ok`, 186 of 186 steps; meanwhile
`out/modal-train2.log` prints `supervision: 19693 tokens from the stored masks +
427 <|im_end|> tokens across 427 assistant spans = 20120 supervised tokens; train
split 18260`; `out/modal-check2.log` `parse: ok`, `matches required_calls: True`;
`out/modal-check3.log` `summary: 14/14 parse ok, 14/14 match required_calls`;
`git diff --check` exit 0, no output. Verified earlier on the pre-fix tree:
`validate` 284 valid, `bench` exit 0 (both holdouts reported, 229/229 honest
approved), `leakcheck` 178 sessions CLEAN.

Open: the composed student is proven on the schema only — 13 trained cases and
one val case, no case the corpus never held; nothing constrains the action
choice; base 27B and 27B + adapter are unscored on bench and holdouts; Phase B
judge unwired; no composed address trace (N-86, N-109, N-110); 52 corpus rows
still carry brief revision 1; the exact `modal run` line is in neither log
(`Unverified:`); both check logs predate the banner fix, so their text still
claims every case trained; nothing committed.

## What this is

A store support agent whose replies are gated on database state. A separator
between "the model's claim" and "what actually happened". Locked stack: a 27B
Qwen3.8 student tuned 4-bit with Unsloth on Modal, DeepSeek 4.1 Flash as teacher
and judge, served by Modal vLLM, inside the $30 Modal credit.

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
  constrained decoding. One prompt per mode names the contract; free text keeps
  the teacher prompt unchanged.

## Next

1. Stage 4: wire the Phase B judge, then evaluate base 27B and 27B + adapter on
   the frozen bench and both holdouts, and report honestly. Bench scenarios are
   now seen; the two holdouts are the outside check.
2. Score the composed mode, not just its shape: the action choice has no number
   yet, and the 20 bench composed cases test the renderer.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `docs/COMPOSED-MODE.md` for the composed promise, the failure, the fix and the
  limit that travels with 14/14.
- `out/bench-results.json` after `python -m shopagent.cli bench`.
