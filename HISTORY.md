# HISTORY - eval-gated-lora-agent

Overflow from `NOW.md`: what the project is, the decisions it holds, and where
the evidence lives. `NOW.md` keeps the current status, next step and open items.

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
  constrained decoding. One prompt per mode names the contract; the free-text
  prompt is now the teacher prompt plus an output-shape paragraph, superseding the
  earlier "unchanged" decision.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `docs/COMPOSED-MODE.md` for the composed promise, the failure, the fix and the
  limit that travels with 14/14.
- `EVIDENCE.md` for the measured runs, spends and gate outputs.
- `out/bench-results.json` after `python -m shopagent.cli bench`.

## Verified history

Moved out of `NOW.md` to keep it inside the line budget. `NOW.md` carries the
current verification line only.

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
