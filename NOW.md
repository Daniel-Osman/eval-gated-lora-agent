# NOW - eval-gated-lora-agent

Status: stage 3 complete and gated, not yet accepted — the fresh-context verdict
is outstanding, and a PASS is what accepts it — SFT data, one adapter, and a load
check. 178 records, 289,759 rendered tokens, 17,748 supervised across both splits
(6.1 %); QLoRA on a Modal A100 40 GB, 3 epochs, 120 steps, loss 0.3892 → 0.1545
(0.7537 → 0.1571), 1045.8 s wall, 21.22 GB peak, **estimated $0.7388** (compute
only), adapter on the Volume at `/vol/runs/full-20261113T180308Z/adapter`. The
three runs estimate **$1.0423** together; the metered figure for this month is
**$5.99564250**, billed **$0.00** (compute $2.945, storage $3.013, egress $0.033;
covered by credits). Load check passed the weights (116,727,808 parameters,
`lora_B` `torch.equal`) and free text N-31 gave a plausible reply, but composed
N-105 failed: the student wrote the free-text plan list, so the composed
validator refused with `unknown action None`. **Composed mode — the project's
safety claim — has no working student output yet**: 14 composed rows against 164
free-text rows under one system prompt taught the dominant shape; the format
fix is an open decision, not a patch. The gate stayed strict by Daniel's call:
the 71 `gate:REJECTED` and 28 `gate:ESCALATED` refund replies remain rejects, no
brief softening, no re-run. Gates green on this tree: 255 tests OK; `validate`
284 valid; `bench` exit 0; `leakcheck` CLEAN. Nothing committed.
Next: stage 4 — wire the Phase B judge and evaluate base 27B against 27B +
adapter on the frozen bench and both holdouts; decide the composed format.
Verified 2026-11-13: `python -m unittest discover -s tests -q` exit 0
(`Ran 255 tests`, `OK`); `python -m shopagent.cli validate` exit 0 (`284 cases:
134 normal, 150 adversarial`, `valid`); `python -m shopagent.cli bench` exit 0
(holdout 1 32/32 caught, 30/32 exact; holdout 2 30/40 caught, 16/20 honest,
43/60 exact; 229/229 honest approved); `python -m shopagent.cli leakcheck
--corpus data/train` exit 0 (178 sessions, 92 holdout replies, CLEAN, closest
pair `N-79 ~ shard-000.jsonl:63` 0.75); `python -m shopagent.sft` 178 rows,
digest `5cab4f46…`; `full-20261113T180308Z` `run.json` `status: ok`;
`out/modal-check.log` `EXIT=0`.
Open: composed output shape (separate prompt per mode, rebalanced composed slice,
or constrained decoding — Daniel's call); the training loss was measured on
16,230 supervised tokens (5.6 % of 289,759; stored masks hold 17,748 across both
splits) and that volume is untested; all 18 val rows are seen scenarios, so
val loss watches overfit only and no bench score is generalisation — the two
holdouts are the outside check; base 27B and 27B + adapter are unscored; 52
corpus rows still carry brief revision 1; the exact `modal run` line is not in
either log (`Unverified:`); Phase B judge unwired; nothing committed.

## What this is

A store support agent whose replies are gated on database state. A separator
between "the model's claim" and "what actually happened". Locked stack: a
27B Qwen3.8 student tuned 4-bit with Unsloth on Modal, DeepSeek 4.1 Flash as
teacher and judge, served by Modal vLLM, inside the $30 Modal credit.

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

## Next

1. Stage 4: wire the Phase B judge, then evaluate base 27B and 27B + adapter on
   the frozen bench and both holdouts, and report honestly. Bench scenarios are
   now seen; the two holdouts are the outside check.
2. Decide the composed output format: a separate system prompt per mode, a
   rebalanced composed slice, constrained decoding, or something else. No patch
   until Daniel chooses.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `docs/COMPOSED-MODE.md` for the composed promise and the missing student
  output.
- `out/bench-results.json` after `python -m shopagent.cli bench`.
