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
