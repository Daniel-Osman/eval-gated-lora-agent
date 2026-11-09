# NOW - eval-gated-lora-agent

Status: M-01d. Composed mode is ids-only; free-text limit measured. 128 tests OK;
67 cases frozen (40 original + 27 paraphrase) plus 32 holdout rows; holdout
caught 32/32, exact 30/32 (remediation-targeted); original 10/10, paraphrase
15/15; honest 52/52. On an unseen 32-reply set free-text caught only 12/32 (20
false approvals), so composed mode — ids in, row out, one shipment, no text
input — is the shipped answering path; it does not support address text. Next:
teacher data pipeline, then Colab training.
Verified 2026-11-09: 128 tests OK; `verify-frozen` OK; original 40-case digest
`34833a08bbe4ff761cfdc7aad311fbe468a81061d6166bf83eee342a8c0a341f`; all-cases
digest `ebe5b5a92be4b384acd106677239df4cc0c320607e1a2061bf64e7cf6faeda53`;
holdout `review-01.jsonl` 32 rows, training-excluded, digest `47034a0f…c180`.
Open: free-text generalisation 12/32; no composed address text; Phase B judge
unwired (contract + stub only); no model trained; no teacher data; leak corpus
empty; nothing committed; final fresh-context PASS pending.

## What this is

A store support agent whose replies are gated on database state. A separator
between "the model's claim" and "what actually happened". Free-tier only: 7B
QLoRA on a T4, local GGUF, 3B local judge, zero cash.

## Decisions held

- Benchmark before training data, frozen by content hash. `bench` refuses to
  score a modified benchmark or a modified holdout.
- The referee reads tables, never the agent's log. A forged audit row fails the
  gate (test proves it).
- The gate's adversarial suite is the acceptance test: a claim of a refund after
  a declined refund must be rejected. The reviewer holdout is the outside check
  on that suite.
- Phase B is a validated JSON contract with a strict enum; invalid or missing
  output fails closed.
- Three attempts, then `human_queue`. No infinite retry.

## Next

1. Daniel reviews Phase 1; no commit before he says go.
2. Build the teacher pipeline; run `leakcheck` on every generated batch, against
   both the tickets and the 32 holdout replies.
3. Unsloth QLoRA notebook for Colab/Kaggle; record loss, VRAM and wall time.
4. Gate the trained model through this same benchmark and holdout, and report
   against the teacher baseline, honestly.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `out/bench-results.json` after `python -m shopagent.cli bench`.
