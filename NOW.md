# NOW - eval-gated-lora-agent

Status: spec locked, harness done. `SPEC.md` holds Daniel's locked decisions:
Qwen3.8-27B 4-bit QLoRA (Unsloth) on Modal A100 40 GB, Modal vLLM serving on an
A10, $30 Modal credit.
Stages 1-6 are in the spec. Composed mode is ids-only; free-text limit measured.
Teacher and judge call `opencode-go/deepseek-v4.1-flash` over the OpenCode Go
subscription API; Modal's $30 credit is the only metered spend.
128 tests OK; 67 cases frozen (40 original + 27 paraphrase) plus 32 holdout rows;
holdout caught 32/32, exact 30/32 (remediation-targeted); original 10/10,
paraphrase 15/15; honest 52/52. On an unseen 32-reply set free-text caught only
12/32 (20 false approvals), so composed mode — ids in, row out, one shipment, no
text input — is the shipped answering path; it does not support address text.
Next: teacher data pipeline (grow the benchmark to 200+ and freeze it first),
then Modal training.
Verified 2026-11-09: 128 tests OK; `verify-frozen` OK; original 40-case digest
`34833a08bbe4ff761cfdc7aad311fbe468a81061d6166bf83eee342a8c0a341f`; all-cases
digest `ebe5b5a92be4b384acd106677239df4cc0c320607e1a2061bf64e7cf6faeda53`;
holdout `review-01.jsonl` 32 rows, training-excluded, digest `47034a0f…c180`.
Open: free-text generalisation 12/32; no composed address text; Phase B judge
unwired (stub still names a local endpoint, re-pointed to DeepSeek in stage 4);
no model trained; no teacher data; benchmark at 67 of 200+ cases; leak corpus
empty; nothing committed; final fresh-context PASS pending.

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
- Phase B is a validated JSON contract with a strict enum; invalid or missing
  output fails closed.
- Three attempts, then `human_queue`. No infinite retry.

## Next

1. Daniel reviews `SPEC.md`; no commit before he says go.
2. Build the teacher pipeline; run `leakcheck` on every generated batch, against
   both the tickets and the 32 holdout replies. Freeze the grown benchmark
   first.
3. Unsloth QLoRA run for Qwen3.8-27B on Modal A100 40 GB; record loss, VRAM and
   wall time, weights cached on a Modal Volume.
4. Gate the trained model through this same benchmark and holdout, and report
   against base 27B, honestly.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `out/bench-results.json` after `python -m shopagent.cli bench`.
