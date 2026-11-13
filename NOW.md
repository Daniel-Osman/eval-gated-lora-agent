# NOW - eval-gated-lora-agent

Status: benchmark grown and frozen — 284 cases (134 normal, 150 adversarial),
20 composed; 171 tests OK; gates green. `SPEC.md` holds Daniel's locked stack:
Qwen3.8-27B 4-bit QLoRA (Unsloth) on Modal A100 40 GB, Modal vLLM on an A10,
$30 Modal credit; teacher and judge call `opencode-go/deepseek-v4.1-flash`.
Composed mode renders status, refund, reshipment and address from ids only;
address is confirm-or-refuse and cannot originate a change. Free-text limit
unchanged: 12/32 caught, 20 false approvals on unseen phrasing. The 79-case
paraphrase suite was authored with knowledge of the gate, so its 55/55 is not
generalisation.
Next: teacher data pipeline against the frozen 284-case bank, then Modal
training.
Verified 2026-11-11: `validate` 284 valid; `freeze` 284 and `verify-frozen` OK;
bench oracle 205/205, scripted 205/205, sabotage 95/95 original + 55/55
paraphrase, honest 229/229, holdout 32/32 caught (30/32 exact); `unittest` 171
OK; `leakcheck --corpus data/train` CLEAN; `git diff --check` clean. Digests:
original 40-case `34833a08bbe4ff761cfdc7aad311fbe468a81061d6166bf83eee342a8c0a341f`;
all-cases (284) `a76a732fa9b1f0126a76a2465f00e1b9bee6eca66e3fd22fa993598621c66364`;
holdout `47034a0f…c180` (32 rows, training-excluded).
Open: free-text generalisation 12/32; composed address has no text; Phase B
judge unwired (stub re-pointed to DeepSeek in stage 4); no model trained; no
teacher data; leak corpus empty; nothing committed; fresh-context PASS pending.

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
