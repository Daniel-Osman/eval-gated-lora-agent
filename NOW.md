# NOW - eval-gated-lora-agent

Status: teacher corpus generated — 178 admitted traces of the frozen 284 cases
(164 free text, 14 composed), from a full pass plus a resumed pass after a brief
change; 106 rejects with reason codes; three shards `18ecdb3a…` / `89bacee4…` /
`7e5f1cb0…`; manifest carries `brief_revision` 2 (runs 1, 2); 975 requests,
1,389 s; 231 tests OK; gates green. `SPEC.md` holds Daniel's locked stack:
Qwen3.8-27B 4-bit QLoRA (Unsloth) on Modal A100 40 GB, Modal vLLM on an A10, $30
Modal credit; teacher and judge call `opencode-go/deepseek-v4.1-flash`.
Next: fresh-context review of the corpus checkpoint, then a decision on the
99-reject gate cluster, then hand-written seeds and expansion toward the 1,000+
sessions `SPEC.md` §3 targets, then Modal training.
Verified 2026-11-13: `teacher --out data --session-id teacher-20261113T011445Z`
exit 0 (284 attempted, 52 admitted, 232 rejected); the same command with
`--resume` exit 0 (232 re-run, 126 admitted; corpus 178 of 284); `validate` 284
valid; `bench` exit 0 (holdout 1 32/32 caught, 30/32 exact; holdout 2 30/40
caught, 16/20 honest, 43/60 exact); `unittest` 231 OK; `leakcheck --corpus
data/train` 178 sessions, 92 holdout replies, CLEAN, closest pair `N-79 ~
shard-000.jsonl:63` (0.75). Digests: shards `18ecdb3a…`, `89bacee4…`,
`7e5f1cb0…`; cases `a76a732f…c66364`; template `68464655…7070f5`; holdout
`47034a0f…c180`.
Open: 71 `gate:REJECTED` and 28 `gate:ESCALATED` are 99 of the 106 rejects —
replies that read a pre-existing refund or shipment as a new claim. Whether that
is a gate reading to accept, a brief to sharpen, or a Phase B job is Daniel's
call. Also open: the corpus mixes brief revisions (52 rows rev 1, 126 rev 2);
bench scenarios are now seen, so later bench scores are not generalisation — the
two holdouts are the outside check; composed address has no trace; free-text
generalisation still 12/32; Phase B judge unwired; no model trained; nothing
committed.

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

1. Fresh-context review of the teacher corpus checkpoint; no commit before
   Daniel says go.
2. Decide the 99-reject gate cluster (`gate:REJECTED` 71, `gate:ESCALATED` 28):
   accept the reading, sharpen the teacher brief, or defer to Phase B.
3. Grow the corpus: hand-written seed tasks and schema-driven expansion toward
   the 1,000+ sessions in `SPEC.md` §3, re-running `leakcheck` on every batch.
4. Unsloth QLoRA run for Qwen3.8-27B on Modal A100 40 GB; record loss, VRAM and
   wall time, weights cached on a Modal Volume.
5. Gate the trained model through this same benchmark and holdout, and report
   against base 27B, honestly. Bench scenarios are now seen; the two holdouts
   are the outside check.

## Evidence

- `README.md` numbers table, each row bound to a command.
- `docs/BENCHMARK.md` for case format, holdout shape and the freeze rule.
- `docs/GATE.md` for the rules, full code table, contract and retry semantics.
- `out/bench-results.json` after `python -m shopagent.cli bench`.
