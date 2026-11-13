# Training data

Generated 2026-11-13: three shards hold 178 admitted traces. No synthetic
session may be generated until `python -m shopagent.cli verify-frozen` passes.

Layout written by `python -m shopagent.cli teacher`:

- `train/shard-NNN.jsonl` — admitted traces only, 64 rows per shard. A trace
  reaches a shard only after the gate returns `APPROVED`.
- `raw/raw.jsonl` — every teacher response verbatim, beside the parsed row, so
  a parse bug is recoverable without regenerating.
- `rejects/rejected.jsonl` — one row per rejected case with its reason code
  (`plan_mismatch`, `state_mismatch`, `facts_missing`, `gate:REJECTED`,
  `leak:<case>` and so on). Never part of the corpus.
- `manifest.json` — counts, per-shard digest, model, gateway, session id,
  prompt-template digest, timestamp and requests made.

Rules:

- One object per line. Fields: `messages` (system, customer message, plan, tool
  outputs, reply), `plan`, `target` and `provenance`. A row without provenance
  is not trainable.
- Composed rows carry the plan only; the sentence is rendered by code and the
  student never learns it. Free-text rows carry the reply that passed Phase A.
- Never copy a benchmark ticket into a seed, a prompt, or a teacher
  instruction. The teacher is briefed from the case's title, ids, amounts and
  fault modes. `bench/` is read-only from the pipeline's point of view.
- A generated row that collides with a bench ticket or a holdout reply is
  dropped and counted in `manifest.json`; it is never written to a shard.
- Run `python -m shopagent.cli leakcheck --corpus data/train` after every batch.
  Any exact or near match fails the batch.
- Files here are gitignored; the folder and this guide are not.
