# Training data

Empty on purpose. No synthetic session may be generated until
`python -m shopagent.cli verify-frozen` passes.

Rules:

- Sessions land here as JSONL, one object per line, strict format: user prompt,
  tool call, tool output, final reply.
- Raw teacher output goes to `raw.jsonl`; the validated set goes to
  `train.jsonl`. They are kept apart so the filter is auditable.
- Never copy a benchmark ticket into a seed, a prompt, or a teacher
  instruction. `bench/` is read-only from the pipeline's point of view.
- Run `python -m shopagent.cli leakcheck --corpus data/train` after every batch.
  Any exact or near match fails the batch.
- Files here are gitignored; the folder and this guide are not.

Planned pipeline (not built yet):

1. 15-20 hand-written seed tasks covering status, refund, reshipment, address,
   refusal and multi-step tickets.
2. Schema-driven scenario expansion (order ids, amounts, fault types) with a
   fixed seed per batch.
3. Strict session assembly against the real tool layer: every tool call must
   execute, every reply is checked against final state.
4. Validators reject malformed or non-executable sessions; keep rejects in
   `rejected.jsonl` with the reason.
5. Leak check, then train.
