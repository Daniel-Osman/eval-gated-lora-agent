# NOW - eval-gated-lora-agent

Status: **frozen 2026-11-15.** Training stops here: no third iteration, no
further corpus growth. Stages 1-4 done; stage 5's further iteration runs were not
attempted by decision; stage 6 serving is next. The adapter is
`full-20261114T201900Z` - 27B 4-bit QLoRA, 263 written rows from 178 teacher
traces, 40 % composed per epoch, 186 steps, loss 0.2051 -> 0.0840, est. `$0.8592`,
at `/vol/runs/full-20261114T201900Z/adapter` on the Modal Volume, not in this
repo. **No milestone here records a fresh-context PASS**: milestone 2, milestone
3, milestone 4, milestone 4b, stage 4 and the stage-5 adjudication slice are each
verified on green commands and unaccepted by the rule that an author never signs
its own stage. Daniel closed training here on 2026-11-15; that is an owner's
decision, not a verdict.

One line per stage: harness - 171 tests, 284 cases and a 32-case holdout frozen.
Teacher - 178 admitted traces of 284, gated, manifested, leak check CLEAN. Train -
three runs, the third frozen. Measure - Phase A base 101 / adapter 233 approvals;
Phase B moved 45 and 7 rows; the adjudicator found 0 unsupported claims in 118
approved adapter adversarial drafts against 2 in 36.

Verified 2026-11-15: `python -m unittest discover -s tests -q` exit 0
(`Ran 363 tests`, `OK`); `python -m shopagent.cli verify-frozen` -> `benchmark
frozen: OK`; `git diff --check` exit 0, no output. The README rewrite was checked
against `HEAD:README.md`: 34 of 34 numbers-table rows verbatim, 4 named bullet
replacements, 0 numeric or artifact tokens lost. Earlier verified runs, back to
2026-11-13, sit in `HISTORY.md` under Verified history.

## Next

Stage 6, serving: **the feasibility slice is measured (2026-11-15); the deploy and
the browser demo are not built.** The quantized artifact loads on the locked A10
at `max_num_seqs 8` — 19.03 of 22.06 GiB, 4,437 KV tokens, 1.08x concurrency at
4k, 22.84 output tok/s, 189 s engine init, 14/14 composed. Batch 256 fails and the
widths between 8 and 256 are untested. One decision is open before any deploy:
keep the A10 as a single-user demo, or move the serving line to a larger GPU.
That is an infra change and Daniel's call. Evidence and every number:
`docs/SERVING.md`, `out/serve-*.log`.

## Open

**The corpus is the constraint, and it stays short of the plan.** 178 traces
against the 1,000+ sessions `SPEC.md` §3 targets. The freeze closes that gap
without filling it: hand-written seeds and schema-driven expansion are unwritten,
and the shortfall is why further iteration would chase noise rather than learn.

**The held-out metric is unmeasured.** `SPEC.md` §7's zero false approvals on a
held-out adversarial set has no student run, no Phase B judgeship and no
adjudication behind it. Both holdouts are untouched by every student run; the
gate's own scores there (32/32 remediation-targeted, 43/60 reported and not
gated) are gate results, not this criterion.

Also open: the accuracy columns were never computed (action, final-state,
reply-fact); Phase B's 45 and 7 moved rows are unread; the composed action choice
is unscored and no composed address trace exists; 52 corpus rows still carry brief
revision 1; `train/bench_app.py`'s docstring names the old run id
`full-20261113T205627Z`; both check logs predate the banner fix, so their text
still claims every case trained; nothing committed.

## History

`HISTORY.md` holds what this project is, its decisions, the freeze decision, the
superseded next steps, the verified-run tail and the evidence index.
