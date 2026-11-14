# NOW - eval-gated-lora-agent

Status: stage 3 gated, not yet accepted — the fresh-context verdict is
outstanding, and a PASS is what accepts it. Stage 4 is **measured, not accepted**:
base 27B and 27B + adapter each answered all 284 frozen cases, and Phase B then
scored every row Phase A approved. Base: 101 approved / 141 rejected / 39
escalated / 3 no draft, 109 plan errors, 1557 s, est. `$1.10`. Adapter
`full-20261114T201900Z` (186 steps, `run_status ok`): 233 / 31 / 18 / 2, 40 plan
errors, 1805 s, est. `$1.28`. Phase A + B: base 56 / 186 / 39 over 101 judge
calls; adapter 226 / 38 / 18 over 233; 0 invalid, 0 unavailable; 61.4 s and
116.6 s. 337 tests OK and `git diff --check` clean on this tree; all eleven Modal
estimates total `$5.4430`, metered `$11.10346860`, billed `$0.00`. Nothing
committed.

Next: settle the false-approval metric, or go to stage 5 (reuse the harness
against the holdouts) or stage 6 (serve on Modal vLLM with the browser demo). The
metric comes first: no safety number here can be read until it is fixed.

Verified 2026-11-15: `python -m unittest discover -s tests -q` exit 0
(`Ran 337 tests`, `OK`, 16 of them `tests/test_judgebench.py`). Verified
2026-11-14 on this tree:
`python -m shopagent.cli judge-bench --rows out/bench-base-284.jsonl
--student base` exit 0 → 284 rows, 101 judged, 0 not-scored, 0 invalid, 0
unavailable, A+B 56 / 186 / 39, 61.4 s; the adapter twin → 233 judged, 0
not-scored, 0 invalid, 0 unavailable, A+B 226 / 38 / 18, 116.6 s; both write
`out/judged-<student>.jsonl` + manifest; replay fidelity checked over both
artifacts: 281 and 282 rows replayed, 0 envelope mismatches, 0 Phase A
mismatches; `python -m shopagent.cli validate` prints
`284 cases: 134 normal, 150 adversarial … valid`; `git diff --check` exit 0, no
output. Both bench artifacts carry `benchmark.verified: true` against
`cases_digest a76a732fa9b1f012…` and 2816-token windows. Verified 2026-11-13 on
the pre-slice tree:
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

## Next

1. Settle the false-approval metric: the case-kind rule scores a correct refusal
   as a false approval, so the judged artifacts cannot yield a false-claim rate
   until the expectation per row is the claim, not the case kind.
2. Stage 5 or 6: run the two holdouts with Phase B on, or serve the student on
   Modal vLLM with the browser demo. The holdouts are the outside check, and no
   holdout row has been touched.
3. Score the composed mode, not just its shape: the action choice has no number
   yet, and the bench composed cases test the renderer.

## Open

**The case-kind rule counts a correct refusal as a false approval.** Many
adversarial tickets can be answered truthfully, so the bench sample holds true
statements, not false claims, and the rule over-counts. What the numbers support:
the adapter follows both output contracts where base emits tool plans as replies
(109 plan errors against 40), and the judge moved 45 base rows off approval
against 7 adapter rows. `SPEC.md` §7's "zero false approvals on a held-out
adversarial set" stays **unmeasured**: the holdouts are untouched and every bench
case is seen.

Also open: the composed student is proven on the schema only — 13 trained cases
and one val case, no case the corpus never held, and nothing constrains the
action choice; base 27B and 27B + adapter are unscored on the two holdouts;
Phase B is measured on the bench but not on a holdout, and its agreement there is
partly self-confirming (teacher and judge are one model); the judge's own 45 and
7 moved rows have not been hand-read; no composed address trace (N-86, N-109,
N-110); 52 corpus rows still carry brief revision 1; `train/bench_app.py`'s
docstring still names the old adapter run id `full-20261113T205627Z` as its
example, where these artifacts used `full-20261114T201900Z`; the exact
`modal run` line is in neither bench log (`Unverified:`); both check logs predate
the banner fix, so their text still claims every case trained; nothing committed.

## History

`HISTORY.md` holds what this project is, the decisions it holds, and the evidence
index.
