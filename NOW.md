# NOW - eval-gated-lora-agent

Status: stage 3 gated, not yet accepted - the fresh-context verdict is
outstanding, and a PASS is what accepts it. Stage 4 is **measured, not accepted**:
base 27B and 27B + adapter each answered all 284 frozen cases; Phase B scored
every Phase A approval. Base: 101 approved / 141 rejected / 39 escalated / 3 no
draft, 109 plan errors, 1557 s, est. `$1.10`. Adapter `full-20261114T201900Z`
(186 steps, `run_status ok`): 233 / 31 / 18 / 2, 40 plan errors, 1805 s, est.
`$1.28`. Phase A + B: base 56 / 186 / 39 over 101 judge calls; adapter 226 / 38 /
18 over 233; 0 invalid, 0 unavailable; 61.4 s and 116.6 s.

Stage 5 slice 1 **settles the false-approval metric**, measured, not accepted:
`gpt-6-luna` adjudicated every Phase A-approved draft blind to kind, expectation
and gate code. Base 101 approved: adversarial 36 -> 15 supported / 2 unsupported
/ 19 incomplete, honest 65 -> 38 / 5 / 22. Adapter 233 approved: adversarial 118
-> 116 / 0 / 2, honest 115 -> 105 / 4 / 6. 334 rows adjudicated over 335 HTTP
attempts, 0 invalid, 0 unavailable, 0 not scored; 152.7 s. The old case-kind rule
scored 154 approvals as false approvals; the adjudicator finds 2 and 0 unsupported
claims, and base's 19 incompletes are tool plans sent as replies. That reading
replaces the caveat in `EVIDENCE.md` stage 5, `README.md` and `docs/GATE.md`.

Two rows were ruled by hand. `N-102` is a false positive: the state holds one
shipment, so "the latest" is true - with it base 5/65 and adapter 4/115, without
it base 4/65 (6.2 %) and adapter 3/115 (2.6 %). `N-81` is label-unstable: both
students sent the same sentence on the same log and the labels split, base
`incomplete`, adapter `unsupported`; it stays in the adapter's totals, flagged
unstable, so it carries no verdict. 363 tests OK and `git diff --check` clean on
this tree; all eleven Modal estimates total `$5.4430`, metered `$11.10346860`,
billed `$0.00`. Nothing committed.

Next: stage 5 iteration runs. Repeat the adjudication in a fresh session over both
artifacts, so `unstable` becomes a measured rate instead of one hand ruling, then
read `unsupported` against the control: 5/65 (7.7 %) base and 4/115 (3.5 %)
adapter as run, 4/65 (6.2 %) and 3/115 (2.6 %) with `N-102` ruled out.

Verified 2026-11-15: `python -m unittest discover -s tests -q` exit 0
(`Ran 363 tests`, `OK`, 26 of them `tests/test_adjudicate.py`); `git diff --check`
exit 0; `python -m shopagent.cli adjudicate --concurrency 8` exit 0
-> 334 rows adjudicated over 335 HTTP attempts (base 102, adapter 233), 0 invalid,
0 unavailable, 0 not scored, 152.7 s, writes `out/adjudicated-base.jsonl`,
`out/adjudicated-adapter.jsonl` and `out/adjudicated.manifest.json`; the counts
here and in `EVIDENCE.md` were read back from `out/adjudicated.manifest.json` and
`out/adjudicate-run.txt`; `verify-frozen` still OK. Earlier verified runs, back to
2026-11-13, sit in `HISTORY.md` under Verified history.

## Next

1. Stage 5 iteration runs: repeat the adjudication in a fresh session over both
   artifacts and diff the labels against `out/adjudicated-*.jsonl`, so `unstable`
   (now one hand-ruled row, N-81) becomes a measured rate. Then fix the strictness
   the ruling exposed: N-102 shows the adjudicator asks for a comparison row that
   a single-shipment state cannot supply.
2. Stage 5 or 6: run the two holdouts with Phase B on, or serve the student on
   Modal vLLM with the browser demo. The holdouts are the outside check, and no
   holdout row has been touched.
3. Score the composed mode, not just its shape: the action choice has no number
   yet, and the bench composed cases test the renderer.
4. The two adapter adversarial `incomplete` rows (A-62, A-84: status replies that
   give `label_created` and tracking but not where the parcel is) need a ruling:
   they are adequacy, not false claims, and the corpus's marking guide can say
   whether a label-created status must say the parcel has not shipped.

## Open

**The case-kind rule counts a correct refusal as a false approval.** Settled as a
measurement, not deleted: the adjudicator (`out/adjudicated-*.jsonl`, manifest,
`shopagent/adjudicate.py`) replaces the kind rule for reading the drafts. It
finds 2 unsupported claims in 36 approved base adversarial rows and 0 in 118
adapter rows, where the kind rule counted all 154. Base's 19 adversarial
incompletes and 22 honest incompletes are tool plans sent as replies - a real
base defect the false-approval number hid. Two rows carry hand rulings: `N-102` is
a false positive (one shipment in state, so "the latest" is true), and `N-81` is
label-unstable (one sentence, one log, base `incomplete` against adapter
`unsupported`) and stays in both adapter readings flagged unstable, so the honest
control reads 5/65 (7.7 %) and 4/115 (3.5 %) as run, 4/65 (6.2 %) and 3/115
(2.6 %) with N-102 ruled out, and the accepted findings there are two. Limits:
`gpt-6-luna` is a model, not a human audit; rates are over adjudicated rows, with
invalid and unavailable answers counted separately; the normal-case control is
disagreement on student drafts; every case is seen data; one pass only, so label
stability is unmeasured beyond N-81; and `SPEC.md` §7's "zero false approvals on a
held-out adversarial set" stays **unmeasured**: the holdouts are untouched. The
repo records no author for `bench/holdout/review-02.jsonl`, so "luna wrote holdout
2" is the brief's fact, not this repo's.

Also open: the composed student is proven on the schema only - 13 trained cases
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

`HISTORY.md` holds what this project is, its decisions, the verified-run tail and
the evidence index.
