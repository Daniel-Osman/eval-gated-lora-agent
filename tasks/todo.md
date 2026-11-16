# Todo — the web-layer check and the portfolio piece, after review FAIL

One task in flight. Acceptance, verify command and files per task; a box ticks only
with the output that closed it quoted here.

Spec: the parent session's brief of 2026-11-16 (record the landed web-layer preset
check, then write the stage-6 portfolio piece) plus `NOW.md` and
`docs/SERVING.md`. This turn's acceptance is the reviewer's four findings,
2026-11-16, quoted per slice below.

Closed previously: stage 6 slice 4, the RPC door. Landed as commit `6073ba7`;
`Ran 437 tests` `OK`; the record is `EVIDENCE.md` stage 6.

---

## 1. Record the slice and write the portfolio piece (landed, then failed review)

- [x] `docs/SERVING.md`: the web-layer refusal, its two timings (0.96 s warm,
      4.85 s cold health), the app-log proof and the fail-open rule; the history
      that the earlier behaviour woke the A10; table rows; spend.
- [x] `NOW.md`: status, stage-6 line, Verified, Next, Open, and the
      no-fresh-context-PASS paragraph.
- [x] `EVIDENCE.md` stage 6: table row, prose line, spend and gates.
- [x] `README.md`: the web-layer refusal, numbers rows, 443-test line, billing
      scope, the `docs/PORTFOLIO.md` link.
- [x] `docs/PORTFOLIO.md`: eight sections, prose, no hype, no em dashes.
- [x] `tasks/todo.md`: this ledger.
- Closed on `python -m unittest discover -s tests -q` -> `Ran 443 tests`, `OK`.
- Failed review: four findings. The reviewer's own words replace the old spend
  acceptance line: the uncaptured month read is unverifiable and drops out;
  `N-81` stays counted and flagged unstable; a later read is not part of an
  earlier record.

## 2. Fix 1: a failed health response must not authorise a refusal

- [x] `serve/demo.py`: cache only on a fully healthy read, meaning HTTP 200, a
      parsed body with `ok is True`, and a non-empty `presets` list whose every
      element is a dict with a non-empty string `id`. A 503, an `ok:false`, a
      partial list or a malformed element caches nothing and returns `None`, so
      the caller forwards (fail open). The `/health` route caches on the same
      rule and still passes body and status through. Keep the single-refresh
      rule and the cold-cache second read.
- [x] `tests/test_demo.py`: four route tests with the fake transport - (a) health
      503 with a valid-looking body forwards a good preset, `FakeAgent.calls == 1`,
      nothing cached, and it forwards again after; (b) health 200 with `ok:false`
      forwards; (c) health 200 with a proper list refuses a bad id 400 and the
      good one forwards with no second health read; (d) a partial or malformed
      list caches nothing and forwards. Assert the 503 body and status still pass
      through `/health` unchanged.
- Files: `serve/demo.py`, `tests/test_demo.py`.
- Acceptance: quote the reviewer's finding 1.
- Verify: `python -m unittest discover -s tests -q`. No model deploy.
- Closed: `Ran 448 tests in 12.360s`, `OK`; five new route tests, and four of the
      five print `AssertionError: 400 != 200` against the pre-fix read rule
      (runtime patch, no file change). `python -m unittest tests.test_demo -q` ->
      `Ran 41 tests`, `OK`. No model deploy.

## 3. Fixes 2 to 4: the portfolio and the history

- [x] `docs/PORTFOLIO.md` line 73: raw control counts first, 5 of 65 base and
      4 of 115 adapter; then the narrower reading, `N-102` ruled a false positive
      and removed, `N-81` flagged unstable and still counted, 4 of 65 (6.2 %) and
      3 of 115 (2.6 %). No "two drafts in a hundred".
- [x] `docs/PORTFOLIO.md` *What is not proven*: 178 traces against `SPEC.md`
      section 3's 1,000+ target with stage 5's further runs not run by decision
      (the stage-5 adjudication slice did run); composed mode cannot originate a
      change the rows do not carry, so an address change needs the free-text path,
      whose quality is unmeasured; no milestone carries a fresh-context PASS, each
      is verified on green commands and none is accepted, because acceptance is
      the owner's call and no reviewer has signed one; and this slice's own review
      found four defects before this fix.
- [x] `docs/PORTFOLIO.md` *Cost* and `HISTORY.md` verified history: each read named
      with when it was taken, the two this repo took naming their command -
      `17.18067970` at the stage-6 record, `17.90068367` before the last cold
      forward call - and `17.97068375` named as the reviewer's read after that
      call, a read with no output retained here. Drop the uncaptured month read
      and its arithmetic. Say the counter moves with use and no snapshot is a
      bill. `HISTORY.md` keeps the later read out of the earlier record.
- Files: `docs/PORTFOLIO.md`, `HISTORY.md`, this ledger.
- Verify: `git diff --check`, `git status --short`.
- Closed: `git diff --check` exit 0, no output.
- [x] Correction, docs only: the uncaptured month read leaves every file it
      reached. Each place now carries a figure the repo can show: metered
      18.16068483 read at 21:51 on 2026-11-16 by `python -m modal billing
      summary --for "this month" --json`, output not retained, beside 17.18067970
      at the stage-6 record and 17.90068367 before the last cold forward call,
      and 17.97068375 for the reviewer's read. Arithmetic restated:
      18.16068483 − 14.78476210 = 3.37592273, of which 2.39591760 at the record
      and 0.98000513 after it. Files: `NOW.md`, `README.md`, `EVIDENCE.md`,
      `docs/SERVING.md`, this ledger.
- Closed: `rg -n "18\.01068374"` empty; `Ran 448 tests`, `OK`; `git diff --check`
      exit 0.

## 4. Gates and the live demo

- [ ] `python -m unittest discover -s tests -q` -> quote the line.
- [ ] `$env:PYTHONIOENCODING='utf-8'; python -m modal deploy serve/demo.py` ->
      time and URL.
- [ ] At 0 agent containers (`modal container list --json`), `POST /run
      {"scenario":"N-104"}` -> 200 within 900 s; `POST /run {"scenario":"N-999"}`
      -> 400 with no GPU engine line in `modal app logs eval-gated-lora-agent`.
- [ ] `git diff --check`, `git status --short`, `(Get-Content NOW.md).Count`.
- No commit. No model deploy.
