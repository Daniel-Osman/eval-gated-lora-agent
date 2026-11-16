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

Section 4's deploy and demo boxes were run and recorded on 2026-11-16 in
`NOW.md` (400 in 0.96 s warm and 4.85 s cold, 200 in 319 s for N-100, 448
tests). They are not re-run here; this turn's one in-flight task is section 5.

## 5. Publish the served artifact and the adapter to a private HF repo

Owner approval, 2026-11-16: private repo only, HF account read from the Modal
secret `scenesift-hf`, nothing public, no commit. Plan reviewed by
`helm/plan-reviewer` before code: REVISE with six findings. The fail-closed
privacy rule, the ledger-before-code rule, explicit `token=` on every authenticated
client, a read-back that asserts rather than prints, the deprecated `hf_transfer`
backend, and the card's limits were all folded in.

- [x] `serve/export_hf.py`, one new file, one app
      `eval-gated-lora-agent-export`, CPU only, `retries=0`, Volume mounted
      read-only at `/vol`, `secrets=[from_name("scenesift-hf")]`:
      `probe()` (whoami, account type, orgs, `sorted(os.environ)` key names only,
      both directories listed), `export()` (fail closed unless private, create,
      upload root and `adapter/`, write the model card, return ids and sizes),
      `verify()` (`model_info`, `list_repo_files`, per-file sizes, anonymous
      denial, asserts the two weight sizes), `main()`.
- [x] Privacy is enforced, not configured: the function refuses
      `private is not True` before `create_repo`, and re-reads the repo after
      creation, aborting before the first artifact upload if it is not private.
- Acceptance: a repo the owner cannot make public by accident; probe names only,
      never a value; the token never printed, returned or written; the verified
      total equals 18,568,390,152 + 467,062,560 bytes plus the small files; an
      anonymous read is refused.
- Verify: `$env:PYTHONIOENCODING='utf-8'; python -m modal run
      serve/export_hf.py --probe-only`, then `python -m modal run
      serve/export_hf.py`, then `python -m modal run serve/export_hf.py
      --verify-only <repo_id>`; plus `python -m unittest discover -s tests -q`,
      `git diff --check`, `git status --short`, and `python -m modal billing
      summary --for "this month" --json` before and after.
- Files: `serve/export_hf.py`, this ledger, `NOW.md`.
- Closed, upload: `Daneil960/eval-gated-qwen3.8-27b`, `private True`, 16 files,
      `total bytes 19,075,486,047` = 18,588,402,779 + 487,076,061 + 5,637 card
      + 1,570 `.gitattributes`; commits `42b45b5` (quantized), `a54a3dd`
      (adapter), `417d1ff192989b5fa431397db79b95c2a3fad5cd` (card and head).
- Closed, verify: `python -m modal run serve/export_hf.py --verify-only
      Daneil960/eval-gated-qwen3.8-27b` -> `ok private is True`;
      `ok list_repo_files agrees with model_info on 16 files`;
      `ok model.safetensors is 18,568,390,152 bytes`;
      `ok adapter/adapter_model.safetensors is 467,062,560 bytes`;
      `ok every one of the 15 source files is in the repo`;
      `ok anonymous read refused with HTTP 401`.
- Closed, probe: account `Daneil960`, type `user`, orgs `none`, `HF_TOKEN`
      present in the key list, both `*_dir_present` true.
- Closed, gates: `Ran 448 tests`, `OK`. Spend before the export **18.17068511**,
      after **18.68353523**, delta **0.51285012** metered, 0.00 billed. No commit.
- [x] Record the publication, docs only: `docs/SERVING.md` "Published artifact"
      after the deployed-endpoint record (what, where, the private flag, the file and
      byte summary, the verify command and its five checks, the metered delta, the
      secret-only token rule); `EVIDENCE.md` one measurements row and one Verdict
      line; `README.md` one stage-6 sentence; `NOW.md` the verify command and result;
      this ledger.
- Acceptance: every figure in the record is one of the closed readings above, and
      the record claims no quality the artifact does not have.
- Verify: `python -m unittest discover -s tests -q`, `git diff --check`,
      `git status --short`, `(Get-Content NOW.md).Count`.
- Files: `docs/SERVING.md`, `EVIDENCE.md`, `README.md`, `NOW.md`, this ledger.
- Closed, record gates: `python -m unittest discover -s tests -q` ->
      `Ran 448 tests in 14.242s`, `OK`, exit 0; `git diff --check` exit 0, no
      output; `NOW.md` 100 lines. No code, no deploy, no upload, no commit.
- Not done, so not ticked: section 4's deploy and live-demo boxes, and any commit.
- Two brief shapes did not survive the client: `secret=` is `secrets=[...]`, and
      `--probe-only true` is `--probe-only` / `--no-probe-only`. `hf_transfer` is
      deprecated upstream, so the image pins `huggingface_hub[hf_xet]==2.4.0`.
