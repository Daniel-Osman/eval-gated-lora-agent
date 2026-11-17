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
- **Independent review of this section, 2026-11-16: FAIL.** The reviewer's three
      findings and one record gap, quoted, against the tree it read (`76099ee`).
      The fixes below sit in the working tree and are **unproven pending the
      re-review**; this ledger claims no PASS and no holdout result.
      1. High — "the privacy check accepts a non-refusal (`serve/export_hf.py:416-428`).
         Today only HTTP 200 fails, so 503 (and any other status) is printed as
         'anonymous read refused'."
      2. High — "the model card ships a loader that silently drops the adapter
         (`serve/export_hf.py:133-143`). The card tells readers to use
         `AutoModelForCausalLM`, and this repo's own record (`docs/SERVING.md:109-115`,
         `serve/feasibility.py:666-670`) says that path loads without the trained
         adapter and gives no error."
      3. Medium — "verification compares names, not sizes (`serve/export_hf.py:409-414`)."
      4. Record gap — the milestone had no `Holdout:` line and no verdict.
- Fix 1, privacy allow-list. Red first: `tests/test_export_hf.py` ->
      `Ran 7 tests`, `FAILED (errors=9)`; the rule and its helper did not exist.
      Fix: `REFUSAL_CODES = frozenset({401, 403, 404})`; `_anonymous_status` sends no
      `Authorization` header, opens through `_NoRedirect` so a 3xx is never followed,
      and returns the code plus a 200-character body excerpt; any other code raises
      naming the code and the excerpt. Green: `Ran 7 tests`, `OK`; the 503, 200 and
      302 cases each raise, 401/403/404 pass with `anonymous read refused with HTTP
      <code>`. Negative proof after the fix: `REFUSAL_CODES` set to empty at runtime
      reddens the refusal test with `AssertionError: an anonymous read of
      someone/eval-gated-qwen3.8-27b returned HTTP 404, which is not a refusal: only
      [] count`, `FAILED (failures=3)`.
- Follow-up on fix 1, medium: the mocked opener could not tell whether `_NoRedirect`
      was installed — replacing `build_opener(_NoRedirect)` with `build_opener()` left
      all 18 tests green. Fixed in the working tree, unproven pending re-review: the
      test now starts a real server on `127.0.0.1:0`, answers the API path with a 307
      to `/refuse`, and answers `/refuse` with 401. Driven through `_anonymous_status`,
      the read returns 307, `/refuse` is never hit, and no `Authorization` header
      reaches the server. With the handler bypassed inside the test, the same chain
      answers 401, so the chain discriminates. `HUB_BASE` is now a module constant, so
      the test points the read at loopback and leaves production on the Hub. Mutation
      proof: `build_opener()` in the module reddens it with `AssertionError: 307 != 401`
      and `AssertionError: AssertionError not raised`; reverted, green.
- Follow-up on fix 1, medium: cleanup was unbounded. The helper used a synchronous
      `HTTPServer` with no accepted-socket timeout, so an incomplete request pinned
      `shutdown()` past the join (the reviewer measured "cleanup finished within 1
      second: False"). Fixed in the working tree, unproven pending re-review: the
      helper is now a `ThreadingHTTPServer` with `daemon_threads = True`, its handler
      sets a 0.5 s socket timeout after `super().setup()`, `handle_error` stays quiet,
      and the `finally` runs `shutdown()`, `server_close()`, `join(timeout=2)`. New
      test: one socket sends a partial request with no blank line and stays open; the
      helper still returns inside a second. Red shown by mutation: with the
      synchronous server and the socket timeout removed, that single test ran past a
      15 s limit and was killed with no output; restored, `Ran 20 tests`, `OK`.
- Fix 2, the card loader. Red first: `Ran 5 tests in 0.002s`, `FAILED (failures=3)`,
      including `AssertionError: 'Qwen3_5ForConditionalGeneration' not found in ...`.
      Fix: the block loads `Qwen3_5ForConditionalGeneration`, the architecture
      `docs/SERVING.md:114-115` records, rebinds `PeftModel.from_pretrained`, then
      asserts a non-empty `model.peft_config` and non-zero `lora_A.` and `lora_B.` in
      the first module ending `.self_attn.q_proj`; the sentences say PEFT zero-inits
      `lora_B`, so a dropped adapter is silent. The test execs the block parsed out of
      `MODEL_CARD`, with its import lines removed and fakes injected. Green: `Ran 12
      tests`, `OK`. One test was corrected after it failed for the wrong reason: it
      read the block's explanatory comment as code, so it now strips comment lines
      before refusing `AutoModelForCausalLM`.
- Fix 3, manifest by size. Red first: `Ran 18 tests`, three manifest failures,
      `AssertionError: AssertionError not raised` for the wrong size and the
      unexpected file, and the old check line `every one of the 5 source files is in
      the repo`. Fix: one name-to-size map from both Volume listings plus
      `README.md`; it raises on a missing name, on a size difference quoting both
      numbers, and on a Hub file no source accounts for. `.gitattributes` is the one
      allowed extra, on the evidence at `docs/SERVING.md:443` and the fact that
      `export` uploads only the two folders and the card. Checks read `5 source files
      match the repo in name and size` and `1 Hub-written file allowed: .gitattributes`.
      Green: `Ran 18 tests`, `OK`.
- Gates on the final tree, 2026-11-17: `python -m unittest discover -s tests -q` ->
      `Ran 468 tests in 16.513s`, `OK`, exit 0; `python -m unittest
      tests.test_export_hf -q` -> `Ran 20 tests in 2.175s`, `OK`, exit 0; `git diff
      --check` exit 0, no output; `git status --short` -> ` M serve/export_hf.py`,
      `M tasks/todo.md`, ` M docs/SERVING.md`, ` M NOW.md`,
      `?? tests/test_export_hf.py`.
- Card-only upload, 2026-11-17, on the owner's approval: added `upload_card()` and
      `--card-only` to `serve/export_hf.py`; it refuses a non-private repo, writes
      only `README.md`, and re-reads the listing before and after. Run: commit
      `c44ae4c7a83988e8ce35b4cdd7ec075cb3310169`, `card bytes 6,890`, `files 16`,
      `total bytes 19,075,487,300`, `other files kept 15 at their previous size`.
      Spend: `python -m modal billing summary --for "this month" --json` read
      **18.65353634** before and **18.65353634** after, so **0.00000000** metered
      and **0.00** billed. Counter lag noted; it moves with use.
- Verify after the upload: `python -m modal run serve/export_hf.py --verify-only
      Daneil960/eval-gated-qwen3.8-27b` -> exit 0, `ok private is True`;
      `ok list_repo_files agrees with model_info on 16 files`; `ok model.safetensors
      is 18,568,390,152 bytes`; `ok adapter/adapter_model.safetensors is 467,062,560
      bytes`; `ok 15 source files match the repo in name and size`; `ok 1 Hub-written
      file allowed: .gitattributes`; `ok anonymous read refused with HTTP 401`.
- Independent verdict, 2026-11-17: **PASS on the code-fix slice**, as the owner
      reported the reviewer's disposition. Finding 1, the privacy allow-list: fixed,
      and the test was red for real (`307 != 401` when the handler is dropped).
      Finding 2, the card loader: fixed, the block is executed out of the card itself
      and a zero `lora_B` raises. Finding 3, the manifest: fixed, a wrong-sized
      `config.json` and an unexplained Hub file both raise, and `.gitattributes` is
      the only allowed extra. The two follow-ups the reviewer raised against the
      fixes — a mocked opener that could not tell the handler was installed, and
      unbounded cleanup on an incomplete request — are fixed with a loopback chain
      and a threading server, each proved red by mutation. The record gap is closed
      by this block. No held-out case was run.
- Holdout: none run — the FAIL review reported three findings and no held-out case,
      and the fixes were reviewed without one. This line records the gap, not a
      result; the held-out case is still owed and is the first thing a later review
      should run.
- Publication milestone: **accepted 2026-11-17**, on the green verify above and the
      owner's say-so. The card drift is closed: the 6,890-byte card is on the Hub and
      the seven checks pass.
- Not done, so not ticked: section 4's deploy and live-demo boxes were run earlier
      and are not re-run here. The commit that carries this record is local only;
      this repo has no remote, so nothing is pushed.
- Two brief shapes did not survive the client: `secret=` is `secrets=[...]`, and
      `--probe-only true` is `--probe-only` / `--no-probe-only`. `hf_transfer` is
      deprecated upstream, so the image pins `huggingface_hub[hf_xet]==2.4.0`.
