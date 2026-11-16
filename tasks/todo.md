# Todo — stage 6 slice 4: the RPC door

One task in flight. Acceptance, verify command and files per task; a box ticks only
with the output that closed it quoted here.

Spec: the operator's directive of 2026-11-16 (the demo's `/run` hop is wrong; the
proxy chases a 303 that Modal does not send) plus `docs/SERVING.md`. The deployed
facts behind it: `eval-gated-lora-agent` (agent, A10, `Agent.run` web endpoint) and
`eval-gated-lora-agent-demo` (`serve/demo.py`, thin proxy) are both live.

Problem being fixed: `serve/demo.py` re-POSTs to a `Location` on a 303 the endpoint
never returns, so a cold POST is lost; and the browser talks to the model through
two HTTP hops that add nothing. One RPC door on the class removes both.

---

## 1. `serve/app.py` — one request body, two doors

- [ ] Extract the body of `Agent.run` into `Agent.run_request(self, payload: dict) -> dict`.
      It validates, bumps `_REQUESTS`, calls the module-level loop, and attaches `engine`.
      It raises `fastapi.HTTPException(400, detail=str(exc))` on `RequestError`, exactly as
      `run` did, so the HTTP path is byte-identical.
- [ ] `run` keeps `@modal.fastapi_endpoint(method="POST", docs=False)` and returns
      `self.run_request(item)`.
- [ ] Add `@modal.method()` `serve(self, payload: dict) -> dict`. It calls the same
      `run_request` and converts a refusal into data, because a non-web door has no status
      line: `{"error": {"status": 400, "detail": <the endpoint's own wording>}}`.
- [ ] No behaviour change on the HTTP path: same success body, same `_REQUESTS` position,
      same `PHASE_B` label.
- Files: `serve/app.py`, `tests/test_serve_app.py`
- Acceptance: the source has exactly one request body; `run` and `serve` both call it;
      `run` is still the fastapi endpoint; `serve` is a `modal.method`; a refusal over
      `serve` carries the 400 text and lets no `HTTPException` escape.
- Verify: `python -m unittest discover -s tests -q`

## 2. `serve/demo.py` — `/run` over the class door, no HTTP hop

- [ ] `/run` calls `modal.Cls.from_name("eval-gated-lora-agent", "Agent")().serve.remote.aio(payload)`
      with the same body `forward_body` already builds, under a 900 s allowance.
- [ ] Delete `post_following_303`, `TooManyRedirects`, `MAX_REDIRECT_HOPS`, `AGENT_URL` and
      the httpx POST path. Keep `forward_body`, `MAX_TICKET_CHARS`, and `/health` on HTTP —
      the health endpoint is model-free and unchanged.
- [ ] Map the door's answers: `{"error": {...}}` → that status with that `detail` verbatim;
      the 900 s allowance → the existing 504 text; a Modal transport error → 502.
- [ ] Delete `tests/test_demo.py::Redirects` and every assertion that pins the false 303
      premise. Add a test that `/run` goes through the class call and sends nothing over HTTP.
- Files: `serve/demo.py`, `tests/test_demo.py`
- Acceptance: no `AGENT_URL`, no `post_following_303`, no `MAX_REDIRECT_HOPS` in the proxy;
      `/run` posts zero times through `httpx`; a faked class call records the payload and its
      answer is mirrored, including a 400.
- Verify: `python -m unittest discover -s tests -q`

## 3. Deploy, then prove it in a browser and over the wire

- [ ] `modal deploy serve/app.py` — report deploy time; `/health` still answers 200.
- [ ] `modal deploy serve/demo.py` — report the URL.
- [ ] Browser: open the demo, click **Run N-104**, wait through the cold start, confirm plan,
      tool calls, verdict, reply, `matches_required_calls: false`, the boundary block and the
      timings; screenshot to `out/demo-n104.png`.
- [ ] Warm, POST `{"scenario":"N-999"}` to the demo `/run`; quote the 400 verbatim.
- [ ] Billing before/after, delta. `git status --short`. No commit.
- Verify: the quoted 400 ends with the endpoint's own preset list; the screenshot shows the
      panel filled; the billing delta is reported as a number.
