"""Stage 6 slice 3 — the demo page: one HTML file, a thin proxy, no GPU.

The page is `serve/demo.html`, self-contained: no framework, no CDN, no external
font, no absolute endpoint in the browser. It calls its own origin (`/health`,
`/run`), and this app answers both: `/health` forwards to the agent's own health
endpoint, and `/run` calls `Agent.serve` in the agent app through the Modal SDK.
So the browser still sees one origin, and nothing here touches the model image,
the Volume or the A10.

It deploys as its **own** Modal app (`eval-gated-lora-agent-demo`). Adding the
page to `serve/app.py` would redeploy the agent app, rebuild its vLLM image and
cold-start the live endpoint for 312 s; the demo is not worth that. Two apps, one
of them asleep.

The proxy rebuilds the request body from validated fields, so a caller cannot
smuggle anything else to the agent, and it passes the agent's own 400 text back
unchanged. `MAX_TICKET_CHARS` is duplicated from `serve/app.py` — a test
asserts the two agree.

`Agent.serve` answers one JSON string, `{"status_code", "body"}` encoded: an
exception raised in a container arrives as traceback text, so a refusal is a value
there, not a raise. The string, not a dict, is the point — this image has no torch,
so the agent's own objects cannot be unpickled here.
Modal's own queueing needs no 303 walk either — `.remote()` waits for a slot.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import httpx
import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse

REPO_ROOT = Path(__file__).resolve().parents[1]
PAGE_LOCAL = REPO_ROOT / "serve" / "demo.html"
PAGE_REMOTE = "/root/demo.html"

# The agent app and the Cls inside it. `Cls.from_name` is lazy: it hydrates on
# first use, in the container, against the active Modal workspace. It returns the
# class handle, so the call site writes `from_name(...)()` — `Agent.serve` is a
# method on the instance, not on the handle.
AGENT_APP = "eval-gated-lora-agent"
AGENT_CLS = "Agent"

HEALTH_URL = os.environ.get(
    "HEALTH_URL",
    "https://daniel-osman--eval-gated-lora-agent-health.us-east.modal.run",
)

# Must equal `serve.app.MAX_TICKET_CHARS`; `tests/test_demo.py` holds the two together.
MAX_TICKET_CHARS = 1200
MAX_BODY_BYTES = 8 * 1024

HEALTH_TIMEOUT = httpx.Timeout(60.0, connect=20.0)

app = modal.App("eval-gated-lora-agent-demo")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install("fastapi[standard]", "httpx")
    .add_local_file(str(PAGE_LOCAL), PAGE_REMOTE)
)


class BadRequest(ValueError):
    """Bad input. The text is shown to the visitor as-is."""


def forward_body(payload: object) -> dict:
    """Exactly one of `scenario` / `ticket`, in the endpoint's own wording."""
    if not isinstance(payload, dict):
        raise BadRequest("body must be a JSON object")
    scenario = payload.get("scenario")
    ticket = payload.get("ticket")
    if (scenario is None) == (ticket is None):
        raise BadRequest(
            'send exactly one of "scenario" (a composed preset id) or "ticket" (free text)'
        )
    if scenario is not None:
        if not isinstance(scenario, str) or not scenario.strip():
            raise BadRequest('"scenario" must be a non-empty string')
        return {"scenario": scenario.strip()}
    if not isinstance(ticket, str) or not ticket.strip():
        raise BadRequest('"ticket" must be a non-empty string')
    if len(ticket) > MAX_TICKET_CHARS:
        raise BadRequest(
            f"ticket is {len(ticket)} characters; the cap is {MAX_TICKET_CHARS}. "
            "Shorten the message and send it again."
        )
    return {"ticket": ticket}


def _page() -> str:
    """Read once per container; the file is baked into the image."""
    return Path(PAGE_REMOTE).read_text(encoding="utf-8")


def build_app() -> FastAPI:
    api = FastAPI(title="eval-gated-lora-agent demo", docs_url=None, redoc_url=None, openapi_url=None)
    page = {"html": None}

    @api.get("/")
    def index() -> HTMLResponse:
        if page["html"] is None:
            page["html"] = _page()
        return HTMLResponse(
            page["html"],
            headers={"cache-control": "no-store", "x-robots-tag": "noindex"},
        )

    @api.get("/health")
    async def health() -> JSONResponse:
        """The preset list, model-free, straight from the agent app."""
        try:
            async with httpx.AsyncClient(timeout=HEALTH_TIMEOUT) as client:
                response = await client.get(HEALTH_URL, headers={"accept": "application/json"})
        except httpx.HTTPError as exc:
            return JSONResponse(
                {"ok": False, "error": f"health endpoint unreachable: {type(exc).__name__}"},
                status_code=502,
            )
        try:
            body = response.json()
        except ValueError:
            return JSONResponse(
                {"ok": False, "error": "health endpoint returned no JSON"}, status_code=502
            )
        return JSONResponse(body, status_code=response.status_code)

    @api.post("/run")
    async def run(request: Request) -> JSONResponse:
        """Run one request in the agent app, keep the connection open, pass the answer back."""
        raw = await request.body()
        if len(raw) > MAX_BODY_BYTES:
            return JSONResponse({"detail": "body is too large"}, status_code=413)
        try:
            payload = json.loads(raw or b"")
        except (json.JSONDecodeError, UnicodeDecodeError):
            # `json.loads` takes bytes and decodes them before parsing, so a body
            # that is not valid UTF-8 (`b'{"ticket":"\xff"}'`) raises
            # `UnicodeDecodeError`, not `JSONDecodeError`. Both are malformed bodies.
            return JSONResponse({"detail": "body must be JSON"}, status_code=400)
        try:
            body = forward_body(payload)
        except BadRequest as exc:
            return JSONResponse({"detail": str(exc)}, status_code=400)

        try:
            agent = modal.Cls.from_name(AGENT_APP, AGENT_CLS)()
            answer = await agent.serve.remote.aio(body)
        except modal.exception.FunctionTimeoutError:
            return JSONResponse(
                {
                    "detail": (
                        "the model did not answer within 900 s. The container may still be "
                        "loading; send the same request again — a warm one answers in seconds."
                    )
                },
                status_code=504,
            )
        except Exception as exc:
            print(f"agent call failed: {type(exc).__name__}: {exc}", flush=True)
            return JSONResponse(
                {"detail": f"the model endpoint could not be reached: {type(exc).__name__}"},
                status_code=502,
            )

        try:
            envelope = json.loads(answer)
        except (TypeError, ValueError):
            return JSONResponse({"detail": "the agent returned no JSON"}, status_code=502)
        if not isinstance(envelope, dict) or "body" not in envelope:
            return JSONResponse(
                {"detail": "the agent returned an unexpected shape"}, status_code=502
            )
        return JSONResponse(envelope["body"], status_code=envelope.get("status_code", 200))

    return api


@app.function(
    image=image,
    cpu=0.5,
    memory=512,
    timeout=1200,
    min_containers=0,
    max_containers=4,
    scaledown_window=300,
)
@modal.asgi_app()
def web():
    return build_app()
