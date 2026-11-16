"""The demo page: self-contained, honest, and a proxy that adds no policy.

The page is a single HTML file with no external fetch, no framework, no CDN and no
external font, so the tests below are mostly about what it *says* — the measured
cold start, the boundary case, the unproven free text — and about the two things a
proxy must not do: invent a second request shape, or swallow the endpoint's 400.

No network here. The route tests replace `httpx.AsyncClient` with a stub, so they
prove the forwarding, the status mirror and the timeout text offline.
"""

from __future__ import annotations

import ast
import contextlib
import io
import json
import unittest
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from serve import app as app_mod
from serve import demo

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "serve" / "demo.html"
SOURCE = REPO / "serve" / "demo.py"


def code_only() -> str:
    """The proxy's source without its docstring or comments, so prose cannot pass as a gate."""
    text = SOURCE.read_text(encoding="utf-8")
    tree = ast.parse(text)
    docstring = ast.get_docstring(tree, clean=False) or ""
    text = text.replace(docstring, "") if docstring else text
    return "\n".join(line for line in text.splitlines() if not line.strip().startswith("#"))


def page_text() -> str:
    return PAGE.read_text(encoding="utf-8")


class FakeResponse:
    def __init__(self, status_code: int, payload, headers: dict | None = None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if isinstance(self._payload, ValueError):
            raise ValueError("no json")
        return self._payload

    @property
    def text(self) -> str:
        if isinstance(self._payload, ValueError):
            return "<html>not json</html>"
        return json.dumps(self._payload)


class FakeAsyncClient:
    """Canned answers, one per call when a queue is set; records what the proxy sent."""

    behaviour: dict = {}

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    def _answer(self):
        if self.behaviour.get("error") is not None:
            raise self.behaviour["error"]
        queued = self.behaviour.get("responses")
        if queued:
            return queued.pop(0)
        return self.behaviour["response"]

    def _record(self, call):
        self.behaviour["called"] = call
        self.behaviour.setdefault("calls", []).append(call)

    async def get(self, url, headers=None):
        self._record(("GET", url))
        return self._answer()

    async def post(self, url, json=None, headers=None):
        self._record(("POST", url, json))
        return self._answer()


class FakeAgentInstance:
    """What `FakeAgent(...)` returns: `Agent.serve` reached as `.serve.remote.aio`."""

    def __init__(self):
        self.serve = self
        self.remote = self

    async def aio(self, body):
        FakeAgent.calls.append(body)
        if FakeAgent.error is not None:
            raise FakeAgent.error
        return FakeAgent.answer


class FakeAgent:
    """Stands in for `modal.Cls.from_name(...)`: the handle, not the instance.

    `from_name` returns the class handle and `serve` lives on the instance, so this
    handle carries no `serve` of its own: a proxy that forgets the trailing `()`
    fails here instead of in a container.
    """

    answer: dict = {}
    error: Exception | None = None
    calls: list = []

    def __init__(self, app_name, cls_name):
        self.app_name = app_name
        self.cls_name = cls_name

    def __call__(self):
        return FakeAgentInstance()


class PageContract(unittest.TestCase):
    def test_no_external_resource(self):
        text = page_text()
        for needle in ("http://", "https://", "//cdn", "@import", "<link", "fonts.googleapis"):
            self.assertNotIn(needle, text, f"the page must not reach outside itself: {needle}")
        self.assertEqual(text.count("<script"), 1, "one inline script, no external script")
        self.assertIn("<!DOCTYPE html>", text)
        self.assertIn('name="color-scheme" content="light dark"', text)

    def test_says_the_wait_before_the_first_click(self):
        text = page_text().lower()
        # 309 s end to end and 291.4 s of it engine init are what this page's own
        # path measured (out/serve-deployed3-cold.json); the split between
        # container start and engine init is not measured, so the page must not
        # imply one. 1.5-2.4 s and 6.3 s are the earlier warm timings.
        self.assertIn("the first request after idle waits about 5 minutes", text)
        for measured in ("309", "291.4", "1.5–2.4", "6.3", "1.08×", "20", "14 training-seen"):
            self.assertIn(measured, text, f"missing measured fact {measured}")
        self.assertIn("not measured", text)
        for invented in ("Building the vLLM engine", "5 s container", "293 s engine init"):
            self.assertNotIn(invented.lower(), text, f"the page still invents a step: {invented}")

    def test_counter_is_live_and_not_a_spinner(self):
        text = page_text()
        self.assertIn('id="clock"', text)
        self.assertIn("elapsed", text)
        self.assertIn("setInterval", text)
        self.assertIn("it does not know which step is running", text.lower())
        self.assertNotIn("spinner", text.lower())

    def test_boundary_is_named_plainly(self):
        text = page_text()
        self.assertIn("matches_required_calls", text)
        self.assertIn(
            "Phase A checks that the sentence is true against state, not that the decision was right.",
            text,
        )
        lowered = text.lower()
        self.assertIn("wrong decision", lowered)
        self.assertIn("phase a", lowered)

    def test_includes_n_104(self):
        text = page_text()
        self.assertIn("N-104", text)
        self.assertIn("2100 != 2599", text)
        self.assertIn("$21.00", text)
        self.assertIn("$25.99", text)

    def test_free_text_is_labelled_unmeasured(self):
        lowered = page_text().lower()
        self.assertIn("quality on unseen tickets is unmeasured", lowered)
        self.assertIn("checks truth, not wisdom", lowered)

    def test_cap_matches_the_endpoint(self):
        self.assertEqual(demo.MAX_TICKET_CHARS, app_mod.MAX_TICKET_CHARS)
        self.assertIn("MAX_TICKET_CHARS = 1200", page_text())

    def test_accessible_basics(self):
        text = page_text()
        self.assertIn('for="ticket"', text)
        self.assertIn('id="ticket"', text)
        self.assertIn('aria-live="polite"', text)
        self.assertIn('role="status"', text)
        self.assertIn("aria-busy", text)
        self.assertIn(":focus-visible", text)
        self.assertNotIn("outline: none", text)
        self.assertNotIn("outline:none", text)
        self.assertIn("prefers-color-scheme: dark", text)
        self.assertIn('[data-theme="dark"]', text)

    def test_only_same_origin_calls(self):
        text = page_text()
        self.assertIn('fetch("/run"', text)
        self.assertIn('fetch("/health"', text)
        self.assertNotIn("modal.run", text)

    def test_retry_button_follows_busy(self):
        # The wait panel promises every control is disabled. A health failure
        # mid-request builds a Retry button, so it must land disabled while busy.
        text = page_text()
        self.assertIn("retry.disabled = state.busy;", text)


class RequestContract(unittest.TestCase):
    def setUp(self):
        self.presets = app_mod.load_presets(REPO / "bench")

    def test_composed_body_is_one_field(self):
        self.assertEqual(demo.forward_body({"scenario": "N-104"}), {"scenario": "N-104"})
        self.assertEqual(demo.forward_body({"scenario": " N-104 "}), {"scenario": "N-104"})

    def test_free_text_body_is_one_field(self):
        self.assertEqual(demo.forward_body({"ticket": "Ordinary ticket"}), {"ticket": "Ordinary ticket"})

    def test_a_scenario_and_a_ticket_is_refused(self):
        with self.assertRaises(demo.BadRequest):
            demo.forward_body({"scenario": "N-104", "ticket": "also this"})

    def test_non_string_scenario_is_refused_before_the_network(self):
        for bad in (7, "", "   ", ["N-104"]):
            with self.assertRaises(demo.BadRequest):
                demo.forward_body({"scenario": bad})

    def test_missing_or_double_selection_uses_the_endpoint_wording(self):
        for payload in ({}, {"scenario": "N-104", "ticket": "x"}, {"ticket": ""}):
            with self.assertRaises(demo.BadRequest) as caught:
                demo.forward_body(payload)
            with self.assertRaises(app_mod.RequestError) as theirs:
                app_mod.validate_request(payload, self.presets)
            self.assertEqual(str(caught.exception), str(theirs.exception))

    def test_over_cap_uses_the_endpoint_wording(self):
        long = "x" * (demo.MAX_TICKET_CHARS + 1)
        with self.assertRaises(demo.BadRequest) as caught:
            demo.forward_body({"ticket": long})
        with self.assertRaises(app_mod.RequestError) as theirs:
            app_mod.validate_request({"ticket": long}, self.presets)
        self.assertEqual(str(caught.exception), str(theirs.exception))
        self.assertIn("1200", str(caught.exception))

    def test_no_gpu_and_no_volume(self):
        code = code_only()
        self.assertNotIn("gpu=", code)
        self.assertNotIn("volumes=", code)
        self.assertIn("modal.Cls.from_name(AGENT_APP, AGENT_CLS)", code)
        self.assertIn('modal.App("eval-gated-lora-agent-demo")', code)

    def test_the_agent_is_reached_by_name_not_by_url(self):
        self.assertEqual(demo.AGENT_APP, app_mod.app.name)
        self.assertTrue(hasattr(app_mod.Agent, "serve"))
        self.assertNotIn("AGENT_URL", code_only())


class Routes(unittest.TestCase):
    def setUp(self):
        FakeAsyncClient.behaviour = {
            "response": FakeResponse(200, {"ok": True}),
            "error": None,
            "responses": [],
            "calls": [],
        }
        patcher = mock.patch.object(demo.httpx, "AsyncClient", FakeAsyncClient)
        patcher.start()
        self.addCleanup(patcher.stop)
        FakeAgent.answer = {}
        FakeAgent.error = None
        FakeAgent.calls = []
        agent_patch = mock.patch.object(demo.modal.Cls, "from_name", FakeAgent)
        agent_patch.start()
        self.addCleanup(agent_patch.stop)
        page_patch = mock.patch.object(demo, "PAGE_REMOTE", str(PAGE))
        page_patch.start()
        self.addCleanup(page_patch.stop)
        self.client = TestClient(demo.build_app())

    def test_page_is_served(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("text/html", response.headers["content-type"])
        self.assertIn("Eval-gated LoRA agent", response.text)
        self.assertEqual(response.headers["cache-control"], "no-store")

    def test_run_calls_the_agent_with_the_built_body_and_mirrors_success(self):
        FakeAgent.answer = json.dumps(
            {"status_code": 200, "body": {"mode": "composed", "verdict": {"verdict": "APPROVED"}}}
        )
        response = self.client.post("/run", json={"scenario": "N-104"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["verdict"]["verdict"], "APPROVED")
        self.assertEqual(FakeAgent.calls, [{"scenario": "N-104"}])

    def test_run_mirrors_the_agent_400_verbatim(self):
        detail = "unknown scenario; the composed presets are: N-100, N-104"
        FakeAgent.answer = json.dumps({"status_code": 400, "body": {"detail": detail}})
        response = self.client.post("/run", json={"scenario": "N-999"})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json()["detail"], detail)

    def test_run_never_calls_the_agent_for_a_bad_body(self):
        self.assertEqual(self.client.post("/run", content=b"not json").status_code, 400)
        self.assertEqual(self.client.post("/run", json={}).status_code, 400)
        self.assertEqual(FakeAgent.calls, [])

    def test_a_non_utf8_body_is_the_same_400_and_no_hop(self):
        """`json.loads` decodes bytes first, so bad UTF-8 raises `UnicodeDecodeError`.

        The plain malformed-JSON test above cannot reach that branch: its bytes are
        valid UTF-8. A proxy that catches `JSONDecodeError` only turns this body into
        a 500, and this client shows the 500 instead of raising inside the test.
        """
        client = TestClient(demo.build_app(), raise_server_exceptions=False)
        response = client.post("/run", content=b'{"ticket":"\xff"}')
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "body must be JSON"})
        self.assertEqual(FakeAgent.calls, [])

    def test_timeout_is_a_504_with_advice(self):
        FakeAgent.error = demo.modal.exception.FunctionTimeoutError("slow")
        response = self.client.post("/run", json={"ticket": "Where is ORD-1037?"})
        self.assertEqual(response.status_code, 504)
        self.assertIn("900 s", response.json()["detail"])

    def test_a_failed_call_is_a_502(self):
        FakeAgent.error = demo.modal.exception.ExecutionError("nope")
        captured = io.StringIO()
        with contextlib.redirect_stdout(captured):
            response = self.client.post("/run", json={"ticket": "Where is ORD-1037?"})
        self.assertEqual(response.status_code, 502)
        # The visitor sees the type name only; the container log carries the detail.
        self.assertIn("ExecutionError", response.json()["detail"])
        self.assertIn("agent call failed: ExecutionError: nope", captured.getvalue())

    def test_an_unexpected_answer_shape_is_a_502(self):
        FakeAgent.answer = json.dumps({"mode": "composed"})
        response = self.client.post("/run", json={"ticket": "Where is ORD-1037?"})
        self.assertEqual(response.status_code, 502)

    def test_a_non_text_answer_is_a_502(self):
        """The old dict contract: a pickled object cannot cross into this torch-free image."""
        FakeAgent.answer = {"status_code": 200, "body": {"mode": "composed"}}
        response = self.client.post("/run", json={"ticket": "Where is ORD-1037?"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "the agent returned no JSON")

    def test_text_that_is_not_json_is_a_502(self):
        FakeAgent.answer = "not json at all"
        response = self.client.post("/run", json={"ticket": "Where is ORD-1037?"})
        self.assertEqual(response.status_code, 502)
        self.assertEqual(response.json()["detail"], "the agent returned no JSON")

    def test_health_passes_the_preset_list_through(self):
        payload = {
            "ok": True,
            "presets": [{"id": "N-104", "title": "x", "training_seen": True, "ticket": "t"}],
        }
        FakeAsyncClient.behaviour["response"] = FakeResponse(200, payload)
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), payload)
        self.assertEqual(FakeAsyncClient.behaviour["called"], ("GET", demo.HEALTH_URL))

    def test_health_failure_is_a_502_the_page_can_report(self):
        FakeAsyncClient.behaviour["error"] = demo.httpx.ConnectError("nope")
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 502)
        self.assertFalse(response.json()["ok"])


if __name__ == "__main__":
    unittest.main()
