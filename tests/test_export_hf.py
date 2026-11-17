"""The publish script: a refusal allow-list, a loader that keeps the adapter, a manifest by size.

No network and no Modal call. The Hub client is a stub module, the Volume listings are
small fake maps, and the loader snippet is executed straight out of the model card
against fake classes, so what the test proves is what the card publishes.

The three rules under test are the ones a fresh review found broken: a 503 counts as
a refusal if only 200 fails; the card's loader drops the adapter without an error; and
the manifest compares names while ignoring the sizes that would catch a truncation.
"""

from __future__ import annotations

import contextlib
import http.server
import re
import socket
import sys
import threading
import time
import types
import unittest
import urllib.error
import urllib.request
from types import SimpleNamespace
from unittest import mock

from serve import export_hf

REPO_ID = "someone/eval-gated-qwen3.8-27b"
SAMPLE = "base_model.model.model.language_model.layers.3.self_attn.q_proj"


@contextlib.contextmanager
def redirect_chain(accepted: list[str] | None = None):
    """A loopback server: any API path answers 307 to /refuse, and /refuse answers 401.

    A real redirect is the only way to prove the no-redirect handler is installed.
    A mocked opener cannot tell `build_opener(_NoRedirect)` from `build_opener()`.
    `accepted` collects one entry per handler that started, so a test can wait until
    its request is in flight instead of sleeping and hoping.
    """
    hits: list[str] = []
    seen_headers: list[set[str]] = []
    accepted = [] if accepted is None else accepted

    class Handler(http.server.BaseHTTPRequestHandler):
        def setup(self):
            super().setup()
            self.connection.settimeout(0.5)
            accepted.append(str(self.client_address))

        def do_GET(self):  # noqa: N802 - the base class names the method
            hits.append(self.path)
            seen_headers.append({name.lower() for name in self.headers.keys()})
            if self.path == "/refuse":
                body = b"unauthorized"
                self.send_response(401)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            self.send_response(307)
            self.send_header("Location", "/refuse")
            self.send_header("Content-Length", "0")
            self.end_headers()

        def log_message(self, *args):  # noqa: ARG002 - quiet server
            return

    class Server(http.server.ThreadingHTTPServer):
        daemon_threads = True

        def handle_error(self, request, client_address):  # noqa: ARG002
            """A handler that dies on a half-sent request is not a test failure."""
            return

    server = Server(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}", hits, seen_headers
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def fake_sources() -> tuple[dict, dict]:
    """The two directory listings, small enough to read."""
    quant = {"model.safetensors": export_hf.QUANT_WEIGHTS_BYTES, "config.json": 1200}
    adapter = {
        "adapter_model.safetensors": export_hf.ADAPTER_WEIGHTS_BYTES,
        "adapter_config.json": 900,
    }
    return quant, adapter


def hub_manifest(quant: dict, adapter: dict, extra: dict | None = None) -> dict:
    """What the Hub should report for those listings: the same names and sizes."""
    files = dict(quant)
    files.update({f"adapter/{name}": size for name, size in adapter.items()})
    files["README.md"] = len(export_hf.MODEL_CARD.encode("utf-8"))
    if extra:
        files.update(extra)
    return files


def run_verify(files: dict, *, anonymous=(401, ""), quant=None, adapter=None) -> dict:
    """One local run of verify() with the Hub, the token and the read stubbed out.

    `anonymous=None` leaves the real read in place, for the loopback redirect test.
    """
    quant = quant if quant is not None else fake_sources()[0]
    adapter = adapter if adapter is not None else fake_sources()[1]
    info = SimpleNamespace(
        private=True,
        sha="deadbeef",
        siblings=[SimpleNamespace(rfilename=name, size=size) for name, size in files.items()],
    )
    api = mock.Mock()
    api.model_info.return_value = info
    api.list_repo_files.return_value = sorted(files)

    hub = types.ModuleType("huggingface_hub")
    hub.HfApi = lambda token=None: api  # noqa: ARG005 - the signature is the real one
    with contextlib.ExitStack() as stack:
        stack.enter_context(mock.patch.dict(sys.modules, {"huggingface_hub": hub}))
        stack.enter_context(mock.patch.object(export_hf, "_token", return_value="hf_not_a_real_token"))
        stack.enter_context(
            mock.patch.object(export_hf, "_listing", side_effect=[dict(quant), dict(adapter)])
        )
        if anonymous is not None:
            stack.enter_context(mock.patch.object(export_hf, "_anonymous_status", return_value=anonymous))
        return export_hf.verify.local(REPO_ID)


class AnonymousReadTests(unittest.TestCase):
    """Only 401, 403 and 404 are a refusal. Everything else is a finding."""

    def test_anonymous_503_is_not_a_refusal(self):
        quant, adapter = fake_sources()
        with self.assertRaises(AssertionError) as caught:
            run_verify(hub_manifest(quant, adapter), anonymous=(503, "service unavailable"))
        message = str(caught.exception)
        self.assertIn("HTTP 503", message)
        self.assertIn("not a refusal", message)
        self.assertIn("service unavailable", message)

    def test_anonymous_200_is_not_a_refusal(self):
        quant, adapter = fake_sources()
        with self.assertRaises(AssertionError) as caught:
            run_verify(hub_manifest(quant, adapter), anonymous=(200, "<html>login</html>"))
        self.assertIn("HTTP 200", str(caught.exception))

    def test_anonymous_302_is_not_a_refusal(self):
        quant, adapter = fake_sources()
        with self.assertRaises(AssertionError) as caught:
            run_verify(hub_manifest(quant, adapter), anonymous=(302, "Found"))
        self.assertIn("HTTP 302", str(caught.exception))

    def test_the_three_refusal_codes_pass_and_keep_the_record_wording(self):
        quant, adapter = fake_sources()
        for code in (401, 403, 404):
            with self.subTest(code=code):
                result = run_verify(hub_manifest(quant, adapter), anonymous=(code, ""))
                self.assertIn(f"anonymous read refused with HTTP {code}", result["checks"])

    def test_the_failure_message_carries_no_token(self):
        quant, adapter = fake_sources()
        with self.assertRaises(AssertionError) as caught:
            run_verify(hub_manifest(quant, adapter), anonymous=(503, "unavailable"))
        self.assertNotIn("hf_not_a_real_token", str(caught.exception))

    def test_a_redirect_is_not_followed_and_reports_its_own_code(self):
        with redirect_chain() as (base, hits, headers):
            with mock.patch.object(export_hf, "HUB_BASE", base):
                code, _excerpt = export_hf._anonymous_status(REPO_ID)
        self.assertEqual(307, code)
        self.assertEqual([f"/api/models/{REPO_ID}"], hits)
        self.assertNotIn("/refuse", hits)
        self.assertTrue(all("authorization" not in seen for seen in headers))

    def test_without_the_handler_the_same_chain_answers_401(self):
        real_build_opener = urllib.request.build_opener
        with redirect_chain() as (base, hits, _headers):
            with mock.patch.object(export_hf, "HUB_BASE", base), mock.patch.object(
                urllib.request, "build_opener", lambda *handlers: real_build_opener()
            ):
                code, _excerpt = export_hf._anonymous_status(REPO_ID)
        self.assertEqual(401, code)
        self.assertIn("/refuse", hits)

    def test_verify_rejects_the_307_and_names_the_reason(self):
        with redirect_chain() as (base, hits, _headers):
            with mock.patch.object(export_hf, "HUB_BASE", base):
                with self.assertRaises(AssertionError) as caught:
                    run_verify(hub_manifest(*fake_sources()), anonymous=None)
        message = str(caught.exception)
        self.assertIn("HTTP 307", message)
        self.assertIn("not a refusal", message)
        self.assertNotIn("/refuse", hits)

    def test_cleanup_is_bounded_when_a_request_never_finishes(self):
        accepted: list[str] = []
        started = time.monotonic()
        sock = None
        try:
            with redirect_chain(accepted) as (base, _hits, _headers):
                sock = socket.create_connection(("127.0.0.1", int(base.rsplit(":", 1)[1])), timeout=1)
                sock.sendall(b"GET /api/models/x HTTP/1.1\r\nHost: local")  # no blank line, ever
                deadline = time.monotonic() + 0.5
                while not accepted and time.monotonic() < deadline:
                    time.sleep(0.01)
                self.assertEqual(1, len(accepted), "the server never accepted the partial request")
            elapsed = time.monotonic() - started
        finally:
            if sock is not None:
                sock.close()
        self.assertLess(elapsed, 1.0, "cleanup blocked on an incomplete request")


def loader_block() -> str:
    """The python block exactly as MODEL_CARD publishes it."""
    match = re.search(r"```python\n(.*?)```", export_hf.MODEL_CARD, re.DOTALL)
    if match is None:
        raise AssertionError("the card has no ```python block")
    return match.group(1)


class FakeTensor:
    def __init__(self, values):
        self.values = list(values)

    def count_nonzero(self):
        return sum(1 for value in self.values if value)


class FakeModule:
    """One adapted target module: its own named parameters are the LoRA factors."""

    def __init__(self, lora_a, lora_b):
        self._params = {
            "lora_A.default.weight": FakeTensor(lora_a),
            "lora_B.default.weight": FakeTensor(lora_b),
            "base_layer.weight": FakeTensor([1.0]),
        }

    def named_parameters(self):
        return iter(self._params.items())


class FakePeftModel:
    def __init__(self, peft_config, lora_a, lora_b):
        self.peft_config = peft_config
        self._modules = [(SAMPLE, FakeModule(lora_a, lora_b))]

    def named_modules(self):
        return iter(self._modules)


class FakeBaseModel:
    @classmethod
    def from_pretrained(cls, *args, **kwargs):  # noqa: ARG003
        return object()


def exec_loader(peft_config, lora_a, lora_b):
    """Run the card's own block, imports removed, against fake classes.

    PEFT and torch are not installed for this suite and the block must not reach
    the Hub, so the names it imports are injected instead and every network call
    lands on a fake.
    """
    source = "\n".join(
        line
        for line in loader_block().splitlines()
        if not line.strip().startswith(("import ", "from "))
    )
    peft = FakePeftModel(peft_config, lora_a, lora_b)
    wrapper = type("FakePeft", (), {"from_pretrained": staticmethod(lambda *a, **k: peft)})
    tokenizer = type("FakeTokenizer", (), {"from_pretrained": staticmethod(lambda *a, **k: object())})
    namespace = {
        "Qwen3_5ForConditionalGeneration": FakeBaseModel,
        "AutoModelForCausalLM": FakeBaseModel,
        "PeftModel": wrapper,
        "AutoTokenizer": tokenizer,
    }
    exec(compile(source, "<model card>", "exec"), namespace)  # noqa: S102 - executing the published text is the test
    return namespace


class CardLoaderTests(unittest.TestCase):
    """The card's loader must keep the adapter, and say so when it cannot."""

    def test_the_block_names_the_architecture_the_record_found(self):
        block = loader_block()
        code = "\n".join(line for line in block.splitlines() if not line.strip().startswith("#"))
        self.assertIn("Qwen3_5ForConditionalGeneration.from_pretrained", code)
        self.assertNotIn("AutoModelForCausalLM", code)

    def test_the_peft_wrapper_is_rebound(self):
        self.assertIn("model = PeftModel.from_pretrained", loader_block())

    def test_a_dropped_adapter_with_zero_lora_b_raises(self):
        with self.assertRaises(AssertionError) as caught:
            exec_loader({"default": object()}, lora_a=[0.5, 0.5], lora_b=[0.0, 0.0])
        message = str(caught.exception)
        self.assertIn("did not attach", message)
        self.assertIn("silently", message)

    def test_an_attached_adapter_passes(self):
        exec_loader({"default": object()}, lora_a=[0.5, 0.5], lora_b=[0.25, -0.25])

    def test_an_empty_peft_config_raises(self):
        with self.assertRaises(AssertionError) as caught:
            exec_loader({}, lora_a=[0.5, 0.5], lora_b=[0.25, -0.25])
        self.assertIn("peft_config", str(caught.exception))


class ManifestSizeTests(unittest.TestCase):
    """A name that is present is not a file that arrived. The sizes must match too."""

    def test_a_wrong_size_on_a_non_weight_file_fails(self):
        quant, adapter = fake_sources()
        files = hub_manifest(quant, adapter)
        files["config.json"] = 1201
        with self.assertRaises(AssertionError) as caught:
            run_verify(files)
        message = str(caught.exception)
        self.assertIn("config.json", message)
        self.assertIn("1201", message)
        self.assertIn("1200", message)

    def test_a_wrong_size_on_a_weight_file_fails(self):
        quant, adapter = fake_sources()
        files = hub_manifest(quant, adapter)
        files["adapter/adapter_model.safetensors"] = export_hf.ADAPTER_WEIGHTS_BYTES - 1
        with self.assertRaises(AssertionError):
            run_verify(files)

    def test_an_unexplained_hub_file_fails(self):
        quant, adapter = fake_sources()
        files = hub_manifest(quant, adapter, extra={"extra.bin": 10})
        with self.assertRaises(AssertionError) as caught:
            run_verify(files)
        self.assertIn("extra.bin", str(caught.exception))

    def test_the_hub_written_gitattributes_is_the_only_allowed_extra(self):
        quant, adapter = fake_sources()
        files = hub_manifest(quant, adapter, extra={".gitattributes": 1570})
        result = run_verify(files)
        self.assertIn("5 source files match the repo in name and size", result["checks"])
        self.assertIn("1 Hub-written file allowed: .gitattributes", result["checks"])

    def test_a_missing_source_file_still_fails(self):
        quant, adapter = fake_sources()
        files = hub_manifest(quant, adapter)
        del files["adapter/adapter_config.json"]
        with self.assertRaises(AssertionError) as caught:
            run_verify(files, quant=quant, adapter=adapter)
        self.assertIn("adapter/adapter_config.json", str(caught.exception))


class PublicRepoAbortTests(unittest.TestCase):
    """An existing public repo must stop the run before the first upload."""

    def test_a_public_repo_raises_before_any_upload(self):
        api = mock.Mock()
        api.model_info.return_value = SimpleNamespace(private=False, sha="deadbeef", siblings=[])
        create_repo = mock.Mock()
        upload_folder = mock.Mock()
        hub = types.ModuleType("huggingface_hub")
        hub.HfApi = lambda token=None: api  # noqa: ARG005
        hub.create_repo = create_repo
        hub.upload_folder = upload_folder

        with mock.patch.dict(sys.modules, {"huggingface_hub": hub}), mock.patch.object(
            export_hf, "_token", return_value="hf_not_a_real_token"
        ), mock.patch.object(
            export_hf, "_whoami", return_value={"name": "someone"}
        ), mock.patch.object(
            export_hf.os.path, "isdir", return_value=True
        ), mock.patch.object(
            export_hf, "_listing", side_effect=[dict(fake_sources()[0]), dict(fake_sources()[1])]
        ):
            with self.assertRaises(PermissionError) as caught:
                export_hf.export.local(True)

        self.assertIn("public", str(caught.exception))
        self.assertEqual(0, upload_folder.call_count)
        self.assertEqual(0, api.upload_file.call_count)


if __name__ == "__main__":
    unittest.main()
