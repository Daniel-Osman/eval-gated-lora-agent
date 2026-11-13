"""The teacher client: wire shape, retries, key handling, concurrency."""

from __future__ import annotations

import json
import os
import threading
import time
import unittest
from unittest import mock

from shopagent.teacher import (
    GATEWAY,
    MAX_CONCURRENCY,
    TeacherClient,
    TeacherParseError,
    TeacherUnavailable,
    template_digest,
)

# Not key-shaped on purpose: the secret scanner must not read this as a
# credential, and no test depends on its shape, only on it never leaving.
SENTINEL = "sentinel-value-not-a-secret-0000"


def completion(text: str) -> tuple[int, bytes]:
    return (
        200,
        json.dumps(
            {
                "model": "deepseek-v4.1-flash",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            }
        ).encode(),
    )


class FakeTransport:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls: list[dict] = []

    def __call__(self, url, headers, body, timeout):
        self.calls.append({"url": url, "headers": dict(headers), "body": json.loads(body)})
        item = self.responses[0] if len(self.responses) == 1 else self.responses.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def client(transport, **kwargs) -> TeacherClient:
    options = {
        "session_id": "run-test-1",
        "api_key": SENTINEL,
        "transport": transport,
        "sleep": lambda _seconds: None,
    }
    options.update(kwargs)
    return TeacherClient(**options)


class RequestShapeTests(unittest.TestCase):
    def test_request_carries_model_path_and_headers(self) -> None:
        transport = FakeTransport([completion('{"ok": true}')])
        reply = client(transport).chat("system text", "user text")

        self.assertEqual(reply.text, '{"ok": true}')
        self.assertEqual(reply.attempts, 1)
        self.assertEqual(reply.status, 200)

        call = transport.calls[0]
        self.assertEqual(call["url"], f"{GATEWAY}/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {SENTINEL}")
        self.assertEqual(call["headers"]["x-opencode-session"], "run-test-1")
        self.assertEqual(call["headers"]["Content-Type"], "application/json")
        self.assertTrue(call["headers"]["User-Agent"])
        self.assertEqual(call["body"]["model"], "deepseek-v4.1-flash")
        self.assertEqual(call["body"]["temperature"], 0)
        self.assertFalse(call["body"]["stream"])
        self.assertEqual(
            call["body"]["messages"],
            [
                {"role": "system", "content": "system text"},
                {"role": "user", "content": "user text"},
            ],
        )

    def test_key_comes_from_primary_env_name(self) -> None:
        env = {"OPENCODE_API_KEY": SENTINEL}
        with mock.patch.dict(os.environ, env, clear=True):
            transport = FakeTransport([completion("hi")])
            client(transport, api_key=None).chat("s", "u")
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], f"Bearer {SENTINEL}")

    def test_alias_env_name_is_a_fallback(self) -> None:
        env = {"OPENCODE_GO_API_KEY": SENTINEL}
        with mock.patch.dict(os.environ, env, clear=True):
            transport = FakeTransport([completion("hi")])
            client(transport, api_key=None).chat("s", "u")
        self.assertEqual(transport.calls[0]["headers"]["Authorization"], f"Bearer {SENTINEL}")

    def test_missing_key_raises_before_any_request(self) -> None:
        transport = FakeTransport([completion("hi")])
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(TeacherUnavailable):
                client(transport, api_key=None)
        self.assertEqual(transport.calls, [])

    def test_key_never_appears_in_repr_or_errors(self) -> None:
        transport = FakeTransport([(401, b"{}")])
        teacher = client(transport)
        self.assertNotIn(SENTINEL, repr(teacher))
        self.assertIn("<redacted>", repr(teacher))
        with self.assertRaises(TeacherUnavailable) as caught:
            teacher.chat("s", "u")
        self.assertNotIn(SENTINEL, str(caught.exception))

    def test_bad_session_id_and_wide_concurrency_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            client(FakeTransport([completion("hi")]), session_id="  ")
        with self.assertRaises(ValueError):
            client(FakeTransport([completion("hi")]), max_concurrency=MAX_CONCURRENCY + 1)


class RetryTests(unittest.TestCase):
    def test_429_is_retried_then_succeeds(self) -> None:
        transport = FakeTransport([(429, b""), completion("ok")])
        teacher = client(transport)
        reply = teacher.chat("s", "u")
        self.assertEqual(reply.text, "ok")
        self.assertEqual(reply.attempts, 2)
        self.assertEqual(teacher.requests, 2)

    def test_5xx_exhausts_three_attempts(self) -> None:
        transport = FakeTransport([(503, b"")])
        teacher = client(transport)
        with self.assertRaises(TeacherUnavailable):
            teacher.chat("s", "u")
        self.assertEqual(teacher.requests, 3)

    def test_400_is_not_retried(self) -> None:
        transport = FakeTransport([(400, b"")])
        teacher = client(transport)
        with self.assertRaises(TeacherUnavailable):
            teacher.chat("s", "u")
        self.assertEqual(teacher.requests, 1)

    def test_403_is_retried_once(self) -> None:
        transport = FakeTransport([(403, b"")])
        teacher = client(transport)
        with self.assertRaises(TeacherUnavailable):
            teacher.chat("s", "u")
        self.assertEqual(teacher.requests, 2)

    def test_network_error_is_retried_then_succeeds(self) -> None:
        transport = FakeTransport([OSError("reset"), completion("ok")])
        teacher = client(transport)
        self.assertEqual(teacher.chat("s", "u").text, "ok")
        self.assertEqual(teacher.requests, 2)

    def test_backoff_uses_the_injected_sleep(self) -> None:
        waits: list[float] = []
        transport = FakeTransport([(500, b""), completion("ok")])
        client(transport, sleep=waits.append).chat("s", "u")
        self.assertEqual(waits, [1.0])


class ParseTests(unittest.TestCase):
    def test_non_json_body_keeps_the_text(self) -> None:
        transport = FakeTransport([(200, b"<html>not json</html>")])
        with self.assertRaises(TeacherParseError) as caught:
            client(transport).chat("s", "u")
        self.assertIn("not json", caught.exception.text)

    def test_empty_content_is_rejected(self) -> None:
        transport = FakeTransport([completion("   ")])
        with self.assertRaises(TeacherParseError):
            client(transport).chat("s", "u")

    def test_template_digest_is_stable(self) -> None:
        self.assertEqual(template_digest(), template_digest())
        self.assertRegex(template_digest(), r"^[0-9a-f]{64}$")


class ConcurrencyTests(unittest.TestCase):
    def test_map_stays_at_or_below_eight_and_keeps_order(self) -> None:
        live = 0
        peak = 0
        guard = threading.Lock()

        def transport(url, headers, body, timeout):
            nonlocal live, peak
            with guard:
                live += 1
                peak = max(peak, live)
            time.sleep(0.02)
            text = json.loads(body)["messages"][1]["content"]
            with guard:
                live -= 1
            return completion(text)

        prompts = [("s", f"prompt-{index}") for index in range(20)]
        replies = client(transport).map(prompts)

        self.assertEqual([reply.text for reply in replies], [f"prompt-{index}" for index in range(20)])
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, MAX_CONCURRENCY)

    def test_empty_map_makes_no_call(self) -> None:
        transport = FakeTransport([completion("unused")])
        self.assertEqual(client(transport).map([]), [])
        self.assertEqual(transport.calls, [])


if __name__ == "__main__":
    unittest.main()
