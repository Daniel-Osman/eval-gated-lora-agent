"""The Phase B judge: reused wire, strict contract, fail-closed, provenance."""

from __future__ import annotations

import json
import os
import unittest
import urllib.request
from unittest import mock

from shopagent.judge import (
    GATEWAY,
    REASON_MAX_CHARS,
    SYSTEM_PROMPT,
    JudgeClient,
    JudgeInvalid,
    JudgeUnavailable,
    JudgeVerdict,
    build_judge_prompt,
    check_verdict,
    parse_judge_output,
)

# Not key-shaped on purpose: the secret scanner must not read this as a
# credential, and no test depends on its shape, only on it never leaving.
SENTINEL = "sentinel-value-not-a-secret-0001"

_URLOPEN = urllib.request.urlopen


def _blocked(*_args, **_kwargs):
    raise AssertionError("a judge test tried to reach the network")


def setUpModule() -> None:
    # Every test injects a transport. If one forgets, this fails loudly instead
    # of dialling the gateway.
    urllib.request.urlopen = _blocked


def tearDownModule() -> None:
    urllib.request.urlopen = _URLOPEN


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


def answer(verdict: str, reason: str = "the log supports it") -> tuple[int, bytes]:
    return completion(json.dumps({"verdict": verdict, "reason": reason}))


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


def client(transport, **kwargs) -> JudgeClient:
    options = {
        "session_id": "judge-run-1",
        "api_key": SENTINEL,
        "transport": transport,
        "sleep": lambda _seconds: None,
    }
    options.update(kwargs)
    return JudgeClient(**options)


PROMPT = {
    "case_id": "G-2",
    "calls": [
        {
            "tool": "refund_order",
            "args": {"order_id": "ORD-1001", "amount_cents": 2400},
            "result": {"ok": True, "refund_id": "RF-1001", "status": "succeeded"},
        }
    ],
    "reply": "I've refunded $24.00 for ORD-1001.",
    "state": {"refunds": {"added": [{"id": "RF-1001", "amount_cents": 2400}], "changed": []}},
}


def judge_call(judge: JudgeClient, **overrides) -> JudgeVerdict:
    payload = dict(PROMPT)
    payload.update(overrides)
    return judge(**payload)


class RequestShapeTests(unittest.TestCase):
    def test_request_carries_model_path_and_headers(self) -> None:
        transport = FakeTransport([answer("APPROVED")])
        transport_call = judge_call(client(transport))

        self.assertEqual(transport_call.verdict, "APPROVED")
        call = transport.calls[0]
        self.assertEqual(call["url"], f"{GATEWAY}/chat/completions")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {SENTINEL}")
        self.assertEqual(call["headers"]["x-opencode-session"], "judge-run-1")
        self.assertEqual(call["body"]["model"], "deepseek-v4.1-flash")
        self.assertEqual(call["body"]["temperature"], 0)
        self.assertFalse(call["body"]["stream"])
        self.assertEqual(call["body"]["messages"][0]["content"], SYSTEM_PROMPT)
        self.assertIn("refund_order", call["body"]["messages"][1]["content"])

    def test_key_alias_is_a_fallback_and_primary_wins(self) -> None:
        for env, expected in (
            ({"OPENCODE_API_KEY": SENTINEL}, SENTINEL),
            ({"OPENCODE_GO_API_KEY": SENTINEL}, SENTINEL),
            ({"OPENCODE_API_KEY": "primary", "OPENCODE_GO_API_KEY": "alias"}, "primary"),
        ):
            with mock.patch.dict(os.environ, env, clear=True):
                transport = FakeTransport([answer("APPROVED")])
                judge_call(client(transport, api_key=None))
            self.assertEqual(transport.calls[0]["headers"]["Authorization"], f"Bearer {expected}")

    def test_missing_key_raises_before_any_request(self) -> None:
        transport = FakeTransport([answer("APPROVED")])
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(JudgeUnavailable):
                client(transport, api_key=None)
        self.assertEqual(transport.calls, [])

    def test_key_never_appears_in_repr_or_errors_or_records(self) -> None:
        transport = FakeTransport([(401, b"{}")])
        records: list[dict] = []
        judge = client(transport, record=records.append)
        self.assertNotIn(SENTINEL, repr(judge))
        with self.assertRaises(JudgeUnavailable) as caught:
            judge_call(judge)
        self.assertNotIn(SENTINEL, str(caught.exception))
        self.assertEqual(len(records), 1)
        self.assertNotIn(SENTINEL, json.dumps(records))


class ContractTests(unittest.TestCase):
    def test_valid_approve(self) -> None:
        transport = FakeTransport([answer("APPROVED")])
        self.assertEqual(judge_call(client(transport)).verdict, "APPROVED")

    def test_valid_reject(self) -> None:
        transport = FakeTransport([answer("REJECTED", "the refund row is missing")])
        verdict = judge_call(client(transport))
        self.assertEqual(verdict.verdict, "REJECTED")
        self.assertEqual(verdict.reason, "the refund row is missing")

    def test_valid_escalate(self) -> None:
        transport = FakeTransport([answer("ESCALATED", "the log is ambiguous")])
        self.assertEqual(judge_call(client(transport)).verdict, "ESCALATED")

    def test_prose_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output("The reply looks fine to me.")

    def test_fence_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('```json\n{"verdict": "APPROVED", "reason": "ok"}\n```')

    def test_wrong_enum_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "MAYBE", "reason": "ok"}')

    def test_lowercase_enum_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "approved", "reason": "ok"}')

    def test_missing_reason_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "APPROVED"}')

    def test_blank_reason_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "APPROVED", "reason": "  "}')

    def test_non_string_reason_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "APPROVED", "reason": 7}')

    def test_overlong_reason_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output(
                json.dumps({"verdict": "APPROVED", "reason": "x" * (REASON_MAX_CHARS + 1)})
            )

    def test_extra_key_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output(
                '{"verdict": "APPROVED", "reason": "ok", "confidence": 0.9}'
            )

    def test_duplicate_verdict_key_is_invalid(self) -> None:
        # A repeated key is not "last one wins": plain `json.loads` hands back
        # APPROVED for this line. A duplicate is off contract and never parses.
        with self.assertRaises(ValueError):
            parse_judge_output(
                '{"verdict":"REJECTED","verdict":"APPROVED","reason":"ok"}'
            )

    def test_duplicate_reason_key_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output(
                '{"verdict":"APPROVED","reason":"ok","reason":"the log supports it"}'
            )

    def test_a_duplicate_key_nested_anywhere_is_invalid(self) -> None:
        # The hook runs for every object the parser reads, not only the top one.
        # The message names the repeat, so this cannot pass on the extra-key rule.
        with self.assertRaisesRegex(ValueError, "repeats the key 'a'"):
            parse_judge_output(
                '{"verdict":"APPROVED","reason":"ok","meta":{"a":1,"a":2}}'
            )

    def test_empty_text_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output("   ")

    def test_typed_verdict_takes_the_same_rules(self) -> None:
        for verdict in (
            JudgeVerdict("MAYBE", "ok"),
            JudgeVerdict("APPROVED", " "),
            JudgeVerdict("APPROVED", "x" * (REASON_MAX_CHARS + 1)),
            JudgeVerdict("APPROVED", 7),  # type: ignore[arg-type]
        ):
            with self.assertRaises(ValueError):
                check_verdict(verdict)

    def test_a_malformed_answer_raises_invalid_not_approval(self) -> None:
        transport = FakeTransport([completion("looks fine")])
        with self.assertRaises(JudgeInvalid):
            judge_call(client(transport))


class RetryTests(unittest.TestCase):
    def test_429_then_success(self) -> None:
        sleeps: list[float] = []
        transport = FakeTransport([(429, b"{}"), answer("APPROVED")])
        judge = client(transport, sleep=sleeps.append)
        self.assertEqual(judge_call(judge).verdict, "APPROVED")
        self.assertEqual(judge.requests, 2)
        self.assertEqual(sleeps, [1.0])

    def test_5xx_then_success_sleeps_before_the_retry(self) -> None:
        sleeps: list[float] = []
        transport = FakeTransport([(503, b"{}"), answer("APPROVED")])
        judge = client(transport, sleep=sleeps.append)
        self.assertEqual(judge_call(judge).verdict, "APPROVED")
        self.assertEqual(judge.requests, 2)
        self.assertEqual(sleeps, [1.0])

    def test_transport_error_after_retries_is_unavailable(self) -> None:
        transport = FakeTransport([OSError("connection reset")])
        judge = client(transport)
        with self.assertRaises(JudgeUnavailable):
            judge_call(judge)
        self.assertEqual(judge.requests, 3)

    def test_403_gets_one_retry_then_stops(self) -> None:
        transport = FakeTransport([(403, b"{}")])
        judge = client(transport)
        with self.assertRaises(JudgeUnavailable):
            judge_call(judge)
        self.assertEqual(judge.requests, 2)

    def test_400_is_not_retried(self) -> None:
        transport = FakeTransport([(400, b"{}")])
        judge = client(transport)
        with self.assertRaises(JudgeUnavailable):
            judge_call(judge)
        self.assertEqual(judge.requests, 1)

    def test_exhausted_5xx_is_unavailable(self) -> None:
        transport = FakeTransport([(500, b"{}")])
        judge = client(transport)
        with self.assertRaises(JudgeUnavailable):
            judge_call(judge)
        self.assertEqual(judge.requests, 3)


class PromptTests(unittest.TestCase):
    def test_prompt_carries_the_log_and_the_reply(self) -> None:
        text = build_judge_prompt(
            calls=PROMPT["calls"], reply=PROMPT["reply"], state=PROMPT["state"]
        )
        self.assertIn("refund_order", text)
        self.assertIn("RF-1001", text)
        self.assertIn("succeeded", text)
        self.assertIn(PROMPT["reply"], text)

    def test_prompt_carries_no_gate_code_and_no_expected_verdict(self) -> None:
        text = build_judge_prompt(
            calls=PROMPT["calls"], reply=PROMPT["reply"], state=PROMPT["state"]
        )
        for forbidden in ("JUDGE_", "UNVERIFIED_CLAIM", "required_facts", "gate:"):
            self.assertNotIn(forbidden, text)

    def test_prompt_keeps_a_read_only_result_envelope(self) -> None:
        calls = [
            {
                "tool": "get_order",
                "args": {"order_id": "ORD-1001"},
                "result": {"ok": True, "order": {"id": "ORD-1001", "status": "paid"}},
            }
        ]
        text = build_judge_prompt(calls=calls, reply="ORD-1001 is paid.", state={})
        self.assertIn('"status": "paid"', text)


class ProvenanceTests(unittest.TestCase):
    def test_one_record_per_call_with_the_six_fields(self) -> None:
        records: list[dict] = []
        transport = FakeTransport([answer("REJECTED", "amount mismatch")])
        judge_call(client(transport, record=records.append))
        self.assertEqual(len(records), 1)
        record = records[0]
        self.assertEqual(
            set(record), {"model", "gateway", "case_id", "checked_at", "verdict", "reason"}
        )
        self.assertEqual(record["model"], "deepseek-v4.1-flash")
        self.assertEqual(record["gateway"], GATEWAY)
        self.assertEqual(record["case_id"], "G-2")
        self.assertEqual(record["verdict"], "REJECTED")
        self.assertEqual(record["reason"], "amount mismatch")
        self.assertTrue(record["checked_at"])

    def test_a_failed_call_records_escalation_not_approval(self) -> None:
        for transport, code in (
            (FakeTransport([OSError("down")]), "JUDGE_UNAVAILABLE"),
            (FakeTransport([completion("prose")]), "JUDGE_INVALID"),
        ):
            records: list[dict] = []
            judge = client(transport, record=records.append)
            with self.assertRaises((JudgeUnavailable, JudgeInvalid)):
                judge_call(judge)
            self.assertEqual(len(records), 1)
            self.assertEqual(records[0]["verdict"], "ESCALATED")
            self.assertTrue(records[0]["reason"].startswith(code))

    def test_the_verdict_carries_its_own_provenance(self) -> None:
        transport = FakeTransport([answer("APPROVED")])
        verdict = judge_call(client(transport))
        self.assertEqual(verdict.model, "deepseek-v4.1-flash")
        self.assertEqual(verdict.gateway, GATEWAY)
        self.assertEqual(verdict.case_id, "G-2")
        self.assertTrue(verdict.checked_at)


if __name__ == "__main__":
    unittest.main()
