"""judge-bench: the join, the expectation rule and the A+B counting.

Every test drives the real `JudgeClient` over a fake transport, so the parse and
fail-closed paths are exercised without a network call and without a key.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopagent.bench import load_all_cases
from shopagent.bench import _check_reply as check_reply
from shopagent.judge import JudgeClient
from shopagent.judgebench import (
    load_rows,
    score,
    summarise,
    write_outputs,
)

SENTINEL = "sentinel-value-not-a-secret-0000"
BASE_CASE_ID = "N-32"  # normal, one read-only call, an approving honest reply


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


def judge_says(verdict: str) -> tuple[int, bytes]:
    return completion(json.dumps({"verdict": verdict, "reason": "checked against the log"}))


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


def client(transport) -> JudgeClient:
    return JudgeClient(
        session_id="judge-test-1",
        api_key=SENTINEL,
        transport=transport,
        sleep=lambda _seconds: None,
    )


def make_case(kind: str | None = None) -> dict:
    cases = {case["id"]: case for case in load_all_cases()}
    case = dict(cases[BASE_CASE_ID])
    if kind is not None:
        case["kind"] = kind
    return case


def make_row(case: dict, *, verdict: str = "APPROVED", codes=(), reply: str | None = None,
             recorded=None) -> dict:
    """A bench row as the artifact stores it, with the gate's reply marks filled."""
    reply = case["honest_reply"] if reply is None else reply
    calls = []
    for item in case.get("required_calls", []):
        call = {"tool": item["name"], "args": item.get("args", {})}
        if recorded is not None:
            call["result"] = recorded
        calls.append(call)
    facts_ok, phrases_ok, _ = check_reply(case, reply) if reply else (False, False, [])
    return {
        "id": case["id"],
        "kind": case["kind"],
        "file": "test.jsonl",
        "path": "free_text",
        "status": "ok" if verdict != "NO_DRAFT" else "rejected",
        "phase_a": (
            None
            if verdict == "NO_DRAFT"
            else {"verdict": verdict, "codes": list(codes), "reasons": [], "phase": "A"}
        ),
        "reply": reply,
        "tool_calls": calls,
        "reply_facts_ok": facts_ok,
        "reply_phrases_ok": phrases_ok,
        "state_errors": [],
    }


class JoinTests(unittest.TestCase):
    def test_unknown_case_id_raises(self) -> None:
        row = {**make_row(make_case()), "id": "N-999"}
        with self.assertRaises(ValueError):
            score([row], client(FakeTransport([judge_says("APPROVED")])), cases={BASE_CASE_ID: make_case()})

    def test_duplicate_row_id_raises(self) -> None:
        case = make_case()
        rows = [make_row(case), make_row(case)]
        with self.assertRaises(ValueError):
            score(rows, client(FakeTransport([judge_says("APPROVED")])), cases={case["id"]: case})

    def test_kind_mismatch_raises(self) -> None:
        case = make_case()
        row = {**make_row(case), "kind": "adversarial"}
        cases = {case["id"]: case}
        with self.assertRaises(ValueError):
            score([row], client(FakeTransport([judge_says("APPROVED")])), cases=cases)


class ExpectationAndCountingTests(unittest.TestCase):
    def test_adversarial_approval_is_a_false_approval(self) -> None:
        case = make_case(kind="adversarial")
        row = make_row(case)
        result = score([row], client(FakeTransport([judge_says("APPROVED")])), cases={case["id"]: case})

        summary = result["summary"]
        self.assertTrue(summary["reconciliation"]["balances"])
        self.assertEqual(summary["phase_a"]["false_approvals"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["APPROVED"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["judged"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["missed_by_judge"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["caught_by_judge"], 0)
        # The draft states every required fact, so the literal marking check passes.
        self.assertEqual(summary["phase_a"]["marking_check_passing_approvals"], 1)

    def test_normal_approval_is_not_a_false_approval(self) -> None:
        case = make_case(kind="normal")
        row = make_row(case)
        summary = score(
            [row], client(FakeTransport([judge_says("APPROVED")])), cases={case["id"]: case}
        )["summary"]

        self.assertEqual(summary["phase_a"]["false_approvals"], 0)
        self.assertEqual(summary["phase_a_plus_b"]["agreed"], 1)

    def test_judge_rejection_turns_an_approval_into_a_rejection(self) -> None:
        case = make_case(kind="adversarial")
        summary = score(
            [make_row(case)], client(FakeTransport([judge_says("REJECTED")])), cases={case["id"]: case}
        )["summary"]

        self.assertEqual(summary["phase_a_plus_b"]["REJECTED"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["turned_rejected"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["false_approvals"], 0)
        self.assertEqual(summary["phase_a_plus_b"]["caught_by_judge"], 1)

    def test_judge_escalation_escalates_the_row(self) -> None:
        case = make_case(kind="normal")
        summary = score(
            [make_row(case)], client(FakeTransport([judge_says("ESCALATED")])), cases={case["id"]: case}
        )["summary"]

        self.assertEqual(summary["phase_a_plus_b"]["ESCALATED"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["turned_escalated"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["false_escalations"], 1)


class NoDraftAndReplayTests(unittest.TestCase):
    def test_no_draft_is_not_judged(self) -> None:
        case = make_case()
        row = make_row(case, verdict="NO_DRAFT", reply="", codes=())
        row["reply"] = None
        transport = FakeTransport([judge_says("APPROVED")])
        summary = score([row], client(transport), cases={case["id"]: case})["summary"]

        self.assertEqual(transport.calls, [])
        self.assertEqual(summary["phase_a"]["NO_DRAFT"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["judged"], 0)
        self.assertEqual(summary["reconciliation"]["no_draft"], 1)
        self.assertTrue(summary["reconciliation"]["balances"])

    def test_stored_refusal_is_carried_without_a_judge_call(self) -> None:
        case = make_case()
        transport = FakeTransport([judge_says("APPROVED")])
        row = make_row(case, verdict="REJECTED", codes=["NO_REPLY"])
        result = score([row], client(transport), cases={case["id"]: case})

        self.assertEqual(transport.calls, [])
        self.assertEqual(result["rows"][0]["phase_a_plus_b"]["verdict"], "REJECTED")
        self.assertFalse(result["rows"][0]["judged"])
        self.assertEqual(result["summary"]["phase_a_plus_b"]["REJECTED"], 1)

    def test_replay_mismatch_is_not_scored(self) -> None:
        case = make_case()
        # get_order succeeds on a fresh seed; a recorded decline cannot be this run.
        row = make_row(case, recorded={"ok": False, "code": "declined"})
        transport = FakeTransport([judge_says("APPROVED")])
        result = score([row], client(transport), cases={case["id"]: case})["summary"]

        self.assertEqual(transport.calls, [])
        self.assertEqual(result["reconciliation"]["not_scored"], 1)
        self.assertEqual(result["reconciliation"]["judged"], 0)
        self.assertTrue(result["reconciliation"]["balances"])
        self.assertEqual(result["phase_a_plus_b"]["claimed"], 0)


class TransportTests(unittest.TestCase):
    def test_invalid_output_escalates_once_and_is_not_retried(self) -> None:
        case = make_case()
        transport = FakeTransport([completion("MAYBE, it looks fine")])
        result = score([make_row(case)], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 1)
        summary = result["summary"]
        self.assertTrue(result["rows"][0]["invalid"])
        self.assertEqual(summary["phase_a_plus_b"]["invalid"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["ESCALATED"], 1)

    def test_transport_retries_429_with_backoff(self) -> None:
        case = make_case()
        transport = FakeTransport([(429, b"{}"), judge_says("APPROVED")])
        result = score([make_row(case)], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 2)
        self.assertEqual(result["summary"]["phase_a_plus_b"]["APPROVED"], 1)
        self.assertEqual(result["summary"]["phase_a_plus_b"]["invalid"], 0)

    def test_unavailable_after_retries_escalates_and_completes(self) -> None:
        case = make_case()
        transport = FakeTransport([(500, b"{}")])
        result = score([make_row(case)], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 3)  # the transport retry, not a judge retry
        summary = result["summary"]
        self.assertEqual(summary["phase_a_plus_b"]["unavailable"], 1)
        self.assertEqual(summary["phase_a_plus_b"]["ESCALATED"], 1)
        self.assertTrue(summary["reconciliation"]["balances"])


class OutputTests(unittest.TestCase):
    def test_manifest_carries_no_key(self) -> None:
        case = make_case()
        rows = [make_row(case)]
        judge = client(FakeTransport([judge_says("APPROVED")]))
        result = score(rows, judge, cases={case["id"]: case})
        with tempfile.TemporaryDirectory() as tmp:
            rows_path = Path(tmp) / "rows.jsonl"
            rows_path.write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
            written = write_outputs(
                tmp,
                "base",
                rows_path,
                result,
                model=judge.model,
                gateway=judge.base_url,
                session_id="judge-test-1",
                concurrency=1,
                calls=judge.requests,
                wall_seconds=0.5,
            )
            rows_text = written["rows"].read_text(encoding="utf-8")
            manifest_text = written["manifest"].read_text(encoding="utf-8")
        self.assertNotIn(SENTINEL, rows_text)
        self.assertNotIn(SENTINEL, manifest_text)
        manifest = json.loads(manifest_text)
        self.assertEqual(manifest["judge"]["session_id"], "judge-test-1")
        self.assertEqual(manifest["cost"]["usd"], 0.0)
        self.assertEqual(manifest["rows"], 1)
        self.assertEqual(manifest["counts"]["phase_a_plus_b"]["judged"], 1)

    def test_summary_rows_keep_input_order(self) -> None:
        first = make_case()
        second_case = {**make_case(), "id": "N-36"}
        rows = [
            make_row(first),
            make_row(second_case, verdict="NO_DRAFT", reply=None),
        ]
        transport = FakeTransport([judge_says("APPROVED")])
        cases = {first["id"]: first, second_case["id"]: second_case}
        result = score(rows, client(transport), cases=cases)
        self.assertEqual([row["id"] for row in result["rows"]], [first["id"], second_case["id"]])


class ArtifactShapeTests(unittest.TestCase):
    def test_load_rows_reads_the_bench_artifact(self) -> None:
        path = Path(__file__).resolve().parent.parent / "out" / "bench-base-284.jsonl"
        if not path.exists():
            self.skipTest("bench artifact not on disk")
        rows = load_rows(path)
        self.assertEqual(len(rows), 284)
        summary = summarise([])
        self.assertTrue(summary["reconciliation"]["balances"])


if __name__ == "__main__":
    unittest.main()
