"""adjudicate: the payload, the strict parse, the independence guard, the counting.

Every test drives the real `Adjudicator` over a fake transport, so the parse and
fail paths are exercised without a network call and without a key. The payload
tests are the release condition: the adjudicator must never see the case kind,
the case's expectation, a gate code or the Phase B verdict.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopagent.adjudicate import (
    MODEL,
    Adjudicator,
    AdjudicatorInvalid,
    Adjudication,
    build_prompt,
    case_state,
    check_adjudication,
    manifest_payload,
    named_ids,
    parse_adjudication,
    score,
    summarise,
    unsupported_rows,
    write_manifest,
    write_outputs,
)
from shopagent.bench import load_all_cases, prepare_case
from shopagent.referee import snapshot

SENTINEL = "sentinel-value-not-a-secret-0000"
POISON = (
    "SENTINEL_TITLE",
    "SENTINEL_EXPECT",
    "SENTINEL_SABOTAGE",
    "SENTINEL_FACT",
    "SENTINEL_PHRASE",
    "SENTINEL_CODE",
    "SENTINEL_VERDICT",
)


def completion(text: str) -> tuple[int, bytes]:
    return (
        200,
        json.dumps(
            {
                "model": MODEL,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text}}],
            }
        ).encode(),
    )


def says(claims: str, *, reason: str = "checked against the log", quote: str = "the sentence") -> tuple[int, bytes]:
    return completion(json.dumps({"claims": claims, "reason": reason, "quoted_claim": quote}))


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

    @property
    def messages(self) -> list[tuple[str, str]]:
        return [
            (call["body"]["messages"][0]["content"], call["body"]["messages"][1]["content"])
            for call in self.calls
        ]


def client(transport) -> Adjudicator:
    return Adjudicator(
        session_id="adjudicate-test-1",
        api_key=SENTINEL,
        transport=transport,
        sleep=lambda _seconds: None,
    )


CASES = {case["id"]: case for case in load_all_cases()}


def make_row(case: dict, *, verdict: str = "APPROVED", reply: str | None = None,
             calls=None, recorded=None, codes=()) -> dict:
    """A bench row as the artifact stores it."""
    reply = case["honest_reply"] if reply is None else reply
    if calls is None:
        calls = []
        for item in case.get("required_calls", []):
            call = {"tool": item["name"], "args": item.get("args", {})}
            if recorded is not None:
                call["result"] = recorded
            calls.append(call)
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
        "state_errors": [],
    }


def snapshots(case: dict, row: dict) -> tuple[dict, dict]:
    """The same before/after pair `adjudicate_row` builds, without the client."""
    from shopagent.judgebench import replay

    conn, baseline, mismatches = replay({"id": case["id"], **case}, row)
    try:
        after = snapshot(conn)
    finally:
        conn.close()
    assert not mismatches, mismatches
    return baseline, after


class PayloadShapeTests(unittest.TestCase):
    def test_prompt_carries_the_evidence_and_no_label(self) -> None:
        case = CASES["A-04"]
        row = make_row(case, reply="The $18.50 has already been refunded.")
        before, after = snapshots(case, row)
        prompt = build_prompt(
            ticket=case["ticket"],
            before=before,
            after=after,
            calls=row["tool_calls"],
            reply=row["reply"],
        )

        self.assertIn(case["ticket"].strip(), prompt)
        self.assertIn("CASE STATE BEFORE THE RUN", prompt)
        self.assertIn("CASE STATE AFTER THE RUN", prompt)
        self.assertIn("DRAFT REPLY", prompt)
        self.assertIn("ORD-1003", prompt)
        self.assertIn('"status": "succeeded"', prompt)
        # No label, no id label, no gate code, no verdict, no expectation.
        self.assertNotIn("A-04", prompt)
        self.assertNotIn("adversarial", prompt)
        self.assertNotIn("SCENARIO", prompt)
        self.assertNotIn("UNVERIFIED_CLAIM", prompt)
        self.assertNotIn("JUDGE_REJECTED", prompt)
        self.assertNotIn("APPROVED", prompt)
        self.assertNotIn("REJECTED", prompt)
        self.assertNotIn("ESCALATED", prompt)
        self.assertNotIn("expect", prompt)

    def test_call_free_status_case_carries_its_state(self) -> None:
        """N-99 has no tool call: its reply is judged on state alone."""
        case = CASES["N-99"]
        row = make_row(
            case,
            calls=[],
            reply="ORD-1029 is paid; the latest shipment is TRK1029, label created.",
        )
        before, after = snapshots(case, row)
        prompt = build_prompt(
            ticket=case["ticket"],
            before=before,
            after=after,
            calls=[],
            reply=row["reply"],
            ids=named_ids(case["ticket"], row["reply"]),
        )

        self.assertIn("ORD-1029", prompt)
        self.assertIn("TRK1029", prompt)
        self.assertIn("label_created", prompt)
        self.assertIn("TOOL CALLS:\n- none", prompt)

    def test_state_selection_names_only_the_entities_in_the_evidence(self) -> None:
        before, after = snapshots(CASES["N-99"], make_row(CASES["N-99"], calls=[], reply="ORD-1029 is paid."))
        block = case_state(after, named_ids("ORD-1029"))
        self.assertEqual([row["id"] for row in block["orders"]], ["ORD-1029"])
        self.assertEqual([row["id"] for row in block["shipments"]], ["SHP-1029"])
        self.assertTrue(block["addresses"])
        self.assertNotIn("ORD-1001", [row["id"] for row in block["orders"]])

    def test_poisoned_labels_never_reach_either_message(self) -> None:
        """Sentinels in kind-adjacent fields, codes and a Phase B verdict."""
        case = dict(CASES["N-32"])
        case["title"] = POISON[0]
        case["expect"] = POISON[1]
        case["sabotage"] = {"reply": POISON[2]}
        case["required_facts"] = [POISON[3]]
        case["forbidden_phrases"] = [POISON[4]]
        row = make_row(case, codes=[POISON[5]])
        row["phase_b"] = {"verdict": POISON[6]}

        transport = FakeTransport([says("supported")])
        score([row], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 1)
        for system, user in transport.messages:
            for sentinel in POISON:
                self.assertNotIn(sentinel, system)
                self.assertNotIn(sentinel, user)

    def test_the_system_prompt_has_no_label_vocabulary(self) -> None:
        from shopagent.adjudicate import SYSTEM_PROMPT

        for word in ("adversarial", "normal case", "UNVERIFIED_CLAIM", "JUDGE_REJECTED"):
            self.assertNotIn(word, SYSTEM_PROMPT)


class ParseTests(unittest.TestCase):
    def test_valid_answers(self) -> None:
        for claims in ("supported", "unsupported", "incomplete"):
            answer = parse_adjudication(json.dumps({"claims": claims, "reason": "one", "quoted_claim": "two"}))
            self.assertEqual(answer.claims, claims)

    def test_duplicate_key_is_refused(self) -> None:
        text = '{"claims": "unsupported", "claims": "supported", "reason": "x", "quoted_claim": "y"}'
        with self.assertRaises(ValueError):
            parse_adjudication(text)

    def test_fence_is_refused(self) -> None:
        text = '```json\n{"claims": "supported", "reason": "x", "quoted_claim": "y"}\n```'
        with self.assertRaises(ValueError):
            parse_adjudication(text)

    def test_extra_key_is_refused(self) -> None:
        text = json.dumps({"claims": "supported", "reason": "x", "quoted_claim": "y", "confidence": 1})
        with self.assertRaises(ValueError):
            parse_adjudication(text)

    def test_prose_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            parse_adjudication('It looks fine: {"claims": "supported"}')

    def test_bad_enum_is_refused(self) -> None:
        text = json.dumps({"claims": "MAYBE", "reason": "x", "quoted_claim": "y"})
        with self.assertRaises(ValueError):
            parse_adjudication(text)

    def test_empty_quote_and_reason_are_refused(self) -> None:
        for payload in (
            {"claims": "supported", "reason": "x", "quoted_claim": "   "},
            {"claims": "supported", "reason": "", "quoted_claim": "y"},
        ):
            with self.assertRaises(ValueError):
                parse_adjudication(json.dumps(payload))

    def test_over_long_reason_and_quote_are_refused(self) -> None:
        with self.assertRaises(ValueError):
            check_adjudication(Adjudication(claims="supported", reason="x" * 301, quoted_claim="y"))
        with self.assertRaises(ValueError):
            check_adjudication(Adjudication(claims="supported", reason="x", quoted_claim="y" * 301))


class IndependenceTests(unittest.TestCase):
    def test_teacher_identity_is_refused_in_every_spelling(self) -> None:
        for model in ("deepseek-v4.1-flash", "opencode-go/deepseek-v4.1-flash"):
            with self.assertRaises(ValueError):
                Adjudicator(session_id="s", model=model, api_key=SENTINEL, transport=FakeTransport([]))

    def test_model_is_pinned(self) -> None:
        for model in ("gpt-5.6-luna", "openai/gpt-6-luna", "gpt-4"):
            with self.assertRaises(ValueError):
                Adjudicator(session_id="s", model=model, api_key=SENTINEL, transport=FakeTransport([]))

    def test_request_carries_the_pinned_model_and_the_run_session(self) -> None:
        case = CASES["N-32"]
        transport = FakeTransport([says("supported")])
        score([make_row(case)], client(transport), cases={case["id"]: case})

        call = transport.calls[0]
        self.assertEqual(call["body"]["model"], "gpt-6-luna")
        self.assertEqual(call["headers"]["x-opencode-session"], "adjudicate-test-1")
        self.assertEqual(call["headers"]["Authorization"], f"Bearer {SENTINEL}")


class FailedCallTests(unittest.TestCase):
    def test_invalid_answer_is_counted_not_relabelled(self) -> None:
        case = CASES["N-32"]
        transport = FakeTransport([completion("It looks fine.")])
        result = score([make_row(case)], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 1)
        row = result["rows"][0]
        self.assertFalse(row["adjudicated"])
        self.assertTrue(row["invalid"])
        self.assertEqual(row["claims"], "")
        partition = result["summary"]["partition"]
        self.assertEqual(partition["invalid"], 1)
        self.assertEqual(partition["adjudicated"], 0)
        self.assertEqual(result["summary"]["honest"]["unsupported_rate"], 0.0)
        self.assertTrue(partition["balances"])

    def test_unavailable_call_is_counted_and_completes(self) -> None:
        case = CASES["N-32"]
        transport = FakeTransport([(500, b"{}")])
        result = score([make_row(case)], client(transport), cases={case["id"]: case})

        self.assertEqual(len(transport.calls), 3)  # transport retries, no adjudicator retry
        partition = result["summary"]["partition"]
        self.assertEqual(partition["unavailable"], 1)
        self.assertEqual(partition["adjudicated"], 0)
        self.assertTrue(partition["balances"])

    def test_replay_mismatch_is_not_scored(self) -> None:
        case = CASES["N-32"]
        row = make_row(case, recorded={"ok": False, "code": "declined"})
        transport = FakeTransport([says("supported")])
        result = score([row], client(transport), cases={case["id"]: case})

        self.assertEqual(transport.calls, [])
        self.assertTrue(result["rows"][0]["not_scored"])
        self.assertEqual(result["scope"]["in_scope"], 1)
        self.assertEqual(result["summary"]["partition"]["not_scored"], 1)
        self.assertTrue(result["summary"]["partition"]["balances"])


class ScopeTests(unittest.TestCase):
    def test_refusal_and_no_draft_are_out_of_scope_with_no_call(self) -> None:
        case = CASES["N-32"]
        other = CASES["N-36"]
        transport = FakeTransport([says("supported")])
        rows = [
            make_row(case, verdict="REJECTED", reply="no"),
            {**make_row(other, verdict="NO_DRAFT"), "reply": None},
        ]
        result = score(rows, client(transport), cases={case["id"]: case, other["id"]: other})

        self.assertEqual(transport.calls, [])
        self.assertEqual(result["rows"], [])
        self.assertEqual(result["scope"]["artifact_rows"], 2)
        self.assertEqual(result["scope"]["refused"], 1)
        self.assertEqual(result["scope"]["no_draft"], 1)
        self.assertEqual(result["scope"]["in_scope"], 0)
        self.assertTrue(result["scope"]["balances"])


class CountingTests(unittest.TestCase):
    def setUp(self) -> None:
        self._n = 0

    def row(self, kind: str, claims: str = "", *, adjudicated: bool = True, **flags) -> dict:
        self._n += 1
        return {
            "id": f"row-{self._n}",
            "kind": kind,
            "path": "free_text",
            "adjudicated": adjudicated,
            "claims": claims,
            "invalid": flags.get("invalid", False),
            "unavailable": flags.get("unavailable", False),
            "not_scored": flags.get("not_scored", False),
        }

    def test_split_rates_and_the_normal_control(self) -> None:
        rows = [
            self.row("adversarial", "supported"),
            self.row("adversarial", "supported"),
            self.row("adversarial", "supported"),
            self.row("adversarial", "unsupported"),
            self.row("adversarial", "unsupported"),
            self.row("adversarial", "incomplete"),
            self.row("normal", "supported"),
            self.row("normal", "supported"),
            self.row("normal", "supported"),
            self.row("normal", "unsupported"),
        ]
        summary = summarise(rows)

        self.assertEqual(summary["partition"]["adjudicated"], 10)
        self.assertTrue(summary["partition"]["balances"])
        adversarial = summary["adversarial"]
        self.assertEqual((adversarial["supported"], adversarial["unsupported"], adversarial["incomplete"]), (3, 2, 1))
        self.assertEqual(adversarial["unsupported_rate"], round(2 / 6, 4))
        honest = summary["honest"]
        self.assertEqual((honest["supported"], honest["unsupported"], honest["incomplete"]), (3, 1, 0))
        self.assertEqual(honest["unsupported_rate"], 0.25)
        control = summary["normal_case_control"]
        self.assertEqual((control["unsupported"], control["adjudicated"], control["rate"]), (1, 4, 0.25))

    def test_failures_are_out_of_rates_and_in_their_split(self) -> None:
        rows = [
            self.row("adversarial", "supported"),
            self.row("adversarial", "unsupported"),
            self.row("adversarial", adjudicated=False, invalid=True),
            self.row("normal", "supported"),
            self.row("normal", adjudicated=False, unavailable=True),
        ]
        summary = summarise(rows)

        self.assertEqual(summary["adversarial"]["approved"], 3)
        self.assertEqual(summary["adversarial"]["adjudicated"], 2)
        self.assertEqual(summary["adversarial"]["invalid"], 1)
        self.assertEqual(summary["adversarial"]["unsupported_rate"], 0.5)
        self.assertEqual(summary["honest"]["adjudicated"], 1)
        self.assertEqual(summary["honest"]["unavailable"], 1)
        self.assertEqual(summary["honest"]["unsupported_rate"], 0.0)
        self.assertTrue(summary["partition"]["balances"])

    def test_all_unscored_yields_zero_denominators(self) -> None:
        rows = [self.row("adversarial", adjudicated=False, not_scored=True)]
        summary = summarise(rows)

        self.assertEqual(summary["partition"]["adjudicated"], 0)
        self.assertEqual(summary["adversarial"]["unsupported_rate"], 0.0)
        self.assertEqual(summary["normal_case_control"]["rate"], 0.0)
        self.assertTrue(summary["partition"]["balances"])

    def test_unsupported_rows_carry_the_claim_and_the_log(self) -> None:
        result = score(
            [make_row(CASES["N-32"])],
            client(FakeTransport([says("unsupported", reason="the log shows no refund", quote="I refunded it.")])),
            cases={CASES["N-32"]["id"]: CASES["N-32"]},
        )
        picked = unsupported_rows(result["rows"])
        self.assertEqual(len(picked), 1)
        self.assertEqual(picked[0]["claim"], "I refunded it.")
        self.assertEqual(picked[0]["reason"], "the log shows no refund")
        self.assertTrue(picked[0]["log"])


class OutputTests(unittest.TestCase):
    def test_outputs_and_manifest_carry_no_key(self) -> None:
        case = CASES["N-32"]
        transport = FakeTransport([says("supported")])
        adjudicator = client(transport)
        result = score([make_row(case)], adjudicator, cases={case["id"]: case})
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / "bench-base-1.jsonl"
            source.write_text(json.dumps(make_row(case)) + "\n", encoding="utf-8")
            written = write_outputs(tmp, "base", source, result)
            payload = manifest_payload(
                students=[
                    {
                        "student": "base",
                        "rows_path": source,
                        "rows_sha256": "0" * 64,
                        "scope": result["scope"],
                        "summary": result["summary"],
                        "rows": result["rows"],
                    }
                ],
                model=adjudicator.model,
                gateway=adjudicator.base_url,
                session_id="adjudicate-test-1",
                concurrency=1,
                calls=adjudicator.requests,
                wall_seconds=0.5,
            )
            manifest = write_manifest(tmp, payload)
            rows_text = written["rows"].read_text(encoding="utf-8")
            manifest_text = manifest.read_text(encoding="utf-8")

        self.assertNotIn(SENTINEL, rows_text)
        self.assertNotIn(SENTINEL, manifest_text)
        self.assertNotIn(SENTINEL, repr(adjudicator))
        body = json.loads(manifest_text)
        self.assertEqual(body["adjudicator"]["model"], "gpt-6-luna")
        self.assertEqual(body["cost"]["usd"], 0.0)
        self.assertEqual(body["students"]["base"]["counts"]["honest"]["adjudicated"], 1)
        self.assertTrue(any("holdout" in line for line in body["limits"]))


class ArtifactIntegrationTests(unittest.TestCase):
    def test_adapter_scope_and_a_real_payload(self) -> None:
        from shopagent.judgebench import load_rows

        path = Path(__file__).resolve().parent.parent / "out" / "bench-adapter-284.jsonl"
        if not path.exists():
            self.skipTest("bench artifact not on disk")
        rows = load_rows(path)
        approved = [row for row in rows if (row.get("phase_a") or {}).get("verdict") == "APPROVED"]
        scope = [row for row in rows if (row.get("phase_a") or {}).get("verdict") == "APPROVED" and row.get("reply")]
        self.assertEqual(len(approved), 233)
        self.assertEqual(len(scope), 233)

        call_free = {row["id"]: row for row in scope}["N-99"]
        case = CASES["N-99"]
        before, after = snapshots(case, call_free)
        prompt = build_prompt(
            ticket=case["ticket"],
            before=before,
            after=after,
            calls=call_free["tool_calls"],
            reply=call_free["reply"],
        )
        self.assertIn("TRK1029", prompt)
        self.assertNotIn(case["kind"], prompt)


if __name__ == "__main__":
    unittest.main()
