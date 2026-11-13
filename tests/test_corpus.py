"""Admission for the teacher corpus: free text and composed."""

from __future__ import annotations

import json
import unittest

from shopagent.bench import load_all_cases
from shopagent.corpus import Admission, composed_unsupported, generate
from shopagent.teacher import (
    GATEWAY,
    TeacherReply,
    build_reply_brief,
    build_scenario_brief,
    status_order_id,
)


class Scripted:
    """A client stand-in: returns canned texts in call order."""

    model = "deepseek-v4.1-flash"
    base_url = GATEWAY
    session_id = "run-test-1"

    def __init__(self, texts):
        self.texts = list(texts)
        self.prompts: list[str] = []

    def chat(self, system, user):
        self.prompts.append(user)
        return TeacherReply(text=self.texts.pop(0), model=self.model, attempts=1, latency_ms=1, status=200)


def stage1(ticket: str, plan) -> str:
    return json.dumps({"ticket": ticket, "plan": plan})


def stage2(reply: str) -> str:
    return json.dumps({"reply": reply})


def synthetic(**overrides) -> dict:
    case = {
        "id": "TX-01",
        "kind": "normal",
        "title": "Order status",
        "ticket": "Where is ORD-1001?",
        "faults": [],
        "required_calls": [{"name": "get_order", "args": {"order_id": "ORD-1001"}}],
        "forbidden_calls": [],
        "state_expect": [
            {"table": "orders", "match": {"id": "ORD-1001"}, "fields": {"status": "paid"}}
        ],
        "required_facts": ["ORD-1001"],
        "forbidden_phrases": [],
    }
    case.update(overrides)
    return case


def bank(case_id: str) -> dict:
    return next(case for case in load_all_cases() if case["id"] == case_id)


GET_1001 = [{"name": "get_order", "args": {"order_id": "ORD-1001"}}]


class FreeTextAdmissionTests(unittest.TestCase):
    def test_a_truthful_trace_is_admitted(self) -> None:
        client = Scripted([stage1("Where is ORD-1001, please?", GET_1001), stage2("Order ORD-1001 is paid.")])
        result = generate(client, synthetic())

        self.assertTrue(result.admitted, result.detail)
        self.assertEqual(result.path, "free_text")
        self.assertEqual(result.row["plan"], GET_1001)
        self.assertEqual(result.row["provenance"]["verdict"], "APPROVED")
        self.assertEqual(result.row["target"]["type"], "plan_and_reply")
        # The reply is built after the tool output exists: two calls, stage 2 last.
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("TOOL RESULTS", client.prompts[1])

    def test_state_mismatch_is_rejected(self) -> None:
        case = synthetic(
            state_expect=[
                {"table": "orders", "match": {"id": "ORD-1001"}, "fields": {"status": "refunded"}}
            ]
        )
        client = Scripted([stage1("Where is ORD-1001?", GET_1001), stage2("Order ORD-1001 is paid.")])
        result = generate(client, case)
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "state_mismatch")

    def test_missing_required_fact_is_rejected_before_the_gate(self) -> None:
        client = Scripted([stage1("Where is ORD-1001?", GET_1001), stage2("Your order is on its way.")])
        result = generate(client, synthetic())
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "facts_missing")
        self.assertIsNone(result.row)

    def test_forbidden_phrase_is_rejected(self) -> None:
        case = synthetic(forbidden_phrases=["already refunded"])
        client = Scripted(
            [stage1("Where is ORD-1001?", GET_1001), stage2("ORD-1001 was already refunded.")]
        )
        result = generate(client, case)
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "forbidden_phrase")

    def test_invalid_plan_shape_is_retried_once_then_rejected(self) -> None:
        plan = [{"name": "send_reply", "args": {"ticket_id": "TX-01", "message": "hi"}}]
        client = Scripted([stage1("Where is ORD-1001?", plan), stage1("Where is ORD-1001?", plan)])
        result = generate(client, synthetic())
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "invalid_plan")
        self.assertEqual(len(client.prompts), 2)
        self.assertIn("previous answer was rejected", client.prompts[1])
        self.assertIn('"plan": [', client.prompts[1])

    def test_plan_arguments_must_match_the_case(self) -> None:
        wrong = [{"name": "get_order", "args": {"order_id": "ORD-1002"}}]
        client = Scripted([stage1("Where is ORD-1001?", wrong)])
        result = generate(client, synthetic())
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "plan_mismatch")

    def test_gate_rejection_keeps_the_row_out(self) -> None:
        # The card is declined, so no refund row succeeds. A success claim is a
        # contradiction, not an unverifiable claim: the gate must reject it.
        refund = {"name": "refund_order", "args": {"order_id": "ORD-1001", "amount_cents": 2400}}
        case = synthetic(
            kind="adversarial",
            faults=[{"operation": "refund", "key": "ORD-1001", "mode": "card_declined", "times": 1}],
            required_calls=[{"name": "get_order", "args": {"order_id": "ORD-1001"}}, {**refund, "ok": False, "code": "card_declined"}],
            required_facts=["ORD-1001", "$24.00"],
            state_expect=[
                {
                    "table": "refunds",
                    "match": {"order_id": "ORD-1001", "status": "failed", "amount_cents": 2400},
                    "count": 1,
                }
            ],
        )
        client = Scripted(
            [
                stage1("Please refund $24.00 for ORD-1001.", [GET_1001[0], refund]),
                stage2("Your $24.00 refund for ORD-1001 has been processed; it should appear in a few days."),
            ]
        )
        result = generate(client, case)
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "gate:REJECTED")
        self.assertIsNone(result.row)

    def test_gate_escalation_keeps_the_row_out(self) -> None:
        client = Scripted(
            [
                stage1("Where is ORD-1001?", GET_1001),
                stage2("I have forwarded your question about ORD-1001 to a manager."),
            ]
        )
        result = generate(client, synthetic())
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "gate:ESCALATED")

    def test_parse_failure_keeps_the_raw_text_and_retries_once(self) -> None:
        client = Scripted(["not json at all", "still not json"])
        result = generate(client, synthetic())
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "parse_error")
        self.assertEqual(result.raw[0]["text"], "not json at all")
        self.assertEqual(len(result.raw), 2)
        self.assertTrue(result.raw[1].get("retry"))

    def test_a_shape_error_is_retried_and_can_be_admitted(self) -> None:
        client = Scripted(
            ["not json at all", stage1("Where is ORD-1001, please?", GET_1001), stage2("Order ORD-1001 is paid.")]
        )
        result = generate(client, synthetic())
        self.assertTrue(result.admitted, result.detail)
        self.assertEqual(len(client.prompts), 3)
        self.assertIn('"reply": "..."', client.prompts[2])

    def test_a_quality_failure_is_not_retried(self) -> None:
        # One stage-1 call and one stage-2 call. A missing fact is not resent.
        client = Scripted([stage1("Where is ORD-1001?", GET_1001), stage2("Your order is on its way.")])
        result = generate(client, synthetic())
        self.assertEqual(result.reason, "facts_missing")
        self.assertEqual(len(client.prompts), 2)

    def test_the_brief_lists_the_required_facts_for_free_text(self) -> None:
        case = synthetic(required_facts=["ORD-1001", "$24.00"])
        brief = build_scenario_brief(case, path="free_text")
        self.assertIn("must state these values", brief)
        self.assertIn("- ORD-1001", brief)
        self.assertIn("- $24.00", brief)
        self.assertNotIn("must state these values", build_scenario_brief(case, path="composed"))

    def test_the_brief_names_the_required_calls_with_their_arguments(self) -> None:
        case = synthetic(
            required_calls=[
                {"name": "get_order", "args": {"order_id": "ORD-1001"}},
                {"name": "refund_order", "args": {"order_id": "ORD-1001", "amount_cents": 2400}},
            ]
        )
        brief = build_scenario_brief(case, path="free_text")
        self.assertIn("needs exactly these tool calls", brief)
        self.assertIn('- get_order({"order_id": "ORD-1001"})', brief)
        self.assertIn('- refund_order({"amount_cents": 2400, "order_id": "ORD-1001"})', brief)

    def test_the_composed_brief_does_not_offer_address_text(self) -> None:
        case = bank("N-86")
        brief = build_scenario_brief(case, path="composed")
        self.assertIn("Address row: ADR-2008", brief)
        self.assertNotIn("New street", brief)
        self.assertIn("New street", build_scenario_brief(case, path="free_text"))

    def test_the_reply_brief_repeats_the_required_facts(self) -> None:
        case = synthetic(required_facts=["ORD-1001", "in transit"])
        brief = build_reply_brief(
            ticket="Where is ORD-1001?",
            plan=GET_1001,
            results=[{"tool": "get_order", "args": {"order_id": "ORD-1001"}, "result": {}}],
            required_facts=case["required_facts"],
        )
        self.assertIn("TOOL RESULTS", brief)
        self.assertIn("must state these values", brief)
        self.assertIn("- in transit", brief)


class ComposedAdmissionTests(unittest.TestCase):
    def test_refund_is_admitted(self) -> None:
        plan = {"action": "refund", "order_id": "ORD-1047", "amount_cents": 9900}
        client = Scripted([stage1("I would like $99.00 back for ORD-1047.", plan)])
        result = generate(client, bank("N-81"))
        self.assertTrue(result.admitted, result.detail)
        self.assertEqual(result.path, "composed")
        self.assertIsNone(result.row["reply"])
        self.assertEqual(result.row["target"], {"type": "plan", "plan": plan})
        self.assertTrue(result.row["rendered_reply"])

    def test_reshipment_is_admitted(self) -> None:
        plan = {"action": "reshipment", "order_id": "ORD-1002", "shipment_id": "TRK2010"}
        client = Scripted([stage1("My ORD-1002 replacement never arrived. Please send it again.", plan)])
        result = generate(client, bank("N-88"))
        self.assertTrue(result.admitted, result.detail)

    def test_locked_address_is_admitted(self) -> None:
        plan = {"action": "address", "address_id": "ADR-2008"}
        client = Scripted([stage1("Please change the address on ORD-1020.", plan)])
        result = generate(client, bank("N-86"))
        self.assertTrue(result.admitted, result.detail)
        self.assertIn("could not update the address", result.row["rendered_reply"])

    def test_status_is_admitted(self) -> None:
        plan = {"action": "status", "order_id": "ORD-1029"}
        client = Scripted([stage1("Where is ORD-1029?", plan)])
        result = generate(client, bank("N-99"))
        self.assertTrue(result.admitted, result.detail)

    def test_status_id_comes_from_state_expect_when_there_are_no_calls(self) -> None:
        for case_id, order_id in (
            ("N-99", "ORD-1029"),
            ("N-100", "ORD-1037"),
            ("N-101", "ORD-1057"),
            ("N-102", "ORD-1051"),
        ):
            with self.subTest(case=case_id):
                case = bank(case_id)
                self.assertEqual(case["required_calls"], [])
                self.assertEqual(status_order_id(case), order_id)

    def test_wrong_action_is_a_plan_mismatch(self) -> None:
        plan = {"action": "status", "order_id": "ORD-1047"}
        client = Scripted([stage1("Where is ORD-1047?", plan)])
        result = generate(client, bank("N-81"))
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "plan_mismatch")

    def test_wrong_amount_is_a_plan_mismatch(self) -> None:
        plan = {"action": "refund", "order_id": "ORD-1047", "amount_cents": 8800}
        client = Scripted([stage1("I would like $88.00 back for ORD-1047.", plan)])
        result = generate(client, bank("N-81"))
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "plan_mismatch")

    def test_shipment_id_must_belong_to_the_order(self) -> None:
        plan = {"action": "reshipment", "order_id": "ORD-1002", "shipment_id": "TRK9999"}
        client = Scripted([stage1("Please resend ORD-1002.", plan)])
        result = generate(client, bank("N-88"))
        self.assertFalse(result.admitted)
        self.assertEqual(result.reason, "plan_mismatch")


class UnsupportedComposedTests(unittest.TestCase):
    def test_the_rule_flags_exactly_three_bank_cases(self) -> None:
        flagged = {case["id"] for case in load_all_cases() if composed_unsupported(case)}
        self.assertEqual(flagged, {"N-83", "N-98", "N-110"})

    def test_locked_address_cases_are_supported(self) -> None:
        for case_id in ("N-86", "N-109"):
            with self.subTest(case=case_id):
                self.assertIsNone(composed_unsupported(bank(case_id)))

    def test_an_address_write_routes_to_free_text_with_expectations_untouched(self) -> None:
        case = bank("N-110")
        plan = [
            {"name": "get_order", "args": {"order_id": "ORD-1030"}},
            {
                "name": "update_address",
                "args": {"address_id": "ADR-2018", "line1": "14 Cedar Lane", "city": "Byblos"},
            },
        ]
        client = Scripted(
            [
                stage1("Please send my ORD-1030 order to 14 Cedar Lane, Byblos.", plan),
                stage2("The address for ORD-1030 is now 14 Cedar Lane, Byblos."),
            ]
        )
        result = generate(client, case)
        self.assertEqual(result.path, "free_text")
        self.assertTrue(result.admitted, result.detail)
        self.assertEqual(result.row["provenance"]["routed"], "unsupported_composed")
        self.assertEqual(case["state_expect"][0]["fields"]["line1"], "14 Cedar Lane")

    def test_the_rule_routes_without_changing_the_case(self) -> None:
        case = bank("N-83")
        before = json.dumps(case, sort_keys=True)
        client = Scripted([stage1("Refund ORD-1051 please.", [{"name": "get_order", "args": {"order_id": "ORD-1051"}}])])
        result = generate(client, case)
        self.assertEqual(result.path, "free_text")
        self.assertEqual(json.dumps(case, sort_keys=True), before)


if __name__ == "__main__":
    unittest.main()
