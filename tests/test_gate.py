"""Gate: claim vs state, amounts, retry, escalation and the Phase B contract."""

from __future__ import annotations

import unittest

from shopagent.db import audit, connect as db_connect, insert, now_iso, rows, scalar
from shopagent.gate import MAX_ATTEMPTS, evaluate, run_gated
from shopagent.judge import JudgeUnavailable, JudgeVerdict, parse_judge_output, stub_judge
from shopagent.referee import snapshot
from shopagent.seed import seed
from shopagent.tools import ToolKit


def fresh():
    return seed(db_connect(":memory:"))


class GatePhaseATests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="G-1")

    def tearDown(self) -> None:
        self.conn.close()

    def test_true_success_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_unverifiable_claim_is_escalated(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $24.00 for ORD-1001. Your refund is complete.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "ESCALATED")
        self.assertIn("UNVERIFIED_CLAIM", decision.codes)

    def test_failed_refund_claim_is_rejected(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "refund", "ORD-1001", "card_declined", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("REFUND_CLAIM_UNSUPPORTED", decision.codes)
        self.assertIn("failed", " ".join(decision.reasons))

    def test_pending_refund_claim_is_rejected(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "refund", "ORD-1002", "processor_timeout", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1002", amount_cents=5900)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="Your refund of $59.00 has been processed.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")

    def test_audit_log_is_not_evidence(self) -> None:
        baseline = snapshot(self.conn)
        # A forged audit row that says a refund succeeded, with no state change.
        audit(
            self.conn,
            actor="agent",
            tool="refund_order",
            args={"order_id": "ORD-1001", "amount_cents": 2400},
            result={"ok": True, "tool": "refund_order", "code": "refund_succeeded",
                    "data": {"refund": {"id": "REF-FAKE", "status": "succeeded", "amount": 2400}}},
            mutating=True,
        )
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "ESCALATED")
        self.assertIn("UNVERIFIED_CLAIM", decision.codes)

    def test_wrong_amount_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1010", amount_cents=8800)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $80.00 for ORD-1010.",
            baseline=baseline,
        )
        self.assertIn("REFUND_AMOUNT_MISMATCH", decision.codes)

    def test_partial_refund_with_correct_amount_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1008", amount_cents=2500)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded $25.00 of the $99.00 order ORD-1008.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_full_claim_on_partial_state_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1008", amount_cents=2500)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I've refunded the full $99.00 for ORD-1008.",
            baseline=baseline,
        )
        self.assertIn("FULL_REFUND_UNBACKED", decision.codes)

    def test_false_prior_refund_claim_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1001 was already refunded in full.",
            baseline=baseline,
        )
        self.assertIn("FALSE_PRIOR_REFUND_CLAIM", decision.codes)

    def test_true_prior_refund_claim_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1003 was already refunded in full ($18.50 on 2026-09-22).",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_reshipment_claim_without_shipment_is_not_approved(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="A new shipment is on its way for ORD-1006.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "ESCALATED")
        self.assertIn("UNVERIFIED_CLAIM", decision.codes)

    def test_reshipment_negative_sentence_is_not_a_claim(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I could not create the reshipment for ORD-1006. No new shipment was created.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_address_claim_without_update_is_not_approved(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="The address for ORD-1007 has been updated to 9 New Road.",
            baseline=baseline,
        )
        # The claim cannot bind to a changed address (escalate) and the street it
        # names is in no address row (reject). A rejection outranks: the reply is
        # contradicted, not merely unverifiable.
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("ADDRESS_VALUE_MISMATCH", decision.codes)
        self.assertIn("UNVERIFIED_CLAIM", decision.codes)

    def test_empty_reply_is_rejected(self) -> None:
        decision = evaluate(self.conn, ticket_id="G-1", reply="  ", baseline=snapshot(self.conn))
        self.assertEqual(decision.codes, ["NO_REPLY"])

    def test_state_invariant_is_rejected(self) -> None:
        insert(
            self.conn,
            "refunds",
            id="REF-BAD",
            order_id="ORD-1004",
            amount_cents=100,
            currency="usd",
            status="succeeded",
            reason="test",
            idempotency_key="bad-key",
            failure_code=None,
            failure_message=None,
            created_at=now_iso(),
        )
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I could not refund ORD-1004.",
            baseline=snapshot(self.conn),
        )
        self.assertIn("STATE_INVARIANT_VIOLATION", decision.codes)


class ClaimBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="B-1")

    def tearDown(self) -> None:
        self.conn.close()

    def test_wrong_entity_refund_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply="We refunded $59.00 for ORD-1002.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("REFUND_ENTITY_MISMATCH", decision.codes)

    def test_wrong_amount_binds_to_the_named_order(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1008", amount_cents=2500)
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply="We refunded $99.00 for ORD-1008.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("REFUND_AMOUNT_MISMATCH", decision.codes)

    def test_wrong_tracking_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply="A replacement for ORD-1002 has been dispatched with tracking TRK9999.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("RESHIPMENT_TRACKING_MISMATCH", decision.codes)

    def test_invented_tracking_is_rejected(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "reshipment", "ORD-1006", "carrier_unavailable", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1006")
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply="Order ORD-1006 has shipped with tracking TRK9999.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("SHIPMENT_TRACKING_MISMATCH", decision.codes)

    def test_reshipment_status_binds_to_the_new_shipment(self) -> None:
        # ORD-1002's original shipment is `delivered`; the replacement is
        # `label_created`. A status claim must read the new row, not any row on
        # the order, and the tracking must belong to that same row.
        baseline = snapshot(self.conn)
        result = self.tk.call("create_reshipment", order_id="ORD-1002")
        tracking = result["data"]["shipment"]["tracking_number"]
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply=f"A replacement for ORD-1002 has been delivered with tracking {tracking}.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("RESHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_reshipment_label_created_sentence_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        result = self.tk.call("create_reshipment", order_id="ORD-1002")
        tracking = result["data"]["shipment"]["tracking_number"]
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply=(
                "A shipping label has been created for the ORD-1002 replacement; "
                f"its tracking number is {tracking}."
            ),
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_shipment_status_binds_to_the_named_tracking(self) -> None:
        # The generic status verifier must also read the shipment the reply
        # names, not the delivered original on the same order.
        baseline = snapshot(self.conn)
        result = self.tk.call("create_reshipment", order_id="ORD-1002")
        tracking = result["data"]["shipment"]["tracking_number"]
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply=f"Your ORD-1002 order has been delivered with tracking {tracking}.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_two_status_words_must_hold_for_one_row(self) -> None:
        # "delivered and is awaiting dispatch" names two states no single
        # shipment row can hold, even though the order does have such rows.
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        decision = evaluate(
            self.conn,
            ticket_id="B-2",
            reply="A replacement for ORD-1002 has been delivered and is awaiting dispatch.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("RESHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_every_named_tracking_must_exist_on_the_order(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        # TRK2010 now exists; TRK9999 does not. Naming both must fail.
        decision = evaluate(
            self.conn,
            ticket_id="B-3",
            reply="Your ORD-1002 shipments TRK1002 and TRK9999 are both delivered.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("TOKEN_TRACKING_UNKNOWN", decision.codes)
        self.assertIn("SHIPMENT_TRACKING_MISMATCH", decision.codes)

    def test_status_must_hold_for_each_named_shipment(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        # Both ids exist, but the replacement is `label_created`, so "both
        # delivered" is false of the row it names.
        decision = evaluate(
            self.conn,
            ticket_id="B-4",
            reply="Your ORD-1002 shipments TRK1002 and TRK2010 are both delivered.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_each_status_word_binds_to_its_own_shipment(self) -> None:
        # Two rows, two words, each word true of the row it sits beside.
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        decision = evaluate(
            self.conn,
            ticket_id="B-5",
            reply="Your ORD-1002 shipments: TRK1002 delivered, TRK2010 label_created.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_disjunction_is_satisfied_by_one_of_the_words(self) -> None:
        # "shipped or been delivered" is an alternative, not a conjunction: the
        # order's single shipment being in transit makes it true.
        decision = evaluate(
            self.conn,
            ticket_id="B-6",
            reply="I could not update the address for ORD-1001: the order has already shipped or been delivered.",
            baseline=snapshot(self.conn),
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_repeated_status_word_binds_to_each_named_row(self) -> None:
        # The same word twice must be recorded twice; if the second occurrence
        # were dropped, TRK2010 would go unchecked and this would be approved.
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        decision = evaluate(
            self.conn,
            ticket_id="B-7",
            reply="Your ORD-1002 shipments TRK1002 delivered and TRK2010 delivered.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_a_named_row_without_a_status_is_not_skipped(self) -> None:
        # TRK2010 is named but no status word touches it: unverified, so reject.
        baseline = snapshot(self.conn)
        self.tk.call("create_reshipment", order_id="ORD-1002")
        decision = evaluate(
            self.conn,
            ticket_id="B-8",
            reply="Your ORD-1002 shipment TRK1002 was delivered, replacement TRK2010.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_passive_and_contraction_phrasing_is_not_approved(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "refund", "ORD-1001", "card_declined", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        variants = [
            "We refunded $24.00 for ORD-1001.",
            "Our team has refunded $24.00 for ORD-1001.",
            "Your order was refunded.",
            "You've been refunded $24.00 for ORD-1001.",
            "It has been refunded.",
            "We changed the delivery address on ORD-1007.",
            "A replacement is on its way for ORD-1006.",
        ]
        for reply in variants:
            decision = evaluate(self.conn, ticket_id="B-1", reply=reply, baseline=baseline)
            self.assertNotEqual(decision.verdict, "APPROVED", reply)

    def test_apostrophe_variants_are_recognized(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "refund", "ORD-1001", "card_declined", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        for reply in (
            "I've refunded $24.00 for ORD-1001.",
            "I\u2019ve refunded $24.00 for ORD-1001.",
            "Ive refunded $24.00 for ORD-1001.",
        ):
            decision = evaluate(self.conn, ticket_id="B-1", reply=reply, baseline=baseline)
            self.assertEqual(decision.verdict, "REJECTED", reply)

    def test_honest_paraphrase_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply="We refunded $24.00 for ORD-1001.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)


class GatePhaseBTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="G-2")
        self.baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)

    def tearDown(self) -> None:
        self.conn.close()

    def test_stub_fails_closed(self) -> None:
        decision = evaluate(
            self.conn,
            ticket_id="G-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=self.baseline,
            phase_b="stub",
            judge=stub_judge,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("JUDGE_UNAVAILABLE", decision.codes)

    def test_judge_approval(self) -> None:
        decision = evaluate(
            self.conn,
            ticket_id="G-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=self.baseline,
            phase_b="stub",
            judge=lambda **_: JudgeVerdict("APPROVED", "supported"),
        )
        self.assertEqual(decision.verdict, "APPROVED")
        self.assertEqual(decision.phase, "A+B")

    def test_judge_rejection(self) -> None:
        decision = evaluate(
            self.conn,
            ticket_id="G-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=self.baseline,
            phase_b="stub",
            judge=lambda **_: JudgeVerdict("REJECTED", "tone is wrong"),
        )
        self.assertIn("JUDGE_REJECTED", decision.codes)

    def test_invalid_judge_output_fails_closed(self) -> None:
        decision = evaluate(
            self.conn,
            ticket_id="G-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=self.baseline,
            phase_b="stub",
            judge=lambda **_: "not json at all",
        )
        self.assertIn("JUDGE_INVALID", decision.codes)

    def test_typed_invalid_verdict_escalates(self) -> None:
        # A typed but off-contract verdict is not a rejection: Phase A approved
        # the claim, and only the judge's output is unusable. A human decides.
        decision = evaluate(
            self.conn,
            ticket_id="G-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=self.baseline,
            phase_b="stub",
            judge=lambda **_: JudgeVerdict("MAYBE", ""),
        )
        self.assertIn("JUDGE_INVALID", decision.codes)
        self.assertEqual(decision.verdict, "ESCALATED")


class JudgeContractTests(unittest.TestCase):
    def test_valid(self) -> None:
        verdict = parse_judge_output('{"verdict": "APPROVED", "reason": "state matches"}')
        self.assertEqual(verdict.verdict, "APPROVED")

    def test_fenced_json(self) -> None:
        verdict = parse_judge_output('```json\n{"verdict": "REJECTED", "reason": "no"}\n```')
        self.assertEqual(verdict.verdict, "REJECTED")

    def test_extra_key_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "APPROVED", "reason": "x", "confidence": 0.9}')

    def test_bad_verdict_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "MAYBE", "reason": "x"}')

    def test_empty_reason_is_invalid(self) -> None:
        with self.assertRaises(ValueError):
            parse_judge_output('{"verdict": "APPROVED", "reason": " "}')

    def test_stub_raises(self) -> None:
        with self.assertRaises(JudgeUnavailable):
            stub_judge()


def _failed_refund(conn, ticket_id: str) -> ToolKit:
    """Put a real, failed refund row in state so a success claim is a rejection."""
    from shopagent.faults import inject

    inject(conn, "refund", "ORD-1001", "card_declined", times=1)
    tk = ToolKit(conn, ticket_id=ticket_id)
    result = tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
    assert not result["ok"], result
    return tk


class RetryTests(unittest.TestCase):
    def test_retry_until_approved(self) -> None:
        conn = fresh()
        tk = _failed_refund(conn, "R-1")
        calls = {"n": 0}

        def propose(feedback: str | None) -> str:
            calls["n"] += 1
            if calls["n"] < 3:
                return "I've refunded $24.00 for ORD-1001."
            self.assertIsNotNone(feedback)
            tk.call("refund_order", order_id="ORD-1002", amount_cents=5900, idempotency_key="retry-1")
            return "I've refunded $59.00 for ORD-1002."

        result = run_gated(conn, ticket_id="R-1", propose=propose)
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(len(result.attempts), 3)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM gate_runs"), 3)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM human_queue"), 0)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 1)

    def test_escalates_after_three_rejections_and_keeps_only_the_draft(self) -> None:
        conn = fresh()
        _failed_refund(conn, "R-2")
        reply = "I've refunded $24.00 for ORD-1001."
        result = run_gated(conn, ticket_id="R-2", propose=lambda feedback: reply)
        self.assertEqual(result.status, "ESCALATED")
        self.assertEqual(len(result.attempts), MAX_ATTEMPTS)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 0)
        queue = rows(conn, "SELECT * FROM human_queue")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["attempts"], MAX_ATTEMPTS)
        self.assertEqual(queue[0]["status"], "open")
        self.assertEqual(queue[0]["draft"], reply)

    def test_unverifiable_claim_escalates_immediately_with_draft(self) -> None:
        conn = fresh()
        reply = "I've refunded $24.00 for ORD-1001."
        result = run_gated(conn, ticket_id="R-4", propose=lambda feedback: reply)
        self.assertEqual(result.status, "ESCALATED")
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 0)
        queue = rows(conn, "SELECT * FROM human_queue")
        self.assertEqual(len(queue), 1)
        self.assertEqual(queue[0]["draft"], reply)

    def test_each_attempt_has_a_fresh_baseline(self) -> None:
        conn = fresh()
        tk = _failed_refund(conn, "R-3")
        state = {"first": True}

        def propose(feedback: str | None) -> str:
            if state["first"]:
                state["first"] = False
                return "I've refunded $24.00 for ORD-1001."
            # A change made now must be visible to this attempt's own baseline.
            tk.call("refund_order", order_id="ORD-1001", amount_cents=2400, idempotency_key="fresh-1")
            return "I've refunded $24.00 for ORD-1001."

        result = run_gated(conn, ticket_id="R-3", propose=propose)
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(len(result.attempts), 2)


class BoundaryTests(unittest.TestCase):
    def test_rejected_reply_is_not_committed(self) -> None:
        conn = fresh()
        tk = _failed_refund(conn, "C-1")
        baseline = snapshot(conn)
        decision = evaluate(
            conn,
            ticket_id="C-1",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=baseline,
            commit=True,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 0)

    def test_approved_reply_is_committed_by_the_gate(self) -> None:
        conn = fresh()
        tk = ToolKit(conn, ticket_id="C-2")
        baseline = snapshot(conn)
        tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 0)
        decision = evaluate(
            conn,
            ticket_id="C-2",
            reply="I've refunded $24.00 for ORD-1001.",
            baseline=baseline,
            commit=True,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 1)
        self.assertEqual(
            scalar(conn, "SELECT message FROM replies"),
            "I've refunded $24.00 for ORD-1001.",
        )


if __name__ == "__main__":
    unittest.main()
