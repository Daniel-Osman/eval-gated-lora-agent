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

    def test_cancelled_order_refund_refusal_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1059", amount_cents=2100)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="I could not refund ORD-1059: the order is cancelled and no payment has been captured.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

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


class LineItemPriceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()

    def tearDown(self) -> None:
        self.conn.close()

    def test_true_line_item_price_is_a_bound_amount(self) -> None:
        # ORD-1005: 2 x Wool blanket at $37.00 each, $74.00 total. A status
        # reply that lists the line item names a real row's price.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="Order ORD-1005 is partially refunded. Items: 2 x Wool blanket ($37.00).",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_wrong_line_item_price_is_rejected(self) -> None:
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="Order ORD-1005 is partially refunded. Items: 2 x Wool blanket ($38.00).",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED", decision.reasons)
        self.assertIn("AMOUNT_UNBOUND", decision.codes)


class NegatedShipmentStatusTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()

    def tearDown(self) -> None:
        self.conn.close()

    def test_negated_status_on_an_unshipped_order_is_approved(self) -> None:
        # ORD-1004 is pending payment and has no shipment row.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1004 has not shipped yet, so there is no shipment to replace.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_negated_status_on_a_shipped_order_is_rejected(self) -> None:
        # ORD-1001 is in transit; "has not shipped" is false there.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1001 has not shipped yet.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED", decision.reasons)
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_negated_delivery_on_an_in_transit_order_is_approved(self) -> None:
        # ORD-1001 is in transit: "has not been delivered" is true.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1001 has not been delivered yet.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_negated_delivery_on_a_delivered_order_is_rejected(self) -> None:
        # ORD-1010 is delivered: the negated predicate is false.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1010 has not been delivered yet.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED", decision.reasons)
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_negated_return_on_a_returned_order_is_rejected(self) -> None:
        # ORD-1011 is returned: "has not been returned" is false.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1011 has not been returned.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED", decision.reasons)
        self.assertIn("SHIPMENT_STATUS_UNSUPPORTED", decision.codes)

    def test_negated_return_on_a_delivered_order_is_approved(self) -> None:
        # ORD-1007 is delivered and not returned: the negation holds.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1007 has not been returned.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_shipment_named_in_its_own_clause_binds_its_own_order(self) -> None:
        # The status clause names SHP-1011 and no order: it must bind ORD-1011,
        # the order that parcel belongs to, not the order named a sentence ago.
        baseline = snapshot(self.conn)
        decision = evaluate(
            self.conn,
            ticket_id="G-1",
            reply="ORD-1007 was delivered. Shipment: SHP-1011 returned, tracking TRK1011.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)


class CompletionFillerTests(unittest.TestCase):
    """A completion adjective inside a typed claim is that claim's predicate.

    "Your refund is complete" names the refund the clause already states; it is
    not a second, untyped action. The typed claim still binds to state, so the
    filler buys no approval the action did not earn.
    """

    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="F-1")

    def tearDown(self) -> None:
        self.conn.close()

    def test_true_refund_with_completion_filler_is_approved(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1014", amount_cents=12875)
        decision = evaluate(
            self.conn,
            ticket_id="F-1",
            reply="I've refunded $128.75 for ORD-1014. Your refund is complete.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_settled_refund_with_completion_filler_is_approved(self) -> None:
        # The refund predates this run: the status clause binds to the row.
        decision = evaluate(
            self.conn,
            ticket_id="F-1",
            reply="ORD-1045 is refunded and the refund is complete.",
            baseline=snapshot(self.conn),
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

    def test_failed_refund_with_completion_filler_is_rejected(self) -> None:
        from shopagent.faults import inject

        inject(self.conn, "refund", "ORD-1014", "card_declined", times=1)
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1014", amount_cents=12875)
        decision = evaluate(
            self.conn,
            ticket_id="F-1",
            reply="I've refunded $128.75 for ORD-1014. Your refund is complete.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "REJECTED")
        self.assertIn("REFUND_CLAIM_UNSUPPORTED", decision.codes)

    def test_standalone_completion_phrase_still_escalates(self) -> None:
        # No typed action in the clause: the cue stands alone and a human looks.
        for reply in ("It's all set.", "We have taken care of it for ORD-1001."):
            decision = evaluate(
                self.conn, ticket_id="F-1", reply=reply, baseline=snapshot(self.conn)
            )
            self.assertEqual(decision.verdict, "ESCALATED", reply)
            self.assertIn("UNVERIFIED_CLAIM", decision.codes)

    def test_unrelated_action_after_a_typed_claim_still_escalates(self) -> None:
        # Only a completion filler about the typed action may be dropped. An
        # unrelated generic action in the same clause is still a claim.
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        for reply in (
            "We refunded $24.00 for ORD-1001 and resolved your billing dispute.",
            "We refunded $24.00 for ORD-1001 and closed the ticket.",
            "We refunded $24.00 for ORD-1001 and cancelled your subscription.",
        ):
            decision = evaluate(self.conn, ticket_id="F-1", reply=reply, baseline=baseline)
            self.assertEqual(decision.verdict, "ESCALATED", reply)
            self.assertIn("UNVERIFIED_CLAIM", decision.codes, reply)

    def test_bare_completion_filler_after_a_typed_claim_is_dropped(self) -> None:
        # "it is now complete" has no object of its own: it is the refund's
        # predicate, and the refund claim is still bound to state.
        baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        decision = evaluate(
            self.conn,
            ticket_id="F-1",
            reply="We refunded $24.00 for ORD-1001 and it is now complete.",
            baseline=baseline,
        )
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)


class UnverifiedActionTests(unittest.TestCase):
    """A handoff the state cannot record must not ride on a true refund.

    Nothing in the shop records "passed to a manager" or "noted". Each such
    action is an untyped claim and escalates, exactly like "resolved your
    billing dispute".
    """

    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="U-1")
        self.baseline = snapshot(self.conn)
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)

    def tearDown(self) -> None:
        self.conn.close()

    def _assert_escalates(self, tails: tuple[str, ...]) -> None:
        for tail in tails:
            reply = f"We refunded $24.00 for ORD-1001 and {tail}."
            decision = evaluate(
                self.conn, ticket_id="U-1", reply=reply, baseline=self.baseline
            )
            self.assertEqual(decision.verdict, "ESCALATED", reply)
            self.assertIn("UNVERIFIED_CLAIM", decision.codes, reply)

    def test_passed_to_someone_is_unverified(self) -> None:
        self._assert_escalates(("passed this to our manager", "passed it on"))

    def test_escalated_is_unverified(self) -> None:
        self._assert_escalates(("escalated your case", "escalated this to a manager"))

    def test_noted_is_unverified(self) -> None:
        self._assert_escalates(("noted your case", "noted it in the file"))

    def test_flagged_is_unverified(self) -> None:
        self._assert_escalates(("flagged this for review", "flagged your account"))

    def test_forwarded_is_unverified(self) -> None:
        self._assert_escalates(("forwarded this to billing", "forwarded it to the team"))

    def test_filed_is_unverified(self) -> None:
        self._assert_escalates(("filed a report", "filed this with the carrier"))


class ClaimBindingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = fresh()
        self.tk = ToolKit(self.conn, ticket_id="B-1")

    def tearDown(self) -> None:
        self.conn.close()

    def test_refund_record_binds_to_its_own_order_after_an_address_line(self) -> None:
        baseline = snapshot(self.conn)
        self.tk.call("update_address", address_id="ADR-2044", line1="9 Ash Lane", city="Tripoli")
        self.tk.call("refund_order", order_id="ORD-1027", amount_cents=5000)
        decision = evaluate(
            self.conn,
            ticket_id="B-1",
            reply=(
                "The delivery address for ORD-1056 has been updated to 9 Ash Lane, Tripoli. "
                "I've refunded $50.00 for ORD-1027. Refund REF-1014 is succeeded. "
                "The order is now partially refunded."
            ),
            baseline=baseline,
        )
        # "Refund REF-1014 is succeeded" names no order; the record's own order
        # decides, not the first order id in the reply (ORD-1056).
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)

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
