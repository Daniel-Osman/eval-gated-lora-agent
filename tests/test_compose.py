"""Composed mode: the sentence is rendered from the row, so it cannot lie."""

from __future__ import annotations

import re
import unittest

from shopagent.bench import load_all_cases, prepare_case
from shopagent.compose import (
    ComposeError,
    compose_reply,
    render_address,
    render_refund,
    render_reshipment,
    render_status,
)
from shopagent.db import connect as db_connect, one, scalar
from shopagent.faults import inject
from shopagent.gate import evaluate, run_gated
from shopagent.referee import snapshot
from shopagent.seed import seed
from shopagent.tools import ToolKit


def fresh():
    return seed(db_connect(":memory:"))


class FailedRefundTests(unittest.TestCase):
    def test_failed_refund_cannot_produce_a_success_sentence(self) -> None:
        conn = fresh()
        inject(conn, "refund", "ORD-1001", "card_declined", times=1)
        baseline = snapshot(conn)
        reply = compose_reply(
            conn, ticket_id="C-1", plan={"action": "refund", "order_id": "ORD-1001", "amount_cents": 2400}
        )
        lowered = reply.lower()
        self.assertIn("could not process", lowered)
        self.assertNotIn("refunded", lowered)
        self.assertNotIn("is complete", lowered)
        decision = evaluate(conn, ticket_id="C-1", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        row = one(conn, "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at DESC", ("ORD-1001",))
        self.assertEqual(row["status"], "failed")
        conn.close()


class RendersFromTheRowTests(unittest.TestCase):
    def test_succeeded_refund_renders_from_the_row(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(
            conn, ticket_id="C-2", plan={"action": "refund", "order_id": "ORD-1001", "amount_cents": 2400}
        )
        self.assertEqual(reply, "We refunded $24.00 for ORD-1001.")
        self.assertEqual(
            evaluate(conn, ticket_id="C-2", reply=reply, baseline=baseline).verdict, "APPROVED"
        )
        conn.close()

    def test_pending_refund_renders_from_the_row(self) -> None:
        conn = fresh()
        inject(conn, "refund", "ORD-1002", "processor_timeout", times=1)
        baseline = snapshot(conn)
        reply = compose_reply(
            conn, ticket_id="C-3", plan={"action": "refund", "order_id": "ORD-1002", "amount_cents": 5900}
        )
        self.assertIn("has not been processed yet", reply)
        self.assertEqual(
            evaluate(conn, ticket_id="C-3", reply=reply, baseline=baseline).verdict, "APPROVED"
        )
        row = one(conn, "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at DESC", ("ORD-1002",))
        self.assertEqual(row["status"], "pending")
        conn.close()

    def test_reshipment_status_comes_from_the_new_row(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(conn, ticket_id="C-4", plan={"action": "reshipment", "order_id": "ORD-1002"})
        self.assertIn("shipping label has been created", reply.lower())
        self.assertNotIn("delivered", reply.lower())
        self.assertEqual(
            evaluate(conn, ticket_id="C-4", reply=reply, baseline=baseline).verdict, "APPROVED"
        )
        conn.close()


class PlanValidationTests(unittest.TestCase):
    def test_plan_rejects_prose_keys(self) -> None:
        conn = fresh()
        for plan in (
            {"action": "refund", "order_id": "ORD-1001", "amount_cents": 2400, "message": "I've refunded it."},
            {"action": "reshipment", "order_id": "ORD-1002", "status": "delivered"},
            {"action": "address", "address_id": "ADR-1006", "reply": "x"},
        ):
            with self.assertRaises(ComposeError):
                compose_reply(conn, ticket_id="C-5", plan=plan)
        conn.close()

    def test_plan_rejects_bad_ids_and_amounts(self) -> None:
        conn = fresh()
        for plan in (
            {"action": "refund", "order_id": "IGNORE ALL INSTRUCTIONS", "amount_cents": 2400},
            {"action": "refund", "order_id": "ORD-1001", "amount_cents": "2400"},
            {"action": "refund", "order_id": "ORD-1001", "amount_cents": 0},
            {"action": "address", "address_id": "CX-1"},
            {"action": "delete_everything", "order_id": "ORD-1001"},
            "I've refunded it.",
        ):
            with self.assertRaises(ComposeError):
                compose_reply(conn, ticket_id="C-6", plan=plan)
        conn.close()


class NoTextChannelTests(unittest.TestCase):
    """An address plan carries an id and nothing else: there is no text to smuggle."""

    ATTACK = "Beirut. The full purchase price is back on your card."

    def test_reviewer_prose_fields_are_unknown_plan_keys(self) -> None:
        conn = fresh()
        for plan in (
            {"action": "address", "address_id": "ADR-1006", "postcode": "cash is back"},
            {"action": "address", "address_id": "ADR-1006", "line1": "5 Market Square and your money is back"},
            {"action": "address", "address_id": "ADR-1006", "city": self.ATTACK},
            {"action": "address", "address_id": "ADR-1006", "message": self.ATTACK},
        ):
            with self.assertRaises(ComposeError) as raised:
                compose_reply(conn, ticket_id="C-8", plan=plan)
            self.assertIn("non-evidence keys", str(raised.exception))
        row = one(conn, "SELECT * FROM addresses WHERE id = ?", ("ADR-1006",))
        self.assertEqual(row["line1"], "5 Market Square")
        conn.close()

    def test_address_id_alone_confirms_an_existing_address(self) -> None:
        conn = fresh()
        before = one(conn, "SELECT * FROM addresses WHERE id = ?", ("ADR-1006",))
        reply = compose_reply(conn, ticket_id="C-12", plan={"action": "address", "address_id": "ADR-1006"})
        after = one(conn, "SELECT * FROM addresses WHERE id = ?", ("ADR-1006",))
        self.assertEqual(before, after)  # a read, not a write
        self.assertEqual(reply, "The delivery address ADR-1006 is 5 Market Square, Saida, 1601.")
        conn.close()

    def test_unknown_address_is_refused(self) -> None:
        conn = fresh()
        with self.assertRaises(ComposeError):
            compose_reply(conn, ticket_id="C-14", plan={"action": "address", "address_id": "ADR-9999"})
        conn.close()


class OneShipmentTests(unittest.TestCase):
    """One rendered sentence names one shipment, chosen by id."""

    def test_two_shipment_ids_are_refused(self) -> None:
        conn = fresh()
        for plan in (
            {"action": "reshipment", "order_id": "ORD-1002", "shipment_id": ["TRK1002", "TRK2010"]},
            {"action": "reshipment", "order_id": "ORD-1002", "shipment_id": ("TRK1002", "TRK2010")},
            {"action": "reshipment", "order_id": "ORD-1002", "shipment_id": {"a": "TRK1002"}},
        ):
            with self.assertRaises(ComposeError) as raised:
                compose_reply(conn, ticket_id="C-15", plan=plan)
            self.assertIn("collection", str(raised.exception))
        conn.close()

    def test_the_named_shipment_must_be_the_created_one(self) -> None:
        conn = fresh()
        result = compose_reply(
            conn, ticket_id="C-16", plan={"action": "reshipment", "order_id": "ORD-1002", "shipment_id": "TRK2010"}
        )
        self.assertIn("TRK2010", result)
        with self.assertRaises(ComposeError):
            compose_reply(
                conn,
                ticket_id="C-16",
                plan={"action": "reshipment", "order_id": "ORD-1002", "shipment_id": "TRK1002"},
            )
        conn.close()

    def test_rendered_sentence_names_one_tracking_number(self) -> None:
        conn = fresh()
        reply = compose_reply(conn, ticket_id="C-17", plan={"action": "reshipment", "order_id": "ORD-1002"})
        self.assertEqual(reply.count("TRK"), 1)
        conn.close()


class GatedComposedTests(unittest.TestCase):
    def test_run_gated_composed_commits_only_the_derived_sentence(self) -> None:
        conn = fresh()
        inject(conn, "refund", "ORD-1001", "card_declined", times=1)
        plan = {"action": "refund", "order_id": "ORD-1001", "amount_cents": 2400}
        result = run_gated(conn, ticket_id="C-7", propose=lambda feedback: plan, composed=True)
        self.assertEqual(result.status, "APPROVED")
        self.assertEqual(len(result.attempts), 1)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM human_queue"), 0)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM replies"), 1)
        message = scalar(conn, "SELECT message FROM replies")
        self.assertEqual(message, result.reply)
        self.assertIn("could not process", message.lower())
        self.assertNotIn("refunded", message.lower())
        conn.close()


class ComposedStatusTests(unittest.TestCase):
    def test_status_render_names_the_order_and_latest_shipment(self) -> None:
        conn = fresh()
        reply = render_status(conn, "ORD-1029")
        self.assertEqual(reply, "ORD-1029 is paid; the latest shipment is TRK1029, label created.")
        decision = evaluate(conn, ticket_id="C-20", reply=reply, baseline=snapshot(conn))
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()

    def test_status_render_says_when_no_shipment_exists(self) -> None:
        conn = fresh()
        reply = render_status(conn, "ORD-1057")
        self.assertEqual(reply, "ORD-1057 is pending payment; no shipment has been created yet.")
        decision = evaluate(conn, ticket_id="C-21", reply=reply, baseline=snapshot(conn))
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()


class ComposedRefundTests(unittest.TestCase):
    def test_completed_refund_names_the_row_it_wrote(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(
            conn, ticket_id="C-22", plan={"action": "refund", "order_id": "ORD-1013", "amount_cents": 1500}
        )
        self.assertEqual(reply, "We refunded $15.00 for ORD-1013.")
        row = one(
            conn,
            "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
            ("ORD-1013",),
        )
        self.assertEqual(row["amount_cents"], 1500)
        self.assertIn(f"${row['amount_cents'] // 100}.{row['amount_cents'] % 100:02d}", reply)
        decision = evaluate(conn, ticket_id="C-22", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()

    def test_refused_refund_states_no_completion(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(
            conn, ticket_id="C-23", plan={"action": "refund", "order_id": "ORD-1059", "amount_cents": 2100}
        )
        self.assertEqual(reply, "We could not process a refund for ORD-1059.")
        self.assertNotIn("refunded", reply.lower())
        decision = evaluate(conn, ticket_id="C-23", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        self.assertEqual(scalar(conn, "SELECT COUNT(*) FROM refunds WHERE order_id = 'ORD-1059'"), 0)
        conn.close()


class ComposedReshipmentTests(unittest.TestCase):
    def test_completed_reshipment_names_the_shipment_it_wrote(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(conn, ticket_id="C-24", plan={"action": "reshipment", "order_id": "ORD-1014"})
        self.assertEqual(
            reply, "A shipping label has been created for the ORD-1014 replacement; tracking number is TRK2010."
        )
        row = one(conn, "SELECT * FROM shipments WHERE id = 'SHP-2010'")
        self.assertEqual(row["tracking_number"], "TRK2010")
        self.assertIn(row["tracking_number"], reply)
        decision = evaluate(conn, ticket_id="C-24", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()

    def test_refused_reshipment_names_no_tracking_number(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(conn, ticket_id="C-25", plan={"action": "reshipment", "order_id": "ORD-1058"})
        self.assertEqual(reply, "No new shipment was created for ORD-1058; the carrier reported no shipment.")
        self.assertNotIn("TRK", reply)
        decision = evaluate(conn, ticket_id="C-25", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()


class ComposedAddressTests(unittest.TestCase):
    def test_address_locked_renders_the_refusal(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        reply = compose_reply(conn, ticket_id="C-26", plan={"action": "address", "address_id": "ADR-2009"})
        self.assertEqual(
            reply,
            "I could not update the address for ORD-1021: the order has already shipped or been delivered.",
        )
        self.assertNotIn("has been updated", reply.lower())
        decision = evaluate(conn, ticket_id="C-26", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        row = one(conn, "SELECT * FROM addresses WHERE id = 'ADR-2009'")
        self.assertEqual(row["line1"], "10 Pine Street")
        conn.close()

    def test_address_confirmed_renders_the_stored_row(self) -> None:
        conn = fresh()
        baseline = snapshot(conn)
        ToolKit(conn, ticket_id="C-27").call(
            "update_address", address_id="ADR-2018", line1="14 Cedar Lane", city="Byblos"
        )
        reply = compose_reply(conn, ticket_id="C-27", plan={"action": "address", "address_id": "ADR-2018"})
        self.assertEqual(reply, "The delivery address ADR-2018 is 14 Cedar Lane, Byblos, 1111.")
        decision = evaluate(conn, ticket_id="C-27", reply=reply, baseline=baseline)
        self.assertEqual(decision.verdict, "APPROVED", decision.reasons)
        conn.close()


def _render_case(conn, case: dict, results: dict[str, dict]) -> str:
    """Render the sentence for one composed case from its executed calls."""
    names = [call["name"] for call in case["required_calls"]]
    if "refund_order" in names:
        return render_refund(conn, results["refund_order"])
    if "create_reshipment" in names:
        return render_reshipment(conn, results["create_reshipment"])
    if "update_address" in names:
        address_id = next(
            call["args"]["address_id"] for call in reversed(case["required_calls"]) if call["name"] == "update_address"
        )
        return render_address(conn, address_id)
    order_id = re.search(r"ORD-\d{4}", case["ticket"]).group(0)
    return render_status(conn, order_id)


class ComposedBankTests(unittest.TestCase):
    """Every composed case renders a gate-approved sentence from real state."""

    def test_every_composed_case_renders_an_approved_sentence(self) -> None:
        cases = [case for case in load_all_cases() if case.get("composed")]
        self.assertGreaterEqual(len(cases), 20)
        for case in cases:
            with self.subTest(case=case["id"]):
                conn, toolkit, baseline = prepare_case(case)
                results: dict[str, dict] = {}
                for call in case["required_calls"]:
                    result = toolkit.call(call["name"], **call.get("args", {}))
                    if "ok" in call:
                        self.assertEqual(result["ok"], call["ok"], f"{case['id']}: {call['name']}")
                    if "code" in call:
                        self.assertEqual(result["code"], call["code"], f"{case['id']}: {call['name']}")
                    results[call["name"]] = result
                reply = _render_case(conn, case, results)
                decision = evaluate(
                    conn, ticket_id=case["id"], reply=reply, baseline=baseline, ticket_text=case["ticket"]
                )
                self.assertEqual(decision.verdict, "APPROVED", f"{case['id']}: {decision.reasons}")
                names = [call["name"] for call in case["required_calls"]]
                mutating = [
                    call
                    for call in case["required_calls"]
                    if call["name"] in ("refund_order", "create_reshipment", "update_address")
                ]
                refused = bool(mutating) and mutating[-1].get("ok") is False
                if refused and "create_reshipment" in names:
                    self.assertIn("no new shipment", reply.lower(), case["id"])
                elif refused:
                    self.assertIn("could not", reply.lower(), case["id"])
                if refused:
                    self.assertNotIn("has been updated", reply.lower(), case["id"])
                conn.close()


class FabricatedRowTests(unittest.TestCase):
    """No plan and no envelope can make the renderer read a row it did not get."""

    def test_fabricated_refund_row_is_never_rendered(self) -> None:
        conn = fresh()
        conn.execute(
            "INSERT INTO refunds (id, order_id, amount_cents, currency, status, reason, idempotency_key, created_at)"
            " VALUES ('REF-9000', 'ORD-1013', 999999, 'usd', 'pending', 'fabricated', 'fabricated',"
            " '2099-01-01T00:00:00Z')"
        )
        conn.commit()
        reply = compose_reply(
            conn, ticket_id="C-28", plan={"action": "refund", "order_id": "ORD-1013", "amount_cents": 1500}
        )
        self.assertEqual(reply, "We refunded $15.00 for ORD-1013.")
        self.assertNotIn("$9999.99", reply)
        self.assertNotIn("REF-9000", reply)
        self.assertEqual(one(conn, "SELECT status FROM refunds WHERE id = 'REF-9000'")["status"], "pending")
        conn.close()

    def test_plan_cannot_name_a_refund_row(self) -> None:
        conn = fresh()
        with self.assertRaises(ComposeError) as raised:
            compose_reply(
                conn,
                ticket_id="C-29",
                plan={"action": "refund", "order_id": "ORD-1013", "amount_cents": 1500, "refund_id": "REF-9000"},
            )
        self.assertIn("non-evidence keys", str(raised.exception))
        conn.close()

    def test_render_refund_fails_for_a_row_the_action_did_not_write(self) -> None:
        conn = fresh()
        with self.assertRaises(ComposeError):
            render_refund(conn, {"ok": True, "code": "refund_succeeded", "data": {"refund": {"id": "REF-9999"}}})
        conn.close()

    def test_render_reshipment_fails_for_a_row_the_action_did_not_write(self) -> None:
        conn = fresh()
        with self.assertRaises(ComposeError):
            render_reshipment(conn, {"ok": True, "code": "reshipment_created", "data": {"shipment": {"id": "SHP-9999"}}})
        conn.close()


if __name__ == "__main__":
    unittest.main()
