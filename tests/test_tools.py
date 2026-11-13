"""Tool layer and billing behaviour. No model, no gate."""

from __future__ import annotations

import json
import unittest

from shopagent.db import connect, rows, scalar
from shopagent.faults import inject
from shopagent.seed import seed
from shopagent.tools import ToolKit


class ToolTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = seed(connect(":memory:"))
        self.tk = ToolKit(self.conn, ticket_id="T-1")

    def tearDown(self) -> None:
        self.conn.close()

    def refunds(self, order_id: str) -> list[dict]:
        return rows(self.conn, "SELECT * FROM refunds WHERE order_id = ? ORDER BY id", (order_id,))

    def test_seed_is_fixed(self) -> None:
        self.assertEqual(scalar(self.conn, "SELECT COUNT(*) FROM orders"), 60)
        self.assertEqual(scalar(self.conn, "SELECT COUNT(*) FROM refunds"), 13)
        self.assertEqual(scalar(self.conn, "SELECT COUNT(*) FROM shipments"), 52)

    def test_refund_success_updates_order_and_audits(self) -> None:
        result = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "refund_succeeded")
        self.assertEqual(result["data"]["order"]["status"], "refunded")
        self.assertEqual(len(self.refunds("ORD-1001")), 1)
        audit = rows(self.conn, "SELECT * FROM audit_log ORDER BY seq")
        self.assertEqual(audit[-1]["tool"], "refund_order")
        self.assertEqual(audit[-1]["ok"], 1)
        self.assertEqual(json.loads(audit[-1]["args_json"])["amount_cents"], 2400)

    def test_partial_then_full_refund(self) -> None:
        first = self.tk.call("refund_order", order_id="ORD-1008", amount_cents=2500)
        self.assertTrue(first["ok"])
        self.assertEqual(first["data"]["order"]["status"], "partially_refunded")
        second = self.tk.call("refund_order", order_id="ORD-1008", amount_cents=7400)
        self.assertTrue(second["ok"])
        self.assertEqual(second["data"]["order"]["status"], "refunded")
        self.assertEqual(second["data"]["remaining_cents"], 0)

    def test_idempotent_replay_returns_same_refund(self) -> None:
        key = "test-key-1"
        first = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=500, idempotency_key=key)
        second = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=500, idempotency_key=key)
        self.assertTrue(first["ok"] and second["ok"])
        self.assertEqual(first["data"]["refund"]["id"], second["data"]["refund"]["id"])
        self.assertTrue(second["data"]["replay"])
        self.assertEqual(len(self.refunds("ORD-1001")), 1)

    def test_idempotent_replay_does_not_consume_another_fault(self) -> None:
        inject(self.conn, "refund", "ORD-1001", "card_declined", times=1)
        first = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400, idempotency_key="k2")
        second = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400, idempotency_key="k2")
        self.assertFalse(first["ok"])
        self.assertFalse(second["ok"])
        self.assertEqual(second["code"], "card_declined")
        self.assertEqual(len(self.refunds("ORD-1001")), 1)

    def test_refund_refused_when_unpaid(self) -> None:
        result = self.tk.call("refund_order", order_id="ORD-1004", amount_cents=100)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "order_not_paid")
        self.assertEqual(self.refunds("ORD-1004"), [])

    def test_refund_refused_when_already_refunded(self) -> None:
        result = self.tk.call("refund_order", order_id="ORD-1003", amount_cents=100)
        self.assertEqual(result["code"], "already_refunded")
        self.assertEqual(len(self.refunds("ORD-1003")), 1)

    def test_refund_refused_when_amount_exceeds_remaining(self) -> None:
        result = self.tk.call("refund_order", order_id="ORD-1005", amount_cents=7400)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "amount_exceeds_remaining")
        self.assertEqual(result["data"]["remaining_cents"], 5400)
        self.assertEqual(result["data"]["prior_refunded_cents"], 2000)
        self.assertEqual(len(self.refunds("ORD-1005")), 1)

    def test_card_decline_writes_failed_refund(self) -> None:
        inject(self.conn, "refund", "ORD-1009", "card_declined", times=1)
        result = self.tk.call("refund_order", order_id="ORD-1009", amount_cents=1250)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "card_declined")
        refunds = self.refunds("ORD-1009")
        self.assertEqual(len(refunds), 1)
        self.assertEqual(refunds[0]["status"], "failed")
        order = rows(self.conn, "SELECT * FROM orders WHERE id = 'ORD-1009'")[0]
        self.assertEqual(order["status"], "paid")

    def test_processor_timeout_leaves_pending_refund(self) -> None:
        inject(self.conn, "refund", "ORD-1002", "processor_timeout", times=1)
        result = self.tk.call("refund_order", order_id="ORD-1002", amount_cents=5900)
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "processor_timeout")
        self.assertEqual(self.refunds("ORD-1002")[0]["status"], "pending")

    def test_fault_is_consumed_once(self) -> None:
        inject(self.conn, "refund", "ORD-1001", "card_declined", times=1)
        first = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        second = self.tk.call("refund_order", order_id="ORD-1001", amount_cents=100)
        self.assertFalse(first["ok"])
        self.assertTrue(second["ok"])

    def test_reshipment_creates_shipment(self) -> None:
        result = self.tk.call("create_reshipment", order_id="ORD-1002")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["shipment"]["id"], "SHP-2010")
        count = scalar(self.conn, "SELECT COUNT(*) FROM shipments WHERE order_id = 'ORD-1002'")
        self.assertEqual(count, 2)

    def test_reshipment_refused_without_shipment(self) -> None:
        result = self.tk.call("create_reshipment", order_id="ORD-1012")
        self.assertEqual(result["code"], "no_shipment")

    def test_carrier_fault_creates_nothing(self) -> None:
        inject(self.conn, "reshipment", "ORD-1006", "carrier_unavailable", times=1)
        result = self.tk.call("create_reshipment", order_id="ORD-1006")
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "carrier_unavailable")
        count = scalar(self.conn, "SELECT COUNT(*) FROM shipments WHERE order_id = 'ORD-1006'")
        self.assertEqual(count, 1)

    def test_address_update_allowed_before_dispatch(self) -> None:
        result = self.tk.call("update_address", address_id="ADR-1006", city="Jounieh")
        self.assertTrue(result["ok"])
        self.assertEqual(result["data"]["address"]["city"], "Jounieh")

    def test_address_locked_after_dispatch(self) -> None:
        result = self.tk.call("update_address", address_id="ADR-1001", city="Jounieh")
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "address_locked")

    def test_address_update_without_fields(self) -> None:
        result = self.tk.call("update_address", address_id="ADR-1006")
        self.assertEqual(result["code"], "invalid_request")

    def test_address_service_fault(self) -> None:
        inject(self.conn, "address", "ADR-1006", "service_unavailable", times=1)
        result = self.tk.call("update_address", address_id="ADR-1006", city="Jounieh")
        self.assertEqual(result["code"], "service_unavailable")

    def test_send_reply_prepares_without_persisting(self) -> None:
        result = self.tk.call("send_reply", ticket_id="T-1", message="hello")
        self.assertTrue(result["ok"])
        self.assertEqual(result["code"], "reply_prepared")
        self.assertEqual(self.tk.draft, "hello")
        self.assertEqual(scalar(self.conn, "SELECT COUNT(*) FROM replies"), 0)

    def test_empty_reply_is_refused(self) -> None:
        result = self.tk.call("send_reply", ticket_id="T-1", message="   ")
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "empty_message")
        self.assertIsNone(self.tk.draft)

    def test_unknown_tool_is_audited(self) -> None:
        result = self.tk.call("delete_everything")
        self.assertEqual(result["code"], "unknown_tool")
        self.assertEqual(rows(self.conn, "SELECT * FROM audit_log")[-1]["tool"], "delete_everything")

    def test_billing_log_records_api_calls(self) -> None:
        self.tk.call("refund_order", order_id="ORD-1001", amount_cents=2400)
        log = rows(self.conn, "SELECT * FROM billing_log")
        self.assertEqual(len(log), 1)
        self.assertEqual(log[0]["http_status"], 200)


if __name__ == "__main__":
    unittest.main()
