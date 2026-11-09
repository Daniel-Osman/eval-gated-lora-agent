"""Referee: snapshots, diffs, declarative state expectations."""

from __future__ import annotations

import unittest

from shopagent.db import connect
from shopagent.referee import check_expect, check_tables, describe, diff, is_empty, snapshot
from shopagent.seed import seed


class RefereeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = seed(connect(":memory:"))

    def tearDown(self) -> None:
        self.conn.close()

    def test_snapshot_is_stable_and_excludes_timestamps(self) -> None:
        first = snapshot(self.conn)
        second = snapshot(self.conn)
        self.assertEqual(first, second)
        self.assertNotIn("created_at", first["orders"][0])
        self.assertEqual(len(first["orders"]), 12)

    def test_diff_detects_change(self) -> None:
        before = snapshot(self.conn)
        self.conn.execute("UPDATE orders SET status = 'refunded' WHERE id = 'ORD-1001'")
        after = snapshot(self.conn)
        changes = diff(before, after)
        self.assertFalse(is_empty(changes))
        self.assertEqual(changes["orders"]["changed"][0]["id"], "ORD-1001")
        self.assertTrue(any("ORD-1001" in line for line in describe(changes)))

    def test_diff_detects_insert(self) -> None:
        before = snapshot(self.conn)
        self.conn.execute(
            "INSERT INTO shipments (id, order_id, status, carrier, tracking_number, address_id, created_at, updated_at)"
            " VALUES ('SHP-9999', 'ORD-1001', 'label_created', 'DHL', 'T', 'ADR-1001', 'x', 'x')"
        )
        changes = diff(before, snapshot(self.conn))
        self.assertEqual(changes["shipments"]["added"][0]["id"], "SHP-9999")

    def test_check_expect_count(self) -> None:
        failures = check_expect(
            self.conn, [{"table": "refunds", "match": {"order_id": "ORD-1003", "status": "succeeded"}, "count": 1}]
        )
        self.assertEqual(failures, [])
        failures = check_expect(self.conn, [{"table": "refunds", "match": {"order_id": "ORD-1003"}, "count": 2}])
        self.assertIn("expected count 2, found 1", failures[0])

    def test_check_expect_fields(self) -> None:
        failures = check_expect(
            self.conn, [{"table": "orders", "match": {"id": "ORD-1001"}, "fields": {"status": "paid"}}]
        )
        self.assertEqual(failures, [])
        failures = check_expect(
            self.conn, [{"table": "orders", "match": {"id": "ORD-1001"}, "fields": {"status": "refunded"}}]
        )
        self.assertIn("status expected 'refunded'", failures[0])

    def test_check_tables_rejects_unknown_names(self) -> None:
        errors = check_tables([{"table": "nope", "match": {"id": "x"}}])
        self.assertTrue(any("unknown table" in error for error in errors))
        errors = check_tables([{"table": "orders", "match": {"nope": "x"}}])
        self.assertTrue(any("unknown columns" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
