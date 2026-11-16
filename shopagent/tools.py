"""The six tools the agent may call.

Every call writes an audit row. State-changing tools write real rows in the
same transaction; a failure writes nothing but the failed attempt (and, for
refunds, the API's failed/pending refund object, because that is what Stripe
would show).

The tool layer enforces policy. The model cannot refund more than remains, cannot
reship an order that was never shipped, and cannot update a delivered address.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from . import billing
from .db import audit, insert, now_iso, one, rows, scalar
from .faults import FAULT_MODES, consume

MUTATING_STATE = frozenset({"refund_order", "create_reshipment", "update_address"})
# send_reply is deliberately not a mutation: it validates and prepares a draft.
# Only the gate may persist a customer-visible reply (gate.commit_reply).
MUTATING = MUTATING_STATE

ORDER_STATUS_TEXT = {
    "pending_payment": "pending payment",
    "paid": "paid",
    "partially_refunded": "partially refunded",
    "refunded": "refunded",
    "cancelled": "cancelled",
}

TOOL_SPECS: list[dict[str, Any]] = [
    {
        "name": "get_order",
        "description": "Read one order with its items, shipments and refunds.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string", "pattern": "^ORD-[0-9]{4}$"}},
            "required": ["order_id"],
        },
        "mutating": False,
    },
    {
        "name": "list_orders",
        "description": "List orders for a customer, optionally filtered by status.",
        "parameters": {
            "type": "object",
            "properties": {
                "customer_id": {"type": "string", "pattern": "^CUS-[0-9]{3}$"},
                "status": {"type": "string"},
                "limit": {"type": "integer", "minimum": 1, "maximum": 50},
            },
            "required": ["customer_id"],
        },
        "mutating": False,
    },
    {
        "name": "refund_order",
        "description": "Refund an amount on a paid order through the billing API.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "amount_cents": {"type": "integer", "minimum": 1},
                "reason": {"type": "string"},
                "idempotency_key": {"type": "string"},
            },
            "required": ["order_id", "amount_cents"],
        },
        "mutating": True,
    },
    {
        "name": "create_reshipment",
        "description": "Create a new shipment for an order that already has one.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "reason": {"type": "string"},
                "address_id": {"type": "string"},
            },
            "required": ["order_id"],
        },
        "mutating": True,
    },
    {
        "name": "update_address",
        "description": "Update a delivery address while the parcel has not shipped.",
        "parameters": {
            "type": "object",
            "properties": {
                "address_id": {"type": "string"},
                "line1": {"type": "string"},
                "city": {"type": "string"},
                "postcode": {"type": "string"},
            },
            "required": ["address_id"],
        },
        "mutating": True,
    },
    {
        "name": "send_reply",
        "description": "Send the final customer reply for this ticket. This is the last call.",
        "parameters": {
            "type": "object",
            "properties": {
                "ticket_id": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["ticket_id", "message"],
        },
        "mutating": True,
    },
]


def _envelope(tool: str, ok: bool, code: str, data: dict | None = None) -> dict:
    return {"ok": bool(ok), "tool": tool, "code": code, "data": data or {}}


def _order_bundle(conn: sqlite3.Connection, order_id: str) -> dict | None:
    order = one(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
    if order is None:
        return None
    order["customer"] = one(conn, "SELECT id, name, email FROM customers WHERE id = ?", (order["customer_id"],))
    order["address"] = one(conn, "SELECT * FROM addresses WHERE id = ?", (order["shipping_address_id"],))
    order["items"] = rows(conn, "SELECT * FROM order_items WHERE order_id = ? ORDER BY id", (order_id,))
    order["shipments"] = rows(
        conn, "SELECT * FROM shipments WHERE order_id = ? ORDER BY created_at, id", (order_id,)
    )
    order["refunds"] = rows(conn, "SELECT * FROM refunds WHERE order_id = ? ORDER BY created_at, id", (order_id,))
    return order


def refunded_cents(conn: sqlite3.Connection, order_id: str) -> int:
    value = scalar(
        conn,
        "SELECT COALESCE(SUM(amount_cents), 0) FROM refunds WHERE order_id = ? AND status = 'succeeded'",
        (order_id,),
    )
    return int(value or 0)


class ToolKit:
    """Execute tool calls for one ticket, auditing every one."""

    def __init__(self, conn: sqlite3.Connection, ticket_id: str, actor: str = "agent"):
        self.conn = conn
        self.ticket_id = ticket_id
        self.actor = actor
        # The last reply the agent prepared. It is a draft until the gate
        # approves it; nothing here is customer-visible.
        self.draft: str | None = None

    # -- dispatch -------------------------------------------------------
    def call(self, name: str, **args) -> dict:
        handler = getattr(self, f"_tool_{name}", None)
        if handler is None:
            result = _envelope(name, False, "unknown_tool", {"message": f"no such tool: {name}"})
        else:
            try:
                result = handler(**args)
            except TypeError:
                # `str(exc)` here would read "ToolKit._tool_get_order() got an
                # unexpected keyword argument 'unexpected'" into a page a stranger
                # sees. The envelope stays a result; the detail stays out of it.
                result = _envelope(
                    name, False, "invalid_arguments", {"message": "the tool was called with arguments it does not accept"}
                )
        audit(
            self.conn,
            actor=self.actor,
            tool=name,
            args=args,
            result=result,
            mutating=name in MUTATING,
        )
        return result

    # -- reads ----------------------------------------------------------
    def _tool_get_order(self, order_id: str) -> dict:
        bundle = _order_bundle(self.conn, order_id)
        if bundle is None:
            return _envelope("get_order", False, "order_not_found", {"order_id": order_id})
        return _envelope("get_order", True, "ok", {"order": bundle})

    def _tool_list_orders(self, customer_id: str, status: str | None = None, limit: int = 10) -> dict:
        if one(self.conn, "SELECT id FROM customers WHERE id = ?", (customer_id,)) is None:
            return _envelope("list_orders", False, "customer_not_found", {"customer_id": customer_id})
        limit = max(1, min(int(limit), 50))
        sql = "SELECT * FROM orders WHERE customer_id = ?"
        params: list[Any] = [customer_id]
        if status:
            sql += " AND status = ?"
            params.append(status)
        sql += " ORDER BY created_at LIMIT ?"
        params.append(limit)
        found = rows(self.conn, sql, params)
        return _envelope("list_orders", True, "ok", {"orders": found, "count": len(found)})

    # -- mutations ------------------------------------------------------
    def _tool_refund_order(
        self,
        order_id: str,
        amount_cents: int,
        reason: str = "customer_request",
        idempotency_key: str | None = None,
    ) -> dict:
        order = one(self.conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
        if order is None:
            return _envelope("refund_order", False, "order_not_found", {"order_id": order_id})
        if not isinstance(amount_cents, int) or isinstance(amount_cents, bool):
            return _envelope("refund_order", False, "invalid_amount", {"message": "amount_cents must be an integer"})
        if amount_cents <= 0:
            return _envelope("refund_order", False, "invalid_amount", {"message": "amount_cents must be positive"})
        if order["status"] == "refunded":
            prior = refunded_cents(self.conn, order_id)
            return _envelope(
                "refund_order",
                False,
                "already_refunded",
                {"order_id": order_id, "refunded_cents": prior, "total_cents": order["total_cents"]},
            )
        if order["status"] not in ("paid", "partially_refunded"):
            return _envelope(
                "refund_order",
                False,
                "order_not_paid",
                {"order_id": order_id, "status": order["status"], "total_cents": order["total_cents"]},
            )

        prior = refunded_cents(self.conn, order_id)
        remaining = int(order["total_cents"]) - prior
        if amount_cents > remaining:
            return _envelope(
                "refund_order",
                False,
                "amount_exceeds_remaining",
                {
                    "order_id": order_id,
                    "requested_cents": amount_cents,
                    "remaining_cents": remaining,
                    "prior_refunded_cents": prior,
                    "total_cents": order["total_cents"],
                },
            )

        key = idempotency_key or f"{self.ticket_id}:{order_id}:{amount_cents}"
        api = billing.create_refund(
            self.conn,
            order_id=order_id,
            amount_cents=amount_cents,
            idempotency_key=key,
            reason=reason,
        )
        refund = api["refund"]
        if api["ok"]:
            total_refunded = refunded_cents(self.conn, order_id)
            new_status = "refunded" if total_refunded >= int(order["total_cents"]) else "partially_refunded"
            self.conn.execute("UPDATE orders SET status = ? WHERE id = ?", (new_status, order_id))
            self.conn.commit()
            return _envelope(
                "refund_order",
                True,
                "refund_succeeded",
                {
                    "refund": refund,
                    "order": {"id": order_id, "status": new_status, "total_cents": order["total_cents"]},
                    "remaining_cents": int(order["total_cents"]) - total_refunded,
                    "replay": bool(api.get("replay")),
                },
            )

        code = api["error"]["code"]
        return _envelope(
            "refund_order",
            False,
            code,
            {
                "refund": refund,
                "order": {"id": order_id, "status": order["status"], "total_cents": order["total_cents"]},
                "error": api["error"],
            },
        )

    def _tool_create_reshipment(
        self,
        order_id: str,
        reason: str = "customer_request",
        address_id: str | None = None,
    ) -> dict:
        order = one(self.conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
        if order is None:
            return _envelope("create_reshipment", False, "order_not_found", {"order_id": order_id})
        shipment = one(
            self.conn,
            "SELECT * FROM shipments WHERE order_id = ? ORDER BY created_at DESC LIMIT 1",
            (order_id,),
        )
        if shipment is None:
            return _envelope(
                "create_reshipment", False, "no_shipment", {"order_id": order_id, "message": "order has no shipment"}
            )

        mode = consume(self.conn, "reshipment", order_id)
        if mode is not None:
            spec = FAULT_MODES["reshipment"][mode]
            return _envelope(
                "create_reshipment",
                False,
                spec["failure_code"],
                {"order_id": order_id, "error": {"code": spec["failure_code"], "message": spec["message"]}},
            )

        # Tool-created shipments own the 2010+ series. Count only that series:
        # seeded shipments live in the 1000 series and must not shift the first
        # id a frozen case expects (SHP-2010). Counting every row made the
        # allocator depend on dataset size, so extending the seed broke old cases.
        created = int(scalar(self.conn, "SELECT COUNT(*) FROM shipments WHERE id >= 'SHP-2010'") or 0)
        shipment_id = f"SHP-{2010 + created}"
        tracking = f"TRK{2010 + created}"
        target_address = address_id or order["shipping_address_id"]
        insert(
            self.conn,
            "shipments",
            id=shipment_id,
            order_id=order_id,
            status="label_created",
            carrier=shipment["carrier"],
            tracking_number=tracking,
            address_id=target_address,
            created_at=now_iso(),
            updated_at=now_iso(),
        )
        self.conn.commit()
        created = one(self.conn, "SELECT * FROM shipments WHERE id = ?", (shipment_id,))
        return _envelope("create_reshipment", True, "reshipment_created", {"shipment": created, "order_id": order_id})

    def _tool_update_address(
        self,
        address_id: str,
        line1: str | None = None,
        city: str | None = None,
        postcode: str | None = None,
    ) -> dict:
        address = one(self.conn, "SELECT * FROM addresses WHERE id = ?", (address_id,))
        if address is None:
            return _envelope("update_address", False, "address_not_found", {"address_id": address_id})
        fields = {k: v for k, v in {"line1": line1, "city": city, "postcode": postcode}.items() if v is not None}
        if not fields:
            return _envelope("update_address", False, "invalid_request", {"message": "no fields to update"})

        locked = one(
            self.conn,
            "SELECT o.id FROM orders o JOIN shipments s ON s.order_id = o.id"
            " WHERE o.shipping_address_id = ? AND s.status IN ('in_transit','delivered') LIMIT 1",
            (address_id,),
        )
        if locked is not None:
            return _envelope(
                "update_address",
                False,
                "address_locked",
                {"address_id": address_id, "order_id": locked["id"], "message": "parcel already shipped"},
            )

        mode = consume(self.conn, "address", address_id)
        if mode is not None:
            spec = FAULT_MODES["address"][mode]
            return _envelope(
                "update_address",
                False,
                spec["failure_code"],
                {"address_id": address_id, "error": {"code": spec["failure_code"], "message": spec["message"]}},
            )

        assignments = ", ".join(f"{k} = ?" for k in fields)
        self.conn.execute(
            f"UPDATE addresses SET {assignments}, updated_at = ? WHERE id = ?",
            (*fields.values(), now_iso(), address_id),
        )
        self.conn.commit()
        updated = one(self.conn, "SELECT * FROM addresses WHERE id = ?", (address_id,))
        return _envelope("update_address", True, "address_updated", {"address": updated})

    def _tool_send_reply(self, ticket_id: str, message: str) -> dict:
        """Validate and prepare the reply. The gate owns persistence.

        Nothing is written to `replies` here: a reply is a draft until the gate
        approves it, so a rejected attempt cannot leave a false message behind.
        """
        if not isinstance(message, str) or not message.strip():
            return _envelope("send_reply", False, "empty_message", {"ticket_id": ticket_id})
        self.draft = message
        return _envelope(
            "send_reply",
            True,
            "reply_prepared",
            {"ticket_id": ticket_id, "message": message},
        )
