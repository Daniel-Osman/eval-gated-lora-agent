"""Mock billing API with Stripe-shaped refund objects.

This is deliberately small: the point is that a failed refund looks like a real
failure (a `refund` object with `status: failed` and a `failure_code`) and that
the response can be forced deterministically. There is no network and no key.

Contract notes
--------------
* `create_refund` is idempotent per `idempotency_key`: a replay returns the
  stored object and never consumes another injected fault.
* A `processor_timeout` fault returns HTTP 504 with an error envelope *and*
  leaves a `refund` row in `pending`. That is the honest hard case: the caller
  cannot claim either success or failure.
"""

from __future__ import annotations

import json
import sqlite3

from .db import insert, now_iso, one, rows
from .faults import FAULT_MODES, consume

STRIPE_OBJECT = "refund"


def _refund_object(row: dict) -> dict:
    obj = {
        "id": row["id"],
        "object": STRIPE_OBJECT,
        "amount": int(row["amount_cents"]),
        "currency": row["currency"],
        "status": row["status"],
        "reason": row["reason"],
        "created": row["created_at"],
        "metadata": {"order_id": row["order_id"]},
    }
    if row.get("failure_code"):
        obj["failure_code"] = row["failure_code"]
        obj["failure_reason"] = row.get("failure_message") or ""
    return obj


def _log(conn: sqlite3.Connection, request: dict, response: dict, http_status: int) -> None:
    insert(
        conn,
        "billing_log",
        ts=now_iso(),
        operation="refunds.create",
        request_json=json.dumps(request, sort_keys=True),
        response_json=json.dumps(response, sort_keys=True),
        http_status=http_status,
    )
    conn.commit()


def _next_refund_id(conn: sqlite3.Connection) -> str:
    count = int(one(conn, "SELECT COUNT(*) AS n FROM refunds")["n"])
    return f"REF-{1000 + count + 1}"


def create_refund(
    conn: sqlite3.Connection,
    *,
    order_id: str,
    amount_cents: int,
    idempotency_key: str,
    reason: str = "customer_request",
) -> dict:
    """Create a refund. Returns {ok, refund, error?, replay?}."""
    request = {
        "order_id": order_id,
        "amount": amount_cents,
        "currency": "usd",
        "idempotency_key": idempotency_key,
        "reason": reason,
    }

    existing = one(conn, "SELECT * FROM refunds WHERE idempotency_key = ?", (idempotency_key,))
    if existing is not None:
        refund = _refund_object(existing)
        ok = existing["status"] == "succeeded"
        response = {"ok": ok, "refund": refund, "replay": True}
        if not ok:
            response["error"] = {
                "type": "card_error" if existing["status"] == "failed" else "api_error",
                "code": existing["failure_code"] or "refund_failed",
                "message": existing["failure_message"] or "",
            }
        _log(conn, request, response, 200 if ok else 402)
        return response

    mode = consume(conn, "refund", order_id)
    if mode is None:
        refund_id = _next_refund_id(conn)
        insert(
            conn,
            "refunds",
            id=refund_id,
            order_id=order_id,
            amount_cents=amount_cents,
            currency="usd",
            status="succeeded",
            reason=reason,
            idempotency_key=idempotency_key,
            failure_code=None,
            failure_message=None,
            created_at=now_iso(),
        )
        conn.commit()
        row = one(conn, "SELECT * FROM refunds WHERE id = ?", (refund_id,))
        response = {"ok": True, "refund": _refund_object(row)}
        _log(conn, request, response, 200)
        return response

    spec = FAULT_MODES["refund"][mode]
    refund_id = _next_refund_id(conn)
    insert(
        conn,
        "refunds",
        id=refund_id,
        order_id=order_id,
        amount_cents=amount_cents,
        currency="usd",
        status=spec["status"],
        reason=reason,
        idempotency_key=idempotency_key,
        failure_code=spec["failure_code"],
        failure_message=spec["message"],
        created_at=now_iso(),
    )
    conn.commit()
    row = one(conn, "SELECT * FROM refunds WHERE id = ?", (refund_id,))
    refund = _refund_object(row)
    if spec["status"] == "pending":
        response = {
            "ok": False,
            "error": {
                "type": "api_error",
                "code": spec["failure_code"],
                "message": spec["message"],
            },
            "refund": refund,
        }
        _log(conn, request, response, 504)
    else:
        response = {
            "ok": False,
            "error": {
                "type": "card_error",
                "code": spec["failure_code"],
                "message": spec["message"],
            },
            "refund": refund,
        }
        _log(conn, request, response, 402)
    return response


def list_billing_log(conn: sqlite3.Connection) -> list[dict]:
    return rows(conn, "SELECT * FROM billing_log ORDER BY id")
