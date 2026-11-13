"""Composed replies: the customer text is built by code from verified state.

Free-text mode lets the model write the sentence and asks the gate to catch a
false claim. That is a detection problem, and detection over open-ended language
does not generalise. Composed mode removes the channel instead: the model chooses
an action and supplies only ids and an amount in cents. No string reaches the
customer text from the model. The call runs through the same tools, then the
sentence is rendered here from the row the action actually wrote.

So the text cannot assert a state the database does not show. A failed refund
has no success sentence to render, and an address sentence can only confirm the
row the id names.

The gate still runs on the composed reply. It is a second check, not the first
line of defence.
"""

from __future__ import annotations

import re
import sqlite3

from .db import one
from .tools import ORDER_STATUS_TEXT, ToolKit

ORDER_ID = re.compile(r"^ORD-\d{4}$")
ADDRESS_ID = re.compile(r"^ADR-\d{4}$")
SHIPMENT_ID = re.compile(r"^(?:SHP-\d{4}|TRK\d{4,})$", re.I)

# The whole plan. There is no text field: an address change names an existing
# row by id, and nothing else. Anything unknown is refused.
PLAN_KEYS = {
    "refund": frozenset({"action", "order_id", "amount_cents"}),
    "reshipment": frozenset({"action", "order_id", "shipment_id"}),
    "address": frozenset({"action", "address_id"}),
    "status": frozenset({"action", "order_id"}),
}


class ComposeError(ValueError):
    """The plan is not a valid composed-mode request."""


def _money(cents: int) -> str:
    return f"${int(cents) // 100}.{int(cents) % 100:02d}"


def _check_plan(plan: dict) -> str:
    if not isinstance(plan, dict):
        raise ComposeError("plan must be an object")
    action = plan.get("action")
    allowed = PLAN_KEYS.get(action) if isinstance(action, str) else None
    if allowed is None:
        raise ComposeError(
            f"unknown action {action!r}; composed mode renders status, refund, reshipment or address"
        )
    unknown = sorted(set(plan) - allowed)
    if unknown:
        raise ComposeError(f"plan carries non-evidence keys {unknown}; composed mode takes ids and amounts, never text")
    for key, value in plan.items():
        if key == "action":
            continue
        if isinstance(value, (list, tuple, set, dict)):
            # A collection is how a plan would name two shipments. One per field.
            raise ComposeError(f"{key} must be a single value, not a collection")
        if not isinstance(value, (str, int, bool)):
            raise ComposeError(f"{key} must be an id, an integer amount or a boolean, got {type(value).__name__}")
    return action


def _order_id(plan: dict) -> str:
    value = plan.get("order_id")
    if not isinstance(value, str) or not ORDER_ID.match(value):
        raise ComposeError(f"order_id must be ORD-NNNN, got {value!r}")
    return value


def _address_id(plan: dict) -> str:
    value = plan.get("address_id")
    if not isinstance(value, str) or not ADDRESS_ID.match(value):
        raise ComposeError(f"address_id must be ADR-NNNN, got {value!r}")
    return value


def _shipment_id(plan: dict) -> str | None:
    value = plan.get("shipment_id")
    if value is None:
        return None
    if not isinstance(value, str) or not SHIPMENT_ID.match(value):
        raise ComposeError(f"shipment_id must be one SHP-NNNN or TRK number, got {value!r}")
    return value.upper()


def render_refund(conn: sqlite3.Connection, result: dict) -> str:
    """Render a refund sentence from the tool envelope, never from a plan.

    The row id comes from the result the action just returned. A plan cannot
    name a refund row, so this function cannot be handed one.
    """
    refund = result["data"].get("refund")
    if not isinstance(refund, dict) or not refund.get("id"):
        # The tool refused before writing a refund row (unpaid order, unknown
        # order, amount over the remainder). Say only that no refund happened.
        order_id = result["data"].get("order_id") or (result["data"].get("order") or {}).get("id")
        order = one(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
        if order is None:
            raise ComposeError(f"order {order_id} is not in the database")
        return f"We could not process a refund for {order['id']}."
    row = one(conn, "SELECT * FROM refunds WHERE id = ?", (refund["id"],))
    if row is None:
        raise ComposeError(f"refund {refund['id']} is not in the database")
    status = row["status"]
    shown = _money(row["amount_cents"])
    order_ref = row["order_id"]
    if status == "succeeded":
        # "We refunded ..." is the shape Phase A binds to a new succeeded row.
        # It names only the amount and order read from that row.
        return f"We refunded {shown} for {order_ref}."
    if status == "pending":
        return (
            f"We could not confirm the {shown} refund for {order_ref}; "
            "it has not been processed yet."
        )
    if status == "failed":
        code = row["failure_code"] or (result["data"].get("error") or {}).get("code") or "a failure"
        return (
            f"We could not process the {shown} refund for {order_ref}; "
            f"the payment provider reported {str(code).replace('_', ' ')}."
        )
    raise ComposeError(f"refund status {status!r} has no composed sentence")


def render_reshipment(conn: sqlite3.Connection, result: dict, wanted: str | None = None) -> str:
    """Render a reshipment sentence from the shipment row the tool wrote."""
    shipment = result["data"].get("shipment")
    if not isinstance(shipment, dict) or not shipment.get("id"):
        # Nothing was written; name the order only after reading it back.
        order_id = result["data"].get("order_id") or (result["data"].get("order") or {}).get("id")
        order = one(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
        if order is None:
            raise ComposeError(f"order {order_id} is not in the database")
        code = str(result.get("code") or "a failure").replace("_", " ")
        return f"No new shipment was created for {order['id']}; the carrier reported {code}."
    row = one(conn, "SELECT * FROM shipments WHERE id = ?", (shipment["id"],))
    if row is None:
        raise ComposeError(f"shipment {shipment['id']} is not in the database")
    # One sentence names one shipment. If the plan named one, it must be this
    # row: the sentence describes the shipment the action wrote, nothing else.
    if wanted is not None and wanted not in {row["id"].upper(), str(row["tracking_number"]).upper()}:
        raise ComposeError(
            f"shipment_id {wanted} is not the shipment this action created ({row['id']})"
        )
    order_ref = row["order_id"]
    tracking = row["tracking_number"] or "not assigned"
    status = row["status"]
    if status == "label_created":
        return (
            f"A shipping label has been created for the {order_ref} replacement; "
            f"tracking number is {tracking}."
        )
    if status == "in_transit":
        return f"The {order_ref} replacement has been shipped; tracking number is {tracking}."
    if status == "delivered":
        return f"The {order_ref} replacement has been delivered; tracking number is {tracking}."
    if status == "returned":
        return f"The {order_ref} replacement was returned; tracking number is {tracking}."
    return (
        f"The {order_ref} replacement shipment is {str(status).replace('_', ' ')}; "
        f"tracking number is {tracking}."
    )


def render_address(conn: sqlite3.Connection, address_id: str) -> str:
    """Confirm a stored address, or state the lock that forbids changing it.

    There is no text in the plan and no write here: the sentence confirms the
    stored row, so not one character of it can come from the model. When an
    order using the address has already shipped, the same check the tool makes
    turns the sentence into a refusal.
    """
    row = one(conn, "SELECT * FROM addresses WHERE id = ?", (address_id,))
    if row is None:
        raise ComposeError(f"address {address_id} is not in the database")
    locked = one(
        conn,
        "SELECT o.id FROM orders o JOIN shipments s ON s.order_id = o.id"
        " WHERE o.shipping_address_id = ? AND s.status IN ('in_transit','delivered') LIMIT 1",
        (address_id,),
    )
    if locked is not None:
        return f"I could not update the address for {locked['id']}: the order has already shipped or been delivered."
    place = ", ".join(part for part in (row["line1"], row["city"], row["postcode"]) if part)
    return f"The delivery address {row['id']} is {place}."


def render_status(conn: sqlite3.Connection, order_id: str) -> str:
    """State the order status and the latest shipment, both read from rows."""
    order = one(conn, "SELECT * FROM orders WHERE id = ?", (order_id,))
    if order is None:
        raise ComposeError(f"order {order_id} is not in the database")
    head = f"{order['id']} is {ORDER_STATUS_TEXT.get(order['status'], order['status'].replace('_', ' '))}"
    shipment = one(
        conn,
        "SELECT * FROM shipments WHERE order_id = ? ORDER BY created_at DESC, id DESC LIMIT 1",
        (order["id"],),
    )
    if shipment is None:
        return f"{head}; no shipment has been created yet."
    tracking = shipment["tracking_number"] or "not assigned"
    status = shipment["status"]
    if status == "label_created":
        tail = f"the latest shipment is {tracking}, label created"
    elif status == "in_transit":
        tail = f"the latest shipment is {tracking} and it is in transit"
    elif status == "delivered":
        tail = f"the latest shipment is {tracking} and it was delivered"
    elif status == "returned":
        tail = f"the latest shipment is {tracking} and it was returned"
    else:
        tail = f"the latest shipment is {tracking}, status {str(status).replace('_', ' ')}"
    return f"{head}; {tail}."


def _compose_refund(conn: sqlite3.Connection, ticket_id: str, plan: dict) -> str:
    order_id = _order_id(plan)
    amount = plan.get("amount_cents")
    if not isinstance(amount, int) or isinstance(amount, bool) or amount <= 0:
        raise ComposeError(f"amount_cents must be a positive integer, got {amount!r}")
    result = ToolKit(conn, ticket_id=ticket_id).call("refund_order", order_id=order_id, amount_cents=amount)
    return render_refund(conn, result)


def _compose_reshipment(conn: sqlite3.Connection, ticket_id: str, plan: dict) -> str:
    order_id = _order_id(plan)
    wanted = _shipment_id(plan)
    result = ToolKit(conn, ticket_id=ticket_id).call("create_reshipment", order_id=order_id)
    return render_reshipment(conn, result, wanted)


def _compose_address(conn: sqlite3.Connection, ticket_id: str, plan: dict) -> str:
    return render_address(conn, _address_id(plan))


def _compose_status(conn: sqlite3.Connection, ticket_id: str, plan: dict) -> str:
    return render_status(conn, _order_id(plan))


_COMPOSERS = {
    "status": _compose_status,
    "refund": _compose_refund,
    "reshipment": _compose_reshipment,
    "address": _compose_address,
}


def compose_reply(conn: sqlite3.Connection, *, ticket_id: str, plan: dict) -> str:
    """Run the planned action, then render the sentence from the row it wrote."""
    action = _check_plan(plan)
    return _COMPOSERS[action](conn, ticket_id, plan)
