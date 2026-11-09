"""Scripted fake models.

These are not models. They are deterministic policies used to prove the harness,
the benchmark and the gate before any real model exists:

* `scripted_policy` reads the ticket, calls the real tools, and reports what the
  tools actually returned. It should pass action/state/fact checks on the
  benchmark and be approved by the gate.
* `lying_reply` is the known-bad reply carried by each adversarial case. It is
  the gate's acceptance test, not a model output.

The policy is tuned to this shop's ticket shapes on purpose. It is a fixture,
and the README says so.
"""

from __future__ import annotations

import re

from .tools import ORDER_STATUS_TEXT, ToolKit

ORDER_ID = re.compile(r"ORD-\d{4}", re.I)
ADRESS_ID = re.compile(r"ADR-\d{4}", re.I)
CUSTOMER_ID = re.compile(r"CUS-\d{3}", re.I)
MONEY = re.compile(r"\$\s?([\d,]+(?:\.\d{1,2})?)")
TO_PHRASE = re.compile(r"\bto\s+([^.;]+)", re.I)


def fmt_cents(cents: int) -> str:
    cents = int(cents)
    return f"${cents // 100}.{cents % 100:02d}"


def status_text(status: str) -> str:
    return ORDER_STATUS_TEXT.get(status, status.replace("_", " "))


def money_from_ticket(ticket: str) -> int | None:
    match = MONEY.search(ticket)
    if not match:
        return None
    raw = match.group(1).replace(",", "")
    if "." in raw:
        whole, frac = raw.split(".", 1)
        frac = (frac + "00")[:2]
    else:
        whole, frac = raw, "00"
    return int(whole) * 100 + int(frac)


def _nearest_order(ticket: str, keyword_index: int, fallback: list[str]) -> str | None:
    match = ORDER_ID.search(ticket, keyword_index)
    if match:
        return match.group(0).upper()
    match = ORDER_ID.search(ticket)
    if match:
        return match.group(0).upper()
    return fallback[0] if fallback else None


def _load(tk: ToolKit, order_id: str, cache: dict) -> dict | None:
    if order_id in cache:
        return cache[order_id]
    result = tk.call("get_order", order_id=order_id)
    order = result["data"].get("order") if result["ok"] else None
    cache[order_id] = order
    return order


def _prior_refunded(order: dict) -> int:
    return sum(int(r["amount_cents"]) for r in order["refunds"] if r["status"] == "succeeded")


def _items_text(order: dict) -> str:
    return "; ".join(
        f"{item['quantity']} x {item['name']} ({fmt_cents(item['unit_price_cents'])})" for item in order["items"]
    ) or "none"


def _shipment_text(order: dict) -> str:
    if not order["shipments"]:
        return "no shipment"
    latest = order["shipments"][-1]
    return f"{latest['id']} {status_text(latest['status'])}, tracking {latest['tracking_number']}"


def _refund_text(order: dict) -> str:
    if not order["refunds"]:
        return "none"
    return "; ".join(
        f"{r['id']} {fmt_cents(r['amount_cents'])} {r['status']} on {r['created_at'][:10]}" for r in order["refunds"]
    )


def _refund_line(tk: ToolKit, order: dict, ticket: str) -> str:
    order_id = order["id"]
    asked = money_from_ticket(ticket) or (int(order["total_cents"]) - _prior_refunded(order))
    result = tk.call(
        "refund_order",
        order_id=order_id,
        amount_cents=asked,
        reason="customer_request",
        idempotency_key=f"auto-{tk.ticket_id}-{order_id}-{asked}",
    )

    if result["code"] == "amount_exceeds_remaining":
        data = result["data"]
        remaining = int(data["remaining_cents"])
        prior = int(data["prior_refunded_cents"])
        retry = tk.call(
            "refund_order",
            order_id=order_id,
            amount_cents=remaining,
            reason="customer_request",
            idempotency_key=f"auto-{tk.ticket_id}-{order_id}-{remaining}",
        )
        if retry["ok"]:
            return (
                f"You asked for {fmt_cents(asked)} for {order_id}, but {fmt_cents(prior)} was already refunded, "
                f"so I've refunded the remaining {fmt_cents(remaining)}."
            )
        return f"I could not refund {order_id}: {retry['code']}."

    if result["ok"]:
        refund = result["data"]["refund"]
        new_status = result["data"]["order"]["status"]
        return (
            f"I've refunded {fmt_cents(refund['amount'])} for {order_id}. "
            f"Refund {refund['id']} is {refund['status']}. The order is now {status_text(new_status)}."
        )

    code = result["code"]
    refund = result["data"].get("refund") or {}
    if code == "already_refunded":
        total = int(result["data"].get("refunded_cents", 0))
        count = len([r for r in order["refunds"] if r["status"] == "succeeded"])
        noun = "refund" if count == 1 else "refunds"
        return (
            f"{order_id} was already refunded in full ({fmt_cents(total)} across {count} {noun}). "
            "No new refund was created."
        )
    if code == "order_not_paid":
        return (
            f"I could not refund {order_id}: the order is {status_text(order['status'])} "
            "and no payment has been captured."
        )
    if code == "processor_timeout":
        return (
            f"I could not confirm the refund of {fmt_cents(asked)} for {order_id}: the processor did not confirm. "
            f"Refund {refund.get('id', '?')} is pending and is not a completed refund."
        )
    if code in ("card_declined", "insufficient_funds"):
        return (
            f"I could not process the refund of {fmt_cents(asked)} for {order_id}: "
            f"{code.replace('_', ' ')}. No new refund was created."
        )
    return f"I could not process the refund of {fmt_cents(asked)} for {order_id}: {code}."


def _reship_line(tk: ToolKit, order_id: str) -> str:
    result = tk.call("create_reshipment", order_id=order_id, reason="customer_request")
    if result["ok"]:
        shipment = result["data"]["shipment"]
        return (
            f"A reshipment for {order_id} has been created: shipment {shipment['id']}, "
            f"tracking {shipment['tracking_number']}."
        )
    if result["code"] == "carrier_unavailable":
        return (
            f"I could not create the reshipment for {order_id}: the carrier is unavailable. "
            "No new shipment was created."
        )
    if result["code"] == "no_shipment":
        return f"I could not create a reshipment for {order_id}: the order has no shipment to replace."
    return f"I could not create the reshipment for {order_id}: {result['code']}."


def _address_line(tk: ToolKit, order: dict, ticket: str) -> str:
    order_id = order["id"]
    address_id = order["shipping_address_id"]
    match = ADRESS_ID.search(ticket)
    if match:
        address_id = match.group(0).upper()

    parsed = TO_PHRASE.search(ticket)
    target = parsed.group(1).strip() if parsed else ""
    fields: dict[str, str] = {}
    if "city" in ticket.lower():
        if target:
            fields["city"] = target
    elif target:
        parts = [part.strip() for part in target.split(",") if part.strip()]
        if parts:
            fields["line1"] = parts[0]
        if len(parts) > 1:
            fields["city"] = parts[1]

    if not fields:
        return f"I could not update the address for {order_id}: no new address was given."

    result = tk.call("update_address", address_id=address_id, **fields)
    if result["ok"]:
        address = result["data"]["address"]
        return f"The delivery address for {order_id} has been updated to {address['line1']}, {address['city']}."
    if result["code"] == "address_locked":
        return (
            f"I could not update the address for {order_id}: the order has already shipped or been delivered."
        )
    return f"I could not update the address for {order_id}: {result['code']}."


def _status_line(tk: ToolKit, order_id: str, cache: dict) -> str:
    order = _load(tk, order_id, cache)
    if order is None:
        return f"I could not find order {order_id}."
    return (
        f"Order {order_id} is {status_text(order['status'])}. Items: {_items_text(order)}. "
        f"Shipment: {_shipment_text(order)}. Refunds: {_refund_text(order)}."
    )


def _intents(ticket: str) -> list[tuple[str, int]]:
    lowered = ticket.lower()
    found: list[tuple[str, int]] = []
    patterns = [
        ("refund", r"\brefund\b"),
        ("reship", r"\breship(?:ment)?\b|\breplacement\b|\bnew shipment\b"),
        ("address", r"\baddress\b"),
        ("status", r"\bstatus\b|\bwhere is\b|\btracking\b"),
    ]
    for kind, pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            found.append((kind, match.start()))
    if not found and ORDER_ID.search(ticket):
        found.append(("status", 0))
    found.sort(key=lambda item: item[1])
    return found


def scripted_policy(ticket: str, tk: ToolKit) -> str:
    """Deterministic honest agent: act with tools, then describe what happened."""
    lowered = ticket.lower()
    lines: list[str] = []
    cache: dict[str, dict | None] = {}
    ticket_orders = [match.upper() for match in ORDER_ID.findall(ticket)]

    if re.search(r"\b(?:list|show)\b.{0,25}\borders\b|\borders for\b", lowered):
        customer = CUSTOMER_ID.search(ticket)
        if customer:
            result = tk.call("list_orders", customer_id=customer.group(0).upper())
            if result["ok"]:
                found = result["data"]["orders"]
                body = "; ".join(
                    f"{order['id']} ({status_text(order['status'])}, {fmt_cents(order['total_cents'])})"
                    for order in found
                )
                lines.append(f"Orders for {customer.group(0).upper()}: {len(found)} found. {body}.")
            else:
                lines.append(f"I could not list orders for {customer.group(0).upper()}: {result['code']}.")
        reply = " ".join(lines)
        tk.call("send_reply", ticket_id=tk.ticket_id, message=reply)
        return reply

    if re.search(r"\brefunds for\b|\bshow me the refunds\b|\blist refunds\b", lowered) and ticket_orders:
        order = _load(tk, ticket_orders[0], cache)
        if order is None:
            lines.append(f"I could not find order {ticket_orders[0]}.")
        else:
            lines.append(f"Refunds for {order['id']}: {_refund_text(order)}.")
        reply = " ".join(lines)
        tk.call("send_reply", ticket_id=tk.ticket_id, message=reply)
        return reply

    intents = _intents(ticket)
    if not intents:
        lines.append("I could not tell what action this ticket needs. A human should review it.")

    for kind, position in intents:
        order_id = _nearest_order(ticket, position, ticket_orders)
        if order_id is None:
            lines.append("I could not find an order number in the ticket.")
            continue
        order = _load(tk, order_id, cache)
        if order is None:
            lines.append(f"I could not find order {order_id}.")
            continue
        if kind == "refund":
            lines.append(_refund_line(tk, order, ticket))
        elif kind == "reship":
            lines.append(_reship_line(tk, order_id))
        elif kind == "address":
            lines.append(_address_line(tk, order, ticket))
        elif kind == "status":
            lines.append(_status_line(tk, order_id, cache))

    reply = " ".join(lines)
    tk.call("send_reply", ticket_id=tk.ticket_id, message=reply)
    return reply


def teachable_policy(ticket: str, tk: ToolKit, feedback: str | None, first_reply: str) -> str:
    """A stand-in for a model that can be corrected: lie once, then answer honestly."""
    if feedback is None:
        tk.call("send_reply", ticket_id=tk.ticket_id, message=first_reply)
        return first_reply
    return scripted_policy(ticket, tk)
