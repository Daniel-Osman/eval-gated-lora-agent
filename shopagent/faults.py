"""Deterministic failure injection.

A fault is a row in `fault_injection`, consumed one call at a time. Nothing is
random: the same case, seeded the same way, fails the same way every run.
"""

from __future__ import annotations

import sqlite3

from .db import rows

FAULT_MODES: dict[str, dict[str, dict]] = {
    "refund": {
        "card_declined": {
            "status": "failed",
            "failure_code": "card_declined",
            "message": "Your card was declined.",
        },
        "insufficient_funds": {
            "status": "failed",
            "failure_code": "insufficient_funds",
            "message": "Your card has insufficient funds.",
        },
        "processor_timeout": {
            "status": "pending",
            "failure_code": "processor_timeout",
            "message": "The processor did not confirm the refund.",
        },
    },
    "reshipment": {
        "carrier_unavailable": {
            "failure_code": "carrier_unavailable",
            "message": "The carrier is unavailable for new labels.",
        },
        "address_invalid": {
            "failure_code": "address_invalid",
            "message": "The carrier rejected the delivery address.",
        },
    },
    "address": {
        "service_unavailable": {
            "failure_code": "service_unavailable",
            "message": "The address service did not respond.",
        },
    },
}


def inject(conn: sqlite3.Connection, operation: str, key: str, mode: str, times: int = 1) -> None:
    if operation not in FAULT_MODES:
        raise ValueError(f"unknown fault operation {operation!r}")
    if mode not in FAULT_MODES[operation]:
        raise ValueError(f"unknown mode {mode!r} for {operation!r}")
    if times < 1:
        raise ValueError("times must be >= 1")
    conn.execute(
        "INSERT INTO fault_injection (operation, key, mode, remaining) VALUES (?, ?, ?, ?)",
        (operation, key, mode, times),
    )
    conn.commit()


def consume(conn: sqlite3.Connection, operation: str, key: str) -> str | None:
    """Return the mode that fires for this call, or None. Key '*' matches any."""
    row = conn.execute(
        "SELECT id, mode, remaining FROM fault_injection"
        " WHERE operation = ? AND (key = ? OR key = '*') AND remaining > 0"
        " ORDER BY (key = ?) DESC, id LIMIT 1",
        (operation, key, key),
    ).fetchone()
    if row is None:
        return None
    remaining = int(row["remaining"]) - 1
    if remaining <= 0:
        conn.execute("DELETE FROM fault_injection WHERE id = ?", (row["id"],))
    else:
        conn.execute("UPDATE fault_injection SET remaining = ? WHERE id = ?", (remaining, row["id"]))
    conn.commit()
    return str(row["mode"])


def list_faults(conn: sqlite3.Connection) -> list[dict]:
    return rows(conn, "SELECT * FROM fault_injection ORDER BY id")


def clear(conn: sqlite3.Connection) -> None:
    conn.execute("DELETE FROM fault_injection")
    conn.commit()
