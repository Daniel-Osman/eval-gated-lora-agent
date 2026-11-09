"""SQLite schema, connections and small query helpers.

The whole project runs on the Python standard library. One database file is
the single source of truth: tools write it, the referee reads it, and the gate
believes it before it believes any sentence the model writes.
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE customers (
    id          TEXT PRIMARY KEY,
    name        TEXT NOT NULL,
    email       TEXT NOT NULL
);

CREATE TABLE addresses (
    id          TEXT PRIMARY KEY,
    customer_id TEXT NOT NULL REFERENCES customers(id),
    line1       TEXT NOT NULL,
    city        TEXT NOT NULL,
    postcode    TEXT NOT NULL,
    country     TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE orders (
    id                  TEXT PRIMARY KEY,
    customer_id         TEXT NOT NULL REFERENCES customers(id),
    status              TEXT NOT NULL CHECK (status IN
                          ('pending_payment','paid','partially_refunded','refunded','cancelled')),
    total_cents         INTEGER NOT NULL CHECK (total_cents >= 0),
    currency            TEXT NOT NULL DEFAULT 'usd',
    shipping_address_id TEXT REFERENCES addresses(id),
    created_at          TEXT NOT NULL
);

CREATE TABLE order_items (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    order_id         TEXT NOT NULL REFERENCES orders(id),
    sku              TEXT NOT NULL,
    name             TEXT NOT NULL,
    quantity         INTEGER NOT NULL CHECK (quantity > 0),
    unit_price_cents INTEGER NOT NULL CHECK (unit_price_cents >= 0)
);

CREATE TABLE refunds (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    amount_cents    INTEGER NOT NULL CHECK (amount_cents > 0),
    currency        TEXT NOT NULL DEFAULT 'usd',
    status          TEXT NOT NULL CHECK (status IN ('succeeded','failed','pending')),
    reason          TEXT NOT NULL,
    idempotency_key TEXT UNIQUE,
    failure_code    TEXT,
    failure_message TEXT,
    created_at      TEXT NOT NULL
);

CREATE TABLE shipments (
    id              TEXT PRIMARY KEY,
    order_id        TEXT NOT NULL REFERENCES orders(id),
    status          TEXT NOT NULL CHECK (status IN
                      ('label_created','in_transit','delivered','returned','cancelled')),
    carrier         TEXT NOT NULL,
    tracking_number TEXT NOT NULL,
    address_id      TEXT REFERENCES addresses(id),
    created_at      TEXT NOT NULL,
    updated_at      TEXT NOT NULL
);

CREATE TABLE replies (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ticket_id TEXT NOT NULL,
    message   TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- One row per tool invocation, including reads. Append-only by contract: no
-- UPDATE or DELETE statement in this project targets audit_log.
CREATE TABLE audit_log (
    seq         INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    actor       TEXT NOT NULL,
    tool        TEXT NOT NULL,
    args_json   TEXT NOT NULL,
    result_json TEXT NOT NULL,
    ok          INTEGER NOT NULL,
    mutating    INTEGER NOT NULL
);

-- Deterministic failure injection. `remaining` counts calls still to fail.
CREATE TABLE fault_injection (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    operation TEXT NOT NULL,
    key       TEXT NOT NULL,
    mode      TEXT NOT NULL,
    remaining INTEGER NOT NULL
);

-- Every Phase A/B decision, kept for the audit trail.
CREATE TABLE gate_runs (
    id        INTEGER PRIMARY KEY AUTOINCREMENT,
    ts        TEXT NOT NULL,
    ticket_id TEXT NOT NULL,
    attempt   INTEGER NOT NULL,
    phase     TEXT NOT NULL,
    verdict   TEXT NOT NULL CHECK (verdict IN ('APPROVED','REJECTED','ESCALATED')),
    codes     TEXT NOT NULL,
    reason    TEXT NOT NULL,
    reply     TEXT NOT NULL
);

-- Replies that failed the gate, or that Phase A could not verify, and go to a
-- human instead. `draft` is the reply the gate refused to send; it is never
-- copied into `replies`.
CREATE TABLE human_queue (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    ts          TEXT NOT NULL,
    ticket_id   TEXT NOT NULL,
    attempts    INTEGER NOT NULL,
    last_reason TEXT NOT NULL,
    draft       TEXT NOT NULL DEFAULT '',
    status      TEXT NOT NULL DEFAULT 'open'
);

-- Mock billing API request/response evidence, Stripe-shaped.
CREATE TABLE billing_log (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    ts            TEXT NOT NULL,
    operation     TEXT NOT NULL,
    request_json  TEXT NOT NULL,
    response_json TEXT NOT NULL,
    http_status   INTEGER NOT NULL
);

CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    if str(path) != ":memory:":
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(conn: sqlite3.Connection) -> sqlite3.Connection:
    conn.executescript(SCHEMA)
    conn.execute("INSERT INTO meta(key, value) VALUES ('schema_version', ?)", (str(SCHEMA_VERSION),))
    conn.commit()
    return conn


def rows(conn: sqlite3.Connection, sql: str, params: tuple | list = ()) -> list[dict]:
    return [dict(r) for r in conn.execute(sql, params).fetchall()]


def one(conn: sqlite3.Connection, sql: str, params: tuple | list = ()) -> dict | None:
    row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def scalar(conn: sqlite3.Connection, sql: str, params: tuple | list = ()):
    row = conn.execute(sql, params).fetchone()
    return None if row is None else row[0]


def insert(conn: sqlite3.Connection, table: str, **fields) -> None:
    cols = ", ".join(fields)
    marks = ", ".join("?" for _ in fields)
    conn.execute(f"INSERT INTO {table} ({cols}) VALUES ({marks})", tuple(fields.values()))


def audit(
    conn: sqlite3.Connection,
    *,
    actor: str,
    tool: str,
    args: dict,
    result: dict,
    mutating: bool,
) -> int:
    cur = conn.execute(
        "INSERT INTO audit_log (ts, actor, tool, args_json, result_json, ok, mutating)"
        " VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            now_iso(),
            actor,
            tool,
            json.dumps(args, sort_keys=True, default=str),
            json.dumps(result, sort_keys=True, default=str),
            1 if result.get("ok") else 0,
            1 if mutating else 0,
        ),
    )
    conn.commit()
    return int(cur.lastrowid)


def audit_after(conn: sqlite3.Connection, seq: int) -> list[dict]:
    return rows(conn, "SELECT * FROM audit_log WHERE seq > ? ORDER BY seq", (seq,))


def last_audit_seq(conn: sqlite3.Connection) -> int:
    value = scalar(conn, "SELECT COALESCE(MAX(seq), 0) FROM audit_log")
    return int(value or 0)
