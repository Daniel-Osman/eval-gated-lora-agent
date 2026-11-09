"""Fixed demo dataset.

Every benchmark case starts from this exact state. Timestamps are literals so a
state snapshot is reproducible; the referee ignores timestamps when it diffs.
"""

from __future__ import annotations

import sqlite3

from .db import init_db, insert

CUSTOMERS = [
    ("CUS-001", "Amina Haddad", "amina@example.test"),
    ("CUS-002", "Omar Nassar", "omar@example.test"),
    ("CUS-003", "Lina Farah", "lina@example.test"),
    ("CUS-004", "Karim Aziz", "karim@example.test"),
]

# One address per order, so an address update has one unambiguous owner.
ADDRESSES = [
    ("ADR-1001", "CUS-001", "14 Cedar Street", "Beirut", "1103", "LB", "2026-10-01T09:00:00Z"),
    ("ADR-1002", "CUS-001", "9 Orchard Lane", "Beirut", "1105", "LB", "2026-10-04T09:00:00Z"),
    ("ADR-1003", "CUS-002", "8 Palm Road", "Tripoli", "1300", "LB", "2026-09-18T09:00:00Z"),
    ("ADR-1004", "CUS-002", "31 Station Street", "Tripoli", "1302", "LB", "2026-10-30T09:00:00Z"),
    ("ADR-1005", "CUS-003", "22 Hill Lane", "Saida", "1600", "LB", "2026-09-26T09:00:00Z"),
    ("ADR-1006", "CUS-003", "5 Market Square", "Saida", "1601", "LB", "2026-11-03T09:00:00Z"),
    ("ADR-1007", "CUS-004", "12 Bay View", "Byblos", "1400", "LB", "2026-10-10T09:00:00Z"),
    ("ADR-1008", "CUS-004", "77 River Road", "Byblos", "1402", "LB", "2026-10-31T09:00:00Z"),
    ("ADR-1009", "CUS-001", "3 Garden Row", "Beirut", "1104", "LB", "2026-10-18T09:00:00Z"),
    ("ADR-1010", "CUS-002", "40 Olive Street", "Tripoli", "1301", "LB", "2026-10-23T09:00:00Z"),
    ("ADR-1011", "CUS-003", "18 Harbour Walk", "Saida", "1602", "LB", "2026-09-08T09:00:00Z"),
    ("ADR-1012", "CUS-004", "6 Cedar Court", "Beirut", "1109", "LB", "2026-11-05T09:00:00Z"),
]

ORDERS = [
    # id, customer, status, total, address, created
    ("ORD-1001", "CUS-001", "paid", 2400, "ADR-1001", "2026-10-02T10:00:00Z"),
    ("ORD-1002", "CUS-001", "paid", 5900, "ADR-1002", "2026-10-05T10:00:00Z"),
    ("ORD-1003", "CUS-002", "refunded", 1850, "ADR-1003", "2026-09-20T10:00:00Z"),
    ("ORD-1004", "CUS-002", "pending_payment", 3200, "ADR-1004", "2026-11-01T10:00:00Z"),
    ("ORD-1005", "CUS-003", "partially_refunded", 7400, "ADR-1005", "2026-09-28T10:00:00Z"),
    ("ORD-1006", "CUS-003", "paid", 1299, "ADR-1006", "2026-11-05T10:00:00Z"),
    ("ORD-1007", "CUS-004", "paid", 4600, "ADR-1007", "2026-10-12T10:00:00Z"),
    ("ORD-1008", "CUS-004", "paid", 9900, "ADR-1008", "2026-11-02T10:00:00Z"),
    ("ORD-1009", "CUS-001", "paid", 1250, "ADR-1009", "2026-10-20T10:00:00Z"),
    ("ORD-1010", "CUS-002", "paid", 8800, "ADR-1010", "2026-10-25T10:00:00Z"),
    ("ORD-1011", "CUS-003", "refunded", 3300, "ADR-1011", "2026-09-10T10:00:00Z"),
    ("ORD-1012", "CUS-004", "pending_payment", 2100, "ADR-1012", "2026-11-06T10:00:00Z"),
]

ITEMS = [
    ("ORD-1001", "THROW-COTTON", "Cotton throw", 1, 2400),
    ("ORD-1002", "LAMP-DESK", "Desk lamp", 1, 5900),
    ("ORD-1003", "MUG-SET", "Ceramic mug set", 1, 1850),
    ("ORD-1004", "SHIRT-LINEN", "Linen shirt", 1, 3200),
    ("ORD-1005", "BLANKET-WOOL", "Wool blanket", 2, 3700),
    ("ORD-1006", "TEA-SAMPLER", "Tea sampler", 1, 1299),
    ("ORD-1007", "NOTEBOOK-LEATHER", "Leather notebook", 1, 4600),
    ("ORD-1008", "ESPRESSO-MACHINE", "Espresso machine", 1, 9900),
    ("ORD-1009", "CANDLE-BEESWAX", "Beeswax candles", 1, 1250),
    ("ORD-1010", "TABLE-WALNUT", "Walnut side table", 1, 8800),
    ("ORD-1011", "CLOCK-WALL", "Wall clock", 1, 3300),
    ("ORD-1012", "CARAFE-GLASS", "Glass carafe", 1, 2100),
]

REFUNDS = [
    # id, order, amount, status, reason, key, failure_code, failure_message, created
    ("REF-1001", "ORD-1003", 1850, "succeeded", "damaged_item", "seed-ref-1003", None, None, "2026-09-22T11:00:00Z"),
    ("REF-1002", "ORD-1005", 2000, "succeeded", "damaged_item", "seed-ref-1005", None, None, "2026-10-01T11:00:00Z"),
    ("REF-1003", "ORD-1011", 3300, "succeeded", "returned_item", "seed-ref-1011", None, None, "2026-09-14T11:00:00Z"),
]

SHIPMENTS = [
    # id, order, status, carrier, tracking, address, created, updated
    ("SHP-1001", "ORD-1001", "in_transit", "DHL", "TRK1001", "ADR-1001", "2026-10-03T08:00:00Z", "2026-10-04T08:00:00Z"),
    ("SHP-1002", "ORD-1002", "delivered", "DHL", "TRK1002", "ADR-1002", "2026-10-06T08:00:00Z", "2026-10-09T08:00:00Z"),
    ("SHP-1003", "ORD-1003", "delivered", "Aramex", "TRK1003", "ADR-1003", "2026-09-21T08:00:00Z", "2026-09-25T08:00:00Z"),
    ("SHP-1005", "ORD-1005", "delivered", "Aramex", "TRK1005", "ADR-1005", "2026-09-29T08:00:00Z", "2026-10-03T08:00:00Z"),
    ("SHP-1006", "ORD-1006", "label_created", "DHL", "TRK1006", "ADR-1006", "2026-11-06T08:00:00Z", "2026-11-06T08:00:00Z"),
    ("SHP-1007", "ORD-1007", "delivered", "DHL", "TRK1007", "ADR-1007", "2026-10-13T08:00:00Z", "2026-10-16T08:00:00Z"),
    ("SHP-1008", "ORD-1008", "in_transit", "DHL", "TRK1008", "ADR-1008", "2026-11-03T08:00:00Z", "2026-11-04T08:00:00Z"),
    ("SHP-1009", "ORD-1009", "delivered", "Aramex", "TRK1009", "ADR-1009", "2026-10-21T08:00:00Z", "2026-10-24T08:00:00Z"),
    ("SHP-1010", "ORD-1010", "delivered", "Aramex", "TRK1010", "ADR-1010", "2026-10-26T08:00:00Z", "2026-10-29T08:00:00Z"),
    ("SHP-1011", "ORD-1011", "returned", "Aramex", "TRK1011", "ADR-1011", "2026-09-11T08:00:00Z", "2026-09-18T08:00:00Z"),
]


def seed(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the schema and load the fixed dataset. Safe on an empty DB."""
    init_db(conn)
    for row in CUSTOMERS:
        insert(conn, "customers", id=row[0], name=row[1], email=row[2])
    for row in ADDRESSES:
        insert(
            conn,
            "addresses",
            id=row[0],
            customer_id=row[1],
            line1=row[2],
            city=row[3],
            postcode=row[4],
            country=row[5],
            updated_at=row[6],
        )
    for row in ORDERS:
        insert(
            conn,
            "orders",
            id=row[0],
            customer_id=row[1],
            status=row[2],
            total_cents=row[3],
            currency="usd",
            shipping_address_id=row[4],
            created_at=row[5],
        )
    for row in ITEMS:
        insert(
            conn,
            "order_items",
            order_id=row[0],
            sku=row[1],
            name=row[2],
            quantity=row[3],
            unit_price_cents=row[4],
        )
    for row in REFUNDS:
        insert(
            conn,
            "refunds",
            id=row[0],
            order_id=row[1],
            amount_cents=row[2],
            currency="usd",
            status=row[3],
            reason=row[4],
            idempotency_key=row[5],
            failure_code=row[6],
            failure_message=row[7],
            created_at=row[8],
        )
    for row in SHIPMENTS:
        insert(
            conn,
            "shipments",
            id=row[0],
            order_id=row[1],
            status=row[2],
            carrier=row[3],
            tracking_number=row[4],
            address_id=row[5],
            created_at=row[6],
            updated_at=row[7],
        )
    conn.commit()
    return conn


def fresh() -> sqlite3.Connection:
    from .db import connect

    return seed(connect(":memory:"))
