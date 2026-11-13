"""Fixed demo dataset.

Every benchmark case starts from this exact state. Timestamps are literals so a
state snapshot is reproducible; the referee ignores timestamps when it diffs.

Two blocks exist on purpose. `ORD-1001..1012` and `CUS-001..004` are the rows
the frozen 40 cases assert, byte for byte. The `_EXT` block is stage 2 data:
new customers own every new order, so a list-orders count for an old customer
cannot move when the dataset grows. New rows keep their own id ranges
(`ADR-2001+`, `REF-2001+`) and seeded shipments stay in the 1000 series, which
is the range the created-shipment allocator deliberately ignores.
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

# -- stage 2 extension -----------------------------------------------------
# `ORDER_DATA_EXT` is the single source for the new addresses, orders, items
# and (where the order has one) shipments below, so an address cannot point at
# the wrong order. One address per order, one item line per order.

CUSTOMERS_EXT = [
    ("CUS-005", "Nour Khalil", "nour@example.test"),
    ("CUS-006", "Rami Salameh", "rami@example.test"),
    ("CUS-007", "Maya Chidiac", "maya@example.test"),
    ("CUS-008", "Tarek Hobeika", "tarek@example.test"),
    ("CUS-009", "Dana Aoun", "dana@example.test"),
    ("CUS-010", "Fadi Rizk", "fadi@example.test"),
    ("CUS-011", "Hala Mansour", "hala@example.test"),
    ("CUS-012", "Ziad Barakat", "ziad@example.test"),
    ("CUS-013", "Lara Khoury", "lara@example.test"),
    ("CUS-014", "Samir Gemayel", "samir@example.test"),
    ("CUS-015", "Rita Sfeir", "rita@example.test"),
    ("CUS-016", "Jad Frem", "jad@example.test"),
]

# order, customer, status, total_cents, line1, city, postcode, created, sku, item
ORDER_DATA_EXT = [
    # delivered
    ("ORD-1013", "CUS-005", "delivered", 3499, "7 Almond Street", "Beirut", "1107", "2026-07-02T10:00:00Z", "THROW-LINEN", "Linen throw"),
    ("ORD-1014", "CUS-005", "delivered", 12875, "19 Sunflower Row", "Beirut", "1108", "2026-07-05T10:00:00Z", "LAMP-FLOOR", "Floor lamp"),
    ("ORD-1015", "CUS-006", "delivered", 6420, "3 Pebble Lane", "Tripoli", "1305", "2026-07-09T10:00:00Z", "KETTLE-STEEL", "Steel kettle"),
    ("ORD-1016", "CUS-006", "delivered", 990, "44 Jasmine Street", "Tripoli", "1306", "2026-07-12T10:00:00Z", "SOAP-OLIVE", "Olive soap"),
    ("ORD-1017", "CUS-007", "delivered", 21500, "2 Harbour View", "Saida", "1605", "2026-07-16T10:00:00Z", "RUG-WOOL", "Wool rug"),
    ("ORD-1018", "CUS-007", "delivered", 4999, "88 Cedar Ridge", "Saida", "1606", "2026-07-20T10:00:00Z", "VASE-CLAY", "Clay vase"),
    ("ORD-1019", "CUS-008", "delivered", 7350, "15 Orchard Walk", "Byblos", "1405", "2026-07-23T10:00:00Z", "TRAY-OAK", "Oak tray"),
    ("ORD-1020", "CUS-008", "delivered", 2899, "6 Marina Street", "Byblos", "1406", "2026-07-27T10:00:00Z", "CUP-GLASS", "Glass cup"),
    # in transit
    ("ORD-1021", "CUS-009", "in_transit", 5600, "10 Pine Street", "Jounieh", "1201", "2026-08-01T10:00:00Z", "TOWEL-COTTON", "Cotton towel"),
    ("ORD-1022", "CUS-009", "in_transit", 17400, "27 Willow Lane", "Jounieh", "1202", "2026-08-04T10:00:00Z", "MIRROR-BRASS", "Brass mirror"),
    ("ORD-1023", "CUS-010", "in_transit", 2599, "5 Cedar Close", "Zahle", "1801", "2026-08-08T10:00:00Z", "BOWL-STONE", "Stone bowl"),
    ("ORD-1024", "CUS-010", "in_transit", 8800, "31 Vine Street", "Zahle", "1802", "2026-08-11T10:00:00Z", "CHAIR-OAK", "Oak chair"),
    ("ORD-1025", "CUS-011", "in_transit", 14250, "9 Bay Street", "Tyre", "1701", "2026-08-15T10:00:00Z", "SHELF-PINE", "Pine shelf"),
    ("ORD-1026", "CUS-011", "in_transit", 4599, "14 Palm Court", "Tyre", "1702", "2026-08-18T10:00:00Z", "MAT-DOOR", "Door mat"),
    ("ORD-1027", "CUS-012", "in_transit", 31000, "21 Temple Road", "Baalbek", "1901", "2026-08-22T10:00:00Z", "DESK-WALNUT", "Walnut desk"),
    ("ORD-1028", "CUS-012", "in_transit", 780, "40 Alley Street", "Baalbek", "1902", "2026-08-25T10:00:00Z", "SPONGE-SEA", "Sea sponge"),
    # label created
    ("ORD-1029", "CUS-013", "label_created", 1899, "8 Rose Street", "Beirut", "1110", "2026-09-01T10:00:00Z", "MUG-CLAY", "Clay mug"),
    ("ORD-1030", "CUS-013", "label_created", 6740, "12 Fig Lane", "Beirut", "1111", "2026-09-04T10:00:00Z", "JUG-CERAMIC", "Ceramic jug"),
    ("ORD-1031", "CUS-014", "label_created", 11200, "4 Olive Grove", "Tripoli", "1307", "2026-09-08T10:00:00Z", "STOOL-BEECH", "Beech stool"),
    ("ORD-1032", "CUS-014", "label_created", 3350, "26 Sage Street", "Tripoli", "1308", "2026-09-11T10:00:00Z", "LAMP-TABLE", "Table lamp"),
    ("ORD-1033", "CUS-015", "label_created", 4999, "7 Quarry Road", "Saida", "1607", "2026-09-15T10:00:00Z", "BASKET-REED", "Reed basket"),
    ("ORD-1034", "CUS-015", "label_created", 15800, "33 Anchor Street", "Saida", "1608", "2026-09-18T10:00:00Z", "MIRROR-OAK", "Oak mirror"),
    ("ORD-1035", "CUS-016", "label_created", 2275, "2 Lime Street", "Byblos", "1407", "2026-09-22T10:00:00Z", "CANDLE-SOY", "Soy candle"),
    ("ORD-1036", "CUS-016", "label_created", 9600, "18 Mill Lane", "Byblos", "1408", "2026-09-25T10:00:00Z", "SHEET-LINEN", "Linen sheet"),
    # returned
    ("ORD-1037", "CUS-005", "returned", 4200, "5 Stone Street", "Jounieh", "1203", "2026-09-29T10:00:00Z", "POT-COPPER", "Copper pot"),
    ("ORD-1038", "CUS-006", "returned", 1299, "51 Field Road", "Zahle", "1803", "2026-10-02T10:00:00Z", "TEA-GREEN", "Green tea"),
    ("ORD-1039", "CUS-007", "returned", 7800, "11 Cliff Street", "Tyre", "1703", "2026-10-05T10:00:00Z", "SHADE-LAMP", "Lamp shade"),
    ("ORD-1040", "CUS-008", "returned", 3150, "23 Grove Lane", "Baalbek", "1903", "2026-10-08T10:00:00Z", "BOOK-JOURNAL", "Journal"),
    ("ORD-1041", "CUS-009", "returned", 21500, "1 Port Street", "Beirut", "1112", "2026-10-11T10:00:00Z", "SOFA-LINEN", "Linen sofa"),
    ("ORD-1042", "CUS-010", "returned", 5899, "30 Mill Street", "Tripoli", "1309", "2026-10-14T10:00:00Z", "PAN-CAST", "Cast pan"),
    ("ORD-1043", "CUS-011", "returned", 1999, "16 Fern Road", "Saida", "1609", "2026-10-17T10:00:00Z", "CLOCK-DESK", "Desk clock"),
    ("ORD-1044", "CUS-012", "returned", 8780, "45 Ridge Street", "Byblos", "1409", "2026-10-20T10:00:00Z", "CABINET-PINE", "Pine cabinet"),
    # refunded, one succeeded refund each
    ("ORD-1045", "CUS-013", "refunded", 5600, "3 Wood Street", "Jounieh", "1204", "2026-10-06T10:00:00Z", "LAMP-ARC", "Arc lamp"),
    ("ORD-1046", "CUS-014", "refunded", 2340, "9 Moss Lane", "Zahle", "1804", "2026-10-09T10:00:00Z", "MUG-STONE", "Stone mug"),
    ("ORD-1047", "CUS-015", "refunded", 9900, "20 Beacon Road", "Tyre", "1704", "2026-10-12T10:00:00Z", "RUG-COTTON", "Cotton rug"),
    ("ORD-1048", "CUS-016", "refunded", 1290, "6 Spring Street", "Baalbek", "1904", "2026-10-15T10:00:00Z", "SOAP-ROSE", "Rose soap"),
    ("ORD-1049", "CUS-005", "refunded", 4750, "13 Elm Street", "Beirut", "1113", "2026-10-18T10:00:00Z", "TRAY-BRASS", "Brass tray"),
    # partially refunded, one succeeded partial refund each
    ("ORD-1050", "CUS-006", "partially_refunded", 8000, "17 Ash Lane", "Tripoli", "1310", "2026-10-21T10:00:00Z", "STOOL-ASH", "Ash stool"),
    ("ORD-1051", "CUS-007", "partially_refunded", 12300, "8 Rill Road", "Saida", "1610", "2026-10-24T10:00:00Z", "DESK-OAK", "Oak desk"),
    ("ORD-1052", "CUS-008", "partially_refunded", 6400, "29 Beach Street", "Byblos", "1410", "2026-10-27T10:00:00Z", "TOWEL-LINEN", "Linen towel"),
    ("ORD-1053", "CUS-009", "partially_refunded", 990, "37 Fir Street", "Jounieh", "1205", "2026-10-30T10:00:00Z", "CUP-STONE", "Stone cup"),
    ("ORD-1054", "CUS-010", "partially_refunded", 15800, "24 Crest Lane", "Zahle", "1805", "2026-11-02T10:00:00Z", "BED-FRAME", "Bed frame"),
    # pending payment, no shipment
    ("ORD-1055", "CUS-011", "pending_payment", 3200, "43 Cedar Bend", "Tyre", "1705", "2026-11-03T10:00:00Z", "SHIRT-COTTON", "Cotton shirt"),
    ("ORD-1056", "CUS-012", "pending_payment", 4500, "7 Bay Lane", "Baalbek", "1905", "2026-11-04T10:00:00Z", "SOCKS-WOOL", "Wool socks"),
    ("ORD-1057", "CUS-013", "pending_payment", 1299, "34 Palm Street", "Beirut", "1114", "2026-11-05T10:00:00Z", "TEA-MINT", "Mint tea"),
    ("ORD-1058", "CUS-014", "pending_payment", 7600, "12 Stone Lane", "Tripoli", "1311", "2026-11-06T10:00:00Z", "PAN-GRIDDLE", "Griddle pan"),
    # cancelled, no shipment
    ("ORD-1059", "CUS-015", "cancelled", 2100, "5 Lark Street", "Saida", "1611", "2026-11-06T10:00:00Z", "BOWL-GLASS", "Glass bowl"),
    ("ORD-1060", "CUS-016", "cancelled", 1850, "39 Oak Lane", "Byblos", "1411", "2026-11-07T10:00:00Z", "MUG-TIN", "Tin mug"),
]

ADDRESSES_EXT = [
    (f"ADR-{2001 + index}", row[1], row[4], row[5], row[6], "LB", row[7])
    for index, row in enumerate(ORDER_DATA_EXT)
]

# `ORDER_DATA_EXT` names the customer-visible situation. A paid order that has
# shipped, arrived or come back is still `paid` in `orders.status`; the
# shipment row carries that part of the situation.
_ORDER_STATUS_EXT = {
    "delivered": "paid",
    "in_transit": "paid",
    "label_created": "paid",
    "returned": "paid",
}

ORDERS_EXT = [
    (row[0], row[1], _ORDER_STATUS_EXT.get(row[2], row[2]), row[3], f"ADR-{2001 + index}", row[7])
    for index, row in enumerate(ORDER_DATA_EXT)
]

ITEMS_EXT = [(row[0], row[8], row[9], 1, row[3]) for row in ORDER_DATA_EXT]

# Shipment status follows the order situation. Orders that were never paid have
# no shipment row, which is what the reshipment tool refuses on.
_SHIPMENT_STATUS_EXT = {
    "delivered": "delivered",
    "in_transit": "in_transit",
    "label_created": "label_created",
    "returned": "returned",
    "refunded": "returned",
    "partially_refunded": "delivered",
}

SHIPMENTS_EXT = [
    (
        f"SHP-{row[0][4:]}",
        row[0],
        _SHIPMENT_STATUS_EXT[row[2]],
        "DHL" if index % 2 == 0 else "Aramex",
        f"TRK{row[0][4:]}",
        f"ADR-{2001 + index}",
        row[7],
        row[7],
    )
    for index, row in enumerate(ORDER_DATA_EXT)
    if row[2] in _SHIPMENT_STATUS_EXT
]

REFUNDS_EXT = [
    # id, order, amount, status, reason, key, failure_code, failure_message, created
    ("REF-2001", "ORD-1045", 5600, "succeeded", "damaged_item", "seed-ref-2045", None, None, "2026-10-09T11:00:00Z"),
    ("REF-2002", "ORD-1046", 2340, "succeeded", "damaged_item", "seed-ref-2046", None, None, "2026-10-12T11:00:00Z"),
    ("REF-2003", "ORD-1047", 9900, "succeeded", "returned_item", "seed-ref-2047", None, None, "2026-10-15T11:00:00Z"),
    ("REF-2004", "ORD-1048", 1290, "succeeded", "damaged_item", "seed-ref-2048", None, None, "2026-10-18T11:00:00Z"),
    ("REF-2005", "ORD-1049", 4750, "succeeded", "customer_request", "seed-ref-2049", None, None, "2026-10-21T11:00:00Z"),
    ("REF-2006", "ORD-1050", 2000, "succeeded", "damaged_item", "seed-ref-2050", None, None, "2026-10-24T11:00:00Z"),
    ("REF-2007", "ORD-1051", 4300, "succeeded", "damaged_item", "seed-ref-2051", None, None, "2026-10-27T11:00:00Z"),
    ("REF-2008", "ORD-1052", 1400, "succeeded", "returned_item", "seed-ref-2052", None, None, "2026-10-30T11:00:00Z"),
    ("REF-2009", "ORD-1053", 490, "succeeded", "damaged_item", "seed-ref-2053", None, None, "2026-11-02T11:00:00Z"),
    ("REF-2010", "ORD-1054", 5800, "succeeded", "damaged_item", "seed-ref-2054", None, None, "2026-11-05T11:00:00Z"),
]


def seed(conn: sqlite3.Connection) -> sqlite3.Connection:
    """Create the schema and load the fixed dataset. Safe on an empty DB."""
    init_db(conn)
    for row in CUSTOMERS + CUSTOMERS_EXT:
        insert(conn, "customers", id=row[0], name=row[1], email=row[2])
    for row in ADDRESSES + ADDRESSES_EXT:
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
    for row in ORDERS + ORDERS_EXT:
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
    for row in ITEMS + ITEMS_EXT:
        insert(
            conn,
            "order_items",
            order_id=row[0],
            sku=row[1],
            name=row[2],
            quantity=row[3],
            unit_price_cents=row[4],
        )
    for row in REFUNDS + REFUNDS_EXT:
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
    for row in SHIPMENTS + SHIPMENTS_EXT:
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
