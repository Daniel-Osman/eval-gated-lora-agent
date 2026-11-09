"""The referee: state snapshots, diffs and declarative state expectations.

The referee never reads the model's log. It reads tables. `snapshot` is taken
before a run, `diff` reports what actually changed, and `check_expect` evaluates
a benchmark case's marking guide against the final database.
"""

from __future__ import annotations

import sqlite3
from typing import Any

from .db import rows

# Columns that define state. Timestamps are deliberately excluded: two runs of
# the same case must diff equal even on separate days.
SNAPSHOT: dict[str, list[str]] = {
    "orders": ["id", "customer_id", "status", "total_cents"],
    "refunds": ["id", "order_id", "amount_cents", "status", "failure_code", "idempotency_key"],
    "shipments": ["id", "order_id", "status", "carrier", "tracking_number", "address_id"],
    "addresses": ["id", "customer_id", "line1", "city", "postcode", "country"],
}

TABLE_COLUMNS = {table: cols for table, cols in SNAPSHOT.items()}


def snapshot(conn: sqlite3.Connection) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for table, cols in SNAPSHOT.items():
        select = ", ".join(cols)
        result[table] = rows(conn, f"SELECT {select} FROM {table} ORDER BY id")
    return result


def diff(before: dict[str, list[dict]], after: dict[str, list[dict]]) -> dict[str, dict]:
    changes: dict[str, dict] = {}
    for table in SNAPSHOT:
        old = {row["id"]: row for row in before.get(table, [])}
        new = {row["id"]: row for row in after.get(table, [])}
        added = [new[key] for key in sorted(new) if key not in old]
        removed = [old[key] for key in sorted(old) if key not in new]
        changed = []
        for key in sorted(new):
            if key in old and old[key] != new[key]:
                changed.append({"id": key, "before": old[key], "after": new[key]})
        if added or removed or changed:
            changes[table] = {"added": added, "removed": removed, "changed": changed}
    return changes


def is_empty(changes: dict[str, dict]) -> bool:
    return not any(v["added"] or v["removed"] or v["changed"] for v in changes.values())


def describe(changes: dict[str, dict]) -> list[str]:
    lines: list[str] = []
    for table, change in changes.items():
        for row in change["added"]:
            lines.append(f"+ {table} {row['id']}")
        for row in change["removed"]:
            lines.append(f"- {table} {row['id']}")
        for entry in change["changed"]:
            fields = ", ".join(
                f"{k}: {entry['before'][k]!r} -> {entry['after'][k]!r}"
                for k in entry["after"]
                if entry["before"].get(k) != entry["after"].get(k)
            )
            lines.append(f"~ {table} {entry['id']} ({fields})")
    return lines


def check_expect(conn: sqlite3.Connection, expectations: list[dict]) -> list[str]:
    """Evaluate a case's `state_expect` list. Returns human-readable failures."""
    failures: list[str] = []
    for index, expectation in enumerate(expectations):
        table = expectation["table"]
        match = expectation.get("match", {})
        where = " AND ".join(f"{key} = ?" for key in match)
        sql = f"SELECT * FROM {table}" + (f" WHERE {where}" if where else "")
        found = rows(conn, sql, tuple(match.values()))
        label = f"state_expect[{index}] {table} {match or '*'}"

        if "count" in expectation and len(found) != int(expectation["count"]):
            failures.append(f"{label}: expected count {expectation['count']}, found {len(found)}")
            continue
        if "fields" in expectation:
            if len(found) != 1:
                failures.append(f"{label}: expected exactly 1 row for field check, found {len(found)}")
                continue
            for key, value in expectation["fields"].items():
                if found[0].get(key) != value:
                    failures.append(f"{label}: field {key} expected {value!r}, found {found[0].get(key)!r}")
        if "absent" in expectation and expectation["absent"] and found:
            failures.append(f"{label}: expected no rows, found {len(found)}")
    return failures


def check_tables(expectations: list[dict]) -> list[str]:
    """Static validation of a marking guide, used by the benchmark validator."""
    errors: list[str] = []
    for index, expectation in enumerate(expectations):
        table = expectation.get("table")
        if table not in SNAPSHOT:
            errors.append(f"state_expect[{index}]: unknown table {table!r}")
            continue
        keys = set(expectation.get("match", {}))
        keys |= set(expectation.get("fields", {}))
        bad = keys - set(SNAPSHOT[table])
        if bad:
            errors.append(f"state_expect[{index}]: unknown columns {sorted(bad)} for {table}")
    return errors
