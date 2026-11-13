# Composed mode — the promise, and the missing student

Composed mode is the only mode with a provable no-false-claim property: the model
returns a plan of ids and an amount, and the sentence is rendered from the row
the action wrote. `docs/GATE.md` states the rules; `README.md` states the
guarantee. This page records the one fact those pages cannot show: **the trained
student does not yet emit a composed plan.**

## The promise, from code

- Accepted fields: `order_id`, `address_id`, `shipment_id`, integer
  `amount_cents`. Any other key raises `ComposeError`, so no text channel exists.
- Actions: status, refund, reshipment, address. Address is confirm-or-refuse.
- Every one of the 20 `composed: true` bench cases renders a gate-`APPROVED`
  sentence from its `required_calls` state (`tests/test_compose.py`).

None of that changed.

## What the student did

`out/modal-check.log`, run `full-20261113T180308Z`, case N-105 (composed,
training split, seen). Target plan:

```json
{"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059"}
```

Raw output:

```json
[{"args": {"order_id": "ORD-1059"}, "name": "get_order"}, {"args": {"amount_cents": 2100, "order_id": "ORD-1059"}, "name": "refund_order"}]
```

The student wrote the free-text plan list. The validator refused:
`unknown action None`.

Case N-31 (free text, same log) produced a plausible reply, so the adapter loads
and the prompt format works for the dominant path.

## The likely cause

14 composed rows against 164 free-text rows, under one shared system prompt. The
model learned the shape the corpus repeats. The converter, the renderer, the
validator and the adapter load are all proven separately; the missing piece is a
model that chooses the composed shape at all.

## What is open

How the student learns the composed plan shape is Daniel's decision, not a
patch:

1. a separate system prompt per mode, so the shape is selected by the prompt;
2. a rebalanced composed slice, so the corpus stops teaching one shape;
3. constrained decoding against the composed schema;
4. something else.

Until one is chosen and measured, treat composed mode as a property of the
renderer, not of the model. The bench's composed cases test the renderer, not the
student.

Record: `EVIDENCE.md`, milestone 4.
