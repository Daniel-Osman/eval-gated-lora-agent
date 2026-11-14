# Composed mode — the promise, the failure, and the fix

Composed mode is the only mode with a provable no-false-claim property: the model
returns a plan of ids and an amount, and the sentence is rendered from the row
the action wrote. `docs/GATE.md` states the rules; `README.md` states the
guarantee. This page records the student side of that promise.

It used to record a failure: the trained student emitted a free-text plan list
where a composed plan was due. That failure is kept below as history. The
composed path now has a student output — and one limit that travels with it.

## The promise, from code

- Accepted fields: `order_id`, `address_id`, `shipment_id`, integer
  `amount_cents`. Any other key raises `ComposeError`, so no text channel exists.
- Actions: status, refund, reshipment, address. Address is confirm-or-refuse.
- Every one of the 20 `composed: true` bench cases renders a gate-`APPROVED`
  sentence from its `required_calls` state (`tests/test_compose.py`).

None of that changed. The promise is still the gate's and the renderer's.

## The failure, as history

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
`unknown action None`. Case N-31 (free text, same log) produced a plausible
reply, so the adapter loaded and the prompt format worked for the dominant path.

The cause was structural. All rows shared one system prompt, so a composed ticket
and a free-text refund ticket read alike while demanding different output shapes:
a plan object for one, a plan list plus a customer reply for the other. Nothing in
the prompt named the shape, so the model read the shape from the corpus — 14
composed rows against 164 free-text rows. Oversampling alone could not fix that:
more copies of a row whose prompt does not name the shape still ask the model to
guess, and repeats teach frequency, not contract.

## What changed, in two parts

Both live in `shopagent/sft.py`. Neither is a training-loop change; the Modal app
consumes the converted rows unchanged.

1. **One system prompt per mode.** `COMPOSED_SYSTEM_PROMPT` (`sft.py:60`) names
   the output contract: one JSON object, never a list of tool calls, never a
   customer reply, the four action names, the allowed keys, `amount_cents` as a
   positive integer on a refund only, ids only from the ticket. The sentence is
   declared to be rendered from the database row, not written by the model.
   Free-text prompting is byte-identical to the teacher's: `sft.py:43` imports
   `shopagent.teacher.SYSTEM_PROMPT`, `convert_row` swaps the system message only
   on `path == "composed"` (`sft.py:379-381`), and any row whose stored system
   message is not the teacher prompt is refused as `system_prompt_drift`
   (`sft.py:353-357`), so the mode marker is applied once and never stacked.
   The manifest records both prompt digests (`free_text b009947d…`,
   `composed e761165d…`), and `tests/test_sft.py` checks the rendered system
   turn against each constant: `test_composed_rows_carry_the_composed_prompt`,
   `test_free_text_rows_keep_the_teacher_prompt`.
2. **A solved rebalance of the composed train slice.** `COMPOSED_TRAIN_SHARE =
   0.40` (`sft.py:99`) with the target solved from the free-text count:
   `round(0.4 / 0.6 * 147) = 98`. Each of the 13 composed train rows keeps a
   copy; the 85 remaining copies spread 6 per row, and the seed
   (`REPEAT_SEED = 20261114`) gives 7 rows an eighth copy — 6 rows at 7, 7 at 8.
   Copies go to the training split only, and the order refuses to put two
   copies of one id side by side.

The result, from `data/sft/manifest.json`:

| | before | after |
|---|---|---|
| written rows | 178 | 263 |
| composed train instances | 13 (8.13 %) | 98 (**40.00 %**) |
| free-text train rows | 147 | 147 |
| supervised tokens (stored masks) | 17,748 | 19,693 |
| val rows | 18 | 18, unchanged |
| optimizer steps (cap 200) | 120 | 62 × 3 = 186 |
| digest | `5cab4f46…` | `bfe5bb3e61676dd4…` |

Val keeps its 18 rows, ids and order, and it is **not** byte-identical to the
previous run: the composed prompt swap rewrites the system turn of `T-N-104`,
the split's one composed row, and that is the only change to it. Val loss is
comparable with the previous curve, not measured on the same bytes.
`rebalance.val.changed` is `false`, and `test_val_split_and_bytes_are_unchanged`
proves the split and its bytes survive the **rebalance** — it does not compare
pre-fix and post-fix artifacts. The guard on "only composed rows moved" is the
free-text digest: 164 records, source order, `9519a868…87479`, the pre-fix value
(`tests/test_sft.py`, `GOLDEN_FREE_TEXT_SHA256`).

The retrain, `full-20261113T205627Z`, logged in `out/modal-train2.log`: 186
steps, loss 0.2042 → 0.0850 (first and last five logged points; points 0.2757 →
0.0994), 1174.7 s wall, 6.18 s/step, peak 21.22 GB, estimated $0.8299 against a
$3.39 cap. The loss was measured on 18,260 supervised tokens in the train split.

## What the checks show

`out/modal-check2.log`, one case: N-105 now returns exactly

```
{"action": "refund", "amount_cents": 2100, "order_id": "ORD-1059"}<|im_end|>
```

`parse: ok`; `matches required_calls: True` on `action`, `amount_cents` and
`order_id`. The adapter-active proof passed in the same log: peft names
`['default']`, active `['default']`, 116,727,808 trainable parameters equal to the
trained count, and a `lora_B` tensor `torch.equal` to the file on the Volume.

`out/modal-check3.log`, every composed case in the SFT file: **14/14 parse ok,
14/14 match `required_calls`.**

Both logs predate the banner fix: they still print the old `every case here was
seen in training` line. The code now states the split, and the logs stand as
records of what ran.

## The limit that travels with 14/14

**The split, stated exactly: 13 of the 14 composed cases were in the training
split, and N-104 is the validation row** — the only composed record among the 18
val rows. So 14/14 **includes one case the adapter never trained on.** That one
row buys one extra observation, not a generalisation claim: N-104 is still a
scenario from the same corpus, written by the same teacher under the same brief.

What 14/14 measures is the output schema across 14 seen composed cases. What it
does not measure is a plan for a case the corpus never held, and it says nothing
about choosing the right action.

So the composed property still belongs to the gate and the renderer. The student
now emits a plan that parses and matches on seen cases; a model that picks the
wrong action still causes a wrong action. The bench's composed cases test the
renderer, not the student.

## What is still open

- Stage 4: wire the Phase B judge, then score base 27B and 27B + adapter on the
  frozen bench and both holdouts. That is the first number that is neither a
  seen-case shape check nor a renderer test.
- No composed address trace exists (N-86, N-109, N-110 are unchanged), so
  composed mode still cannot originate an address change.
- 98 instances are 85 copies of 13 rows: the 40 % share buys the output contract,
  not ticket coverage.

Record: `EVIDENCE.md`, milestone 4b.
