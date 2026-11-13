# Token-level SFT data

Generated 2026-11-13 from the 178 admitted traces in `data/train/`. Read it with
`manifest.json`; regenerate it with `python -m shopagent.sft`.

## Files

- `sft.jsonl` - one record per admitted trace, deterministic order (shards by
  name, rows in file order).
- `manifest.json` - source shard digests recomputed from bytes, converted count,
  per-shard and per-brief-revision counts, token lengths and histogram, the
  tokenizer id and revision, the exact config, the mask rule, the val ids and
  every drop.

Both are gitignored. This guide is not.

## Record

```json
{"id": "T-A-01", "case": "A-01", "path": "free_text", "shard": "shard-000.jsonl",
 "brief_revision": 2, "split": "train", "text": "<|im_start|>system\n...",
 "input_ids": [248045, 8678, ...], "labels": [-100, -100, ...]}
```

`text` is the rendered chat (`apply_chat_template`, `add_generation_prompt=False`,
`enable_thinking=False`). `input_ids` is that text under the pinned tokenizer.
`labels` is `input_ids` where the token is supervised and `-100` everywhere else,
so a `Trainer` can read it without a second mask step.

## Mask rule

Loss is kept only on tokens that decode to an assistant plan text and, for
free-text rows, the assistant reply text; the system prompt, the customer
ticket, every tool result, the empty think block and the chat markers are masked
with `-100`.

Spans are anchored to rendered turns, not to message index: the chat template
merges consecutive `tool` messages into one `user` turn, so a search over the
whole text could pick identical text from a ticket or a tool result. Each kept
span must decode back to its assistant content, and the converter refuses
otherwise.

## Limits

- Every benchmark scenario is in the corpus, so `split: "val"` (10 %, fixed
  seed) watches overfit only. It proves nothing about generalisation.
- The assistant `<|im_end|>` and the empty think block are masked, so EOS is
  never in the loss. The Modal training slice must resolve that before training.

## Commands

```bash
python -m shopagent.sft                      # convert and write the manifest
python -m shopagent.sft --fetch              # populate the pinned tokenizer first
python -m unittest tests.test_sft -q         # the converter's tests
```

The tokenizer is `unsloth/Qwen3.8-27B-unsloth-bnb-4bit` at revision
`8aa5f05d…`; its files differ from the plain `Qwen/Qwen3.8-27B` repository, so
the converter never substitutes one for the other.
