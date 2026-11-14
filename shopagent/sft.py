"""Convert the admitted teacher traces into token-level SFT records.

Input: `data/train/shard-*.jsonl`, written by `python -m shopagent.cli teacher`.
Output: `data/sft/sft.jsonl` and `data/sft/manifest.json`.

A record carries the rendered chat text, its token ids, and a ``labels`` array
where every supervised token keeps its id and every other token is
``IGNORE_INDEX``. Supervision is the assistant plan text, plus the assistant
reply text for free-text rows. The system prompt, the customer ticket, every
tool result, the empty think block and the chat markers are masked.

The tokenizer is the one the Modal run will load: ``unsloth/Qwen3.8-27B`` at a
pinned revision, with thinking disabled. The unsloth repository's tokenizer
files differ from the plain Qwen repository's, so this module never substitutes
one for the other.

Span anchoring: the chat template merges consecutive ``tool`` messages into one
``user`` turn and wraps each result in ``<tool_response>``. Message index is
therefore not turn index. Spans are found by scanning the rendered turn markers
and searching for each assistant content inside its own turn only, so identical
text in a ticket or a tool result can never be picked as a target.

Every refusal is a hard ``SftError`` naming the case id and a reason code. The
one unavoidable drop is a source row whose id is a reviewer holdout id: it is
skipped and named in the manifest, never written.

Nothing here trains, fetches weights, or touches Modal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

from .teacher import SYSTEM_PROMPT as FREE_TEXT_SYSTEM_PROMPT

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CORPUS = ROOT / "data" / "train"
DEFAULT_OUT = ROOT / "data" / "sft"
DEFAULT_REVISION_MANIFEST = ROOT / "data" / "manifest.json"
DEFAULT_BENCH = ROOT / "bench" / "holdout"

TOKENIZER_ID = "unsloth/Qwen3.8-27B-unsloth-bnb-4bit"
TOKENIZER_REVISION = "8aa5f05d26b7205477066e1449e0af13f762a299"
TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "config.json")
TOKENIZER_CLASS = "Qwen3_5Tokenizer"
ENABLE_THINKING = False

# One mode, one prompt. The corpus carries the free-text prompt on every row;
# the composed path gets its own, so the output shape is decidable from the
# prompt instead of guessed from a ticket that reads the same in both modes.
COMPOSED_SYSTEM_PROMPT = """You serve the composed mode of a small online shop's support agent. You read
one customer ticket and return one action object.

Return one JSON object and nothing else. The object names one action and the ids
that action needs, with exactly these keys:
- {"action": "status", "order_id": "ORD-1001"}
- {"action": "refund", "order_id": "ORD-1001", "amount_cents": 2100}
- {"action": "reshipment", "order_id": "ORD-1001"}   (add "shipment_id": "SHP-1001" when the ticket names one shipment)
- {"action": "address", "address_id": "ADR-1001"}

Rules:
- Return an object, never a list of tool calls, and never write a customer reply.
- Use only the action names and keys above. Any other key is refused.
- amount_cents is a positive integer number of cents, and only a refund takes it.
- Use only the ids and amounts named in the ticket. Never invent an order, an
  amount, an address or a shipment.
- The sentence the customer reads is rendered from the database row this action
  writes, not written here.
""".strip(
    "\n"
)

MODE_PROMPTS = {"free_text": FREE_TEXT_SYSTEM_PROMPT, "composed": COMPOSED_SYSTEM_PROMPT}

# Copied from train/modal_app.py, where the trainer enforces them. The test
# suite parses those literals back out of that file, so the two cannot drift.
TRAIN_MAX_SEQ_LEN = 2560
TRAIN_DATASET_PIN = "bfe5bb3e61676dd4079cf2f81285ed82b9219b2659913924a6a20f7c2bd0e9cc"

IGNORE_INDEX = -100
SEED = 20261113
VAL_FRACTION = 0.1
TOKEN_BIN_WIDTH = 256
SHARD_GLOB = "shard-*.jsonl"
OUT_NAME = "sft.jsonl"
MANIFEST_NAME = "manifest.json"

# The composed train slice is rebalanced to this share, because the student read
# one shape from a corpus where 13 composed rows sat against 147 free-text rows.
COMPOSED_TRAIN_SHARE = 0.40
REPEAT_SEED = SEED + 1

# Copied from train/modal_app.py, where the trainer enforces them. The test
# suite parses those literals back out of that file, so the two cannot drift.
TRAIN_GRAD_ACCUM = 4
TRAIN_EPOCHS = 3
TRAIN_STEP_CAP = 200

REQUIRED_ROW_KEYS = ("case", "id", "messages", "path", "plan", "target")
RECORD_KEYS = ("brief_revision", "case", "id", "input_ids", "labels", "path", "shard", "split", "text")

MASK_RULE = (
    "Loss is kept only on tokens that decode to an assistant plan text and, for free-text rows, "
    "the assistant reply text; the system prompt, the customer ticket, every tool result, "
    "the empty think block and the chat markers are masked with -100."
)
VAL_LABEL = (
    "overfit watch only; every scenario is seen, so val loss is not generalisation"
)
OPEN_ITEMS = (
    "the assistant <|im_end|> and the empty think block are masked, so EOS is never in the loss; "
    "resolve in the Modal training slice",
)

TURN_MARKER = re.compile(r"<\|im_start\|>(system|user|assistant)\n")
TURN_END = "<|im_end|>"


class SftError(ValueError):
    """A row or an input that cannot become an SFT record. Always names why."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def display_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def load_tokenizer(fetch: bool = False):
    """The pinned training-target tokenizer, or a named refusal.

    Never falls back to another repository: the unsloth tokenizer files differ
    from the plain Qwen ones, so a fallback would compute a mask the training
    run cannot reproduce.
    """
    try:
        import transformers
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover - environment failure
        raise SftError("tokenizer_unavailable", "transformers is not installed; pip install 'transformers==5.19.0'") from exc

    if fetch:
        from huggingface_hub import snapshot_download

        snapshot_download(TOKENIZER_ID, revision=TOKENIZER_REVISION, allow_patterns=list(TOKENIZER_FILES))

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            TOKENIZER_ID, revision=TOKENIZER_REVISION, local_files_only=True
        )
    except Exception as exc:
        raise SftError(
            "tokenizer_uncached",
            f"{TOKENIZER_ID}@{TOKENIZER_REVISION[:8]} is not cached; run python -m shopagent.sft --fetch once",
        ) from exc

    name = type(tokenizer).__name__
    if name != TOKENIZER_CLASS:
        raise SftError("tokenizer_class", f"expected {TOKENIZER_CLASS}, loaded {name}")
    return tokenizer


def tokenizer_json_sha256() -> str | None:
    """Digest of the pinned tokenizer.json, or None when the cache cannot name it."""
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:  # pragma: no cover - environment failure
        return None
    cached = try_to_load_from_cache(TOKENIZER_ID, "tokenizer.json", revision=TOKENIZER_REVISION)
    if not isinstance(cached, str):
        return None
    path = Path(cached)
    return sha256_bytes(path.read_bytes()) if path.is_file() else None


def load_shards(corpus_dir: Path) -> list[dict]:
    """Every shard, its rows and a digest recomputed from the file bytes."""
    files = sorted(corpus_dir.glob(SHARD_GLOB))
    if not files:
        raise SftError("corpus_missing", f"no {SHARD_GLOB} under {display_path(corpus_dir)}")
    shards: list[dict] = []
    for path in files:
        payload = path.read_bytes()
        rows = []
        for lineno, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise SftError("shard_unreadable", f"{path.name}:{lineno} is not JSON: {exc}") from exc
        if not rows:
            raise SftError("shard_empty", path.name)
        shards.append(
            {"name": path.name, "path": display_path(path), "rows": len(rows), "sha256": sha256_bytes(payload), "records": rows}
        )
    return shards


def revisions_from_manifest(path: Path) -> dict[str, int]:
    """Map each run's `generated_at` to the teacher brief revision of that run.

    Rows carry no revision field, so attribution comes from the corpus manifest
    the teacher wrote. An unknown timestamp is a refusal, not a guess.
    """
    if not path.is_file():
        raise SftError("revision_manifest_missing", display_path(path))
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SftError("revision_manifest_unreadable", f"{display_path(path)}: {exc}") from exc
    mapping: dict[str, int] = {}
    for run in payload.get("runs") or []:
        stamp = run.get("generated_at")
        revision = run.get("brief_revision")
        if stamp is None or revision is None:
            continue
        if stamp in mapping and mapping[stamp] != revision:
            raise SftError("revision_conflict", f"{stamp} maps to both {mapping[stamp]} and {revision}")
        mapping[stamp] = int(revision)
    if not mapping:
        raise SftError("revision_manifest_empty", f"{display_path(path)} has no runs with a brief_revision")
    return mapping


def holdout_ids(bench_dir: Path) -> frozenset[str]:
    """Ids of every reviewer holdout row. No training row may wear one."""
    files = sorted(bench_dir.glob("*.jsonl"))
    if not files:
        raise SftError("holdout_missing", f"no holdout rows under {display_path(bench_dir)}")
    found: set[str] = set()
    for path in files:
        for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SftError("holdout_unreadable", f"{path.name}:{lineno}: {exc}") from exc
            if row.get("id"):
                found.add(row["id"])
    if not found:
        raise SftError("holdout_empty", display_path(bench_dir))
    return frozenset(found)


def render_chat(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, enable_thinking=ENABLE_THINKING
    )


def assistant_turns(text: str) -> list[tuple[str, int, int]]:
    """Every rendered turn as (role, body_start, body_end), in order."""
    matches = list(TURN_MARKER.finditer(text))
    if not matches or matches[0].start() != 0:
        raise SftError("render_start", "the rendered chat does not begin with a turn marker")
    if matches[0].group(1) != "system":
        raise SftError("render_order", f"the first turn is {matches[0].group(1)}, not system")
    turns: list[tuple[str, int, int]] = []
    for match in matches:
        body_start = match.end()
        body_end = text.find(TURN_END, body_start)
        if body_end < 0:
            raise SftError("turn_unclosed", f"{match.group(1)} turn at offset {body_start} has no closing marker")
        turns.append((match.group(1), body_start, body_end))
    return turns


def supervised_spans(text: str, messages: list[dict], tokenizer) -> tuple[list[int], list[tuple[int, int]]]:
    """Token ids for the rendered text and one kept token span per assistant turn.

    Every span is anchored to its own rendered turn, so text repeated in a
    ticket or a tool result is never selected. Each span must decode back to the
    assistant content exactly.
    """
    assistant_messages = [message for message in messages if message.get("role") == "assistant"]
    turns = [turn for turn in assistant_turns(text) if turn[0] == "assistant"]
    if len(assistant_messages) != len(turns):
        raise SftError(
            "turn_mismatch",
            f"{len(assistant_messages)} assistant messages but {len(turns)} assistant turns in the render",
        )

    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids: list[int] = encoded["input_ids"]
    offsets = encoded["offset_mapping"]

    spans: list[tuple[int, int]] = []
    for message, (_, body_start, body_end) in zip(assistant_messages, turns):
        content = message.get("content")
        if not isinstance(content, str) or not content:
            raise SftError("assistant_empty", "an assistant message carries no content")
        start = text.find(content, body_start)
        if start < 0 or start >= body_end or start + len(content) > body_end:
            raise SftError(
                "content_absent",
                f"assistant content not found inside its own turn ({body_start}..{body_end})",
            )
        if text.find(content, start + 1, body_end) >= 0:
            raise SftError("content_ambiguous", "assistant content appears twice inside its own turn")
        end = start + len(content)
        keep = [index for index, (left, right) in enumerate(offsets) if left >= start and right <= end and right > left]
        if not keep:
            raise SftError("mask_empty", "no token lies inside the assistant content")
        if tokenizer.decode([ids[index] for index in keep]) != content:
            raise SftError("mask_misaligned", "kept tokens do not decode back to the assistant content")
        spans.append((keep[0], keep[-1] + 1))
    return ids, spans


def convert_row(tokenizer, row: dict, shard: str, revision: int) -> dict:
    """One admitted trace to one SFT record. Refuses with a named reason."""
    missing = [key for key in REQUIRED_ROW_KEYS if key not in row]
    if missing:
        raise SftError("shape", f"row is missing {missing}")
    case = row.get("case")
    row_id = row.get("id")
    label = f"{row_id} ({case})"
    path = row.get("path")
    messages = row.get("messages")
    target = row.get("target") or {}

    if not isinstance(messages, list) or len(messages) < 3:
        raise SftError("shape", f"{label}: messages must hold at least three entries")
    roles = [message.get("role") for message in messages]
    if roles[0] != "system" or roles[1] != "user":
        raise SftError("shape", f"{label}: expected system, user first, got {roles[:2]}")
    if roles[2] != "assistant":
        raise SftError("shape", f"{label}: the third message must be the assistant plan")
    if messages[2].get("content") != json.dumps(row.get("plan")):
        raise SftError("plan_text_mismatch", f"{label}: messages[2] is not json.dumps(plan)")

    if messages[0].get("content") != FREE_TEXT_SYSTEM_PROMPT:
        raise SftError(
            "system_prompt_drift",
            f"{label}: messages[0] is not the teacher system prompt; the mode marker is applied once, not stacked",
        )

    if path == "free_text":
        if target.get("type") != "plan_and_reply":
            raise SftError("target_type", f"{label}: expected plan_and_reply, got {target.get('type')!r}")
        if roles[-1] != "assistant":
            raise SftError("shape", f"{label}: a free-text row must end with the assistant reply")
        if len(messages) < 5:
            raise SftError("shape", f"{label}: a free-text row needs plan, a tool result and a reply")
        if any(role != "tool" for role in roles[3:-1]):
            raise SftError("shape", f"{label}: expected only tool results between plan and reply, got {roles[3:-1]}")
        if messages[-1].get("content") != target.get("reply"):
            raise SftError("reply_text_mismatch", f"{label}: messages[-1] is not target.reply")
    elif path == "composed":
        if target.get("type") != "plan":
            raise SftError("target_type", f"{label}: expected plan, got {target.get('type')!r}")
        if len(messages) != 3:
            raise SftError("shape", f"{label}: a composed row must hold exactly system, user, plan")
    else:
        raise SftError("path", f"{label}: unknown path {path!r}")

    rendered = messages
    if path == "composed":
        rendered = [dict(message) for message in messages]
        rendered[0]["content"] = MODE_PROMPTS["composed"]

    text = render_chat(tokenizer, rendered)
    ids, spans = supervised_spans(text, rendered, tokenizer)

    labels = [IGNORE_INDEX] * len(ids)
    for start, end in spans:
        labels[start:end] = ids[start:end]

    return {
        "id": row_id,
        "case": case,
        "path": path,
        "shard": shard,
        "brief_revision": revision,
        "text": text,
        "input_ids": ids,
        "labels": labels,
    }


def split_records(records: list[dict], seed: int = SEED, fraction: float = VAL_FRACTION) -> set[str]:
    """Deterministic val ids: shuffle the sorted ids with a fixed seed."""
    val_count = round(fraction * len(records))
    shuffled = sorted(record["id"] for record in records)
    random.Random(seed).shuffle(shuffled)
    return set(sorted(shuffled[:val_count]))


def target_composed_train(free_text_train: int) -> int:
    """Composed train rows that put composed at COMPOSED_TRAIN_SHARE of the split.

    ``composed / (free + composed) = share`` gives
    ``composed = share / (1 - share) * free``.
    """
    if not 0 < COMPOSED_TRAIN_SHARE < 1:
        raise SftError("share", f"COMPOSED_TRAIN_SHARE must sit between 0 and 1, got {COMPOSED_TRAIN_SHARE}")
    return round(COMPOSED_TRAIN_SHARE / (1 - COMPOSED_TRAIN_SHARE) * free_text_train)


def plan_composed_repeats(records: list[dict], seed: int = SEED) -> dict:
    """How many copies of each composed train row to write.

    Repeats go to the training split only. Every base row keeps one copy; the
    remainder is spread one extra copy each, and the seed decides which rows
    take it.
    """
    train = [record for record in records if record["split"] == "train"]
    free_text = sum(1 for record in train if record["path"] == "free_text")
    composed = sorted(record["id"] for record in train if record["path"] == "composed")
    target = target_composed_train(free_text)
    copies = max(0, target - len(composed))
    per_row, remainder = divmod(copies, len(composed)) if composed else (0, 0)
    shuffled = list(composed)
    random.Random(seed + 1).shuffle(shuffled)
    extra = sorted(shuffled[:remainder])
    counts = {row_id: 1 + per_row + (1 if row_id in extra else 0) for row_id in composed}
    return {
        "free_text_train": free_text,
        "base_composed": len(composed),
        "target": target,
        "copies": copies,
        "per_row": per_row,
        "remainder": remainder,
        "extra_copy_ids": extra,
        "counts": counts,
    }


def order_train(records: list[dict], counts: dict[str, int], seed: int = SEED) -> list[dict]:
    """Train rows in a deterministic order that never puts one id beside itself.

    Free-text rows keep their file order. Composed copies are drawn by a seeded
    scheduler that refuses the previous id, then spread across the free-text
    gaps, so no two neighbours share an id.
    """
    train = [record for record in records if record["split"] == "train"]
    free_text = [record for record in train if record["path"] == "free_text"]
    by_id = {record["id"]: record for record in train if record["path"] == "composed"}

    remaining = dict(counts)
    total = sum(remaining.values())
    rng = random.Random(seed + 2)
    priority = list(remaining)
    rng.shuffle(priority)
    rank = {row_id: index for index, row_id in enumerate(priority)}
    sequence: list[dict] = []
    previous: str | None = None
    while len(sequence) < total:
        candidates = [row_id for row_id, left in remaining.items() if left > 0 and row_id != previous]
        if not candidates:
            raise SftError("repeat_order", f"cannot separate {previous} from itself")
        pick = min(candidates, key=lambda row_id: (-remaining[row_id], rank[row_id]))
        sequence.append(by_id[pick])
        remaining[pick] -= 1
        previous = pick

    ordered: list[dict] = []
    cursor = 0
    for position, record in enumerate(free_text):
        ordered.append(record)
        target_seen = (position + 1) * len(sequence) // (len(free_text) + 1)
        while cursor < target_seen:
            ordered.append(sequence[cursor])
            cursor += 1
    ordered.extend(sequence[cursor:])
    return ordered


def histogram(values: list[int], width: int = TOKEN_BIN_WIDTH) -> dict:
    counts: dict[int, int] = {}
    for value in values:
        low = (value // width) * width
        counts[low] = counts.get(low, 0) + 1
    return {f"{low}-{low + width - 1}": counts[low] for low in sorted(counts)}


def _schema(records: list[dict]) -> dict | None:
    """One shape for a group of records, or None when the group disagrees with itself."""
    keys = None
    for record in records:
        current = tuple(sorted(record))
        if keys is None:
            keys = current
        elif keys != current:
            return None
        if len(record["input_ids"]) != len(record["labels"]):
            return None
        if not isinstance(record["text"], str) or not record["text"]:
            return None
        kept = {label for label in record["labels"] if label != IGNORE_INDEX}
        if not kept <= set(record["input_ids"]):
            return None
    return {"keys": list(keys), "label_type": "int", "labels_match_ids": True}


def build(
    corpus_dir: Path = DEFAULT_CORPUS,
    out_dir: Path = DEFAULT_OUT,
    revision_manifest: Path = DEFAULT_REVISION_MANIFEST,
    bench_dir: Path = DEFAULT_BENCH,
    fetch: bool = False,
    tokenizer=None,
    rebalance: bool = True,
) -> dict:
    """Convert every shard row, rebalance the composed train slice, write both files.

    `rebalance=False` writes the base records only; the test suite uses it to
    prove the val split and val bytes do not move when the rebalance is on.
    """
    tokenizer = tokenizer or load_tokenizer(fetch=fetch)
    shards = load_shards(corpus_dir)
    revisions = revisions_from_manifest(revision_manifest)
    holdout = holdout_ids(bench_dir)

    records: list[dict] = []
    drops: list[dict] = []
    by_shard: dict[str, dict] = {}
    by_revision: dict[str, dict] = {}
    plan_shape: dict[str, str] = {}

    for shard in shards:
        stats = {"source_rows": shard["rows"], "converted": 0, "drops": 0, "sha256": shard["sha256"]}
        for lineno, row in enumerate(shard["records"], start=1):
            row_id = row.get("id")
            revision = revisions.get((row.get("provenance") or {}).get("generated_at"))
            if revision is None:
                raise SftError(
                    "brief_revision_unknown",
                    f"{row_id} ({row.get('case')}): generated_at is not in {display_path(revision_manifest)}",
                )
            bucket = by_revision.setdefault(str(revision), {"source_rows": 0, "converted": 0, "free_text": 0, "composed": 0})
            bucket["source_rows"] += 1

            if row_id in holdout:
                drops.append(
                    {
                        "case": row.get("case"),
                        "id": row_id,
                        "reason": "holdout_id",
                        "shard": shard["name"],
                        "line": lineno,
                    }
                )
                stats["drops"] += 1
                continue

            record = convert_row(tokenizer, row, shard["name"], revision)
            records.append(record)
            plan_shape[record["id"]] = "list" if isinstance(row.get("plan"), list) else "dict"
            stats["converted"] += 1
            bucket["converted"] += 1
            bucket[record["path"]] += 1
        by_shard[shard["name"]] = stats

    source_rows = sum(shard["rows"] for shard in shards)
    if source_rows != len(records) + len(drops):
        raise SftError(
            "count_mismatch",
            f"{source_rows} source rows but {len(records)} converted and {len(drops)} named drops",
        )

    longest = max(records, key=lambda record: len(record["input_ids"]))
    if len(longest["input_ids"]) > TRAIN_MAX_SEQ_LEN:
        raise SftError(
            "record_too_long",
            f"{longest['id']} is {len(longest['input_ids'])} tokens, above MAX_SEQ_LEN {TRAIN_MAX_SEQ_LEN}",
        )

    val_ids = split_records(records)
    for record in records:
        record["split"] = "val" if record["id"] in val_ids else "train"

    base_records = records
    base_tokens = sum(len(record["input_ids"]) for record in base_records)
    base_kept = sum(sum(1 for label in record["labels"] if label != IGNORE_INDEX) for record in base_records)

    groups: dict[str, dict[str, dict]] = {}
    for record in base_records:
        group = groups.setdefault(record["path"], {}).setdefault(
            f"revision_{record['brief_revision']}", {"rows": 0, "plan_shapes": {}, "schema": None}
        )
        group["rows"] += 1
        shape = plan_shape.get(record["id"], "unknown")
        group["plan_shapes"][shape] = group["plan_shapes"].get(shape, 0) + 1

    for path, revisions_in_path in groups.items():
        schemas = {}
        for name, group in revisions_in_path.items():
            subset = [
                record
                for record in base_records
                if record["path"] == path and f"revision_{record['brief_revision']}" == name
            ]
            schema = _schema(subset)
            if schema is None:
                raise SftError("schema_drift", f"{path} {name}: records disagree on their shape")
            group["schema"] = schema
            schemas[name] = json.dumps(schema, sort_keys=True)
        if len(set(schemas.values())) > 1:
            raise SftError("schema_drift", f"{path}: record schema differs between brief revisions")

    rebalance_plan = None
    if rebalance:
        rebalance_plan = plan_composed_repeats(base_records)
        val_records = [record for record in base_records if record["split"] == "val"]
        records = val_records + order_train(base_records, rebalance_plan["counts"])
    else:
        records = base_records

    train_records = [record for record in records if record["split"] == "train"]
    steps_per_epoch = math.ceil(len(train_records) / TRAIN_GRAD_ACCUM) if train_records else 0
    total_steps = steps_per_epoch * TRAIN_EPOCHS

    lengths = [len(record["input_ids"]) for record in records]
    kept_lengths = [sum(1 for label in record["labels"] if label != IGNORE_INDEX) for record in records]
    total_tokens = sum(lengths)
    total_kept = sum(kept_lengths)

    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = out_dir / OUT_NAME
    payload = "\n".join(json.dumps(record, sort_keys=True, ensure_ascii=False) for record in records) + "\n"
    output_bytes = payload.encode("utf-8")
    output_path.write_bytes(output_bytes)

    rebalance_block = None
    if rebalance_plan is not None:
        share = COMPOSED_TRAIN_SHARE
        product = share / (1 - share) * rebalance_plan["free_text_train"]
        before_composed = rebalance_plan["base_composed"]
        after_composed = rebalance_plan["target"]
        before_rows = rebalance_plan["free_text_train"] + before_composed
        after_rows = rebalance_plan["free_text_train"] + after_composed
        counts = sorted(rebalance_plan["counts"].values())
        rebalance_block = {
            "applied": True,
            "rule": (
                "composed_train = round(0.4 / 0.6 * free_text_train); every composed row keeps one copy, "
                "the seed spreads the remainder one copy each, then the copies are shuffled apart"
            ),
            "share_target": share,
            "repeat_seed": REPEAT_SEED,
            "derivation": {
                "free_text_train": rebalance_plan["free_text_train"],
                "expression": f"round({share} / {1 - share:.1f} * {rebalance_plan['free_text_train']})",
                "product": round(product, 4),
                "rounded": rebalance_plan["target"],
                "base_composed_train": before_composed,
                "copies_added": rebalance_plan["copies"],
                "copies_per_row": rebalance_plan["per_row"],
                "remainder": rebalance_plan["remainder"],
                "extra_copy_ids": rebalance_plan["extra_copy_ids"],
                "repeat_factor": round(after_composed / before_composed, 4) if before_composed else None,
                "counts": {str(count): counts.count(count) for count in sorted(set(counts))},
            },
            "epoch_composition": {
                "before": {
                    "train_rows": before_rows,
                    "free_text": rebalance_plan["free_text_train"],
                    "composed": before_composed,
                    "composed_share": round(before_composed / before_rows, 4) if before_rows else 0,
                },
                "after": {
                    "train_rows": after_rows,
                    "free_text": rebalance_plan["free_text_train"],
                    "composed": after_composed,
                    "composed_share": round(after_composed / after_rows, 4) if after_rows else 0,
                },
            },
            "val": {
                "rows": len(val_ids),
                "free_text": sum(
                    1 for record in base_records if record["split"] == "val" and record["path"] == "free_text"
                ),
                "composed": sum(
                    1 for record in base_records if record["split"] == "val" and record["path"] == "composed"
                ),
                "changed": False,
            },
        }

    manifest = {
        "version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "corpus": display_path(corpus_dir),
        "source": {
            "rows": source_rows,
            "shards": [
                {"name": shard["name"], "path": shard["path"], "rows": shard["rows"], "sha256": shard["sha256"]}
                for shard in shards
            ],
        },
        "converted": len(base_records),
        "composition": {
            path: sum(1 for record in base_records if record["path"] == path) for path in ("free_text", "composed")
        },
        "rebalance": rebalance_block,
        "token_delta": {
            "supervised": {"before": base_kept, "after": total_kept, "added": total_kept - base_kept},
            "rendered": {"before": base_tokens, "after": total_tokens, "added": total_tokens - base_tokens},
            "why": (
                "a written copy repeats its composed record byte for byte, so every added supervised token is a "
                "composed plan token already present in the base rows; the rendered rise also repeats each copy's "
                "system prompt and ticket"
            ),
        },
        "budget": {
            "grad_accum": TRAIN_GRAD_ACCUM,
            "epochs": TRAIN_EPOCHS,
            "step_cap": TRAIN_STEP_CAP,
            "train_rows": len(train_records),
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "fits_cap": total_steps <= TRAIN_STEP_CAP,
        },
        "drops": drops,
        "by_shard": by_shard,
        "by_brief_revision": by_revision,
        "schema": {
            "identical_within_path": True,
            "keys": list(RECORD_KEYS),
            "groups": groups,
        },
        "tokenizer": {
            "id": TOKENIZER_ID,
            "revision": TOKENIZER_REVISION,
            "class": type(tokenizer).__name__,
            "tokenizer_json_sha256": tokenizer_json_sha256(),
            "enable_thinking": ENABLE_THINKING,
            "environment": {"transformers": _transformers_version()},
        },
        "prompts": {
            "free_text": {
                "source": "shopagent.teacher.SYSTEM_PROMPT",
                "text": FREE_TEXT_SYSTEM_PROMPT,
                "sha256": sha256_bytes(FREE_TEXT_SYSTEM_PROMPT.encode("utf-8")),
            },
            "composed": {
                "source": "shopagent.sft.COMPOSED_SYSTEM_PROMPT",
                "text": COMPOSED_SYSTEM_PROMPT,
                "sha256": sha256_bytes(COMPOSED_SYSTEM_PROMPT.encode("utf-8")),
            },
        },
        "train_app_pin": {
            "constant": "DATASET_SHA256",
            "value": TRAIN_DATASET_PIN,
            "matches_output": TRAIN_DATASET_PIN == sha256_bytes(output_bytes),
            "note": (
                "train/modal_app.py pins the dataset digest; retraining is blocked until a later "
                "slice accepts this artifact and repins that line"
            ),
        },
        "config": {
            "seed": SEED,
            "val_fraction": VAL_FRACTION,
            "ignore_index": IGNORE_INDEX,
            "add_generation_prompt": False,
            "enable_thinking": ENABLE_THINKING,
            "shard_glob": SHARD_GLOB,
            "token_bin_width": TOKEN_BIN_WIDTH,
            "mask_rule": MASK_RULE,
            "split_rule": "shuffle the sorted record ids with random.Random(seed); the first round(fraction*n) are val",
        },
        "mask_rule": MASK_RULE,
        "token_lengths": {
            "total": total_tokens,
            "min": min(lengths),
            "median": statistics.median(lengths),
            "max": max(lengths),
            "histogram": {"bin_width": TOKEN_BIN_WIDTH, "bins": histogram(lengths)},
        },
        "kept_tokens": {
            "total": total_kept,
            "min": min(kept_lengths),
            "median": statistics.median(kept_lengths),
            "max": max(kept_lengths),
            "fraction_of_all": round(total_kept / total_tokens, 4),
        },
        "val": {"fraction": VAL_FRACTION, "rows": len(val_ids), "ids": sorted(val_ids), "label": VAL_LABEL},
        "output": {
            "dir": display_path(out_dir),
            "file": OUT_NAME,
            "rows": len(records),
            "source_records": len(base_records),
            "sha256": sha256_bytes(output_bytes),
        },
        "open_items": list(OPEN_ITEMS),
    }

    manifest_path = out_dir / MANIFEST_NAME
    manifest_path.write_bytes((json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    return manifest


def _transformers_version() -> str:
    try:
        import transformers

        return transformers.__version__
    except ImportError:  # pragma: no cover - environment failure
        return "absent"


def _report(manifest: dict) -> list[str]:
    lines = [
        f"sft: {manifest['converted']} source records -> {manifest['output']['rows']} written rows "
        f"-> {manifest['output']['dir']}/{OUT_NAME} (sha256 {manifest['output']['sha256'][:16]})"
    ]
    for shard in manifest["source"]["shards"]:
        stats = manifest["by_shard"][shard["name"]]
        lines.append(
            f"  {shard['name']}: {shard['rows']} rows, sha256 {shard['sha256'][:16]}, "
            f"{stats['converted']} converted, {stats['drops']} dropped"
        )
    for revision, stats in sorted(manifest["by_brief_revision"].items()):
        lines.append(
            f"  brief revision {revision}: {stats['converted']} converted "
            f"({stats['free_text']} free text, {stats['composed']} composed), {stats['source_rows']} source rows"
        )
    if manifest["drops"]:
        for drop in manifest["drops"]:
            lines.append(f"  dropped {drop['id']} ({drop['case']}) from {drop['shard']}:{drop['line']}: {drop['reason']}")
    else:
        lines.append("  drops: none")
    tokens = manifest["token_lengths"]
    kept = manifest["kept_tokens"]
    lines.append(
        f"  tokens: min {tokens['min']} median {tokens['median']} max {tokens['max']}; "
        f"kept min {kept['min']} median {kept['median']} max {kept['max']} "
        f"({kept['fraction_of_all']:.1%} of all)"
    )
    lines.append(f"  val: {manifest['val']['rows']} rows, {manifest['val']['label']}")
    rebalance = manifest.get("rebalance")
    if rebalance:
        before = rebalance["epoch_composition"]["before"]
        after = rebalance["epoch_composition"]["after"]
        derivation = rebalance["derivation"]
        lines.append(
            f"  rebalance: {derivation['copies_added']} composed repeats, {derivation['copies_per_row']} per row "
            f"+ {derivation['remainder']} extra (factor {derivation['repeat_factor']})"
        )
        lines.append(
            f"  epoch composition: before {before['free_text']} free + {before['composed']} composed "
            f"({before['composed_share']:.2%}); after {after['free_text']} free + {after['composed']} composed "
            f"({after['composed_share']:.2%}) over {after['train_rows']} rows"
        )
    delta = manifest["token_delta"]
    lines.append(
        f"  supervised tokens: {delta['supervised']['before']} -> {delta['supervised']['after']} "
        f"(+{delta['supervised']['added']}); rendered {delta['rendered']['before']} -> "
        f"{delta['rendered']['after']} (+{delta['rendered']['added']})"
    )
    budget = manifest["budget"]
    lines.append(
        f"  budget: {budget['train_rows']} train rows / {budget['grad_accum']} = {budget['steps_per_epoch']} "
        f"steps per epoch x {budget['epochs']} = {budget['total_steps']} (cap {budget['step_cap']}, "
        f"fits {budget['fits_cap']})"
    )
    prompts = manifest["prompts"]
    lines.append(
        f"  prompts: free_text sha256 {prompts['free_text']['sha256'][:16]}, "
        f"composed sha256 {prompts['composed']['sha256'][:16]}"
    )
    pin = manifest["train_app_pin"]
    if not pin["matches_output"]:
        lines.append(
            f"  dataset pin: {pin['constant']} is {pin['value'][:16]} and does not match this output; "
            "retraining is blocked until it is repinned"
        )
    lines.append(f"  histogram: bin width {TOKEN_BIN_WIDTH}")
    for bucket, count in tokens["histogram"]["bins"].items():
        lines.append(f"    {bucket:>9}: {count}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m shopagent.sft", description=__doc__.splitlines()[0])
    parser.add_argument("--corpus", default=str(DEFAULT_CORPUS), help="shard directory (default data/train)")
    parser.add_argument("--out", default=str(DEFAULT_OUT), help="output directory (default data/sft)")
    parser.add_argument(
        "--revision-manifest", default=str(DEFAULT_REVISION_MANIFEST), help="corpus manifest with runs (default data/manifest.json)"
    )
    parser.add_argument("--bench", default=str(DEFAULT_BENCH), help="holdout directory (default bench/holdout)")
    parser.add_argument("--fetch", action="store_true", help="download the pinned tokenizer files, then convert")
    args = parser.parse_args(argv)

    try:
        manifest = build(
            corpus_dir=Path(args.corpus),
            out_dir=Path(args.out),
            revision_manifest=Path(args.revision_manifest),
            bench_dir=Path(args.bench),
            fetch=args.fetch,
        )
    except SftError as exc:
        print(f"sft: {exc}", file=sys.stderr)
        return 2
    for line in _report(manifest):
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
