"""The SFT converter: span anchoring, masking, digests, revisions, refusals.

Every test needs the real tokenizer, because the mask is a token-level fact of
that tokenizer. `setUpModule` loads it once; a missing pinned snapshot raises
`SftError` naming the setup command, so the module goes red instead of skipping.
The two real-corpus tests do the same for `data/train` through `require_corpus`.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from shopagent import sft
from shopagent.sft import (
    IGNORE_INDEX,
    SEED,
    TOKENIZER_ID,
    TOKENIZER_REVISION,
    SftError,
    build,
    convert_row,
    holdout_ids,
    load_tokenizer,
    supervised_spans,
)

STAMP_1 = "2026-11-13T01:14:45+00:00"
STAMP_2 = "2026-11-13T01:39:48+00:00"

PLAN = [
    {"name": "get_order", "args": {"order_id": "ORD-1001"}},
    {"name": "refund_order", "args": {"amount_cents": 2400, "order_id": "ORD-1001"}},
]
COMPOSED_PLAN = {"action": "status", "order_id": "ORD-1037"}
TICKET = "I need a $24.00 refund for order ORD-1001, please."
REPLY = "The $24.00 refund for ORD-1001 failed: the card was declined and the order remains paid."
TOOL_OK = json.dumps({"args": {"order_id": "ORD-1001"}, "result": {"code": "ok", "data": {"order": {"id": "ORD-1001"}}}})
TOOL_FAIL = json.dumps(
    {"args": {"amount_cents": 2400}, "result": {"code": "card_declined", "data": {"refund": {"id": "REF-1014"}}}}
)
RENDERED_REPLY = "ORD-1037 is paid; the latest shipment is TRK1037 and it was returned."

# The 164 free-text records as written by the build with the free-text contract
# in place, one `json.dumps(sort_keys=True, ensure_ascii=False)` line each, joined
# by "\n" and closed with "\n". Supersedes b009947d…'s build digest
# `9519a8683d87562186de773ea1ca1d027329f45e784b477bf0daabff95687479`, which was
# taken before the free-text turn named its output shape: that prompt carried no
# shape, so a base-model miss on it measured format, not quality.
GOLDEN_FREE_TEXT_SHA256 = "0bb6d13347b42662287f105e2497640ca16ad3571d58894a12969c4ac077d85b"

TOKENIZER = None


def system_body(text: str) -> str:
    """The system turn's content as the chat template rendered it."""
    head = "<|im_start|>system\n"
    assert text.startswith(head), text[:40]
    return text[len(head) : text.index("<|im_end|>")]


def setUpModule() -> None:  # noqa: N802 - unittest hook
    global TOKENIZER
    TOKENIZER = load_tokenizer()


def require_corpus() -> None:
    """Fail, never skip: a missing corpus means acceptance cannot be claimed."""
    if not (sft.DEFAULT_CORPUS / "shard-000.jsonl").is_file():
        raise AssertionError(
            f"{sft.display_path(sft.DEFAULT_CORPUS)} is absent: run python -m shopagent.cli teacher first"
        )


def free_row(
    row_id: str = "T-A-01",
    case: str = "A-01",
    ticket: str = TICKET,
    tools: tuple[str, ...] = (TOOL_OK, TOOL_FAIL),
    reply: str = REPLY,
    stamp: str = STAMP_1,
) -> dict:
    messages = [
        {"role": "system", "content": sft.TEACHER_SYSTEM_PROMPT},
        {"role": "user", "content": ticket},
        {"role": "assistant", "content": json.dumps(PLAN)},
    ]
    messages += [{"role": "tool", "content": content} for content in tools]
    messages.append({"role": "assistant", "content": reply})
    return {
        "case": case,
        "id": row_id,
        "kind": "normal",
        "messages": messages,
        "path": "free_text",
        "plan": PLAN,
        "target": {"plan": PLAN, "reply": reply, "type": "plan_and_reply"},
        "provenance": {"generated_at": stamp},
    }


def composed_row(
    row_id: str = "T-N-100",
    case: str = "N-100",
    stamp: str = STAMP_1,
    ticket: str = "Where is order ORD-1037?",
) -> dict:
    return {
        "case": case,
        "id": row_id,
        "kind": "normal",
        "messages": [
            {"role": "system", "content": sft.TEACHER_SYSTEM_PROMPT},
            {"role": "user", "content": ticket},
            {"role": "assistant", "content": json.dumps(COMPOSED_PLAN)},
        ],
        "path": "composed",
        "plan": COMPOSED_PLAN,
        "rendered_reply": RENDERED_REPLY,
        "target": {"plan": COMPOSED_PLAN, "type": "plan"},
        "provenance": {"generated_at": stamp},
    }


def write_corpus(
    root: Path,
    rows: list[dict],
    revisions: dict[str, int] | None = None,
    holdout: list[dict] | None = None,
) -> tuple[Path, Path, Path]:
    """A synthetic corpus, revision manifest and holdout directory under `root`."""
    corpus = root / "train"
    corpus.mkdir(parents=True, exist_ok=True)
    payload = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    (corpus / "shard-000.jsonl").write_bytes(payload.encode("utf-8"))

    revision_manifest = root / "manifest.json"
    revision_manifest.write_text(
        json.dumps({"runs": [{"generated_at": stamp, "brief_revision": rev} for stamp, rev in (revisions or {STAMP_1: 1}).items()]}),
        encoding="utf-8",
    )

    bench = root / "holdout"
    bench.mkdir(parents=True, exist_ok=True)
    rows_out = holdout if holdout is not None else [{"id": "HR-01", "case": "A-01", "reply": "a holdout reply"}]
    (bench / "holdout-01.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows_out) + "\n", encoding="utf-8"
    )
    return corpus, revision_manifest, bench


def build_synthetic(
    root: Path, rows: list[dict], revisions: dict[str, int] | None = None, rebalance: bool = True
) -> dict:
    corpus, revision_manifest, bench = write_corpus(root, rows, revisions=revisions)
    return build(
        corpus_dir=corpus,
        out_dir=root / "sft",
        revision_manifest=revision_manifest,
        bench_dir=bench,
        tokenizer=TOKENIZER,
        rebalance=rebalance,
    )


def kept_text(record: dict) -> list[str]:
    decode = TOKENIZER.decode
    segments: list[str] = []
    labels = record["labels"]
    ids = record["input_ids"]
    index = 0
    while index < len(labels):
        if labels[index] == IGNORE_INDEX:
            index += 1
            continue
        end = index
        while end < len(labels) and labels[end] != IGNORE_INDEX:
            end += 1
        segments.append(decode(ids[index:end]))
        index = end
    return segments


def covered_tokens(record: dict, needle: str) -> list[int]:
    """Token indices that overlap one occurrence of `needle` in the rendered text."""
    text = record["text"]
    encoded = TOKENIZER(text, add_special_tokens=False, return_offsets_mapping=True)
    start = text.index(needle)
    end = start + len(needle)
    return [
        index
        for index, (left, right) in enumerate(encoded["offset_mapping"])
        if right > start and left < end
    ]


class Conversion(unittest.TestCase):
    def test_mask_excludes_ticket_and_every_tool_result(self) -> None:
        record = convert_row(TOKENIZER, free_row(), "shard-000.jsonl", 1)
        self.assertEqual(len(record["labels"]), len(record["input_ids"]))
        segments = kept_text(record)
        self.assertEqual(segments, [json.dumps(PLAN), REPLY])

        self.assertEqual(record["text"].count(TICKET), 1)
        for content in (TICKET, TOOL_OK, TOOL_FAIL):
            covered = covered_tokens(record, content)
            self.assertTrue(covered, content[:20])
            self.assertTrue(
                all(record["labels"][index] == IGNORE_INDEX for index in covered),
                f"a token of {content[:20]!r} is supervised",
            )
        plan_covered = covered_tokens(record, json.dumps(PLAN))
        self.assertTrue(any(record["labels"][index] != IGNORE_INDEX for index in plan_covered))
        reply_covered = covered_tokens(record, REPLY)
        self.assertTrue(all(record["labels"][index] != IGNORE_INDEX for index in reply_covered))

    def test_plan_repeated_in_ticket_and_tool_stays_masked(self) -> None:
        plan_text = json.dumps(PLAN)
        row = free_row(
            ticket=f"Here is the plan you gave me: {plan_text} - please run it again.",
            tools=(TOOL_OK, json.dumps({"echo": REPLY, "result": {"code": "ok"}})),
        )
        record = convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        self.assertEqual(record["text"].count(plan_text), 2)
        self.assertEqual(record["text"].count(REPLY), 2)

        _, spans = supervised_spans(record["text"], row["messages"], TOKENIZER)
        kept = sum(end - start for start, end in spans)
        supervised = sum(1 for label in record["labels"] if label != IGNORE_INDEX)
        self.assertEqual(supervised, kept)
        self.assertEqual(kept_text(record), [plan_text, REPLY])

        ticket_covered = covered_tokens(record, plan_text)
        kept_indices = [index for index, label in enumerate(record["labels"]) if label != IGNORE_INDEX]
        self.assertTrue(ticket_covered)
        self.assertFalse(set(ticket_covered) & set(kept_indices))
        self.assertEqual(len(kept_indices), sum(end - start for start, end in spans))
        tool_copy = covered_tokens(record, REPLY)
        self.assertFalse(set(tool_copy) & set(kept_indices))

    def test_composed_row_carries_no_reply_text(self) -> None:
        row = composed_row()
        record = convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        for key in ("reply", "rendered_reply", "ticket", "plan"):
            self.assertNotIn(key, record)
        self.assertEqual(set(record), set(sft.RECORD_KEYS) - {"split"})
        self.assertEqual(record["path"], "composed")
        self.assertNotIn(RENDERED_REPLY, record["text"])
        segments = kept_text(record)
        self.assertEqual(len(segments), 1)
        self.assertEqual(json.loads(segments[0]), COMPOSED_PLAN)
        self.assertIsInstance(json.loads(segments[0]), dict)

    def test_both_plan_shapes_round_trip(self) -> None:
        free = convert_row(TOKENIZER, free_row(), "shard-000.jsonl", 1)
        composed = convert_row(TOKENIZER, composed_row(), "shard-000.jsonl", 1)
        free_segments = kept_text(free)
        self.assertEqual(len(free_segments), 2)
        self.assertIsInstance(json.loads(free_segments[0]), list)
        self.assertEqual(json.loads(free_segments[0]), PLAN)
        self.assertEqual(free_segments[1], REPLY)
        composed_segments = kept_text(composed)
        self.assertIsInstance(json.loads(composed_segments[0]), dict)
        self.assertEqual(json.loads(composed_segments[0]), COMPOSED_PLAN)


class ModePrompts(unittest.TestCase):
    def test_composed_rows_carry_the_composed_prompt(self) -> None:
        row = composed_row()
        record = convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        self.assertEqual(system_body(record["text"]), sft.COMPOSED_SYSTEM_PROMPT)
        self.assertIn("composed mode", system_body(record["text"]))
        self.assertIn("never write a customer reply", system_body(record["text"]))
        self.assertNotEqual(system_body(record["text"]), sft.FREE_TEXT_SYSTEM_PROMPT.strip())
        self.assertEqual(row["messages"][0]["content"], sft.TEACHER_SYSTEM_PROMPT)

    def test_free_text_rows_carry_the_teacher_prompt_plus_the_contract(self) -> None:
        record = convert_row(TOKENIZER, free_row(), "shard-000.jsonl", 1)
        body = system_body(record["text"])
        self.assertTrue(body.startswith(sft.TEACHER_SYSTEM_PROMPT.strip()), body[:120])
        self.assertTrue(body.endswith(sft.FREE_TEXT_CONTRACT), body[-160:])
        self.assertEqual(body, sft.FREE_TEXT_SYSTEM_PROMPT.strip())
        self.assertNotIn("composed mode", record["text"])

    def test_composed_target_keeps_the_action_object(self) -> None:
        from shopagent.compose import PLAN_KEYS

        record = convert_row(TOKENIZER, composed_row(), "shard-000.jsonl", 1)
        plan = json.loads(kept_text(record)[0])
        self.assertEqual(plan, COMPOSED_PLAN)
        self.assertIsInstance(plan, dict)
        self.assertTrue(set(plan) <= PLAN_KEYS[plan["action"]])

    def test_every_action_appears_with_allowed_keys_only(self) -> None:
        from shopagent.compose import PLAN_KEYS

        plans = [
            {"action": "status", "order_id": "ORD-1037"},
            {"action": "refund", "order_id": "ORD-1037", "amount_cents": 2100},
            {"action": "reshipment", "order_id": "ORD-1037"},
            {"action": "reshipment", "order_id": "ORD-1037", "shipment_id": "SHP-1037"},
            {"action": "address", "address_id": "ADR-1001"},
        ]
        for plan in plans:
            row = composed_row(row_id=f"T-X-{plan['action']}", case="X-1")
            row["messages"][2]["content"] = json.dumps(plan)
            row["plan"] = plan
            row["target"] = {"plan": plan, "type": "plan"}
            record = convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
            kept = json.loads(kept_text(record)[0])
            self.assertEqual(kept, plan)
            self.assertTrue(set(kept) <= PLAN_KEYS[plan["action"]], plan)

    def test_prompt_already_marked_is_refused(self) -> None:
        row = composed_row()
        row["messages"][0]["content"] = sft.COMPOSED_SYSTEM_PROMPT
        with self.assertRaises(SftError) as caught:
            convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        self.assertEqual(caught.exception.reason, "system_prompt_drift")
        self.assertIn("T-N-100", str(caught.exception))

    def test_prompt_digests_are_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_synthetic(Path(temp), [free_row(), composed_row()])
        prompts = manifest["prompts"]
        self.assertEqual(prompts["free_text"]["text"], sft.FREE_TEXT_SYSTEM_PROMPT)
        self.assertEqual(prompts["composed"]["text"], sft.COMPOSED_SYSTEM_PROMPT)
        self.assertEqual(prompts["free_text"]["sha256"], hashlib.sha256(sft.FREE_TEXT_SYSTEM_PROMPT.encode()).hexdigest())
        self.assertEqual(prompts["composed"]["sha256"], hashlib.sha256(sft.COMPOSED_SYSTEM_PROMPT.encode()).hexdigest())
        self.assertNotEqual(prompts["free_text"]["sha256"], prompts["composed"]["sha256"])
        self.assertIn("teacher.SYSTEM_PROMPT", prompts["free_text"]["source"])
        superseded = prompts["free_text"]["superseded"]
        self.assertEqual(len(superseded), 1)
        self.assertEqual(
            superseded[0]["sha256"], hashlib.sha256(sft.TEACHER_SYSTEM_PROMPT.encode()).hexdigest()
        )
        self.assertIn("no output shape", superseded[0]["reason"])
        self.assertNotEqual(superseded[0]["sha256"], prompts["free_text"]["sha256"])

    def test_record_over_max_seq_len_is_refused(self) -> None:
        filler = json.dumps({"args": {"order_id": "ORD-1001"}, "result": {"data": {"blob": "stack " * 3000}}})
        row = free_row(tools=(filler,))
        with self.assertRaises(SftError) as caught:
            with tempfile.TemporaryDirectory() as temp:
                build_synthetic(Path(temp), [row])
        self.assertEqual(caught.exception.reason, "record_too_long")
        self.assertIn(str(sft.TRAIN_MAX_SEQ_LEN), str(caught.exception))

    def test_train_app_pin_matches_the_app_file(self) -> None:
        app = (sft.ROOT / "train" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn(f'DATASET_SHA256 = "{sft.TRAIN_DATASET_PIN}"', app)
        self.assertIn(f"MAX_SEQ_LEN = {sft.TRAIN_MAX_SEQ_LEN}", app)
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_synthetic(Path(temp), [free_row()])
        pin = manifest["train_app_pin"]
        self.assertEqual(pin["value"], sft.TRAIN_DATASET_PIN)
        self.assertFalse(pin["matches_output"])
        self.assertIn("retraining is blocked", pin["note"])


class Refusals(unittest.TestCase):
    def test_reply_mismatch_names_case_and_reason(self) -> None:
        row = free_row()
        row["target"]["reply"] = "A different reply."
        with self.assertRaises(SftError) as caught:
            convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        self.assertEqual(caught.exception.reason, "reply_text_mismatch")
        self.assertIn("T-A-01", str(caught.exception))

    def test_plan_mismatch_names_case_and_reason(self) -> None:
        row = free_row()
        row["messages"][2]["content"] = json.dumps(COMPOSED_PLAN)
        with self.assertRaises(SftError) as caught:
            convert_row(TOKENIZER, row, "shard-000.jsonl", 1)
        self.assertEqual(caught.exception.reason, "plan_text_mismatch")
        self.assertIn("A-01", str(caught.exception))

    def test_unknown_revision_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            corpus, revision_manifest, bench = write_corpus(root, [free_row(stamp=STAMP_2)], revisions={STAMP_1: 1})
            with self.assertRaises(SftError) as caught:
                build(
                    corpus_dir=corpus,
                    out_dir=root / "sft",
                    revision_manifest=revision_manifest,
                    bench_dir=bench,
                    tokenizer=TOKENIZER,
                )
            self.assertEqual(caught.exception.reason, "brief_revision_unknown")
            self.assertFalse((root / "sft" / sft.OUT_NAME).exists())


class Corpus(unittest.TestCase):
    def test_holdout_id_never_reaches_sft(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = [free_row(), free_row(row_id="HR-01", case="A-01")]
            manifest = build_synthetic(root, rows)
            payload = (root / "sft" / sft.OUT_NAME).read_text(encoding="utf-8")
        self.assertEqual(manifest["converted"], 1)
        self.assertEqual(len(manifest["drops"]), 1)
        drop = manifest["drops"][0]
        self.assertEqual(drop["id"], "HR-01")
        self.assertEqual(drop["reason"], "holdout_id")
        self.assertEqual(drop["case"], "A-01")
        self.assertEqual(manifest["source"]["rows"], 2)
        self.assertEqual(manifest["converted"] + len(manifest["drops"]), 2)
        self.assertNotIn("HR-01", payload)

    def test_digests_are_recomputed_from_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifest = build_synthetic(root, [free_row(), composed_row()])
            shard = root / "train" / "shard-000.jsonl"
            self.assertEqual(manifest["source"]["shards"][0]["sha256"], hashlib.sha256(shard.read_bytes()).hexdigest())
            output = root / "sft" / sft.OUT_NAME
            self.assertEqual(manifest["output"]["sha256"], hashlib.sha256(output.read_bytes()).hexdigest())
            self.assertEqual(manifest["output"]["rows"], 2)

            shard.write_bytes(shard.read_bytes() + b"\n")
            rebuilt = build(
                corpus_dir=root / "train",
                out_dir=root / "sft2",
                revision_manifest=root / "manifest.json",
                bench_dir=root / "holdout",
                tokenizer=TOKENIZER,
            )
            self.assertNotEqual(manifest["source"]["shards"][0]["sha256"], rebuilt["source"]["shards"][0]["sha256"])

    def test_build_is_deterministic(self) -> None:
        rows = [free_row(row_id=f"T-A-{index:02d}", case=f"A-{index:02d}", ticket=f"Refund question {index}.") for index in range(12)]
        with tempfile.TemporaryDirectory() as one, tempfile.TemporaryDirectory() as two:
            first = build_synthetic(Path(one), rows)
            second = build_synthetic(Path(two), rows)
            first_bytes = (Path(one) / "sft" / sft.OUT_NAME).read_bytes()
            second_bytes = (Path(two) / "sft" / sft.OUT_NAME).read_bytes()
        self.assertEqual(first_bytes, second_bytes)
        self.assertEqual(first["output"]["sha256"], second["output"]["sha256"])
        self.assertEqual(first["val"]["ids"], second["val"]["ids"])
        self.assertEqual(first["val"]["rows"], round(sft.VAL_FRACTION * len(rows)))
        self.assertEqual(first["config"]["seed"], SEED)

    def test_schema_is_compared_per_path_across_revisions(self) -> None:
        rows = [
            free_row(row_id="T-A-01", case="A-01", stamp=STAMP_1),
            free_row(row_id="T-A-02", case="A-02", stamp=STAMP_1),
            free_row(row_id="T-A-03", case="A-03", stamp=STAMP_2),
            composed_row(row_id="T-N-100", case="N-100", stamp=STAMP_1),
            composed_row(row_id="T-N-101", case="N-101", stamp=STAMP_2),
        ]
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_synthetic(Path(temp), rows, revisions={STAMP_1: 1, STAMP_2: 2})
        self.assertTrue(manifest["schema"]["identical_within_path"])
        groups = manifest["schema"]["groups"]
        self.assertEqual(set(groups), {"free_text", "composed"})
        self.assertEqual(groups["free_text"]["revision_1"]["rows"], 2)
        self.assertEqual(groups["free_text"]["revision_2"]["rows"], 1)
        self.assertEqual(groups["composed"]["revision_1"]["plan_shapes"], {"dict": 1})
        self.assertEqual(groups["free_text"]["revision_1"]["plan_shapes"], {"list": 2})
        for path in groups.values():
            shapes = {json.dumps(group["schema"], sort_keys=True) for group in path.values()}
            self.assertEqual(len(shapes), 1)
        self.assertEqual(manifest["by_brief_revision"]["1"]["converted"], 3)
        self.assertEqual(manifest["by_brief_revision"]["2"]["converted"], 2)
        self.assertEqual(manifest["by_brief_revision"]["1"]["free_text"], 2)
        self.assertEqual(manifest["by_brief_revision"]["2"]["composed"], 1)


class Manifest(unittest.TestCase):
    def test_manifest_contract(self) -> None:
        rows = [free_row(row_id=f"T-A-{index:02d}", case=f"A-{index:02d}") for index in range(10)]
        with tempfile.TemporaryDirectory() as temp:
            manifest = build_synthetic(Path(temp), rows)
        tokenizer = manifest["tokenizer"]
        self.assertEqual(tokenizer["id"], TOKENIZER_ID)
        self.assertEqual(tokenizer["revision"], TOKENIZER_REVISION)
        self.assertEqual(tokenizer["class"], "Qwen3_5Tokenizer")
        self.assertFalse(tokenizer["enable_thinking"])
        self.assertEqual(tokenizer["tokenizer_json_sha256"], "06b9509352d2af50381ab2247e083b80d32d5c0aba91c272ca9ff729b6a0e523")
        self.assertEqual(tokenizer["environment"]["transformers"], "5.19.0")

        config = manifest["config"]
        self.assertEqual(config["seed"], SEED)
        self.assertEqual(config["val_fraction"], sft.VAL_FRACTION)
        self.assertEqual(config["ignore_index"], IGNORE_INDEX)
        self.assertFalse(config["add_generation_prompt"])
        self.assertEqual(config["mask_rule"], manifest["mask_rule"])
        self.assertTrue(manifest["mask_rule"].endswith("."))
        self.assertEqual(manifest["mask_rule"].count(". "), 0)

        lengths = manifest["token_lengths"]
        self.assertLessEqual(lengths["min"], lengths["median"])
        self.assertLessEqual(lengths["median"], lengths["max"])
        self.assertEqual(lengths["histogram"]["bin_width"], 256)
        self.assertEqual(sum(lengths["histogram"]["bins"].values()), manifest["output"]["rows"])
        self.assertGreater(manifest["kept_tokens"]["min"], 0)
        self.assertLessEqual(manifest["kept_tokens"]["fraction_of_all"], 1)

        self.assertIn("overfit watch only", manifest["val"]["label"])
        self.assertTrue(any("EOS" in item for item in manifest["open_items"]))
        self.assertEqual(Path(manifest["output"]["dir"]).name, "sft")
        self.assertEqual(manifest["output"]["file"], sft.OUT_NAME)
        self.assertEqual(manifest["converted"], len(rows))
        self.assertEqual(manifest["drops"], [])

    def test_budget_constants_match_the_app_file(self) -> None:
        app = (sft.ROOT / "train" / "modal_app.py").read_text(encoding="utf-8")
        self.assertIn(f"GRAD_ACCUM = {sft.TRAIN_GRAD_ACCUM}", app)
        self.assertIn(f"EPOCHS = {sft.TRAIN_EPOCHS}", app)
        self.assertIn(f"STEP_CAP = {sft.TRAIN_STEP_CAP}", app)

    def test_holdout_ids_loads_every_holdout_row(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            bench = Path(temp)
            (bench / "a.jsonl").write_text('{"id": "HR-01", "case": "A-01", "reply": "x"}\n', encoding="utf-8")
            (bench / "b.jsonl").write_text('{"id": "HR2-01", "case": "A-02", "reply": "y"}\n', encoding="utf-8")
            self.assertEqual(holdout_ids(bench), frozenset({"HR-01", "HR2-01"}))
            empty = bench / "empty"
            empty.mkdir()
            with self.assertRaises(SftError) as caught:
                holdout_ids(empty)
            self.assertEqual(caught.exception.reason, "holdout_missing")


class Rebalance(unittest.TestCase):
    """The composed train slice: 40 % of the epoch, repeats in train only."""

    @classmethod
    def corpus_rows(cls) -> list[dict]:
        rows = [
            free_row(row_id=f"T-A-{index:02d}", case=f"A-{index:02d}", ticket=f"Refund question {index}.")
            for index in range(20)
        ]
        rows += [
            composed_row(
                row_id=f"T-N-{index:02d}",
                case=f"N-{index:02d}",
                ticket=f"Where is order ORD-{1037 + index}?",
            )
            for index in range(4)
        ]
        return rows

    @classmethod
    def setUpClass(cls) -> None:
        cls._temp = tempfile.TemporaryDirectory()
        root = Path(cls._temp.name)
        rows = cls.corpus_rows()
        cls.base = build_synthetic(root / "base", rows, rebalance=False)
        cls.balanced = build_synthetic(root / "balanced", rows)
        cls.base_lines = (root / "base" / "sft" / sft.OUT_NAME).read_text(encoding="utf-8").splitlines()
        cls.balanced_lines = (root / "balanced" / "sft" / sft.OUT_NAME).read_text(encoding="utf-8").splitlines()
        cls.base_rows = [json.loads(line) for line in cls.base_lines]
        cls.balanced_rows = [json.loads(line) for line in cls.balanced_lines]

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_target_arithmetic(self) -> None:
        self.assertEqual(sft.COMPOSED_TRAIN_SHARE, 0.4)
        self.assertEqual(sft.target_composed_train(147), 98)
        self.assertEqual(sft.target_composed_train(150), 100)

    def test_base_artifact_has_no_repeats(self) -> None:
        self.assertIsNone(self.base["rebalance"])
        self.assertEqual(self.base["output"]["rows"], self.base["converted"])
        self.assertEqual(len(self.base_rows), len({row["id"] for row in self.base_rows}))

    def test_val_split_and_bytes_are_unchanged(self) -> None:
        self.assertEqual(self.base["val"]["ids"], self.balanced["val"]["ids"])
        self.assertEqual(self.balanced["val"]["rows"], 2)
        base_lines = {row["id"]: line for row, line in zip(self.base_rows, self.base_lines)}
        val = [(row, line) for row, line in zip(self.balanced_rows, self.balanced_lines) if row["split"] == "val"]
        self.assertEqual(len(val), 2)
        for row, line in val:
            self.assertEqual(line, base_lines[row["id"]])
        self.assertEqual(self.balanced["rebalance"]["val"]["rows"], 2)
        self.assertFalse(self.balanced["rebalance"]["val"]["changed"])

    def test_repeats_are_train_composed_rows_only(self) -> None:
        counts: dict[str, int] = {}
        for row in self.balanced_rows:
            counts[row["id"]] = counts.get(row["id"], 0) + 1
        self.assertEqual(set(counts), {row["id"] for row in self.base_rows})
        for row in self.balanced_rows:
            if counts[row["id"]] > 1:
                self.assertEqual(row["path"], "composed")
                self.assertEqual(row["split"], "train")
        for row_id in self.balanced["val"]["ids"]:
            self.assertEqual(counts[row_id], 1)

    def test_every_repeat_is_byte_identical(self) -> None:
        lines: dict[str, list[str]] = {}
        for row, line in zip(self.balanced_rows, self.balanced_lines):
            lines.setdefault(row["id"], []).append(line)
        repeated = {row_id: found for row_id, found in lines.items() if len(found) > 1}
        self.assertTrue(repeated)
        for found in repeated.values():
            self.assertEqual(len(set(found)), 1)
        self.assertGreater(self.balanced["token_delta"]["supervised"]["added"], 0)

    def test_no_two_neighbours_share_an_id(self) -> None:
        ids = [row["id"] for row in self.balanced_rows]
        copies = self.balanced["rebalance"]["derivation"]["copies_added"]
        self.assertEqual(len(ids), len(set(ids)) + copies)
        for left, right in zip(ids, ids[1:]):
            self.assertNotEqual(left, right)

    def test_rebalance_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            again = build_synthetic(Path(temp), self.corpus_rows())
            again_bytes = (Path(temp) / "sft" / sft.OUT_NAME).read_bytes()
        balanced_bytes = (Path(self._temp.name) / "balanced" / "sft" / sft.OUT_NAME).read_bytes()
        self.assertEqual(again_bytes, balanced_bytes)
        self.assertEqual(again["rebalance"], self.balanced["rebalance"])

    def test_derivation_and_budget_arithmetic(self) -> None:
        rebalance = self.balanced["rebalance"]
        derivation = rebalance["derivation"]
        before = rebalance["epoch_composition"]["before"]
        after = rebalance["epoch_composition"]["after"]
        self.assertEqual(derivation["rounded"], sft.target_composed_train(derivation["free_text_train"]))
        self.assertEqual(derivation["copies_added"], derivation["rounded"] - derivation["base_composed_train"])
        self.assertEqual(
            derivation["copies_added"],
            derivation["copies_per_row"] * derivation["base_composed_train"] + derivation["remainder"],
        )
        self.assertEqual(
            sum((int(count) - 1) * rows for count, rows in derivation["counts"].items()),
            derivation["copies_added"],
        )
        self.assertEqual(before["composed"], derivation["base_composed_train"])
        self.assertEqual(after["composed"], derivation["rounded"])
        self.assertEqual(before["train_rows"], before["free_text"] + before["composed"])
        self.assertEqual(after["train_rows"], after["free_text"] + after["composed"])
        self.assertAlmostEqual(after["composed_share"], after["composed"] / after["train_rows"], places=4)
        budget = self.balanced["budget"]
        self.assertEqual(budget["train_rows"], after["train_rows"])
        self.assertEqual(budget["steps_per_epoch"], -(-budget["train_rows"] // budget["grad_accum"]))
        self.assertEqual(budget["total_steps"], budget["steps_per_epoch"] * budget["epochs"])
        self.assertEqual(budget["fits_cap"], budget["total_steps"] <= budget["step_cap"])
        delta = self.balanced["token_delta"]
        self.assertEqual(delta["supervised"]["after"] - delta["supervised"]["before"], delta["supervised"]["added"])
        self.assertEqual(delta["rendered"]["after"] - delta["rendered"]["before"], delta["rendered"]["added"])


class RealCorpus(unittest.TestCase):
    """The full corpus: count, drop accounting, and the supervised total."""

    @classmethod
    def setUpClass(cls) -> None:
        require_corpus()
        cls._temp = tempfile.TemporaryDirectory()
        cls.manifest = build(
            corpus_dir=sft.DEFAULT_CORPUS,
            out_dir=Path(cls._temp.name) / "sft",
            revision_manifest=sft.DEFAULT_REVISION_MANIFEST,
            bench_dir=sft.DEFAULT_BENCH,
            tokenizer=TOKENIZER,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        cls._temp.cleanup()

    def test_converted_count_is_178_or_every_drop_is_named(self) -> None:
        manifest = self.manifest
        self.assertEqual(manifest["source"]["rows"], 178)
        self.assertEqual(manifest["converted"] + len(manifest["drops"]), 178)
        if manifest["converted"] != 178:
            self.assertTrue(manifest["drops"])
            for drop in manifest["drops"]:
                self.assertTrue(drop["case"])
                self.assertEqual(drop["reason"], "holdout_id")
        self.assertEqual(manifest["composition"]["free_text"] + manifest["composition"]["composed"], manifest["converted"])
        self.assertEqual(manifest["val"]["rows"], round(sft.VAL_FRACTION * manifest["converted"]))
        for shard in manifest["source"]["shards"]:
            path = sft.ROOT / shard["path"]
            self.assertEqual(shard["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertLessEqual(manifest["kept_tokens"]["min"], manifest["kept_tokens"]["median"])
        self.assertLessEqual(manifest["kept_tokens"]["median"], manifest["kept_tokens"]["max"])
        self.assertGreater(manifest["kept_tokens"]["fraction_of_all"], 0)

    def test_supervised_total_is_reported(self) -> None:
        manifest = self.manifest
        rows = [json.loads(line) for line in (Path(self._temp.name) / "sft" / sft.OUT_NAME).read_text(encoding="utf-8").splitlines() if line]
        total = sum(sum(1 for label in row["labels"] if label != IGNORE_INDEX) for row in rows)
        self.assertEqual(total, manifest["kept_tokens"]["total"])
        self.assertEqual(sum(len(row["input_ids"]) for row in rows), manifest["token_lengths"]["total"])
        self.assertLess(manifest["kept_tokens"]["total"], manifest["token_lengths"]["total"])
        self.assertEqual(len(rows), manifest["output"]["rows"])
        self.assertEqual(len(rows), manifest["converted"] + manifest["rebalance"]["derivation"]["copies_added"])

    def test_free_text_records_hash_to_the_pre_fix_digest(self) -> None:
        """The mode marker moved composed rows only; free-text bytes are pinned.

        `GOLDEN_FREE_TEXT_SHA256` was taken from the corpus as written before the
        composed prompt existed, in source order. A build with the rebalance off
        keeps that order, so a single byte moved on a free-text row — a prompt
        change, a mask change, a re-render — changes this hash. Without this
        test, "the fix touched only composed rows" is an assertion; with it, it
        is a measurement.
        """
        with tempfile.TemporaryDirectory() as temp:
            manifest = build(
                corpus_dir=sft.DEFAULT_CORPUS,
                out_dir=Path(temp) / "sft",
                revision_manifest=sft.DEFAULT_REVISION_MANIFEST,
                bench_dir=sft.DEFAULT_BENCH,
                tokenizer=TOKENIZER,
                rebalance=False,
            )
            payload = (Path(temp) / "sft" / sft.OUT_NAME).read_text(encoding="utf-8")
        rows = [json.loads(line) for line in payload.splitlines() if line]
        free_text = [row for row in rows if row["path"] == "free_text"]
        self.assertEqual(len(rows), manifest["output"]["rows"])
        self.assertEqual(len(free_text), 164)
        digest = hashlib.sha256(
            ("\n".join(json.dumps(row, sort_keys=True, ensure_ascii=False) for row in free_text) + "\n").encode("utf-8")
        ).hexdigest()
        self.assertEqual(digest, GOLDEN_FREE_TEXT_SHA256)

    def test_rebalance_hits_the_decided_numbers(self) -> None:
        manifest = self.manifest
        rebalance = manifest["rebalance"]
        derivation = rebalance["derivation"]
        before = rebalance["epoch_composition"]["before"]
        after = rebalance["epoch_composition"]["after"]
        self.assertEqual(derivation["free_text_train"], 147)
        self.assertEqual(derivation["base_composed_train"], 13)
        self.assertEqual(derivation["rounded"], 98)
        self.assertEqual(derivation["copies_added"], 85)
        self.assertEqual(derivation["copies_per_row"], 6)
        self.assertEqual(derivation["remainder"], 7)
        self.assertEqual(len(derivation["extra_copy_ids"]), 7)
        self.assertEqual(derivation["counts"], {"7": 6, "8": 7})
        self.assertEqual(derivation["repeat_factor"], 7.5385)
        self.assertEqual(before["composed"], 13)
        self.assertEqual(before["train_rows"], 160)
        self.assertEqual(after["train_rows"], 245)
        self.assertEqual(after["composed"], 98)
        self.assertEqual(after["composed_share"], 0.4)
        self.assertEqual(manifest["output"]["rows"], 263)
        self.assertEqual(manifest["output"]["source_records"], 178)
        self.assertEqual(manifest["output"]["rows"], manifest["converted"] + derivation["copies_added"])
        self.assertEqual(rebalance["val"]["rows"], 18)
        self.assertFalse(rebalance["val"]["changed"])
        budget = manifest["budget"]
        self.assertEqual(budget["train_rows"], 245)
        self.assertEqual(budget["steps_per_epoch"], 62)
        self.assertEqual(budget["total_steps"], 186)
        self.assertTrue(budget["fits_cap"])

    def test_real_repeats_are_train_only_and_neighbours_differ(self) -> None:
        rows = [
            json.loads(line)
            for line in (Path(self._temp.name) / "sft" / sft.OUT_NAME).read_text(encoding="utf-8").splitlines()
            if line
        ]
        counts: dict[str, int] = {}
        for row in rows:
            counts[row["id"]] = counts.get(row["id"], 0) + 1
        repeated = [row for row in rows if counts[row["id"]] > 1]
        self.assertTrue(repeated)
        for row in repeated:
            self.assertEqual(row["path"], "composed")
            self.assertEqual(row["split"], "train")
        train_composed = {row["id"] for row in rows if row["split"] == "train" and row["path"] == "composed"}
        self.assertEqual(len(train_composed), 13)
        self.assertEqual(sum(counts[row_id] for row_id in train_composed), 98)
        for row_id in self.manifest["val"]["ids"]:
            self.assertEqual(counts[row_id], 1)
        ids = [row["id"] for row in rows]
        for left, right in zip(ids, ids[1:]):
            self.assertNotEqual(left, right)


if __name__ == "__main__":
    unittest.main()
