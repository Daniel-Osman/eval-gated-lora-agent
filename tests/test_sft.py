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

TOKENIZER = None


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
        {"role": "system", "content": "You are the support model for a small online shop."},
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


def composed_row(row_id: str = "T-N-100", case: str = "N-100", stamp: str = STAMP_1) -> dict:
    return {
        "case": case,
        "id": row_id,
        "kind": "normal",
        "messages": [
            {"role": "system", "content": "You are the support model for a small online shop."},
            {"role": "user", "content": "Where is order ORD-1037?"},
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


def build_synthetic(root: Path, rows: list[dict], revisions: dict[str, int] | None = None) -> dict:
    corpus, revision_manifest, bench = write_corpus(root, rows, revisions=revisions)
    return build(
        corpus_dir=corpus,
        out_dir=root / "sft",
        revision_manifest=revision_manifest,
        bench_dir=bench,
        tokenizer=TOKENIZER,
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
        self.assertEqual(sum(lengths["histogram"]["bins"].values()), manifest["converted"])
        self.assertGreater(manifest["kept_tokens"]["min"], 0)
        self.assertLessEqual(manifest["kept_tokens"]["fraction_of_all"], 1)

        self.assertIn("overfit watch only", manifest["val"]["label"])
        self.assertTrue(any("EOS" in item for item in manifest["open_items"]))
        self.assertEqual(Path(manifest["output"]["dir"]).name, "sft")
        self.assertEqual(manifest["output"]["file"], sft.OUT_NAME)
        self.assertEqual(manifest["converted"], len(rows))
        self.assertEqual(manifest["drops"], [])

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
        self.assertEqual(len(rows), manifest["converted"])


if __name__ == "__main__":
    unittest.main()
