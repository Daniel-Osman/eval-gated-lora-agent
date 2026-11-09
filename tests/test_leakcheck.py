"""Leak check: exact duplicates and near duplicates must be caught."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from shopagent.leakcheck import check, jaccard, load_corpus, normalize, shingles

CASES = [
    {"id": "N-01", "ticket": "Where is order ORD-1001?"},
    {"id": "N-02", "ticket": "Refund the remaining $54.00 for ORD-1005."},
]

HOLDOUT = [
    {"id": "HX-01", "case": "A-01", "reply": "They refunded $24.00 for ORD-1001."},
]


class LeakCheckTests(unittest.TestCase):
    def test_normalisation(self) -> None:
        self.assertEqual(normalize("Where is ORD-1001?!"), "where is ord 1001")

    def test_exact_duplicate_is_caught(self) -> None:
        report = check(CASES, [("train.jsonl:1", "Where is order ORD-1001?")])
        self.assertFalse(report["clean"])
        self.assertEqual(report["exact_matches"][0]["case_id"], "N-01")

    def test_wrapped_ticket_is_caught(self) -> None:
        # Extra words around the ticket: after normalisation the ticket is a
        # substring of the session, so this is an exact leak, not a near one.
        prompt = "Where is order ORD-1001? I need it today please."
        report = check(CASES, [("train.jsonl:1", prompt)])
        self.assertFalse(report["clean"])
        self.assertEqual(report["exact_matches"][0]["case_id"], "N-01")

    def test_unrelated_prompt_is_clean(self) -> None:
        report = check(CASES, [("train.jsonl:1", "How do I open a wholesale account in Dubai?")])
        self.assertTrue(report["clean"])
        self.assertEqual(report["near_matches"], [])

    def test_empty_corpus_is_clean(self) -> None:
        report = check(CASES, [])
        self.assertTrue(report["clean"])
        self.assertEqual(report["sessions"], 0)

    def test_jaccard_bounds(self) -> None:
        left = shingles("the quick brown fox jumps")
        right = shingles("the quick brown fox jumps")
        self.assertEqual(jaccard(left, right), 1.0)
        self.assertEqual(jaccard(shingles("one two three"), shingles("four five six")), 0.0)

    def test_corpus_loader_reads_the_whole_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            rows = [
                {"prompt": "first ticket"},
                {"messages": [{"role": "user", "content": "second ticket"}]},
                {"nothing": "here"},
            ]
            path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
            corpus = load_corpus(Path(tmp))
        self.assertEqual(len(corpus), 3)  # one entry per record, nothing skipped
        blob = "\n".join(text for _, text in corpus)
        for needle in ("first ticket", "second ticket", "here"):
            self.assertIn(needle, blob)

    def test_assistant_message_cannot_hide_a_holdout_reply(self) -> None:
        from shopagent.bench import load_holdout

        holdout = load_holdout()
        smuggled = holdout[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            record = {
                "messages": [
                    {"role": "system", "content": "You are a support agent."},
                    {"role": "user", "content": "How do I open a wholesale account in Dubai?"},
                    {"role": "assistant", "content": smuggled["reply"]},
                ]
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            corpus = load_corpus(Path(tmp))
        report = check(CASES, corpus, holdout=holdout)
        self.assertFalse(report["clean"])
        self.assertEqual(
            [match["holdout_id"] for match in report["holdout_exact_matches"]],
            [smuggled["id"]],
        )

    def test_nested_status_field_cannot_hide_a_holdout_reply(self) -> None:
        from shopagent.bench import load_holdout

        holdout = load_holdout()
        smuggled = holdout[0]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            record = {
                "messages": [{"role": "user", "content": "How do I open a wholesale account in Dubai?"}],
                "metadata": {"nested": [{"status": smuggled["reply"]}]},
            }
            path.write_text(json.dumps(record), encoding="utf-8")
            corpus = load_corpus(Path(tmp))
        report = check(CASES, corpus, holdout=holdout)
        self.assertFalse(report["clean"])
        self.assertEqual(
            [match["holdout_id"] for match in report["holdout_exact_matches"]],
            [smuggled["id"]],
        )

    def test_reply_split_across_list_tokens_is_caught(self) -> None:
        # The reviewer's payload: the reply is not one string anywhere, it is
        # five list tokens. Old parsed-key walking missed it.
        tokens = ["They", "refunded", "$24.00", "for", "ORD-1001."]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            record = {"messages": [{"role": "assistant", "content": tokens}]}
            path.write_text(json.dumps(record), encoding="utf-8")
            corpus = load_corpus(Path(tmp))
        report = check(CASES, corpus, holdout=HOLDOUT)
        self.assertFalse(report["clean"])
        self.assertEqual(
            [match["holdout_id"] for match in report["holdout_exact_matches"]], ["HX-01"]
        )

    def test_jsonl_array_line_is_caught(self) -> None:
        # A line that is itself a JSON array, not an object.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sessions.jsonl"
            line = json.dumps([{"role": "assistant", "content": HOLDOUT[0]["reply"]}])
            path.write_text(line + "\n", encoding="utf-8")
            corpus = load_corpus(Path(tmp))
        report = check(CASES, corpus, holdout=HOLDOUT)
        self.assertFalse(report["clean"])
        self.assertEqual(
            [match["holdout_id"] for match in report["holdout_exact_matches"]], ["HX-01"]
        )

    def test_json_attachment_line_is_caught(self) -> None:
        # A .json file whose payload is a list of records.
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "session.json"
            path.write_text(
                json.dumps([{"content": [{"text": HOLDOUT[0]["reply"]}]}]), encoding="utf-8"
            )
            corpus = load_corpus(Path(tmp))
        report = check(CASES, corpus, holdout=HOLDOUT)
        self.assertFalse(report["clean"])


if __name__ == "__main__":
    unittest.main()
