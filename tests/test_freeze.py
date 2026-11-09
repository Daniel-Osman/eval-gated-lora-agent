"""Freeze: a frozen benchmark must change identity when any case changes."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from shopagent.bench import ROOT
from shopagent.freeze import ORIGINAL_CASES_DIGEST, canonical_digest, verify, write


class FreezeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        cases_dir = self.root / "bench" / "cases"
        cases_dir.mkdir(parents=True)
        for name in ("normal.jsonl", "adversarial.jsonl", "paraphrase.jsonl"):
            shutil.copy(ROOT / "bench" / "cases" / name, cases_dir / name)
        holdout_dir = self.root / "bench" / "holdout"
        holdout_dir.mkdir(parents=True)
        shutil.copy(ROOT / "bench" / "holdout" / "review-01.jsonl", holdout_dir / "review-01.jsonl")

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_freeze_then_verify(self) -> None:
        record = write(self.root)
        self.assertEqual(record["count"], 67)
        self.assertEqual(record["original_count"], 40)
        self.assertEqual(record["original_cases_digest"], ORIGINAL_CASES_DIGEST)
        self.assertTrue(record["holdout"]["training_excluded"])
        self.assertEqual(record["holdout"]["rows"], 32)
        self.assertEqual(record["holdout"]["file"], "bench/holdout/review-01.jsonl")
        self.assertIn("bench/holdout/review-01.jsonl", record["files"])
        ok, problems = verify(self.root)
        self.assertTrue(ok, problems)

    def test_holdout_edit_breaks_verification(self) -> None:
        write(self.root)
        path = self.root / "bench" / "holdout" / "review-01.jsonl"
        path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        ok, problems = verify(self.root)
        self.assertFalse(ok)
        self.assertIn("bench/holdout/review-01.jsonl bytes changed", " ".join(problems))

    def test_edit_breaks_verification(self) -> None:
        write(self.root)
        path = self.root / "bench" / "cases" / "normal.jsonl"
        path.write_text(path.read_text(encoding="utf-8").replace("Where is order ORD-1001?", "Where is order ORD-1001 now?"),
                        encoding="utf-8")
        ok, problems = verify(self.root)
        self.assertFalse(ok)
        self.assertTrue(problems)
        self.assertIn("normal.jsonl bytes changed", " ".join(problems))

    def test_missing_file_fails(self) -> None:
        ok, problems = verify(self.root)
        self.assertFalse(ok)
        self.assertIn("FROZEN.json is missing", " ".join(problems))

    def test_canonical_digest_is_order_sensitive(self) -> None:
        first = [{"id": "A", "ticket": "x"}]
        second = [{"ticket": "x", "id": "A"}]
        self.assertEqual(canonical_digest(first), canonical_digest(second))
        third = [{"id": "A", "ticket": "y"}]
        self.assertNotEqual(canonical_digest(first), canonical_digest(third))


if __name__ == "__main__":
    unittest.main()
