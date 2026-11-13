"""Run, store and manifest: shards, digests, rejects, leak screen, CLI."""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from shopagent.bench import load_all_cases
from shopagent.cli import build_parser, cmd_teacher
from shopagent.corpus import generate, reindex_manifest, rejected_case_ids, run_generation
from shopagent.teacher import BRIEF_REVISION, GATEWAY, TeacherReply


def stage1(ticket, plan):
    return json.dumps({"ticket": ticket, "plan": plan})


def stage2(reply):
    return json.dumps({"reply": reply})


GET_1001 = [{"name": "get_order", "args": {"order_id": "ORD-1001"}}]


def synthetic(case_id: str, **overrides) -> dict:
    case = {
        "id": case_id,
        "kind": "normal",
        "title": "Order status",
        "ticket": f"benchmark fixture text for {case_id}",
        "faults": [],
        "required_calls": [{"name": "get_order", "args": {"order_id": "ORD-1001"}}],
        "forbidden_calls": [],
        "state_expect": [
            {"table": "orders", "match": {"id": "ORD-1001"}, "fields": {"status": "paid"}}
        ],
        "required_facts": ["ORD-1001"],
        "forbidden_phrases": [],
    }
    case.update(overrides)
    return case


class KeyedClient:
    """Per-case canned replies; the stage is found from the prompt itself."""

    model = "deepseek-v4.1-flash"
    base_url = GATEWAY
    session_id = "run-test-s3"
    max_concurrency = 4

    def __init__(self, replies: dict[str, list[str]]):
        self.replies = {key: list(value) for key, value in replies.items()}
        self.by_ticket: dict[str, str] = {}
        self.requests = 0

    def chat(self, system, user):
        self.requests += 1
        if "TOOL RESULTS" in user:
            ticket = user.split("CUSTOMER MESSAGE:\n", 1)[1].split("\nPLAN:", 1)[0]
            case_id = self.by_ticket[ticket]
            text = self.replies[case_id].pop(0)
        else:
            case_id = re.search(r"SCENARIO (\S+)", user).group(1)
            text = self.replies[case_id].pop(0)
            self.by_ticket[json.loads(text)["ticket"]] = case_id
        return TeacherReply(text=text, model=self.model, attempts=1, latency_ms=1, status=200)


def three_cases() -> list[dict]:
    return [
        synthetic("TX-01"),
        synthetic("TX-02", required_facts=["ORD-1001", "$24.00"]),
        synthetic("TX-03"),
    ]


def three_replies() -> dict[str, list[str]]:
    return {
        "TX-01": [stage1("Please check the status of the order with id ORD-1001 for me.", GET_1001), stage2("Order ORD-1001 is paid.")],
        "TX-02": [stage1("Could you tell me where the parcel for ORD-1001 has got to?", GET_1001), stage2("Your order is on its way.")],
        "TX-03": [
            stage1("Kindly review the order ORD-1001 and let me know what is happening.", GET_1001),
            stage2("I have forwarded your question about ORD-1001 to a manager."),
        ],
    }


class RunStoreTests(unittest.TestCase):
    def run_three(self, directory: str):
        return run_generation(
            three_cases(), KeyedClient(three_replies()), out_dir=directory, now="2026-11-13T00:00:00Z"
        )

    def test_manifest_arithmetic_and_digests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_three(directory)
            manifest = result.manifest

            self.assertEqual(manifest["attempted"], 3)
            self.assertEqual(manifest["admitted"], 1)
            self.assertEqual(manifest["rejected"], 2)
            self.assertEqual(manifest["admitted"] + manifest["rejected"], manifest["attempted"])
            self.assertEqual(sum(shard["rows"] for shard in manifest["shards"]), manifest["admitted"])
            self.assertEqual(manifest["session_id"], "run-test-s3")
            self.assertEqual(manifest["model"], "deepseek-v4.1-flash")
            self.assertEqual(manifest["gateway"], GATEWAY)
            self.assertRegex(manifest["template_digest"], r"^[0-9a-f]{64}$")
            self.assertRegex(manifest["cases_digest"], r"^[0-9a-f]{64}$")
            self.assertEqual(manifest["requests"], 6)

            for shard in manifest["shards"]:
                path = Path(directory) / shard["file"]
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                self.assertEqual(digest, shard["sha256"])
                self.assertEqual(len(path.read_text(encoding="utf-8").strip().splitlines()), shard["rows"])

    def test_rejects_keep_their_reason_and_never_reach_a_shard(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            result = self.run_three(directory)
            rejects = [
                json.loads(line)
                for line in (Path(directory) / "rejects" / "rejected.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            reasons = {row["case"]: row["reason"] for row in rejects}
            self.assertEqual(reasons, {"TX-02": "facts_missing", "TX-03": "gate:ESCALATED"})
            self.assertEqual(result.manifest["rejects_by_reason"], {"facts_missing": 1, "gate:ESCALATED": 1})

            admitted = []
            for shard in result.manifest["shards"]:
                admitted.extend(
                    json.loads(line)
                    for line in (Path(directory) / shard["file"]).read_text(encoding="utf-8").splitlines()
                )
            self.assertEqual([row["case"] for row in admitted], ["TX-01"])
            self.assertNotIn("TX-02", [row["case"] for row in admitted])

    def test_raw_output_is_kept_verbatim(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.run_three(directory)
            raw = [
                json.loads(line)
                for line in (Path(directory) / "raw" / "raw.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(len(raw), 6)
            rejected_texts = [row["text"] for row in raw if not row["admitted"]]
            self.assertIn(stage2("Your order is on its way."), rejected_texts)

    def test_a_holdout_id_cannot_enter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "corpus"
            with self.assertRaises(ValueError):
                run_generation(
                    [synthetic("HR-01")], KeyedClient({}), out_dir=target, now="2026-11-13T00:00:00Z"
                )
            self.assertFalse((target / "manifest.json").exists())

    def test_a_bench_ticket_collision_is_dropped_and_counted(self) -> None:
        case = next(item for item in load_all_cases() if item["id"] == "N-01")
        reply = {
            "N-01": [
                stage1(case["ticket"], [{"name": "get_order", "args": {"order_id": "ORD-1001"}}]),
                stage2("Order ORD-1001 is in transit and holds the Cotton throw."),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            result = run_generation(
                [case], KeyedClient(reply), out_dir=directory, now="2026-11-13T00:00:00Z"
            )
            self.assertEqual(result.manifest["admitted"], 0)
            self.assertEqual(result.manifest["shards"], [])
            self.assertEqual(len(result.dropped), 1)
            self.assertTrue(result.dropped[0]["reason"].startswith("leak"))
            rejects = (Path(directory) / "rejects" / "rejected.jsonl").read_text(encoding="utf-8")
            self.assertIn("leak", rejects)

    def test_no_credential_is_written(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            self.run_three(directory)
            for path in Path(directory).rglob("*"):
                if path.is_file():
                    text = path.read_text(encoding="utf-8", errors="replace")
                    self.assertNotIn("Bearer", text)
                    self.assertNotIn("OPENCODE_API_KEY", text)


class ResumeTests(unittest.TestCase):
    def test_a_resumed_pass_merges_rows_and_recomputes_the_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = run_generation(
                three_cases(), KeyedClient(three_replies()), out_dir=directory, now="2026-11-13T00:00:00Z"
            )
            self.assertEqual(first.manifest["admitted"], 1)
            self.assertEqual(first.manifest["rejected"], 2)
            first_requests = first.manifest["requests"]

            fixed = {
                "TX-02": [
                    stage1("Could you tell me where the parcel for ORD-1001 has got to?", GET_1001),
                    stage2("Your ORD-1001 order is paid and has cost $24.00."),
                ],
                "TX-03": [
                    stage1("Kindly review the order ORD-1001 and let me know what is happening.", GET_1001),
                    stage2("I have forwarded your question about ORD-1001 to a manager."),
                ],
            }
            second = run_generation(
                [three_cases()[1], three_cases()[2]],
                KeyedClient(fixed),
                out_dir=directory,
                now="2026-11-13T01:00:00Z",
                resume=True,
            )
            manifest = second.manifest
            self.assertEqual(manifest["attempted"], 3)
            self.assertEqual(manifest["admitted"], 2)
            self.assertEqual(manifest["rejected"], 1)
            self.assertEqual(manifest["admitted"] + manifest["rejected"], manifest["attempted"])
            self.assertEqual(manifest["requests"], first_requests + 4)
            self.assertEqual(manifest["resumed_requests"], 4)
            self.assertEqual(len(manifest["runs"]), 2)
            self.assertEqual(manifest["generated_at"], "2026-11-13T00:00:00Z")
            self.assertEqual(manifest["updated_at"], "2026-11-13T01:00:00Z")

            admitted = []
            for shard in manifest["shards"]:
                admitted.extend(
                    json.loads(line)
                    for line in (Path(directory) / shard["file"]).read_text(encoding="utf-8").splitlines()
                )
            self.assertEqual([row["case"] for row in admitted], ["TX-01", "TX-02"])
            digest = hashlib.sha256((Path(directory) / manifest["shards"][0]["file"]).read_bytes()).hexdigest()
            self.assertEqual(digest, manifest["shards"][0]["sha256"])

            rejects = [
                json.loads(line)
                for line in (Path(directory) / "rejects" / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual([row["case"] for row in rejects], ["TX-03"])
            self.assertEqual(manifest["rejects_by_reason"], {"gate:ESCALATED": 1})

    def test_a_resume_with_an_empty_ledger_keeps_every_row_and_exits_clean(self) -> None:
        """A complete corpus has no rejects, so `--resume` must select no case."""

        class Disturbed:
            model = "deepseek-v4.1-flash"
            base_url = GATEWAY
            session_id = "run-test-s3"
            max_concurrency = 1

            def chat(self, system, user):
                raise AssertionError("a resume with an empty ledger called the teacher")

        with tempfile.TemporaryDirectory() as directory:
            run_generation(
                [synthetic("TX-01")],
                KeyedClient(
                    {
                        "TX-01": [
                            stage1(
                                "Please check the status of the order with id ORD-1001 for me.",
                                GET_1001,
                            ),
                            stage2("Order ORD-1001 is paid."),
                        ]
                    }
                ),
                out_dir=directory,
                now="2026-11-13T00:00:00Z",
            )
            self.assertEqual(rejected_case_ids(directory), [])
            shards = sorted(Path(directory).glob("train/shard-*.jsonl"))
            before = [path.read_bytes() for path in shards]
            self.assertEqual(len(before), 1)

            args = build_parser().parse_args(
                ["teacher", "--resume", "--out", directory, "--session-id", "s"]
            )
            with mock.patch("shopagent.teacher.TeacherClient", lambda **kwargs: Disturbed()):
                code = cmd_teacher(args)

            self.assertEqual(code, 0)
            after = [path.read_bytes() for path in sorted(Path(directory).glob("train/shard-*.jsonl"))]
            self.assertEqual(after, before)

    def test_resume_without_a_manifest_refuses(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):
                run_generation(
                    three_cases()[:1], KeyedClient({}), out_dir=directory, resume=True
                )

    def test_rejected_case_ids_reads_the_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_generation(three_cases(), KeyedClient(three_replies()), out_dir=directory)
            self.assertEqual(rejected_case_ids(directory), ["TX-02", "TX-03"])


class ReindexTests(unittest.TestCase):
    def test_reindex_rebuilds_the_derived_fields_from_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            first = run_generation(
                three_cases(), KeyedClient(three_replies()), out_dir=directory, now="2026-11-13T00:00:00Z"
            )
            manifest_path = Path(directory) / "manifest.json"
            spoiled = json.loads(manifest_path.read_text(encoding="utf-8"))
            spoiled["admitted"] = 99
            spoiled["dropped_leaks"] = [{"case": "X", "reason": "leak:X"}]
            spoiled["shards"] = []
            manifest_path.write_text(json.dumps(spoiled), encoding="utf-8")

            fixed = reindex_manifest(directory)
            self.assertEqual(fixed["admitted"], 1)
            self.assertEqual(fixed["rejected"], 2)
            self.assertEqual(fixed["attempted"], 3)
            self.assertEqual(fixed["dropped_leaks"], [])
            self.assertEqual(fixed["shards"], first.manifest["shards"])
            self.assertEqual(fixed["brief_revision"], BRIEF_REVISION)
            on_disk = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(on_disk["dropped_leaks"], [])

    def test_reindex_derives_a_leak_drop_from_the_reject_ledger(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            run_generation(three_cases(), KeyedClient(three_replies()), out_dir=directory)
            rejects_path = Path(directory) / "rejects" / "rejected.jsonl"
            rows = [json.loads(line) for line in rejects_path.read_text(encoding="utf-8").splitlines()]
            rows[0]["reason"] = "leak_near:N-24"
            rejects_path.write_text(
                "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8"
            )

            fixed = reindex_manifest(directory)
            self.assertEqual(fixed["dropped_leaks"], [{"case": "TX-02", "reason": "leak_near:N-24"}])
            self.assertEqual(fixed["rejects_by_reason"]["leak_near:N-24"], 1)

    def test_a_resumed_pass_keeps_the_leak_drop(self) -> None:
        case = next(item for item in load_all_cases() if item["id"] == "N-01")
        replies = {
            "N-01": [
                stage1(case["ticket"], [{"name": "get_order", "args": {"order_id": "ORD-1001"}}]),
                stage2("Order ORD-1001 is in transit and holds the Cotton throw."),
            ]
        }
        with tempfile.TemporaryDirectory() as directory:
            first = run_generation([case], KeyedClient(replies), out_dir=directory)
            self.assertEqual(len(first.manifest["dropped_leaks"]), 1)
            drop = first.manifest["dropped_leaks"][0]
            self.assertEqual(drop["case"], "N-01")
            self.assertTrue(drop["reason"].startswith("leak"))

            second = run_generation(
                [synthetic("TX-01")],
                KeyedClient(
                    {
                        "TX-01": [
                            stage1("Please check the status of the order with id ORD-1001 for me.", GET_1001),
                            stage2("Order ORD-1001 is paid."),
                        ]
                    }
                ),
                out_dir=directory,
                now="2026-11-13T01:00:00Z",
                resume=True,
            )
            self.assertEqual(second.manifest["dropped_leaks"], first.manifest["dropped_leaks"])
            self.assertEqual(second.manifest["attempted"], 2)
            self.assertEqual(second.manifest["admitted"], 1)
            self.assertEqual(second.manifest["rejected"], 1)

    def test_a_resume_still_screens_against_the_whole_bank(self) -> None:
        case = next(item for item in load_all_cases() if item["id"] == "N-01")
        with tempfile.TemporaryDirectory() as directory:
            run_generation(three_cases(), KeyedClient(three_replies()), out_dir=directory)
            result = run_generation(
                [synthetic("TX-09")],
                KeyedClient(
                    {
                        "TX-09": [
                            stage1(case["ticket"], GET_1001),
                            stage2("Order ORD-1001 is paid."),
                        ]
                    }
                ),
                out_dir=directory,
                resume=True,
            )
            self.assertEqual([row["case"] for row in result.manifest["dropped_leaks"]], ["TX-09"])
            self.assertEqual(result.manifest["admitted"], 1)
            # The two old rejects survive, plus this pass's dropped row.
            self.assertEqual(result.manifest["rejected"], 3)
            self.assertEqual(result.manifest["attempted"], 4)


class CliTests(unittest.TestCase):
    def test_flags_parse(self) -> None:
        args = build_parser().parse_args(
            [
                "teacher",
                "--dry-run",
                "5",
                "--out",
                "out/x",
                "--limit",
                "3",
                "--only",
                "N-01,A-01",
                "--session-id",
                "s-1",
            ]
        )
        self.assertEqual(args.dry_run, 5)
        self.assertEqual(args.out, "out/x")
        self.assertEqual(args.limit, 3)
        self.assertEqual(args.only, "N-01,A-01")
        self.assertEqual(args.session_id, "s-1")
        self.assertFalse(args.resume)

        resumed = build_parser().parse_args(["teacher", "--resume", "--out", "data"])
        self.assertTrue(resumed.resume)

    def test_missing_key_exits_two_and_writes_nothing(self) -> None:
        args = build_parser().parse_args(["teacher", "--dry-run", "1", "--out", "out/must-not-exist"])
        with mock.patch.dict(os.environ, {}, clear=True):
            code = cmd_teacher(args)
        self.assertEqual(code, 2)
        self.assertFalse(Path("out/must-not-exist").exists())


if __name__ == "__main__":
    unittest.main()
