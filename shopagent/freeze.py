"""Freeze the benchmark by content hash.

The benchmark must be frozen before the first synthetic training session is
generated. `write` records the SHA-256 of every case file and a canonical digest
of the parsed cases. `verify` recomputes both. Any edit -- even whitespace inside
a string -- changes the canonical digest and fails verification.

Two canonical digests are kept on purpose:

* `original_cases_digest` -- the first frozen 40 cases only. It must stay
  `34833a08...` so the pre-fix checkpoint is comparable.
* `cases_digest` -- every case, original plus the paraphrase suite. This is the
  digest that changes when the paraphrase suite changes.

The original two case files are never edited; the paraphrase suite is separate.

The reviewer-authored holdout is frozen too, as its own file digest plus a
`holdout` block that records `training_excluded: true`. A change to the holdout
fails verification like a change to any case file.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from .bench import ALL_CASE_FILES, HOLDOUT_REL, ROOT

FROZEN_NAME = "FROZEN.json"
ORIGINAL_CASES_DIGEST = "34833a08bbe4ff761cfdc7aad311fbe468a81061d6166bf83eee342a8c0a341f"


def canonical_digest(cases: list[dict]) -> str:
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_digests(root: Path | None = None) -> dict[str, str]:
    """SHA-256 of every frozen benchmark file, case files and the holdout."""
    base = root or ROOT
    digests: dict[str, str] = {}
    for name in ALL_CASE_FILES:
        path = base / "bench" / "cases" / name
        digests[f"bench/cases/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    holdout = base / "bench" / "holdout" / Path(HOLDOUT_REL).name
    if not holdout.exists():
        raise FileNotFoundError(f"missing holdout file: {holdout}")
    digests[HOLDOUT_REL] = hashlib.sha256(holdout.read_bytes()).hexdigest()
    return digests


def write(root: Path | None = None) -> dict:
    from .bench import load_all_cases, load_cases, load_holdout

    base = root or ROOT
    originals = load_cases(base)
    cases = load_all_cases(base)
    holdout = load_holdout(base)
    files = file_digests(base)
    record = {
        "version": 3,
        "count": len(cases),
        "normal": sum(1 for case in cases if case["kind"] == "normal"),
        "adversarial": sum(1 for case in cases if case["kind"] == "adversarial"),
        "original_count": len(originals),
        "original_normal": sum(1 for case in originals if case["kind"] == "normal"),
        "original_adversarial": sum(1 for case in originals if case["kind"] == "adversarial"),
        "files": files,
        "holdout": {
            "file": HOLDOUT_REL,
            "sha256": files[HOLDOUT_REL],
            "rows": len(holdout),
            "training_excluded": True,
        },
        "original_cases_digest": canonical_digest(originals),
        "cases_digest": canonical_digest(cases),
        "frozen_at": date.today().isoformat(),
        "note": "Training data may not be generated before this file exists and verifies. The holdout was authored by a reviewer and may never enter training data.",
    }
    path = base / "bench" / FROZEN_NAME
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    return record


def read(root: Path | None = None) -> dict | None:
    path = (root or ROOT) / "bench" / FROZEN_NAME
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def verify(root: Path | None = None) -> tuple[bool, list[str]]:
    from .bench import load_all_cases, load_cases, load_holdout

    base = root or ROOT
    record = read(base)
    if record is None:
        return False, [f"{FROZEN_NAME} is missing; run `python -m shopagent.cli freeze`"]
    problems: list[str] = []
    originals = load_cases(base)
    cases = load_all_cases(base)
    holdout = load_holdout(base)
    if canonical_digest(originals) != record.get("original_cases_digest"):
        problems.append("original 40-case digest changed")
    if canonical_digest(cases) != record.get("cases_digest"):
        problems.append("canonical case digest changed")
    current_files = file_digests(base)
    if current_files != record.get("files"):
        for name, digest in current_files.items():
            if record.get("files", {}).get(name) != digest:
                problems.append(f"{name} bytes changed")
        for name in record.get("files", {}):
            if name not in current_files:
                problems.append(f"{name} is missing")
    marked = record.get("holdout", {})
    if marked.get("training_excluded") is not True:
        problems.append("holdout is not marked training-excluded")
    if marked.get("sha256") != current_files.get(HOLDOUT_REL):
        problems.append("holdout digest changed")
    if int(marked.get("rows", -1)) != len(holdout):
        problems.append(f"holdout row count changed: {marked.get('rows')} -> {len(holdout)}")
    if int(record.get("count", -1)) != len(cases):
        problems.append(f"case count changed: {record.get('count')} -> {len(cases)}")
    if int(record.get("original_count", -1)) != len(originals):
        problems.append(f"original case count changed: {record.get('original_count')} -> {len(originals)}")
    return (not problems), problems
