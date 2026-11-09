"""Leak check between the frozen benchmark and synthetic training data.

Training data must never touch the benchmark. This check reports:

* exact matches after normalisation (case, punctuation, whitespace)
* near duplicates by token 3-shingle Jaccard similarity >= threshold

It compares training prompts against two forbidden sets: the benchmark tickets
and the reviewer-authored holdout replies. A generated session that reproduces a
holdout reply would teach the model the answer, so it is a leak even when it
matches no ticket.

It runs on a corpus directory (`data/train/` by default). Before any data exists
it reports zero sessions; the tests prove it catches planted duplicates.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_THRESHOLD = 0.8
SHINGLE_SIZE = 3


def normalize(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokens(text: str) -> list[str]:
    normalised = normalize(text)
    return normalised.split() if normalised else []


def shingles(text: str, size: int = SHINGLE_SIZE) -> set[str]:
    words = tokens(text)
    if len(words) < size:
        return {" ".join(words)} if words else set()
    return {" ".join(words[index : index + size]) for index in range(len(words) - size + 1)}


def jaccard(left: set[str], right: set[str]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 1.0
    return len(left & right) / len(union)


def containment(left: set[str], right: set[str]) -> float:
    """How much of the smaller prompt is covered by the other one.

    Catches a training prompt that wraps a benchmark ticket in extra words, which
    plain Jaccard misses when the lengths differ.
    """
    if not left and not right:
        return 1.0
    smaller = min(len(left), len(right))
    if smaller == 0:
        return 0.0
    return len(left & right) / smaller


def similarity(left: set[str], right: set[str]) -> float:
    return max(jaccard(left, right), containment(left, right))


def _strings_in(value) -> list[str]:
    """Every string in a record, in document order."""
    found: list[str] = []

    def walk(node) -> None:
        if isinstance(node, str):
            found.append(node)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        elif isinstance(node, dict):
            for child in node.values():
                walk(child)

    walk(value)
    return found


def _session_text(record) -> str:
    """The whole record as searchable text.

    Two views, both searched: the raw JSON, so a reply copied straight into a
    field is found; and every string joined by spaces, so a reply split across
    tokens (`["They","refunded",...]`) is found too. Neither view cares about
    key names, nesting or field boundaries.
    """
    raw = json.dumps(record, ensure_ascii=False)
    joined = " ".join(_strings_in(record))
    return raw + "\n" + joined


def load_corpus(corpus_dir: Path) -> list[tuple[str, str]]:
    """Return [(source, text)] for every record in the corpus.

    One entry per record, holding the whole record: raw JSON plus its strings
    joined. Nothing is skipped by key name or by value shape, so a reply hidden
    in a list, a nested object or a split field is still searched.
    """
    sessions: list[tuple[str, str]] = []
    if not corpus_dir.exists():
        return sessions
    for path in sorted(corpus_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        if path.name in {"README.md", "FROZEN.json"}:
            continue
        if path.suffix == ".jsonl":
            with path.open("r", encoding="utf-8") as handle:
                for lineno, line in enumerate(handle, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    sessions.append((f"{path.name}:{lineno}", _session_text(record)))
        elif path.suffix == ".json":
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, UnicodeDecodeError):
                continue
            sessions.append((path.name, _session_text(payload)))
        elif path.suffix == ".txt":
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                sessions.append((path.name, text))
    return sessions


def contains(haystack: str, needle: str) -> bool:
    """True when the needle appears inside the haystack after normalisation.

    Works on the serialized record, so a reply is found whether it sits in one
    field, is split across list tokens, or is nested several keys deep.
    """
    pin = normalize(needle)
    return bool(pin) and pin in normalize(haystack)


def check(
    cases: list[dict],
    corpus: list[tuple[str, str]],
    threshold: float = DEFAULT_THRESHOLD,
    holdout: list[dict] | None = None,
) -> dict:
    """Compare the corpus against case tickets and holdout replies.

    `holdout` rows need `id`, `case` and `reply`; the replies are forbidden in
    training data exactly like the tickets are.
    """
    holdout = holdout or []
    case_shingles = {case["id"]: shingles(case["ticket"]) for case in cases}
    holdout_shingles = {row["id"]: shingles(row["reply"]) for row in holdout}
    exact: list[dict] = []
    near: list[dict] = []
    holdout_exact: list[dict] = []
    holdout_near: list[dict] = []
    best = {"id": None, "source": None, "similarity": 0.0}

    for source, prompt in corpus:
        prompt_shingles = shingles(prompt)
        for case in cases:
            if contains(prompt, case["ticket"]):
                exact.append({"case_id": case["id"], "source": source})
                continue
            score = similarity(case_shingles[case["id"]], prompt_shingles)
            if score > best["similarity"]:
                best = {"id": case["id"], "source": source, "similarity": round(score, 4)}
            if score >= threshold:
                near.append(
                    {"case_id": case["id"], "source": source, "similarity": round(score, 4), "prompt": prompt[:200]}
                )
        for row in holdout:
            if contains(prompt, row["reply"]):
                holdout_exact.append({"holdout_id": row["id"], "case_id": row["case"], "source": source})
                continue
            score = similarity(holdout_shingles[row["id"]], prompt_shingles)
            if score >= threshold:
                holdout_near.append(
                    {
                        "holdout_id": row["id"],
                        "case_id": row["case"],
                        "source": source,
                        "similarity": round(score, 4),
                        "prompt": prompt[:200],
                    }
                )

    return {
        "sessions": len(corpus),
        "cases": len(cases),
        "holdout": len(holdout),
        "threshold": threshold,
        "exact_matches": exact,
        "near_matches": near,
        "holdout_exact_matches": holdout_exact,
        "holdout_near_matches": holdout_near,
        "best_match": best,
        "clean": not exact and not near and not holdout_exact and not holdout_near,
    }
