"""The prompt and parse helpers the composed suite needs, copied, not invented.

`_strip_stop_token`, `check_prompt`, `parse_plan`, `training_target` and
`composed_verdict` are byte-for-byte the functions in `train/modal_app.py` at
commit 4499d17 — `tests/test_serve_suite.py` compares their ASTs so the copies
cannot drift. The two checks that decide a verdict — `shopagent.compose._check_plan`
and `shopagent.corpus._check_composed_plan` — are **imported** here, never copied.

No Modal import and no file read at import time: this module is importable in a
test run with no network and no Volume.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"
ASSISTANT_TURN = "<|im_start|>assistant"
IM_END = "<|im_end|>"


class RunError(RuntimeError):
    """A check that must stop the run. Always names what failed."""


def _strip_stop_token(text: str) -> str:
    for stop in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(stop):
            return text[: -len(stop)].rstrip()
    return text


def parse_plan(text: str) -> tuple[dict | None, str]:
    """The output must be one JSON object at the root, and nothing else.

    One object, or one fence whose whole content is that object. A list of tool
    calls is a valid free-text plan and an invalid composed plan, so a dict inside
    a list is refused rather than extracted. Prose outside a fence is refused, and
    the fence must wrap the entire output. Keys are judged by the project's own
    validator, so a prose key is refused here too.
    """
    from shopagent.compose import ComposeError, _check_plan

    cleaned = _strip_stop_token(text.strip())
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", cleaned, re.S)
    if fenced:
        cleaned = _strip_stop_token(fenced.group(1).strip())
    elif "```" in cleaned:
        return None, "text outside the fence: a fence must wrap the whole output"
    if not cleaned:
        return None, "empty output"
    try:
        payload, end = json.JSONDecoder().raw_decode(cleaned)
    except json.JSONDecodeError as exc:
        return None, f"not one JSON value at the root: {exc}"
    if not isinstance(payload, dict):
        return None, f"plan is {type(payload).__name__}, not an object at the root"
    trailing = cleaned[end:].strip()
    if trailing:
        return None, f"trailing text after the plan: {trailing[:60]!r}"
    try:
        _check_plan(payload)
    except ComposeError as exc:
        return None, f"composed-plan validator refused: {exc}"
    return payload, "ok"


def check_prompt(text: str, case_id: str, tokenizer, input_ids: list[int], first: bool = False) -> str:
    """The exact prompt training saw before a case's assistant content.

    The SFT record keeps the rendered chat, so this is a byte slice of it, not a
    re-render that a template change could move. The slice must tokenize to the
    record's own id prefix, or it is not the training prompt. `first` takes the
    opening assistant turn — where the free-text plan belongs — instead of the
    last one, which is the reply.
    """
    marker = ASSISTANT_TURN + "\n"
    start = text.index(marker) if first else text.rindex(marker)
    empty_think = THINK_CLOSE + "\n\n"
    stop = text.find(empty_think, start)
    if stop < 0:
        raise RunError(f"check_prompt: {case_id} has no empty {THINK_OPEN} block after its assistant marker")
    prompt = text[: stop + len(empty_think)]
    ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    if ids != input_ids[: len(ids)]:
        raise RunError(f"check_prompt: {case_id} prompt does not tokenize to its training id prefix")
    return prompt


def training_target(text: str, case_id: str) -> dict:
    """The plan the record trains for: the assistant content after its think block."""
    marker = text.rindex(ASSISTANT_TURN + "\n")
    stop = text.find(THINK_CLOSE + "\n\n", marker)
    if stop < 0:
        raise RunError(f"check_target: {case_id} has no empty {THINK_OPEN} block")
    content = text[stop + len(THINK_CLOSE + "\n\n") :].strip()
    if content.endswith(IM_END):
        content = content[: -len(IM_END)].strip()
    try:
        target = json.loads(content)
    except json.JSONDecodeError as exc:
        raise RunError(f"check_target: {case_id} target is not JSON: {exc}") from exc
    if not isinstance(target, dict):
        raise RunError(f"check_target: {case_id} target is {type(target).__name__}, not an object")
    return target


def composed_verdict(case: dict, plan: dict | None) -> tuple[bool, str]:
    """The project's own check: action, ids and amount against `required_calls`."""
    if plan is None:
        return False, "no plan to check"
    from shopagent.corpus import InvalidPlan, PlanMismatch, _check_composed_plan

    try:
        action = _check_composed_plan(case, plan)
    except InvalidPlan as exc:
        return False, f"invalid plan: {exc}"
    except PlanMismatch as exc:
        return False, f"plan mismatch: {exc}"
    return True, f"action {action} matches required_calls"


def composed_case_ids(rows: list[dict]) -> list[str]:
    """Every composed case in the SFT dataset: the suite is what was trained on."""
    ids = []
    for row in rows:
        if row["path"] == "composed" and row["case"] not in ids:
            ids.append(row["case"])
    if not ids:
        raise RunError("check_cases: the SFT dataset has no composed row")
    return sorted(ids)


def check_payload(rows: list[dict], bench_root: Path) -> dict:
    """The composed suite, with each case dict carried for the validator."""
    from shopagent.bench import load_all_cases

    composed = composed_case_ids(rows)
    wanted = {case_id: "composed" for case_id in composed}
    payload = {}
    for case in load_all_cases(Path(bench_root)):
        kind = wanted.get(case["id"])
        if kind is None:
            continue
        payload[case["id"]] = {"kind": kind, "case": case}
    missing = sorted(set(wanted) - set(payload))
    if missing:
        raise RunError(f"check_cases: {missing} are not in bench/cases")
    return payload
