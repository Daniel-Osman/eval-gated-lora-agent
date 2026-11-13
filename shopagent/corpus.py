"""Turn one frozen case into one trainable trace, or one recorded reject.

The teacher writes the customer message and a plan. The plan runs against the
real tool layer. The student's target is the plan (composed) or the plan plus
the reply (free text). A trace is admitted only when the tail of this chain
holds:

    plan matches required_calls -> state matches state_expect -> reply carries
    the required facts -> the gate returns APPROVED

Composed sentences are rendered by code, so their required facts are not
checked: those facts describe a free-text reply. Everything else is checked in
both modes.

Nothing here writes to disk. The run, the shards and the manifest are the next
slice.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .bench import load_all_cases, load_all_holdouts, match_plan, prepare_case
from .compose import ComposeError, compose_reply
from .db import last_audit_seq, rows
from .freeze import canonical_digest, verify as verify_frozen
from .gate import evaluate
from .leakcheck import check as leak_check
from .referee import check_expect
from .teacher import (
    ALLOWED_PLAN_TOOLS,
    BRIEF_REVISION,
    MUTATING_TOOLS,
    SYSTEM_PROMPT,
    TeacherParseError,
    build_reply_brief,
    build_scenario_brief,
    expected_action,
    parse_stage1,
    parse_stage2,
    status_order_id,
    template_digest,
)

COMPOSED_ACTIONS = ("status", "refund", "reshipment", "address")


class InvalidPlan(ValueError):
    """The plan is not a shape this pipeline can run."""


class PlanMismatch(ValueError):
    """The plan is well-formed but does not do what the case requires."""


@dataclass
class Admission:
    case_id: str
    path: str
    admitted: bool
    reason: str = ""
    detail: str = ""
    row: dict | None = None
    raw: list[dict] = field(default_factory=list)
    brief: str = ""
    results: list[dict] = field(default_factory=list)
    decision: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "case": self.case_id,
            "path": self.path,
            "admitted": self.admitted,
            "reason": self.reason,
            "detail": self.detail,
        }


def composed_unsupported(case: dict) -> str | None:
    """Why a composed case cannot be expressed as a composed plan, or None.

    No expectation is changed for these cases: they take the free-text path
    with `required_calls`, `state_expect` and their facts untouched.
    """
    if not case.get("composed"):
        return None
    mutating = [call for call in case.get("required_calls", []) if call.get("name") in MUTATING_TOOLS]
    if len(mutating) > 1:
        return "more than one mutating call"
    if mutating:
        call = mutating[0]
        wrote_text = any(key in call.get("args", {}) for key in ("line1", "city", "postcode"))
        if call["name"] == "update_address" and call.get("ok", True) and wrote_text:
            return "address write cannot be expressed as an address id"
    return None


# -- checks --------------------------------------------------------------


def _reject(admission: Admission, reason: str, detail: str = "") -> Admission:
    admission.admitted = False
    admission.reason = reason
    admission.detail = detail
    return admission


def _check_plan_list(plan) -> list[dict]:
    """A free-text plan: a list of {name, args} over the allowed tools."""
    if not isinstance(plan, list):
        raise InvalidPlan("free-text plan must be a list of calls")
    calls: list[dict] = []
    for call in plan:
        if not isinstance(call, dict):
            raise InvalidPlan("each call must be an object")
        unknown = set(call) - {"name", "args"}
        if unknown:
            raise InvalidPlan(f"call carries unknown keys {sorted(unknown)}")
        name = call.get("name")
        if name not in ALLOWED_PLAN_TOOLS:
            raise InvalidPlan(f"tool {name!r} is not a plan tool")
        args = call.get("args", {})
        if not isinstance(args, dict):
            raise InvalidPlan(f"{name}: args must be an object")
        calls.append({"name": name, "args": args})
    return calls


def _check_composed_plan(case: dict, plan) -> str:
    """Validate the composed action and its ids before anything runs."""
    if not isinstance(plan, dict):
        raise InvalidPlan("composed plan must be one object")
    action = plan.get("action")
    if action not in COMPOSED_ACTIONS:
        raise InvalidPlan(f"unknown composed action {action!r}")
    wanted = expected_action(case)
    if action != wanted:
        raise PlanMismatch(f"action {action!r} does not match the case's {wanted!r}")

    mutating = next(
        (call for call in case.get("required_calls", []) if call.get("name") in MUTATING_TOOLS),
        None,
    )
    if action == "refund":
        args = (mutating or {}).get("args", {})
        if plan.get("order_id") != args.get("order_id"):
            raise PlanMismatch(f"refund order_id {plan.get('order_id')!r} != {args.get('order_id')!r}")
        if plan.get("amount_cents") != args.get("amount_cents"):
            raise PlanMismatch(
                f"refund amount {plan.get('amount_cents')!r} != {args.get('amount_cents')!r}"
            )
    elif action == "reshipment":
        args = (mutating or {}).get("args", {})
        if plan.get("order_id") != args.get("order_id"):
            raise PlanMismatch(f"reshipment order_id {plan.get('order_id')!r} != {args.get('order_id')!r}")
    elif action == "address":
        args = (mutating or {}).get("args", {})
        if plan.get("address_id") != args.get("address_id"):
            raise PlanMismatch(f"address_id {plan.get('address_id')!r} != {args.get('address_id')!r}")
    else:
        wanted_id = status_order_id(case)
        if plan.get("order_id") != wanted_id:
            raise PlanMismatch(f"status order_id {plan.get('order_id')!r} != {wanted_id!r}")
    return action


def _check_reply(case: dict, reply: str) -> list[str]:
    lowered = reply.lower()
    missing = [fact for fact in case.get("required_facts", []) if fact.lower() not in lowered]
    present = [phrase for phrase in case.get("forbidden_phrases", []) if phrase.lower() in lowered]
    problems: list[str] = []
    if missing:
        problems.append(f"missing facts: {missing}")
    if present:
        problems.append(f"forbidden phrases: {present}")
    return problems


def _provenance(client, case: dict, *, path: str, verdict: str, codes: list[str], now: str) -> dict:
    return {
        "model": getattr(client, "model", ""),
        "gateway": getattr(client, "base_url", ""),
        "session_id": getattr(client, "session_id", ""),
        "case_id": case["id"],
        "generated_at": now,
        "verdict": verdict,
        "codes": list(codes),
        "source_ticket_sha256": hashlib.sha256(case["ticket"].encode("utf-8")).hexdigest(),
        "routed": "unsupported_composed" if path == "free_text" and case.get("composed") else None,
    }


def _message_row(
    *, case: dict, path: str, ticket: str, plan, reply: str | None, results: list[dict]
) -> tuple[list[dict], dict]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": ticket},
        {"role": "assistant", "content": json.dumps(plan, sort_keys=True)},
    ]
    if path == "free_text":
        for item in results:
            messages.append({"role": "tool", "content": json.dumps(item, sort_keys=True)})
        messages.append({"role": "assistant", "content": reply or ""})
        target = {"type": "plan_and_reply", "plan": plan, "reply": reply}
    else:
        target = {"type": "plan", "plan": plan}
    return messages, target


# -- retry on a shape error only -----------------------------------------

SHAPE_PLAN_LIST = '{"ticket": "...", "plan": [{"name": "...", "args": {...}}]}'
SHAPE_PLAN_OBJECT = '{"ticket": "...", "plan": {"action": "status|refund|reshipment|address", ...}}'
SHAPE_REPLY = '{"reply": "..."}'


def _correction(brief: str, error: str, shape: str) -> str:
    return (
        f"{brief}\n\nYour previous answer was rejected: {error}\n"
        "Return one JSON object and nothing else, in this shape:\n"
        f"{shape}"
    )


def _retry_text(client, brief: str, error: str, shape: str, admission: Admission, stage: int) -> str:
    """One resend, only for a shape error. A wrong verdict or a missing fact
    is a quality failure and goes to rejects instead."""
    text = client.chat(SYSTEM_PROMPT, _correction(brief, error, shape)).text
    admission.raw.append({"stage": stage, "text": text, "attempts": 2, "retry": True})
    return text


# -- generation ----------------------------------------------------------


def generate(client, case: dict, *, now: str | None = None) -> Admission:
    """One case, one admission. `client` needs `chat`, `model`, `base_url`,
    `session_id`; tests pass a scripted stand-in."""
    composed = bool(case.get("composed")) and composed_unsupported(case) is None
    if composed:
        return _generate_composed(client, case, now=now)
    return _generate_free_text(client, case, now=now)


def _generate_free_text(client, case: dict, *, now: str | None = None) -> Admission:
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    admission = Admission(case_id=case["id"], path="free_text", admitted=False)
    brief = build_scenario_brief(case, path="free_text")
    admission.brief = brief

    text = client.chat(SYSTEM_PROMPT, brief).text
    admission.raw.append({"stage": 1, "text": text, "attempts": 1})
    for attempt in (1, 2):
        try:
            ticket, plan = parse_stage1(text)
            plan = _check_plan_list(plan)
            break
        except (TeacherParseError, InvalidPlan) as exc:
            if attempt == 2:
                reason = "parse_error" if isinstance(exc, TeacherParseError) else "invalid_plan"
                return _reject(admission, reason, str(exc))
            text = _retry_text(client, brief, str(exc), SHAPE_PLAN_LIST, admission, 1)

    conn, toolkit, baseline = prepare_case(case)
    seq = last_audit_seq(conn)
    results: list[dict] = []
    try:
        for call in plan:
            results.append(
                {"tool": call["name"], "args": call["args"], "result": toolkit.call(call["name"], **call["args"])}
            )
        audit = [dict(row) for row in rows(conn, "SELECT * FROM audit_log WHERE seq > ? ORDER BY seq", (seq,))]

        problems = match_plan(case, audit)
        if problems:
            return _reject(admission, "plan_mismatch", "; ".join(problems[:3]))

        state_problems = check_expect(conn, case.get("state_expect", []))
        if state_problems:
            return _reject(admission, "state_mismatch", "; ".join(state_problems[:3]))

        reply_brief = build_reply_brief(
            ticket=ticket,
            plan=plan,
            results=results,
            required_facts=case.get("required_facts", []),
        )
        text = client.chat(SYSTEM_PROMPT, reply_brief).text
        admission.raw.append({"stage": 2, "text": text, "attempts": 1})
        for attempt in (1, 2):
            try:
                reply = parse_stage2(text)
                break
            except TeacherParseError as exc:
                if attempt == 2:
                    return _reject(admission, "parse_error", str(exc))
                text = _retry_text(client, reply_brief, str(exc), SHAPE_REPLY, admission, 2)

        lowered = reply.lower()
        missing = [f for f in case.get("required_facts", []) if f.lower() not in lowered]
        if missing:
            return _reject(admission, "facts_missing", f"missing facts: {missing}")
        present = [p for p in case.get("forbidden_phrases", []) if p.lower() in lowered]
        if present:
            return _reject(admission, "forbidden_phrase", f"forbidden phrases: {present}")

        decision = evaluate(
            conn,
            ticket_id=case["id"],
            reply=reply,
            baseline=baseline,
            ticket_text=ticket,
        )
        admission.decision = {"verdict": decision.verdict, "codes": list(decision.codes), "phase": decision.phase}
        admission.results = results
        if decision.verdict != "APPROVED":
            return _reject(admission, f"gate:{decision.verdict}", ", ".join(decision.codes))

        messages, target = _message_row(
            case=case, path="free_text", ticket=ticket, plan=plan, reply=reply, results=results
        )
        admission.admitted = True
        admission.row = {
            "id": f"T-{case['id']}",
            "case": case["id"],
            "kind": case["kind"],
            "path": "free_text",
            "ticket": ticket,
            "plan": plan,
            "reply": reply,
            "rendered_reply": None,
            "messages": messages,
            "target": target,
            "provenance": _provenance(
                client, case, path="free_text", verdict=decision.verdict, codes=decision.codes, now=stamp
            ),
        }
        return admission
    finally:
        conn.close()


def _generate_composed(client, case: dict, *, now: str | None = None) -> Admission:
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    admission = Admission(case_id=case["id"], path="composed", admitted=False)
    brief = build_scenario_brief(case, path="composed")
    admission.brief = brief

    text = client.chat(SYSTEM_PROMPT, brief).text
    admission.raw.append({"stage": 1, "text": text, "attempts": 1})
    for attempt in (1, 2):
        try:
            ticket, plan = parse_stage1(text)
            _check_composed_plan(case, plan)
            break
        except PlanMismatch as exc:
            return _reject(admission, "plan_mismatch", str(exc))
        except (TeacherParseError, InvalidPlan) as exc:
            if attempt == 2:
                reason = "parse_error" if isinstance(exc, TeacherParseError) else "invalid_plan"
                return _reject(admission, reason, str(exc))
            text = _retry_text(client, brief, str(exc), SHAPE_PLAN_OBJECT, admission, 1)

    conn, _toolkit, baseline = prepare_case(case)
    try:
        try:
            rendered = compose_reply(conn, ticket_id=case["id"], plan=plan)
        except ComposeError as exc:
            # A plan that names a shipment the renderer cannot find is a
            # mismatch, not a malformed plan.
            reason = "plan_mismatch" if plan.get("shipment_id") else "invalid_plan"
            return _reject(admission, reason, str(exc))

        if plan.get("shipment_id"):
            want = str(plan["shipment_id"]).upper()
            found = rows(
                conn,
                "SELECT id, tracking_number FROM shipments WHERE order_id = ?",
                (plan.get("order_id"),),
            )
            if not any(want in {str(row["id"]).upper(), str(row["tracking_number"] or "").upper()} for row in found):
                return _reject(admission, "plan_mismatch", f"shipment {want} does not belong to this order")

        state_problems = check_expect(conn, case.get("state_expect", []))
        if state_problems:
            return _reject(admission, "state_mismatch", "; ".join(state_problems[:3]))

        present = [p for p in case.get("forbidden_phrases", []) if p.lower() in rendered.lower()]
        if present:
            return _reject(admission, "forbidden_phrase", f"forbidden phrases: {present}")

        decision = evaluate(
            conn,
            ticket_id=case["id"],
            reply=rendered,
            baseline=baseline,
            ticket_text=ticket,
        )
        admission.decision = {"verdict": decision.verdict, "codes": list(decision.codes), "phase": decision.phase}
        if decision.verdict != "APPROVED":
            return _reject(admission, f"gate:{decision.verdict}", ", ".join(decision.codes))

        messages, target = _message_row(
            case=case, path="composed", ticket=ticket, plan=plan, reply=None, results=[]
        )
        admission.admitted = True
        admission.row = {
            "id": f"T-{case['id']}",
            "case": case["id"],
            "kind": case["kind"],
            "path": "composed",
            "ticket": ticket,
            "plan": plan,
            "reply": None,
            "rendered_reply": rendered,
            "messages": messages,
            "target": target,
            "provenance": _provenance(
                client, case, path="composed", verdict=decision.verdict, codes=decision.codes, now=stamp
            ),
        }
        return admission
    finally:
        conn.close()


# -- run, store, manifest ------------------------------------------------

HOLDOUT_ID = re.compile(r"^HR2?-\d+$", re.I)
SHARD_SIZE = 64


@dataclass
class RunResult:
    admissions: list[Admission]
    dropped: list[dict]
    manifest: dict
    out_dir: Path


def _guard_ids(cases: list[dict]) -> None:
    """A holdout row's id can never name a training trace."""
    for case in cases:
        if HOLDOUT_ID.match(str(case.get("id", ""))):
            raise ValueError(f"holdout id {case['id']!r} cannot enter the corpus")


def _write_jsonl(path: Path, records: list[dict], base: Path) -> dict:
    payload = "".join(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n" for record in records)
    data = payload.encode("utf-8")
    path.write_bytes(data)
    return {
        "file": str(path.relative_to(base)).replace("\\", "/"),
        "rows": len(records),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def _screen(admitted: list[Admission], cases: list[dict], holdouts: list[dict]) -> dict[str, str]:
    """Which admitted rows collide with a bench ticket or a holdout reply."""
    if not admitted:
        return {}
    records = [(item.row["id"], json.dumps(item.row, ensure_ascii=False)) for item in admitted]
    report = leak_check(cases, records, holdout=holdouts)
    reasons: dict[str, str] = {}
    for match in report["exact_matches"]:
        reasons.setdefault(match["source"], f"leak:{match['case_id']}")
    for match in report["near_matches"]:
        reasons.setdefault(match["source"], f"leak_near:{match['case_id']}")
    for match in report["holdout_exact_matches"]:
        reasons.setdefault(match["source"], f"leak_holdout:{match['holdout_id']}")
    for match in report["holdout_near_matches"]:
        reasons.setdefault(match["source"], f"leak_holdout_near:{match['holdout_id']}")
    return reasons


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _previous_state(base: Path) -> dict | None:
    """The corpus already on disk, for a resumed pass. None when there is none."""
    manifest_path = base / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    for shard in manifest.get("shards", []):
        rows.extend(_read_jsonl(base / shard["file"]))
    return {
        "manifest": manifest,
        "rows": rows,
        "rejects": _read_jsonl(base / "rejects" / "rejected.jsonl"),
        "raw": _read_jsonl(base / "raw" / "raw.jsonl"),
    }


def reindex_manifest(out_dir: str | Path = "data") -> dict:
    """Recompute the derived manifest fields from the files as they lie.

    Identity and history fields are kept. Digests are taken from the bytes on
    disk, never from a re-serialisation, so a stale digest cannot hide here.
    A leak drop is derived from the reject ledger, which is the only place a
    drop is recorded.
    """
    base = Path(out_dir)
    manifest_path = base / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    shards: list[dict] = []
    rows: list[dict] = []
    for path in sorted((base / "train").glob("shard-*.jsonl")):
        data = path.read_bytes()
        records = [json.loads(line) for line in data.decode("utf-8").splitlines() if line.strip()]
        shards.append(
            {
                "file": str(path.relative_to(base)).replace("\\", "/"),
                "rows": len(records),
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
        rows.extend(records)

    def sidecar(relative: str) -> dict:
        data = (base / relative).read_bytes()
        return {
            "file": relative,
            "rows": len([line for line in data.decode("utf-8").splitlines() if line.strip()]),
            "sha256": hashlib.sha256(data).hexdigest(),
        }

    rejected = _read_jsonl(base / "rejects" / "rejected.jsonl")
    manifest.update(
        {
            "brief_revision": BRIEF_REVISION,
            "admitted": len(rows),
            "rejected": len(rejected),
            "attempted": len({row["case"] for row in rows} | {row["case"] for row in rejected}),
            "rejects_by_reason": dict(sorted(Counter(row["reason"] for row in rejected).items())),
            "composition": dict(sorted(Counter(row["path"] for row in rows).items())),
            "dropped_leaks": [
                {"case": row["case"], "reason": row["reason"]}
                for row in rejected
                if row["reason"].startswith("leak")
            ],
            "shards": shards,
            "raw": sidecar("raw/raw.jsonl"),
            "rejects": sidecar("rejects/rejected.jsonl"),
        }
    )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def rejected_case_ids(out_dir: str | Path = "data") -> list[str]:
    """Case ids in the reject ledger, in file order. A resumed pass runs these."""
    seen: list[str] = []
    for row in _read_jsonl(Path(out_dir) / "rejects" / "rejected.jsonl"):
        case = row.get("case")
        if case and case not in seen:
            seen.append(case)
    return seen


def run_generation(
    cases: list[dict],
    client,
    *,
    out_dir: str | Path = "data",
    shard_size: int = SHARD_SIZE,
    holdouts: list[dict] | None = None,
    now: str | None = None,
    resume: bool = False,
) -> RunResult:
    """Generate every case, admit what passes, and write the corpus.

    Written only after `verify-frozen`, and only from rows the gate approved.
    A row that collides with a bench ticket or a holdout reply is dropped.

    `resume=True` merges this pass into the corpus already on disk: a re-run
    case's old verdict is replaced, every other row is kept, and the manifest
    is recomputed from the merged files.
    """
    stamp = now or datetime.now(timezone.utc).isoformat(timespec="seconds")
    _guard_ids(cases)
    ok, problems = verify_frozen()
    if not ok:
        raise RuntimeError("benchmark is not frozen: " + "; ".join(problems[:3]))

    base = Path(out_dir)
    previous = _previous_state(base) if resume else None
    if resume and previous is None:
        raise ValueError(f"cannot resume: no corpus manifest under {base}")
    if resume and not cases:
        raise ValueError("cannot resume: no cases to re-run")

    workers = max(1, min(getattr(client, "max_concurrency", 8), len(cases) or 1))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        admissions = list(pool.map(lambda case: generate(client, case, now=stamp), cases))

    heldout = load_all_holdouts() if holdouts is None else holdouts
    admitted = [item for item in admissions if item.admitted]
    # Screen against the whole frozen bank, never the subset this pass ran:
    # a resumed pass generates a few cases and must still refuse a bench ticket.
    collision = _screen(admitted, load_all_cases(), heldout)
    for item in admitted:
        reason = collision.get(item.row["id"])
        if reason:
            item.admitted = False
            item.reason = reason

    payloads = [item.row for item in admissions if item.admitted]
    rejected = [
        {"case": item.case_id, "path": item.path, "reason": item.reason, "detail": item.detail}
        for item in admissions
        if not item.admitted
    ]
    raw_records = [
        {
            "case": item.case_id,
            "stage": entry.get("stage"),
            "text": entry.get("text", ""),
            "attempts": entry.get("attempts"),
            "admitted": item.admitted,
        }
        for item in admissions
        for entry in item.raw
    ]
    old_runs: list[dict] = []
    previous_requests = 0
    previous_generated_at = stamp
    previous_template = template_digest()
    previous_cases_digest = canonical_digest(cases)
    if previous is not None:
        prior = previous["manifest"]
        rerun = {item.case_id for item in admissions}
        payloads = [row for row in previous["rows"] if row["case"] not in rerun] + payloads
        rejected = [row for row in previous["rejects"] if row["case"] not in rerun] + rejected
        raw_records = previous["raw"] + raw_records
        old_runs = prior.get("runs") or [
            {
                "generated_at": prior.get("generated_at", ""),
                "template_digest": prior.get("template_digest", ""),
                "cases": prior.get("attempted", 0),
                "requests": prior.get("requests", 0),
                "admitted": prior.get("admitted", 0),
                "rejected": prior.get("rejected", 0),
            }
        ]
        previous_requests = prior.get("requests", 0)
        previous_generated_at = prior.get("generated_at", stamp)
        previous_template = prior.get("template_digest", previous_template)
        previous_cases_digest = prior.get("cases_digest", previous_cases_digest)

    payloads.sort(key=lambda row: row["case"])
    rejected.sort(key=lambda row: row["case"])

    for folder in ("train", "raw", "rejects"):
        (base / folder).mkdir(parents=True, exist_ok=True)
    for stale in (base / "train").glob("shard-*.jsonl"):
        stale.unlink()

    shards = [
        _write_jsonl(
            base / "train" / f"shard-{index // shard_size:03d}.jsonl",
            payloads[index : index + shard_size],
            base,
        )
        for index in range(0, len(payloads), shard_size)
    ]
    raw_info = _write_jsonl(base / "raw" / "raw.jsonl", raw_records, base)
    reject_info = _write_jsonl(base / "rejects" / "rejected.jsonl", rejected, base)

    # A leak drop is a reject whose reason names a collision, so the ledger is
    # the single source: a resumed pass cannot leave a stale or missing drop.
    dropped = [
        {"case": row["case"], "reason": row["reason"]}
        for row in rejected
        if row["reason"].startswith("leak")
    ]

    current_run = {
        "brief_revision": BRIEF_REVISION,
        "generated_at": stamp,
        "template_digest": template_digest(),
        "cases": len(cases),
        "requests": getattr(client, "requests", 0),
        "admitted": sum(1 for item in admissions if item.admitted),
        "rejected": sum(1 for item in admissions if not item.admitted),
    }
    attempted = len({row["case"] for row in payloads} | {row["case"] for row in rejected})

    manifest = {
        "version": 1,
        "generated_at": previous_generated_at,
        "updated_at": stamp,
        "model": getattr(client, "model", ""),
        "gateway": getattr(client, "base_url", ""),
        "session_id": getattr(client, "session_id", ""),
        "brief_revision": BRIEF_REVISION,
        "template_digest": template_digest(),
        "template_digest_history": sorted({previous_template, template_digest()}),
        "cases_digest": previous_cases_digest,
        "attempted": attempted,
        "admitted": len(payloads),
        "rejected": len(rejected),
        "rejects_by_reason": dict(sorted(Counter(row["reason"] for row in rejected).items())),
        "composition": dict(sorted(Counter(row["path"] for row in payloads).items())),
        "dropped_leaks": dropped,
        "requests": previous_requests + getattr(client, "requests", 0),
        "resumed_requests": getattr(client, "requests", 0) if previous is not None else 0,
        "runs": old_runs + [current_run],
        "holdout_rows_checked": len(heldout),
        "shard_size": shard_size,
        "shards": shards,
        "raw": raw_info,
        "rejects": reject_info,
    }
    (base / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return RunResult(admissions=admissions, dropped=dropped, manifest=manifest, out_dir=base)
