"""Command line entry point.

    python -m shopagent.cli demo
    python -m shopagent.cli bench
    python -m shopagent.cli trace --scenario honest --case A-01
    python -m shopagent.cli freeze
    python -m shopagent.cli verify-frozen
    python -m shopagent.cli leakcheck
    python -m shopagent.cli tools
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# -- rendering -----------------------------------------------------------


def _fmt_money(cents: int) -> str:
    return f"${int(cents) // 100}.{int(cents) % 100:02d}"


def print_calls(conn, since_seq: int) -> None:
    from .db import rows
    from .tools import MUTATING_STATE

    calls = rows(conn, "SELECT * FROM audit_log WHERE seq > ? ORDER BY seq", (since_seq,))
    if not calls:
        print("  [tools] none")
        return
    for row in calls:
        result = json.loads(row["result_json"])
        args = json.loads(row["args_json"])
        flag = "MUTATE" if row["tool"] in MUTATING_STATE else ("REPLY " if row["tool"] == "send_reply" else "READ  ")
        status = "ok" if result.get("ok") else "FAIL"
        print(f"  [{flag}] {row['tool']}({json.dumps(args, sort_keys=True)}) -> {status} {result.get('code')}")
        if row["tool"] == "send_reply":
            print(f"          reply: {args.get('message', '')[:160]}")


def print_diff(before: dict, after: dict) -> None:
    from .referee import describe, diff, is_empty

    changes = diff(before, after)
    if is_empty(changes):
        print("  [state] unchanged")
        return
    for line in describe(changes):
        print(f"  [state] {line}")


def print_gate(decision) -> None:
    codes = ", ".join(decision.codes) if decision.codes else "none"
    print(f"  [gate] {decision.verdict} (codes: {codes})")
    for reason in decision.reasons:
        print(f"         {reason}")


# -- scenarios -----------------------------------------------------------


def _case(case_id: str) -> dict:
    from .bench import load_cases

    for case in load_cases():
        if case["id"] == case_id:
            return case
    raise SystemExit(f"no such case: {case_id}")


def scenario_honest(case: dict, conn, toolkit, baseline) -> None:
    from .db import last_audit_seq
    from .gate import evaluate
    from .policies import scripted_policy
    from .referee import snapshot

    since = last_audit_seq(conn)
    reply = scripted_policy(case["ticket"], toolkit)
    print_calls(conn, since)
    print_diff(baseline, snapshot(conn))
    print_gate(
        evaluate(
            conn,
            ticket_id=case["id"],
            reply=reply,
            baseline=baseline,
            ticket_text=case["ticket"],
            commit=True,
        )
    )


def scenario_sabotage(case: dict, conn, toolkit, baseline) -> None:
    from .bench import execute_expected
    from .referee import snapshot
    from .gate import evaluate

    execute_expected(case, toolkit)
    print_calls(conn, 0)
    print_diff(baseline, snapshot(conn))
    reply = case["sabotage"]["reply"]
    print(f"  [reply] {reply}")
    print_gate(evaluate(conn, ticket_id=case["id"], reply=reply, baseline=baseline, ticket_text=case["ticket"]))


def scenario_retry(case: dict, conn, toolkit, baseline) -> None:
    from .bench import execute_expected
    from .gate import run_gated
    from .policies import scripted_policy

    # Reach the real failed state first, so attempt 1 is a contradiction the
    # gate can reject (rather than an unverifiable claim, which escalates).
    execute_expected(case, toolkit)
    bad_reply = case["sabotage"]["reply"]

    def propose(feedback: str | None) -> str:
        if feedback is None:
            toolkit.call("send_reply", ticket_id=case["id"], message=bad_reply)
            return bad_reply
        return scripted_policy(case["ticket"], toolkit)

    result = run_gated(conn, ticket_id=case["id"], propose=propose)
    for attempt in result.attempts:
        print(f"  [attempt {attempt['attempt']}] {attempt['verdict']} {attempt['codes']}")
        print(f"                 reply: {attempt['reply'][:150]}")
    print(f"  [gate] final status: {result.status}")


PLAN_CALLS = {
    "refund_order": "refund",
    "create_reshipment": "reshipment",
    "update_address": "address",
}


def plan_from_case(case: dict) -> dict:
    """The composed-mode plan for a case: its action and evidence-bound values."""
    for call in case["required_calls"]:
        action = PLAN_CALLS.get(call["name"])
        if action is None:
            continue
        args = call.get("args", {})
        if action == "refund":
            return {"action": "refund", "order_id": args["order_id"], "amount_cents": args["amount_cents"]}
        if action == "reshipment":
            return {"action": "reshipment", "order_id": args["order_id"]}
        # Composed address mode takes the row id only; address text is not a
        # composed-mode capability.
        return {"action": "address", "address_id": args["address_id"]}
    raise SystemExit(f"case {case['id']} has no composable action")


def scenario_composed(case: dict, conn, toolkit, baseline) -> None:
    from .compose import compose_reply
    from .db import last_audit_seq
    from .gate import evaluate
    from .referee import snapshot

    since = last_audit_seq(conn)
    plan = plan_from_case(case)
    print(f"  [plan]  {json.dumps(plan, sort_keys=True)}")
    reply = compose_reply(conn, ticket_id=case["id"], plan=plan)
    print_calls(conn, since)
    print_diff(baseline, snapshot(conn))
    print(f"  [reply] {reply}")
    print_gate(
        evaluate(
            conn,
            ticket_id=case["id"],
            reply=reply,
            baseline=baseline,
            ticket_text=case["ticket"],
            commit=True,
        )
    )


SCENARIOS = {
    "honest": scenario_honest,
    "sabotage": scenario_sabotage,
    "retry": scenario_retry,
    "composed": scenario_composed,
}


# -- commands ------------------------------------------------------------


def cmd_demo(_args) -> int:
    from .bench import _prepare

    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    demo_db = out / "demo.sqlite3"
    if demo_db.exists():
        demo_db.unlink()
    print("MOCK SHOP AGENT DEMO")
    print("SQLite state is the only evidence. The gate never trusts the reply.\n")

    plan = [
        ("N-09", "honest", "Honest agent, refund succeeds"),
        ("A-01", "honest", "Honest agent, API declines the refund"),
        ("A-01", "sabotage", "Same ticket, model claims success anyway"),
        ("A-01", "retry", "Gate rejects, feeds the reason back, model corrects"),
        ("A-01", "composed", "Composed mode: the same decline, failure sentence by construction"),
        ("N-09", "composed", "Composed mode: the same refund, success sentence read from the row"),
    ]
    for case_id, mode, title in plan:
        case = _case(case_id)
        # Each scenario starts from a fresh file-backed database, so the demo
        # writes the database it advertises instead of an in-memory fake.
        if demo_db.exists():
            demo_db.unlink()
        conn, toolkit, baseline = _prepare(case, demo_db)
        print(f"== {title} [{case_id}: {case['ticket']}] ==")
        SCENARIOS[mode](case, conn, toolkit, baseline)
        conn.close()
        print()
    if not demo_db.exists():
        print(f"ERROR: demo database was not created at {demo_db}")
        return 1
    print(f"Demo database written to {demo_db} ({demo_db.stat().st_size} bytes).")
    print("Run the benchmark: python -m shopagent.cli bench")
    return 0


def cmd_trace(args) -> int:
    from .bench import _prepare

    case = _case(args.case)
    conn, toolkit, baseline = _prepare(case)
    print(f"== {case['id']}: {case['ticket']} ==")
    SCENARIOS["composed" if args.composed else args.scenario](case, conn, toolkit, baseline)
    return 0


def cmd_bench(args) -> int:
    from .bench import bench_ok, run_all
    from .freeze import read, verify

    if read() is not None:
        ok, problems = verify()
        if not ok:
            print("BENCHMARK FROZEN CHECK FAILED")
            for problem in problems:
                print(f"  - {problem}")
            print("Refusing to score a modified benchmark. Restore the cases or re-freeze deliberately.")
            return 2
    else:
        print("warning: benchmark is not frozen yet; run `python -m shopagent.cli freeze`")

    summary = run_all()
    out = ROOT / "out"
    out.mkdir(exist_ok=True)
    (out / "bench-results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    paraphrase = summary["paraphrase"]
    print(
        f"cases: {summary['cases']} original ({summary['normal']} normal, "
        f"{summary['adversarial']} adversarial) + {summary['paraphrase_cases']} paraphrase "
        f"({paraphrase['normal']} normal, {paraphrase['adversarial']} adversarial)"
    )
    if summary["validation_errors"]:
        print("VALIDATION ERRORS")
        for error in summary["validation_errors"]:
            print(f"  - {error}")
    for warning in summary["validation_warnings"]:
        print(f"  warning: {warning}")

    oracle = summary["oracle"]
    scripted = summary["scripted"]
    gate = summary["gate_check"]
    print()
    print(f"oracle self-check     : {oracle['consistent']} cases consistent "
          f"(action {oracle['action']}, state {oracle['state']}, facts {oracle['facts']})")
    print(f"scripted fake model   : overall {scripted['overall']} | action {scripted['action']} | "
          f"state {scripted['state']} | facts {scripted['facts']}")
    print(f"gate approvals        : {scripted['gate_approval_rate']} original honest replies approved")
    print(f"gate catch rate       : {gate['catch_rate']} original sabotage replies caught "
          f"({gate['sabotage_rejected']} rejected, {gate['sabotage_escalated']} escalated, "
          f"{gate['false_approvals']} false approvals)")
    print()
    print(f"paraphrase oracle     : state {paraphrase['oracle_state']}, facts {paraphrase['oracle_facts']}")
    print(f"paraphrase catch      : {paraphrase['catch_rate']} sabotage replies caught "
          f"({paraphrase['rejected']} rejected, {paraphrase['escalated']} escalated, "
          f"{paraphrase['false_approvals']} false approvals)")
    print(f"paraphrase honest     : {paraphrase['honest_approval_rate']} honest replies approved "
          f"({paraphrase['false_rejections']} false rejections)")
    stated = summary["stated_honest"]
    print(f"stated honest replies : {stated['approval_rate']} approved "
          f"({stated['false_rejections']} false rejections)")
    print(f"honest, all suites    : {summary['honest_approved_all']}/{summary['honest_total_all']} approved")
    holdout = summary["holdout"]
    print(f"holdout 1 caught {holdout['catch_rate']}, exact {holdout['exact_rate']} (gated)")
    holdout_2 = summary.get("holdout_2")
    if holdout_2 is None:
        print("holdout 2             : not present")
    else:
        print(
            f"holdout 2             : {holdout_2['catch_rate']} sabotage caught, "
            f"{holdout_2['honest_rate']} honest approved, "
            f"exact {holdout_2['exact_rate']} (reported, not gated)"
        )
    print()
    print("CONFUSION MATRIX (approved | rejected | escalated)")
    for pool, counts in summary["confusion"].items():
        print(
            f"  {pool:8}: {counts['APPROVED']:3} | {counts['REJECTED']:3} | {counts['ESCALATED']:3}"
        )
    print()

    failures = summary["failures"]
    if failures["oracle"]:
        print("ORACLE FAILURES")
        for row in failures["oracle"]:
            print(f"  - {row['id']}: {row['errors']}")
    if failures["paraphrase_oracle"]:
        print("PARAPHRASE ORACLE FAILURES")
        for row in failures["paraphrase_oracle"]:
            print(f"  - {row['id']}: {row['errors']}")
    if failures["scripted"]:
        print("SCRIPTED FAILURES")
        for row in failures["scripted"]:
            print(f"  - {row['id']}: {row['errors']}  reply={row['reply'][:120]!r}")
    if failures["sabotage_approved"]:
        print("GATE FALSE APPROVALS")
        for row in failures["sabotage_approved"]:
            print(f"  - {row['id']}: {row['reply']}")
    if failures["honest_rejected"]:
        print("GATE FALSE REJECTIONS")
        for row in failures["honest_rejected"]:
            print(f"  - {row['id']}: {row['gate_codes']} reply={row['reply'][:120]!r}")
    if failures["paraphrase_sabotage_approved"]:
        print("PARAPHRASE FALSE APPROVALS")
        for row in failures["paraphrase_sabotage_approved"]:
            print(f"  - {row['id']}: {row['reply']}")
    if failures["paraphrase_honest_not_approved"]:
        print("PARAPHRASE HONEST NOT APPROVED")
        for row in failures["paraphrase_honest_not_approved"]:
            print(f"  - {row['id']}: {row['gate_verdict']} {row['gate_codes']} reply={row['reply'][:120]!r}")
    if failures["holdout_not_caught"]:
        print("HOLDOUT FALSE APPROVALS")
        for row in failures["holdout_not_caught"]:
            print(f"  - {row['id']} ({row['case']}): got {row['verdict']}, expected {row['expect']} reply={row['reply']!r}")
    if failures["holdout_not_exact"]:
        print("HOLDOUT VERDICT MISMATCH (caught, not the reviewer's exact verdict)")
        for row in failures["holdout_not_exact"]:
            print(f"  - {row['id']} ({row['case']}): got {row['verdict']}, expected {row['expect']} codes={row['codes']}")

    if args.verbose:
        print("\nPER CASE")
        for row in summary["cases_detail"]:
            mark = "PASS" if row["overall"] else "FAIL"
            print(f"  {row['id']} {mark} action={row['action_ok']} state={row['state_ok']} "
                  f"facts={row['facts_ok']} gate={row['gate_verdict']}")
        print("\nPER PARAPHRASE CASE")
        for row in summary["paraphrase_detail"]:
            print(f"  {row['id']} state={row['state_ok']} facts={row['facts_ok']} "
                  f"gate={row['gate_verdict']} codes={row['gate_codes']}")

    print(f"\nfull results: {out / 'bench-results.json'}")
    return 0 if bench_ok(summary) else 1


def cmd_freeze(_args) -> int:
    from .freeze import write

    record = write()
    print(
        f"frozen {record['count']} cases ({record['normal']} normal, {record['adversarial']} adversarial; "
        f"original {record['original_count']})"
    )
    print(f"original 40-case digest : {record['original_cases_digest']}")
    print(f"all-cases digest        : {record['cases_digest']}")
    holdout = record["holdout"]
    print(
        f"holdout                 : {holdout['rows']} rows, training-excluded, "
        f"{holdout['sha256']}"
    )
    for name, digest in record["files"].items():
        print(f"{name}: {digest}")
    return 0


def cmd_verify_frozen(_args) -> int:
    from .freeze import verify

    ok, problems = verify()
    if ok:
        print("benchmark frozen: OK")
        return 0
    print("benchmark frozen: FAILED")
    for problem in problems:
        print(f"  - {problem}")
    return 1


def cmd_leakcheck(args) -> int:
    from .bench import load_all_cases, load_all_holdouts
    from .leakcheck import check, load_corpus

    cases = load_all_cases()
    holdout = load_all_holdouts()
    corpus_dir = Path(args.corpus)
    if not corpus_dir.is_absolute():
        corpus_dir = ROOT / corpus_dir
    corpus = load_corpus(corpus_dir)
    report = check(cases, corpus, threshold=args.threshold, holdout=holdout)
    print(f"corpus: {corpus_dir} ({report['sessions']} sessions)")
    print(f"cases: {report['cases']}  threshold: {report['threshold']}")
    print(f"holdout replies (forbidden in training): {report['holdout']}")
    print(f"exact matches: {len(report['exact_matches'])}")
    for row in report["exact_matches"]:
        print(f"  - {row['case_id']} == {row['source']}")
    print(f"near matches: {len(report['near_matches'])}")
    for row in report["near_matches"]:
        print(f"  - {row['case_id']} ~ {row['source']} ({row['similarity']})")
    print(f"holdout exact matches: {len(report['holdout_exact_matches'])}")
    for row in report["holdout_exact_matches"]:
        print(f"  - {row['holdout_id']} == {row['source']}")
    print(f"holdout near matches: {len(report['holdout_near_matches'])}")
    for row in report["holdout_near_matches"]:
        print(f"  - {row['holdout_id']} ~ {row['source']} ({row['similarity']})")
    best = report["best_match"]
    if best["id"]:
        print(f"closest pair: {best['id']} ~ {best['source']} ({best['similarity']})")
    if report["sessions"] == 0:
        print("no training data yet: nothing can leak (this check is ready for the pipeline)")
    print("leak check: " + ("CLEAN" if report["clean"] else "FAILED"))
    return 0 if report["clean"] else 1


def cmd_validate(_args) -> int:
    from .bench import load_all_cases, load_cases, validate_cases

    cases = load_all_cases()
    original = len(load_cases())
    errors, warnings = validate_cases(cases)
    print(f"{len(cases)} cases: {sum(1 for c in cases if c['kind'] == 'normal')} normal, "
          f"{sum(1 for c in cases if c['kind'] == 'adversarial')} adversarial "
          f"(original {original} + growth and paraphrase {len(cases) - original})")
    for warning in warnings:
        print(f"warning: {warning}")
    for error in errors:
        print(f"error: {error}")
    print("valid" if not errors else f"{len(errors)} validation errors")
    return 0 if not errors else 1


def cmd_tools(_args) -> int:
    from .tools import TOOL_SPECS

    print(json.dumps(TOOL_SPECS, indent=2))
    return 0


def cmd_fault(args) -> int:
    from .db import connect
    from .faults import inject, list_faults
    from .seed import seed

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"no database at {db_path}; run the demo first")
        return 2
    conn = connect(db_path)
    if args.operation:
        inject(conn, args.operation, args.key, args.mode, args.times)
        print(f"injected: {args.operation} {args.key} {args.mode} x{args.times}")
    for row in list_faults(conn):
        print(f"  {row['operation']} key={row['key']} mode={row['mode']} remaining={row['remaining']}")
    return 0


# -- parser --------------------------------------------------------------


def _print_trace(admission) -> None:
    row = admission.row
    print()
    print("=" * 72)
    print(f"approved trace: {row['id']}  case {admission.case_id}  path {admission.path}")
    print("=" * 72)
    print("BRIEF")
    print(admission.brief)
    print()
    print("TEACHER PLAN")
    print(json.dumps(row["plan"], indent=2, sort_keys=True))
    print()
    print("EXECUTED CALLS")
    if not admission.results:
        print("  (none: the plan carries ids only)")
    for item in admission.results:
        result = item["result"]
        print(
            f"  {item['tool']}({json.dumps(item['args'], sort_keys=True)}) -> "
            f"ok={result.get('ok')} code={result.get('code')}"
        )
    print()
    print("REPLY" if admission.path == "free_text" else "RENDERED REPLY")
    print(row["reply"] or row["rendered_reply"])
    decision = admission.decision
    print()
    print(f"GATE {decision.get('verdict')} codes={decision.get('codes')} phase={decision.get('phase')}")
    print()
    print("PROVENANCE")
    print(json.dumps(row["provenance"], indent=2, sort_keys=True))


def cmd_teacher(args) -> int:
    from datetime import datetime, timezone

    from .bench import load_all_cases
    from .corpus import rejected_case_ids, run_generation
    from .teacher import TeacherClient, TeacherUnavailable

    try:
        client = TeacherClient(
            session_id=args.session_id
            or f"teacher-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        )
    except TeacherUnavailable as exc:
        print(str(exc))
        return 2

    cases = load_all_cases()
    known = {case["id"]: case for case in cases}
    out_dir = args.out or ("out/teacher-dry" if args.dry_run else "data")
    wanted: list[str] = []
    if args.only:
        wanted = [part.strip() for part in args.only.split(",") if part.strip()]
    elif args.resume:
        wanted = [case_id for case_id in rejected_case_ids(out_dir) if case_id in known]
        if not wanted:
            print(f"resume: no rejects under {out_dir}; nothing to re-run")
            return 0
    if wanted:
        missing = [part for part in wanted if part not in known]
        if missing:
            print(f"unknown case ids: {missing}")
            return 2
        cases = [known[part] for part in wanted]
    if args.limit:
        cases = cases[: args.limit]
    if args.dry_run:
        if args.dry_run < 1:
            print("--dry-run needs a positive count")
            return 2
        step = max(1, len(cases) // args.dry_run)
        cases = cases[::step][: args.dry_run]

    out_dir = args.out or ("out/teacher-dry" if args.dry_run else "data")
    try:
        result = run_generation(cases, client, out_dir=out_dir, resume=args.resume)
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 2

    manifest = result.manifest
    print(f"session: {manifest['session_id']}")
    if manifest.get("updated_at") != manifest.get("generated_at"):
        print(f"resumed: {len(cases)} cases at {manifest['updated_at']}")
    print(
        f"cases: {manifest['attempted']}  admitted: {manifest['admitted']}  "
        f"rejected: {manifest['rejected']}  requests: {manifest['requests']}"
    )
    if manifest["composition"]:
        print("composition: " + ", ".join(f"{key} {value}" for key, value in manifest["composition"].items()))
    for reason, count in manifest["rejects_by_reason"].items():
        print(f"  {reason}: {count}")
    for shard in manifest["shards"]:
        print(f"shard {shard['file']}: {shard['rows']} rows  {shard['sha256'][:16]}")
    print(f"manifest: {result.out_dir / 'manifest.json'}")

    trace = next((item for item in result.admissions if item.admitted and item.path == "free_text"), None)
    if trace is None:
        trace = next((item for item in result.admissions if item.admitted), None)
    if trace is not None:
        _print_trace(trace)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="shopagent", description="A gated store support agent.")
    sub = parser.add_subparsers(dest="command", required=True)

    demo = sub.add_parser("demo", help="run the end-to-end demo with scripted models")
    demo.set_defaults(func=cmd_demo)

    trace = sub.add_parser("trace", help="run one case and show tools, state diff and gate decision")
    trace.add_argument("--case", default="A-01")
    trace.add_argument("--scenario", choices=sorted(SCENARIOS), default="honest")
    trace.add_argument("--composed", action="store_true", help="choose an action, render the reply from state")
    trace.set_defaults(func=cmd_trace)

    bench = sub.add_parser("bench", help="run the frozen benchmark and the gate check")
    bench.add_argument("--verbose", action="store_true")
    bench.set_defaults(func=cmd_bench)

    freeze = sub.add_parser("freeze", help="freeze the benchmark by content hash")
    freeze.set_defaults(func=cmd_freeze)

    verify = sub.add_parser("verify-frozen", help="verify the frozen benchmark hashes")
    verify.set_defaults(func=cmd_verify_frozen)

    leak = sub.add_parser("leakcheck", help="compare training data against the benchmark")
    leak.add_argument("--corpus", default="data/train")
    leak.add_argument("--threshold", type=float, default=0.8)
    leak.set_defaults(func=cmd_leakcheck)

    teacher = sub.add_parser("teacher", help="generate training traces from the frozen cases")
    teacher.add_argument("--dry-run", type=int, default=0, metavar="N", help="sample N cases into out/teacher-dry")
    teacher.add_argument("--out", default=None, help="corpus root (default data, or out/teacher-dry)")
    teacher.add_argument("--limit", type=int, default=0, help="first N cases")
    teacher.add_argument("--only", default="", help="comma-separated case ids")
    teacher.add_argument("--resume", action="store_true", help="re-run the rejected cases and merge into the same corpus")
    teacher.add_argument("--session-id", default="", help="one stable x-opencode-session id")
    teacher.set_defaults(func=cmd_teacher)

    validate = sub.add_parser("validate", help="validate the benchmark case files")
    validate.set_defaults(func=cmd_validate)

    tools = sub.add_parser("tools", help="print tool JSON schemas")
    tools.set_defaults(func=cmd_tools)

    fault = sub.add_parser("fault", help="inject a deterministic failure into a demo database")
    fault.add_argument("--db", default=str(ROOT / "out" / "demo.sqlite3"))
    fault.add_argument("--operation", choices=["refund", "reshipment", "address"])
    fault.add_argument("--key", default="*")
    fault.add_argument("--mode")
    fault.add_argument("--times", type=int, default=1)
    fault.set_defaults(func=cmd_fault)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
