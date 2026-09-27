"""Synthetic Judger-only direction diagnostic; never a scoring entry point.

The probe fixes one real typed MethodIR and its adapter evidence, then asks the
ordinary independent Judger whether that evidence is sufficient in four
outcome directions.  It excludes extraction, compilation, Oracle selection,
and executor work, so it cannot support a full-E2E or 10x claim.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any, Callable

from experiments.terminal_bench.probe_record_game import (
    _audit_model_work,
    _empty_metrics,
    _ledger_snapshots,
)
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint,
    _validate_execute_output_root,
)
from verifier import offline
from verifier.contract_replay import ReplaySetupError, clone_sealed_inputs
from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    Obligation,
    content_hash,
)
from verifier.engine.game import Attempt, BindingGame
from verifier.execution.adapters import (
    AdapterContext,
    default_adapter_registry,
)
from verifier.execution.instruments import ArtifactStore
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.codex.usage import codex_usage_ledger
from verifier.llm.runtime import capabilities_of
from verifier.llm.model_spans import model_scope, span_sidecar_path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "gv-judger-direction-probe/v1"
SCOPE = "judger-only-typed-record-direction"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium",
           "service_tier": "standard"}
TOOL = "instrument.synthetic.v1"
VALUES = ("alpha", "beta")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_EXCLUDED = ("extract", "Compiler", "Critic", "Oracle", "executor", "setup")
CASES = ("exists-alpha", "exists-beta", "forall-alpha", "forall-beta")
_TASK = (
    "This is a task-neutral synthetic sufficiency audit. The full retained "
    "scope is every tool_call event in events.jsonl, at every sequence; no "
    "tool_call is excluded. Each obligation defines either existence or "
    "universality over the exact string at event.payload.input.value. A "
    "bounded prefix is not authority for completeness of that full scope."
)


def _feed_rows() -> list[dict[str, Any]]:
    return [
        {"type": "tool_call", "payload": {
            "tool": TOOL, "tool_use_id": f"synthetic-{index}",
            "input": {"value": value},
        }}
        for index, value in enumerate(VALUES, 1)
    ]


def _write_synthetic_source(source: Path) -> str:
    source.mkdir(mode=0o700, parents=True, exist_ok=False)
    (source / "task.md").write_text(_TASK, encoding="utf-8")
    (source / "final.txt").write_text(
        "Synthetic verifier diagnostic input; not a candidate answer.\n",
        encoding="utf-8",
    )
    (source / "hook_feed.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
                for row in _feed_rows()),
        encoding="utf-8",
    )
    return offline.write_immutable_input_anchor(os.fspath(source))


def _materialize_synthetic_ledger(state: Path) -> EventLog:
    """Project the sealed feed to deterministic ledger bytes (no live action)."""
    try:
        rows = [json.loads(line) for line in
                (state / "hook_feed.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()]
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ReplaySetupError("synthetic feed cannot be read") from error
    if rows != _feed_rows():
        raise ReplaySetupError("synthetic feed differs from its fixed profile")
    events = [
        {"seq": index, "ts": 0.0, "type": row["type"],
         "payload": row["payload"]}
        for index, row in enumerate(rows, 1)
    ]
    (state / "events.jsonl").write_text(
        "".join(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n"
                for event in events),
        encoding="utf-8",
    )
    return EventLog(os.fspath(state / "events.jsonl"))


def _case_specs() -> list[dict[str, Any]]:
    specs = []
    expected = {
        ("exists", "alpha"): "yes",
        ("exists", "beta"): "no",
        ("forall", "alpha"): "no",
        ("forall", "beta"): "yes",
    }
    for quantifier in ("exists", "forall"):
        for target in VALUES:
            case_id = f"{quantifier}-{target}"
            if quantifier == "exists":
                text = (
                    "In the complete immutable retained tool-call scope, at "
                    f"least one event has input.value exactly {target!r}."
                )
                selector = "any"
                absence = True
            else:
                text = (
                    "In the complete immutable retained tool-call scope, "
                    f"every event has input.value exactly {target!r}."
                )
                selector = "all"
                absence = False
            claim_id = content_hash({
                "schema": SCHEMA, "case": case_id, "obligation": text,
            })
            obligation = Obligation(
                claim_id=claim_id,
                text=text,
                required_basis=BASIS_RECORD,
                severity="must",
                clause=(
                    "The comparison domain is all retained tool_call events, "
                    "not merely an observed prefix."
                ),
                refutable_by_absence=absence,
                premise_scope=(),
            )
            method = {
                "procedure": (
                    "Run ledger.tool-call-string@2 over input.value only "
                    "through ledger sequence 1 using the declared exact "
                    f"{selector}/equals query."
                ),
                "artifact": (
                    "One content-addressed typed-adapter evidence envelope "
                    "for that exact bounded query."
                ),
                "level": "exact bounded ledger query",
                "basis": BASIS_RECORD,
                "strength": "decisive",
                "bridges": [],
                "adapter": "ledger.tool-call-string",
                "adapter_version": 2,
                "adapter_arguments": json.dumps({
                    "selector": selector,
                    "value_pointer": "/input/value",
                    "through_seq": 1,
                    "predicate": "equals",
                    "expected": target,
                }, sort_keys=True, separators=(",", ":")),
                "adapter_inputs": [{
                    "ref": "ledger:events", "provenance": "ledger",
                    "role": "subject",
                }],
            }
            specs.append({
                "case_id": case_id,
                "row": {"id": claim_id, "kind": "assertion", "text": text},
                "obligation": obligation,
                "method": method,
                "expected_sufficient": expected[(quantifier, target)],
            })
    return specs


def _case_spec(case_id: str) -> dict[str, Any]:
    if case_id not in CASES:
        raise ReplaySetupError("case must be one fixed synthetic case")
    return next(spec for spec in _case_specs() if spec["case_id"] == case_id)


def _subject_digests(log: EventLog) -> tuple[str, ...]:
    events = list(log.replay())
    subjects = [event for event in events if event.get("type") == "tool_call"]
    if len(subjects) != 2 or [event.get("seq") for event in subjects] != [1, 2]:
        raise ReplaySetupError("synthetic subject event set changed")
    return tuple(EventLog.event_digest(event) for event in subjects)


def _artifact_integrity(
    artifacts: ArtifactStore, digests: list[str],
) -> bool:
    try:
        return all(content_hash(artifacts.get(digest)) == digest
                   for digest in digests)
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _runtime_backend(
    state: Path, backend_factory: Callable[[Path], Any] | None,
    judger_prefetch: bool = False,
) -> Any:
    backend = (backend_factory(state) if backend_factory is not None else
               CodexRuntime(cwd=os.fspath(state), **RUNTIME,
                            compact_inspection=False,
                            judger_prefetch=judger_prefetch))
    options = capabilities_of(backend).run_options
    if options.compact_inspection is not False:
        raise ReplaySetupError("backend differs from compact_inspection=false")
    if options.judger_prefetch is not judger_prefetch:
        raise ReplaySetupError("backend differs from judger_prefetch profile")
    for field, expected in RUNTIME.items():
        if hasattr(backend, field) and getattr(backend, field) != expected:
            raise ReplaySetupError("backend differs from pinned Luna runtime")
    if getattr(backend, "model_overrides", {}) not in ({}, None):
        raise ReplaySetupError("role model overrides are not permitted")
    capabilities_of(backend).validate_offline_tool_scope(
        os.fspath(state), None, None)
    return backend


def run_probe(
    output: Path,
    *,
    case_id: str,
    expected_source_fingerprint: str,
    backend_factory: Callable[[Path], Any] | None = None,
    judger_prefetch: bool = False,
) -> dict[str, Any]:
    """Run one isolated Judger call; ``backend_factory`` is test-only."""
    if type(judger_prefetch) is not bool:
        raise ReplaySetupError("judger_prefetch must be a bool")
    output = output.absolute()
    if (not output.is_absolute() or ".." in output.parts
            or os.path.lexists(output)):
        raise ReplaySetupError("output root must be a new absolute path")
    if (_SHA256.fullmatch(expected_source_fingerprint or "") is None
            or _source_fingerprint(SOURCE_ROOT)
            != expected_source_fingerprint):
        raise ReplaySetupError("running verifier differs from operator pin")
    spec = _case_spec(case_id)

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    source = output / "synthetic-source"
    state = output / "state"
    source_seal = _write_synthetic_source(source)
    clone_sealed_inputs(source, state)
    log = _materialize_synthetic_ledger(state)
    subject_before = _subject_digests(log)
    artifacts = ArtifactStore(os.fspath(state / "artifacts"))
    registry = default_adapter_registry()
    context = AdapterContext(os.fspath(state), False, log)
    backend = _runtime_backend(state, backend_factory, judger_prefetch)
    game = BindingGame(
        backend, artifacts, log, None, live_world=False,
        method_cache=None,
    )

    probe_id = "judger-direction-" + uuid.uuid4().hex
    usage = output / "usage.jsonl"
    spans = Path(span_sidecar_path(os.fspath(usage)))
    usage.touch()
    spans.touch()
    started_ns = time.monotonic_ns()
    start = {
        "schema": SCHEMA, "status": "started", "probe_id": probe_id,
        "scope": SCOPE, "diagnostic_only": True,
        "full_e2e_acceptance": False,
        "test_only": backend_factory is not None,
        "runtime": dict(RUNTIME), "compact_inspection": False,
        "judger_prefetch": judger_prefetch,
        "judger_prefetch_protocol": getattr(
            backend, "_judger_prefetch_protocol", None),
        "source_fingerprint": expected_source_fingerprint,
        "source_seal": source_seal,
        "excluded_stages": list(_EXCLUDED),
        "excluded_stage_costs": {name: None for name in _EXCLUDED},
        "fixture_adapter": "ledger.tool-call-string@2",
        "fixture_adapter_executions_planned": 1,
        "case_id": case_id,
        "started_at_monotonic_ns": started_ns,
    }
    # Expectations are deliberately absent until every model call has ended.
    offline._write_json_atomic(os.fspath(output / "probe-start.json"), start)

    results: list[dict[str, Any]] = []
    evidence_digests: list[str] = []
    error_class: str | None = None
    interrupted: BaseException | None = None
    ledger_context = {"probe_id": probe_id}
    try:
        with codex_usage_ledger(os.fspath(usage), {os.fspath(state): ledger_context}):
            row = spec["row"]
            obligation = spec["obligation"]
            method = spec["method"]
            result = registry.execute(
                method, row, obligation, context, artifacts)
            if result is None:
                raise ReplaySetupError("typed fixture method was not executed")
            evidence_digests.append(result.artifact)
            attempt = Attempt(
                method=method,
                discharge_artifact=result.artifact,
                discharge_outcome=result.outcome,
            )
            with model_scope(
                claim_id=row["id"], claim_kind=row["kind"],
                locus="support", obligation_id=obligation.id,
            ):
                verdict = game._judge(
                    obligation, method, result, attempt,
                    method_attempt=1, independent_judger=True,
                )
            expected_sufficient = spec["expected_sufficient"]
            direction_matches = (
                verdict.get("sufficient") == expected_sufficient
                and (expected_sufficient == "no" or (
                    verdict.get("faithful") == "yes"
                    and verdict.get("degradation") == "n/a"
                ))
            )
            results.append({
                "case_id": spec["case_id"],
                "claim_id": row["id"],
                "obligation_id": obligation.id,
                "method_sha256": content_hash(method),
                "adapter_outcome": result.outcome,
                "witness_kind": result.witness_kind,
                "evidence_artifact": result.artifact,
                "evidence_artifact_bytes": (
                    Path(artifacts.root) / f"{result.artifact}.json"
                ).stat().st_size,
                "actual": {key: verdict.get(key) for key in (
                    "sufficient", "faithful", "degradation", "reason"
                )},
                "expected_sufficient": expected_sufficient,
                "direction_matches": direction_matches,
            })
    except BaseException as error:
        error_class = type(error).__name__
        if not isinstance(error, Exception):
            interrupted = error

    finished_ns = time.monotonic_ns()
    wall = max(0.0, (finished_ns - started_ns) / 1e9)
    try:
        cost = _audit_model_work(
            usage, spans, wall_seconds=wall,
            context=ledger_context, state=state,
        )
    except BaseException as error:
        cost = {"complete": False, "findings": ["accounting_audit_failed"],
                "metrics": _empty_metrics(), "span_report": None}
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error

    source_seal_after = state_seal_after = fingerprint_after = None
    subject_after: tuple[str, ...] | None = None
    try:
        source_seal_after = offline.verify_immutable_input_anchor(
            os.fspath(source))
        state_seal_after = offline.verify_immutable_input_anchor(
            os.fspath(state))
        subject_after = _subject_digests(log)
        fingerprint_after = _source_fingerprint(SOURCE_ROOT)
    except BaseException as error:
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error
    input_unchanged = (
        source_seal_after == source_seal
        and state_seal_after == source_seal
        and subject_after == subject_before
        and fingerprint_after == expected_source_fingerprint
    )
    artifacts_unchanged = _artifact_integrity(artifacts, evidence_digests)
    snapshots = _ledger_snapshots(usage, spans)
    if any(value is None for value in snapshots.values()):
        error_class = error_class or "LedgerSnapshotError"
    profile_complete = False
    attached_attempts = 0
    try:
        bills = [json.loads(line) for line in (snapshots["usage"] or b"").splitlines()
                 if line.strip()]
        profile_complete = bool(bills) and all(
            bill.get("judger_prefetch") is judger_prefetch
            and bill.get("judger_prefetch_protocol")
            == start["judger_prefetch_protocol"]
            for bill in bills)
        attached_attempts = sum(
            bill.get("judger_prefetch_attached") is True for bill in bills)
    except (TypeError, ValueError, AttributeError):
        error_class = error_class or "ProfileAccountingError"
    # Recheck after accounting and snapshot reads, the final trust boundary.
    try:
        final_unchanged = (
            offline.verify_immutable_input_anchor(os.fspath(source))
            == source_seal
            and offline.verify_immutable_input_anchor(os.fspath(state))
            == source_seal
            and _subject_digests(log) == subject_before
            and _source_fingerprint(SOURCE_ROOT) == expected_source_fingerprint
            and _artifact_integrity(artifacts, evidence_digests)
        )
    except BaseException as error:
        final_unchanged = False
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error
    valid_verdicts = (
        len(results) == 1
        and all(item["actual"]["sufficient"] in {"yes", "no"}
                and item["actual"]["faithful"] in {"yes", "no"}
                and item["actual"]["degradation"] == "n/a"
                for item in results)
    )
    complete = (
        valid_verdicts and cost.get("complete") is True
        and profile_complete
        and input_unchanged and artifacts_unchanged and final_unchanged
        and error_class is None
    )
    observed_match = len(results) == 1 and results[0]["direction_matches"]
    terminal = {
        **start, "status": "terminal",
        "diagnostic_status": "complete" if complete else "inconclusive",
        "complete": complete, "semantic_pass": complete and observed_match,
        "observed_expectations_match": observed_match,
        "error_class": error_class,
        "finished_at_monotonic_ns": finished_ns, "wall_seconds": wall,
        "source_seal_after": source_seal_after,
        "state_input_seal_after": state_seal_after,
        "source_fingerprint_after": fingerprint_after,
        "input_unchanged": input_unchanged,
        "subject_event_digests": list(subject_before),
        "subject_events_unchanged": subject_after == subject_before,
        "evidence_artifacts_unchanged": artifacts_unchanged and final_unchanged,
        "state_path": "state", "source_path": "synthetic-source",
        "usage_path": usage.name, "spans_path": spans.name,
        "usage_sha256": (hashlib.sha256(snapshots["usage"]).hexdigest()
                         if snapshots["usage"] is not None else None),
        "spans_sha256": (hashlib.sha256(snapshots["spans"]).hexdigest()
                         if snapshots["spans"] is not None else None),
        "cost_report": cost,
        "fixture_adapter_executions": len(evidence_digests),
        "runtime_profile_accounting_complete": profile_complete,
        "prefetch_attached_attempts": attached_attempts,
        # Written only after the paid/model scope has closed.
        "cases": results,
        "note": (
            "This fixes MethodIR and real adapter evidence to isolate Judger "
            "directional sufficiency. The local adapter fixture execution is "
            "included in wall time; source setup and the other excluded stages "
            "are unmeasured, not zero."
        ),
    }
    offline._write_json_atomic(os.fspath(output / "probe-result.json"), terminal)
    if interrupted is not None:
        raise interrupted
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--judger-prefetch", choices=("0", "1"), default="0")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    if args.execute and args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    if (_SHA256.fullmatch(args.source_fingerprint or "") is None
            or _source_fingerprint(SOURCE_ROOT) != args.source_fingerprint):
        parser.error("running verifier differs from --source-fingerprint")
    if os.path.lexists(args.output_root):
        parser.error("output-root must be new")
    if not args.execute:
        print(json.dumps({
            "schema": SCHEMA, "scope": SCOPE, "diagnostic_only": True,
            "full_e2e_acceptance": False,
            "output_root": os.fspath(args.output_root.absolute()),
            "runtime": RUNTIME, "compact_inspection": False,
            "judger_prefetch": args.judger_prefetch == "1",
            "source_fingerprint": args.source_fingerprint,
            "case": {"case_id": _case_spec(args.case)["case_id"],
                     "expected_sufficient":
                         _case_spec(args.case)["expected_sufficient"]},
        }, indent=2))
        return 0
    _validate_execute_output_root(args.output_root)
    result = run_probe(
        args.output_root,
        case_id=args.case,
        expected_source_fingerprint=args.source_fingerprint,
        judger_prefetch=args.judger_prefetch == "1",
    )
    print(json.dumps({
        "output": os.fspath(args.output_root),
        "complete": result["complete"],
        "semantic_pass": result["semantic_pass"],
    }))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
