"""One-obligation offline BindingGame diagnostic; never a scoring entry point.

The operator supplies a previously compiled record obligation.  This probe
clones only sealed inputs and runs the ordinary Oracle -> executor -> Judger
game once, with or without the default typed-adapter router.  It deliberately
does not run or price extraction, contract compilation, or contract criticism.
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

from experiments.terminal_bench.replay_prepared_contracts import (
    _audit_model_work, _empty_metrics, _ledger_snapshots,
)
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint, _validate_execute_output_root,
)
from experiments.terminal_bench.validate_prepared_contract_entry import (
    MAX_RECEIPT_BYTES, _private_read,
)
from verifier import offline
from verifier.contract_replay import ReplaySetupError, clone_sealed_inputs
from verifier.core.events import EventLog
from verifier.core.model import Obligation, content_hash
from verifier.engine.contract_bundle import _strict_json
from verifier.engine.game import BindingGame
from verifier.execution.discharge import AgentDischarger
from verifier.execution.instruments import ArtifactStore
from verifier.llm.codex_backend import CodexBackend, codex_usage_ledger
from verifier.llm.model_spans import span_sidecar_path


SOURCE_ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "gv-precompiled-record-obligation-probe/v1"
SCOPE = "precompiled-record-obligation-binding-only"
RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium",
           "service_tier": "standard"}
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_CASE_KEYS = frozenset(("row", "obligation"))
_ROW_KEYS = frozenset(("id", "kind", "text"))
_OBLIGATION_KEYS = frozenset((
    "claim_id", "text", "required_basis", "severity", "clause",
    "refutable_by_absence", "premise_scope",
))


def _case(path: Path) -> tuple[dict[str, Any], Obligation, str]:
    raw = _private_read(path.resolve(strict=True), MAX_RECEIPT_BYTES)
    value = _strict_json(raw)
    if type(value) is not dict or set(value) != _CASE_KEYS:
        raise ReplaySetupError("case must contain exactly row and obligation")
    row, spec = value["row"], value["obligation"]
    if (type(row) is not dict or set(row) != _ROW_KEYS
            or not isinstance(row.get("id"), str)
            or _SHA256.fullmatch(row["id"]) is None
            or row.get("kind") not in {"assertion", "final"}
            or not isinstance(row.get("text"), str) or not row["text"]):
        raise ReplaySetupError("case row is malformed")
    if (type(spec) is not dict or set(spec) != _OBLIGATION_KEYS
            or spec.get("claim_id") != row["id"]
            or not isinstance(spec.get("text"), str) or not spec["text"]
            or spec.get("required_basis") != "record"
            or spec.get("severity") not in {"must", "should"}
            or not isinstance(spec.get("clause"), str)
            or type(spec.get("refutable_by_absence")) is not bool
            or spec.get("premise_scope") != []):
        raise ReplaySetupError(
            "case must be one context-free record obligation for its row")
    obligation = Obligation(
        claim_id=spec["claim_id"], text=spec["text"],
        required_basis=spec["required_basis"], severity=spec["severity"],
        clause=spec["clause"],
        refutable_by_absence=spec["refutable_by_absence"],
        premise_scope=(),
    )
    return dict(row), obligation, hashlib.sha256(raw).hexdigest()


def _build_game(state: Path, typed: bool, backend: Any) -> BindingGame:
    log = EventLog(str(state / "events.jsonl"))
    artifacts = ArtifactStore(str(state / "artifacts"))
    fallback = AgentDischarger(
        backend, artifacts, workdir_base=str(state / "work"), cleanup=False,
        state_dir=str(state), live_world=False, store=None,
    )
    discharger: Any = fallback
    if typed:
        from verifier.execution.adapters import (
            AdapterRoutingDischarger, default_adapter_registry,
        )
        discharger = AdapterRoutingDischarger(
            fallback, default_adapter_registry(), artifacts, log, str(state),
            live_world=False,
        )
    return BindingGame(
        backend, artifacts, log, discharger, live_world=False,
        method_cache=None,
    )


def _binding(result: Any, artifacts: ArtifactStore) -> dict[str, Any]:
    artifact = Path(artifacts.root) / f"{result.artifact}.json"
    return {
        "outcome": result.outcome, "artifact": result.artifact,
        "artifact_bytes": artifact.stat().st_size,
        "instrument": result.instrument, "basis": result.basis,
        "strength": result.strength, "witness_kind": result.witness_kind,
        "bridges": list(result.bridges),
        "assumed_premises": list(result.assumed_premises),
        "inputs": list(result.inputs),
        "witness_sha256": content_hash(result.witness),
    }


def _event_summary(log: EventLog, artifacts: ArtifactStore) -> dict[str, Any]:
    events = list(log.replay())
    counts: dict[str, int] = {}
    for event in events:
        counts[event["type"]] = counts.get(event["type"], 0) + 1
    adapters = []
    for event in events:
        if event["type"] != "adapter_discharge":
            continue
        item = dict(event["payload"])
        artifact = Path(artifacts.root) / f"{item.get('artifact', '')}.json"
        item["artifact_bytes"] = artifact.stat().st_size
        adapters.append(item)
    return {
        "counts": counts,
        "methods": [event["payload"].get("methods", []) for event in events
                    if event["type"] == "methods_enumerated"],
        "adapter_discharges": adapters,
        "adapter_fallbacks": [event["payload"] for event in events
                              if event["type"] == "adapter_fallback"],
        "judger_verdicts": [event["payload"] for event in events
                            if event["type"] == "judger_verdict"],
    }


def run_probe(
    source: Path,
    case_path: Path,
    output: Path,
    *,
    typed_adapters: bool,
    expected_source_seal: str,
    expected_source_fingerprint: str,
    compact_inspection: bool = False,
    backend_factory: Callable[[Path], Any] | None = None,
) -> dict[str, Any]:
    """Run one arm. ``backend_factory`` is a local-test seam, not a CLI flag."""
    if type(typed_adapters) is not bool:
        raise ReplaySetupError("typed_adapters must be a boolean")
    if type(compact_inspection) is not bool:
        raise ReplaySetupError("compact_inspection must be a boolean")
    source = source.resolve(strict=True)
    output = output.absolute()
    if (not output.is_absolute() or ".." in output.parts
            or os.path.lexists(output)):
        raise ReplaySetupError("output root must be a new absolute path")
    if os.path.commonpath((str(source), str(output.resolve()))) in {
            str(source), str(output.resolve())}:
        raise ReplaySetupError("source and output must be disjoint")
    if (_SHA256.fullmatch(expected_source_seal or "") is None
            or offline.verify_immutable_input_anchor(str(source))
            != expected_source_seal):
        raise ReplaySetupError("source differs from the operator seal")
    if (_SHA256.fullmatch(expected_source_fingerprint or "") is None
            or _source_fingerprint(SOURCE_ROOT)
            != expected_source_fingerprint):
        raise ReplaySetupError("running verifier differs from the operator pin")
    row, obligation, case_sha256 = _case(case_path)

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    probe_id = "record-probe-" + uuid.uuid4().hex
    arm = "typed" if typed_adapters else "control"
    start_ns = time.monotonic_ns()
    start = {
        "schema": SCHEMA, "status": "started", "probe_id": probe_id,
        "scope": SCOPE, "diagnostic_only": True,
        "full_e2e_acceptance": False, "arm": arm,
        "test_only": backend_factory is not None,
        "typed_adapters": typed_adapters, "runtime": dict(RUNTIME),
        "compact_inspection": compact_inspection,
        "source": str(source), "source_seal": expected_source_seal,
        "source_root": str(SOURCE_ROOT),
        "source_fingerprint": expected_source_fingerprint,
        "case_sha256": case_sha256, "claim_id": row["id"],
        "obligation_id": obligation.id,
        "excluded_stages": ["extract", "obligations", "critic", "setup"],
        "excluded_stage_costs": {
            "extract": None, "obligations": None, "critic": None,
            "setup": None,
        },
        "started_at_monotonic_ns": start_ns,
    }
    offline._write_json_atomic(str(output / "probe-start.json"), start)

    usage = output / "usage.jsonl"
    spans = Path(span_sidecar_path(str(usage)))
    usage.touch()
    spans.touch()
    state: Path | None = None
    result = game = None
    error_class: str | None = None
    interrupted: BaseException | None = None
    context = {"probe_id": probe_id, "arm": arm}
    try:
        clone_sealed_inputs(source, output / "state")
        state = (output / "state").resolve(strict=True)
        log = EventLog(str(state / "events.jsonl"))
        offline.ingest_feed(str(state / "hook_feed.jsonl"), log)
        backend = (backend_factory(state) if backend_factory is not None else
                   CodexBackend(cwd=str(state), **RUNTIME,
                                compact_inspection=compact_inspection))
        if getattr(backend, "compact_inspection", False) is not compact_inspection:
            raise ReplaySetupError("backend differs from inspection profile")
        validate = getattr(backend, "validate_offline_tool_scope", None)
        if callable(validate):
            validate(str(state), None, None)
        game = _build_game(state, typed_adapters, backend)
        with codex_usage_ledger(str(usage), {str(state): context}):
            result = game.run(row, obligation)
    except BaseException as error:
        error_class = type(error).__name__
        if not isinstance(error, Exception):
            interrupted = error

    finished_ns = time.monotonic_ns()
    wall = max(0.0, (finished_ns - start_ns) / 1e9)
    try:
        cost = _audit_model_work(
            usage, spans, wall_seconds=wall, context=context, state=state)
    except BaseException as error:
        cost = {"complete": False, "findings": ["accounting_audit_failed"],
                "metrics": _empty_metrics(), "span_report": None}
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error

    seal_after = None
    fingerprint_after = None
    cloned_seal_after = None
    try:
        seal_after = offline.verify_immutable_input_anchor(str(source))
        if state is not None:
            cloned_seal_after = offline.verify_immutable_input_anchor(str(state))
        fingerprint_after = _source_fingerprint(SOURCE_ROOT)
    except BaseException as error:
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error
    source_unchanged = (seal_after == expected_source_seal and
                        (state is None or cloned_seal_after == expected_source_seal) and
                        fingerprint_after == expected_source_fingerprint)
    binding = events = None
    try:
        if result is not None and game is not None:
            binding = _binding(result, game.artifacts)
            events = _event_summary(game.log, game.artifacts)
    except BaseException as error:
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error
    snapshots = _ledger_snapshots(usage, spans)
    if any(value is None for value in snapshots.values()):
        error_class = error_class or "LedgerSnapshotError"
    # This is the final boundary read: post-processing and accounting may not
    # race a source or code mutation into an apparently complete record.
    try:
        seal_after = offline.verify_immutable_input_anchor(str(source))
        if state is not None:
            cloned_seal_after = offline.verify_immutable_input_anchor(str(state))
        fingerprint_after = _source_fingerprint(SOURCE_ROOT)
    except BaseException as error:
        error_class = error_class or type(error).__name__
        if not isinstance(error, Exception) and interrupted is None:
            interrupted = error
    source_unchanged = (seal_after == expected_source_seal and
                        (state is None or
                         cloned_seal_after == expected_source_seal) and
                        fingerprint_after == expected_source_fingerprint)
    complete = (result is not None and cost.get("complete") is True
                and source_unchanged and error_class is None)
    terminal = {
        **start, "status": "terminal",
        "diagnostic_status": "complete" if complete else "inconclusive",
        "complete": complete, "error_class": error_class,
        "finished_at_monotonic_ns": finished_ns, "wall_seconds": wall,
        "source_seal_after": seal_after,
        "source_fingerprint_after": fingerprint_after,
        "source_unchanged": source_unchanged,
        "state_path": "state" if state is not None else None,
        "usage_path": usage.name, "spans_path": spans.name,
        "usage_sha256": (hashlib.sha256(snapshots["usage"]).hexdigest()
                         if snapshots["usage"] is not None else None),
        "spans_sha256": (hashlib.sha256(snapshots["spans"]).hexdigest()
                         if snapshots["spans"] is not None else None),
        "cost_report": cost,
        "binding_result": binding, "events": events,
        "judger_ledger_rescan_observable": False,
        "note": (
            "Upstream stages were excluded, not measured as zero. The probe "
            "records the receipt passed to Judger but cannot prove which "
            "permitted files the model read internally."
        ),
    }
    offline._write_json_atomic(str(output / "probe-result.json"), terminal)
    if interrupted is not None:
        raise interrupted
    return terminal


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--case", dest="case_path", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--typed-adapters", choices=("0", "1"), required=True)
    parser.add_argument("--compact-inspection", choices=("0", "1"), default="0")
    parser.add_argument("--source-seal", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    if args.execute and args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    source = args.source.resolve(strict=True)
    _case(args.case_path)
    if offline.verify_immutable_input_anchor(str(source)) != args.source_seal:
        parser.error("source differs from --source-seal")
    if _source_fingerprint(SOURCE_ROOT) != args.source_fingerprint:
        parser.error("running verifier differs from --source-fingerprint")
    if os.path.lexists(args.output_root):
        parser.error("output-root must be new")
    if not args.execute:
        print(json.dumps({
            "schema": SCHEMA, "scope": SCOPE, "diagnostic_only": True,
            "full_e2e_acceptance": False, "source": str(source),
            "case": str(args.case_path.resolve()),
            "output_root": str(args.output_root.absolute()),
            "typed_adapters": args.typed_adapters == "1",
            "compact_inspection": args.compact_inspection == "1",
            "runtime": RUNTIME,
        }, indent=2))
        return 0
    _validate_execute_output_root(args.output_root)
    result = run_probe(
        source, args.case_path, args.output_root,
        typed_adapters=args.typed_adapters == "1",
        compact_inspection=args.compact_inspection == "1",
        expected_source_seal=args.source_seal,
        expected_source_fingerprint=args.source_fingerprint,
    )
    print(json.dumps({"output": str(args.output_root),
                      "complete": result["complete"],
                      "outcome": ((result.get("binding_result") or {})
                                  .get("outcome"))}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
