"""Paid diagnostic for one fresh Compiler/Critic -> Oracle planning pass.

This deliberately stops before execution and judgment.  It is useful for
rejecting a bad prompt or adapter-selection change before a full candidate
run, but it is never result-quality or end-to-end performance evidence.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if os.fspath(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, os.fspath(SOURCE_ROOT))

from experiments.terminal_bench import analyze_model_spans
from experiments.terminal_bench.run_typed_adapter_eval import (
    _source_fingerprint,
    _validate_execute_output_root,
)
from verifier import offline
from verifier.contract_replay import clone_sealed_inputs, fresh_graph_snapshot
from verifier.core.events import EventLog
from verifier.core.model import CLAIM_FINAL
from verifier.core.store import TRIGGER_LAZY
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.codex.usage import codex_usage_ledger
from verifier.llm.model_spans import model_scope, span_sidecar_path, submit_with_context
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service


SCHEMA = "gv-compile-oracle-plan-probe/v1"
RUNTIME = {
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "service_tier": "standard",
}
MAX_OBLIGATIONS = 16
_PLAN_EVENT_TYPES = frozenset({"decomposition_attacked", "methods_enumerated"})


def _write(path: Path, value) -> None:
    offline._write_json_atomic(os.fspath(path), value)


def _source_prefix(reference: Path) -> list[dict]:
    """Return the contiguous post-extraction, pre-verification event prefix."""
    prefix = []
    saw_commit = False
    for event in EventLog(os.fspath(reference / "events.jsonl")).replay():
        event_type = event["type"]
        allowed_before_commit = (
            event_type in offline.HOOK_EVENT_TYPES
            or offline.is_offline_final_response(event)
        )
        if event_type in {"commit", "extracted_commit"}:
            saw_commit = True
        elif saw_commit or not allowed_before_commit:
            break
        prefix.append({key: event[key] for key in ("seq", "type", "payload")})
    if not saw_commit or not prefix:
        raise ValueError("reference has no contiguous extracted source prefix")
    if [event["seq"] for event in prefix] != list(range(1, len(prefix) + 1)):
        raise ValueError("reference source prefix has non-contiguous sequence numbers")
    finals = [event for event in prefix
              if event["type"] in {"commit", "extracted_commit"}
              and event["payload"].get("kind") == CLAIM_FINAL]
    if len(finals) != 1:
        raise ValueError("reference source prefix must contain exactly one final")
    return prefix


def _fresh_state(reference: Path, state: Path, expected_seal: str) -> None:
    if offline.verify_immutable_input_anchor(os.fspath(reference)) != expected_seal:
        raise ValueError("reference immutable-input seal differs from operator pin")
    prefix = _source_prefix(reference)
    clone_sealed_inputs(reference, state)
    extract = offline._read_regular_json(os.fspath(reference / "extract.json"))
    if extract.get("immutable_input_seal") != expected_seal:
        raise ValueError("reference extraction is not bound to immutable inputs")
    _write(state / "extract.json", extract)
    log = EventLog(os.fspath(state / "events.jsonl"))
    for expected in prefix:
        actual = log.append(expected["type"], expected["payload"])
        if any(actual[key] != expected[key] for key in ("seq", "type", "payload")):
            raise ValueError("fresh source reconstruction changed an event")
    if offline.verify_immutable_input_anchor(os.fspath(state)) != expected_seal:
        raise ValueError("fresh probe state changed immutable inputs")


def _obligation_document(obligation) -> dict:
    return {
        "id": obligation.id,
        "claim_id": obligation.claim_id,
        "text": obligation.text,
        "required_basis": obligation.required_basis,
        "severity": obligation.severity,
        "clause": obligation.clause,
        "refutable_by_absence": obligation.refutable_by_absence,
        "premise_scope": list(obligation.premise_scope),
    }


def _enumerate_one(kernel, row, obligation) -> dict:
    with model_scope(
        claim_id=row["id"], claim_kind=row["kind"], locus="final_task",
        obligation_id=obligation.id,
    ):
        methods = kernel.binder.oracle.enumerate(row, obligation)
    binding = []
    registry = getattr(kernel.binder.discharger, "registry", None)
    for method in methods:
        if "adapter" not in method:
            binding.append({"adapter": None, "valid": None, "reason": "custom"})
            continue
        try:
            rebound = registry.bind_inputs(method) if registry is not None else None
            valid = rebound == method
            reason = "bound" if valid else "method was not fully materialized"
        except Exception as error:
            valid = False
            reason = type(error).__name__ + ": " + str(error)[:300]
        binding.append({
            "adapter": method.get("adapter"),
            "version": method.get("adapter_version"),
            "valid": valid,
            "reason": reason,
        })
    return {
        "obligation": _obligation_document(obligation),
        "methods": methods,
        "static_adapter_binding": binding,
    }


def run_probe(
    reference: Path,
    output: Path,
    *,
    expected_reference_seal: str,
    expected_source_fingerprint: str,
) -> dict:
    reference = reference.resolve(strict=True)
    output = output.absolute()
    _validate_execute_output_root(output)
    if _source_fingerprint(SOURCE_ROOT) != expected_source_fingerprint:
        raise ValueError("running source differs from operator fingerprint")
    output.mkdir(parents=True, exist_ok=False)
    state = output / "state"
    usage = output / "native-usage.jsonl"
    spans = Path(span_sidecar_path(os.fspath(usage)))
    started = time.monotonic()
    _write(output / "probe-start.json", {
        "schema": SCHEMA,
        "status": "started",
        "diagnostic_only": True,
        "full_e2e_acceptance": False,
        "excluded_stages": ["extract", "discharge", "judger", "certificate"],
        "reference": os.fspath(reference),
        "reference_seal": expected_reference_seal,
        "source_root": os.fspath(SOURCE_ROOT),
        "source_fingerprint": expected_source_fingerprint,
        "runtime": RUNTIME,
    })
    findings = []
    plans = []
    payload = None
    error = None
    try:
        _fresh_state(reference, state, expected_reference_seal)
        backend = CodexRuntime(
            cwd=os.fspath(state), compact_inspection=True, **RUNTIME)

        def factory(sc):
            return build_kernel(
                sc, RUNTIME["model"], backend=backend, critic_rounds=3,
                parallel_obligations=4, parallel_claims=4)

        service = Service(
            os.fspath(state), kernel_factory=factory, trigger=TRIGGER_LAZY,
            live_world=False, critic_rounds=3,
            parallel_obligations=4, parallel_claims=4)
        if service.kernel.binder is None:
            raise ValueError("probe requires the ordinary BindingGame")
        graph = fresh_graph_snapshot(
            service.kernel, service.builder.finals[-1])
        row = service.store.get(graph["final_id"])
        with codex_usage_ledger(
            os.fspath(usage), {os.fspath(state): {"probe": SCHEMA}}
        ):
            with model_scope(
                claim_id=row["id"], claim_kind=row["kind"],
                locus="final_task",
            ):
                payload = service.kernel.contracts.compile_payload(row)
                obligations = service.kernel.contracts.materialize_obligations(
                    row, payload, record=False)
            if not obligations or len(obligations) > MAX_OBLIGATIONS:
                raise ValueError("compiled obligation count is outside probe bounds")
            with ThreadPoolExecutor(
                max_workers=min(4, len(obligations)),
                thread_name_prefix="gv-plan-probe",
            ) as pool:
                futures = [submit_with_context(
                    pool, _enumerate_one, service.kernel, row, obligation)
                    for obligation in obligations]
                plans = [future.result() for future in futures]
        new_events = list(service.log.replay())[len(_source_prefix(reference)):]
        unexpected = sorted({event["type"] for event in new_events}
                            - _PLAN_EVENT_TYPES)
        if unexpected:
            findings.append("unexpected_events:" + ",".join(unexpected))
        if sum(event["type"] == "methods_enumerated" for event in new_events) \
                != len(plans):
            findings.append("method_event_count_mismatch")
        if any(item["obligation"]["claim_id"] != row["id"] for item in plans):
            findings.append("plan_claim_mismatch")
        if any(binding["valid"] is False for item in plans
               for binding in item["static_adapter_binding"]):
            findings.append("invalid_static_adapter_binding")
    except BaseException as caught:  # Retain every started call and diagnosis.
        error = type(caught).__name__ + ": " + str(caught)
        findings.append("probe_raised:" + type(caught).__name__)

    analysis = None
    if usage.exists() and spans.exists():
        analysis = analyze_model_spans.analyze(usage, spans)
        if not analysis["cost_complete"]:
            findings.append("usage_incomplete")
        if not analysis["attribution_complete"]:
            findings.append("attribution_incomplete")
        roles = {group.get("role") for group in analysis["groups"]}
        if not roles <= {"obligations", "critic", "oracles"}:
            findings.append("unexpected_model_role")
    else:
        findings.append("usage_or_spans_missing")
    if offline.verify_immutable_input_anchor(os.fspath(reference)) \
            != expected_reference_seal:
        findings.append("reference_seal_changed")
    if _source_fingerprint(SOURCE_ROOT) != expected_source_fingerprint:
        findings.append("source_fingerprint_changed")
    result = {
        "schema": SCHEMA,
        "status": "complete" if not findings else "inconclusive",
        "diagnostic_only": True,
        "full_e2e_acceptance": False,
        "wall_seconds": time.monotonic() - started,
        "findings": sorted(set(findings)),
        "error": error,
        "compiler_payload": payload,
        "plans": plans,
        "model_analysis": analysis,
    }
    _write(output / "probe-result.json", result)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-state", type=Path, required=True)
    parser.add_argument("--reference-seal", required=True)
    parser.add_argument("--source-fingerprint", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    if args.execute and args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    if os.path.lexists(args.output_root):
        parser.error("output-root must be new")
    if not args.execute:
        print(json.dumps({
            "schema": SCHEMA,
            "diagnostic_only": True,
            "full_e2e_acceptance": False,
            "reference_state": os.fspath(args.reference_state.resolve()),
            "output_root": os.fspath(args.output_root.absolute()),
            "runtime": RUNTIME,
        }, indent=2))
        return 0
    result = run_probe(
        args.reference_state,
        args.output_root,
        expected_reference_seal=args.reference_seal,
        expected_source_fingerprint=args.source_fingerprint,
    )
    print(json.dumps({
        "output": os.fspath(args.output_root),
        "status": result["status"],
        "findings": result["findings"],
    }))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
