"""One real Compiler/Critic invocation for an offline isolation diagnostic.

Not a scoring entry point or a replacement scheduler. The caller must charge
extraction, every claim preparation (including failures), and replay separately.
This producer never retries a failed preparation and never exports a verdict.
"""
from __future__ import annotations

import hashlib
import os
import re
import stat
import time
from dataclasses import asdict
from pathlib import Path
from typing import Callable

from experiments.terminal_bench import analyze_model_spans
from verifier import offline
from verifier.contract_replay import (
    MAX_SOURCE_EVENTS, MAX_SOURCE_LOG_BYTES, ReplaySetupError,
    clone_extracted_inputs, compilation_config_sha256, compile_input_digests,
    fresh_graph_snapshot, normalized_compile_inputs, pipeline_sha256,
    replay_identity,
)
from verifier.core.model import CLAIM_FINAL, content_hash
from verifier.engine.contract_bundle import (
    ContractBundleIdentity, contract_record_sha256, passed_contract_entry,
)
from verifier.engine.contract_cache import CompiledFinalContractCache
from verifier.llm.codex_backend import codex_usage_ledger
from verifier.llm.model_spans import model_scope, span_sidecar_path


RECEIPT_SCHEMA = "gv-direct-contract-preparation/v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_RUNTIME = {"model": "gpt-5.6-luna", "effort": "medium", "service_tier": "standard"}
_ROLE_STAGES = {"obligations": "contract_compile", "critic": "contract_critic"}


def _private_bytes(path: Path, cap: int) -> bytes:
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1 or info.st_size > cap:
        raise ReplaySetupError("diagnostic ledger must be private and bounded")
    with path.open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ReplaySetupError("diagnostic ledger grew beyond its size cap")
    return raw


def _event_rows(kernel):
    _private_bytes(Path(kernel.log.path), MAX_SOURCE_LOG_BYTES)
    rows = []
    for event in kernel.log.replay():
        if len(rows) >= MAX_SOURCE_EVENTS:
            raise ReplaySetupError("diagnostic ledger exceeds its event cap")
        rows.append({key: event[key] for key in ("seq", "type", "payload")})
    return rows


def _unchanged_graph(kernel, snapshot):
    ids = [row[0] for row in kernel.store.db.execute(
        "SELECT id FROM claims ORDER BY rowid LIMIT ?", (len(snapshot["claims"]) + 1,))]
    actual = []
    for cid in ids:
        row = kernel.store.get(cid)
        if row.get("basis") is not None or row.get("strength") is not None:
            return False
        actual.append({
            "id": cid, "kind": row["kind"], "text": row["text"],
            "deps": sorted(kernel.store.parents(cid)), "revises": row.get("revises"),
            "initial_status": row["status"],
        })
    return actual == snapshot["claims"]


def _receipt_calls(usage_path, spans_path, claim_id, claim_kind, events, *, expected_context=None):
    """Join complete role lifecycles and all attempts, never by elapsed time."""
    report = analyze_model_spans.analyze(usage_path, spans_path)
    usage, usage_errors = analyze_model_spans._read_jsonl(usage_path)
    spans, span_errors = analyze_model_spans._read_jsonl(spans_path)
    findings = [*usage_errors, *span_errors,
                *report["cost_findings"], *report["attribution_findings"]]
    if not report["cost_complete"]:
        findings.append("incomplete_usage")
    if not report["attribution_complete"]:
        findings.append("incomplete_attribution")
    for row in usage:
        if any(row.get(field) != value for field, value in _RUNTIME.items()):
            findings.append("runtime_mismatch")
        if (row.get("record_type") != "codex_cli_attempt"
                or row.get("schema_version") != 1
                or row.get("attempt_schema_version") != 1):
            findings.append("attempt_schema_mismatch")
        if expected_context is not None and any(
                row.get(key) != value for key, value in expected_context.items()):
            findings.append("usage_context_mismatch")
    role_starts = [row for row in spans if row.get("record_type") == "role_span_started"]
    calls = []
    for event in events:
        if type(event.get("seq")) is not int or event["seq"] < 1:
            findings.append("invalid_event_seq")
        if event["type"] == "decomposition_attacked":
            for key in ("round", "repair_attempt"):
                if key == "round" or key in event["payload"]:
                    value = event["payload"].get(key)
                    if type(value) is not int or value < 0:
                        findings.append("invalid_event_round")
    for start in role_starts:
        role = start.get("role")
        if (role not in _ROLE_STAGES or start.get("stage") != _ROLE_STAGES.get(role)
                or start.get("claim_id") != claim_id
                or start.get("claim_kind") != claim_kind
                or type(start.get("round")) is not int or start["round"] < 0
                or start.get("locus") != ("final_task" if claim_kind == CLAIM_FINAL else "support")):
            findings.append("unexpected_role_scope")
        sid = start.get("role_span_id")
        finishes = [row for row in spans if row.get("record_type") == "role_span_finished"
                    and row.get("role_span_id") == sid]
        model_calls = [row for row in spans if row.get("record_type") == "model_call_started"
                       and row.get("role_span_id") == sid]
        if not model_calls:
            findings.append("role_without_model_call")
        if len(finishes) != 1 or finishes[0].get("status") != "succeeded":
            findings.append("role_did_not_succeed")
        attempts = [item for item in spans if item.get("record_type") == "role_attempt_started"
                    and item.get("role_span_id") == sid]
        if [item.get("role_attempt_index") for item in attempts] not in ([1], [1, 2]):
            findings.append("unexpected_schema_attempt_sequence")
        schema_records = []
        for index, attempt in enumerate(attempts):
            linked = [item for item in model_calls
                      if item.get("role_attempt_id") == attempt.get("role_attempt_id")]
            if len(linked) != 1:
                findings.append("schema_attempt_model_call_count")
                continue
            call_id = linked[0]["call_id"]
            model_end = [item for item in spans if item.get("record_type") == "model_call_finished"
                         and item.get("call_id") == call_id]
            attempt_end = [item for item in spans if item.get("record_type") == "role_attempt_finished"
                           and item.get("role_attempt_id") == attempt.get("role_attempt_id")]
            billed = [item for item in usage if item.get("call_id") == call_id]
            expected_status = "succeeded" if index == len(attempts) - 1 else "error"
            expected_usage_status = "succeeded" if expected_status == "succeeded" else "schema_error"
            if (len(model_end) != 1 or model_end[0].get("status") != expected_status
                    or len(attempt_end) != 1 or attempt_end[0].get("status") != expected_status
                    or not billed or billed[-1].get("attempt_status") != expected_usage_status):
                findings.append("schema_result_status_mismatch")
            schema_records.append({
                "role_attempt_id": attempt.get("role_attempt_id"),
                "role_attempt_index": attempt.get("role_attempt_index"), "call_id": call_id,
                "role_attempt_status": attempt_end[0].get("status") if len(attempt_end) == 1 else None,
                "model_call_status": model_end[0].get("status") if len(model_end) == 1 else None,
            })
        for call in model_calls:
            billed = [item for item in usage if item.get("call_id") == call["call_id"]]
            if (not billed or any(item.get("attempt_status") != "provider_retry" for item in billed[:-1])
                    or billed[-1].get("attempt_status") not in {"succeeded", "schema_error"}):
                findings.append("unexpected_provider_attempt_sequence")
        calls.append({
            "role_span_id": sid, "role": role, "stage": start.get("stage"),
            "round": start.get("round"),
            "terminal_status": finishes[0].get("status") if len(finishes) == 1 else None,
            "call_ids": [row["call_id"] for row in model_calls],
            "schema_attempts": schema_records,
            "successful_call_id": schema_records[-1]["call_id"] if schema_records else None,
            "attempts": [{"attempt_id": row.get("attempt_id"),
                          "attempt_index": row.get("attempt_index"),
                          "call_id": row.get("call_id"),
                          "attempt_status": row.get("attempt_status"),
                          "usage_complete": row.get("usage_complete"),
                          "prompt_sha256": row.get("prompt_sha256"),
                          "row_sha256": content_hash(row)}
                         for row in usage if row.get("call_id") in {
                             item["call_id"] for item in model_calls}],
        })
    reviews = [event for event in events if event["type"] == "decomposition_attacked"
               and "source" not in event["payload"]]
    critic_calls = [call for call in calls if call["role"] == "critic"]
    if len(reviews) != len(critic_calls):
        findings.append("critic_receipt_count_mismatch")
    for event, call in zip(reviews, critic_calls):
        if (event["payload"].get("claim") != claim_id
                or type(event["payload"].get("round")) is not int
                or event["payload"].get("round") != call["round"]
                or call["terminal_status"] != "succeeded"):
            findings.append("critic_receipt_scope_mismatch")
    terminal = None
    if (reviews and critic_calls and reviews[-1]["payload"].get("verdict") == "pass"
            and calls[-1] == critic_calls[-1]):
        event, call = reviews[-1], critic_calls[-1]
        terminal = {"event_seq": event["seq"], "event_sha256": content_hash(event),
                    "round": call["round"], "verdict": "pass",
                    "role_span_id": call["role_span_id"], "call_ids": call["call_ids"],
                    "successful_call_id": call["successful_call_id"]}
    else:
        findings.append("no_terminal_critic_pass")
    if not calls or calls[0]["role"] != "obligations":
        findings.append("no_initial_compiler")
    compilers = [call for call in calls if call["role"] == "obligations"]
    if ([call["round"] for call in compilers] != list(range(len(compilers)))
            or [call["round"] for call in critic_calls] != list(range(len(critic_calls)))
            or len(calls) < 2 or calls[1]["role"] != "critic"
            or any(left["role"] == right["role"] == "critic"
                   for left, right in zip(calls, calls[1:]))
            or any(event["payload"].get("verdict") != "insufficient" for event in reviews[:-1])):
        findings.append("invalid_compile_critic_sequence")
    guards = [event for event in events if event["type"] == "decomposition_attacked"
              and "source" in event["payload"]]
    expected_guards = [(left["round"], sum(item["role"] == "critic" for item in calls[:index + 1]) - 1)
                       for index, (left, right) in enumerate(zip(calls, calls[1:]))
                       if left["role"] == right["role"] == "obligations"]
    actual_guards = [(event["payload"].get("repair_attempt"), event["payload"].get("round"))
                     for event in guards]
    if (actual_guards != expected_guards
            or any(event["payload"].get("source") != "mechanical_append_guard"
                   or event["payload"].get("verdict") != "insufficient" for event in guards)):
        findings.append("invalid_mechanical_guard_sequence")
    return calls, terminal, report, findings


def prepare_contract_entry(
    source: str | Path, destination: str | Path, *, setup_id: str, claim_id: str,
    service_factory: Callable, config: dict,
    expected_identity: ContractBundleIdentity,
    expected_compile_input_sha256: str,
) -> dict:
    """Compile one claim once on its own freshly cloned post-extract ledger.

    ``service_factory(state)`` must build the normal offline, lazy Service,
    with the intended existing roles/backend. The independent identity pins
    come from the selected shared input graph. This is not a paid CLI and
    does not itself establish a complete setup: extraction and other claims
    are deliberately outside this receipt's accounting boundary.

    The returned entry is an UNTRUSTED candidate for a future independently
    validated bundle, not an authorization to attach it or run paid replay.
    A failed/missing Critic pass, mutation, unknown usage, or missing causal
    join yields an inconclusive receipt and no bundle entry. All new files
    remain for diagnosis. Interrupted processes retain their start record
    and whatever terminal attempt rows were captured; absence of a receipt
    cannot be interpreted as zero cost or permission to resample.
    """
    if not isinstance(setup_id, str) or _IDENTIFIER.fullmatch(setup_id) is None:
        raise ReplaySetupError("invalid setup id")
    expected_identity = ContractBundleIdentity.from_value(expected_identity)
    started = time.monotonic_ns()
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    resolved = destination.resolve()
    if os.path.commonpath((source, resolved)) in (str(source), str(resolved)):
        raise ReplaySetupError("source and preparation root must be disjoint")
    if os.path.lexists(destination):
        raise ReplaySetupError("preparation root must be new")
    destination.mkdir(parents=True, exist_ok=False)
    # Operator receipts/usage are siblings, outside the role-readable state.
    clone_extracted_inputs(source, destination / "state")
    state = (destination / "state").resolve(strict=True)
    sc = service_factory(state)
    kernel = sc.kernel
    if Path(kernel.log.path).resolve() != state / "events.jsonl":
        raise ReplaySetupError("factory returned a kernel for another state")
    snapshot = fresh_graph_snapshot(kernel, sc.builder.finals[-1])
    digests = compile_input_digests(kernel, snapshot)
    if claim_id not in digests or digests[claim_id] != expected_compile_input_sha256:
        raise ReplaySetupError("claim compile input differs from operator pin")
    identity = replay_identity(kernel, snapshot, config)
    if identity != expected_identity:
        raise ReplaySetupError("contract identity differs from operator pin")
    if kernel.contract_replay is not None:
        raise ReplaySetupError("preparation may not consume replay contracts")
    # Cold direct receipts only. An ambient populated final cache must never
    # masquerade as a fresh Compiler/Critic call. Warm lineage is not enabled.
    kernel.contract_cache = CompiledFinalContractCache()
    row = kernel.store.get(claim_id)
    before = _event_rows(kernel)
    prefix = _private_bytes(state / "events.jsonl", MAX_SOURCE_LOG_BYTES)
    usage_path = destination / "prepare-usage.jsonl"
    spans_path = Path(span_sidecar_path(str(usage_path)))
    if any(os.path.lexists(path) for path in (usage_path, spans_path)):
        raise ReplaySetupError("preparation usage ledgers must be new")
    provenance = {"schema": RECEIPT_SCHEMA, "setup_id": setup_id,
                  "claim_id": claim_id, "claim_kind": row["kind"],
                  "identity": asdict(identity),
                  "compile_input_sha256": digests[claim_id],
                  "diagnostic_only": True, "mode": "direct_compile",
                  "binding_mode": "trusted-kernel-terminal-pass-control-flow",
                  "accounting_scope": "one_claim_excludes_extraction_and_replay",
                  "started_at_monotonic_ns": started}
    offline._write_json_atomic(str(destination / "prepare-start.json"), provenance)
    payload, obligations, error_class = None, None, None
    with codex_usage_ledger(str(usage_path), {str(state): {"setup_id": setup_id}}):
        try:
            with model_scope(claim_id=claim_id, claim_kind=row["kind"],
                             locus="final_task" if row["kind"] == CLAIM_FINAL else "support"):
                payload = kernel._compile_obligation_payload(row)
                obligations = kernel._materialize_obligations(row, payload, record=False)
        except Exception as error:
            # No outer retry; provider/schema retries inside the unchanged
            # role implementation still run and every attempt stays charged.
            error_class = type(error).__name__
    findings = []
    raw_events_before_audit = _private_bytes(state / "events.jsonl", MAX_SOURCE_LOG_BYTES)
    after = _event_rows(kernel)
    if (after[:len(before)] != before
            or not _private_bytes(state / "events.jsonl", MAX_SOURCE_LOG_BYTES).startswith(prefix)):
        findings.append("ledger_prefix_changed")
    events = after[len(before):]
    stable_events = _private_bytes(state / "events.jsonl", MAX_SOURCE_LOG_BYTES)
    if stable_events != raw_events_before_audit:
        findings.append("events_changed_during_audit")
    allowed = {"decomposition_attacked", "contract_cache_miss", "contract_cache_published"}
    if any(event["type"] not in allowed or event["payload"].get("claim") != claim_id
           for event in events):
        findings.append("unexpected_compile_event")
    if row["kind"] == CLAIM_FINAL:
        cache_events = [event["type"] for event in events if event["type"].startswith("contract_cache_")]
        if cache_events != ["contract_cache_miss", "contract_cache_published"]:
            findings.append("not_a_direct_authority_compile")
        else:
            miss, published = [event["payload"] for event in events
                               if event["type"].startswith("contract_cache_")]
            if (published.get("source_digest") != content_hash(payload)
                    or published.get("key") != miss.get("key")
                    or published.get("authority") != miss.get("authority")
                    or events[0]["type"] != "contract_cache_miss"
                    or events[-1]["type"] != "contract_cache_published"
                    or events[-2]["type"] != "decomposition_attacked"
                    or events[-2]["payload"].get("verdict") != "pass"
                    or miss.get("leader") is not True or miss.get("waited") is not False
                    or published.get("waited") is not False
                    or published.get("after_duplicate_fill") is not False):
                findings.append("authority_publication_mismatch")
    elif any(event["type"].startswith("contract_cache_") for event in events):
        findings.append("unexpected_support_cache")
    elif not events or events[-1]["type"] != "decomposition_attacked" or events[-1]["payload"].get("verdict") != "pass":
        findings.append("support_terminal_event_mismatch")
    captured = {}
    for label, path in (("usage", usage_path), ("spans", spans_path)):
        try:
            captured[label] = _private_bytes(path, analyze_model_spans.MAX_LEDGER_BYTES)
        except (OSError, ReplaySetupError):
            captured[label] = None
            findings.append(f"{label}_ledger_unavailable")
    calls, terminal, report, audit_findings = _receipt_calls(
        usage_path, spans_path, claim_id, row["kind"], events,
        expected_context={"setup_id": setup_id, "state_dir": str(state), "backend": "codex-cli"})
    findings.extend(audit_findings)
    if error_class:
        findings.append("compilation_raised")
    if not payload or not obligations:
        findings.append("empty_or_failed_contract")
    candidate = None
    if not findings:
        try:
            candidate = passed_contract_entry(
                claim_id=claim_id, compile_input_sha256=digests[claim_id],
                critic_receipt_sha256="0" * 64, payload=payload, obligations=obligations)
        except ValueError:
            findings.append("invalid_compiled_payload")
    ledger_hashes = {}
    for label, path in (("usage", usage_path), ("spans", spans_path)):
        try:
            raw = _private_bytes(path, analyze_model_spans.MAX_LEDGER_BYTES)
            if raw != captured[label]:
                findings.append(f"{label}_ledger_changed_during_audit")
            ledger_hashes[label] = hashlib.sha256(raw).hexdigest()
        except (OSError, ReplaySetupError):
            ledger_hashes[label] = None
            findings.append(f"{label}_ledger_unavailable")
    if _private_bytes(state / "events.jsonl", MAX_SOURCE_LOG_BYTES) != stable_events:
        findings.append("events_changed_during_audit")
    # Revalidate after the accounting read too: a source revision during the
    # audit must not be exported under the earlier selected input identity.
    try:
        if (offline.verify_immutable_input_anchor(str(state)) != identity.source_sha256
                or offline.verify_immutable_input_anchor(str(source)) != identity.source_sha256
                or pipeline_sha256() != identity.pipeline_sha256
                or compilation_config_sha256(kernel, config) != identity.config_sha256
                or content_hash(normalized_compile_inputs(kernel, row)) != digests[claim_id]
                or not _unchanged_graph(kernel, snapshot)):
            findings.append("compile_inputs_changed")
    except Exception:
        findings.append("compile_inputs_cannot_revalidate")
    for table in ("evidence", "obligation_edges"):
        if kernel.store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
            findings.append("preparation_contains_verification_state")
    receipt = {
        **provenance, "status": "passed" if not findings else "inconclusive",
        "finished_at_monotonic_ns": time.monotonic_ns(),
        "findings": sorted(set(findings)), "error_class": error_class,
        "event_prefix_sha256": hashlib.sha256(prefix).hexdigest(),
        "event_ledger_sha256": hashlib.sha256(stable_events).hexdigest(),
        "event_range": {"before_seq": before[-1]["seq"] if before else 0,
                        "after_seq": after[-1]["seq"] if after else 0,
                        "events": [{"seq": event["seq"], "type": event["type"],
                                    "sha256": content_hash(event)} for event in events]},
        "calls": calls, "terminal_critic": terminal,
        "ledger_sha256": ledger_hashes,
        "usage_complete": report["cost_complete"],
        "attribution_complete": report["attribution_complete"],
        "payload_sha256": candidate["payload_sha256"] if candidate else None,
        "obligations_sha256": content_hash(candidate["obligations"]) if candidate else None,
    }
    receipt_sha256 = contract_record_sha256(receipt)
    offline._write_json_atomic(str(destination / "prepare-receipt.json"), receipt)
    entry = None
    if not findings:
        entry = {**candidate, "critic_receipt_sha256": receipt_sha256}
    return {"receipt": receipt, "receipt_sha256": receipt_sha256, "entry": entry}
