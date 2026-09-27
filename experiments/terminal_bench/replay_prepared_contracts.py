"""Run one explicit, offline sealed-contract replay diagnostic.

This module is not a scoring entry point.  It clones only a fresh extracted
input boundary, replaces only Compiler/Critic output with an externally pinned
bundle, and runs the ordinary Oracle/executor/Judger certificate path.  It
never writes ``score.json`` and never retries an arm-level failure.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import stat
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from experiments.terminal_bench import analyze_model_spans, shared_setup
from verifier import offline
from verifier.contract_replay import (
    ReplaySetupError, attach_replay_contracts, clone_extracted_inputs,
    load_replay_contracts,
)
from verifier.core.model import content_hash
from verifier.core.store import TRIGGER_LAZY
from verifier.engine.certificate import request_certificate
from verifier.engine.contract.bundle import (
    MAX_CONTRACT_BUNDLE_BYTES, ContractBundleIdentity,
)
from verifier.engine.binding.game import BindingGame
from verifier.engine.binding.method_cache import AcceptedMethodCache
from verifier.llm.codex.usage import codex_usage_ledger
from verifier.llm.model_spans import span_sidecar_path


RESULT_SCHEMA = "gv-offline-contract-replay-diagnostic/v1"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_RUNTIME = {
    "model": "gpt-5.6-luna",
    "effort": "medium",
    "service_tier": "standard",
}


def _identifier(value: Any, name: str) -> str:
    if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
        raise ReplaySetupError(f"invalid {name}")
    return value


def _snapshot_config(value: Any) -> dict[str, Any]:
    """Take one strict-JSON snapshot; later caller mutation is irrelevant."""
    if type(value) is not dict:
        raise ReplaySetupError("replay config must be an object")
    try:
        raw = json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
            allow_nan=False,
        )
        config = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ReplaySetupError("replay config must be strict JSON") from error
    if any(config.get(field) != expected
           for field, expected in _RUNTIME.items()):
        raise ReplaySetupError("replay runtime must be the pinned Luna profile")
    return config


def _empty_metrics() -> dict[str, Any]:
    return {
        "complete": False,
        "cost_complete": False,
        "wall_complete": False,
        "started_attempts": None,
        "uncached_input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "wall_seconds": None,
    }


def _private_ledger_bytes(path: Path) -> bytes:
    """Read one bounded external ledger without following writable aliases."""
    expected = path.lstat()
    if (not stat.S_ISREG(expected.st_mode) or expected.st_nlink != 1
            or expected.st_size > analyze_model_spans.MAX_LEDGER_BYTES):
        raise ReplaySetupError("replay usage ledger is not private and bounded")
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if ((before.st_dev, before.st_ino) != (expected.st_dev, expected.st_ino)
                or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1):
            raise ReplaySetupError("replay usage ledger changed before opening")
        chunks, total = [], 0
        while True:
            block = os.read(
                descriptor,
                min(1024 * 1024,
                    analyze_model_spans.MAX_LEDGER_BYTES + 1 - total),
            )
            if not block:
                break
            chunks.append(block)
            total += len(block)
            if total > analyze_model_spans.MAX_LEDGER_BYTES:
                raise ReplaySetupError("replay usage ledger exceeds its size cap")
        after = os.fstat(descriptor)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field)
               for field in stable):
            raise ReplaySetupError("replay usage ledger changed while reading")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _ledger_snapshots(usage_path: Path, spans_path: Path) \
        -> dict[str, bytes | None]:
    snapshots: dict[str, bytes | None] = {}
    for label, path in (("usage", usage_path), ("spans", spans_path)):
        try:
            snapshots[label] = _private_ledger_bytes(path)
        except (OSError, ReplaySetupError):
            snapshots[label] = None
    return snapshots


def _audit_model_work(
    usage_path: Path,
    spans_path: Path,
    *,
    wall_seconds: float,
    context: Mapping[str, str],
    state: Path | None,
) -> dict[str, Any]:
    """Reuse the shared role/runtime audit and additionally bind arm context."""
    audit = shared_setup.audit_model_work(
        usage_path,
        spans_path,
        scope="replay",
        wall_seconds=wall_seconds,
        expected_runtime=_RUNTIME,
    )
    findings = list(audit.get("findings") or [])
    rows, errors = analyze_model_spans._read_jsonl(usage_path)
    findings.extend(f"usage:{error}" for error in errors)
    expected = {
        **dict(context),
        "state_dir": str(state) if state is not None else None,
        "backend": "codex-cli",
    }
    for index, row in enumerate(rows, 1):
        if any(row.get(key) != value for key, value in expected.items()):
            findings.append(f"usage_context_mismatch:{index}")
    findings = sorted(set(findings))
    if findings:
        audit["complete"] = False
        audit["findings"] = findings
        audit["metrics"] = _empty_metrics()
    return audit


def _service_boundary(service: Any, state: Path) -> tuple[Any, str]:
    """Validate the caller-built Service before attaching trusted contracts."""
    required = ("kernel", "store", "log", "artifacts", "builder", "state_dir")
    if any(not hasattr(service, name) for name in required):
        raise ReplaySetupError("service factory returned an incomplete Service")
    try:
        service_state = Path(service.state_dir).resolve(strict=True)
        log_path = Path(service.log.path).resolve()
        kernel_log = Path(service.kernel.log.path).resolve()
    except (OSError, TypeError, ValueError) as error:
        raise ReplaySetupError("service boundary cannot be resolved") from error
    if (service_state != state or log_path != state / "events.jsonl"
            or kernel_log != log_path
            or service.kernel.store is not service.store
            or service.kernel.log is not service.log
            or service.kernel.artifacts is not service.artifacts):
        raise ReplaySetupError("service factory is bound to another state")
    if service.store.trigger != TRIGGER_LAZY or service.kernel.live_world:
        raise ReplaySetupError("contract replay requires a lazy offline Service")
    if service.kernel.contract_replay is not None:
        raise ReplaySetupError("service factory pre-attached replay state")
    finals = list(service.builder.finals)
    if len(finals) != 1:
        raise ReplaySetupError("replay requires exactly one final claim")
    backend = service.kernel.backend
    backend_cwd = getattr(backend, "cwd", None)
    if backend_cwd is not None:
        try:
            if Path(backend_cwd).resolve(strict=True) != state:
                raise ReplaySetupError("replay backend is bound to another state")
        except (OSError, TypeError, ValueError) as error:
            raise ReplaySetupError("replay backend state cannot be resolved") from error
    for field, expected in _RUNTIME.items():
        if hasattr(backend, field) and getattr(backend, field) != expected:
            raise ReplaySetupError("service backend differs from the pinned Luna profile")
    overrides = getattr(backend, "model_overrides", {})
    if overrides not in ({}, None):
        raise ReplaySetupError("replay backend may not override the pinned model")
    binder = service.kernel.binder
    if (not isinstance(binder, BindingGame)
            or binder.backend is not backend
            or binder.log is not service.log
            or binder.artifacts is not service.artifacts
            or binder.live_world):
        raise ReplaySetupError("replay requires the state-bound offline binding game")
    # The sealed bundle is the diagnostic's only cross-arm reuse.  Keep the
    # configured cache behavior within this arm, but never import a method or
    # a prior Judger signature from setup/control/another candidate.
    method_cache = binder.method_cache
    if method_cache is not None:
        if type(method_cache) is not AcceptedMethodCache:
            raise ReplaySetupError("replay method cache must be verifier-owned")
        binder.method_cache = AcceptedMethodCache(
            max_entries=method_cache.max_entries)
    return service.kernel, finals[0]


def _source_seal(source: Path, expected: str) -> None:
    if offline.verify_immutable_input_anchor(str(source)) != expected:
        raise ReplaySetupError("source seal differs from the operator pin")


def replay_prepared_contracts(
    source: str | Path,
    destination: str | Path,
    *,
    replay_id: str,
    setup_id: str,
    arm: str,
    service_factory: Callable[[Path], Any],
    config: dict[str, Any],
    extracted_rounds: int,
    bundle_bytes: bytes,
    expected_bundle_sha256: str,
    expected_identity: ContractBundleIdentity | Mapping[str, str],
) -> dict[str, Any]:
    """Execute one no-retry adapter-isolation replay arm.

    ``service_factory`` is invoked exactly once, inside the external usage
    ledger, and must return the intended production-shaped offline Service.
    Semantic output and accounting completeness are separate: a deterministic
    Fixture may exercise the full certificate path, but absent Codex usage is
    reported as unknown and the diagnostic remains inconclusive.
    """
    started_ns = time.monotonic_ns()
    replay_id = _identifier(replay_id, "replay id")
    setup_id = _identifier(setup_id, "setup id")
    arm = _identifier(arm, "arm")
    if not callable(service_factory):
        raise ReplaySetupError("service factory must be callable")
    if type(extracted_rounds) is not int or extracted_rounds < 1:
        raise ReplaySetupError("extracted rounds must be a positive exact integer")
    if type(bundle_bytes) is not bytes:
        raise ReplaySetupError("pinned bundle must be bytes")
    raw_bundle = bytes(bundle_bytes)
    if len(raw_bundle) > MAX_CONTRACT_BUNDLE_BYTES:
        raise ReplaySetupError("pinned bundle exceeds its size cap")
    if (not isinstance(expected_bundle_sha256, str)
            or _SHA256.fullmatch(expected_bundle_sha256) is None
            or hashlib.sha256(raw_bundle).hexdigest()
            != expected_bundle_sha256):
        raise ReplaySetupError("bundle bytes differ from the operator pin")
    identity = ContractBundleIdentity.from_value(expected_identity)
    frozen_config = _snapshot_config(config)

    source_argument, destination_argument = Path(source), Path(destination)
    if (not source_argument.is_absolute()
            or not destination_argument.is_absolute()
            or ".." in source_argument.parts
            or ".." in destination_argument.parts):
        raise ReplaySetupError("source and replay root must be absolute normalized paths")
    source = source_argument.resolve(strict=True)
    destination = destination_argument.absolute()
    resolved_destination = destination.resolve()
    try:
        overlaps = os.path.commonpath((source, resolved_destination)) in (
            str(source), str(resolved_destination))
    except ValueError as error:
        raise ReplaySetupError("source and replay root cannot be compared") from error
    if overlaps:
        raise ReplaySetupError("source and replay root must be disjoint")
    if os.path.lexists(destination):
        raise ReplaySetupError("replay root must be new")
    destination.mkdir(parents=True, exist_ok=False)

    context = {"replay_id": replay_id, "setup_id": setup_id, "arm": arm}
    start_record = {
        "schema": RESULT_SCHEMA,
        **context,
        "status": "started",
        "diagnostic_only": True,
        "accounting_scope":
            "replay_only_excludes_extraction_and_contract_preparation",
        "runtime": copy.deepcopy(_RUNTIME),
        "extracted_rounds": extracted_rounds,
        "bundle_sha256": expected_bundle_sha256,
        "contract_identity": identity.as_dict(),
        "wrapper_config_sha256": content_hash(frozen_config),
        "started_at_monotonic_ns": started_ns,
    }
    offline._write_json_atomic(
        str(destination / "replay-start.json"), start_record)

    state: Path | None = None
    service = kernel = provider = certificate = features = None
    findings: list[str] = []
    error_class: str | None = None
    error_detail: str | None = None
    interrupt: BaseException | None = None
    stamp: dict[str, Any] | None = None
    scope_started_ns: int | None = None
    usage_path = destination / "replay-usage.jsonl"
    spans_path = Path(span_sidecar_path(str(usage_path)))
    try:
        stamp = clone_extracted_inputs(source, destination / "state")
        state = (destination / "state").resolve(strict=True)
        if (stamp.get("rounds") != extracted_rounds
                or type(stamp.get("rounds")) is not int):
            raise ReplaySetupError("extracted rounds differ from the cloned stamp")
        if stamp.get("immutable_input_seal") != identity.source_sha256:
            raise ReplaySetupError("cloned source identity differs from the operator pin")
        _source_seal(source, identity.source_sha256)

        scope_started_ns = time.monotonic_ns()
        with codex_usage_ledger(str(usage_path), {str(state): context}):
            service = service_factory(state)
            kernel, final_id = _service_boundary(service, state)
            provider = load_replay_contracts(
                kernel,
                final_id,
                frozen_config,
                raw_bundle,
                expected_sha256=expected_bundle_sha256,
            )
            if (provider.sha256 != expected_bundle_sha256
                    or provider.identity != identity):
                raise ReplaySetupError("loaded bundle differs from external pins")
            attach_replay_contracts(kernel, provider)
            certificate = request_certificate(
                service.store, kernel, service.log, service.artifacts, final_id)
    except BaseException as error:
        # Role-level provider/schema retries remain the unchanged verifier's
        # responsibility.  This coordinator never retries the replay arm.
        error_class = type(error).__name__
        error_detail = str(error)[:300]
        findings.append("replay_execution_failed")
        if not isinstance(error, Exception):
            interrupt = error
    scope_finished_ns = time.monotonic_ns()
    execution_scope_wall_seconds = (
        max(0.0, (scope_finished_ns - scope_started_ns) / 1e9)
        if scope_started_ns is not None else None
    )

    # Evidence may legitimately add statuses, obligation nodes and artifacts.
    # The bound provider rechecks only immutable graph/source/config identity;
    # do not call fresh_graph_snapshot after execution.
    if certificate is not None and provider is not None and kernel is not None:
        try:
            provider.validate_inputs()
            _source_seal(source, identity.source_sha256)
            if (offline._read_regular_json(str(state / "extract.json")) != stamp
                    or hashlib.sha256(raw_bundle).hexdigest()
                    != expected_bundle_sha256):
                raise ReplaySetupError("replay metadata or bundle bytes changed")
            features = offline._features(
                certificate, extracted_rounds, offline.pipeline_fingerprint())
            features.update(offline._final_contract_audit(kernel.log, certificate))
            # Final export guard after all ledger reads used by feature/audit.
            provider.validate_inputs()
            _source_seal(source, identity.source_sha256)
        except BaseException as error:
            features = None
            error_class = error_class or type(error).__name__
            error_detail = error_detail or str(error)[:300]
            findings.append("replay_final_validation_failed")
            if not isinstance(error, Exception) and interrupt is None:
                interrupt = error

    ledger_before_audit = _ledger_snapshots(usage_path, spans_path)
    try:
        cost_report = _audit_model_work(
            usage_path,
            spans_path,
            wall_seconds=max(0.0, (time.monotonic_ns() - started_ns) / 1e9),
            context=context,
            state=state,
        )
    except BaseException as error:
        cost_report = {
            "complete": False,
            "findings": ["accounting_audit_failed"],
            "metrics": _empty_metrics(),
            "span_report": None,
        }
        error_class = error_class or type(error).__name__
        error_detail = error_detail or str(error)[:300]
        if not isinstance(error, Exception) and interrupt is None:
            interrupt = error
    if cost_report.get("complete") is not True:
        findings.append("replay_accounting_incomplete")
    ledger_after_audit = _ledger_snapshots(usage_path, spans_path)
    if ledger_after_audit != ledger_before_audit:
        findings.append("replay_usage_ledgers_changed_during_audit")
        cost_report["complete"] = False
        cost_report["metrics"] = _empty_metrics()
    # Reading the usage ledgers is part of the arm's diagnostic makespan.
    # Revalidate after that read, then charge the resulting complete wall to
    # the same interval reported at the top level.
    try:
        _source_seal(source, identity.source_sha256)
        if provider is not None:
            provider.validate_inputs()
        if state is not None and stamp is not None \
                and offline._read_regular_json(str(state / "extract.json")) != stamp:
            raise ReplaySetupError("replay metadata changed before export")
    except BaseException as error:
        features = None
        error_class = error_class or type(error).__name__
        error_detail = error_detail or str(error)[:300]
        findings.append("replay_final_validation_failed")
        if not isinstance(error, Exception) and interrupt is None:
            interrupt = error
    ledger_before_export = _ledger_snapshots(usage_path, spans_path)
    if ledger_before_export != ledger_after_audit:
        findings.append("replay_usage_ledgers_changed_before_export")
        cost_report["complete"] = False
        cost_report["metrics"] = _empty_metrics()
    try:
        _source_seal(source, identity.source_sha256)
        if provider is not None:
            provider.validate_inputs()
        if state is not None and stamp is not None \
                and offline._read_regular_json(str(state / "extract.json")) != stamp:
            raise ReplaySetupError("replay metadata changed at export")
    except BaseException as error:
        features = None
        error_class = error_class or type(error).__name__
        error_detail = error_detail or str(error)[:300]
        findings.append("replay_final_validation_failed")
        if not isinstance(error, Exception) and interrupt is None:
            interrupt = error
    finished_ns = time.monotonic_ns()
    arm_wall_seconds = max(0.0, (finished_ns - started_ns) / 1e9)
    if cost_report.get("complete") is True:
        cost_report["metrics"]["wall_seconds"] = arm_wall_seconds

    verification_complete = features is not None and not any(
        item in findings for item in (
            "replay_execution_failed", "replay_final_validation_failed"))
    accounting_complete = cost_report.get("complete") is True
    complete = verification_complete and accounting_complete
    result = {
        **start_record,
        "status": "terminal",
        "diagnostic_status": "complete" if complete else "inconclusive",
        "complete": complete,
        "verification_complete": verification_complete,
        "accounting_complete": accounting_complete,
        "findings": sorted(set(findings)),
        "error_class": error_class,
        "error": error_detail,
        "finished_at_monotonic_ns": finished_ns,
        "arm_wall_seconds": arm_wall_seconds,
        "execution_scope_wall_seconds": execution_scope_wall_seconds,
        "usage_path": usage_path.name,
        "spans_path": spans_path.name,
        "state_path": "state" if state is not None else None,
        "source_stamp_sha256": content_hash(stamp) if stamp is not None else None,
        "usage_sha256": (
            hashlib.sha256(ledger_before_export["usage"]).hexdigest()
            if ledger_before_export["usage"] is not None else None),
        "spans_sha256": (
            hashlib.sha256(ledger_before_export["spans"]).hexdigest()
            if ledger_before_export["spans"] is not None else None),
        "certificate": certificate if verification_complete else None,
        "features": features,
        "cost_report": cost_report,
    }
    offline._write_json_atomic(
        str(destination / "replay-result.json"), result)
    if interrupt is not None:
        raise interrupt
    return result
