"""Strict accounting primitives for a shared adapter-isolation setup.

This module is deliberately not an executor.  It validates an operator-owned
setup record and combines already-audited metric vectors without copying a
shared usage row into either arm.  The normal full-E2E runner does not import
or use it.

The one physical setup may be charged analytically to both arms.  Reports keep
that distinct from physical spend, where the setup occurred exactly once.
Unknown usage, missing causal spans, or an invalid terminal record is never
coerced to zero.
"""
from __future__ import annotations

import hashlib
import math
import os
import re
from pathlib import Path
from typing import Any, Mapping, Optional

from experiments.terminal_bench import analyze_model_spans
from experiments.terminal_bench import run_typed_adapter_eval
from verifier.engine.contract_bundle import (
    MAX_CONTRACT_BUNDLE_BYTES,
    ContractBundleIdentity,
    ContractReplayError,
    SealedContractBundle,
)


SETUP_SCHEMA = "gv-terminal-bench-shared-setup/v1"
CHARGE_SCHEMA = "gv-terminal-bench-shared-setup-charge/v1"
PHASES = frozenset({"cold", "warm", "batch"})
COST_FIELDS = (
    "started_attempts", "uncached_input_tokens", "cached_input_tokens",
    "output_tokens",
)
SETUP_ROLE_STAGES = frozenset({
    ("extract", "extract"),
    ("obligations", "contract_compile"),
    ("critic", "contract_critic"),
})
REPLAY_ROLE_STAGES = frozenset({
    ("oracles", "oracle_enumerate"),
    ("oracles", "oracle_repair"),
    ("discharge", "discharge"),
    ("judger", "judger"),
})
_HEX40 = re.compile(r"[0-9a-f]{40}\Z")
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,255}\Z")
_RECORD_FIELDS = frozenset({
    "schema", "setup_id", "status", "comparison", "tier", "phase",
    "rep", "group_key", "job", "tasks", "trials", "source_root",
    "source_commit", "verifier_fingerprint",
    "verifier_fingerprint_after", "source_fingerprint_unchanged",
    "manifest_sha256", "bundle_path", "bundle_sha256",
    "bundle_expected_sha256", "usage_path", "spans_path",
    "started_at_monotonic_ns", "finished_at_monotonic_ns", "wall_seconds",
    "exit_code", "usage_audit_complete", "span_attribution_complete",
    "allowed_model_work_only", "fresh_setup_root", "fresh_process_cache",
    "model", "seats", "effort", "service_tier",
})
_RUNTIME_FIELDS = ("model", "effort", "service_tier")
EXPECTED_IDENTITY_FIELDS = frozenset({
    "comparison", "tier", "phase", "rep", "group_key", "job", "tasks",
    "trials", "source_root", "source_commit", "verifier_fingerprint",
    "manifest_sha256", "model", "seats", "effort", "service_tier",
})


def _exact_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _monotonic_ns(value: Any) -> bool:
    return _exact_int(value) and value.bit_length() <= 128


def _number(value: Any) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool) \
            or value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


def _bundle_bytes(path: Path) -> bytes:
    with path.open("rb") as handle:
        value = handle.read(MAX_CONTRACT_BUNDLE_BYTES + 1)
    if len(value) > MAX_CONTRACT_BUNDLE_BYTES:
        raise ValueError("bundle exceeds its size cap")
    return value


def _private_path(
    output_root: Path, setup_id: str, value: Any, field: str,
) -> tuple[Optional[Path], list[str]]:
    if not isinstance(value, str) or not value or os.path.isabs(value):
        return None, [f"{field}:must_be_relative"]
    try:
        output = output_root.resolve()
    except (OSError, RuntimeError):
        return None, [f"{field}:cannot_resolve_output_root"]
    relative = Path(value)
    if len(relative.parts) < 3 \
            or relative.parts[:2] != ("setups", setup_id) \
            or any(part == ".." for part in relative.parts):
        return None, [f"{field}:wrong_private_setup_prefix"]
    raw_path = output / relative
    try:
        expected_root = (output / "setups" / setup_id).resolve()
        path = raw_path.resolve()
    except (OSError, RuntimeError):
        return None, [f"{field}:cannot_resolve"]
    try:
        inside = os.path.commonpath((expected_root, path)) \
            == os.fspath(expected_root)
    except ValueError:
        inside = False
    if not inside:
        return None, [f"{field}:escapes_private_setup_root"]
    cursor = raw_path
    while cursor != output:
        if cursor.is_symlink():
            return None, [f"{field}:symlink_not_allowed"]
        cursor = cursor.parent
    if not path.is_file():
        return None, [f"{field}:missing_regular_file"]
    try:
        if path.stat().st_nlink != 1:
            return None, [f"{field}:hardlink_not_allowed"]
    except OSError:
        return None, [f"{field}:cannot_stat"]
    return path, []


def _empty_metrics() -> dict[str, Any]:
    return {
        "complete": False,
        "cost_complete": False,
        "wall_complete": False,
        **{field: None for field in COST_FIELDS},
        "wall_seconds": None,
    }


def _complete_metrics(report: Mapping[str, Any], wall_seconds: float) \
        -> dict[str, Any]:
    totals = report["totals"]
    return {
        "complete": True,
        "cost_complete": True,
        "wall_complete": True,
        "started_attempts": totals["started_attempt_count"],
        "uncached_input_tokens":
            totals["tokens"]["uncached_input_tokens"]["known_sum"],
        "cached_input_tokens":
            totals["tokens"]["cached_input_tokens"]["known_sum"],
        "output_tokens": totals["tokens"]["output_tokens"]["known_sum"],
        "wall_seconds": float(wall_seconds),
    }


def _zero_metrics(wall_seconds: float) -> dict[str, Any]:
    return {
        "complete": True,
        "cost_complete": True,
        "wall_complete": True,
        **{field: 0 for field in COST_FIELDS},
        "wall_seconds": float(wall_seconds),
    }


def audit_model_work(
    usage_path: Path,
    spans_path: Path,
    *,
    scope: str,
    wall_seconds: float,
    expected_runtime: Optional[Mapping[str, str]] = None,
    model_calls_expected: bool = True,
) -> dict[str, Any]:
    """Audit model attempts and their causal spans for one isolated scope.

    ``model_calls_expected=False`` is an explicit operator assertion for a
    mechanically settled effect-only replay.  It is accepted only when both
    ledgers are absent or empty.  All ordinary setup/replay work requires a
    nonempty attempt ledger and a complete span join.
    """
    if scope not in {"setup", "replay"}:
        raise ValueError("scope must be 'setup' or 'replay'")
    if not _number(wall_seconds):
        return {
            "complete": False,
            "findings": ["invalid_wall_seconds"],
            "metrics": _empty_metrics(),
            "span_report": None,
        }
    usage_path, spans_path = Path(usage_path), Path(spans_path)
    if not model_calls_expected:
        nonempty = []
        for label, path in (("usage", usage_path), ("spans", spans_path)):
            try:
                if path.exists() and path.stat().st_size:
                    nonempty.append(f"unexpected_{label}_for_zero_model_work")
            except OSError:
                nonempty.append(f"cannot_stat_{label}_ledger")
        return {
            "complete": not nonempty,
            "findings": nonempty,
            "metrics": (_zero_metrics(float(wall_seconds))
                        if not nonempty else _empty_metrics()),
            "span_report": None,
        }

    findings: list[str] = []
    report = analyze_model_spans.analyze(usage_path, spans_path)
    if report.get("cost_complete") is not True:
        findings.extend(
            f"cost:{value}" for value in report.get("cost_findings") or [])
    if report.get("attribution_complete") is not True:
        findings.extend(
            f"span:{value}"
            for value in report.get("attribution_findings") or [])
    if not _exact_int(report.get("call_count")) or report.get("call_count") < 1:
        findings.append("no_model_calls")

    allowed = SETUP_ROLE_STAGES if scope == "setup" else REPLAY_ROLE_STAGES
    observed_pairs = set()
    for index, group in enumerate(report.get("groups") or []):
        pair = (group.get("role"), group.get("stage"))
        observed_pairs.add(pair)
        if group.get("attribution") != "attributed" or pair not in allowed:
            findings.append(f"forbidden_role_stage:{index}")
    if scope == "setup":
        observed_roles = {role for role, _stage in observed_pairs}
        for required in ("extract", "obligations", "critic"):
            if required not in observed_roles:
                findings.append(f"missing_setup_role:{required}")

    runtime = dict(expected_runtime or {})
    if any(key not in _RUNTIME_FIELDS or not isinstance(value, str)
           or not value for key, value in runtime.items()):
        findings.append("invalid_expected_runtime")
    usage_rows, read_errors = analyze_model_spans._read_jsonl(usage_path)
    findings.extend(f"usage:{error}" for error in read_errors)
    for row_no, row in enumerate(usage_rows, 1):
        if (row.get("record_type") != analyze_model_spans.ATTEMPT_RECORD
                or row.get("schema_version") != 1
                or row.get("attempt_schema_version") != 1):
            findings.append(f"usage:non_attempt_schema:{row_no}")
        for field, expected in runtime.items():
            if row.get(field) != expected:
                findings.append(f"runtime_mismatch:{field}:{row_no}")

    complete = not findings
    return {
        "complete": complete,
        "findings": sorted(set(findings)),
        "metrics": (_complete_metrics(report, float(wall_seconds))
                    if complete else _empty_metrics()),
        "span_report": report,
    }


def validate_setup_record(
    record: Mapping[str, Any],
    output_root: Path,
    *,
    expected_bundle_sha256: Optional[str] = None,
    expected_identity: Optional[Mapping[str, Any]] = None,
    expected_contract_identity: Optional[
        ContractBundleIdentity | Mapping[str, str]] = None,
    expected_compile_inputs: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Strictly validate one terminal operator-owned shared-setup record."""
    findings: list[str] = []
    if not isinstance(record, Mapping):
        return {
            "complete": False, "findings": ["record:not_an_object"],
            "metrics": _empty_metrics(), "model_work": None,
        }
    missing = sorted(_RECORD_FIELDS - set(record))
    extra = sorted(set(record) - _RECORD_FIELDS)
    if missing:
        findings.append("record:missing_fields:" + ",".join(missing))
    if extra:
        findings.append("record:unknown_fields:" + ",".join(extra))
    if record.get("schema") != SETUP_SCHEMA:
        findings.append("record:wrong_schema")
    setup_id = record.get("setup_id")
    if not isinstance(setup_id, str) or _IDENTIFIER.fullmatch(setup_id) is None:
        findings.append("record:invalid_setup_id")
        setup_id = "invalid"
    for field in ("tier", "rep", "group_key", "job"):
        value = record.get(field)
        if not isinstance(value, str) or _IDENTIFIER.fullmatch(value) is None:
            findings.append(f"record:invalid_{field}")
    if record.get("status") != "terminal":
        findings.append("record:not_terminal")
    if record.get("comparison") != "adapter-isolation":
        findings.append("record:wrong_comparison")
    if not isinstance(record.get("phase"), str) \
            or record.get("phase") not in PHASES:
        findings.append("record:invalid_phase")
    for field in ("tasks", "trials"):
        value = record.get(field)
        if (not isinstance(value, list) or not value
                or any(not isinstance(item, str) or not item
                       or len(item) > 512 for item in value)
                or value != sorted(set(value))):
            findings.append(f"record:invalid_{field}")
    if record.get("phase") == "cold" and (
            not isinstance(record.get("tasks"), list)
            or not isinstance(record.get("trials"), list)
            or len(record["tasks"]) != 1 or len(record["trials"]) != 1):
        findings.append("record:invalid_cold_group")
    if record.get("phase") == "warm" and (
            not isinstance(record.get("tasks"), list)
            or len(record["tasks"]) != 1):
        findings.append("record:invalid_warm_group")
    if not isinstance(record.get("source_commit"), str) \
            or _HEX40.fullmatch(record["source_commit"]) is None:
        findings.append("record:invalid_source_commit")
    for field in (
            "verifier_fingerprint", "verifier_fingerprint_after",
            "manifest_sha256", "bundle_sha256", "bundle_expected_sha256"):
        value = record.get(field)
        if not isinstance(value, str) or _HEX64.fullmatch(value) is None:
            findings.append(f"record:invalid_{field}")
    if (record.get("source_fingerprint_unchanged") is not True
            or record.get("verifier_fingerprint_after")
            != record.get("verifier_fingerprint")):
        findings.append("record:source_identity_changed")
    if record.get("bundle_sha256") != record.get("bundle_expected_sha256"):
        findings.append("record:bundle_expected_sha_mismatch")
    if (not isinstance(expected_bundle_sha256, str)
            or _HEX64.fullmatch(expected_bundle_sha256) is None):
        findings.append("record:operator_bundle_sha_not_pinned")
    elif record.get("bundle_sha256") != expected_bundle_sha256:
        findings.append("record:unexpected_bundle_sha")
    if not isinstance(expected_identity, Mapping) \
            or set(expected_identity) != EXPECTED_IDENTITY_FIELDS:
        findings.append("record:operator_identity_not_fully_pinned")
    else:
        for field in sorted(EXPECTED_IDENTITY_FIELDS):
            if record.get(field) != expected_identity[field]:
                findings.append(f"record:identity_mismatch:{field}")
    if expected_contract_identity is None or expected_compile_inputs is None:
        findings.append("record:replay_validation_inputs_not_pinned")
    for field in (
            "usage_audit_complete", "span_attribution_complete",
            "allowed_model_work_only", "fresh_setup_root",
            "fresh_process_cache"):
        if record.get(field) is not True:
            findings.append(f"record:{field}_not_true")
    if type(record.get("exit_code")) is not int \
            or record.get("exit_code") != 0:
        findings.append("record:nonzero_exit")
    for field in ("model", "seats", "effort", "service_tier"):
        value = record.get(field)
        if not isinstance(value, str) or not value or len(value) > 128:
            findings.append(f"record:invalid_{field}")

    start = record.get("started_at_monotonic_ns")
    finish = record.get("finished_at_monotonic_ns")
    wall = record.get("wall_seconds")
    wall_valid = (_monotonic_ns(start) and _monotonic_ns(finish)
                  and finish > start
                  and _number(wall))
    if wall_valid:
        measured = (finish - start) / 1_000_000_000
        if abs(float(wall) - measured) > max(0.01, measured * 1e-6):
            wall_valid = False
    if not wall_valid:
        findings.append("record:invalid_monotonic_wall")

    source_root_value = record.get("source_root")
    source_root = None
    if not isinstance(source_root_value, str) or not os.path.isabs(
            source_root_value):
        findings.append("record:invalid_source_root")
    else:
        source_root = Path(source_root_value)
        if (not source_root.is_dir() or source_root.is_symlink()
                or source_root.resolve() != source_root):
            findings.append("record:source_root_not_stable")
        else:
            try:
                actual_fingerprint = \
                    run_typed_adapter_eval._source_fingerprint(source_root)
            except (OSError, RuntimeError, ValueError):
                findings.append("record:cannot_fingerprint_source")
            else:
                if actual_fingerprint != record.get("verifier_fingerprint"):
                    findings.append("record:source_fingerprint_mismatch")

    paths: dict[str, Path] = {}
    for field in ("bundle_path", "usage_path", "spans_path"):
        path, path_errors = _private_path(
            Path(output_root), setup_id, record.get(field), field)
        findings.extend(path_errors)
        if path is not None:
            paths[field] = path
    if len(set(paths.values())) != len(paths):
        findings.append("record:private_paths_reused")
    if "usage_path" in paths and "spans_path" in paths:
        expected_spans = Path(str(paths["usage_path"]) + ".spans.jsonl")
        if paths["spans_path"] != expected_spans:
            findings.append("record:spans_not_adjacent_to_usage")
    replay_bundle = None
    if "bundle_path" in paths:
        try:
            raw_bundle = _bundle_bytes(paths["bundle_path"])
            if hashlib.sha256(raw_bundle).hexdigest() \
                    != record.get("bundle_sha256"):
                findings.append("record:bundle_content_sha_mismatch")
            elif (expected_contract_identity is not None
                  and expected_compile_inputs is not None
                  and isinstance(expected_bundle_sha256, str)):
                replay_bundle = SealedContractBundle.from_bytes(
                    raw_bundle,
                    expected_sha256=expected_bundle_sha256,
                    expected_identity=expected_contract_identity,
                    expected_compile_inputs=expected_compile_inputs,
                )
                replay_bundle.require_replayable()
        except (OSError, ValueError, ContractReplayError):
            findings.append("record:bundle_replay_validation_failed")

    model_work = None
    if "usage_path" in paths and "spans_path" in paths and wall_valid:
        model_work = audit_model_work(
            paths["usage_path"], paths["spans_path"], scope="setup",
            wall_seconds=float(wall),
            expected_runtime={field: record[field]
                              for field in _RUNTIME_FIELDS
                              if isinstance(record.get(field), str)},
        )
        if model_work["complete"] is not True:
            findings.extend(
                f"model_work:{value}" for value in model_work["findings"])
        if record.get("usage_audit_complete") \
                is not bool(model_work["span_report"] and
                            model_work["span_report"].get("cost_complete")):
            findings.append("record:usage_audit_claim_mismatch")
        if record.get("span_attribution_complete") \
                is not bool(model_work["span_report"] and
                            model_work["span_report"].get(
                                "attribution_complete")):
            findings.append("record:span_audit_claim_mismatch")
        observed_allowed = not any(
            value.startswith("forbidden_role_stage:")
            for value in model_work["findings"])
        if record.get("allowed_model_work_only") is not observed_allowed:
            findings.append("record:allowed_work_claim_mismatch")

    complete = (not findings and replay_bundle is not None
                and model_work is not None
                and model_work["complete"] is True)
    return {
        "complete": complete,
        "findings": sorted(set(findings)),
        "metrics": (model_work["metrics"] if complete
                    else _empty_metrics()),
        "model_work": model_work,
        "bundle_validation": {
            "complete": replay_bundle is not None,
            "bundle_sha256": (replay_bundle.sha256
                              if replay_bundle is not None else None),
        },
    }


def _validate_metrics(value: Mapping[str, Any]) -> tuple[bool, bool]:
    cost_complete = value.get("cost_complete") is True
    wall_complete = value.get("wall_complete") is True
    if cost_complete and any(not _exact_int(value.get(field))
                             for field in COST_FIELDS):
        cost_complete = False
    if wall_complete and not _number(value.get("wall_seconds")):
        wall_complete = False
    declared_complete = value.get("complete")
    if not isinstance(declared_complete, bool) \
            or declared_complete is not (cost_complete and wall_complete):
        return False, False
    return cost_complete, wall_complete


def charged_metrics(
    common_setup: Mapping[str, Any], private_arm: Mapping[str, Any],
) -> dict[str, Any]:
    """Add one common setup to one private arm without inventing unknowns."""
    common_cost, common_wall = _validate_metrics(common_setup)
    arm_cost, arm_wall = _validate_metrics(private_arm)
    cost_complete = common_cost and arm_cost
    wall_complete = common_wall and arm_wall
    wall_total = None
    if wall_complete:
        try:
            candidate = (float(common_setup["wall_seconds"])
                         + float(private_arm["wall_seconds"]))
        except (OverflowError, ValueError):
            wall_complete = False
        else:
            if _number(candidate):
                wall_total = candidate
            else:
                wall_complete = False
    result = {
        "complete": cost_complete and wall_complete,
        "cost_complete": cost_complete,
        "wall_complete": wall_complete,
        **{field: (common_setup[field] + private_arm[field]
                   if cost_complete else None)
           for field in COST_FIELDS},
        "wall_seconds": wall_total if wall_complete else None,
    }
    return result


def _physical_metrics(*values: Mapping[str, Any]) -> dict[str, Any]:
    if not values:
        raise ValueError("at least one metric vector is required")
    result = dict(values[0])
    for value in values[1:]:
        result = charged_metrics(result, value)
    return result


def charge_pair(
    common_setup: Mapping[str, Any],
    control_private: Mapping[str, Any],
    treatment_private: Mapping[str, Any],
) -> dict[str, Any]:
    """Report fair per-arm charges and actual one-setup physical spend."""
    control = charged_metrics(common_setup, control_private)
    treatment = charged_metrics(common_setup, treatment_private)
    physical = _physical_metrics(
        common_setup, control_private, treatment_private)
    return {
        "schema": CHARGE_SCHEMA,
        "complete": (control["complete"] and treatment["complete"]
                     and physical["complete"]),
        "charged": {"control": control, "treatment": treatment},
        "physical": physical,
        "common_setup": dict(common_setup),
        "note": (
            "charged arms each include the common setup once; physical spend "
            "includes the one actual setup once and does not copy usage rows"
        ),
    }
