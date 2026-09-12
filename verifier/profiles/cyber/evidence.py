"""Controller-authenticated execution capabilities, not candidate verdicts.

The controller builds this data from its own spec, candidate leases, artifact
store and event log.  A JSON manifest supplied by the solver is never an input.
The ordinary Effect remains a record.  A capsule additionally identifies what
the host actually captured, at which time, without claiming that stdout is
true, that the remote target build is known, or that a current replay occurred.
"""
from __future__ import annotations

import base64
import json
import math
import os
from dataclasses import asdict, dataclass
from typing import Any, Callable, Iterable

from ...core.events import EventLog
from ...core.model import content_hash
from .candidates import _read_fd, _regular_fd
from .evaluator import EvaluatorSpec

MANIFEST_SCHEMA = "gv-cyber-execution-capabilities-v1"
CAPSULE_SCHEMA = "gv-cyber-authenticated-execution-v1"


class EvidenceBindingError(ValueError):
    """A retained execution cannot be authenticated against controller state."""


@dataclass(frozen=True)
class SeatCapabilities:
    """Actual host-selected seat permissions; missing capabilities fail closed."""

    read_retained_artifacts: bool = False
    read_candidate_blobs: bool = False
    execute_local_checks: bool = False
    invoke_controller_evaluator: bool = False

    def __post_init__(self) -> None:
        if any(type(value) is not bool for value in asdict(self).values()):
            raise EvidenceBindingError("seat capabilities must be host-supplied booleans")


def _digest(value: Any) -> bool:
    return (isinstance(value, str) and len(value) == 64
            and all(char in "0123456789abcdef" for char in value))


def _timestamp(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value > 0


def _host_event_payload(payload: Any) -> bool:
    # Legacy host captures predate these host-only fields. Explicit solver or
    # foreign authority metadata must never be normalized into a host capsule.
    return (isinstance(payload, dict)
            and payload.get("origin") in (None, "verifier_controller")
            and payload.get("authority_scope") in (None, "host_evaluator_execution"))


def _inventory(spec: EvaluatorSpec | None, capabilities: SeatCapabilities) -> dict[str, Any]:
    evaluator = None
    if spec is not None:
        identity = spec.identity
        evaluator = {
            "evaluator_id": spec.id,
            "environment": spec.environment,
            "authority_scope": spec.authority_scope,
            "configuration_revision": spec.id,
            "entrypoint": {"path": spec.argv[0],
                           "sha256": identity["executable_sha256"]},
            "dependency_pins": identity["pinned_files"],
            # An adapter/interpreter/config pin is not a remote target build pin.
            "target_build_identity": None,
        }
    return {
        "seat": asdict(capabilities),
        "evaluator": evaluator,
        "retained_execution": {
            "capture_kind": "controller_authenticated_host_execution",
            "effect_basis": "record",
            "automatic_candidate_verdict": False,
            "time_scope": "the captured execution, not the current world",
            "potential_capture_scope": ["exact_candidate_presented_to_pinned_evaluator",
                                        "evaluator_process_result", "captured_output_bytes"],
            "unestablished_scope": ["truth_of_output_content", "remote_target_build",
                                    "current_target_state", "task_completion"],
        },
    }


def _capsule(*, spec: EvaluatorSpec, observation: dict[str, Any],
             artifact_root: str,
             candidate_provider: Callable[[str], dict[str, Any]],
             artifact_provider: Callable[[str], dict[str, Any]],
             event_provider: Callable[[int], dict[str, Any] | None],
             tool_call_provider: Callable[[str], dict[str, Any] | None],
             effect_resolver: Callable[[str], str]) -> dict[str, Any]:
    if not isinstance(observation, dict) or not _digest(observation.get("candidate_id")):
        raise EvidenceBindingError("invalid controller observation identity")
    candidate = candidate_provider(observation["candidate_id"])
    if (not isinstance(candidate, dict)
            or candidate.get("candidate_id") != observation["candidate_id"]
            or candidate.get("sha256") != observation["candidate_id"]
            or type(candidate.get("size")) is not int or candidate["size"] < 0
            or not isinstance(candidate.get("blob_path"), str)
            or not os.path.isabs(candidate["blob_path"])):
        raise EvidenceBindingError("invalid authenticated candidate identity")
    artifact_id = observation.get("artifact_id")
    if not _digest(artifact_id):
        raise EvidenceBindingError("invalid execution artifact identity")
    artifact = artifact_provider(artifact_id)
    if not isinstance(artifact, dict) or content_hash(artifact) != artifact_id:
        raise EvidenceBindingError("execution artifact changed")
    if (artifact.get("schema") != "cyber-evaluator-observation-v1"
            or not isinstance(artifact.get("execution_id"), str)
            or not artifact["execution_id"]
            or artifact.get("candidate_id") != candidate["candidate_id"]
            or artifact.get("candidate_sha256") != candidate["sha256"]
            or type(artifact.get("candidate_size")) is not int
            or artifact.get("candidate_size") != candidate["size"]
            or artifact.get("evaluator_id") != spec.id
            or any(artifact.get(key) != observation.get(key) for key in
                   ("evaluator_id", "environment", "authority_scope"))
            or artifact.get("environment") != spec.environment
            or artifact.get("authority_scope") != spec.authority_scope):
        raise EvidenceBindingError("execution has a different subject or authority context")
    seq = observation.get("event_seq")
    if type(seq) is not int or seq <= 0:
        raise EvidenceBindingError("invalid execution event identity")
    event = event_provider(seq)
    payload = event.get("payload") if isinstance(event, dict) else None
    if (not isinstance(event, dict) or event.get("seq") != seq
            or event.get("type") != "tool_result"
            or not _host_event_payload(payload) or payload.get("tool") != "cyber_evaluator"
            or EventLog.event_digest(event) != observation.get("event_digest")):
        raise EvidenceBindingError("execution is not a bound host evaluator event")
    call_id = payload.get("tool_use_id")
    call = tool_call_provider(call_id) if isinstance(call_id, str) and call_id else None
    call_payload = call.get("payload") if isinstance(call, dict) else None
    if (not isinstance(call, dict) or type(call.get("seq")) is not int
            or not 0 < call["seq"] < seq or call.get("type") != "tool_call"
            or not _host_event_payload(call_payload) or call_payload.get("tool") != "cyber_evaluator"
            or call_payload.get("tool_use_id") != call_id
            or call_payload.get("input") != {"candidate_id": candidate["candidate_id"],
                                             "evaluator_id": spec.id}
            or not _timestamp(call.get("ts"))):
        raise EvidenceBindingError("execution invocation does not bind the candidate and evaluator")
    output = payload.get("output")
    path = observation.get("artifact_path")
    expected_path = os.path.join(os.path.abspath(artifact_root), artifact_id + ".json")
    if (not isinstance(path, str) or path != expected_path
            or not isinstance(output, dict) or output.get("artifact_id") != artifact_id
            or output.get("artifact_path") != path
            or {key: value for key, value in output.items()
                if key not in ("artifact_id", "artifact_path")} != artifact):
        raise EvidenceBindingError("execution event does not bind the complete raw artifact")
    # The path exposed to signers must denote the same real controller file,
    # not an equivalent-looking basename, alias, symlink, FIFO or JSON mirror.
    try:
        fd = _regular_fd(os.path.sep, expected_path)
        try:
            on_disk = json.loads(_read_fd(fd).decode("utf-8"))
        finally:
            os.close(fd)
    except (OSError, ValueError, UnicodeDecodeError) as exc:
        raise EvidenceBindingError("execution capture is not a safe authoritative file") from exc
    if on_disk != artifact or content_hash(on_disk) != artifact_id:
        raise EvidenceBindingError("execution capture file differs from its authoritative identity")
    effect_ref = f"effect:{seq}"
    if (observation.get("effect_ref") != effect_ref
            or effect_resolver(effect_ref) != observation.get("effect_claim_id")):
        raise EvidenceBindingError("execution effect binding changed")
    completed = event.get("ts")
    if not _timestamp(completed):
        raise EvidenceBindingError("execution event has no authenticated timestamp")
    if call["ts"] > completed:
        raise EvidenceBindingError("execution invocation follows its result")
    started_unix, completed_unix = artifact.get("started_unix"), artifact.get("completed_unix")
    if started_unix is None and completed_unix is None:
        time_binding = {"started_unix": None, "completed_unix": completed,
                        "precision": "host_completion_event_only"}
    elif (not _timestamp(started_unix) or not _timestamp(completed_unix)
          or started_unix < call["ts"]
          or completed_unix < started_unix or completed_unix > completed):
        raise EvidenceBindingError("invalid host execution time interval")
    else:
        time_binding = {"started_unix": started_unix, "completed_unix": completed_unix,
                        "precision": "host_execution_interval"}
    if ((artifact.get("exit_code") is not None and type(artifact["exit_code"]) is not int)
            or type(artifact.get("timed_out")) is not bool
            or (artifact.get("execution_error") is not None
                and not isinstance(artifact["execution_error"], str))
            or any(artifact.get(key) is not None and type(artifact[key]) is not bool
                   for key in ("process_started", "process_completed"))
            or (artifact.get("process_completed") is True
                and artifact.get("process_started") is not True)):
        raise EvidenceBindingError("invalid execution process metadata")
    raw_streams = {}
    for name in ("stdout", "stderr"):
        encoded = artifact.get(name + "_base64")
        observed = artifact.get(name + "_bytes_observed")
        truncated = artifact.get(name + "_truncated")
        if (not isinstance(encoded, str) or not isinstance(artifact.get(name), str)
                or type(observed) is not int or observed < 0 or type(truncated) is not bool):
            raise EvidenceBindingError("invalid captured stream metadata")
        try:
            raw = base64.b64decode(encoded, validate=True)
        except (ValueError, UnicodeEncodeError) as exc:
            raise EvidenceBindingError("invalid captured stream encoding") from exc
        if (raw.decode("utf-8", errors="replace") != artifact[name]
                or observed < len(raw) or (not truncated and observed != len(raw))):
            raise EvidenceBindingError("captured stream bytes differ from their metadata")
        raw_streams[name] = {"truncated": truncated, "bytes_observed": observed,
                             "bytes_retained": len(raw), "encoding": "base64"}
    captured_host_scope = []
    if artifact.get("process_started") is True:
        captured_host_scope.extend(["exact_candidate_presented_to_pinned_evaluator",
                                    "evaluator_process_started", "captured_output_bytes"])
    if artifact.get("process_completed") is True:
        captured_host_scope.append("evaluator_process_completed")
    return {
        "schema": CAPSULE_SCHEMA,
        "execution_id": artifact["execution_id"],
        "subject": {key: candidate[key] for key in ("candidate_id", "sha256", "size", "blob_path")},
        "source": {"artifact_id": artifact_id, "artifact_path": path,
                   "event_seq": seq, "event_digest": observation["event_digest"],
                   "tool_use_id": call_id, "call_event_seq": call["seq"],
                   "call_event_digest": EventLog.event_digest(call),
                   "effect_ref": effect_ref, "effect_claim_id": observation["effect_claim_id"],
                   "effect_basis": "record"},
        "evaluator_id": spec.id, "configuration_revision": spec.id,
        "environment": spec.environment, "authority_scope": spec.authority_scope,
        "target_build_identity": None,
        "time_binding": time_binding,
        "captured_host_scope": captured_host_scope,
        "raw_streams": raw_streams,
        # These are observations, not parsed success signals. No score or
        # semantic candidate outcome is derived from the process result.
        "process": {"exit_code": artifact.get("exit_code"),
                    "timed_out": artifact.get("timed_out"),
                    "started": artifact.get("process_started"),
                    "completed": artifact.get("process_completed"),
                    "execution_error": artifact.get("execution_error")},
    }


def build_evidence_manifest(*, evaluator_spec: EvaluatorSpec | None,
                            capabilities: SeatCapabilities,
                            observations: Iterable[dict[str, Any]] = (),
                            artifact_root: str | None = None,
                            candidate_provider=None, artifact_provider=None,
                            event_provider=None, tool_call_provider=None, effect_resolver=None,
                            answer_blind: bool = False) -> dict[str, Any]:
    """Build a data-only capability map and authenticated execution capsules.

    The Service must select observations from its own ledger and the claim's
    real transitive Effect dependency closure (or its validated final binding).
    The candidate callback must return a leased/current-byte-validated subject;
    the artifact callback must use the controller's authoritative store. Inputs
    and emitted JSON are not a new trust anchor. Signers still audit provenance,
    time, completeness and whether an observation entails their exact contract.
    Answer-blind seats receive no candidate, event, time, output or result data.
    """
    if not isinstance(capabilities, SeatCapabilities) or type(answer_blind) is not bool:
        raise EvidenceBindingError("invalid host capability configuration")
    inventory = _inventory(evaluator_spec, capabilities)
    manifest = {"schema": MANIFEST_SCHEMA, "inventory": inventory,
                "capability_revision": content_hash({"schema": MANIFEST_SCHEMA,
                                                     "inventory": inventory})}
    if answer_blind:
        return manifest
    capsules = []
    seen = set()
    for observation in observations:
        if (evaluator_spec is None or not isinstance(artifact_root, str)
                or not os.path.isabs(artifact_root)
                or not all(callable(callback) for callback in
                           (candidate_provider, artifact_provider,
                            event_provider, tool_call_provider, effect_resolver))):
            raise EvidenceBindingError("execution authentication callbacks are required")
        capsule = _capsule(spec=evaluator_spec, observation=observation,
                           artifact_root=artifact_root,
                           candidate_provider=candidate_provider,
                           artifact_provider=artifact_provider,
                           event_provider=event_provider, tool_call_provider=tool_call_provider,
                           effect_resolver=effect_resolver)
        identity = (capsule["source"]["event_seq"], capsule["source"]["artifact_id"])
        if identity not in seen:
            capsules.append(capsule)
            seen.add(identity)
    manifest["executions"] = capsules
    return manifest
