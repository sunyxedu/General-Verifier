"""Freeze completed Harbor Codex trials into immutable candidate seals.

The adapter contains no benchmark-task logic.  It cross-checks Harbor's job
and trial config, lock, and result records; binds the completed trajectory and
successfully captured artifact world; and keeps held-out reward only in the
private spec/seal files.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import re
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from experiments.common import candidate_seal
from experiments.common import harbor_artifacts


ADAPTER_SCHEMA = "harbor-rec-codex-candidate/v1"
WORLD_SCHEMA = "harbor-captured-world/v1"
BUILTIN_CODEX_IMPLEMENTATION = "harbor.agents.installed.codex:Codex"
# Import paths are executable configuration, so accepting arbitrary values
# merely because the class name contains "Codex" would weaken the provenance
# check.  This is an intentionally small allowlist of implementations whose
# runtime contract is the logical Harbor ``codex`` agent.
CODEX_IMPLEMENTATIONS = frozenset({
    BUILTIN_CODEX_IMPLEMENTATION,
    "codex_agent:TieredCodex",
    "experiments.terminal_bench.codex_agent:TieredCodex",
})

# These failures are deterministic solver/model output boundaries rather than
# orchestration failures.  Harbor deliberately continues to the official
# verifier after them, so a fully evidenced attempt is a legitimate sample.
# Transient API, authentication, network, setup and timeout failures are not
# listed and therefore remain excluded.
AGENT_OUTCOME_EXCEPTION_TYPES = frozenset({
    "AgentSafetyRefusalError",
    "ContextWindowExceededError",
    "OutputTokenExceededError",
})
RUNTIME_SNAPSHOT_SOURCE = "/gv-runtime-snapshot"
RUNTIME_SNAPSHOT_SCHEMA = "gv-post-run-runtime-snapshot/v1"
RUNTIME_SNAPSHOT_BOUNDARY = (
    "after solver return, before official verifier"
)
RUNTIME_SNAPSHOT_SHELL = "posix-sh-fallback"

# Codex persists provider failures in the terminal turn event.  Prefer its
# structured classification, while retaining the exact Harbor error patterns
# for rollout versions which predate ``codex_error_info``.
_CODEX_EXCEPTION_INFO = {
    "AgentSafetyRefusalError": frozenset({
        "cyber_policy", "misalignment_policy_violation",
    }),
    "ContextWindowExceededError": frozenset({"context_window_exceeded"}),
    "OutputTokenExceededError": frozenset({"session_budget_exceeded"}),
}
_CODEX_EXCEPTION_PATTERNS = {
    "AgentSafetyRefusalError": re.compile(
        r"safety measures that flagged|Cyber Verification Program|"
        r"flagged for possible cybersecurity risk|Request blocked|"
        r"Output blocked by content filtering policy",
        re.IGNORECASE,
    ),
    "ContextWindowExceededError": re.compile(
        r"input token count exceeds the maximum number of tokens",
        re.IGNORECASE,
    ),
    "OutputTokenExceededError": re.compile(
        r"response exceeded .+ output token maximum",
        re.IGNORECASE,
    ),
}


@dataclass(frozen=True)
class JobContext:
    directory: Path
    config: dict[str, Any]
    lock: dict[str, Any]
    input_digests: dict[Path, str]


@dataclass(frozen=True)
class Candidate:
    spec: dict[str, Any]
    seal: dict[str, Any]


@dataclass(frozen=True)
class TrialMetadata:
    """Payload-independent Harbor evidence used for incremental discovery."""

    directory: Path
    spec: dict[str, Any]
    source_identity: dict[str, Any]
    implementation: str
    input_digests: dict[Path, str]

    @property
    def source_id(self) -> str:
        return candidate_seal.candidate_source_id(self.source_identity)


@dataclass(frozen=True)
class FrozenCandidate:
    spec: dict[str, Any]
    seal: dict[str, Any]
    view: dict[str, Any]


def _json_object(path: Path) -> tuple[dict[str, Any], str]:
    before = candidate_seal.file_digest(path)
    try:
        value = json.loads(path.read_bytes())
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise candidate_seal.SealError(f"invalid JSON object {path}: {error}") \
            from error
    after = candidate_seal.file_digest(path)
    if before != after:
        raise candidate_seal.SealError(f"file changed while reading: {path}")
    if not isinstance(value, dict):
        raise candidate_seal.SealError(f"expected a JSON object: {path}")
    return value, before


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise candidate_seal.SealError(f"{field} must be an object")
    return value


def _same_text(field: str, values: Iterable[Any]) -> str:
    items = list(values)
    if not items or any(not isinstance(value, str) or not value.strip()
                        for value in items):
        raise candidate_seal.SealError(f"{field} is missing")
    normalized = [value.strip() for value in items]
    if len(set(normalized)) != 1:
        raise candidate_seal.SealError(
            f"{field} disagrees across Harbor config/lock/result")
    return normalized[0]


def _sha256_ref(value: Any, field: str) -> str:
    if isinstance(value, str) and value.startswith("sha256:"):
        value = value.removeprefix("sha256:")
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise candidate_seal.SealError(f"{field} is not a SHA-256 digest")
    return value


def _same_digest(field: str, values: Iterable[Any]) -> str:
    digests = [_sha256_ref(value, field) for value in values]
    if not digests:
        raise candidate_seal.SealError(f"{field} is missing")
    if len(set(digests)) != 1:
        raise candidate_seal.SealError(
            f"{field} disagrees across Harbor config/lock/result")
    return digests[0]


def _finite_reward(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise candidate_seal.SealError("completed trial has no numeric reward")
    result = float(value)
    if not math.isfinite(result):
        raise candidate_seal.SealError("completed trial reward is not finite")
    return result


def _optional_text(value: Any, field: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise candidate_seal.SealError(f"{field} must be null or non-empty text")
    return value.strip()


def _codex_implementation(agent: Mapping[str, Any], field: str) -> str:
    """Resolve Harbor's named/import-path agent selector without rewriting it."""
    name = _optional_text(agent.get("name"), f"{field}.name")
    import_path = _optional_text(
        agent.get("import_path"), f"{field}.import_path")

    if name == "codex":
        # AgentFactory gives a registered name precedence.  Reject an ignored
        # import_path instead of claiming that the configured subclass ran.
        if import_path is not None:
            raise candidate_seal.SealError(
                f"{field} ambiguously sets codex name and import_path")
        implementation = BUILTIN_CODEX_IMPLEMENTATION
    elif import_path is None and name is not None and ":" in name:
        # Harbor's unified --agent flag stores custom imports in ``name`` and
        # resolves them as import paths at runtime.
        implementation = name
    elif name is None and import_path is not None:
        implementation = import_path
    else:
        configured = import_path or name
        raise candidate_seal.SealError(
            f"adapter accepts only an allowlisted Harbor Codex implementation, "
            f"got {configured!r}")

    if implementation not in CODEX_IMPLEMENTATIONS:
        raise candidate_seal.SealError(
            "adapter accepts only an allowlisted Harbor Codex implementation, "
            f"got {implementation}")
    return implementation


def _solver_service_tier(kwargs: Mapping[str, Any], field: str) -> str:
    """Normalize the two coupled TieredCodex flags, failing on mixed modes."""
    raw_tier = kwargs.get("service_tier")
    raw_fast_mode = kwargs.get("fast_mode")
    if raw_tier in (None, "default") and raw_fast_mode in (None, "disable"):
        return "standard"
    if raw_tier == "fast" and raw_fast_mode == "enable":
        return "fast"
    raise candidate_seal.SealError(
        f"{field} has inconsistent service_tier={raw_tier!r} and "
        f"fast_mode={raw_fast_mode!r}")


def _agent_outcome_exception(result: Mapping[str, Any]) -> str | None:
    value = result.get("exception_info")
    if value is None:
        return None
    exception = _mapping(value, "result.exception_info")
    exception_type = _optional_text(
        exception.get("exception_type"),
        "result.exception_info.exception_type",
    )
    if exception_type not in AGENT_OUTCOME_EXCEPTION_TYPES:
        raise candidate_seal.SealError(
            f"trial exception is not a sealable agent outcome: {exception_type}")
    return exception_type


def _stable_file_bytes(path: Path, field: str) -> tuple[bytes, str]:
    digest = candidate_seal.file_digest(path)
    try:
        value = path.read_bytes()
    except OSError as error:
        raise candidate_seal.SealError(f"cannot read {field} {path}: {error}") \
            from error
    if digest != candidate_seal.file_digest(path):
        raise candidate_seal.SealError(f"{field} changed while reading: {path}")
    return value, digest


def _unique_raw_rollout(trial_dir: Path) -> Path:
    """Select the one copied Codex rollout without mtime/name guessing."""
    sessions = trial_dir / "agent" / "sessions"
    if sessions.is_symlink() or not sessions.is_dir():
        raise candidate_seal.SealError(
            "agent-outcome exception has no real Codex sessions directory")
    matches: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(sessions, followlinks=False):
        directory = Path(dirpath)
        for name in list(dirnames):
            child = directory / name
            if child.is_symlink():
                dirnames.remove(name)
            elif name.startswith("rollout-") and name.endswith(".jsonl"):
                raise candidate_seal.SealError(
                    f"raw Codex rollout is not a regular file: {child}")
        for name in filenames:
            if name.startswith("rollout-") and name.endswith(".jsonl"):
                matches.append(directory / name)
    if len(matches) != 1:
        raise candidate_seal.SealError(
            "agent-outcome exception requires exactly one raw Codex rollout, "
            f"found {len(matches)}")
    path = matches[0]
    try:
        mode = os.lstat(path).st_mode
    except OSError as error:
        raise candidate_seal.SealError(
            f"cannot inspect raw Codex rollout {path}: {error}") from error
    if not stat.S_ISREG(mode):
        raise candidate_seal.SealError(
            f"raw Codex rollout is not a regular file: {path}")
    return path


def _raw_rollout_events(path: Path) -> tuple[list[dict[str, Any]], str]:
    raw, digest = _stable_file_bytes(path, "raw Codex rollout")
    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except (UnicodeError, json.JSONDecodeError) as error:
            raise candidate_seal.SealError(
                f"invalid raw Codex rollout JSONL {path}:{line_number}: {error}") \
                from error
        if not isinstance(event, dict):
            raise candidate_seal.SealError(
                f"raw Codex rollout row is not an object: {path}:{line_number}")
        events.append(event)
    if not events:
        raise candidate_seal.SealError("raw Codex rollout is empty")
    return events, digest


def _terminal_exception_matches(
    terminal: Mapping[str, Any], exception_type: str,
) -> bool:
    error = terminal.get("error")
    message: str | None = None
    info: str | None = None
    if isinstance(error, dict):
        if isinstance(error.get("message"), str):
            message = error["message"]
        if isinstance(error.get("codex_error_info"), str):
            info = error["codex_error_info"]
    elif isinstance(error, str):
        message = error
    if message is None and isinstance(terminal.get("message"), str):
        message = terminal["message"]
    if info is not None:
        return info in _CODEX_EXCEPTION_INFO[exception_type]
    return bool(message and _CODEX_EXCEPTION_PATTERNS[exception_type].search(message))


def _validate_finished_exception_trajectory(
    path: Path, trial_dir: Path, exception_type: str,
) -> tuple[Path, str]:
    """Bind a completed exceptional ATIF record to its raw Codex session.

    A deterministic immediate refusal can legitimately consume no model tokens,
    so Harbor may serialize ``final_metrics`` as null.  The raw lifecycle is
    authoritative in that case; metrics, when present, remain strictly checked.
    """
    trajectory, _ = _json_object(path)
    schema = trajectory.get("schema_version")
    steps = trajectory.get("steps")
    metrics = trajectory.get("final_metrics")
    session_id = trajectory.get("session_id")
    if (not isinstance(schema, str) or not schema.startswith("ATIF-v")
            or not isinstance(steps, list) or not steps
            or not isinstance(session_id, str) or not session_id.strip()):
        raise candidate_seal.SealError(
            "agent-outcome exception has no finished ATIF trajectory")
    if metrics is not None:
        if not isinstance(metrics, dict):
            raise candidate_seal.SealError(
                "agent-outcome exception has malformed ATIF final_metrics")
        total_steps = metrics.get("total_steps")
        if isinstance(total_steps, bool) or not isinstance(total_steps, int) \
                or total_steps != len(steps):
            raise candidate_seal.SealError(
                "agent-outcome exception has inconsistent ATIF final_metrics")

    rollout_path = _unique_raw_rollout(trial_dir)
    events, rollout_digest = _raw_rollout_events(rollout_path)
    session_meta = [event for event in events
                    if event.get("type") == "session_meta"]
    if len(session_meta) != 1:
        raise candidate_seal.SealError(
            "raw Codex rollout must contain exactly one session_meta event")
    meta_payload = _mapping(
        session_meta[0].get("payload"), "raw session_meta.payload")
    raw_session_id = _same_text(
        "Codex session id", (meta_payload.get("id"), session_id))

    lifecycle: list[tuple[str, dict[str, Any]]] = []
    for event in events:
        if event.get("type") != "event_msg" \
                or not isinstance(event.get("payload"), dict):
            continue
        payload = event["payload"]
        event_type = payload.get("type")
        if event_type == "task_started":
            lifecycle.append(("started", payload))
        elif event_type in ("task_complete", "task_failed"):
            lifecycle.append(("terminal", payload))
    if not lifecycle or len(lifecycle) % 2:
        raise candidate_seal.SealError(
            "raw Codex rollout has an incomplete task lifecycle")
    for index in range(0, len(lifecycle), 2):
        started_kind, started = lifecycle[index]
        terminal_kind, terminal = lifecycle[index + 1]
        if started_kind != "started" or terminal_kind != "terminal":
            raise candidate_seal.SealError(
                "raw Codex rollout task lifecycle is not closed in order")
        _same_text("Codex turn id", (
            started.get("turn_id"), terminal.get("turn_id")))
    terminal = lifecycle[-1][1]
    if not _terminal_exception_matches(terminal, exception_type):
        raise candidate_seal.SealError(
            "raw Codex rollout terminal event does not match "
            f"{exception_type}")
    # Keep the normalized equality value live so a future refactor cannot
    # accidentally replace the explicit session join with mere presence checks.
    assert raw_session_id == session_id.strip()
    return rollout_path, rollout_digest


def _job_context(job_dir: str | os.PathLike[str]) -> JobContext:
    directory = Path(job_dir).expanduser().resolve()
    if directory.is_symlink() or not directory.is_dir():
        raise candidate_seal.SealError(f"expected a Harbor job directory: {directory}")
    config_path = directory / "config.json"
    lock_path = directory / "lock.json"
    config, config_digest = _json_object(config_path)
    lock, lock_digest = _json_object(lock_path)
    trials = lock.get("trials")
    if not isinstance(trials, list):
        raise candidate_seal.SealError("job lock has no trials list")
    return JobContext(
        directory, config, lock,
        {config_path: config_digest, lock_path: lock_digest})


def _validate_runtime_snapshot(capture: harbor_artifacts.CapturedArtifact) -> None:
    root = capture.host_path
    if root.is_symlink() or not root.is_dir():
        raise candidate_seal.SealError(
            "runtime snapshot capture must be a real directory")
    manifest_path = root / "manifest.json"
    manifest, _ = _json_object(manifest_path)
    if manifest.get("schema") != RUNTIME_SNAPSHOT_SCHEMA:
        raise candidate_seal.SealError("runtime snapshot has the wrong schema")
    if manifest.get("status") != "complete":
        raise candidate_seal.SealError("runtime snapshot is not complete")
    if manifest.get("capture_boundary") != RUNTIME_SNAPSHOT_BOUNDARY:
        raise candidate_seal.SealError(
            "runtime snapshot has the wrong capture boundary")
    privacy = _mapping(manifest.get("privacy"), "runtime snapshot privacy")
    if privacy.get("process_environments_captured") is not False:
        raise candidate_seal.SealError(
            "runtime snapshot may not capture process environments")

    returncode_path = root / "collector.returncode"
    returncode, _ = _stable_file_bytes(
        returncode_path, "runtime snapshot collector returncode")
    if returncode != b"0\n":
        raise candidate_seal.SealError(
            "runtime snapshot collector.returncode must be exactly 0")

    if "collector_runtime" not in manifest:
        # The original Python v1 collector predates the discriminator.  Its
        # self-copy is an unambiguous, content-addressed runtime marker.
        collector = _mapping(
            manifest.get("collector"), "runtime snapshot collector")
        if set(collector) != {"path", "bytes", "sha256"} \
                or collector.get("path") != "collector.py":
            raise candidate_seal.SealError(
                "python-v1 runtime snapshot has no complete collector binding")
        byte_count = collector.get("bytes")
        if isinstance(byte_count, bool) or not isinstance(byte_count, int) \
                or byte_count < 0:
            raise candidate_seal.SealError(
                "python-v1 runtime snapshot collector bytes are invalid")
        expected_digest = _sha256_ref(
            collector.get("sha256"), "runtime snapshot collector digest")
        collector_path = root / "collector.py"
        try:
            info = os.lstat(collector_path)
        except OSError as error:
            raise candidate_seal.SealError(
                f"cannot inspect runtime snapshot collector: {error}") from error
        if not stat.S_ISREG(info.st_mode) or info.st_size != byte_count \
                or candidate_seal.file_digest(collector_path) != expected_digest:
            raise candidate_seal.SealError(
                "python-v1 runtime snapshot collector binding does not match")
    elif manifest.get("collector_runtime") != RUNTIME_SNAPSHOT_SHELL:
        raise candidate_seal.SealError(
            "runtime snapshot collector_runtime is unsupported")


def _artifact_snapshot(
    trial_dir: Path, *, require_runtime_snapshot: bool,
) -> tuple[str, str, list[str]]:
    manifest_path = trial_dir / "artifacts" / "manifest.json"
    manifest_digest = candidate_seal.file_digest(manifest_path)
    manifest = harbor_artifacts.load_manifest(trial_dir)
    if manifest is None:
        raise candidate_seal.SealError(
            f"completed trial has no valid artifact manifest: {trial_dir.name}")
    captures = harbor_artifacts.successful_captures(
        trial_dir, manifest=manifest)
    by_source = {capture.source: capture for capture in captures}
    for entry in manifest:
        if not isinstance(entry, dict) or entry.get("status") != "ok":
            continue
        source = entry.get("source")
        if source == "/logs/artifacts":
            continue
        capture = by_source.get(source)
        if capture is None or capture.destination != entry.get("destination"):
            raise candidate_seal.SealError(
                f"invalid or missing status=ok artifact in {trial_dir.name}")

    runtime_capture = by_source.get(RUNTIME_SNAPSHOT_SOURCE)
    if runtime_capture is not None:
        _validate_runtime_snapshot(runtime_capture)
    elif require_runtime_snapshot:
        raise candidate_seal.SealError(
            "TieredCodex candidate has no valid /gv-runtime-snapshot capture")

    roots = []
    for capture in captures:
        host = capture.host_path
        if host.is_symlink():
            kind = "symlink"
            digest = candidate_seal.symlink_digest(host)
        elif host.is_dir():
            kind = "directory"
            digest = candidate_seal.path_digest(host)
        elif host.is_file():
            kind = "file"
            digest = candidate_seal.path_digest(host)
        else:
            raise candidate_seal.SealError(
                f"unsupported captured artifact: {host}")
        roots.append({
            "source": capture.source,
            "kind": kind,
            "sha256": digest,
        })
    world_digest = candidate_seal.stable_digest({
        "schema": WORLD_SCHEMA,
        "roots": roots,
    })
    if manifest_digest != candidate_seal.file_digest(manifest_path):
        raise candidate_seal.SealError(
            f"artifact manifest changed while hashing: {manifest_path}")
    return world_digest, manifest_digest, [row["source"] for row in roots]


def _matching_job_dataset(
    context: JobContext, dataset: str, task: str,
) -> dict[str, Any]:
    datasets = context.config.get("datasets")
    if not isinstance(datasets, list):
        raise candidate_seal.SealError("job config has no datasets list")
    matches = [row for row in datasets if isinstance(row, dict)
               and row.get("name") == dataset]
    if len(matches) != 1:
        raise candidate_seal.SealError(
            f"job config does not uniquely identify dataset {dataset}")
    names = matches[0].get("task_names")
    if isinstance(names, list) and task not in names:
        raise candidate_seal.SealError(
            f"task {task} is not declared by the job dataset")
    return matches[0]


def _trial_metadata(context: JobContext, trial_dir: Path) -> TrialMetadata:
    """Validate one completed trial without opening trajectory/world payloads."""
    config_path = trial_dir / "config.json"
    lock_path = trial_dir / "lock.json"
    result_path = trial_dir / "result.json"
    config, config_hash = _json_object(config_path)
    lock, lock_hash = _json_object(lock_path)
    result, result_hash = _json_object(result_path)

    if not isinstance(result.get("finished_at"), str):
        raise candidate_seal.SealError(f"trial is not finished: {trial_dir.name}")
    agent_exception_type = _agent_outcome_exception(result)
    execution = _mapping(result.get("agent_execution"), "result.agent_execution")
    verifier = _mapping(result.get("verifier"), "result.verifier")
    if not isinstance(execution.get("finished_at"), str) \
            or not isinstance(verifier.get("finished_at"), str):
        raise candidate_seal.SealError(
            f"trial phases are incomplete: {trial_dir.name}")

    config_task = _mapping(config.get("task"), "config.task")
    lock_task = _mapping(lock.get("task"), "lock.task")
    result_task_id = _mapping(result.get("task_id"), "result.task_id")
    result_config = _mapping(result.get("config"), "result.config")
    result_config_task = _mapping(
        result_config.get("task"), "result.config.task")
    task = _same_text("task name", (
        config_task.get("name"), lock_task.get("name"),
        result.get("task_name"),
        f"{result_task_id.get('org')}/{result_task_id.get('name')}",
        result_config_task.get("name"),
    ))
    trial = _same_text("trial name", (
        trial_dir.name, config.get("trial_name"), result.get("trial_name"),
        result_config.get("trial_name"),
    ))
    dataset = _same_text("dataset", (
        config_task.get("source"), lock_task.get("source"),
        result.get("source"), result_config_task.get("source"),
    ))
    dataset_row = _matching_job_dataset(context, dataset, task)
    dataset_digest = _sha256_ref(dataset_row.get("ref"), "dataset digest")

    task_package_digest = _same_digest("task package digest", (
        config_task.get("ref"), lock_task.get("digest"),
        result_task_id.get("ref"), result_config_task.get("ref"),
    ))
    task_digest = _sha256_ref(result.get("task_checksum"), "task checksum")
    checksum_copies = [value for value in (
        config_task.get("checksum"), lock_task.get("checksum"),
        result_task_id.get("checksum"), result_config_task.get("checksum"),
    ) if value is not None]
    if checksum_copies:
        _same_digest("task checksum", (task_digest, *checksum_copies))

    config_agent = _mapping(config.get("agent"), "config.agent")
    lock_agent = _mapping(lock.get("agent"), "lock.agent")
    result_agent = _mapping(result_config.get("agent"), "result.config.agent")
    agent_info = _mapping(result.get("agent_info"), "result.agent_info")
    model_info = _mapping(agent_info.get("model_info"), "result.agent_info.model_info")
    implementations = [
        _codex_implementation(config_agent, "config.agent"),
        _codex_implementation(lock_agent, "lock.agent"),
        _codex_implementation(result_agent, "result.config.agent"),
    ]
    implementation = _same_text("agent implementation", implementations)
    runtime_agent_name = _same_text(
        "runtime agent name", (agent_info.get("name"),))
    if runtime_agent_name != "codex":
        raise candidate_seal.SealError(
            "configured Codex implementation did not report logical runtime "
            f"agent codex, got {runtime_agent_name}")
    model = _same_text("model", (
        config_agent.get("model_name"), lock_agent.get("model_name"),
        result_agent.get("model_name"), model_info.get("name"),
    ))
    config_kwargs = _mapping(config_agent.get("kwargs"), "config.agent.kwargs")
    lock_kwargs = _mapping(lock_agent.get("kwargs"), "lock.agent.kwargs")
    result_kwargs = _mapping(
        result_agent.get("kwargs"), "result.config.agent.kwargs")
    effort = _same_text("reasoning effort", (
        config_kwargs.get("reasoning_effort"),
        lock_kwargs.get("reasoning_effort"),
        result_kwargs.get("reasoning_effort"),
    ))
    service_tier = _same_text("solver service tier", (
        _solver_service_tier(config_kwargs, "config.agent.kwargs"),
        _solver_service_tier(lock_kwargs, "lock.agent.kwargs"),
        _solver_service_tier(result_kwargs, "result.config.agent.kwargs"),
    ))
    if service_tier == "fast" and implementation == BUILTIN_CODEX_IMPLEMENTATION:
        raise candidate_seal.SealError(
            "fast service tier requires the allowlisted TieredCodex implementation")
    job_agents = context.config.get("agents")
    if not isinstance(job_agents, list):
        raise candidate_seal.SealError("job config has no agents list")
    job_matches = []
    for index, row in enumerate(job_agents):
        if not isinstance(row, dict):
            continue
        try:
            job_implementation = _codex_implementation(
                row, f"job.config.agents[{index}]")
            job_kwargs = _mapping(
                row.get("kwargs"), f"job.config.agents[{index}].kwargs")
            job_tier = _solver_service_tier(
                job_kwargs, f"job.config.agents[{index}].kwargs")
        except candidate_seal.SealError:
            continue
        if (row.get("model_name") == model
                and job_kwargs.get("reasoning_effort") == effort
                and job_tier == service_tier):
            job_matches.append(job_implementation)
    if not job_matches:
        raise candidate_seal.SealError(
            "trial solver does not match a job config agent")
    if not any(isinstance(row, dict) and row == lock
               for row in context.lock["trials"]):
        raise candidate_seal.SealError("trial lock is not present in the job lock")

    rewards = _mapping(
        _mapping(result.get("verifier_result"), "result.verifier_result").get(
            "rewards"),
        "result.verifier_result.rewards")
    reward = _finite_reward(rewards.get("reward"))
    spec = {
        "schema": ADAPTER_SCHEMA,
        "dataset": dataset,
        "dataset_digest": dataset_digest,
        "task": task,
        "task_digest": task_digest,
        "task_package_digest": task_package_digest,
        "task_config_path": str(config_path),
        "task_config_digest": config_hash,
        "trial": trial,
        "model": model,
        "effort": effort,
        "service_tier": service_tier,
        "reward": reward,
    }
    if implementation != BUILTIN_CODEX_IMPLEMENTATION:
        # Preserve the exact trial selector in the private audit spec.  It is
        # not exposed to the scorer or substituted into historical records.
        spec["agent_implementation"] = implementation
        spec["agent_config_selector"] = {
            "name": config_agent.get("name"),
            "import_path": config_agent.get("import_path"),
        }
    if agent_exception_type is not None:
        # Termination metadata is useful for audit, but the scorer should rank
        # only the produced trajectory/world, not receive a refusal signal.
        spec["agent_exception_type"] = agent_exception_type
    solver = {"model": model, "effort": effort}
    if service_tier != "standard":
        solver["service_tier"] = service_tier
    source_identity = {
        "schema": candidate_seal.IDENTITY_SCHEMA,
        "state": "completed",
        "dataset": {"name": dataset, "sha256": dataset_digest},
        "task": {
            "name": task,
            "sha256": task_digest,
            "package_sha256": task_package_digest,
            "instruction_sha256": None,
            "config_sha256": config_hash,
        },
        "trial": trial,
        "solver": solver,
    }
    input_digests = {
        config_path: config_hash,
        lock_path: lock_hash,
        result_path: result_hash,
    }
    if any(candidate_seal.file_digest(path) != digest
           for path, digest in input_digests.items()):
        raise candidate_seal.SealError(
            f"trial metadata changed while reading: {trial_dir.name}")
    return TrialMetadata(
        directory=trial_dir,
        spec=spec,
        source_identity=source_identity,
        implementation=implementation,
        input_digests=input_digests,
    )


def _metadata_unchanged(metadata: TrialMetadata) -> None:
    if any(candidate_seal.file_digest(path) != digest
           for path, digest in metadata.input_digests.items()):
        raise candidate_seal.SealError(
            f"trial metadata changed while sealing: {metadata.directory.name}")


def candidate_from_trial(
    context: JobContext, trial_dir: Path, *,
    metadata: TrialMetadata | None = None,
) -> Candidate:
    """Build and race-check one completed rec-codex candidate."""
    metadata = metadata or _trial_metadata(context, trial_dir)
    if metadata.directory != trial_dir:
        raise candidate_seal.SealError("trial metadata belongs to another directory")
    trajectory_path = trial_dir / "agent" / "trajectory.json"
    trajectory_hash = candidate_seal.file_digest(trajectory_path)
    raw_rollout: tuple[Path, str] | None = None
    agent_exception_type = metadata.spec.get("agent_exception_type")
    if isinstance(agent_exception_type, str):
        raw_rollout = _validate_finished_exception_trajectory(
            trajectory_path, trial_dir, agent_exception_type)

    require_runtime = metadata.implementation != BUILTIN_CODEX_IMPLEMENTATION
    world_digest, manifest_digest, captured_roots = _artifact_snapshot(
        trial_dir, require_runtime_snapshot=require_runtime)
    if agent_exception_type is not None and not captured_roots:
        raise candidate_seal.SealError(
            "agent-outcome exception has no successfully captured artifact world")
    manifest_path = trial_dir / "artifacts" / "manifest.json"
    spec = dict(metadata.spec)
    spec.update({
        "trajectory_path": str(trajectory_path),
        "world_digest": world_digest,
        "artifact_manifest_digest": manifest_digest,
        "artifact_manifest_path": str(manifest_path),
        "captured_roots": captured_roots,
    })
    if raw_rollout is not None:
        raw_path, raw_digest = raw_rollout
        spec["raw_trajectory_path"] = str(raw_path)
        spec["raw_trajectory_digest"] = raw_digest
    seal = candidate_seal.seal_from_spec(spec)

    _metadata_unchanged(metadata)
    if candidate_seal.file_digest(trajectory_path) != trajectory_hash:
        raise candidate_seal.SealError(
            f"trial changed while sealing: {trial_dir.name}")
    if raw_rollout is not None \
            and candidate_seal.file_digest(raw_rollout[0]) != raw_rollout[1]:
        raise candidate_seal.SealError(
            f"raw Codex rollout changed while sealing: {trial_dir.name}")
    if _artifact_snapshot(
            trial_dir, require_runtime_snapshot=require_runtime) != (
            world_digest, manifest_digest, captured_roots):
        raise candidate_seal.SealError(
            f"artifact world changed while sealing: {trial_dir.name}")
    return Candidate(spec, seal)


def _frozen_matches_metadata(
    frozen: FrozenCandidate, metadata: TrialMetadata,
) -> None:
    identity = frozen.seal["identity"]
    expected = metadata.source_identity
    if frozen.seal["source_id"] != metadata.source_id \
            or identity.get("dataset") != expected["dataset"] \
            or identity.get("task") != expected["task"] \
            or identity.get("trial") != expected["trial"]:
        raise candidate_seal.SealError(
            "completed trial metadata does not match its frozen source identity")
    if identity.get("solver") != expected["solver"]:
        raise candidate_seal.SealError(
            "completed trial solver does not match its frozen source identity")
    if frozen.seal["evaluation"].get("reward") != metadata.spec["reward"]:
        raise candidate_seal.SealError(
            "completed trial reward does not match its frozen candidate")
    for field in (
        "agent_implementation", "agent_config_selector",
        "agent_exception_type",
    ):
        if frozen.spec.get(field) != metadata.spec.get(field):
            raise candidate_seal.SealError(
                f"completed trial {field} does not match its frozen spec")
    if frozen.spec.get("service_tier", "standard") \
            != metadata.spec.get("service_tier", "standard"):
        raise candidate_seal.SealError(
            "completed trial service tier does not match its frozen spec")
    _metadata_unchanged(metadata)


def _discover_context(
    context: JobContext, frozen: dict[str, FrozenCandidate] | None = None,
) -> tuple[list[Candidate], dict[str, str], list[str]]:
    candidates: list[Candidate] = []
    skipped: dict[str, str] = {}
    reused: list[str] = []
    frozen_trial_configs = {
        _spec_path(candidate.spec, "task_config_path"): candidate
        for candidate in (frozen or {}).values()
    }

    def skip_or_reject_frozen(trial_dir: Path, reason: str) -> None:
        if trial_dir / "config.json" in frozen_trial_configs:
            raise candidate_seal.SealError(
                f"frozen Harbor trial became {reason}: {trial_dir.name}")
        skipped[trial_dir.name] = reason

    for trial_dir in sorted(context.directory.iterdir()):
        if not trial_dir.is_dir() or "__" not in trial_dir.name:
            continue
        required = [trial_dir / name for name in (
            "config.json", "lock.json", "result.json")]
        if not all(path.is_file() and not path.is_symlink() for path in required):
            skip_or_reject_frozen(trial_dir, "incomplete")
            continue
        result, _ = _json_object(trial_dir / "result.json")
        if not result.get("finished_at"):
            skip_or_reject_frozen(trial_dir, "incomplete")
            continue
        if result.get("exception_info") is not None:
            try:
                _agent_outcome_exception(result)
            except candidate_seal.SealError:
                skip_or_reject_frozen(trial_dir, "errored")
                continue
        metadata = _trial_metadata(context, trial_dir)
        prior = frozen.get(metadata.source_id) if frozen is not None else None
        if prior is not None:
            _frozen_matches_metadata(prior, metadata)
            reused.append(metadata.source_id)
            continue
        candidate = candidate_from_trial(
            context, trial_dir, metadata=metadata)
        candidates.append(candidate)
        if frozen is not None:
            frozen[metadata.source_id] = FrozenCandidate(
                candidate.spec,
                candidate.seal,
                candidate_seal.verifier_view(candidate.seal),
            )
    if any(candidate_seal.file_digest(path) != digest
           for path, digest in context.input_digests.items()):
        raise candidate_seal.SealError("job config or lock changed while sealing")
    return candidates, skipped, reused


def discover_completed(
    job_dir: str | os.PathLike[str],
) -> tuple[list[Candidate], dict[str, str], JobContext]:
    """Discover completed trials; leave plainly in-progress dirs unsealed."""
    context = _job_context(job_dir)
    candidates, skipped, _ = _discover_context(context)
    return candidates, skipped, context


def _spec_key(spec: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return spec.get("dataset"), spec.get("task"), spec.get("trial")


def _spec_path(spec: Mapping[str, Any], field: str) -> Path:
    value = spec.get(field)
    if not isinstance(value, str) or not value:
        raise candidate_seal.SealError(f"frozen candidate spec has no {field}")
    return Path(value).expanduser()


def _validate_spec_path_affinity(spec: Mapping[str, Any]) -> None:
    """Check persisted audit paths lexically, without reopening pruned payloads."""
    trial = spec.get("trial")
    config_path = _spec_path(spec, "task_config_path")
    trajectory_path = _spec_path(spec, "trajectory_path")
    manifest_path = _spec_path(spec, "artifact_manifest_path")
    roots = (
        config_path.parent,
        trajectory_path.parent.parent,
        manifest_path.parent.parent,
    )
    if len(set(roots)) != 1 or roots[0].name != trial \
            or config_path.name != "config.json" \
            or trajectory_path.name != "trajectory.json" \
            or manifest_path.name != "manifest.json":
        raise candidate_seal.SealError(
            "frozen candidate spec paths do not identify one Harbor trial")
    raw_value = spec.get("raw_trajectory_path")
    if raw_value is not None:
        raw_path = _spec_path(spec, "raw_trajectory_path")
        sessions = roots[0] / "agent" / "sessions"
        try:
            raw_path.relative_to(sessions)
        except ValueError as error:
            raise candidate_seal.SealError(
                "frozen raw trajectory path is outside the Harbor trial") from error
        if not raw_path.name.startswith("rollout-") \
                or raw_path.suffix != ".jsonl":
            raise candidate_seal.SealError(
                "frozen raw trajectory path is not a Codex rollout")


def _validate_spec_against_seal(
    spec: Mapping[str, Any], seal: Mapping[str, Any],
) -> None:
    if spec.get("schema") != ADAPTER_SCHEMA:
        raise candidate_seal.SealError(
            "frozen Harbor candidate spec has the wrong schema")
    identity = seal["identity"]
    dataset = identity["dataset"]
    task = identity["task"]
    solver = identity["solver"]
    world = identity["world"]
    trajectory = identity["trajectory"]
    if (spec.get("dataset") != dataset.get("name")
            or _sha256_ref(spec.get("dataset_digest"), "spec dataset digest")
            != dataset.get("sha256")
            or spec.get("task") != task.get("name")
            or _sha256_ref(spec.get("task_digest"), "spec task digest")
            != task.get("sha256")
            or _sha256_ref(
                spec.get("task_package_digest"), "spec task package digest")
            != task.get("package_sha256")
            or spec.get("trial") != identity.get("trial")):
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different source identity")
    if task.get("instruction_sha256") is not None:
        raise candidate_seal.SealError(
            "Harbor candidate seal has an unexpected instruction binding")
    _sha256_ref(task.get("config_sha256"), "seal task config digest")
    config_digest = spec.get("task_config_digest")
    if config_digest is not None and _sha256_ref(
            config_digest, "spec task config digest") \
            != task.get("config_sha256"):
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different task config digest")

    expected_solver = {
        "model": spec.get("model"),
        "effort": spec.get("effort"),
    }
    service_tier = spec.get("service_tier", "standard")
    if service_tier == "fast":
        expected_solver["service_tier"] = "fast"
    elif service_tier != "standard":
        raise candidate_seal.SealError(
            "frozen Harbor candidate has an invalid service tier")
    if expected_solver != solver:
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different solver identity")
    implementation = spec.get("agent_implementation")
    selector = spec.get("agent_config_selector")
    if implementation is None:
        if selector is not None:
            raise candidate_seal.SealError(
                "frozen Harbor spec has an orphan agent selector")
    elif implementation not in CODEX_IMPLEMENTATIONS \
            or implementation == BUILTIN_CODEX_IMPLEMENTATION \
            or not isinstance(selector, dict) \
            or set(selector) != {"name", "import_path"} \
            or (selector != {"name": implementation, "import_path": None}
                and selector != {"name": None, "import_path": implementation}):
        raise candidate_seal.SealError(
            "frozen Harbor spec has an invalid agent implementation")
    exception_type = spec.get("agent_exception_type")
    if exception_type is not None \
            and exception_type not in AGENT_OUTCOME_EXCEPTION_TYPES:
        raise candidate_seal.SealError(
            "frozen Harbor spec has an invalid agent outcome exception")
    if world.get("status") != "captured" \
            or _sha256_ref(spec.get("world_digest"), "spec world digest") \
            != world.get("sha256") \
            or _sha256_ref(
                spec.get("artifact_manifest_digest"),
                "spec artifact manifest digest") \
            != world.get("manifest_sha256"):
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different world identity")
    if _finite_reward(spec.get("reward")) != seal["evaluation"].get("reward"):
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different private reward")

    raw_digest = spec.get("raw_trajectory_digest")
    if raw_digest is None:
        if "raw_sha256" in trajectory or spec.get("raw_trajectory_path") is not None:
            raise candidate_seal.SealError(
                "frozen Harbor spec has an incomplete raw trajectory binding")
    elif _sha256_ref(raw_digest, "spec raw trajectory digest") \
            != trajectory.get("raw_sha256") \
            or spec.get("raw_trajectory_path") is None:
        raise candidate_seal.SealError(
            "frozen Harbor spec and seal have different raw trajectory identity")
    captured_roots = spec.get("captured_roots")
    if not isinstance(captured_roots, list) \
            or any(not isinstance(root, str) or not root.startswith("/")
                   for root in captured_roots) \
            or len(captured_roots) != len(set(captured_roots)):
        raise candidate_seal.SealError(
            "frozen Harbor spec has invalid captured roots")
    _validate_spec_path_affinity(spec)


def _validate_frozen_pool(
    specs: Iterable[dict[str, Any]],
    seals: Iterable[dict[str, Any]],
    views: Iterable[dict[str, Any]],
) -> dict[str, FrozenCandidate]:
    specs = list(specs)
    seals = list(seals)
    views = list(views)
    by_spec_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    spec_paths: set[Path] = set()
    for spec in specs:
        key = _spec_key(spec)
        config_path = _spec_path(spec, "task_config_path")
        if key in by_spec_key or config_path in spec_paths:
            raise candidate_seal.SealError(
                "duplicate frozen Harbor candidate spec source")
        by_spec_key[key] = spec
        spec_paths.add(config_path)

    by_source: dict[str, dict[str, Any]] = {}
    by_candidate: dict[str, dict[str, Any]] = {}
    by_seal_key: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for seal in seals:
        candidate_seal.validate_seal(seal)
        identity = seal["identity"]
        key = (
            identity["dataset"]["name"],
            identity["task"]["name"],
            identity["trial"],
        )
        if seal["source_id"] in by_source or seal["candidate_id"] in by_candidate \
                or key in by_seal_key:
            raise candidate_seal.SealError(
                "duplicate frozen Harbor candidate seal")
        by_source[seal["source_id"]] = seal
        by_candidate[seal["candidate_id"]] = seal
        by_seal_key[key] = seal
    if set(by_spec_key) != set(by_seal_key):
        raise candidate_seal.SealError(
            "frozen specs and seals describe different candidate sources")

    by_view: dict[str, dict[str, Any]] = {}
    for view in views:
        candidate_id = view.get("candidate_id")
        if not isinstance(candidate_id, str) or candidate_id in by_view:
            raise candidate_seal.SealError(
                "duplicate or malformed frozen verifier input")
        by_view[candidate_id] = view
    if set(by_view) != set(by_candidate):
        raise candidate_seal.SealError(
            "frozen seals and verifier inputs describe different candidates")

    result: dict[str, FrozenCandidate] = {}
    for key, spec in by_spec_key.items():
        seal = by_seal_key[key]
        view = by_view[seal["candidate_id"]]
        _validate_spec_against_seal(spec, seal)
        if view != candidate_seal.verifier_view(seal):
            raise candidate_seal.SealError(
                "frozen verifier input does not match its private seal")
        result[seal["source_id"]] = FrozenCandidate(spec, seal, view)
    return result


def _load_frozen_pool(
    specs_path: Path, seals_path: Path, verifier_inputs_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]],
           dict[str, FrozenCandidate]]:
    paths = (specs_path, seals_path, verifier_inputs_path)
    exists = tuple(path.exists() for path in paths)
    if any(exists) and not all(exists):
        raise candidate_seal.SealError(
            "candidate specs, seals, and verifier inputs must all exist together")
    if not any(exists):
        return [], [], [], {}
    specs = candidate_seal.load_jsonl(specs_path)
    seals = candidate_seal.load_jsonl(seals_path)
    views = candidate_seal.load_jsonl(verifier_inputs_path)
    return specs, seals, views, _validate_frozen_pool(specs, seals, views)


def _merge_specs(
    existing: Iterable[dict[str, Any]], incoming: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_source: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for spec in [*existing, *incoming]:
        key = _spec_key(spec)
        if any(not isinstance(value, str) or not value for value in key):
            raise candidate_seal.SealError("frozen candidate spec has no source identity")
        prior = by_source.get(key)
        if prior is not None and prior != spec:
            # ``service_tier`` was added after the first standard-tier seals
            # were frozen.  Missing and explicit standard are semantically
            # identical, and retaining the earlier bytes preserves the
            # immutable private audit record.  No analogous migration exists
            # for fast or any other field.
            prior_compare = dict(prior)
            spec_compare = dict(spec)
            if prior_compare.get("service_tier", "standard") == "standard" \
                    and spec_compare.get("service_tier", "standard") == "standard":
                prior_compare.pop("service_tier", None)
                spec_compare.pop("service_tier", None)
            if prior_compare == spec_compare:
                continue
            raise candidate_seal.SealError(
                "attempted to change a frozen Harbor candidate spec")
        if prior is None:
            by_source[key] = spec
    return [by_source[key] for key in sorted(by_source)]


def freeze_jobs(
    job_dirs: Iterable[str | os.PathLike[str]], *,
    specs_path: str | os.PathLike[str],
    seals_path: str | os.PathLike[str],
    verifier_inputs_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Incrementally merge completed candidates from one or more Harbor jobs."""
    job_dirs = [Path(path).expanduser().resolve() for path in job_dirs]
    if not job_dirs:
        raise candidate_seal.SealError("provide at least one Harbor job")
    if len(job_dirs) != len(set(job_dirs)):
        raise candidate_seal.SealError("duplicate Harbor job directory")
    specs_path = Path(specs_path).expanduser().resolve()
    seals_path = Path(seals_path).expanduser().resolve()
    verifier_inputs_path = Path(verifier_inputs_path).expanduser().resolve()
    if len({specs_path, seals_path, verifier_inputs_path}) != 3:
        raise candidate_seal.SealError(
            "candidate specs, seals, and verifier inputs need distinct paths")
    specs_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{specs_path}.lock")
    with lock_path.open("a+") as output_lock:
        fcntl.flock(output_lock.fileno(), fcntl.LOCK_EX)
        existing_specs, existing_seals, _, frozen = _load_frozen_pool(
            specs_path, seals_path, verifier_inputs_path)
        output_digests = {
            path: candidate_seal.file_digest(path)
            for path in (specs_path, seals_path, verifier_inputs_path)
            if path.exists()
        }
        old_source_ids = set(frozen)
        candidates: list[Candidate] = []
        reused: list[str] = []
        skipped_by_job: dict[str, dict[str, str]] = {}
        for job_dir in job_dirs:
            context = _job_context(job_dir)
            discovered, skipped, already_frozen = _discover_context(
                context, frozen)
            candidates.extend(discovered)
            reused.extend(already_frozen)
            skipped_by_job[str(context.directory)] = skipped

        merged_specs = _merge_specs(
            existing_specs, (candidate.spec for candidate in candidates))
        merged_seals = candidate_seal.merge_seals(
            existing_seals, (candidate.seal for candidate in candidates))
        merged_views = [
            candidate_seal.verifier_view(seal) for seal in merged_seals]
        validated = _validate_frozen_pool(
            merged_specs, merged_seals, merged_views)
        if any(candidate_seal.file_digest(path) != digest
               for path, digest in output_digests.items()):
            raise candidate_seal.SealError(
                "candidate output changed during incremental freeze")
        candidate_seal.write_jsonl_atomic(specs_path, merged_specs)
        candidate_seal.write_jsonl_atomic(seals_path, merged_seals)
        candidate_seal.write_jsonl_atomic(
            verifier_inputs_path, merged_views)
    return {
        "schema": ADAPTER_SCHEMA,
        "jobs": [str(path) for path in job_dirs],
        "discovered": len(candidates),
        "reused": len(reused),
        "added": len(set(validated) - old_source_ids),
        "frozen": len(merged_seals),
        "skipped_by_job": skipped_by_job,
        "specs": str(specs_path),
        "seals": str(seals_path),
        "verifier_inputs": str(verifier_inputs_path),
    }


def freeze_job(
    job_dir: str | os.PathLike[str], *,
    specs_path: str | os.PathLike[str],
    seals_path: str | os.PathLike[str],
    verifier_inputs_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Backward-compatible single-job incremental freeze."""
    job_dir = Path(job_dir).expanduser().resolve()
    result = freeze_jobs(
        [job_dir],
        specs_path=specs_path,
        seals_path=seals_path,
        verifier_inputs_path=verifier_inputs_path,
    )
    result["skipped"] = result["skipped_by_job"][str(job_dir)]
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--job-dir", required=True, action="append", type=Path,
        help="Harbor job to merge; repeat for multiple jobs")
    parser.add_argument("--specs", type=Path)
    parser.add_argument("--seals", type=Path)
    parser.add_argument("--verifier-inputs", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    job_dirs = [path.expanduser().resolve() for path in args.job_dir]
    if len(job_dirs) > 1 and any(path is None for path in (
            args.specs, args.seals, args.verifier_inputs)):
        parser.error(
            "multiple --job-dir values require --specs, --seals, and "
            "--verifier-inputs")
    default_job = job_dirs[0]
    kwargs = {
        "specs_path": args.specs or default_job / "candidate-specs.jsonl",
        "seals_path": args.seals or default_job / "candidate-seals.jsonl",
        "verifier_inputs_path": (
            args.verifier_inputs
            or default_job / "candidate-verifier-inputs.jsonl"),
    }
    result = freeze_job(job_dirs[0], **kwargs) if len(job_dirs) == 1 \
        else freeze_jobs(job_dirs, **kwargs)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
