"""Score the official Terminal-Bench 2.1 mini-swe-agent trajectories.

The upstream release contains conversation trajectories and held-out reward
labels, but no post-run filesystem snapshot.  This driver converts each trace
to the verifier's raw offline inputs with ``miniswe.convert`` and deliberately
leaves ``world/`` absent.  The verifier therefore sees the historical world as
``unavailable``; a transcript is evidence that commands ran and produced the
recorded output, never a reconstruction of the final filesystem.

Isolation and identity are part of the experiment, not bookkeeping details:

* seats run in a newly-built state containing exactly the tool feed, any
  content-addressed full-response blobs, semantic task, retained agent
  protocol, final candidate, trace boundary manifest, and a trace-only world
  manifest; the task compiler reads only the semantic task;
* the official reward is never copied into that state or included in a seat
  label/result JSONL (it is joined only while rendering the TSV);
* the batch identity binds the verifier config, dataset tree, adapter source,
  provider endpoint/served model, and repetition;
* every resume key additionally binds a canonical trajectory hash with the
  top-level reward field removed.

Usage::

    uv run python experiments/terminal_bench/score_official.py \
      --trajectory-root /path/to/data/terminal_bench_2.1_trajs \
      --task chess-best-move fix-code-vulnerability largest-eigenval \
      --workers 3 --dry-run
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from collections.abc import Callable, Iterable
from contextlib import contextmanager
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

HERE = os.path.dirname(os.path.abspath(__file__))
EXPERIMENTS = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(EXPERIMENTS)
for _path in (REPO_ROOT, EXPERIMENTS, HERE):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common import results as res
from common import scoring

from verifier import offline

DATASET_NAME = "terminal-bench-2.1-official"
IDENTITY_SCHEMA = "tb2.1-official-gv-v2"
TRACE_MANIFEST_SCHEMA = "terminal-bench-2.1-atif-adapter-v2"
TRACE_PATTERN = "*_trajectory.json"
STATE_FILES = frozenset({
    "hook_feed.jsonl", "response_blobs.jsonl", "task.md", "protocol.md",
    "final.txt", "world_manifest.json", "trace_manifest.json",
})
DERIVED_FILES = frozenset({
    "events.jsonl", "events.extract.jsonl", "extract.json", "score.json",
    "claims.sqlite", "claims.sqlite-wal", "claims.sqlite-shm",
    "claims.sqlite-journal",
})
DERIVED_DIRS = frozenset({"artifacts", "work"})
SOURCE_MANIFEST = ".source_inputs.json"
WORLD_MANIFEST = {
    "captured_roots": [],
    "world_mode": "unavailable",
    "evidence": "trace-only",
}
DEFAULT_RESULTS = os.path.join(HERE, "results", "official_2.1_gv_scores.jsonl")
DEFAULT_SCORING_ROOT = os.path.expanduser(
    os.environ.get("GV_SCORING_ROOT", "~/.gv-scoring/terminal_bench"))
_USAGE_WRITE_LOCK = threading.Lock()


def _usage_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        rendered = dump()
        return rendered if isinstance(rendered, dict) else {}
    return {}


def _usage_int(usage: dict[str, Any], field: str) -> int:
    try:
        return int(usage.get(field, 0) or 0)
    except (TypeError, ValueError):
        return 0


def _append_agent_usage(
    path: Path, run_id: str, backend: Any, role: Any,
    prompt: str, transcript: dict[str, Any],
) -> None:
    state = Path(backend.cwd or "")
    work_id = state.parent.name if state.name == "gv" else None
    usage = _usage_dict(transcript.get("usage"))
    input_tokens = _usage_int(usage, "input_tokens")
    cache_creation = _usage_int(usage, "cache_creation_input_tokens")
    cache_read = _usage_int(usage, "cache_read_input_tokens")
    output_tokens = _usage_int(usage, "output_tokens")
    try:
        cost = float(transcript.get("total_cost_usd") or 0.0)
    except (TypeError, ValueError):
        cost = 0.0
    record = {
        "schema_version": 1,
        "run_id": run_id,
        "work_id": work_id,
        "role": role.name,
        "session_id": transcript.get("session_id"),
        "subtype": transcript.get("subtype"),
        "num_turns": transcript.get("num_turns"),
        "api_error_status": transcript.get("api_error_status"),
        "recorded_at_unix_ns": time.time_ns(),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "input_tokens": input_tokens,
        "cache_creation_input_tokens": cache_creation,
        "cache_read_input_tokens": cache_read,
        "total_input_tokens": input_tokens + cache_creation + cache_read,
        "output_tokens": output_tokens,
        "cost_usd": cost,
    }
    with _USAGE_WRITE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())


def _aggregate_agent_usage(path: Path, run_id: str, work_id: str) -> dict[str, Any]:
    totals: dict[str, Any] = {
        "calls": 0,
        "input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "total_input_tokens": 0,
        "output_tokens": 0,
        "cost_usd": 0.0,
    }
    if not path.is_file():
        return totals
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("run_id") != run_id or row.get("work_id") != work_id:
                continue
            totals["calls"] += 1
            for field in (
                "input_tokens", "cache_creation_input_tokens",
                "cache_read_input_tokens", "total_input_tokens",
                "output_tokens",
            ):
                totals[field] += int(row.get(field, 0) or 0)
            totals["cost_usd"] += float(row.get("cost_usd", 0.0) or 0.0)
    return totals


@contextmanager
def _agent_usage_ledger(path: Path, run_id: str):
    """Persist every SDK attempt and attach per-trace aggregates to features."""
    from verifier.llm.agents import AgentBackend

    original_attempt = AgentBackend._attempt
    original_score = offline.score

    async def tracked_attempt(backend, role, prompt):
        before = len(backend.transcripts)
        try:
            return await original_attempt(backend, role, prompt)
        finally:
            for transcript in backend.transcripts[before:]:
                _append_agent_usage(
                    path, run_id, backend, role, prompt, transcript
                )

    def tracked_score(state_dir, *args, **kwargs):
        features = original_score(state_dir, *args, **kwargs)
        work_id = Path(state_dir).parent.name
        return {
            **features,
            "llm_usage": _aggregate_agent_usage(path, run_id, work_id),
        }

    AgentBackend._attempt = tracked_attempt
    offline.score = tracked_score
    try:
        yield
    finally:
        offline.score = original_score
        AgentBackend._attempt = original_attempt


class DatasetError(ValueError):
    """The official trajectory tree is malformed or internally ambiguous."""


class StateError(ValueError):
    """A converted state violates the trace-only isolation boundary."""


@dataclasses.dataclass(frozen=True)
class OfficialTrace:
    task: str
    trial: str
    path: str
    relative_path: str
    trace_hash: str
    reward: float | None


@dataclasses.dataclass(frozen=True)
class Dataset:
    root: str
    tree_fingerprint: str
    traces: tuple[OfficialTrace, ...]


@dataclasses.dataclass(frozen=True)
class ProviderIdentity:
    provider: str
    endpoint: str
    requested_model: str
    served_model: str
    text_only: bool

    def fields(self) -> dict[str, str]:
        return dataclasses.asdict(self)


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _read_trajectory(path: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        raise DatasetError(f"cannot read trajectory {path}: {err}") from err
    if not isinstance(value, dict):
        raise DatasetError(f"trajectory must be a JSON object: {path}")
    if not isinstance(value.get("trajectory"), dict):
        raise DatasetError(f"trajectory payload is missing: {path}")
    return value


def trace_hash(path: str) -> str:
    """Canonical hash of one release record, excluding only held-out gold.

    The upstream reward is a top-level sibling of ``trajectory``.  Removing it
    before canonicalization means a corrected label does not masquerade as a
    different model rollout, while every byte of semantic trajectory metadata
    (task checksum, steps, observations, final message) remains identity-bound.
    """
    payload = dict(_read_trajectory(path))
    payload.pop("reward", None)
    return _digest(payload)


def _reward(value: Any, path: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DatasetError(f"reward must be numeric or null: {path}")
    return float(value)


def _safe_label(value: str, field: str, path: str) -> str:
    value = value.strip()
    if not value or any(char in value for char in "\x00\r\n\t"):
        raise DatasetError(f"invalid {field} in trajectory: {path}")
    return value


def discover_dataset(root: str) -> Dataset:
    """Load and fingerprint every official ``*_trajectory.json`` below root."""
    root = os.path.realpath(os.path.abspath(os.path.expanduser(root)))
    if not os.path.isdir(root):
        raise DatasetError(f"trajectory root is not a directory: {root}")
    paths = sorted(str(path) for path in Path(root).rglob(TRACE_PATTERN))
    if not paths:
        raise DatasetError(f"no {TRACE_PATTERN} files below {root}")

    traces: list[OfficialTrace] = []
    seen: dict[tuple[str, str], str] = {}
    for path in paths:
        if os.path.islink(path):
            raise DatasetError(f"trajectory may not be a symlink: {path}")
        payload = _read_trajectory(path)
        task = payload.get("task_name") or os.path.basename(os.path.dirname(path))
        trial = payload.get("trial_name") or os.path.basename(path)[:-len(
            "_trajectory.json")]
        if not isinstance(task, str):
            raise DatasetError(f"trajectory has no task_name: {path}")
        if not isinstance(trial, str):
            raise DatasetError(f"trajectory has no trial_name: {path}")
        task = _safe_label(task, "task_name", path)
        trial = _safe_label(trial, "trial_name", path)
        key = (task, trial)
        if key in seen:
            raise DatasetError(
                f"duplicate task/trial {task}/{trial}: {seen[key]} and {path}")
        seen[key] = path
        rel = os.path.relpath(path, root).replace(os.sep, "/")
        no_gold = dict(payload)
        no_gold.pop("reward", None)
        traces.append(OfficialTrace(
            task=task,
            trial=trial,
            path=path,
            relative_path=rel,
            trace_hash=_digest(no_gold),
            reward=_reward(payload.get("reward"), path),
        ))

    # This is a dataset-tree identity, not a gold-label identity.  Paths bind
    # task/run membership and hashes bind the complete non-reward records.
    tree = _digest({
        "schema": "tb2.1-trajectory-tree-v1",
        "files": [
            {"path": trace.relative_path, "trace_hash": trace.trace_hash}
            for trace in traces
        ],
    })
    return Dataset(root=root, tree_fingerprint=tree, traces=tuple(traces))


def _safe_endpoint(url: str) -> str:
    """Canonical endpoint identity without userinfo, query, or fragment."""
    if not url:
        return ""
    parts = urlsplit(url)
    if not parts.scheme or not parts.netloc:
        return url.rstrip("/")
    host = parts.hostname or ""
    if ":" in host and not host.startswith("["):
        host = f"[{host}]"
    if parts.port is not None:
        host += f":{parts.port}"
    return urlunsplit((parts.scheme.lower(), host.lower(),
                       parts.path.rstrip("/"), "", ""))


def resolve_provider_identity(
    model: str | None, seats: str | None,
    served_model_identity: str | None = None,
) -> ProviderIdentity:
    """Resolve the actual backend identity without ever recording a key."""
    if seats == "codex":
        cfg = offline.config_fields(model=model, seats=seats)
        resolved = str(cfg["model"])
        return ProviderIdentity(
            provider="codex-cli",
            endpoint="chatgpt-subscription",
            requested_model=resolved,
            served_model=served_model_identity or resolved,
            text_only=False,
        )

    provider = importlib.import_module("verifier.llm.provider")
    requested = model or str(provider.MODEL)
    served = (served_model_identity
              or os.environ.get("GV_SERVED_MODEL_IDENTITY")
              or os.environ.get("GV_MODEL_ID")
              or str(provider.MODEL))
    return ProviderIdentity(
        provider=str(provider.PROVIDER),
        endpoint=_safe_endpoint(str(provider.ANTHROPIC_BASE_URL)),
        requested_model=requested,
        served_model=served,
        text_only=bool(provider.TEXT_ONLY),
    )


def adapter_fingerprint(adapter: Any = None) -> str:
    """Hash the exact converter source used to construct seat inputs."""
    if adapter is None:
        adapter = importlib.import_module("miniswe")
    path = getattr(adapter, "__file__", None) or os.path.join(HERE, "miniswe.py")
    if not path or not os.path.isfile(path):
        raise StateError("cannot fingerprint experiments/terminal_bench/miniswe.py")
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def batch_identity(
    cfg: dict[str, Any], dataset_tree: str, adapter_hash: str,
    provider: ProviderIdentity, rep: str,
) -> tuple[str, dict[str, Any]]:
    fields: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "dataset": DATASET_NAME,
        "dataset_tree_fingerprint": dataset_tree,
        "adapter_fingerprint": adapter_hash,
        "verifier_config": cfg["config"],
        "provider_identity": provider.fields(),
        "rep": rep,
    }
    return _digest(fields)[:24], fields


def work_identity(run_id: str, trace: OfficialTrace) -> str:
    """Per-trace resume key; trace hash is explicit, not implied by a tree."""
    return _digest({
        "schema": IDENTITY_SCHEMA,
        "run_id": run_id,
        "task": trace.task,
        "trial": trace.trial,
        "trace_hash": trace.trace_hash,
    })[:24]


def _assert_regular(path: str) -> None:
    try:
        mode = os.lstat(path).st_mode
    except OSError as err:
        raise StateError(f"missing sanitized input {path}: {err}") from err
    if not stat.S_ISREG(mode) or os.path.islink(path):
        raise StateError(f"sanitized input is not a regular file: {path}")


def _reject_symlinks(root: str) -> None:
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        for name in [*dirnames, *filenames]:
            path = os.path.join(dirpath, name)
            if os.path.islink(path):
                raise StateError(f"symlink forbidden in sanitized state: {path}")


def _load_json_object(path: str, label: str) -> dict[str, Any]:
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        raise StateError(f"invalid {label}: {err}") from err
    if not isinstance(value, dict):
        raise StateError(f"invalid {label}: expected a JSON object")
    return value


def _validate_trace_manifest(state: str, expected_trace_hash: str | None) \
        -> None:
    path = os.path.join(state, "trace_manifest.json")
    manifest = _load_json_object(path, "trace manifest")
    if manifest.get("schema") != TRACE_MANIFEST_SCHEMA:
        raise StateError(
            "trace manifest has wrong schema: "
            f"{manifest.get('schema')!r}")
    if manifest.get("evidence_mode") != "trace-only":
        raise StateError("trace manifest must declare evidence_mode=trace-only")

    source = manifest.get("source")
    contract = manifest.get("task_contract")
    envelopes = manifest.get("response_envelopes")
    coverage = manifest.get("coverage")
    boundary = manifest.get("final_boundary")
    converted = manifest.get("converted_files")
    if not all(isinstance(value, dict) for value in (
            source, contract, coverage, boundary, converted)):
        raise StateError("trace manifest is missing a structured section")
    if not isinstance(envelopes, list) or any(
            not isinstance(value, dict) for value in envelopes):
        raise StateError("trace manifest response_envelopes must be objects")

    trace_digest = source.get("trajectory_sha256")
    if not isinstance(trace_digest, str) or len(trace_digest) != 64:
        raise StateError("trace manifest has invalid trajectory_sha256")
    if expected_trace_hash is not None and trace_digest != expected_trace_hash:
        raise StateError(
            "trace manifest trajectory hash does not match selected trace")
    if contract.get("splitter") != "exact-delimiter-and-wrapper-digest-v1":
        raise StateError("trace manifest has an unrecognized task splitter")

    for name in (
            "hook_feed.jsonl", "response_blobs.jsonl", "task.md",
            "protocol.md", "final.txt", "world_manifest.json"):
        expected = converted.get(name)
        actual = _file_sha256(os.path.join(state, name))
        if expected != actual:
            raise StateError(
                f"trace manifest digest mismatch for {name}: "
                f"expected {expected!r}, got {actual}")
    if contract.get("task_sha256") != converted.get("task.md"):
        raise StateError("trace manifest task contract digest is inconsistent")
    if contract.get("protocol_sha256") != converted.get("protocol.md"):
        raise StateError("trace manifest protocol digest is inconsistent")
    if boundary.get("final_text_sha256") != converted.get("final.txt"):
        raise StateError("trace manifest final boundary digest is inconsistent")

    tool_use_ids: list[str] = []
    matched_results = 0
    pending_calls = 0
    harness_results = 0
    referenced_response_blobs: set[str] = set()
    for index, envelope in enumerate(envelopes):
        if envelope.get("response_index") != index:
            raise StateError("trace manifest response indexes are not continuous")
        ids = envelope.get("tool_use_ids")
        results = envelope.get("result_summaries")
        pending = envelope.get("pending_tool_use_ids")
        harness = envelope.get("harness_results")
        if not all(isinstance(value, list) for value in (
                ids, results, pending, harness)):
            raise StateError("trace manifest response envelope is malformed")
        if any(not isinstance(value, str) for value in ids):
            raise StateError("trace manifest contains a non-string tool_use_id")
        tool_use_ids.extend(ids)
        matched_results += len(results)
        pending_calls += len(pending)
        harness_results += len(harness)
        for result in [*results, *harness]:
            if not isinstance(result, dict):
                raise StateError("trace manifest result summary is malformed")
            if not result.get("adapter_excerpt_truncated"):
                continue
            ref = result.get("full_response_ref")
            if (not isinstance(ref, dict)
                    or ref.get("file") != "response_blobs.jsonl"
                    or not isinstance(ref.get("sha256"), str)):
                raise StateError(
                    "truncated response has no valid full-response reference")
            referenced_response_blobs.add(ref["sha256"])
    if len(tool_use_ids) != len(set(tool_use_ids)):
        raise StateError("trace manifest contains duplicate tool_use_ids")
    response_blob_digests = set()
    blob_path = os.path.join(state, "response_blobs.jsonl")
    with open(blob_path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                blob = json.loads(line)
            except json.JSONDecodeError as err:
                raise StateError(f"invalid response blob record: {err}") from err
            if not isinstance(blob, dict):
                raise StateError("response blob record must be an object")
            digest = blob.get("sha256")
            canonical_response = blob.get("canonical_response")
            if (not isinstance(digest, str)
                    or not isinstance(canonical_response, str)
                    or hashlib.sha256(canonical_response.encode("utf-8"))
                    .hexdigest() != digest):
                raise StateError("response blob record has an invalid digest")
            if digest in response_blob_digests:
                raise StateError("response blob records must be deduplicated")
            response_blob_digests.add(digest)
    if response_blob_digests != referenced_response_blobs:
        raise StateError(
            "response blob records do not match truncated-response references")

    expected_counts = {
        "agent_responses": len(envelopes),
        "zero_tool_responses": sum(not envelope.get("tool_use_ids")
                                   for envelope in envelopes),
        "multi_tool_responses": sum(
            len(envelope.get("tool_use_ids")) > 1
            for envelope in envelopes),
        "tool_calls": len(tool_use_ids),
        "matched_tool_results": matched_results,
        "pending_tool_calls": pending_calls,
        "harness_results": harness_results,
        "response_blob_records": len(response_blob_digests),
    }
    for field, expected in expected_counts.items():
        if coverage.get(field) != expected:
            raise StateError(
                f"trace manifest coverage mismatch for {field}: "
                f"expected {expected}, got {coverage.get(field)!r}")

    event_count = 0
    with open(os.path.join(state, "hook_feed.jsonl"), encoding="utf-8") \
            as handle:
        for line in handle:
            if line.strip():
                event_count += 1
    if manifest.get("event_count") != event_count:
        raise StateError("trace manifest event_count does not match hook feed")


def validate_state(state: str, *, allow_derived: bool = False,
                   expected_trace_hash: str | None = None) -> None:
    """Enforce the trace-only seat boundary, fail closed.

    A newly converted state must contain exactly the seven source inputs.  A
    resumed state may additionally contain only the verifier's named derived
    files/directories.  The latter is necessary after even a partial
    ``offline.score`` run, which creates its ledger, SQLite store, artifacts,
    and work tree inside ``gv/``.
    """
    state = os.path.abspath(state)
    if (os.path.islink(state) or os.path.realpath(state) != state
            or not os.path.isdir(state)):
        raise StateError(f"sanitized state is not a real directory: {state}")
    names = set(os.listdir(state))
    allowed = STATE_FILES | (DERIVED_FILES | DERIVED_DIRS
                             if allow_derived else frozenset())
    if not STATE_FILES.issubset(names) or not names.issubset(allowed):
        missing = sorted(STATE_FILES - names)
        extra = sorted(names - allowed)
        raise StateError(
            f"sanitized state has wrong top-level entries; missing={missing}, "
            f"extra={extra}")
    if not allow_derived and names != STATE_FILES:
        raise StateError("new sanitized state contains verifier-derived entries")
    for name in sorted(STATE_FILES):
        _assert_regular(os.path.join(state, name))
    for name in sorted(names & DERIVED_FILES):
        _assert_regular(os.path.join(state, name))
    for name in sorted(names & DERIVED_DIRS):
        path = os.path.join(state, name)
        if os.path.islink(path) or not os.path.isdir(path):
            raise StateError(f"invalid verifier-derived directory: {path}")
    # In particular, there is no synthetic world/ reconstructed from shell
    # transcript text.  Derived work trees may be nested, but never linked.
    _reject_symlinks(state)
    for name in (
            "hook_feed.jsonl", "task.md", "protocol.md", "final.txt",
            "trace_manifest.json"):
        path = os.path.join(state, name)
        if os.path.getsize(path) == 0:
            raise StateError(f"sanitized input is empty: {path}")
    try:
        with open(os.path.join(state, "world_manifest.json"),
                  encoding="utf-8") as handle:
            manifest = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        raise StateError(f"invalid trace-only world manifest: {err}") from err
    if manifest != WORLD_MANIFEST:
        raise StateError(
            "official trajectories have no captured world; manifest must be "
            f"exactly {WORLD_MANIFEST!r}")
    _validate_trace_manifest(state, expected_trace_hash)


def _file_sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _source_hashes(state: str) -> dict[str, dict[str, Any]]:
    return {
        name: {
            "sha256": _file_sha256(os.path.join(state, name)),
            "bytes": os.path.getsize(os.path.join(state, name)),
        }
        for name in sorted(STATE_FILES)
    }


def _source_manifest(run_id: str, wid: str, trace: OfficialTrace,
                     state: str) -> dict[str, Any]:
    return {
        "schema": "tb2.1-sanitized-source-v2",
        "run_id": run_id,
        "work_id": wid,
        "trace_hash": trace.trace_hash,
        "inputs": _source_hashes(state),
    }


def _write_json_atomic(path: str, value: dict[str, Any]) -> None:
    parent = os.path.dirname(path)
    fd, temp = tempfile.mkstemp(prefix=".manifest-", dir=parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, sort_keys=True)
            handle.write("\n")
        os.replace(temp, path)
        temp = ""
    finally:
        if temp and os.path.exists(temp):
            os.remove(temp)


def _read_source_manifest(path: str) -> dict[str, Any] | None:
    if not os.path.exists(path):
        return None
    if os.path.islink(path):
        raise StateError(f"source manifest may not be a symlink: {path}")
    try:
        with open(path, encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, json.JSONDecodeError) as err:
        raise StateError(f"invalid source manifest {path}: {err}") from err
    if not isinstance(value, dict):
        raise StateError(f"source manifest is not an object: {path}")
    return value


def _same_hash(trace: OfficialTrace) -> bool:
    return trace_hash(trace.path) == trace.trace_hash


def sanitized_state(
    trace: OfficialTrace, scoring_root: str, run_id: str,
    adapter: Any = None,
) -> str:
    """Atomically build/reuse one immutable, gold-free converted state."""
    adapter = adapter or importlib.import_module("miniswe")
    wid = work_identity(run_id, trace)
    parent = os.path.join(scoring_root, DATASET_NAME, run_id, wid)
    state = os.path.join(parent, "gv")
    source_manifest = os.path.join(parent, SOURCE_MANIFEST)
    if os.path.lexists(state):
        validate_state(
            state, allow_derived=True,
            expected_trace_hash=trace.trace_hash)
        expected = _read_source_manifest(source_manifest)
        actual = _source_manifest(run_id, wid, trace, state)
        if expected is not None:
            if expected != actual:
                raise StateError(
                    f"sanitized source inputs changed since conversion: {state}")
            return state
        # A crash between publishing gv/ and its sibling manifest is the only
        # legitimate manifest-less case.  Reconvert below and compare bytes
        # before adopting it; never trust an unsealed existing source state.
    os.makedirs(parent, exist_ok=True)
    if not _same_hash(trace):
        raise DatasetError(f"trajectory changed after discovery: {trace.path}")
    stage = tempfile.mkdtemp(prefix=".convert-", dir=parent)
    try:
        adapter.convert(trace.path, stage)
        if not _same_hash(trace):
            raise DatasetError(f"trajectory changed during conversion: {trace.path}")
        validate_state(stage, expected_trace_hash=trace.trace_hash)
        converted = _source_manifest(run_id, wid, trace, stage)
        if os.path.lexists(state):
            # Manifest-less existing state: source bytes must equal a fresh,
            # deterministic conversion under the current adapter.
            validate_state(
                state, allow_derived=True,
                expected_trace_hash=trace.trace_hash)
            if _source_manifest(run_id, wid, trace, state) != converted:
                raise StateError(
                    f"existing sanitized inputs differ from conversion: {state}")
        else:
            try:
                os.rename(stage, state)
                stage = ""
            except OSError:
                # A concurrent driver may have won the atomic publish race.
                # Reuse only when both the boundary and source bytes match.
                if not os.path.isdir(state):
                    raise
                validate_state(
                    state, allow_derived=True,
                    expected_trace_hash=trace.trace_hash)
                if _source_manifest(run_id, wid, trace, state) != converted:
                    raise StateError(
                        f"concurrent sanitized inputs differ: {state}")
        _write_json_atomic(source_manifest, converted)
        return state
    finally:
        if stage and os.path.isdir(stage):
            shutil.rmtree(stage)


def _select(
    traces: Iterable[OfficialTrace], tasks: list[str] | None,
    trials: list[str] | None,
) -> list[OfficialTrace]:
    selected = [
        trace for trace in traces
        if (not tasks or trace.task in tasks)
        and (not trials or trace.trial in trials)
    ]
    return sorted(selected, key=lambda trace: (trace.task, trace.trial))


def _reward_text(reward: float | None) -> str:
    return "?" if reward is None else f"{reward:g}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--trajectory-root", required=True,
        help="official data/terminal_bench_2.1_trajs root (searched recursively)")
    parser.add_argument("--results", default=DEFAULT_RESULTS,
                        help="independent append-only GV JSONL")
    parser.add_argument("--scoring-root", default=DEFAULT_SCORING_ROOT,
                        help="isolated converted/derived state root")
    parser.add_argument("--task", nargs="+", default=None,
                        help="only exact task names")
    parser.add_argument("--trial", nargs="+", default=None,
                        help="only exact official trial names")
    parser.add_argument("--workers", type=int, default=4,
                        help="traces scored concurrently (default: 4)")
    parser.add_argument("--rep", default="r1",
                        help="fresh-sampling repetition identity (default: r1)")
    parser.add_argument("--dry-run", action="store_true",
                        help="validate inputs/identity and show plan; write nothing")
    parser.add_argument("--model", default=None,
                        help="requested verifier model (default: provider model)")
    parser.add_argument(
        "--served-model-identity", default=None,
        help="checkpoint/deployment identity behind the requested model; defaults "
             "to GV_SERVED_MODEL_IDENTITY, GV_MODEL_ID, then provider model")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="extraction retry rounds (default: 3)")
    parser.add_argument("--critic-rounds", type=int, default=None,
                        help="decomposition-game round budget")
    parser.add_argument("--seats", choices=["codex"], default=None,
                        help="use Codex CLI seats instead of the SDK provider")
    parser.add_argument(
        "--effort", choices=["low", "medium", "high", "xhigh", "max"],
        default=None, help="Codex reasoning effort")
    parser.add_argument(
        "--service-tier", choices=["standard", "fast"], default="standard",
        help="Codex service tier (default: standard)")
    return parser


def run(
    args: argparse.Namespace, *, adapter: Any = None,
    adapter_hash: str | None = None,
    provider_identity: ProviderIdentity | None = None,
    score_many_fn: Callable[..., Any] | None = None,
) -> int:
    if args.workers < 1:
        raise SystemExit("--workers must be at least 1")
    if not args.rep.strip():
        raise SystemExit("--rep may not be empty")
    if args.service_tier == "fast" and args.seats != "codex":
        raise SystemExit("--service-tier fast requires --seats codex")

    dataset = discover_dataset(args.trajectory_root)
    selected = _select(dataset.traces, args.task, args.trial)
    if not selected:
        raise SystemExit("no official trajectories matched --task/--trial")

    adapter = adapter or importlib.import_module("miniswe")
    ahash = adapter_hash or adapter_fingerprint(adapter)
    provider = provider_identity or resolve_provider_identity(
        args.model, args.seats, args.served_model_identity)
    # Pass the resolved request model explicitly.  "provider-default" is not
    # a sufficient model identity for a reproducible official run.
    model = provider.requested_model
    cfg = offline.config_fields(
        model=model, seats=args.seats, effort=args.effort,
        max_rounds=args.max_rounds, critic_rounds=args.critic_rounds,
        service_tier=args.service_tier)
    run_id, identity = batch_identity(
        cfg, dataset.tree_fingerprint, ahash, provider, args.rep)

    results_path = os.path.realpath(os.path.abspath(os.path.expanduser(
        args.results)))
    usage_path = Path(f"{results_path}.usage.jsonl")
    scoring_root = os.path.realpath(os.path.abspath(
        os.path.expanduser(args.scoring_root)))
    # Keep source/gold and result logs out of the constructed state subtree.
    # This is isolation by construction (the seats are not an OS sandbox).
    common = os.path.commonpath((dataset.root, scoring_root))
    if common in (dataset.root, scoring_root):
        raise SystemExit(
            "--scoring-root and --trajectory-root may not contain one another")
    if os.path.commonpath((results_path, scoring_root)) == scoring_root:
        raise SystemExit("--results may not be inside --scoring-root")
    if os.path.commonpath((results_path, dataset.root)) == dataset.root:
        raise SystemExit("--results may not be inside --trajectory-root")

    rows = res.load(results_path)
    cached = {
        row.get("work_id"): row for row in rows
        if row.get("run_id") == run_id and row.get("features") is not None
    }
    work = {work_identity(run_id, trace): trace for trace in selected}
    pending = [trace for wid, trace in work.items() if wid not in cached]
    print(
        f"[tb:official] dataset={dataset.tree_fingerprint[:16]} "
        f"adapter={ahash[:16]} provider={provider.provider} "
        f"endpoint={provider.endpoint or '-'} model={provider.requested_model} "
        f"served={provider.served_model} text_only={int(provider.text_only)} "
        f"service_tier={cfg['service_tier']} verifier={cfg['config']} "
        f"run={run_id} rep={args.rep}",
        file=sys.stderr, flush=True)
    print(
        f"[tb:official] selected {len(selected)} traces in "
        f"{len({trace.task for trace in selected})} tasks; "
        f"{len(pending)} pending, {len(selected) - len(pending)} cached",
        file=sys.stderr, flush=True)

    if args.dry_run:
        # No state/result directories are made, and the adapter/score pipeline
        # is not invoked.  Fingerprinting its source is the only adapter access.
        print(
            f"[tb:official] DRY-RUN: would score {len(pending)} traces with "
            f"{args.workers} workers; no files written",
            file=sys.stderr, flush=True)
        return 0

    by_task: dict[str, list[OfficialTrace]] = {}
    for trace in selected:
        by_task.setdefault(trace.task, []).append(trace)
    scored = dict(cached)
    remaining: dict[str, int] = {}
    emitted: set[str] = set()
    header = (
        "task\ttrial\ttrace_hash\ttb_reward\tgv_verdict\tgv_score\t"
        "gv_final_obligations_verified_frac\tgv_rank")
    print(header, flush=True)

    def render(task: str) -> None:
        candidates = []
        for trace in by_task[task]:
            row = scored.get(work_identity(run_id, trace))
            if row is not None:
                candidates.append((trace, row))
        ranked = sorted(candidates, key=lambda pair: (
            tuple(-item for item in offline.rank_key(
                pair[1].get("features"))),
            pair[0].trial,
        ))
        rank = {
            work_identity(run_id, trace): position
            for position, (trace, _row) in enumerate(ranked, 1)
        }
        for trace, row in sorted(candidates, key=lambda pair: pair[0].trial):
            features = row.get("features") or {}
            key = offline.rank_key(features)
            wid = work_identity(run_id, trace)
            print(
                f"{trace.task}\t{trace.trial}\t{trace.trace_hash}\t"
                f"{_reward_text(trace.reward)}\t"
                f"{features.get('verdict') or 'unscoreable'}\t"
                f"{scoring.scalar(features):.6f}\t{key[1]:.6f}\t"
                f"{rank[wid]}",
                flush=True)

    jobs: list[tuple[dict[str, Any], str]] = []
    for task in sorted(by_task):
        task_pending = 0
        for trace in sorted(by_task[task], key=lambda value: value.trial):
            wid = work_identity(run_id, trace)
            if wid in scored:
                continue
            state = sanitized_state(trace, scoring_root, run_id, adapter=adapter)
            record: dict[str, Any] = {
                "dataset": DATASET_NAME,
                "dataset_tree_fingerprint": dataset.tree_fingerprint,
                "adapter_fingerprint": ahash,
                "run_id": run_id,
                "work_id": wid,
                "trace_hash": trace.trace_hash,
                "task": trace.task,
                "trial": trace.trial,
                "rep": args.rep,
                "provider_identity": provider.fields(),
                "text_only": provider.text_only,
                "verifier_config": cfg["config"],
                # Plaintext verifier fields make result audits independent of
                # reverse-engineering the hash.  No reward/gold rides along.
                "model": cfg["model"],
                "seats": cfg["seats"],
                "effort": cfg["effort"],
                "max_rounds": cfg["max_rounds"],
                "critic_rounds": cfg["critic_rounds"],
                "identity": identity,
            }
            jobs.append((record, state))
            task_pending += 1
        if task_pending:
            remaining[task] = task_pending
        else:
            emitted.add(task)
            render(task)

    def on_record(record: dict[str, Any]) -> None:
        wid = record["work_id"]
        scored[wid] = record
        task = record["task"]
        remaining[task] -= 1
        if remaining[task] == 0 and task not in emitted:
            emitted.add(task)
            render(task)

    if jobs:
        print(
            f"[tb:official] scoring {len(jobs)} traces with "
            f"{args.workers} workers; usage={usage_path}",
            file=sys.stderr, flush=True)
        with _agent_usage_ledger(usage_path, run_id):
            (score_many_fn or scoring.score_many)(
                jobs, results_path, workers=args.workers,
                model=model, max_rounds=args.max_rounds,
                seats=args.seats, effort=args.effort,
                critic_rounds=args.critic_rounds,
                service_tier=args.service_tier,
                # Deliberately omit held-out reward from progress/log labels.
                label=lambda row: f"{row['task']}/{row['trial']}",
                tag="tb:official", on_record=on_record)
        # score_many reports failures without fabricating records.  Still emit
        # every surviving candidate for affected tasks.
        for task in sorted(remaining):
            if task not in emitted:
                emitted.add(task)
                render(task)

    print("[tb:official] done", file=sys.stderr, flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    return run(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
