"""Preparation primitives for explicit offline adapter-isolation experiments.

This is not a production scheduling policy. A shared contract experiment fixes
the complete post-extraction, pre-verification graph, then runs both evidence
pipelines afresh. Ordinary online/offline scoring never imports this module.
It must not be used to claim equivalence to ordinary interleaved compilation.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Any

from . import offline
from .core.events import EventLog
from .core.model import (
    CLAIM_EFFECT, CLAIM_FINAL, KINDS, OBLIGATION_KINDS, STATUS_STALE, STATUS_UNVERIFIED,
    content_hash,
)
from .core.world import load_world_view
from .engine.contract.authority_view import build_authority_compilation_view
from .engine.contract.bundle import (
    ContractBundleIdentity, SealedContractBundle, claim_graph_sha256,
)
from .llm.identity import backend_identity, role_identity
from .config import APPEND_ONLY_DELTA_PROTOCOL
from .llm.authority_backend import authority_compilation_backend


SNAPSHOT_POLICY = "post-extract-pre-verification-v1"
MAX_SNAPSHOT_CLAIMS = 5_000
MAX_SOURCE_EVENTS = 100_000
MAX_SOURCE_LOG_BYTES = 256 * 1024 * 1024
_LOADED_REPLAY_TOKEN = object()


class ReplaySetupError(ValueError):
    """A diagnostic cannot establish its fixed, unadjudicated input boundary."""


def clone_sealed_inputs(source: str | Path, destination: str | Path) -> str:
    """Copy only immutable inputs to a new sibling boundary, never judgments.

    Existing destinations are not overwritten, and failures leave the new
    partial directory for diagnosis. Neither caller data nor a previous score
    is deleted to make an experiment appear fresh.
    """
    source = Path(source).resolve(strict=True)
    destination = Path(destination).absolute()
    resolved_destination = destination.resolve()
    if os.path.commonpath((source, resolved_destination)) in (
            os.fspath(source), os.fspath(resolved_destination)):
        raise ReplaySetupError("source and replay destination must be disjoint")
    if destination.exists() or destination.is_symlink():
        raise ReplaySetupError("replay destination must be new")
    seal = offline.verify_immutable_input_anchor(os.fspath(source))
    destination.mkdir(parents=True, exist_ok=False)
    for name in offline.IMMUTABLE_INPUT_PATHS:
        original, target = source / name, destination / name
        if not original.exists() and not original.is_symlink():
            continue
        mode = original.lstat().st_mode
        if stat.S_ISLNK(mode):
            target.symlink_to(os.readlink(original))
        elif stat.S_ISDIR(mode):
            shutil.copytree(original, target, symlinks=True)
        elif stat.S_ISREG(mode):
            shutil.copy2(original, target)
        else:
            raise ReplaySetupError("unsupported immutable input file kind")
    if (offline.verify_immutable_input_anchor(os.fspath(source)) != seal
            or offline.immutable_input_seal(os.fspath(destination)) != seal):
        raise ReplaySetupError("immutable inputs changed while copying")
    offline.write_immutable_input_anchor(os.fspath(destination))
    return seal


def pipeline_sha256() -> str:
    """Full package identity, including materialization and replay semantics."""
    root = Path(__file__).resolve().parent
    sources = {
        os.fspath(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.py"))
        if "__pycache__" not in path.parts
    }
    return content_hash(sources)


def clone_extracted_inputs(source: str | Path, destination: str | Path) -> dict[str, Any]:
    """Rebuild a fresh graph from sealed hooks and one extracted commit set.

    This never copies a ClaimStore, artifact, certificate, or verifier log.
    The source must still be at the post-extraction boundary. A caller must
    not export contracts from a scored arm and use them as a shared setup.
    Every caller-created directory is retained on failure for diagnosis.
    """
    source = Path(source).resolve(strict=True)
    events_path = source / "events.jsonl"
    info = events_path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ReplaySetupError("extracted source log must be a private regular file")
    if info.st_size > MAX_SOURCE_LOG_BYTES:
        raise ReplaySetupError("extracted source log exceeds the size cap")
    events, commits, commit_type, saw_commit = [], [], None, False
    final_response = None
    for event in EventLog(os.fspath(events_path)).replay():
        if len(events) >= MAX_SOURCE_EVENTS:
            raise ReplaySetupError("extracted source log exceeds the event cap")
        if event["type"] in ("commit", "extracted_commit"):
            if commit_type is None:
                commit_type = event["type"]
            elif event["type"] != commit_type:
                raise ReplaySetupError("source mixes extracted commit encodings")
            saw_commit = True
            commits.append(copy.deepcopy(event["payload"]))
        elif offline.is_offline_final_response(event) and not saw_commit \
                and final_response is None:
            final_response = event["payload"]["text"]
        elif event["type"] not in offline.HOOK_EVENT_TYPES or saw_commit \
                or final_response is not None:
            raise ReplaySetupError("source is not a fresh post-extraction ledger")
        events.append({key: copy.deepcopy(event[key])
                       for key in ("seq", "type", "payload")})
    if not commits:
        raise ReplaySetupError("extracted source contains no commits")
    metadata = offline._read_regular_json(os.fspath(source / "extract.json"))
    seal = offline.verify_immutable_input_anchor(os.fspath(source))
    if metadata.get("immutable_input_seal") != seal:
        raise ReplaySetupError("extracted source is not bound to its sealed inputs")
    rounds = metadata.get("rounds")
    if type(rounds) is not int or rounds < 1:
        raise ReplaySetupError("extraction attempt count must be retained")
    clone_sealed_inputs(source, destination)
    destination = Path(destination).resolve(strict=True)
    offline._write_json_atomic(
        os.fspath(destination / offline.DIAGNOSTIC_ONLY_FILE),
        {"diagnostic_only": SNAPSHOT_POLICY})
    log = EventLog(os.fspath(destination / "events.jsonl"))
    offline.ingest_feed(os.fspath(destination / "hook_feed.jsonl"), log)
    if final_response is not None:
        if offline._read(os.fspath(destination / "final.txt")) != final_response:
            raise ReplaySetupError(
                "retained final response differs from sealed final input")
        offline.append_final_response(log, final_response)
    error = offline._apply_commits(log, commits, event_type=commit_type)
    if error:
        raise ReplaySetupError(f"extracted commit graph is invalid: {error}")
    actual = [{key: event[key] for key in ("seq", "type", "payload")}
              for event in log.replay()]
    if actual != events:
        raise ReplaySetupError("extracted ledger differs from its sealed source hooks")
    # Keep distinct provenance as well as the durable refusal marker. This
    # metadata alone is not sufficient isolation: stale proof artifacts can
    # survive an ordinary re-extraction, so score() refuses marked states.
    stamp = {
        "rounds": rounds, "commits": len(commits),
        "fingerprint": offline.pipeline_fingerprint(),
        "config": content_hash({"diagnostic_only": SNAPSHOT_POLICY}),
        "immutable_input_seal": seal,
        "diagnostic_only": SNAPSHOT_POLICY,
        "source_events_sha256": content_hash(events),
    }
    offline._write_json_atomic(os.fspath(destination / "extract.json"), stamp)
    if offline.verify_immutable_input_anchor(os.fspath(source)) != seal:
        raise ReplaySetupError("source inputs changed during graph reconstruction")
    return stamp


def compilation_config_sha256(kernel, config: dict[str, Any]) -> str:
    """Bind all semantic configuration except the executor treatment switch.

    No model/effort/role/schema/rounds difference is normalized away. The
    browser observer and typed-adapter switch belong to the later Oracle and
    executor, not the shared Compiler/Critic stage.
    """
    semantic = {key: value for key, value in config.items()
                if key not in {"config", "typed_adapters", "browser_observer"}}
    configured_delta = config.get("append_only_deltas", False)
    configured_protocol = config.get("append_only_delta_protocol")
    if (type(configured_delta) is not bool
            or configured_delta != getattr(
                kernel, "append_only_deltas", False)
            or configured_protocol != (
                APPEND_ONLY_DELTA_PROTOCOL if configured_delta else None)
            or configured_protocol != getattr(
                kernel, "append_only_delta_protocol", None)):
        raise ReplaySetupError(
            "compiler append-repair protocol differs from configuration")
    # Normalize legacy fixture dictionaries which omit the disabled flag so
    # explicit false and implicit default false have one exact identity.
    semantic["append_only_deltas"] = configured_delta
    semantic["append_only_delta_protocol"] = configured_protocol
    backend = backend_identity(kernel.backend)
    if backend is None or kernel.critic_role is None or kernel.critic_rounds < 1:
        raise ReplaySetupError("known compiler backend and independent Critic required")
    if kernel.memory is not None or kernel.memory_hits != "[]":
        raise ReplaySetupError("shared preparation cannot import advisory memory")
    # Capture the same effective authority roles, runtime, date, and capability
    # policy that the actual final Compiler/Critic session uses. Creating this
    # private session does not call a model or change the candidate backend.
    with authority_compilation_backend(
            kernel.backend, kernel.obligations_role, kernel.critic_role) as session:
        authority = {
            "backend": copy.deepcopy(session.identity),
            "compiler": role_identity(session.obligations_role),
            "critic": role_identity(session.critic_role),
        }
        session.revalidate()
    return content_hash({
        "policy": SNAPSHOT_POLICY, "config": semantic, "backend": backend,
        "authority": authority,
        "compiler": role_identity(kernel.obligations_role),
        "critic": role_identity(kernel.critic_role),
        "critic_rounds": kernel.critic_rounds,
    })


def fresh_graph_snapshot(kernel, final_id: str) -> dict[str, Any]:
    """Inspect the complete graph before any evidence work is permitted.

    Revision-induced stale nodes are declarative input state, not replayed
    adjudication. Actual verdicts, obligation nodes, and evidence are refused.
    Call only while the prepare/replay coordinator owns the idle kernel.
    """
    if kernel.live_world:
        raise ReplaySetupError("sealed contract diagnostics are offline only")
    if kernel.store.get(final_id)["kind"] != CLAIM_FINAL:
        raise ReplaySetupError("snapshot requires a final claim")
    for index, event in enumerate(kernel.log.replay()):
        if index >= MAX_SOURCE_EVENTS:
            raise ReplaySetupError("snapshot source ledger exceeds the event cap")
        if event["type"] not in (
                *offline.HOOK_EVENT_TYPES, "message", "commit",
                "extracted_commit"):
            raise ReplaySetupError("snapshot ledger contains verifier-generated context")
        if event["type"] == "message" \
                and not offline.is_offline_final_response(event):
            raise ReplaySetupError("snapshot contains an invalid offline response")
    for table in ("evidence", "obligation_edges"):
        if kernel.store.db.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]:
            raise ReplaySetupError("snapshot contains prior verification state")
    ids = [row[0] for row in kernel.store.db.execute(
        "SELECT id FROM claims ORDER BY rowid LIMIT ?", (MAX_SNAPSHOT_CLAIMS + 1,))]
    if len(ids) > MAX_SNAPSHOT_CLAIMS:
        raise ReplaySetupError("claim graph exceeds the snapshot cap")
    claims = []
    for claim_id in ids:
        row = kernel.store.get(claim_id)
        if (row["kind"] not in KINDS
                or row["status"] not in (STATUS_UNVERIFIED, STATUS_STALE)
                or row.get("basis") is not None or row.get("strength") is not None):
            raise ReplaySetupError("snapshot contains an adjudicated or unknown node")
        claims.append({
            "id": claim_id, "kind": row["kind"], "text": row["text"],
            "deps": sorted(kernel.store.parents(claim_id)),
            "revises": row.get("revises"), "initial_status": row["status"],
        })
    return {
        "policy": SNAPSHOT_POLICY, "final_id": final_id, "claims": claims,
        "slice": kernel.store.backward_slice(final_id),
    }


def _local_locator(path: str, state: str) -> str:
    """Normalize a designated infrastructure slot, not arbitrary claim text."""
    if path == "":
        return path
    if not isinstance(path, str) or not os.path.isabs(path):
        raise ReplaySetupError("compiler locator must be an absolute path")
    absolute = os.path.abspath(path)
    if os.path.commonpath((state, absolute)) != state:
        raise ReplaySetupError("compiler locator escapes its sealed state")
    return "<STATE>/" + os.path.relpath(absolute, state).replace(os.sep, "/")


def normalized_compile_inputs(kernel, row: dict[str, Any]) -> dict[str, Any]:
    """Path-independent identity of the original semantic compile snapshot.

    Preserve claim wording, quantifiers, premise status, graph context and the
    existing authenticated effect call/result references. Only known harness
    locator fields are normalized. Never fuzzy-match or rewrite embedded text.
    """
    state = os.path.abspath(os.path.dirname(kernel.log.path))
    if row["kind"] == CLAIM_FINAL:
        view = build_authority_compilation_view(state)
        return copy.deepcopy(kernel.contracts.authority_contract_variables(view))
    variables = copy.deepcopy(kernel.contracts.compile_variables(row))
    if variables["ledger_path"] != os.path.join(state, "events.jsonl"):
        raise ReplaySetupError("unexpected compiler ledger locator")
    variables["ledger_path"] = _local_locator(variables["ledger_path"], state)
    variables["world_path"] = _local_locator(variables["world_path"], state)
    try:
        task = json.loads(variables["task_context"])
    except (TypeError, ValueError) as error:
        raise ReplaySetupError("diagnostic requires an inlined authoritative task") from error
    if not isinstance(task, dict) or task.get("path") != os.path.join(state, "task.md"):
        raise ReplaySetupError("unexpected compiler task locator")
    task["path"] = "<STATE>/task.md"
    variables["task_context"] = json.dumps(task, ensure_ascii=False)
    description = json.loads(variables["world_map"])
    world = load_world_view(state, live_world=False)
    if description != world.describe():
        raise ReplaySetupError("compiler world description does not match the sealed mapping")
    if "host_root" in description:
        description["host_root"] = _local_locator(description["host_root"], state)
    if world.mode == "captured" and world.scheme == "filesystem":
        # This field is generated by WorldView, unlike the task's free-text
        # note. Check equality above before replacing its infrastructure root.
        description["address_mapping"] = (
            "a task address A under a captured root lives at "
            f"{description['host_root']}/A on this host; addresses outside "
            "every captured root were not captured and their absence proves nothing")
    for item in description.get("artifacts", []):
        item["host"] = _local_locator(item["host"], state)
    variables["world_map"] = json.dumps(description, ensure_ascii=False, sort_keys=True)
    return variables


def compile_input_digests(kernel, snapshot: dict[str, Any]) -> dict[str, str]:
    """Compute every expected digest once, before any premise can settle."""
    if fresh_graph_snapshot(kernel, snapshot["final_id"]) != snapshot:
        raise ReplaySetupError("claim graph changed after snapshot selection")
    return {
        claim_id: content_hash(normalized_compile_inputs(
            kernel, kernel.store.get(claim_id)))
        for claim_id in snapshot["slice"]
        if kernel.store.get(claim_id)["kind"] != CLAIM_EFFECT
    }


def replay_identity(kernel, snapshot: dict[str, Any], config: dict[str, Any]) \
        -> ContractBundleIdentity:
    """Derive trust inputs from this fresh arm, never from bundle declarations."""
    if fresh_graph_snapshot(kernel, snapshot["final_id"]) != snapshot:
        raise ReplaySetupError("claim graph changed before bundle validation")
    state = os.path.dirname(os.path.abspath(kernel.log.path))
    claims = [{key: row[key] for key in ("id", "kind", "text", "deps", "revises")}
              for row in snapshot["claims"]]
    return ContractBundleIdentity(
        pipeline_sha256=pipeline_sha256(),
        config_sha256=compilation_config_sha256(kernel, config),
        source_sha256=offline.verify_immutable_input_anchor(state),
        graph_sha256=claim_graph_sha256(claims),
    )


def load_replay_contracts(kernel, final_id: str, config: dict[str, Any],
                          raw: bytes, *, expected_sha256: str) -> "KernelBoundReplay":
    """Validate an operator-pinned bundle before any arm evidence work starts.

    Returns a provider but does not attach it or run verification. The explicit
    diagnostic coordinator remains responsible for private bundle storage,
    complete setup charging and result provenance. Normal scoring never calls
    this function. A failed setup is not converted into a partial replay.
    """
    graph = fresh_graph_snapshot(kernel, final_id)
    identity = replay_identity(kernel, graph, config)
    inputs = compile_input_digests(kernel, graph)
    bundle = SealedContractBundle.from_bytes(
        raw, expected_sha256=expected_sha256, expected_identity=identity,
        expected_compile_inputs=inputs)
    bundle.require_replayable()
    if replay_identity(kernel, graph, config) != identity:
        raise ReplaySetupError("replay identity changed during bundle validation")
    return KernelBoundReplay(kernel, bundle, graph, config,
                             _token=_LOADED_REPLAY_TOKEN)


class KernelBoundReplay:
    """An operator-loaded contract bound to one exact idle offline kernel.

    Only declarative claim structure is immutable during execution. Premise
    statuses, evidence and materialized obligation nodes are expected to grow;
    they are not imported from the bundle or used to re-hash compiler inputs.
    The diagnostic coordinator must call ``validate_inputs`` again after
    evidence work and before exporting its result.
    """

    def __init__(self, kernel, bundle, snapshot, config, *, _token=None):
        if _token is not _LOADED_REPLAY_TOKEN or not isinstance(bundle, SealedContractBundle):
            raise ReplaySetupError("bound replay providers must be loaded from operator-pinned bytes")
        self._kernel, self._store, self._log = kernel, kernel.store, kernel.log
        self._bundle = bundle
        self._snapshot = copy.deepcopy(snapshot)
        self._config = copy.deepcopy(config)

    @property
    def sha256(self):
        return self._bundle.sha256

    @property
    def identity(self):
        return self._bundle.identity

    @property
    def replayable(self):
        return self._bundle.replayable

    def require_replayable(self):
        self._bundle.require_replayable()

    def validate_kernel(self, kernel):
        if (kernel is not self._kernel or kernel.store is not self._store
                or kernel.log is not self._log or kernel.live_world):
            raise ReplaySetupError("contract replay is bound to another kernel, store, or log")
        state = os.path.dirname(os.path.abspath(self._log.path))
        if not os.path.lexists(os.path.join(state, offline.DIAGNOSTIC_ONLY_FILE)):
            raise ReplaySetupError("replay execution requires a diagnostic-only state marker")
        # The normal store lock makes this structural read coherent with
        # parallel evidence materialization; it does not serialize model work.
        with self._store._lock:
            rows = self._store.db.execute(
                "SELECT id, kind FROM claims WHERE kind != ? ORDER BY rowid LIMIT ?",
                (OBLIGATION_KINDS, MAX_SNAPSHOT_CLAIMS + 1)).fetchall()
            if len(rows) > MAX_SNAPSHOT_CLAIMS or any(row[1] not in KINDS for row in rows):
                raise ReplaySetupError("claim structure changed during diagnostic replay")
            claims = []
            for claim_id, _kind in rows:
                row = self._store.get(claim_id)
                claims.append({
                    "id": claim_id, "kind": row["kind"], "text": row["text"],
                    "deps": sorted(self._store.parents(claim_id)),
                    "revises": row.get("revises"),
                })
            if (claim_graph_sha256(claims) != self.identity.graph_sha256
                    or self._store.backward_slice(self._snapshot["final_id"])
                    != self._snapshot["slice"]):
                raise ReplaySetupError("claim structure changed during diagnostic replay")

    def validate_inputs(self):
        """Coordinator's before/after guard; no model calls or result reuse."""
        self.validate_kernel(self._kernel)
        state = os.path.dirname(os.path.abspath(self._log.path))
        if (offline.verify_immutable_input_anchor(state) != self.identity.source_sha256
                or pipeline_sha256() != self.identity.pipeline_sha256
                or compilation_config_sha256(self._kernel, self._config)
                != self.identity.config_sha256):
            raise ReplaySetupError("sealed replay input or configuration identity changed")

    def lookup(self, claim_id):
        self.validate_kernel(self._kernel)
        return self._bundle.lookup(claim_id)

    def validate_materialized(self, entry, obligations):
        self.validate_kernel(self._kernel)
        return self._bundle.validate_materialized(entry, obligations)


def attach_replay_contracts(kernel, provider: KernelBoundReplay) -> None:
    """Explicit, one-time activation immediately before diagnostic execution."""
    if not isinstance(provider, KernelBoundReplay):
        raise ReplaySetupError("replay activation requires an operator-loaded bound provider")
    if kernel.contract_replay is not None:
        raise ReplaySetupError("a replay provider is already attached")
    provider.validate_kernel(kernel)
    if fresh_graph_snapshot(kernel, provider._snapshot["final_id"]) != provider._snapshot:
        raise ReplaySetupError("replay activation requires the original unadjudicated graph")
    inputs = compile_input_digests(kernel, provider._snapshot)
    if any(provider._bundle.lookup(claim_id).compile_input_sha256 != digest
           for claim_id, digest in inputs.items()):
        raise ReplaySetupError("compiler inputs changed before replay activation")
    provider.validate_inputs()
    provider.require_replayable()
    kernel.contract_replay = provider
