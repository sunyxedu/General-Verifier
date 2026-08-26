"""Immutable candidate seals and reward-safe best-of-N evaluation.

The seal is the join boundary between generation, scoring, and evaluation:

* ``identity`` binds the solver and all candidate inputs;
* ``evaluation`` is private post-hoc metadata and is never returned by
  :func:`verifier_view`;
* scorer outputs refer to ``candidate_id`` only; and
* :func:`evaluate_bon` rejects partial or mixed candidate pools.

The module is benchmark-agnostic.  A JSONL command-line interface is provided
for experiment drivers that do not want to import it directly::

    python -m experiments.common.candidate_seal freeze \
      --specs completed.jsonl --seals private-seals.jsonl \
      --verifier-inputs scorer-inputs.jsonl
    python -m experiments.common.candidate_seal bon \
      --seals private-seals.jsonl --scores scorer-scores.jsonl --expected-n 5

Candidate specs must name ``dataset``, ``dataset_digest`` (or
``dataset_digest_path``), ``task``, ``task_digest`` (or ``task_digest_path``),
``trial``, ``model``, ``effort``, and ``trajectory_path``.
``service_tier`` is optional and defaults to ``standard``; the standard tier
is deliberately omitted from the sealed solver identity for compatibility
with seals created before service tiers were recorded, while non-standard
tiers remain identity- and pool-separating.
``instruction_path``, ``task_config_path``, ``task_package_digest``, and
``artifact_manifest_path`` are optional independent bindings.  ``world_path``
may be null for releases that do not contain a completed filesystem snapshot;
adapters which hash a logical (rather than physically assembled) world may
supply ``world_digest`` instead.  Adapters which retain an authoritative raw
solver transcript may additionally supply ``raw_trajectory_digest`` (or
``raw_trajectory_path``); it is bound as ``identity.trajectory.raw_sha256``
without changing legacy identities which omit it.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
import stat
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


SEAL_SCHEMA = "candidate-seal/v1"
IDENTITY_SCHEMA = "candidate-identity/v1"
VERIFIER_VIEW_SCHEMA = "candidate-verifier-input/v1"
BON_SCHEMA = "bon-top1/v1"
TREE_SCHEMA = "candidate-tree/v1"
SYMLINK_SCHEMA = "candidate-symlink/v1"
UNAVAILABLE_WORLD = {
    "status": "unavailable",
    "sha256": hashlib.sha256(
        b'{"schema":"candidate-world/v1","status":"unavailable"}'
    ).hexdigest(),
}


class SealError(ValueError):
    """A candidate, seal, or score record violates the identity contract."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def stable_digest(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


def file_digest(path: str | os.PathLike[str]) -> str:
    """Hash a stable regular file, rejecting a concurrent modification."""
    path = os.fspath(path)
    if os.path.islink(path) or not os.path.isfile(path):
        raise SealError(f"expected a regular file: {path}")
    with open(path, "rb") as handle:
        before = os.fstat(handle.fileno())
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
        after = os.fstat(handle.fileno())
    fingerprint = lambda value: (  # noqa: E731 - compact immutable stat tuple
        value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
        value.st_ctime_ns, stat.S_IMODE(value.st_mode),
    )
    if fingerprint(before) != fingerprint(after):
        raise SealError(f"file changed while hashing: {path}")
    return digest.hexdigest()


def _tree_records(root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        directory = Path(dirpath)
        relative_dir = directory.relative_to(root).as_posix()
        if relative_dir == ".":
            relative_dir = ""
        # os.walk puts directory symlinks in dirnames.  Record them as links and
        # remove them so their targets never enter the candidate world.
        for name in list(dirnames):
            path = directory / name
            if path.is_symlink():
                dirnames.remove(name)
                relative = (Path(relative_dir) / name).as_posix()
                records.append({
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                })
        dirnames.sort()
        filenames.sort()
        if relative_dir:
            mode = stat.S_IMODE(os.lstat(directory).st_mode)
            records.append({
                "path": relative_dir, "kind": "directory", "mode": mode,
            })
        for name in filenames:
            path = directory / name
            relative = (Path(relative_dir) / name).as_posix()
            info = os.lstat(path)
            if stat.S_ISLNK(info.st_mode):
                records.append({
                    "path": relative,
                    "kind": "symlink",
                    "target": os.readlink(path),
                })
            elif stat.S_ISREG(info.st_mode):
                records.append({
                    "path": relative,
                    "kind": "file",
                    "mode": stat.S_IMODE(info.st_mode),
                    "bytes": info.st_size,
                    "sha256": file_digest(path),
                })
            else:
                raise SealError(f"unsupported world entry: {path}")
    return sorted(records, key=lambda row: (row["path"], row["kind"]))


def tree_digest(path: str | os.PathLike[str]) -> str:
    """Location-independent hash of file contents, modes, and symlinks."""
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise SealError(f"expected a real directory tree: {root}")
    # A second traversal is an inexpensive fail-closed race check relative to
    # the verifier work that follows.  It also catches added/removed files.
    first = _tree_records(root)
    second = _tree_records(root)
    if first != second:
        raise SealError(f"directory changed while hashing: {root}")
    return stable_digest({"schema": TREE_SCHEMA, "entries": first})


def symlink_digest(path: str | os.PathLike[str]) -> str:
    """Hash a symlink node without following or reading its target."""
    path = os.fspath(path)

    def snapshot() -> tuple[tuple[int, int, int, int, int, int], str]:
        try:
            info = os.lstat(path)
            target = os.readlink(path)
        except OSError as error:
            raise SealError(f"cannot read symlink {path}: {error}") from error
        if not stat.S_ISLNK(info.st_mode):
            raise SealError(f"expected a symlink: {path}")
        fingerprint = (
            info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, stat.S_IMODE(info.st_mode),
        )
        return fingerprint, target

    first = snapshot()
    second = snapshot()
    if first != second:
        raise SealError(f"symlink changed while hashing: {path}")
    return stable_digest({"schema": SYMLINK_SCHEMA, "target": first[1]})


def path_digest(path: str | os.PathLike[str]) -> str:
    path = Path(path)
    if path.is_dir() and not path.is_symlink():
        return tree_digest(path)
    return file_digest(path)


def trajectory_digest(path: str | os.PathLike[str]) -> str:
    """Hash a candidate trajectory while excluding top-level held-out reward.

    Official trajectory releases commonly store ``reward`` beside the actual
    trajectory.  Canonical JSON hashing makes label corrections and whitespace
    changes irrelevant to candidate identity.  Non-JSON rollouts are hashed as
    exact bytes.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except OSError as error:
        raise SealError(f"cannot read trajectory {path}: {error}") from error
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return file_digest(path)
    if isinstance(value, dict):
        value = dict(value)
        value.pop("reward", None)
    return stable_digest(value)


def _required_text(spec: Mapping[str, Any], field: str) -> str:
    value = spec.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SealError(f"{field} must be a non-empty string")
    if any(character in value for character in "\x00\r\n\t"):
        raise SealError(f"{field} contains a forbidden control character")
    return value.strip()


def _sha256(value: Any, field: str) -> str:
    if (not isinstance(value, str) or len(value) != 64
            or any(character not in "0123456789abcdef" for character in value)):
        raise SealError(f"{field} must be a lowercase SHA-256 hex digest")
    return value


def _finite_number(value: Any, field: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SealError(f"{field} must be numeric or null")
    result = float(value)
    if not math.isfinite(result):
        raise SealError(f"{field} must be finite")
    return result


def _pool_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields shared by all samples from one solver/task pool.

    A completed-candidate identity deliberately binds its exact Harbor trial
    config.  Harbor's per-trial config includes ``trial_name`` and therefore
    has a different hash for every replicate; that hash belongs in
    ``candidate_id`` but must not split a five-sample BoN pool.  Task package,
    checksum, and instruction hashes are task-wide and remain pool bindings.
    """
    task = identity.get("task")
    stable_task = task
    if isinstance(task, Mapping):
        stable_task = {
            "name": task.get("name"),
            "sha256": task.get("sha256"),
            "package_sha256": task.get("package_sha256"),
            "instruction_sha256": task.get("instruction_sha256"),
        }
    return {
        "schema": "candidate-pool/v2",
        "dataset": identity.get("dataset"),
        "task": stable_task,
        "solver": identity.get("solver"),
    }


def _source_identity(identity: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "candidate-source/v1",
        "dataset": identity.get("dataset"),
        "task": identity.get("task"),
        "trial": identity.get("trial"),
    }


def candidate_source_id(identity: Mapping[str, Any]) -> str:
    """Return the immutable dataset/task/trial source identifier.

    Adapters may construct a metadata-only identity containing these fields to
    decide whether an already-frozen source can safely skip rediscovery of a
    pruned trajectory or world.  The resulting identifier is exactly the one
    validated on a complete seal.
    """
    return stable_digest(_source_identity(identity))


def _contains_gold_key(value: Any) -> bool:
    if isinstance(value, dict):
        if any(key in value for key in ("reward", "tb_reward", "gold",
                                        "evaluation")):
            return True
        return any(_contains_gold_key(child) for child in value.values())
    if isinstance(value, list):
        return any(_contains_gold_key(child) for child in value)
    return False


def _resolved_path(base_dir: Path, value: Any, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SealError(f"{field} must be a path string")
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (base_dir / path).resolve()


def _digest_field(
    spec: Mapping[str, Any], digest_field: str, path_field: str,
    base_dir: Path, *, required: bool,
) -> str | None:
    supplied_digest = spec.get(digest_field)
    supplied_path = spec.get(path_field)
    if supplied_digest is not None and supplied_path is not None:
        actual = path_digest(_resolved_path(base_dir, supplied_path, path_field))
        expected = _sha256(supplied_digest, digest_field)
        if actual != expected:
            raise SealError(f"{digest_field} does not match {path_field}")
        return expected
    if supplied_digest is not None:
        return _sha256(supplied_digest, digest_field)
    if supplied_path is not None:
        return path_digest(_resolved_path(base_dir, supplied_path, path_field))
    if required:
        raise SealError(f"provide {digest_field} or {path_field}")
    return None


def seal_from_spec(
    spec: Mapping[str, Any], *, base_dir: str | os.PathLike[str] = ".",
) -> dict[str, Any]:
    """Create one deterministic seal from a completed-candidate spec."""
    base = Path(base_dir).resolve()
    dataset_name = _required_text(spec, "dataset")
    dataset_digest = _digest_field(
        spec, "dataset_digest", "dataset_digest_path", base, required=True)
    assert dataset_digest is not None
    task = _required_text(spec, "task")
    task_hash = _digest_field(
        spec, "task_digest", "task_digest_path", base, required=True)
    task_package_hash = _digest_field(
        spec, "task_package_digest", "task_package_path", base,
        required=False)
    trial = _required_text(spec, "trial")
    model = _required_text(spec, "model")
    effort = _required_text(spec, "effort")
    service_tier = spec.get("service_tier", "standard")
    if service_tier not in ("standard", "fast"):
        raise SealError("service_tier must be 'standard' or 'fast'")
    instruction_hash = _digest_field(
        spec, "instruction_digest", "instruction_path", base, required=False)
    config_hash = _digest_field(
        spec, "task_config_digest", "task_config_path", base, required=False)
    artifact_manifest_hash = _digest_field(
        spec, "artifact_manifest_digest", "artifact_manifest_path", base,
        required=False)
    trajectory_path = _resolved_path(
        base, spec.get("trajectory_path"), "trajectory_path")
    trajectory_hash = trajectory_digest(trajectory_path)
    raw_trajectory_hash = _digest_field(
        spec, "raw_trajectory_digest", "raw_trajectory_path", base,
        required=False)

    world_value = spec.get("world_path")
    supplied_world_hash = spec.get("world_digest")
    if world_value is not None and supplied_world_hash is not None:
        raise SealError("provide world_path or world_digest, not both")
    if supplied_world_hash is not None:
        world = {
            "status": "captured",
            "sha256": _sha256(supplied_world_hash, "world_digest"),
        }
    elif world_value is None:
        world = dict(UNAVAILABLE_WORLD)
    else:
        world_path = _resolved_path(base, world_value, "world_path")
        world = {"status": "captured", "sha256": tree_digest(world_path)}
    if artifact_manifest_hash is not None:
        world["manifest_sha256"] = artifact_manifest_hash

    solver = {"model": model, "effort": effort}
    # Preserve legacy standard-tier candidate and pool identities exactly.
    # A non-standard tier is scientifically a different sampling arm and must
    # therefore split both identities.
    if service_tier != "standard":
        solver["service_tier"] = service_tier

    trajectory = {"sha256": trajectory_hash}
    if raw_trajectory_hash is not None:
        trajectory["raw_sha256"] = raw_trajectory_hash

    identity: dict[str, Any] = {
        "schema": IDENTITY_SCHEMA,
        "state": "completed",
        "dataset": {"name": dataset_name, "sha256": dataset_digest},
        "task": {
            "name": task,
            "sha256": task_hash,
            "package_sha256": task_package_hash,
            "instruction_sha256": instruction_hash,
            "config_sha256": config_hash,
        },
        "trial": trial,
        "solver": solver,
        "trajectory": trajectory,
        "world": world,
    }
    seal: dict[str, Any] = {
        "schema": SEAL_SCHEMA,
        "candidate_id": stable_digest(identity),
        "pool_id": stable_digest(_pool_identity(identity)),
        "source_id": candidate_source_id(identity),
        "identity": identity,
        # This is deliberately the only home for held-out correctness.
        "evaluation": {"reward": _finite_number(spec.get("reward"), "reward")},
    }
    seal["seal_id"] = stable_digest(seal)
    validate_seal(seal)
    return seal


def validate_seal(seal: Mapping[str, Any]) -> None:
    if seal.get("schema") != SEAL_SCHEMA:
        raise SealError("candidate seal has the wrong schema")
    identity = seal.get("identity")
    evaluation = seal.get("evaluation")
    if not isinstance(identity, dict) or not isinstance(evaluation, dict):
        raise SealError("candidate seal needs identity and evaluation objects")
    if identity.get("schema") != IDENTITY_SCHEMA:
        raise SealError("candidate identity has the wrong schema")
    dataset = identity.get("dataset")
    task = identity.get("task")
    solver = identity.get("solver")
    trajectory = identity.get("trajectory")
    world = identity.get("world")
    if not all(isinstance(value, dict) for value in (
            dataset, task, solver, trajectory, world)):
        raise SealError("candidate identity has malformed components")
    _sha256(dataset.get("sha256"), "identity.dataset.sha256")
    _sha256(task.get("sha256"), "identity.task.sha256")
    for field in ("package_sha256", "instruction_sha256", "config_sha256"):
        value = task.get(field)
        if value is not None:
            _sha256(value, f"identity.task.{field}")
    if set(solver) not in (
            {"model", "effort"}, {"model", "effort", "service_tier"}):
        raise SealError("identity.solver has unsupported fields")
    _required_text(solver, "model")
    _required_text(solver, "effort")
    if solver.get("service_tier") != "fast" and "service_tier" in solver:
        raise SealError(
            "identity.solver.service_tier must be omitted for standard or be 'fast'")
    if set(trajectory) not in ({"sha256"}, {"sha256", "raw_sha256"}):
        raise SealError("identity.trajectory has unsupported fields")
    _sha256(trajectory.get("sha256"), "identity.trajectory.sha256")
    if trajectory.get("raw_sha256") is not None:
        _sha256(
            trajectory.get("raw_sha256"),
            "identity.trajectory.raw_sha256")
    if world.get("status") not in ("captured", "unavailable"):
        raise SealError("identity.world.status is invalid")
    _sha256(world.get("sha256"), "identity.world.sha256")
    if world.get("manifest_sha256") is not None:
        _sha256(
            world.get("manifest_sha256"),
            "identity.world.manifest_sha256")
    if seal.get("candidate_id") != stable_digest(identity):
        raise SealError("candidate_id does not match identity")
    if seal.get("pool_id") != stable_digest(_pool_identity(identity)):
        raise SealError("pool_id does not match candidate pool identity")
    if seal.get("source_id") != candidate_source_id(identity):
        raise SealError("source_id does not match candidate source identity")
    unsigned = dict(seal)
    seal_id = unsigned.pop("seal_id", None)
    if seal_id != stable_digest(unsigned):
        raise SealError("seal_id does not match the complete private seal")
    if set(evaluation) != {"reward"}:
        raise SealError("evaluation object may contain only reward")
    _finite_number(evaluation.get("reward"), "evaluation.reward")
    if _contains_gold_key(identity):
        raise SealError("evaluation data is forbidden inside candidate identity")


def verifier_view(seal: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact reward-free manifest safe to give a scorer."""
    validate_seal(seal)
    return {
        "schema": VERIFIER_VIEW_SCHEMA,
        "candidate_id": seal["candidate_id"],
        "pool_id": seal["pool_id"],
        "identity": seal["identity"],
    }


def _load_jsonl(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    rows = []
    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SealError(f"invalid JSONL at {path}:{line_number}: {error}") \
                    from error
            if not isinstance(row, dict):
                raise SealError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


# Public aliases for small benchmark adapters.  The underscored names remain
# for compatibility with the original single-file CLI implementation.
load_jsonl = _load_jsonl


def _write_jsonl_atomic(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(canonical_bytes(row).decode("utf-8"))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = ""
    finally:
        if temporary and os.path.exists(temporary):
            os.remove(temporary)


write_jsonl_atomic = _write_jsonl_atomic


def merge_seals(
    existing: Iterable[dict[str, Any]],
    incoming: Iterable[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Pure immutable merge used by single- and multi-file freeze adapters."""
    existing = list(existing)
    incoming = list(incoming)
    for seal in incoming:
        validate_seal(seal)
    by_source: dict[str, dict[str, Any]] = {}
    by_candidate: dict[str, dict[str, Any]] = {}
    for seal in [*existing, *incoming]:
        validate_seal(seal)
        source_id = seal["source_id"]
        candidate_id = seal["candidate_id"]
        prior_source = by_source.get(source_id)
        if prior_source is not None and prior_source != seal:
            raise SealError(
                "attempted to change a frozen dataset/task/trial candidate")
        prior_candidate = by_candidate.get(candidate_id)
        if prior_candidate is not None and prior_candidate != seal:
            raise SealError("candidate identity has conflicting private metadata")
        by_source[source_id] = seal
        by_candidate[candidate_id] = seal
    return sorted(by_source.values(), key=lambda row: (
        row["pool_id"], row["identity"]["trial"], row["candidate_id"]))


def freeze_seals(path: str | os.PathLike[str], seals: Iterable[dict[str, Any]]) \
        -> list[dict[str, Any]]:
    """Append new seals atomically; reject mutation of a frozen source id."""
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = Path(f"{path}.lock")
    incoming = list(seals)
    for seal in incoming:
        validate_seal(seal)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        existing = _load_jsonl(path) if path.exists() else []
        merged = merge_seals(existing, incoming)
        _write_jsonl_atomic(path, merged)
        return merged


def evaluate_bon(
    seals: Iterable[Mapping[str, Any]],
    scores: Iterable[Mapping[str, Any]], *,
    expected_n: int | None = None,
    higher_is_better: bool = True,
    correctness_threshold: float = 1.0,
) -> dict[str, Any]:
    """Compute deterministic, per-task BoN top-1 correctness for each scorer.

    A score row has exactly the join fields ``candidate_id``, ``scorer``, and
    numeric ``score`` (extra scorer metadata is allowed, held-out fields are
    not).  A scorer may omit whole pools, but once it scores one member it must
    score every candidate in that sealed pool.
    """
    if expected_n is not None and expected_n < 1:
        raise SealError("expected_n must be at least one")
    threshold = _finite_number(correctness_threshold, "correctness_threshold")
    assert threshold is not None

    by_candidate: dict[str, Mapping[str, Any]] = {}
    pools: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for seal in seals:
        validate_seal(seal)
        candidate_id = seal["candidate_id"]
        if candidate_id in by_candidate:
            raise SealError(f"duplicate candidate seal: {candidate_id}")
        by_candidate[candidate_id] = seal
        pools[seal["pool_id"]].append(seal)
    if not by_candidate:
        raise SealError("no candidate seals")
    if expected_n is not None:
        for pool_id, candidates in pools.items():
            if len(candidates) != expected_n:
                raise SealError(
                    f"pool {pool_id} has {len(candidates)} candidates, "
                    f"expected {expected_n}")

    scorer_scores: dict[str, dict[str, float]] = defaultdict(dict)
    for row in scores:
        if any(field in row for field in ("reward", "tb_reward", "gold",
                                          "evaluation")):
            raise SealError("held-out evaluation data is forbidden in score rows")
        candidate_id = row.get("candidate_id")
        scorer = row.get("scorer")
        if candidate_id not in by_candidate:
            raise SealError(f"score references an unknown candidate: {candidate_id}")
        if not isinstance(scorer, str) or not scorer.strip():
            raise SealError("score row needs a non-empty scorer")
        value = _finite_number(row.get("score"), "score")
        if value is None:
            raise SealError("score may not be null")
        if candidate_id in scorer_scores[scorer]:
            raise SealError(f"duplicate score for {scorer}/{candidate_id}")
        scorer_scores[scorer][candidate_id] = value
    if not scorer_scores:
        raise SealError("no scorer rows")

    summaries: dict[str, Any] = {}
    for scorer, values in sorted(scorer_scores.items()):
        selections = []
        for pool_id, candidates in sorted(pools.items()):
            present = [row for row in candidates if row["candidate_id"] in values]
            if not present:
                continue
            if len(present) != len(candidates):
                raise SealError(
                    f"scorer {scorer} has partial coverage for pool {pool_id}: "
                    f"{len(present)}/{len(candidates)}")
            rewards = []
            for candidate in candidates:
                reward = candidate["evaluation"].get("reward")
                reward = _finite_number(reward, "evaluation.reward")
                if reward is None:
                    raise SealError(
                        f"candidate {candidate['candidate_id']} has no reward")
                rewards.append(reward)
            ordered = sorted(candidates, key=lambda candidate: (
                -values[candidate["candidate_id"]]
                if higher_is_better else values[candidate["candidate_id"]],
                candidate["identity"]["trial"],
                candidate["candidate_id"],
            ))
            selected = ordered[0]
            selected_score = values[selected["candidate_id"]]
            tied = sum(
                values[candidate["candidate_id"]] == selected_score
                for candidate in candidates)
            selected_reward = float(selected["evaluation"]["reward"])
            oracle_reward = max(rewards)
            selections.append({
                "pool_id": pool_id,
                "candidate_set_id": stable_digest(sorted(
                    candidate["candidate_id"] for candidate in candidates)),
                "task": selected["identity"]["task"]["name"],
                "selected_candidate_id": selected["candidate_id"],
                "selected_trial": selected["identity"]["trial"],
                "selected_score": selected_score,
                "selected_reward": selected_reward,
                "oracle_reward": oracle_reward,
                "random_reward": sum(rewards) / len(rewards),
                "top_tie_count": tied,
                "correct": selected_reward >= threshold,
                "oracle_winnable": oracle_reward >= threshold,
                "selector_hit": (
                    oracle_reward >= threshold
                    and selected_reward == oracle_reward),
            })
        if not selections:
            raise SealError(f"scorer {scorer} has no complete candidate pool")
        pool_count = len(selections)
        top1_correct = sum(row["correct"] for row in selections)
        winnable = sum(row["oracle_winnable"] for row in selections)
        hits = sum(row["selector_hit"] for row in selections)
        summaries[scorer] = {
            "pools": pool_count,
            "candidates": sum(
                len(pools[row["pool_id"]]) for row in selections),
            "top1_correct": top1_correct,
            "top1_accuracy": top1_correct / pool_count,
            "oracle_winnable": winnable,
            "selector_hits": hits,
            "selector_hit_rate": hits / winnable if winnable else None,
            "selected_reward_mean": sum(
                row["selected_reward"] for row in selections) / pool_count,
            "random_reward_mean": sum(
                row["random_reward"] for row in selections) / pool_count,
            "oracle_reward_mean": sum(
                row["oracle_reward"] for row in selections) / pool_count,
            "top_ties": sum(row["top_tie_count"] > 1 for row in selections),
            "selections": selections,
        }
    return {
        "schema": BON_SCHEMA,
        "higher_is_better": higher_is_better,
        "correctness_threshold": threshold,
        "expected_n": expected_n,
        "scorers": summaries,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("freeze", help="freeze completed candidates")
    freeze.add_argument("--specs", required=True, type=Path)
    freeze.add_argument("--seals", required=True, type=Path)
    freeze.add_argument(
        "--verifier-inputs", type=Path,
        help="write the complete reward-free scorer manifest")
    bon = commands.add_parser("bon", help="evaluate scorer top-1 selections")
    bon.add_argument("--seals", required=True, type=Path)
    bon.add_argument("--scores", required=True, type=Path)
    bon.add_argument("--expected-n", type=int)
    bon.add_argument("--lower-is-better", action="store_true")
    bon.add_argument("--correctness-threshold", type=float, default=1.0)
    bon.add_argument("--output", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "freeze":
        specs_path = args.specs.expanduser().resolve()
        specs = _load_jsonl(specs_path)
        seals = [seal_from_spec(row, base_dir=specs_path.parent) for row in specs]
        merged = freeze_seals(args.seals, seals)
        if args.verifier_inputs:
            _write_jsonl_atomic(
                args.verifier_inputs.expanduser().resolve(),
                (verifier_view(seal) for seal in merged))
        print(json.dumps({
            "frozen": len(merged),
            "added_or_confirmed": len(seals),
            "seals": str(args.seals.expanduser().resolve()),
            "verifier_inputs": (
                str(args.verifier_inputs.expanduser().resolve())
                if args.verifier_inputs else None),
        }, sort_keys=True))
        return 0

    seals = _load_jsonl(args.seals.expanduser().resolve())
    scores = _load_jsonl(args.scores.expanduser().resolve())
    result = evaluate_bon(
        seals, scores, expected_n=args.expected_n,
        higher_is_better=not args.lower_is_better,
        correctness_threshold=args.correctness_threshold)
    rendered = json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2)
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
