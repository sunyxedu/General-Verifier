"""Prepare task-agnostic, sealed offline evaluation input packages.

The operator manifest is the only authority for case membership.  Model-visible
trees receive only copies of GV's immutable offline inputs.  Official labels
and their externally pinned provenance are verified separately and written to
a disjoint, coordinator-only manifest.

This module never invokes a model or scores a candidate.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
from typing import Any

from verifier import offline
from verifier.contract_replay import clone_sealed_inputs


OPERATOR_SCHEMA = "gv-general-offline-eval-operator-v1"
MODEL_INPUT_SCHEMA = "gv-general-offline-model-inputs-v1"
TRUTH_SCHEMA = "gv-general-offline-coordinator-truth-v1"
TRUTH_IDENTITY_SCHEMA = "gv-general-offline-official-truth-v1"
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_CASES = 1_000

_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CASE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_TOP_KEYS = {
    "schema", "package_id", "model_input_root", "truth_root", "cases",
}
_CASE_KEYS = {
    "case_id", "task_family", "candidate_id", "source_state",
    "input_seal", "truth",
}
_TRUTH_KEYS = {"label", "identity", "provenance"}
_JSON_PROVENANCE_KEYS = {"kind", "path", "sha256", "json_path"}
_TEXT_PROVENANCE_KEYS = {"kind", "path", "sha256"}
_PROVENANCE_KINDS = {"harbor_result_json", "harbor_reward_txt"}


class PackageError(ValueError):
    """The requested package cannot establish a safe evaluation boundary."""


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise PackageError(f"duplicate JSON key: {key}")
        value[key] = item
    return value


def _nonfinite(_value):
    raise PackageError("non-finite JSON constants are forbidden")


def _canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True,
                       separators=(",", ":")) + "\n").encode("utf-8")


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    fd = os.open(path, flags)
    try:
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            raise PackageError(f"truth source must be a private regular file: {path}")
        while block := os.read(fd, 1024 * 1024):
            digest.update(block)
        after = os.fstat(fd)
        stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
                  "st_mtime_ns", "st_ctime_ns")
        if any(getattr(before, field) != getattr(after, field) for field in stable):
            raise PackageError(f"truth source changed while hashing: {path}")
    finally:
        os.close(fd)
    return digest.hexdigest()


def _stable_bytes(path: Path, cap: int, description: str) -> bytes:
    """Read one bounded private file without accepting a mid-read swap."""
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    try:
        fd = os.open(path, flags)
    except OSError as err:
        raise PackageError(f"cannot read {description} {path}: {err}") from err
    try:
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > cap):
            raise PackageError(
                f"{description} must be a bounded private regular file: {path}")
        chunks = []
        remaining = cap + 1
        while remaining:
            block = os.read(fd, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(fd)
    finally:
        os.close(fd)
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
              "st_mtime_ns", "st_ctime_ns")
    if len(raw) > cap:
        raise PackageError(f"{description} exceeds size cap: {path}")
    if any(getattr(before, field) != getattr(after, field)
           for field in stable):
        raise PackageError(f"{description} changed while reading: {path}")
    return raw


def _read_json(path: Path, *, cap: int = MAX_MANIFEST_BYTES) -> tuple[Any, bytes]:
    raw = _stable_bytes(path, cap, "JSON file")
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=_strict_object,
                           parse_constant=_nonfinite)
    except (UnicodeDecodeError, json.JSONDecodeError) as err:
        raise PackageError(f"invalid JSON file {path}: {err}") from err
    return value, raw


def _safe_text(value: Any, field: str, *, path_safe: bool = False) -> str:
    if (not isinstance(value, str) or not value or len(value) > 256
            or any(char in value for char in "\x00\r\n\t")):
        raise PackageError(f"{field} must be a bounded nonempty string")
    if path_safe and not _CASE_ID.fullmatch(value):
        raise PackageError(f"{field} is not a safe path component")
    return value


def _canonical_existing(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise PackageError(f"{field} must be an absolute canonical path")
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as err:
        raise PackageError(f"{field} does not exist: {path}") from err
    if os.fspath(resolved) != value:
        raise PackageError(f"{field} may not use symlinks or aliases: {value}")
    return path


def _canonical_fresh(value: Any, field: str) -> Path:
    if not isinstance(value, str) or not os.path.isabs(value):
        raise PackageError(f"{field} must be an absolute canonical path")
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise PackageError(f"{field} must be a fresh path")
    try:
        resolved = path.resolve(strict=False)
    except OSError as err:
        raise PackageError(f"cannot resolve {field}: {path}") from err
    if os.fspath(resolved) != value:
        raise PackageError(f"{field} may not use symlinks or aliases: {value}")
    if not path.parent.is_dir():
        raise PackageError(f"parent of {field} must already exist")
    return path


def _inside(parent: Path, child: Path) -> bool:
    return os.path.commonpath((os.fspath(parent), os.fspath(child))) == os.fspath(parent)


def _overlap(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _reject_tree_aliases(state: Path) -> set[tuple[int, int]]:
    """Reject link-based escape hatches in every model-visible source input."""
    inodes: set[tuple[int, int]] = set()
    for name in offline.IMMUTABLE_INPUT_PATHS:
        root = state / name
        if not root.exists() and not root.is_symlink():
            continue
        pending = [root]
        while pending:
            path = pending.pop()
            info = path.lstat()
            if stat.S_ISLNK(info.st_mode):
                raise PackageError(f"symlink forbidden in model input source: {path}")
            if stat.S_ISDIR(info.st_mode):
                pending.extend(path / child for child in os.listdir(path))
                continue
            if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
                raise PackageError(f"aliased or special model input is forbidden: {path}")
            inode = (info.st_dev, info.st_ino)
            if inode in inodes:
                raise PackageError(f"duplicate file alias in model input source: {path}")
            inodes.add(inode)
    return inodes


def _label(value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PackageError("official truth label must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise PackageError("official truth label must be finite")
    return result


def truth_identity(label: float, provenance: dict[str, Any]) -> str:
    """Return the content identity an operator must pin in its manifest."""
    return _sha_bytes(_canonical_bytes({
        "schema": TRUTH_IDENTITY_SCHEMA,
        "label": label,
        "provenance": provenance,
    }))


def _json_path(value: Any, segments: Any, source: Path) -> Any:
    if not isinstance(segments, list) or not segments or len(segments) > 32:
        raise PackageError("harbor_result_json requires a bounded json_path")
    current = value
    for segment in segments:
        if isinstance(segment, bool) or not isinstance(segment, (str, int)):
            raise PackageError("json_path segments must be strings or nonnegative integers")
        if isinstance(segment, str):
            if not segment or len(segment) > 256 or not isinstance(current, dict):
                raise PackageError(f"official truth json_path is invalid for {source}")
            if segment not in current:
                raise PackageError(f"official truth json_path is missing in {source}")
            current = current[segment]
        else:
            if segment < 0 or not isinstance(current, list) or segment >= len(current):
                raise PackageError(f"official truth json_path is invalid for {source}")
            current = current[segment]
    return current


def _validate_truth(value: Any) -> tuple[float, str, dict[str, Any], Path]:
    if type(value) is not dict or set(value) != _TRUTH_KEYS:
        raise PackageError("truth must contain exactly label, identity, provenance")
    label = _label(value["label"])
    identity = value["identity"]
    if not isinstance(identity, str) or not _HASH.fullmatch(identity):
        raise PackageError("official truth identity must be a sha256 digest")
    provenance = value["provenance"]
    if not isinstance(provenance, dict) or provenance.get("kind") not in _PROVENANCE_KINDS:
        raise PackageError("missing or unsupported official truth provenance")
    kind = provenance["kind"]
    expected_keys = (_JSON_PROVENANCE_KEYS if kind == "harbor_result_json"
                     else _TEXT_PROVENANCE_KEYS)
    if set(provenance) != expected_keys:
        raise PackageError("official truth provenance has unexpected fields")
    source = _canonical_existing(provenance.get("path"), "truth provenance path")
    if not source.is_file() or source.lstat().st_nlink != 1:
        raise PackageError("truth provenance path must be a private regular file")
    expected_sha = provenance.get("sha256")
    if not isinstance(expected_sha, str) or not _HASH.fullmatch(expected_sha):
        raise PackageError("truth provenance sha256 is invalid")
    if _sha_file(source) != expected_sha:
        raise PackageError("official truth provenance digest mismatch")
    if kind == "harbor_result_json":
        payload, _raw = _read_json(source, cap=64 * 1024 * 1024)
        observed = _label(_json_path(payload, provenance["json_path"], source))
    else:
        try:
            raw = _stable_bytes(source, 256, "Harbor reward file")
            observed = _label(float(raw.decode("ascii").strip()))
        except (OSError, UnicodeDecodeError, ValueError) as err:
            raise PackageError(f"invalid Harbor reward file: {source}") from err
    if observed != label:
        raise PackageError("operator truth label differs from official provenance")
    if truth_identity(label, provenance) != identity:
        raise PackageError("official truth identity mismatch")
    return label, identity, provenance, source


def _write_new(path: Path, payload: bytes, mode: int) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    fd = os.open(path, flags, mode)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def validate_manifest(manifest: Any) -> dict[str, Any]:
    """Preflight every source and pin before creating either output root."""
    if type(manifest) is not dict or set(manifest) != _TOP_KEYS:
        raise PackageError("operator manifest has the wrong fields")
    if manifest.get("schema") != OPERATOR_SCHEMA:
        raise PackageError("wrong operator manifest schema")
    package_id = _safe_text(manifest["package_id"], "package_id")
    model_root = _canonical_fresh(manifest["model_input_root"], "model_input_root")
    truth_root = _canonical_fresh(manifest["truth_root"], "truth_root")
    if _overlap(model_root, truth_root):
        raise PackageError("model input and truth roots must be disjoint")
    cases = manifest["cases"]
    if not isinstance(cases, list) or not cases or len(cases) > MAX_CASES:
        raise PackageError("cases must be a nonempty bounded list")

    seen_cases: set[str] = set()
    seen_candidates: set[str] = set()
    seen_seals: set[str] = set()
    seen_sources: set[tuple[int, int]] = set()
    seen_truth_sources: set[tuple[int, int]] = set()
    seen_truth_identities: set[str] = set()
    normalized = []
    for case in cases:
        if type(case) is not dict or set(case) != _CASE_KEYS:
            raise PackageError("case has the wrong fields")
        case_id = _safe_text(case["case_id"], "case_id", path_safe=True)
        task_family = _safe_text(case["task_family"], "task_family")
        candidate_id = _safe_text(case["candidate_id"], "candidate_id")
        if case_id in seen_cases:
            raise PackageError("duplicate case_id")
        if candidate_id in seen_candidates:
            raise PackageError("duplicate candidate_id")
        seen_cases.add(case_id)
        seen_candidates.add(candidate_id)

        source = _canonical_existing(case["source_state"], "source_state")
        if not source.is_dir():
            raise PackageError("source_state must be a directory")
        source_inode = (source.stat().st_dev, source.stat().st_ino)
        if source_inode in seen_sources:
            raise PackageError("duplicate or aliased source_state")
        seen_sources.add(source_inode)
        if _overlap(source, model_root) or _overlap(source, truth_root):
            raise PackageError("source and destination/truth roots must be disjoint")
        _reject_tree_aliases(source)
        actual_seal = offline.verify_immutable_input_anchor(os.fspath(source))
        expected_seal = case["input_seal"]
        if (not isinstance(expected_seal, str) or not _HASH.fullmatch(expected_seal)
                or actual_seal != expected_seal):
            raise PackageError("source immutable input seal mismatch")
        if actual_seal in seen_seals:
            raise PackageError("duplicate immutable input seal")
        seen_seals.add(actual_seal)

        label, identity, provenance, truth_source = _validate_truth(case["truth"])
        truth_inode = (truth_source.stat().st_dev, truth_source.stat().st_ino)
        if truth_inode in seen_truth_sources:
            raise PackageError("duplicate or aliased truth provenance source")
        if identity in seen_truth_identities:
            raise PackageError("duplicate official truth identity")
        seen_truth_sources.add(truth_inode)
        seen_truth_identities.add(identity)
        if (_overlap(source, truth_source) or _overlap(model_root, truth_source)
                or _overlap(truth_root, truth_source)):
            raise PackageError("source/destination/truth paths must not overlap")
        normalized.append({
            "case_id": case_id,
            "task_family": task_family,
            "candidate_id": candidate_id,
            "source_state": source,
            "input_seal": actual_seal,
            "label": label,
            "truth_identity": identity,
            "provenance": provenance,
            "truth_source": truth_source,
        })
    return {
        "package_id": package_id,
        "model_input_root": model_root,
        "truth_root": truth_root,
        "cases": normalized,
    }


def prepare(manifest_path: str | Path) -> dict[str, Any]:
    """Build one fresh package without copying any derived verifier state."""
    manifest_file = _canonical_existing(os.fspath(manifest_path), "operator manifest")
    if not manifest_file.is_file() or manifest_file.lstat().st_nlink != 1:
        raise PackageError("operator manifest must be a private regular file")
    manifest, manifest_raw = _read_json(manifest_file)
    plan = validate_manifest(manifest)
    model_root: Path = plan["model_input_root"]
    truth_root: Path = plan["truth_root"]
    if (_overlap(manifest_file, model_root) or _overlap(manifest_file, truth_root)):
        raise PackageError("operator manifest must remain outside both output roots")

    model_root.mkdir(mode=0o755)
    (model_root / "cases").mkdir(mode=0o755)
    model_cases = []
    truth_cases = []
    for case in plan["cases"]:
        case_root = model_root / "cases" / case["case_id"]
        case_root.mkdir(mode=0o755)
        destination = case_root / "gv"
        copied_seal = clone_sealed_inputs(case["source_state"], destination)
        if copied_seal != case["input_seal"]:
            raise PackageError("cloned input seal differs from operator pin")
        _reject_tree_aliases(destination)
        if offline.verify_immutable_input_anchor(os.fspath(destination)) != copied_seal:
            raise PackageError("destination immutable input seal mismatch")
        model_cases.append({
            "case_id": case["case_id"],
            "task_family": case["task_family"],
            "candidate_id": case["candidate_id"],
            "state": f"cases/{case['case_id']}/gv",
            "input_seal": copied_seal,
        })
        truth_cases.append({
            "case_id": case["case_id"],
            "task_family": case["task_family"],
            "candidate_id": case["candidate_id"],
            "input_seal": copied_seal,
            "label": case["label"],
            "identity": case["truth_identity"],
            "provenance": case["provenance"],
        })

    # Revalidate every authority immediately before publishing completion
    # manifests.  The model manifest bytes are committed last.
    for case in plan["cases"]:
        if offline.verify_immutable_input_anchor(os.fspath(case["source_state"])) \
                != case["input_seal"]:
            raise PackageError("source immutable inputs changed during packaging")
        destination = model_root / "cases" / case["case_id"] / "gv"
        if offline.verify_immutable_input_anchor(os.fspath(destination)) \
                != case["input_seal"]:
            raise PackageError("destination immutable inputs changed during packaging")
        if _sha_file(case["truth_source"]) != case["provenance"]["sha256"]:
            raise PackageError("official truth source changed during packaging")

    model_payload = _canonical_bytes({
        "schema": MODEL_INPUT_SCHEMA,
        "package_id": plan["package_id"],
        "cases": model_cases,
    })
    truth_payload = _canonical_bytes({
        "schema": TRUTH_SCHEMA,
        "package_id": plan["package_id"],
        "operator_manifest_sha256": _sha_bytes(manifest_raw),
        "model_input_manifest_sha256": _sha_bytes(model_payload),
        "model_input_root": os.fspath(model_root),
        "cases": truth_cases,
    })
    truth_root.mkdir(mode=0o700)
    _write_new(truth_root / "truth-manifest.json", truth_payload, 0o600)
    _write_new(model_root / "input-manifest.json", model_payload, 0o644)
    return {
        "package_id": plan["package_id"],
        "model_input_root": os.fspath(model_root),
        "truth_root": os.fspath(truth_root),
        "cases": len(model_cases),
        "model_input_manifest_sha256": _sha_bytes(model_payload),
        "truth_manifest_sha256": _sha_bytes(truth_payload),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", help="absolute path to the operator manifest")
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest), sort_keys=True))


if __name__ == "__main__":
    main()
