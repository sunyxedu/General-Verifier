"""Build sealed, task-agnostic plans for the full-offline v2 comparison.

This is a no-model schema join.  It combines materialized input/truth records,
one frozen source identity, explicit runtime roots, and pinned evaluator code.
Official truth references are retained only in the coordinator plan; worker
plans contain no expected answer or truth-file identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import stat
import tempfile

try:
    from . import run_full_offline_comparison_v2 as comparison
    from . import run_full_offline_worker_v2 as worker
    from . import run_typed_adapter_eval as source_tools
except ImportError:  # direct script execution
    import run_full_offline_comparison_v2 as comparison
    import run_full_offline_worker_v2 as worker
    import run_typed_adapter_eval as source_tools


BUILD_SCHEMA = "gv-full-offline-plan-build-v1"
RECORDS_SCHEMA = "gv-offline-v2-materialized-records-v1"
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_RECORDS_KEYS = {
    "schema", "package_id", "model_input_manifest_sha256",
    "truth_manifest_sha256", "records",
}
_RECORD_KEYS = {
    "case_id", "candidate_id", "task_family", "input_source",
    "input_seal", "official_truth",
}
_TRUTH_REF_KEYS = {"path", "sha256"}
_TRUTH_KEYS = {
    "schema", "candidate_id", "task_family", "label", "input_seal",
    "official_identity", "source_truth_manifest_sha256",
}
_DEFAULT_ATTESTATIONS = ("task.md",)
_MAX_FILE_BYTES = 512 * 1024 * 1024


class PlanBuildError(ValueError):
    """The requested evaluation plan is unsafe or internally inconsistent."""


def _inside(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((str(parent), str(child))) == str(parent)
    except ValueError:
        return False


def _overlap(left: Path, right: Path) -> bool:
    return _inside(left, right) or _inside(right, left)


def _existing(value: str, field: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise PlanBuildError(f"{field} must be a canonical absolute path")
    path = Path(value)
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise PlanBuildError(f"{field} does not exist") from error
    if str(resolved) != value or (directory and not path.is_dir()):
        raise PlanBuildError(f"{field} must be a canonical existing path")
    return path


def _fresh_root(value: str) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise PlanBuildError("run root must be a canonical absolute path")
    path = Path(value)
    if path.exists() or path.is_symlink():
        raise PlanBuildError("run root must be fresh")
    if not path.parent.is_dir() or str(path.resolve(strict=False)) != value:
        raise PlanBuildError("run-root parent must be canonical and existing")
    temporary_roots = {
        Path(tempfile.gettempdir()).resolve(), Path("/private/tmp").resolve(),
        Path("/private/var/folders").resolve(), Path("/tmp").resolve(),
    }
    if any(_inside(root, path) for root in temporary_roots):
        raise PlanBuildError("paid evaluation outputs require a persistent root")
    return path


def _strict_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PlanBuildError("duplicate JSON key")
        result[key] = value
    return result


def _nonfinite(_value):
    raise PlanBuildError("non-finite JSON constant")


def _stable_bytes(path: Path, cap: int, description: str) -> bytes:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) \
        | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise PlanBuildError(f"cannot open {description}: {path}") from error
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > cap):
            raise PlanBuildError(
                f"{description} must be a bounded private regular file")
        chunks, remaining = [], cap + 1
        while remaining:
            block = os.read(descriptor, min(1024 * 1024, remaining))
            if not block:
                break
            chunks.append(block)
            remaining -= len(block)
        raw = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable = ("st_dev", "st_ino", "st_mode", "st_nlink", "st_size",
              "st_mtime_ns", "st_ctime_ns")
    if len(raw) > cap or any(getattr(before, key) != getattr(after, key)
                             for key in stable):
        raise PlanBuildError(f"{description} changed while it was read")
    return raw


def _read_json(path: Path, cap: int = 64 * 1024 * 1024):
    raw = _stable_bytes(path, cap, "JSON input")
    try:
        value = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as error:
        raise PlanBuildError(f"invalid JSON input: {path}") from error
    return value, hashlib.sha256(raw).hexdigest()


def _sha(path: Path) -> str:
    return hashlib.sha256(
        _stable_bytes(path, _MAX_FILE_BYTES, "pinned file")).hexdigest()


def _safe_text(value, field: str, limit: int = 256) -> str:
    if (not isinstance(value, str) or not value or len(value) > limit
            or any(char in value for char in "\x00\r\n\t")):
        raise PlanBuildError(f"{field} must be a bounded opaque string")
    return value


def _relative_attestations(values) -> tuple[str, ...]:
    requested = [*_DEFAULT_ATTESTATIONS, *(values or ())]
    result = []
    for value in requested:
        if not isinstance(value, str) or not value or len(value) > 512:
            raise PlanBuildError("attestations must be bounded relative paths")
        path = PurePosixPath(value)
        if (path.is_absolute() or path.as_posix() != value
                or any(part in {"", ".", ".."} for part in path.parts)):
            raise PlanBuildError("attestations must be canonical relative paths")
        if value not in result:
            result.append(value)
    if len(result) > 64:
        raise PlanBuildError("too many input attestations")
    return tuple(result)


def _runtime_roots(values, private_roots) -> list[str]:
    if not isinstance(values, (list, tuple)) or not 1 <= len(values) <= 16:
        raise PlanBuildError("runtime read roots must be explicitly provided")
    result = []
    for value in values:
        root = _existing(value, "runtime read root", directory=True)
        if value == "/" or value in result:
            raise PlanBuildError("runtime read roots must be distinct and bounded")
        if any(_overlap(root, private) for private in private_roots):
            raise PlanBuildError("runtime read root overlaps an evaluation boundary")
        result.append(value)
    return result


def _truth_record(reference, record, truth_manifest_sha, private_roots):
    if type(reference) is not dict or set(reference) != _TRUTH_REF_KEYS:
        raise PlanBuildError("official truth reference is malformed")
    path = _existing(reference.get("path"), "official truth")
    if any(_overlap(path, root) for root in private_roots):
        raise PlanBuildError("official truth overlaps model-visible or output roots")
    document, observed = _read_json(path, cap=16 * 1024)
    if (not _HASH.fullmatch(reference.get("sha256", ""))
            or observed != reference["sha256"]
            or type(document) is not dict or set(document) != _TRUTH_KEYS
            or document.get("schema") != comparison.TRUTH_SCHEMA
            or document.get("candidate_id") != record["candidate_id"]
            or document.get("task_family") != record["task_family"]
            or document.get("input_seal") != record["input_seal"]
            or document.get("label") not in {"positive", "negative"}
            or not _HASH.fullmatch(document.get("official_identity", ""))
            or document.get("source_truth_manifest_sha256")
            != truth_manifest_sha):
        raise PlanBuildError("official truth does not match its materialized record")
    return path, document


def _load_records(records_path: str, expected_sha256: str, source: Path,
                  run_root: Path, relative_attestations):
    if not _HASH.fullmatch(expected_sha256 or ""):
        raise PlanBuildError("records digest must be an exact SHA-256")
    path = _existing(records_path, "materialized records")
    if _overlap(path, source) or _overlap(path, run_root):
        raise PlanBuildError("records must stay outside source and outputs")
    manifest, observed = _read_json(path)
    if observed != expected_sha256:
        raise PlanBuildError("materialized records changed")
    if (type(manifest) is not dict or set(manifest) != _RECORDS_KEYS
            or manifest.get("schema") != RECORDS_SCHEMA
            or not _HASH.fullmatch(
                manifest.get("model_input_manifest_sha256", ""))
            or not _HASH.fullmatch(manifest.get("truth_manifest_sha256", ""))
            or type(manifest.get("records")) is not list
            or not 6 <= len(manifest["records"]) <= 60
            or len(manifest["records"]) % 2):
        raise PlanBuildError("materialized records schema or cohort is invalid")
    _safe_text(manifest.get("package_id"), "package_id")
    prepared, candidates, cases, input_roots, truth_paths = [], set(), set(), [], set()
    family_labels = {}
    for record in manifest["records"]:
        if type(record) is not dict or set(record) != _RECORD_KEYS:
            raise PlanBuildError("materialized record schema is invalid")
        case_id = _safe_text(record.get("case_id"), "case_id", 128)
        candidate = _safe_text(record.get("candidate_id"), "candidate_id")
        family = _safe_text(record.get("task_family"), "task_family", 128)
        if case_id in cases or candidate in candidates:
            raise PlanBuildError("case and candidate identities must be unique")
        cases.add(case_id); candidates.add(candidate)
        if not _HASH.fullmatch(record.get("input_seal", "")):
            raise PlanBuildError("input seal must be an exact SHA-256")
        inputs = _existing(record.get("input_source"), "input source",
                           directory=True)
        if (_overlap(inputs, source) or _overlap(inputs, run_root)
                or any(_overlap(inputs, other) for other in input_roots)):
            raise PlanBuildError("input roots must be mutually disjoint")
        input_roots.append(inputs)
        truth_path, truth = _truth_record(
            record["official_truth"], record,
            manifest["truth_manifest_sha256"], (source, inputs, run_root))
        if truth_path in truth_paths:
            raise PlanBuildError("each candidate requires independent truth")
        truth_paths.add(truth_path)
        family_labels.setdefault(family, set()).add(truth["label"])
        attestations = {}
        for relative in relative_attestations:
            attested = inputs.joinpath(*PurePosixPath(relative).parts)
            try:
                resolved = attested.resolve(strict=True)
            except OSError as error:
                raise PlanBuildError(
                    f"missing input attestation {relative!r}") from error
            if resolved != attested or not _inside(inputs, attested):
                raise PlanBuildError("input attestation escapes its candidate root")
            attestations[str(attested)] = _sha(attested)
        prepared.append({
            **record, "input_source": str(inputs),
            "official_truth": {
                "path": str(truth_path), "sha256": record["official_truth"]["sha256"],
            },
            "input_attestations": attestations,
        })
    if (len(family_labels) < 3
            or any(labels != {"positive", "negative"}
                   for labels in family_labels.values())):
        raise PlanBuildError(
            "cohort requires positive and negative truth in at least three families")
    retained_files = {path, *truth_paths}
    if any(_overlap(retained, inputs)
           for retained in retained_files for inputs in input_roots):
        raise PlanBuildError(
            "coordinator-only records and truth must be outside every input")
    return manifest, prepared, input_roots


def _evaluation_support(source: Path):
    root = source / "experiments" / "terminal_bench"
    paths = {
        "worker": root / "run_full_offline_worker_v2.py",
        "projection": root / "project_full_offline_result.py",
        "coordinator": root / "run_full_offline_comparison_v2.py",
        "accounting": root / "run_full_offline_comparison.py",
    }
    for name, path in paths.items():
        if path.resolve(strict=True) != path or not path.is_file():
            raise PlanBuildError(f"{name} evaluator support path is aliased")
    pins = {name: _sha(path) for name, path in paths.items()}
    worker_support = {}
    for name in worker.SUPPORT_FILES:
        path = root / name
        if path.resolve(strict=True) != path or not path.is_file():
            raise PlanBuildError(f"worker support file is missing: {name}")
        worker_support[name] = _sha(path)
    return paths, pins, worker_support


def _write_new(path: Path, value) -> str:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL \
        | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        raise
    return hashlib.sha256(raw).hexdigest()


def build_plan(*, records_path: str, records_sha256: str, source_root: str,
               source_commit: str, run_root: str, comparison_id: str,
               runtime_read_roots, attestation_paths=(),
               expected_source_fingerprint: str | None = None):
    """Create a private plan bundle without invoking the model."""
    source = _existing(source_root, "frozen source", directory=True)
    if (source / ".git").exists() or not _COMMIT.fullmatch(source_commit or ""):
        raise PlanBuildError("source must be a git-free archive of an exact commit")
    output = _fresh_root(run_root)
    if _overlap(source, output):
        raise PlanBuildError("source and run root must be disjoint")
    comparison_id = _safe_text(comparison_id, "comparison_id")
    attest_relative = _relative_attestations(attestation_paths)
    manifest, records, input_roots = _load_records(
        records_path, records_sha256, source, output, attest_relative)
    roots = _runtime_roots(
        runtime_read_roots, (Path.home().resolve(), source, output, *input_roots))
    fingerprint = source_tools._source_fingerprint(source)
    if (expected_source_fingerprint is not None
            and expected_source_fingerprint != fingerprint):
        raise PlanBuildError("frozen source fingerprint differs from operator pin")
    paths, pins, worker_support = _evaluation_support(source)

    # Prepare the full objects before making the fresh bundle visible.
    pair_specs = []
    for ordinal, record in enumerate(records, 1):
        first = "control" if ordinal % 2 else "treatment"
        order = (first, "treatment" if first == "control" else "control")
        jobs = []
        for arm in order:
            jobs.append((arm, {
                "schema": worker.SCHEMA,
                "source_root": str(source), "source_commit": source_commit,
                "source_fingerprint": fingerprint,
                "input_source": record["input_source"],
                "input_seal": record["input_seal"],
                "output_root": str(
                    output / "worker-results" / f"worker-{ordinal:04d}-{arm}"),
                "arm": arm, "runtime_read_roots": roots,
                "support_sha256": worker_support,
            }))
        pair_specs.append((record, jobs))

    os.mkdir(output, 0o700)
    plans_root = output / "plans"
    worker_plans_root = plans_root / "workers"
    os.mkdir(plans_root, 0o700)
    os.mkdir(worker_plans_root, 0o700)
    pairs = []
    for ordinal, (record, plans) in enumerate(pair_specs, 1):
        jobs = []
        for arm, worker_plan in plans:
            plan_path = worker_plans_root / f"worker-{ordinal:04d}-{arm}.json"
            digest = _write_new(plan_path, worker_plan)
            jobs.append({"arm": arm, "plan_path": str(plan_path),
                         "plan_sha256": digest})
        pairs.append({
            "candidate_id": record["candidate_id"],
            "task_family": record["task_family"],
            "official_truth": record["official_truth"],
            "input_attestations": record["input_attestations"],
            "jobs": jobs,
        })
    plan = {
        "schema": comparison.SCHEMA, "comparison_id": comparison_id,
        "regime": "cold",
        "output_root": str(output / "coordinator-result"),
        "source": {"root": str(source), "commit": source_commit,
                   "fingerprint": fingerprint},
        "worker_path": str(paths["worker"]),
        "worker_sha256": pins["worker"],
        "projection_path": str(paths["projection"]),
        "projection_sha256": pins["projection"],
        "coordinator_sha256": pins["coordinator"],
        "accounting_helper_sha256": pins["accounting"],
        "pairs": pairs,
    }
    plan_path = plans_root / "comparison-plan.json"
    plan_sha = _write_new(plan_path, plan)
    if source_tools._source_fingerprint(source) != fingerprint:
        raise PlanBuildError("frozen source changed while plans were built")
    for name, path in paths.items():
        if _sha(path) != pins[name]:
            raise PlanBuildError("evaluation support changed while plans were built")
    for name, digest in worker_support.items():
        if _sha(Path(paths["worker"]).with_name(name)) != digest:
            raise PlanBuildError("worker support changed while plans were built")
    build_record = {
        "schema": BUILD_SCHEMA, "model_calls": 0,
        "package_id": manifest["package_id"],
        "records_path": str(_existing(records_path, "materialized records")),
        "records_sha256": records_sha256,
        "comparison_plan_path": str(plan_path),
        "comparison_plan_sha256": plan_sha,
        "source_fingerprint": fingerprint,
        "attestation_relative_paths": list(attest_relative),
        "worker_plan_count": 2 * len(records),
    }
    build_path = plans_root / "plan-build.json"
    _write_new(build_path, build_record)
    return {**build_record, "build_record_path": str(build_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", required=True)
    parser.add_argument("--records-sha256", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--source-fingerprint")
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--comparison-id", required=True)
    parser.add_argument("--runtime-read-root", action="append", required=True)
    parser.add_argument("--attest-relative", action="append", default=[])
    args = parser.parse_args(argv)
    result = build_plan(
        records_path=args.records, records_sha256=args.records_sha256,
        source_root=args.source_root, source_commit=args.source_commit,
        expected_source_fingerprint=args.source_fingerprint,
        run_root=args.run_root, comparison_id=args.comparison_id,
        runtime_read_roots=args.runtime_read_root,
        attestation_paths=args.attest_relative)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
