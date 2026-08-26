"""Discover safe final-world artifacts from authoritative Harbor task text.

Terminal-Bench tasks do not consistently declare ``artifacts`` in task.toml.
For offline verification that means a job-level ``--artifact /app`` is often
insufficient: system-administration tasks can deliver their answer under
``/etc``, ``/var/www``, or another explicitly named path.  Capturing those
entire system roots is both expensive and unreliable.

This module extracts *explicit absolute paths* from the public task
instruction and declared task artifacts, plus the final literal ``WORKDIR``
from the public environment Dockerfile, validates them conservatively, and
merges them with caller-supplied roots.  Capturing the environment working
directory is necessary when a task names a relative deliverable.  The module
never searches or reads a task's ``tests/`` or ``solution/`` directories.

The project environment intentionally does not depend on Harbor.  The
``resolve`` subcommand is therefore launched with the Python interpreter from
the installed Harbor entry point.  Harbor's dataset client resolves the exact
package digests before any solver trial starts; only the selected task
packages are downloaded.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import posixpath
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping, Sequence


# A candidate must start at a token boundary.  The intentionally broad token
# body lets the audit explain why globs/shell syntax were rejected instead of
# silently failing to match them.  URL paths are excluded by the boundary and
# audited separately through ``_URL``.  SCP-like ``host:/git/project`` paths
# remain candidates because they can be real task deliverables.
_PATH_CANDIDATE = re.compile(
    r"(?<![A-Za-z0-9_./-])(?P<path>/(?!/)[^\s`\"'<>]+)"
)
_URL = re.compile(r"(?P<url>(?:https?|ftp)://[^\s`\"'<>]+)", re.IGNORECASE)
_URI_ROUTE_CONTEXT = re.compile(
    r"(?:\bendpoint(?:\s+at)?(?:\s*:\s*(?:get|post|put|patch|delete))?"
    r"|\b(?:get|post|put|patch|delete)"
    r"|\b(?:url|route)\s*[:=]?"
    r"|\bserves?)\s*[\"'`(]*\s*$",
    re.IGNORECASE,
)

_APPROVED_BROAD_ROOTS = frozenset({"/app", "/gv-state", "/workspace"})
_DANGEROUS_EXACT = frozenset({
    "/", "/bin", "/boot", "/dev", "/etc", "/home", "/lib", "/lib64",
    "/opt", "/proc", "/root", "/run", "/sbin", "/sys", "/tmp", "/usr",
    "/var", "/var/cache", "/var/lib", "/var/log", "/var/tmp",
})
_FORBIDDEN_PREFIXES = (
    "/dev/", "/proc/", "/root/", "/run/", "/sys/",
)
_SENSITIVE_PATHS = frozenset({
    "/etc/gshadow", "/etc/gshadow-", "/etc/shadow", "/etc/shadow-",
    "/etc/sudoers",
})
_SHELL_OR_GLOB = frozenset("\x00\n\r\t*?[]{}$`|;&<>")
_TRAILING_PROSE = ".,;:!?)]}\"'"


class ArtifactDiscoveryError(RuntimeError):
    """The authoritative task package could not be resolved safely."""


@dataclass(frozen=True)
class ArtifactPathAudit:
    """One accepted or rejected path candidate from a public task source."""

    source: str
    raw: str
    normalized: str | None
    decision: str
    reason: str

    def to_json(self) -> dict[str, str | None]:
        return {
            "source": self.source,
            "raw": self.raw,
            "normalized": self.normalized,
            "decision": self.decision,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class DatasetArtifactResolution:
    """Resolved dataset identity and audited artifact paths."""

    dataset: str
    task_packages: Mapping[str, str]
    referenced_by_task: Mapping[str, tuple[str, ...]]
    audit_by_task: Mapping[str, tuple[ArtifactPathAudit, ...]]
    artifacts: tuple[str, ...]


def audit_artifact_path(
    value: Any, *, source: str, allow_approved_broad: bool = True,
) -> ArtifactPathAudit:
    """Classify one candidate and retain the reason for auditability.

    Exact system roots are refused; specific descendants such as
    ``/etc/nginx/nginx.conf`` remain eligible.  Virtual-kernel trees, home/root
    data, obvious secrets, globs, shell syntax, traversal, and network-style
    paths are always rejected.  ``/app`` and ``/gv-state`` are the two broad
    benchmark roots explicitly approved by this harness.
    """
    if not isinstance(value, str):
        return ArtifactPathAudit(source, repr(value), None, "rejected", "not_string")
    raw = value.strip().rstrip(_TRAILING_PROSE)
    # Markdown emphasis commonly produces tokens like ``/app/tool.py**:``.
    # Strip the closing marker, but never reinterpret a path whose final
    # component is itself a glob (``/tmp/**``).
    if raw.endswith("**") and not raw.endswith("/**"):
        raw = raw[:-2].rstrip(_TRAILING_PROSE)
    if not raw:
        return ArtifactPathAudit(source, value, None, "rejected", "empty")
    if len(raw) > 4096:
        return ArtifactPathAudit(source, value, None, "rejected", "too_long")
    if not raw.startswith("/"):
        return ArtifactPathAudit(source, value, None, "rejected", "not_absolute")
    if raw.startswith("//"):
        return ArtifactPathAudit(source, value, None, "rejected", "network_path")
    if any(char in raw for char in _SHELL_OR_GLOB):
        return ArtifactPathAudit(source, value, None, "rejected", "shell_or_glob")
    parts = PurePosixPath(raw).parts
    if ".." in parts:
        return ArtifactPathAudit(source, value, None, "rejected", "traversal")
    normalized = posixpath.normpath(raw)
    if not normalized.startswith("/") or normalized.startswith("//"):
        return ArtifactPathAudit(source, value, None, "rejected", "normalization")
    if normalized in _APPROVED_BROAD_ROOTS:
        if allow_approved_broad:
            return ArtifactPathAudit(source, value, normalized, "accepted", "approved_root")
        return ArtifactPathAudit(source, value, normalized, "rejected", "broad_root")
    if normalized in _DANGEROUS_EXACT:
        return ArtifactPathAudit(source, value, normalized, "rejected", "broad_root")
    if normalized in _SENSITIVE_PATHS:
        return ArtifactPathAudit(source, value, normalized, "rejected", "sensitive_path")
    if any(normalized.startswith(prefix) for prefix in _FORBIDDEN_PREFIXES):
        return ArtifactPathAudit(source, value, normalized, "rejected", "private_system_tree")
    if re.fullmatch(r"/[0-9]+(?:\.[0-9]+)?", normalized):
        return ArtifactPathAudit(source, value, normalized, "rejected", "numeric_expression")
    if normalized.lower().endswith(".sock"):
        return ArtifactPathAudit(
            source, value, normalized, "rejected", "runtime_socket_unsupported"
        )
    normalized_parts = PurePosixPath(normalized).parts
    if (
        len(normalized_parts) == 4
        and normalized_parts[1:3] == ("usr", "local")
        and "." not in normalized_parts[-1]
    ):
        return ArtifactPathAudit(
            source, value, normalized, "rejected", "broad_system_directory"
        )
    if len(normalized_parts) == 2:
        basename = normalized_parts[-1]
        if basename.startswith(".") or "." not in basename:
            return ArtifactPathAudit(
                source, value, normalized, "rejected", "broad_top_level"
            )
    # Refuse credential-bearing home subtrees even when they appear below a
    # nonstandard home prefix.
    lowered_parts = tuple(part.lower() for part in normalized_parts)
    if ".ssh" in lowered_parts or ".aws" in lowered_parts or ".gnupg" in lowered_parts:
        return ArtifactPathAudit(source, value, normalized, "rejected", "credential_tree")
    return ArtifactPathAudit(source, value, normalized, "accepted", "specific_absolute_path")


def normalize_artifact_path(
    value: Any, *, allow_approved_broad: bool = True,
) -> str | None:
    """Return a safe normalized absolute POSIX path, otherwise ``None``."""
    record = audit_artifact_path(
        value, source="direct", allow_approved_broad=allow_approved_broad
    )
    return record.normalized if record.decision == "accepted" else None


def audit_instruction_paths(
    instruction: str, *, source: str = "instruction.md",
) -> list[ArtifactPathAudit]:
    """Audit every absolute-path candidate and every URL path in prose."""
    records: list[ArtifactPathAudit] = []
    url_spans: list[tuple[int, int]] = []
    for match in _URL.finditer(instruction):
        url_spans.append(match.span())
        url = match.group("url").rstrip(_TRAILING_PROSE)
        # Keep URL routes visible in the audit.  They are never artifact
        # candidates even if they look like an absolute POSIX path.
        route_match = re.match(r"[a-z]+://[^/]+(?P<route>/.*)?$", url, re.IGNORECASE)
        route = route_match.group("route") if route_match else None
        if route and route != "/":
            records.append(
                ArtifactPathAudit(source, route, None, "rejected", "url_path")
            )

    for match in _PATH_CANDIDATE.finditer(instruction):
        if any(start <= match.start() < end for start, end in url_spans):
            continue
        record = audit_artifact_path(match.group("path"), source=source)
        if match.start() > 0 and instruction[match.start() - 1] == "~":
            record = ArtifactPathAudit(
                source, record.raw, record.normalized, "rejected", "home_relative"
            )
        elif (
            record.normalized is not None
            and _URI_ROUTE_CONTEXT.search(instruction[max(0, match.start() - 100):match.start()])
        ):
            record = ArtifactPathAudit(
                source, record.raw, record.normalized, "rejected", "uri_route"
            )
        records.append(record)
    return records


def extract_instruction_paths(instruction: str) -> list[str]:
    """Extract safe, explicit absolute paths from task instruction prose."""
    found: list[str] = []
    seen: set[str] = set()
    for record in audit_instruction_paths(instruction):
        path = record.normalized
        if record.decision != "accepted" or path is None or path in seen:
            continue
        seen.add(path)
        found.append(path)
    return found


def _declared_artifact_audit(config: Mapping[str, Any]) -> list[ArtifactPathAudit]:
    """Audit only the path-bearing public artifact fields in task.toml."""
    entries: list[Any] = list(config.get("artifacts", []) or [])
    for step in config.get("steps", []) or []:
        if isinstance(step, Mapping):
            entries.extend(step.get("artifacts", []) or [])
    records: list[ArtifactPathAudit] = []
    for entry in entries:
        value = entry.get("source") if isinstance(entry, Mapping) else entry
        records.append(audit_artifact_path(value, source="task.toml:artifacts"))
    return records


def _dockerfile_final_workdir_audit(task_dir: Path) -> list[ArtifactPathAudit]:
    """Return the final stage's statically resolvable Docker ``WORKDIR``.

    Docker resets the working directory to ``/`` at each ``FROM``.  Relative
    ``WORKDIR`` instructions compose with the preceding value in that stage.
    Variable-expanded or shell-like values are deliberately rejected rather
    than guessed.  Only the final stage matters to the solver container.
    """
    dockerfile = task_dir / "environment" / "Dockerfile"
    if not dockerfile.exists():
        return []
    path = _safe_public_file(task_dir, "environment/Dockerfile")
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as exc:
        raise ArtifactDiscoveryError(
            f"cannot read public environment Dockerfile: {path}"
        ) from exc

    logical: list[str] = []
    pending = ""
    for physical in text.splitlines():
        stripped = physical.strip()
        if not pending and (not stripped or stripped.startswith("#")):
            continue
        pending += (" " if pending else "") + physical.rstrip().rstrip("\\").strip()
        if physical.rstrip().endswith("\\"):
            continue
        logical.append(pending)
        pending = ""
    if pending:
        logical.append(pending)

    current = "/"
    final_raw: str | None = None
    final_normalized: str | None = None
    for line in logical:
        instruction, _, argument = line.partition(" ")
        keyword = instruction.upper()
        argument = argument.strip()
        if keyword == "FROM":
            current = "/"
            final_raw = None
            final_normalized = None
            continue
        if keyword != "WORKDIR":
            continue
        final_raw = argument
        if not argument or any(char in argument for char in _SHELL_OR_GLOB):
            final_normalized = None
            continue
        candidate = argument if argument.startswith("/") else posixpath.join(current, argument)
        final_normalized = posixpath.normpath(candidate)
        current = final_normalized

    if final_raw is None:
        return []
    if final_normalized is None:
        return [ArtifactPathAudit(
            "environment/Dockerfile:WORKDIR", final_raw, None,
            "rejected", "dynamic_workdir",
        )]
    return [audit_artifact_path(
        final_normalized, source="environment/Dockerfile:WORKDIR"
    )]


def _safe_public_file(task_dir: Path, relative: str) -> Path:
    """Resolve a public task file without following a package symlink."""
    candidate = task_dir / relative
    if candidate.is_symlink() or not candidate.is_file():
        raise ArtifactDiscoveryError(
            f"authoritative task file is missing or a symlink: {candidate}"
        )
    task_real = task_dir.resolve()
    candidate_real = candidate.resolve()
    try:
        candidate_real.relative_to(task_real)
    except ValueError as exc:
        raise ArtifactDiscoveryError(
            f"authoritative task file escapes package: {candidate}"
        ) from exc
    return candidate_real


def discover_task_package_audit(
    task_dir: str | os.PathLike[str],
) -> tuple[tuple[str, ...], tuple[ArtifactPathAudit, ...]]:
    """Audit public instruction/config paths, never tests/solution."""
    root = Path(task_dir)
    config_path = _safe_public_file(root, "task.toml")
    try:
        config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise ArtifactDiscoveryError(f"cannot read public task metadata: {root}") from exc

    instructions: list[tuple[str, str]] = []
    root_instruction = root / "instruction.md"
    if root_instruction.exists() or not config.get("steps"):
        instruction_path = _safe_public_file(root, "instruction.md")
        try:
            instructions.append((
                "instruction.md", instruction_path.read_text(encoding="utf-8")
            ))
        except (OSError, UnicodeError) as exc:
            raise ArtifactDiscoveryError(
                f"cannot read public task instruction: {instruction_path}"
            ) from exc

    # Multi-step instructions are public solver inputs too.  Step names come
    # only from task.toml and are constrained to remain under steps/.
    for step in config.get("steps", []) or []:
        if not isinstance(step, Mapping) or not isinstance(step.get("name"), str):
            continue
        name = step["name"]
        if not name or name in (".", "..") or "/" in name or "\\" in name:
            raise ArtifactDiscoveryError(f"unsafe step name in {config_path}: {name!r}")
        step_instruction = _safe_public_file(root, f"steps/{name}/instruction.md")
        try:
            instructions.append((
                f"steps/{name}/instruction.md",
                step_instruction.read_text(encoding="utf-8"),
            ))
        except (OSError, UnicodeError) as exc:
            raise ArtifactDiscoveryError(
                f"cannot read public step instruction: {step_instruction}"
            ) from exc

    raw_records = _declared_artifact_audit(config)
    raw_records.extend(_dockerfile_final_workdir_audit(root))
    for source, text in instructions:
        raw_records.extend(audit_instruction_paths(text, source=source))

    paths: list[str] = []
    records: list[ArtifactPathAudit] = []
    seen: set[str] = set()
    for record in raw_records:
        path = record.normalized
        if record.decision == "accepted" and path is not None:
            if path in seen:
                record = ArtifactPathAudit(
                    record.source, record.raw, path, "ignored", "duplicate"
                )
            else:
                seen.add(path)
                paths.append(path)
        records.append(record)
    return tuple(paths), tuple(records)


def discover_task_package(task_dir: str | os.PathLike[str]) -> tuple[str, ...]:
    """Discover accepted paths from public task sources."""
    return discover_task_package_audit(task_dir)[0]


def _is_parent(parent: str, child: str) -> bool:
    return parent != child and child.startswith(parent.rstrip("/") + "/")


def _coalesce(paths: Iterable[str]) -> list[str]:
    """Remove duplicates/descendants so Harbor sees no overlapping entries."""
    result: list[str] = []
    for path in paths:
        if path in result or any(_is_parent(existing, path) for existing in result):
            continue
        descendants = [existing for existing in result if _is_parent(path, existing)]
        if descendants:
            first = min(result.index(existing) for existing in descendants)
            result = [existing for existing in result if existing not in descendants]
            result.insert(first, path)
        else:
            result.append(path)
    return result


def _coalesce_without_widening(paths: Iterable[str]) -> list[str]:
    """Keep the first overlapping discovered path and never widen it later."""
    result: list[str] = []
    for path in paths:
        if any(
            existing == path or _is_parent(existing, path) or _is_parent(path, existing)
            for existing in result
        ):
            continue
        result.append(path)
    return result


def merge_artifact_paths(
    explicit_roots: Sequence[str],
    referenced_by_task: Mapping[str, Sequence[str]],
) -> tuple[str, ...]:
    """Merge caller roots and task references without widening caller roots.

    If a discovered path overlaps an explicit root in either direction, the
    explicit choice wins.  Among discovered paths, the first explicit mention
    wins; a later parent is not allowed to silently widen an earlier file or
    subdirectory.  Harbor therefore sees no overlapping entries.
    """
    explicit: list[str] = []
    for raw in explicit_roots:
        path = normalize_artifact_path(raw)
        if path is None:
            raise ArtifactDiscoveryError(f"unsafe explicit artifact root: {raw!r}")
        explicit.append(path)
    explicit = _coalesce(explicit)

    discovered: list[str] = []
    for paths in referenced_by_task.values():
        for raw in paths:
            path = normalize_artifact_path(raw)
            if path is None:
                continue
            if any(
                root == path or _is_parent(root, path) or _is_parent(path, root)
                for root in explicit
            ):
                continue
            discovered.append(path)
    return tuple([*explicit, *_coalesce_without_widening(discovered)])


def _harbor_python(harbor_executable: str) -> str:
    executable = shutil.which(harbor_executable) or harbor_executable
    try:
        first = Path(executable).read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError) as exc:
        raise ArtifactDiscoveryError(
            f"cannot inspect Harbor entry point: {harbor_executable!r}"
        ) from exc
    if not first.startswith("#!"):
        raise ArtifactDiscoveryError(f"Harbor entry point has no interpreter: {executable}")
    command = shlex.split(first[2:].strip())
    if not command:
        raise ArtifactDiscoveryError(f"Harbor entry point has an empty shebang: {executable}")
    if Path(command[0]).name == "env":
        if len(command) != 2 or shutil.which(command[1]) is None:
            raise ArtifactDiscoveryError(f"cannot resolve Harbor shebang: {first}")
        return str(shutil.which(command[1]))
    if not Path(command[0]).is_file():
        raise ArtifactDiscoveryError(f"Harbor interpreter does not exist: {command[0]}")
    return command[0]


def resolve_harbor_dataset_artifacts(
    dataset: str,
    task_names: Sequence[str],
    explicit_roots: Sequence[str],
    *,
    download: bool = True,
    harbor_executable: str = "harbor",
) -> DatasetArtifactResolution:
    """Resolve exact packages in Harbor's environment and return safe paths."""
    interpreter = _harbor_python(harbor_executable)
    with tempfile.TemporaryDirectory(prefix="gv-harbor-artifacts-") as temp_dir:
        output = Path(temp_dir) / "resolution.json"
        command = [
            interpreter,
            str(Path(__file__).resolve()),
            "resolve",
            "--dataset", dataset,
            "--output", str(output),
        ]
        if download:
            command.append("--download")
        for task in task_names:
            command.extend(("--task", task))
        proc = subprocess.run(command, text=True, capture_output=True)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            raise ArtifactDiscoveryError(
                f"Harbor dataset artifact resolution failed ({proc.returncode}): {detail}"
            )
        try:
            payload = json.loads(output.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ArtifactDiscoveryError("Harbor resolver did not produce valid JSON") from exc

    task_packages = {
        str(name): str(path) for name, path in payload["task_packages"].items()
    }
    referenced = {
        str(name): tuple(str(path) for path in paths)
        for name, paths in payload["referenced_by_task"].items()
    }
    audit = {
        str(name): tuple(ArtifactPathAudit(**record) for record in records)
        for name, records in payload["audit_by_task"].items()
    }
    artifacts = merge_artifact_paths(explicit_roots, referenced)
    return DatasetArtifactResolution(
        dataset=str(payload["dataset"]),
        task_packages=task_packages,
        referenced_by_task=referenced,
        audit_by_task=audit,
        artifacts=artifacts,
    )


async def _resolve_with_harbor(
    dataset: str, task_names: Sequence[str], *, download: bool,
) -> dict[str, Any]:
    """Harbor-environment half of resolution (imports Harbor lazily)."""
    from harbor.registry.client.package import PackageDatasetClient
    from harbor.tasks.client import TaskClient

    client = PackageDatasetClient()
    metadata = await client.get_dataset_metadata(dataset)
    available = {task_id.get_name(): task_id for task_id in metadata.task_ids}
    if not task_names:
        task_names = list(available)
    missing = [name for name in task_names if name not in available]
    if missing:
        raise ArtifactDiscoveryError(
            f"tasks are not members of resolved dataset {metadata.name}@{metadata.version}: "
            f"{missing}"
        )
    selected = [available[name] for name in task_names]
    if download:
        batch = await TaskClient().download_tasks(task_ids=selected)
        package_paths = batch.paths
    else:
        package_paths = [task_id.get_local_path() for task_id in selected]

    task_packages: dict[str, str] = {}
    referenced: dict[str, list[str]] = {}
    audit: dict[str, list[dict[str, str | None]]] = {}
    for name, task_id, path in zip(task_names, selected, package_paths):
        if not path.is_dir():
            raise ArtifactDiscoveryError(
                f"exact package is not cached for dry-run: {name}@{task_id.ref}; "
                "run once without --dry-run to download it"
            )
        task_packages[name] = str(path)
        accepted, records = discover_task_package_audit(path)
        referenced[name] = list(accepted)
        audit[name] = [record.to_json() for record in records]
    return {
        "dataset": f"{metadata.name}@{metadata.version}",
        "task_packages": task_packages,
        "referenced_by_task": referenced,
        "audit_by_task": audit,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8")
    temp.replace(path)


def _resolver_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["resolve"])
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--download", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = asyncio.run(
            _resolve_with_harbor(args.dataset, args.task, download=args.download)
        )
        _write_json_atomic(args.output, payload)
    except Exception as exc:  # CLI boundary: parent reports stderr verbatim.
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_resolver_main(sys.argv[1:]))
