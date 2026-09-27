"""One source-pinned, fresh-state online verifier evaluation job.

The worker exercises the production live path: ``Service.commit(final)``
performs obligation compilation, Critic review, Oracle enumeration,
execution (agent or typed adapter), and an independent Judger audit.  It is
evaluation-only: expected answers never enter the state and no task-specific
method is selected here.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys
import time
import uuid


SCHEMA = "gv-full-online-worker-plan-v2"
RESULT_SCHEMA = "gv-full-online-worker-result-v2"
RUNTIME = {
    "model": "gpt-5.6-luna", "effort": "medium",
    "service_tier": "standard",
}
SUPPORT_FILES = (
    "run_full_online_worker.py", "observe_codex_attempts.py",
    "analyze_codex_observer.py", "eval_offline_policy.py",
    "run_typed_adapter_eval.py", "project_full_offline_result.py",
)
FINAL_TEXT = (
    "The delivered subject satisfies the complete authoritative task contract."
)
_PLAN_KEYS = {
    "schema", "source_root", "source_commit", "source_fingerprint",
    "input_source", "input_seal", "output_root", "arm",
    "runtime_read_roots", "support_sha256",
}
_HASH = re.compile(r"[0-9a-f]{64}\Z")
_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_ATTEMPT = re.compile(r"[0-9a-f]{32}\Z")
_TASK_CAP = 1024 * 1024
_INPUT_FILE_CAP = 32 * 1024 * 1024
_INPUT_TOTAL_CAP = 128 * 1024 * 1024
_INPUT_FILE_COUNT_CAP = 512
_INPUT_DIRECTORY_COUNT_CAP = 128
_INPUT_DEPTH_CAP = 24
_INPUT_PATH_BYTES_CAP = 4096
_RUNTIME_STATE_TOP_LEVEL = frozenset({
    "artifacts", "work", "workspace", "events.jsonl", "claims.sqlite",
    "claims.sqlite-shm", "claims.sqlite-wal",
    # frozen per-claim ledger prefixes published during verification
    "record",
    # the final claim under check, published for the seats to read
    "candidate_response.txt",
    # how far the hook feed was ingested, and the live server's pid
    "hook_feed.offset", "server.pid",
})
_ARTIFACT_CAP = 32 * 1024 * 1024
_EVENT_CAP = 64 * 1024 * 1024


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def _inside(parent: Path, child: Path) -> bool:
    try:
        return os.path.commonpath((str(parent), str(child))) == str(parent)
    except ValueError:
        return False


def _path(value, *, existing=False) -> Path:
    if not isinstance(value, str) or not value or not os.path.isabs(value):
        raise ValueError("worker paths must be canonical absolute strings")
    path = Path(value)
    if str(path.resolve(strict=existing)) != value:
        raise ValueError("worker paths must contain no aliases")
    return path


def _strict_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON key")
        value[key] = item
    return value


def _nonfinite(_value):
    raise ValueError("non-finite JSON constant")


def _read_json(path, cap=64 * 1024):
    with Path(path).open("rb") as handle:
        raw = handle.read(cap + 1)
    if len(raw) > cap:
        raise ValueError("JSON input exceeds worker limit")
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_strict_object, parse_constant=_nonfinite,
    )
    return value, _sha_bytes(raw)


def _stat_identity(item) -> tuple:
    return (
        item.st_dev, item.st_ino, item.st_mode, item.st_nlink, item.st_uid,
        item.st_gid, item.st_size, item.st_mtime_ns, item.st_ctime_ns,
    )


def _stable_regular_bytes(path: Path, cap: int) -> tuple[bytes, dict]:
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > cap):
            raise ValueError("input must be a bounded private regular file")
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
    if len(raw) > cap:
        raise ValueError("input exceeds worker limit")
    current = os.lstat(path)
    if (_stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)):
        raise ValueError("input changed while it was read")
    return raw, {
        "sha256": _sha_bytes(raw), "bytes": len(raw),
        "mode": stat.S_IMODE(after.st_mode), "nlink": after.st_nlink,
    }


def _stable_regular_bytes_at(parent_fd: int, name: str,
                             cap: int) -> tuple[bytes, dict]:
    before_path = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_NONBLOCK", 0) | getattr(os, "O_CLOEXEC", 0))
    descriptor = os.open(name, flags, dir_fd=parent_fd)
    try:
        before = os.fstat(descriptor)
        if (_stat_identity(before) != _stat_identity(before_path)
                or not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or before.st_size > cap):
            raise ValueError("input must be a bounded private regular file")
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
    current = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if (len(raw) > cap or _stat_identity(before) != _stat_identity(after)
            or _stat_identity(after) != _stat_identity(current)):
        raise ValueError("input changed while it was read")
    return raw, {
        "sha256": _sha_bytes(raw), "bytes": len(raw),
        "mode": stat.S_IMODE(after.st_mode), "nlink": after.st_nlink,
    }


def _stable_tree_file(root: Path, relative: str, cap: int) -> tuple[bytes, dict]:
    parts = Path(relative).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError("input tree file has an invalid relative path")
    directory_flags = (
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    descriptors = [os.open(root, directory_flags)]
    try:
        for part in parts[:-1]:
            descriptors.append(os.open(
                part, directory_flags, dir_fd=descriptors[-1]))
        return _stable_regular_bytes_at(descriptors[-1], parts[-1], cap)
    finally:
        for descriptor in reversed(descriptors):
            os.close(descriptor)


def input_snapshot(root: str | os.PathLike[str], *, runtime_state=False) -> dict:
    """Seal one bounded arbitrary input tree without following any alias."""
    directory = Path(root)
    try:
        root_info = os.lstat(directory)
    except OSError as error:
        raise ValueError("input root is unavailable") from error
    if stat.S_ISLNK(root_info.st_mode) or not stat.S_ISDIR(root_info.st_mode):
        raise ValueError("input root must be a real directory")

    flags = (os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
             | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0))
    root_fd = os.open(directory, flags)
    opened_root = os.fstat(root_fd)
    if _stat_identity(opened_root) != _stat_identity(root_info):
        os.close(root_fd)
        raise ValueError("input root changed before traversal")

    files, directories, payload = {}, {}, []
    total_bytes = 0
    def walk(parent_fd: int, parent_rel: str, parent_identity) -> None:
        nonlocal total_bytes
        try:
            names = sorted(os.listdir(parent_fd))
        except OSError as error:
            raise ValueError("input directory cannot be enumerated") from error
        for name in names:
            if (not parent_rel and runtime_state
                    and name in _RUNTIME_STATE_TOP_LEVEL):
                continue
            if (not parent_rel and not runtime_state
                    and name in _RUNTIME_STATE_TOP_LEVEL):
                raise ValueError(
                    f"input tree reserves verifier state name {name!r}")
            rel = f"{parent_rel}/{name}" if parent_rel else name
            if (len(rel.encode("utf-8")) > _INPUT_PATH_BYTES_CAP
                    or len(Path(rel).parts) > _INPUT_DEPTH_CAP):
                raise ValueError("input tree path exceeds worker bounds")
            try:
                info = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except OSError as error:
                raise ValueError("input tree entry cannot be inspected") from error
            if stat.S_ISLNK(info.st_mode):
                raise ValueError("input tree must not contain symlinks")
            if stat.S_ISDIR(info.st_mode):
                if len(directories) >= _INPUT_DIRECTORY_COUNT_CAP:
                    raise ValueError("input tree has too many directories")
                record = {"mode": stat.S_IMODE(info.st_mode)}
                if record["mode"] & ~0o777:
                    raise ValueError(
                        "input directory mode contains special permission bits")
                directories[rel] = record
                payload.append({"kind": "directory", "path": rel, **record})
                child_fd = os.open(name, flags, dir_fd=parent_fd)
                try:
                    opened = os.fstat(child_fd)
                    if _stat_identity(opened) != _stat_identity(info):
                        raise ValueError("input directory changed before traversal")
                    walk(child_fd, rel, opened)
                    after = os.fstat(child_fd)
                    rebound = os.stat(
                        name, dir_fd=parent_fd, follow_symlinks=False)
                    if (_stat_identity(opened) != _stat_identity(after)
                            or _stat_identity(after)
                            != _stat_identity(rebound)):
                        raise ValueError("input directory changed during traversal")
                finally:
                    os.close(child_fd)
                continue
            if not stat.S_ISREG(info.st_mode):
                raise ValueError("input tree must contain only regular files")
            if len(files) >= _INPUT_FILE_COUNT_CAP:
                raise ValueError("input tree has too many files")
            cap = _TASK_CAP if rel == "task.md" else _INPUT_FILE_CAP
            raw, record = _stable_regular_bytes_at(parent_fd, name, cap)
            if record["mode"] & ~0o777:
                raise ValueError("input file mode contains special permission bits")
            total_bytes += record["bytes"]
            if total_bytes > _INPUT_TOTAL_CAP:
                raise ValueError("input tree exceeds total byte limit")
            files[rel] = record
            payload.append({"kind": "file", "path": rel, **record})
            if rel == "task.md":
                if not raw.decode("utf-8", errors="strict").strip():
                    raise ValueError("task.md must be nonempty UTF-8")
        if _stat_identity(os.fstat(parent_fd)) != _stat_identity(parent_identity):
            raise ValueError("input directory changed during traversal")

    try:
        walk(root_fd, "", opened_root)
        after_root = os.fstat(root_fd)
        rebound_root = os.lstat(directory)
        if (_stat_identity(opened_root) != _stat_identity(after_root)
                or _stat_identity(after_root) != _stat_identity(rebound_root)):
            raise ValueError("input root changed during traversal")
    finally:
        os.close(root_fd)
    if "task.md" not in files:
        raise ValueError("input tree requires a root task.md")
    payload.sort(key=lambda item: (item["path"], item["kind"]))
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return {
        "schema": "gv-online-input-tree-seal-v2",
        "seal": _sha_bytes(encoded), "files": files,
        "directories": directories, "file_count": len(files),
        "directory_count": len(directories), "total_bytes": total_bytes,
    }


def _support(name):
    path = Path(__file__).with_name(name + ".py")
    module_name = "_gv_online_support_" + name + "_" + _sha(path)[:16]
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ValueError("cannot load pinned evaluation support")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def validate_plan(plan):
    if type(plan) is not dict or set(plan) != _PLAN_KEYS or plan.get("schema") != SCHEMA:
        raise ValueError("wrong online worker plan schema or fields")
    if plan.get("arm") not in {"control", "treatment"}:
        raise ValueError("arm must be control or treatment")
    if (not isinstance(plan.get("source_commit"), str)
            or not _COMMIT.fullmatch(plan["source_commit"])):
        raise ValueError("source commit must be an exact Git identity")
    for key in ("source_fingerprint", "input_seal"):
        if not isinstance(plan.get(key), str) or not _HASH.fullmatch(plan[key]):
            raise ValueError("source and input identities must be pinned")
    source = _path(plan["source_root"], existing=True)
    inputs = _path(plan["input_source"], existing=True)
    output = _path(plan["output_root"])
    if not source.is_dir() or not inputs.is_dir() or (source / ".git").exists():
        raise ValueError("source/input must be real dirs and source must be frozen")
    if output.exists() or output.is_symlink():
        raise ValueError("worker output must be fresh")
    for left, right in ((source, inputs), (source, output), (inputs, output)):
        if _inside(left, right) or _inside(right, left):
            raise ValueError("source, input, and output roots must be disjoint")
    for temporary in (Path("/private/tmp"), Path("/private/var/folders")):
        if _inside(temporary, output):
            raise ValueError("paid output must not be in host temporary trees")
    roots = plan.get("runtime_read_roots")
    if type(roots) is not list or len(roots) > 16:
        raise ValueError("runtime roots must be an explicit bounded list")
    seen = set()
    for value in roots:
        root = _path(value, existing=True)
        if not root.is_dir() or value in seen or value == "/":
            raise ValueError("runtime roots must be distinct real directories")
        seen.add(value)
        for private in (Path.home().resolve(), source, inputs, output):
            if _inside(root, private) or _inside(private, root):
                raise ValueError("runtime root overlaps a private boundary")
    pins = plan.get("support_sha256")
    if type(pins) is not dict or set(pins) != set(SUPPORT_FILES):
        raise ValueError("every evaluation support file must be pinned")
    for name, digest in pins.items():
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError("invalid support digest")
        if _sha(Path(__file__).with_name(name)) != digest:
            raise ValueError("evaluation support changed")
    snapshot = input_snapshot(inputs)
    if snapshot["seal"] != plan["input_seal"]:
        raise ValueError("operator input seal mismatch")
    return plan


def _environment(plan):
    for name in list(os.environ):
        if name.startswith("GV_") or name in {
                "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP"}:
            del os.environ[name]
    os.environ.update({
        "GV_TYPED_ADAPTERS": "1" if plan["arm"] == "treatment" else "0",
        "GV_TYPED_JUDGER_PACKET": (
            "1" if plan["arm"] == "treatment" else "0"),
        "GV_CONTRACT_CACHE": "0", "GV_APPEND_ONLY_DELTAS": "0",
        "GV_MAX_CONCURRENT_SEATS": "12",
        "GV_OFFLINE_RUNTIME_READ_ROOTS": json.dumps(
            plan["runtime_read_roots"], separators=(",", ":")),
        "PATH": "/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        "TMPDIR": "/private/tmp", "PYTHONDONTWRITEBYTECODE": "1",
        "TZ": "Europe/London",
    })
    os.environ.pop("GV_SHARED_DIR", None)
    sys.dont_write_bytecode = True



def _frozen_module(source: Path, current: str, legacy: str) -> str:
    """The module path of a runtime part in the frozen tree: its path
    before the refactor when only that file exists there, else its
    current path."""
    def present(name):
        return source.joinpath(*name.split(".")).with_suffix(".py").is_file()
    return legacy if present(legacy) and not present(current) else current

def _load_runtime(source):
    if any(name == "verifier" or name.startswith("verifier.")
           for name in sys.modules):
        raise RuntimeError("worker requires a fresh verifier import process")
    source = Path(source)
    if any((source / "verifier").rglob("*.pyc")):
        raise ValueError("frozen source contains bytecode")
    if any(path.is_symlink() for path in (source / "verifier").rglob("*.py")):
        raise ValueError("frozen source contains aliased Python files")
    sys.path.insert(0, str(source))
    modules = {
        "offline": importlib.import_module("verifier.offline"),
        "backend": importlib.import_module(_frozen_module(
            source, "verifier.llm.codex.runtime",
            "verifier.llm.codex_backend")),
        "mcp": importlib.import_module("verifier.mcp_server"),
        # the composition root; a tree before it had one assembles the
        # kernel in the MCP server
        "assembly": importlib.import_module(_frozen_module(
            source, "verifier.app.assembly", "verifier.mcp_server")),
        "certificate": importlib.import_module("verifier.engine.certificate"),
        "store": importlib.import_module("verifier.core.store"),
        "typed_judger_backend": importlib.import_module(
            "verifier.llm.typed_judger_backend"),
        "typed_judger_packet": importlib.import_module(_frozen_module(
            source, "verifier.execution.judger_packets.typed",
            "verifier.engine.typed_judger_packet")),
    }
    for name, module in list(sys.modules.items()):
        if name == "verifier" or name.startswith("verifier."):
            location = getattr(module, "__file__", None)
            if not location or not _inside(source, Path(location).resolve()):
                raise RuntimeError("selected verifier source was shadowed")
    return modules


def preflight(plan):
    validate_plan(plan)
    _environment(plan)
    source = Path(plan["source_root"])
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen source fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    del observer
    _support("analyze_codex_observer")
    _support("project_full_offline_result")
    policy = _support("eval_offline_policy")
    modules = _load_runtime(source)
    if not hasattr(modules["backend"], "_run_codex_process"):
        raise ValueError("source lacks the pinned observer seam")
    policy_plan = policy.plan_eval_online_policy(
        modules["backend"], bindings=(policy.OfflinePolicyBinding(
            state_dir=plan["input_source"],
            runtime_read_roots=tuple(plan["runtime_read_roots"])),),
    )
    return {
        "schema": SCHEMA, "preflight_complete": True, "model_calls": 0,
        "source_fingerprint": plan["source_fingerprint"],
        "input_seal": plan["input_seal"], "runtime": RUNTIME,
        "online_policy_identity": policy_plan.identity,
        "input_contract": "bounded-sealed-arbitrary-tree-v2",
        "hook_feed_ingest": "production-Service.drain_hooks",
        "browser_runtime_pins": {
            "supported": False,
            "reason": "no source-pinned coordinator schema",
        },
    }


def _clone_inputs(source: Path, state: Path, expected_seal: str) -> dict:
    before = input_snapshot(source)
    if before["seal"] != expected_seal:
        raise ValueError("input changed before cloning")
    state.mkdir(exist_ok=False)
    for name in sorted(before["directories"],
                       key=lambda item: (len(Path(item).parts), item)):
        (state / name).mkdir(mode=0o700)
    for name, expected in sorted(before["files"].items()):
        cap = _TASK_CAP if name == "task.md" else _INPUT_FILE_CAP
        raw, observed = _stable_tree_file(source, name, cap)
        if observed != expected:
            raise ValueError("input changed while cloning")
        target = state / name
        descriptor = os.open(
            target,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        try:
            offset = 0
            while offset < len(raw):
                written = os.write(descriptor, raw[offset:])
                if written <= 0:
                    raise OSError("short input clone write")
                offset += written
            os.fchmod(descriptor, expected["mode"])
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    for name in sorted(before["directories"],
                       key=lambda item: (-len(Path(item).parts), item)):
        os.chmod(state / name, before["directories"][name]["mode"])
    clone = input_snapshot(state)
    if clone["seal"] != expected_seal or input_snapshot(source) != before:
        raise ValueError("input changed while cloning")
    return clone


def _write_new(path: Path, value) -> None:
    raw = (json.dumps(value, ensure_ascii=False, sort_keys=True,
                      allow_nan=False) + "\n").encode("utf-8")
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(raw)
        handle.flush()
        os.fsync(handle.fileno())


def _read_jsonl(path: Path, cap=_EVENT_CAP) -> list[dict]:
    rows, _digest = _read_jsonl_identity(path, cap=cap)
    return rows


def _read_jsonl_identity(path: Path, cap=_EVENT_CAP) -> tuple[list[dict], str]:
    raw, _identity = _stable_regular_bytes(path, cap)
    if not raw or not raw.endswith(b"\n"):
        raise ValueError("audit ledger is oversized or incomplete")
    rows = [json.loads(line, object_pairs_hook=_strict_object,
                       parse_constant=_nonfinite)
            for line in raw.decode("utf-8", errors="strict").split("\n")[:-1]]
    if not rows or any(type(row) is not dict for row in rows):
        raise ValueError("audit ledger must contain objects")
    return rows, _sha_bytes(raw)


def _artifact(state: Path, digest: str) -> tuple[dict, str]:
    if not isinstance(digest, str) or not _HASH.fullmatch(digest):
        raise ValueError("invalid artifact digest")
    path = state / "artifacts" / (digest + ".json")
    raw, _identity = _stable_regular_bytes(path, _ARTIFACT_CAP)
    value = json.loads(
        raw.decode("utf-8", errors="strict"),
        object_pairs_hook=_strict_object, parse_constant=_nonfinite,
    )
    raw_sha = _sha_bytes(raw)
    if type(value) is not dict:
        raise ValueError("artifact must be a JSON object")
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    if _sha_bytes(encoded) != digest:
        raise ValueError("artifact content address mismatch")
    return value, raw_sha


def _artifact_inventory(state: Path) -> list[dict]:
    root = state / "artifacts"
    result = []
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not re.fullmatch(r"[0-9a-f]{64}\.json", path.name):
            raise ValueError("artifact store contains an invalid entry")
        raw, identity = _stable_regular_bytes(path, _ARTIFACT_CAP)
        payload = json.loads(
            raw.decode("utf-8", errors="strict"),
            object_pairs_hook=_strict_object, parse_constant=_nonfinite,
        )
        canonical = json.dumps(
            payload, sort_keys=True, ensure_ascii=False,
            separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
        if _sha_bytes(canonical) != path.stem:
            raise ValueError("artifact filename does not match canonical content")
        result.append({"name": path.name, "bytes": identity["bytes"],
                       "raw_sha256": _sha_bytes(raw)})
    if not result:
        raise ValueError("artifact store is empty")
    return result


_ENTRY_FIELDS = (
    "id", "kind", "text", "status", "effective_status",
    "effective_basis", "effective_strength", "open_premises",
    "premise_refuted", "reason", "basis", "strength",
)
_OBLIGATION_FIELDS = (
    "id", "text", "status", "reason", "basis", "strength", "severity",
    "clause", "refutable_by_absence",
)


def _certificate_semantics(certificate: dict) -> dict:
    if type(certificate) is not dict or type(certificate.get("slice")) is not list:
        raise ValueError("certificate has no exact slice")
    rows = []
    for entry in certificate["slice"]:
        if type(entry) is not dict or any(key not in entry for key in _ENTRY_FIELDS):
            raise ValueError("certificate entry is incomplete")
        obligations = entry.get("obligations")
        if type(obligations) is not list:
            raise ValueError("certificate obligations are malformed")
        projected = {key: entry[key] for key in _ENTRY_FIELDS}
        projected["obligations"] = []
        for obligation in obligations:
            if (type(obligation) is not dict
                    or any(key not in obligation for key in _OBLIGATION_FIELDS)):
                raise ValueError("certificate obligation is incomplete")
            projected["obligations"].append(
                {key: obligation[key] for key in _OBLIGATION_FIELDS})
        rows.append(projected)
    return {
        "final": certificate.get("final"),
        "verdict": certificate.get("verdict"),
        "verdict_strength": certificate.get("verdict_strength"),
        "slice": rows,
    }


def _score_semantics(features: dict) -> dict:
    excluded = {"certificate_id", "fingerprint"}
    if type(features) is not dict or features.get("unscoreable") is not False:
        raise ValueError("online certificate did not yield a scoreable view")
    return {key: features[key] for key in sorted(features) if key not in excluded}


def _content_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, ensure_ascii=False,
        separators=(",", ":"), allow_nan=False,
    ).encode("utf-8")
    return _sha_bytes(encoded)


def _final_certificate_entry(certificate: dict) -> dict:
    if type(certificate) is not dict:
        raise ValueError("certificate must be an object")
    final = certificate.get("final")
    entries = certificate.get("slice")
    if not isinstance(final, str) or not final or type(entries) is not list:
        raise ValueError("certificate has no exact final slice")
    matches = [entry for entry in entries
               if type(entry) is dict and entry.get("id") == final]
    if len(matches) != 1:
        raise ValueError("certificate has no unique final entry")
    return matches[0]


def _feature_expectations(certificate: dict, source_fingerprint: str) -> dict:
    final = _final_certificate_entry(certificate)
    entries = certificate["slice"]
    claims, effective_claims, obligations = {}, {}, {}
    conditional_verified = 0
    for entry in entries:
        if type(entry) is not dict:
            raise ValueError("certificate slice contains a non-object")
        status = entry.get("status")
        effective = entry.get("effective_status", status)
        if not isinstance(status, str) or not isinstance(effective, str):
            raise ValueError("certificate claim status is malformed")
        claims[status] = claims.get(status, 0) + 1
        effective_claims[effective] = effective_claims.get(effective, 0) + 1
        conditional_verified += status == "verified" and effective != "verified"
        values = entry.get("obligations")
        if type(values) is not list:
            raise ValueError("certificate claim obligations are malformed")
        for obligation in values:
            if (type(obligation) is not dict
                    or not isinstance(obligation.get("id"), str)
                    or not isinstance(obligation.get("status"), str)):
                raise ValueError("certificate obligation status is malformed")
            obligations[obligation["id"]] = obligation["status"]
    final_values = final["obligations"]

    def counts(values):
        result = {}
        for value in values:
            result[value] = result.get(value, 0) + 1
        return result

    return {
        "unscoreable": False,
        "verdict": certificate.get("verdict"),
        "verdict_strength": certificate.get("verdict_strength"),
        "certificate_id": certificate.get("certificate_id"),
        "slice_size": len(entries),
        "claims": claims,
        "claims_effective": effective_claims,
        "conditional_verified": conditional_verified,
        "obligations": counts(obligations.values()),
        "obligations_total": len(obligations),
        "final_obligations": counts(
            item.get("status") for item in final_values),
        "final_obligations_total": len(final_values),
        "final_contrary_signals": sum(
            item.get("status") == "refuted"
            and item.get("severity", "must") != "must"
            for item in final_values),
        "extract_rounds": 0,
        "fingerprint": source_fingerprint,
    }


def _validate_result_views(commit: dict, certificate: dict, features: dict,
                           rank: tuple, report: dict,
                           source_fingerprint: str,
                           feature_schema_version) -> dict:
    if type(commit) is not dict:
        raise ValueError("online commit result is malformed")
    final = _final_certificate_entry(certificate)
    claim_id = certificate["final"]
    if (commit.get("claim_id") != claim_id
            or final.get("kind") != "final"
            or final.get("text") != FINAL_TEXT
            or final.get("status") != commit.get("status")
            or final.get("effective_status") != final.get("status")
            or final.get("open_premises") != []
            or final.get("premise_refuted") is not None):
        raise ValueError("commit and final certificate entry disagree")
    status = final["status"]
    verdict = certificate.get("verdict")
    strength = certificate.get("verdict_strength")
    if status == "verified":
        if (verdict not in {"verified", "supported"}
                or strength not in {"decisive", "bounded", "analytic"}
                or (verdict == "verified") != (strength == "decisive")):
            raise ValueError("verified final has an inconsistent certificate verdict")
    elif (verdict != {"refuted": "refuted", "partial": "partial"}.get(
            status, "unverified") or strength is not None):
        raise ValueError("final status and certificate verdict disagree")

    if type(features) is not dict:
        raise ValueError("online score features are malformed")
    expected = _feature_expectations(certificate, source_fingerprint)
    if any(key not in features or features[key] != value
           for key, value in expected.items()):
        raise ValueError("online score features disagree with the certificate")
    if (feature_schema_version is None
            or features.get("feature_schema_version") != feature_schema_version
            or not isinstance(rank, tuple) or len(rank) != 2
            or any(type(item) is not int for item in rank)
            or rank == (-1, 0)):
        raise ValueError("online score rank or feature schema is malformed")

    if (type(report) is not dict or report.get("complete") is not True
            or report.get("findings") != []
            or report.get("final_claim_id") != claim_id
            or not isinstance(report.get("events_sha256"), str)
            or not _HASH.fullmatch(report["events_sha256"])):
        raise ValueError("ordered contract projection is incomplete")
    contract = report.get("ordered_contract")
    outcomes = report.get("ordered_outcomes")
    final_obligations = final.get("obligations")
    if (type(contract) is not list or type(outcomes) is not list
            or type(final_obligations) is not list
            or not final_obligations
            or len(contract) != len(final_obligations)
            or len(outcomes) != len(final_obligations)):
        raise ValueError("quality projection does not cover the final contract")
    by_id = {item.get("id"): item for item in final_obligations
             if type(item) is dict}
    if len(by_id) != len(final_obligations):
        raise ValueError("final certificate obligations are not unique")
    for contract_item, outcome in zip(contract, outcomes):
        oid = contract_item.get("id") if type(contract_item) is dict else None
        certificate_item = by_id.get(oid)
        if (certificate_item is None or type(outcome) is not dict
                or outcome.get("obligation_id") != oid
                or outcome.get("evaluated") is not True
                or any(outcome.get(key) != certificate_item.get(key)
                       for key in ("status", "reason", "basis", "strength"))
                or any(contract_item.get(key) != certificate_item.get(key)
                       for key in ("text", "severity", "clause",
                                   "refutable_by_absence"))):
            raise ValueError("quality projection disagrees with final certificate")
    return final


def _execute_live_chain(modules, state: Path, backend, projection,
                        source_fingerprint: str) -> dict:
    """Execute the production live chain; this helper adds no verifier stage."""
    def factory(service):
        assembly = modules["assembly"]
        build = getattr(assembly, "build_kernel", None) \
            or assembly._default_kernel
        return build(
            service, RUNTIME["model"], backend=backend, critic_rounds=3,
            parallel_obligations=4, parallel_claims=1,
            # this audit pins the separate final path's stage sequence
            unify_final=False,
        )

    service = modules["mcp"].Service(
        str(state), kernel_factory=factory,
        trigger=modules["store"].TRIGGER_ON_COMMIT, live_world=True,
        critic_rounds=3, parallel_obligations=4, parallel_claims=1,
    )
    commit = service.commit("final", "final", FINAL_TEXT, [])
    claim_id = commit.get("claim_id") if type(commit) is dict else None
    if not isinstance(claim_id, str) or not claim_id:
        raise ValueError("online final commit was not retained")
    certificate = modules["certificate"].request_certificate(
        service.store, service.kernel, service.log, service.artifacts, claim_id,
    )
    features = modules["offline"]._features(
        certificate, 0, source_fingerprint)
    rank = modules["offline"].rank_key(features)
    report = projection.project_full_offline_result(
        str(state / "events.jsonl"), certificate)
    _validate_result_views(
        commit, certificate, features, rank, report, source_fingerprint,
        getattr(modules["offline"], "FEATURE_SCHEMA_VERSION", None))
    event_rows, event_sha = _read_jsonl_identity(state / "events.jsonl")
    if event_sha != report["events_sha256"]:
        raise ValueError("projection event identity changed after projection")
    artifact_digest = certificate.get("certificate_id")
    stored_certificate, raw_sha = _artifact(state, artifact_digest)
    expected_stored = {key: value for key, value in certificate.items()
                       if key != "certificate_id"}
    if stored_certificate != expected_stored:
        raise ValueError("published certificate differs from returned certificate")
    semantic_evidence = _semantic_evidence_projection(state, certificate)
    return {
        "commit": commit, "certificate": certificate,
        "certificate_artifact_raw_sha256": raw_sha,
        "features": features,
        "quality_view": {
            "projection": {
                "schema": report.get("schema"),
                "events_sha256": event_sha,
                "events_rows": len(event_rows),
                "final_claim_id": report["final_claim_id"],
            },
            "ordered_contract": report["ordered_contract"],
            "ordered_outcomes": report["ordered_outcomes"],
            "semantic_evidence": semantic_evidence,
            "certificate": _certificate_semantics(certificate),
            "score": {"features": _score_semantics(features),
                      "rank_key": list(rank)},
        },
    }


def _verify_projection_identity(state: Path, chain: dict) -> dict:
    if type(chain) is not dict:
        raise ValueError("online result has no projection identity")
    quality = chain.get("quality_view")
    identity = quality.get("projection") if type(quality) is dict else None
    if (type(identity) is not dict
            or not isinstance(identity.get("events_sha256"), str)
            or not _HASH.fullmatch(identity["events_sha256"])
            or type(identity.get("events_rows")) is not int
            or identity["events_rows"] <= 0):
        raise ValueError("online result has no complete projection identity")
    rows, digest = _read_jsonl_identity(state / "events.jsonl")
    if (identity["events_sha256"] != digest
            or identity["events_rows"] != len(rows)):
        raise ValueError("verification events changed after projection")
    return {"events_sha256": digest, "events_rows": len(rows)}


def _canonical_json_value(value):
    """Copy JSON business data losslessly; canonical encoding supplies order."""
    if type(value) is dict:
        if any(not isinstance(key, str) for key in value):
            raise ValueError("semantic evidence contains a non-string key")
        return {key: _canonical_json_value(value[key]) for key in sorted(value)}
    if type(value) is list:
        return [_canonical_json_value(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("semantic evidence contains a non-JSON value")


def _logical_ref(value, state: Path, inputs: dict) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("evidence input has no logical reference")
    if value == "ledger:events" or re.fullmatch(r"event:[1-9][0-9]*", value):
        return value
    if value.startswith(("ledger:", "event:")):
        raise ValueError("evidence has an unauthenticated ledger reference")
    normalized = os.path.normpath(value)
    if os.path.isabs(normalized):
        state_root = os.path.abspath(os.fspath(state))
        if normalized == os.path.join(state_root, "events.jsonl"):
            return "ledger:events"
        try:
            normalized = os.path.relpath(normalized, state_root)
        except ValueError as error:
            raise ValueError(
                "evidence input cannot be bound to the sealed tree") from error
    if normalized == ".." or normalized.startswith(".." + os.sep):
        raise ValueError("evidence input cannot be bound to the sealed tree")
    logical = Path(normalized).as_posix()
    if logical == "." or logical in inputs["files"] \
            or logical in inputs["directories"]:
        return logical
    raise ValueError("evidence input cannot be bound to the sealed tree")


def _semantic_adapter_arguments(arguments: dict, state: Path,
                                inputs: dict) -> dict:
    """Project only the adapter envelope's declared primary address.

    Nested values such as ``payload_equals`` are task-domain data. Their keys
    and values are never interpreted as verifier metadata.
    """
    result = _canonical_json_value(arguments)
    if "address" in arguments:
        result["address"] = _logical_ref(arguments["address"], state, inputs)
    return result


def _semantic_observed_eof(value: dict, state: Path, inputs: dict) -> dict:
    if type(value) is not dict:
        raise ValueError("observed EOF identity is malformed")
    volatile = {
        "observation_started_at_unix_ns",
        "observation_completed_at_unix_ns",
        "observation_started_monotonic_ns",
        "observation_completed_monotonic_ns",
        "monotonic_clock_scope",
    }
    result = {key: _canonical_json_value(item)
              for key, item in sorted(value.items())
              if key not in volatile and key != "file_identity"}
    if "file_identity" in value:
        identity = value["file_identity"]
        if type(identity) is not dict or "path" not in identity:
            raise ValueError("observed EOF file identity is malformed")
        result["file_identity"] = {
            "logical_ref": _logical_ref(identity["path"], state, inputs),
            **{key: _canonical_json_value(identity[key])
               for key in ("mode", "links", "bytes") if key in identity},
        }
    return result


def _semantic_source_identity(value: dict, state: Path, inputs: dict) -> dict:
    """Project a known adapter consumed-input/source identity envelope."""
    if type(value) is not dict or not isinstance(value.get("ref"), str):
        raise ValueError("adapter source identity is malformed")
    result = {key: _canonical_json_value(item)
              for key, item in sorted(value.items())
              if key not in {"ref", "mtime_ns", "observed_eof"}}
    result["logical_ref"] = _logical_ref(value["ref"], state, inputs)
    if "observed_eof" in value:
        result["observed_eof"] = _semantic_observed_eof(
            value["observed_eof"], state, inputs)
    return result


def _semantic_observation(value, state: Path, inputs: dict):
    """Project known observation-envelope identities, preserving business data."""
    if type(value) is not dict:
        return _canonical_json_value(value)
    result = _canonical_json_value(value)
    for key in ("address", "ref"):
        if key in value:
            result[key] = _logical_ref(value[key], state, inputs)
    if "identity" in value:
        result["identity"] = _semantic_source_identity(
            value["identity"], state, inputs)
    if "snapshot" in value:
        result["snapshot"] = _semantic_source_identity(
            value["snapshot"], state, inputs)
    return result


def _semantic_agent_snapshot(snapshot: dict) -> list[dict]:
    if type(snapshot) is not dict:
        raise ValueError("agent evidence snapshot is malformed")
    result = []
    for name, content in sorted(snapshot.items()):
        if not isinstance(name, str):
            raise ValueError("agent snapshot path is malformed")
        normalized = os.path.normpath(name)
        if (os.path.isabs(normalized) or normalized == ".."
                or normalized.startswith(".." + os.sep)):
            raise ValueError("agent snapshot path escapes its workspace")
        canonical_content = _canonical_json_value(content)
        result.append({
            "logical_ref": Path(normalized).as_posix(),
            "content_sha256": _content_digest(canonical_content),
        })
    return result


def _sealed_ref_digest(logical_ref: str, inputs: dict):
    files = inputs["files"]
    if logical_ref in files:
        return files[logical_ref]["sha256"]
    prefix = "" if logical_ref == "." else logical_ref.rstrip("/") + "/"
    children = [{"path": name[len(prefix):], "sha256": row["sha256"]}
                for name, row in sorted(files.items())
                if name.startswith(prefix)]
    return _content_digest(children) if children else None


def _semantic_inputs(values, execution: dict, inputs: dict,
                     state: Path) -> list[dict]:
    if type(values) is not list:
        raise ValueError("evidence inputs are malformed")
    consumed = execution.get("consumed", []) if type(execution) is dict else []
    if type(consumed) is not list:
        raise ValueError("adapter consumed-input receipt is malformed")
    consumed_by_ref = {}
    for item in consumed:
        if type(item) is not dict or not isinstance(item.get("ref"), str):
            raise ValueError("adapter consumed-input identity is malformed")
        logical = _logical_ref(item["ref"], state, inputs)
        if logical in consumed_by_ref:
            raise ValueError("adapter consumed one logical input more than once")
        consumed_by_ref[logical] = item
    projected = []
    for item in values:
        if (type(item) is not dict
                or not isinstance(item.get("provenance"), str)
                or not isinstance(item.get("role"), str)):
            raise ValueError("evidence declared input is malformed")
        logical = _logical_ref(item.get("ref"), state, inputs)
        consumed_item = consumed_by_ref.get(logical, {})
        digest = next((consumed_item.get(key) for key in (
            "sha256", "event_digest", "events_digest",
            "prefix_events_digest", "raw_sha256",
        ) if isinstance(consumed_item.get(key), str)
            and _HASH.fullmatch(consumed_item[key])), None)
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            digest = _sealed_ref_digest(logical, inputs)
        if (logical.startswith(("ledger:", "event:"))
                and (not isinstance(digest, str)
                     or not _HASH.fullmatch(digest))):
            raise ValueError("ledger input has no authenticated content digest")
        projected.append({
            "logical_ref": logical,
            "provenance": item["provenance"], "role": item["role"],
            "content_sha256": digest,
        })
    return projected


def _semantic_attempt(method: dict, audit: dict, discharge: dict,
                      evidence: dict, inputs: dict, state: Path) -> dict:
    if type(method) is not dict or type(audit) is not dict:
        raise ValueError("bound evidence has no exact Method/Judger pair")
    instrument = discharge.get("instrument")
    execution = discharge.get("execution")
    if isinstance(instrument, str) and instrument.startswith("adapter:"):
        if type(execution) is not dict:
            raise ValueError("typed evidence has no execution envelope")
        arguments = method.get("adapter_arguments", "{}")
        if isinstance(arguments, str):
            arguments = json.loads(
                arguments, object_pairs_hook=_strict_object,
                parse_constant=_nonfinite)
        if type(arguments) is not dict:
            raise ValueError("adapter arguments are not an object")
        mechanism = {
            "kind": "typed_adapter",
            "adapter_identity": instrument.removeprefix("adapter:"),
            "arguments": _semantic_adapter_arguments(
                arguments, state, inputs),
            "consumed": [_semantic_source_identity(item, state, inputs)
                         for item in execution.get("consumed", [])],
            "observation": _semantic_observation(
                execution.get("observation"), state, inputs),
        }
    elif instrument == "agent:discharge":
        reported = discharge.get("reported")
        if type(reported) is not dict:
            raise ValueError("agent evidence has no reported envelope")
        snapshot = discharge.get("files")
        mechanism = {
            "kind": "agent",
            "snapshot": _semantic_agent_snapshot(snapshot),
        }
        if "adapter" in method:
            mechanism["oracle_adapter_request"] = {
                "identity": f"{method['adapter']}@"
                            f"{method.get('adapter_version')}",
                "arguments": _semantic_adapter_arguments(json.loads(
                    method.get("adapter_arguments", "{}"),
                    object_pairs_hook=_strict_object,
                    parse_constant=_nonfinite), state, inputs),
            }
    else:
        raise ValueError("evidence mechanism is not projectable")
    declared_inputs = evidence.get("inputs")
    return {
        "outcome": evidence.get("outcome"),
        "basis": evidence.get("basis"),
        "strength": evidence.get("strength"),
        "witness_kind": evidence.get("witness_kind"),
        "assumed_premises": list(evidence.get("assumed_premises") or []),
        "inputs": _semantic_inputs(
            declared_inputs, execution or {}, inputs, state),
        "bridges": list(evidence.get("bridges") or method.get("bridges") or []),
        "judger": {key: audit.get(key) for key in (
            "sufficient", "faithful", "degradation")},
        "mechanism": mechanism,
    }


def _semantic_evidence_projection(state: Path, certificate: dict) -> dict:
    """Build a path-independent, content-addressed evidence comparison view."""
    inputs = input_snapshot(state, runtime_state=True)
    final = _final_certificate_entry(certificate)
    obligations = []
    evidence_count = 0
    for obligation in final.get("obligations", []):
        projections = []
        values = obligation.get("evidence")
        if type(values) is not list or len(values) > 64:
            raise ValueError("obligation evidence exceeds semantic projection bounds")
        evidence_count += len(values)
        if evidence_count > 512:
            raise ValueError("certificate evidence exceeds semantic projection bounds")
        for evidence in values:
            if type(evidence) is not dict:
                raise ValueError("certificate evidence is malformed")
            binding, _ = _artifact(state, evidence.get("artifact"))
            if binding.get("instrument") != "binding-game":
                raise ValueError("certificate evidence is not a binding game")
            bound = binding.get("bound")
            if bound is None:
                attempts = binding.get("attempts")
                if type(attempts) is not list:
                    raise ValueError("fail-closed binding residue is malformed")
                projections.append({
                    "outcome": evidence.get("outcome"),
                    "basis": evidence.get("basis"),
                    "strength": evidence.get("strength"),
                    "witness_kind": evidence.get("witness_kind"),
                    "assumed_premises": list(
                        evidence.get("assumed_premises") or []),
                    "inputs": _semantic_inputs(
                        evidence.get("inputs") or [], {}, inputs, state),
                    "bridges": list(evidence.get("bridges") or []),
                    "mechanism": {
                        "kind": "fail_closed",
                        "attempted_methods": len(attempts),
                    },
                })
                continue
            if type(bound) is not dict:
                raise ValueError("bound evidence is malformed")
            discharge, _ = _artifact(state, bound.get("discharge_artifact"))
            projections.append(_semantic_attempt(
                bound.get("method"), bound.get("audit"), discharge,
                evidence, inputs, state))
        obligations.append({
            "contract": {key: obligation.get(key) for key in (
                "text", "severity", "clause", "refutable_by_absence")},
            "status": obligation.get("status"),
            "evidence": projections,
        })
    document = {
        "schema": "gv-online-semantic-evidence-v1",
        "input_tree_seal": inputs["seal"],
        "verdict": certificate.get("verdict"),
        "verdict_strength": certificate.get("verdict_strength"),
        # Settle truth only for the verifier's three explicit terminal
        # verdicts. Missing, future, or malformed values fail closed.
        "truth_eligible": certificate.get("verdict")
                          in {"verified", "supported", "refuted"},
        "obligations": obligations,
    }
    return {"content_sha256": _content_digest(document), "document": document}


def _verify_semantic_evidence(state: Path, chain: dict) -> str:
    if (type(chain) is not dict or type(chain.get("certificate")) is not dict
            or type(chain.get("quality_view")) is not dict):
        raise ValueError("online result has no semantic evidence view")
    expected = chain["quality_view"].get("semantic_evidence")
    actual = _semantic_evidence_projection(state, chain["certificate"])
    if expected != actual:
        raise ValueError("semantic evidence changed after projection")
    if actual["content_sha256"] != _content_digest(actual["document"]):
        raise ValueError("semantic evidence content address is invalid")
    return actual["content_sha256"]


def _bound_adapter_digests(state: Path, certificate: dict) -> list[str]:
    final = certificate.get("final")
    entries = [item for item in certificate.get("slice", [])
               if type(item) is dict and item.get("id") == final]
    if len(entries) != 1:
        raise ValueError("certificate has no unique final entry")
    digests = []
    for obligation in entries[0].get("obligations", []):
        for evidence in obligation.get("evidence", []):
            if type(evidence) is not dict:
                raise ValueError("malformed obligation evidence")
            binding_digest = evidence.get("artifact")
            binding, _ = _artifact(state, binding_digest)
            if binding.get("instrument") != "binding-game":
                continue
            bound = binding.get("bound")
            if bound is None:
                continue
            if type(bound) is not dict:
                raise ValueError("binding-game artifact lacks its bound method")
            discharge, _ = _artifact(state, bound.get("discharge_artifact"))
            instrument = discharge.get("instrument")
            if isinstance(instrument, str) and instrument.startswith("adapter:"):
                digests.append(bound["discharge_artifact"])
    return sorted(set(digests))


def _binding_attempt_is_fresh(binding: dict, method: dict,
                              discharge_digest: str, audit: dict) -> bool:
    attempts = binding.get("attempts")
    if type(attempts) is not list:
        return False
    for attempt in attempts:
        if type(attempt) is not dict or attempt.get("method") != method:
            continue
        if (attempt.get("discharge_artifact") == discharge_digest
                and attempt.get("audit") == audit):
            return True
        redos = attempt.get("redos")
        if type(redos) is list and any(
                type(redo) is dict
                and redo.get("artifact") == discharge_digest
                and redo.get("audit") == audit
                for redo in redos):
            return True
    return False


def _audit_event_lineage(state: Path, certificate: dict,
                         events: list[dict]) -> list[dict]:
    for expected_seq, event in enumerate(events, 1):
        if (type(event.get("seq")) is not int or event["seq"] != expected_seq
                or not isinstance(event.get("type"), str)
                or type(event.get("payload")) is not dict):
            raise ValueError("verification event sequence or payload is malformed")

    final = _final_certificate_entry(certificate)
    final_id = certificate["final"]
    expected_final_id = _content_digest({
        "kind": "final", "text": FINAL_TEXT, "deps": [],
    })
    if final_id != expected_final_id:
        raise ValueError("final claim identity does not match the online commit")
    commits = [event for event in events if event["type"] == "commit"]
    if (len(commits) != 1 or commits[0]["payload"] != {
            "label": "final", "kind": "final", "text": FINAL_TEXT,
            "deps": [],
    }):
        raise ValueError("online ledger does not contain exactly its final commit")

    certificate_events = [event for event in events
                          if event["type"] == "certificate"]
    matching_certificates = [event for event in certificate_events
        if event["payload"].get("final") == final_id
        and event["payload"].get("digest") == certificate.get("certificate_id")
        and event["payload"].get("verdict") == certificate.get("verdict")
        and event["payload"].get("verdict_strength")
        == certificate.get("verdict_strength")]
    if len(certificate_events) != 1 or len(matching_certificates) != 1:
        raise ValueError("final certificate event identity is missing or ambiguous")
    certificate_seq = matching_certificates[0]["seq"]
    if certificate_seq != len(events):
        raise ValueError("final certificate must be the terminal verification event")

    compiled_events = [event for event in events
        if event["type"] == "obligations_compiled"
        and event["payload"].get("claim") == final_id
        and event["seq"] < certificate_seq]
    if not compiled_events:
        raise ValueError("final claim has no compiled contract event")
    compiled_event = compiled_events[-1]
    critic_passes = [event for event in events
        if event["type"] == "decomposition_attacked"
        and commits[0]["seq"] < event["seq"] < compiled_event["seq"]
        and event["payload"].get("claim") == final_id
        and event["payload"].get("verdict") == "pass"]
    adjacent_passes = [event for event in critic_passes
                       if event["seq"] + 1 == compiled_event["seq"]]
    if len(adjacent_passes) != 1:
        raise ValueError("compiled final has no unique durable Critic pass")
    critic_pass = adjacent_passes[0]
    if (set(critic_pass["payload"]) != {
                "claim", "round", "verdict", "counterexample_world",
                "missing_condition",
            }
            or type(critic_pass["payload"]["round"]) is not int
            or critic_pass["payload"]["round"] < 0
            or not isinstance(
                critic_pass["payload"]["counterexample_world"], str)
            or not isinstance(
                critic_pass["payload"]["missing_condition"], str)):
        raise ValueError("Critic pass is not adjacent and payload-bound")
    compiled = compiled_event["payload"].get("obligations")
    final_obligations = final.get("obligations")
    if type(compiled) is not list or type(final_obligations) is not list:
        raise ValueError("final obligation lineage is malformed")
    compiled_ids = [item.get("id") if type(item) is dict else None
                    for item in compiled]
    certificate_ids = [item.get("id") if type(item) is dict else None
                       for item in final_obligations]
    if (not compiled_ids
            or any(not isinstance(oid, str) or not oid for oid in compiled_ids)
            or any(not isinstance(oid, str) or not oid for oid in certificate_ids)
            or len(set(compiled_ids)) != len(compiled_ids)
            or len(set(certificate_ids)) != len(certificate_ids)
            or set(compiled_ids) != set(certificate_ids)):
        raise ValueError("compiled and certified final obligations disagree")
    compiled_by_id = {item["id"]: item for item in compiled}
    for item in compiled:
        expected_oid = _content_digest({
            "schema": "gv-obligation-v2", "text": item.get("text"),
            "required": item.get("required_basis"),
            "refutable_by_absence": item.get("refutable_by_absence"),
            "premise_scope": item.get("premise_scope"),
        })
        if item["id"] != expected_oid:
            raise ValueError("compiled obligation identity is not payload-bound")

    lineages = []
    for certificate_item in final_obligations:
        compiled_item = compiled_by_id[certificate_item["id"]]
        oid = compiled_item["id"]
        if (compiled_item.get("premise_scope") != []
                or any(compiled_item.get(key) != certificate_item.get(key)
                       for key in ("text", "severity", "clause",
                                   "refutable_by_absence"))):
            raise ValueError("compiled obligation payload changed in certificate")
        evidence_values = certificate_item.get("evidence")
        if type(evidence_values) is not list or not evidence_values:
            raise ValueError("certified obligation has no retained evidence")

        accepted, residues = None, []
        for evidence in evidence_values:
            if type(evidence) is not dict:
                raise ValueError("certified obligation evidence is malformed")
            binding_digest = evidence.get("artifact")
            binding, _raw_sha = _artifact(state, binding_digest)
            if binding.get("instrument") != "binding-game":
                continue
            bound = binding.get("bound")
            if bound is None:
                residues.append((evidence, binding_digest, binding))
                continue
            if type(bound) is not dict:
                continue
            method = bound.get("method")
            audit = bound.get("audit")
            discharge_digest = bound.get("discharge_artifact")
            if (type(method) is not dict or type(audit) is not dict
                    or not isinstance(method.get("procedure"), str)
                    or not method["procedure"]
                    or audit.get("sufficient") != "yes"
                    or audit.get("faithful") != "yes"
                    or not _binding_attempt_is_fresh(
                        binding, method, discharge_digest, audit)):
                continue
            if binding.get("obligation") != compiled_item.get("text"):
                raise ValueError("binding artifact names a different obligation")
            discharge_artifact, _ = _artifact(state, discharge_digest)
            instrument = discharge_artifact.get("instrument")
            if not (instrument == "agent:discharge"
                    or (isinstance(instrument, str)
                        and instrument.startswith("adapter:"))):
                raise ValueError("bound execution is neither adapter nor agent evidence")

            procedure = method["procedure"][:80]
            method_events = [event for event in events
                if event["type"] == "method_bound"
                and compiled_event["seq"] < event["seq"] < certificate_seq
                and event["payload"].get("claim") == final_id
                and event["payload"].get("obligation") == oid
                and event["payload"].get("procedure") == procedure
                and event["payload"].get("level") == method.get("level", "")
                and event["payload"].get("strength") == method.get("strength")]
            for method_event in reversed(method_events):
                judges = [event for event in events
                    if event["type"] == "judger_verdict"
                    and compiled_event["seq"] < event["seq"] < method_event["seq"]
                    and event["payload"].get("obligation") == oid
                    and event["payload"].get("procedure") == procedure
                    and all(event["payload"].get(key) == audit.get(key)
                            for key in ("sufficient", "faithful",
                                        "degradation", "reason"))]
                if not judges:
                    continue
                judge = judges[-1]
                enumerations = [event for event in events
                    if event["type"] == "methods_enumerated"
                    and compiled_event["seq"] < event["seq"] < judge["seq"]
                    and event["payload"].get("claim") == final_id
                    and event["payload"].get("obligation") == oid
                    and type(event["payload"].get("methods")) is list
                    and any(type(item) is dict
                            and item.get("procedure") == procedure
                            for item in event["payload"]["methods"])]
                if not enumerations:
                    continue
                enumeration = enumerations[-1]
                if isinstance(instrument, str) and instrument.startswith("adapter:"):
                    executions = [event for event in events
                        if event["type"] == "adapter_discharge"
                        and enumeration["seq"] < event["seq"] < judge["seq"]
                        and event["payload"].get("claim") == final_id
                        and event["payload"].get("obligation") == oid
                        and event["payload"].get("artifact") == discharge_digest
                        and event["payload"].get("adapter")
                        == instrument.removeprefix("adapter:")]
                    if not executions:
                        continue
                    execution_seq = executions[-1]["seq"]
                else:
                    # Agent execution has no ledger event of its own. Its exact
                    # content-addressed artifact and matching attempt/audit are
                    # the durable execution record preceding the Judger call.
                    execution_seq = None
                kernel_events = [event for event in events
                    if event["type"] == "discharge"
                    and method_event["seq"] < event["seq"] < certificate_seq
                    and event["payload"].get("claim") == final_id
                    and event["payload"].get("obligation") == oid
                    and event["payload"].get("artifact") == binding_digest
                    and event["payload"].get("evidence_id")
                    == evidence.get("evidence_id")
                    and all(event["payload"].get(key) == evidence.get(key)
                            for key in ("outcome", "basis", "strength"))]
                if not kernel_events:
                    continue
                accepted = {
                    "obligation": oid,
                    "status": certificate_item.get("status"),
                    "commit_seq": commits[0]["seq"],
                    "critic_pass_seq": critic_pass["seq"],
                    "compiled_seq": compiled_event["seq"],
                    "oracle_seq": enumeration["seq"],
                    "execution_kind": (
                        "adapter" if execution_seq is not None else "agent"),
                    "adapter_identity": (
                        instrument.removeprefix("adapter:")
                        if execution_seq is not None else None),
                    "execution_seq": execution_seq,
                    "execution_artifact": discharge_digest,
                    "judger_seq": judge["seq"],
                    "method_bound_seq": method_event["seq"],
                    "discharge_seq": kernel_events[0]["seq"],
                    "binding_artifact": binding_digest,
                    "certificate_seq": certificate_seq,
                    "stages_skipped_reason": None,
                }
                break
            if accepted is not None:
                break
        if accepted is None:
            if certificate_item.get("status") not in {"partial", "unverified"}:
                raise ValueError(
                    "final obligation lacks a fresh complete method lineage")
            enumerations = [event for event in events
                if event["type"] == "methods_enumerated"
                and compiled_event["seq"] < event["seq"] < certificate_seq
                and event["payload"].get("claim") == final_id
                and event["payload"].get("obligation") == oid]
            if not enumerations:
                raise ValueError("unsettled obligation has no Oracle residue")
            residue = None
            for evidence, binding_digest, binding in residues:
                attempts = binding.get("attempts")
                if (binding.get("obligation") != compiled_item.get("text")
                        or type(attempts) is not list):
                    continue
                kernel_events = [event for event in events
                    if event["type"] == "discharge"
                    and enumerations[0]["seq"] < event["seq"] < certificate_seq
                    and event["payload"].get("claim") == final_id
                    and event["payload"].get("obligation") == oid
                    and event["payload"].get("artifact") == binding_digest
                    and event["payload"].get("evidence_id")
                    == evidence.get("evidence_id")
                    and all(event["payload"].get(key) == evidence.get(key)
                            for key in ("outcome", "basis", "strength"))]
                if not kernel_events:
                    continue
                kernel_event = kernel_events[0]
                listed = [method for event in enumerations
                          for method in event["payload"].get("methods", [])
                          if type(method) is dict]
                executions, judges = [], []
                valid_attempts = True
                for attempt in attempts:
                    method = attempt.get("method") if type(attempt) is dict else None
                    if (type(method) is not dict
                            or not any(item.get("procedure")
                                       == method.get("procedure", "")[:80]
                                       for item in listed)):
                        valid_attempts = False
                        break
                    for digest in [attempt.get("discharge_artifact"), *[
                            redo.get("artifact") for redo in attempt.get("redos", [])
                            if type(redo) is dict]]:
                        if digest is not None:
                            _artifact(state, digest)
                            executions.append(digest)
                    for audit in [attempt.get("audit"), *[
                            redo.get("audit") for redo in attempt.get("redos", [])
                            if type(redo) is dict]]:
                        if type(audit) is not dict:
                            continue
                        matches = [event for event in events
                            if event["type"] == "judger_verdict"
                            and event["seq"] < kernel_event["seq"]
                            and event["payload"].get("obligation") == oid
                            and event["payload"].get("procedure")
                            == method.get("procedure", "")[:80]
                            and all(event["payload"].get(key) == audit.get(key)
                                    for key in ("sufficient", "faithful",
                                                "degradation", "reason"))]
                        if not matches:
                            valid_attempts = False
                            break
                        judges.append(matches[-1]["seq"])
                    if not valid_attempts:
                        break
                if not valid_attempts:
                    continue
                no_methods = not listed and not attempts
                residue = {
                    "obligation": oid,
                    "status": certificate_item.get("status"),
                    "commit_seq": commits[0]["seq"],
                    "critic_pass_seq": critic_pass["seq"],
                    "compiled_seq": compiled_event["seq"],
                    "oracle_seqs": [event["seq"] for event in enumerations],
                    "execution_artifacts": executions,
                    "judger_seqs": judges,
                    "discharge_seq": kernel_event["seq"],
                    "binding_artifact": binding_digest,
                    "certificate_seq": certificate_seq,
                    "stages_skipped_reason": (
                        "Oracle returned no executable method; executor, "
                        "Judger, and method_bound were skipped fail-closed"
                        if no_methods else
                        "no attempted method survived execution and fresh "
                        "Judger audit; method_bound was skipped fail-closed"),
                }
                break
            if residue is None:
                raise ValueError(
                    "unsettled obligation lacks a retained fail-closed residue")
            accepted = residue
        lineages.append(accepted)
    if any(item["discharge_seq"] >= certificate_seq for item in lineages):
        raise ValueError("certificate preceded a final obligation discharge")
    return lineages


_ONLINE_ROLES = frozenset({
    "obligations", "critic", "oracles", "discharge", "judger",
})
_ATTEMPT_COUNTER_FIELDS = (
    "input_tokens", "cached_input_tokens", "uncached_input_tokens",
    "cache_write_input_tokens", "output_tokens", "reasoning_tokens",
)


def _typed_transport_profile(modules, backend, arm: str) -> dict:
    normal = getattr(backend, "runtime_instruction_identity", None)
    if not isinstance(normal, str) or not _HASH.fullmatch(normal):
        raise ValueError("ordinary runtime instruction identity is unavailable")
    profile = {
        "enabled": arm == "treatment",
        "protocol": None,
        "boundary": None,
        "raw_prefetch_enabled": arm == "treatment",
        "raw_prefetch_protocol": (
            getattr(backend, "_judger_prefetch_protocol", None)
            if arm == "treatment" else None),
        "ordinary_runtime_instruction_identity": normal,
        "restricted_runtime_instruction_identity": None,
        "eligible_adapter_identities": [],
    }
    if arm == "control":
        return profile
    transport = modules["typed_judger_backend"]
    with transport.restricted_typed_judger_backend(backend) as restricted:
        if restricted is backend:
            raise ValueError("source lacks the restricted typed Judger seam")
        restricted_identity = getattr(
            restricted, "runtime_instruction_identity", None)
    if (not isinstance(restricted_identity, str)
            or not _HASH.fullmatch(restricted_identity)
            or restricted_identity == normal):
        raise ValueError("restricted typed Judger identity is unavailable")
    packet = modules["typed_judger_packet"]
    profile.update({
        "protocol": packet.TYPED_JUDGER_PACKET_PROTOCOL,
        "boundary": transport.TYPED_JUDGER_BOUNDARY_VERSION,
        "restricted_runtime_instruction_identity": restricted_identity,
        "eligible_adapter_identities": sorted(
            f"{name}@{version}" for name, version in packet.ELIGIBLE_ADAPTERS),
    })
    return profile


def _audit_execution(output: Path, state: Path, arm: str,
                     certificate: dict, transport_profile: dict) -> dict:
    roles, role_order, attempts = set(), [], []
    observer_rows = _read_jsonl(output / "observer.jsonl")
    for row in observer_rows:
        role = row.get("role")
        if not isinstance(role, str) or role not in _ONLINE_ROLES:
            raise ValueError("observer recorded an Extractor or unknown online role")
        if row.get("record_type") == "codex_logical_call_started":
            roles.add(role)
            role_order.append(role)
        elif row.get("record_type") == "codex_cli_attempt_finished":
            attempts.append(row)
    required = {"obligations", "critic", "oracles"}
    if not required <= roles:
        raise ValueError("complete online verifier role coverage was not observed")
    if (role_order.index("obligations") >= role_order.index("critic")
            or role_order.index("critic") >= role_order.index("oracles")):
        raise ValueError("online verifier roles began outside pipeline order")
    events, events_sha = _read_jsonl_identity(state / "events.jsonl")
    counts = {}
    for event in events:
        kind = event.get("type")
        if not isinstance(kind, str):
            raise ValueError("verification event has no type")
        counts[kind] = counts.get(kind, 0) + 1
    lineages = _audit_event_lineage(state, certificate, events)
    requires_judger = any(item.get("judger_seq") is not None
                          or item.get("judger_seqs") for item in lineages)
    requires_agent = any(item.get("execution_kind") == "agent"
                         or (arm == "control"
                             and item.get("execution_artifacts"))
                         for item in lineages)
    if ((requires_judger and "judger" not in roles)
            or (requires_agent and "discharge" not in roles)):
        raise ValueError("executed online stages are absent from the observer")
    if (("judger" in roles
         and role_order.index("oracles") >= role_order.index("judger"))
            or ("discharge" in roles
                and role_order.index("oracles") >= role_order.index("discharge"))):
        raise ValueError("online verifier roles began outside pipeline order")
    cache_events = {
        kind: counts.get(kind, 0) for kind in (
            "method_cache_hit", "method_cache_direction_audit",
            "contract_cache_hit", "contract_cache_published")
    }
    if any(cache_events.values()):
        raise ValueError("cold online job reused cached signed work")

    native_rows = _read_jsonl(output / "native-usage.jsonl")
    attached = raw_prefetch_attached = 0
    runtime_identities = set()
    if (type(transport_profile) is not dict
            or transport_profile.get("enabled") is not (arm == "treatment")):
        raise ValueError("typed Judger transport profile disagrees with arm")
    ordinary_identity = transport_profile.get(
        "ordinary_runtime_instruction_identity")
    restricted_identity = transport_profile.get(
        "restricted_runtime_instruction_identity")
    allowed_runtime_identities = {ordinary_identity}
    if arm == "treatment":
        if (not isinstance(transport_profile.get("protocol"), str)
                or not isinstance(transport_profile.get("boundary"), str)
                or transport_profile.get("raw_prefetch_enabled") is not True
                or not isinstance(
                    transport_profile.get("raw_prefetch_protocol"), str)
                or not isinstance(restricted_identity, str)):
            raise ValueError("treatment lacks a restricted transport identity")
        allowed_runtime_identities.add(restricted_identity)
    elif restricted_identity is not None:
        raise ValueError("control declared a restricted transport identity")
    for row in native_rows:
        required_profile = {
            "record_type", "backend", "model", "effort", "service_tier",
            "compact_inspection", "judger_prefetch",
            "judger_prefetch_protocol", "judger_prefetch_attached",
            "judger_prefetch_packet_sha256", "runtime_instruction_identity",
            "typed_judger_transport", "typed_judger_protocol",
            "typed_judger_packet_sha256",
            "thread_id", "role", *_ATTEMPT_COUNTER_FIELDS,
        }
        if not required_profile <= set(row):
            raise ValueError("native attempt omitted its runtime profile")
        if (row["record_type"] != "codex_cli_attempt"
                or row["backend"] != "codex-cli"
                or any(row[key] != value for key, value in RUNTIME.items())
                or row["compact_inspection"] is not (arm == "treatment")
                or type(row["typed_judger_transport"]) is not bool
                or not isinstance(row["runtime_instruction_identity"], str)
                or row["runtime_instruction_identity"]
                not in allowed_runtime_identities):
            raise ValueError("native attempt runtime profile changed")
        runtime_identities.add(row["runtime_instruction_identity"])
        if row["typed_judger_transport"]:
            packet_digest = row["typed_judger_packet_sha256"]
            if (row.get("role") != "judger" or arm != "treatment"
                    or row["typed_judger_protocol"]
                    != transport_profile.get("protocol")
                    or not isinstance(packet_digest, str)
                    or not _HASH.fullmatch(packet_digest)
                    or row["runtime_instruction_identity"]
                    != restricted_identity
                    or row["judger_prefetch"] is not False
                    or row["judger_prefetch_protocol"] is not None
                    or row["judger_prefetch_attached"] is not False
                    or row["judger_prefetch_packet_sha256"] is not None):
                raise ValueError("invalid explicit restricted Judger telemetry")
            attached += 1
        elif (row["typed_judger_protocol"] is not None
              or row["typed_judger_packet_sha256"] is not None
              or row["runtime_instruction_identity"] != ordinary_identity
              or row["judger_prefetch"] is not (arm == "treatment")
              or row["judger_prefetch_protocol"]
              != transport_profile.get("raw_prefetch_protocol")
              or type(row["judger_prefetch_attached"]) is not bool
              or (row["judger_prefetch_attached"]
                  and (row.get("role") != "judger"
                       or not isinstance(
                           row["judger_prefetch_packet_sha256"], str)
                       or not _HASH.fullmatch(
                           row["judger_prefetch_packet_sha256"])))
              or (not row["judger_prefetch_attached"]
                  and row["judger_prefetch_packet_sha256"] is not None)):
            raise ValueError("ordinary attempt carries restricted telemetry")
        if row["judger_prefetch_attached"]:
            raw_prefetch_attached += 1
    if not native_rows:
        raise ValueError("native runtime profile is unavailable")

    def validated_attempt_rows(rows, label):
        result = list(rows)
        attempt_ids = set()
        logical_attempts = set()
        threads = set()
        for row in result:
            attempt_id = row.get("attempt_id")
            call_id = row.get("call_id")
            attempt_index = row.get("attempt_index")
            thread = row.get("thread_id")
            logical_attempt = (call_id, attempt_index)
            if (not isinstance(attempt_id, str)
                    or not _ATTEMPT.fullmatch(attempt_id)
                    or attempt_id in attempt_ids
                    or not isinstance(call_id, str)
                    or not _ATTEMPT.fullmatch(call_id)
                    or type(attempt_index) is not int or attempt_index < 1
                    or logical_attempt in logical_attempts
                    or type(row.get("usage_complete")) is not bool
                    or row.get("usage_status") not in {"complete", "partial"}
                    or row["usage_complete"] is not (
                        row["usage_status"] == "complete")):
                raise ValueError(
                    f"{label} attempt identities are invalid or duplicate")
            attempt_ids.add(attempt_id)
            logical_attempts.add(logical_attempt)
            if thread is None:
                if row["usage_complete"] is not False:
                    raise ValueError(
                        f"{label} threadless attempt is not incomplete")
            elif (not isinstance(thread, str) or not thread
                  or thread in threads):
                raise ValueError(
                    f"{label} attempt thread identities are invalid or duplicate")
            else:
                threads.add(thread)
        return result

    observed_rows = validated_attempt_rows(attempts, "observer")
    native_attempt_rows = validated_attempt_rows(native_rows, "native")
    if len(observed_rows) != len(native_attempt_rows):
        raise ValueError("observer and native attempt inventories disagree")
    for native, observed in zip(native_attempt_rows, observed_rows):
        profile_fields = (
            "attempt_index", "usage_complete", "usage_status", "role",
            "model", "effort", "service_tier",
            "runtime_instruction_identity",
        )
        same_thread = (
            native.get("thread_id") == observed.get("thread_id")
            and isinstance(native.get("thread_id"), str)
            and bool(native["thread_id"])
        )
        if not same_thread:
            same_timeout = (
                not (native.get("thread_id") is not None
                     and observed.get("thread_id") is not None)
                and native.get("usage_complete") is False
                and observed.get("usage_complete") is False
                and native.get("usage_status") == "partial"
                and observed.get("usage_status") == "partial"
                and native.get("attempt_status") == "timeout"
                and observed.get("terminal_status") == "timeout"
            )
            for key in ("started_at_unix_ns", "finished_at_unix_ns",
                        "duration_ns"):
                left, right = native.get(key), observed.get(key)
                same_timeout = same_timeout and type(left) is int \
                    and type(right) is int and abs(left - right) <= 50_000_000
            if not same_timeout:
                raise ValueError(
                    "observer and native attempt thread identities disagree")
        if any(native.get(key) != observed.get(key) for key in profile_fields):
            raise ValueError("observer and native attempt profiles disagree")
        expected_cwd = (
            str(output / "typed-judger-scratch")
            if native["typed_judger_transport"] else str(state))
        if observed.get("cwd") != expected_cwd:
            raise ValueError("observer and native execution cwd disagree")
        if any(observed.get("command_" + key) != native.get(key)
               for key in ("model", "effort", "service_tier")):
            raise ValueError("native attempt disagrees with actual CLI profile")
        if any(type(native.get(key)) is not type(observed.get(key))
               or native.get(key) != observed.get(key)
               for key in (*_ATTEMPT_COUNTER_FIELDS,
                           "typed_judger_transport", "typed_judger_protocol",
                           "typed_judger_packet_sha256")):
            raise ValueError("observer and native token counters disagree")

    adapter_artifacts = _bound_adapter_digests(state, certificate)
    adapter_events = counts.get("adapter_discharge", 0)
    if arm == "control" and (adapter_events or adapter_artifacts):
        raise ValueError("control unexpectedly used a typed adapter")
    eligible = set(transport_profile.get("eligible_adapter_identities", []))
    eligible_bound = [item["adapter_identity"] for item in lineages
                      if item.get("adapter_identity") in eligible]
    ordinary_judger_attempts = sum(
        row.get("role") == "judger"
        and row.get("typed_judger_transport") is False
        for row in native_rows)
    transport_failures = [event for event in events
                          if event["type"] == "judger_call_failed"
                          and event["payload"].get("will_retry") is True]
    return {
        "roles_observed": sorted(roles), "event_counts": counts,
        "events_sha256": events_sha, "events_rows": len(events),
        "final_obligation_lineage": lineages,
        "cache_events": cache_events, "native_attempts": len(native_rows),
        "observer_finished_attempts": len(attempts),
        "runtime_instruction_identities": sorted(runtime_identities),
        "typed_judger_transport": {
            "enabled": arm == "treatment",
            "protocol": transport_profile.get("protocol"),
            "boundary": transport_profile.get("boundary"),
            "packet_eligible_bound_methods": len(eligible_bound),
            # The exact restricted backend refuses calls without a trusted
            # typed packet, so its runtime identity is attachment telemetry.
            "packet_attached_attempts": attached,
            "restricted_attempted": attached > 0,
            "ordinary_judger_attempts": ordinary_judger_attempts,
            "raw_prefetch_attached_attempts": raw_prefetch_attached,
            "restricted_fallback_attempted": bool(
                attached and ordinary_judger_attempts and transport_failures),
            "packet_unavailable_fallback": bool(
                eligible_bound and not attached and ordinary_judger_attempts),
        },
        "adapter_discharge_events": adapter_events,
        "bound_adapter_artifacts": adapter_artifacts,
        # Exercise and final acceptance are deliberately separate. A typed
        # method rejected by the fresh Judger still ran and must retain its
        # cost, but it is not evidence that the optimization settled a result.
        "typed_adapter_exercised": bool(adapter_events),
        "typed_adapter_bound": bool(adapter_artifacts),
    }


def run_worker(plan):
    """Run once; paid authorization is owned exclusively by :func:`main`."""
    validate_plan(plan)
    _environment(plan)
    source, inputs, output = (Path(plan[key]) for key in (
        "source_root", "input_source", "output_root"))
    source_tools = _support("run_typed_adapter_eval")
    if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
        raise ValueError("frozen source fingerprint mismatch")
    observer = _support("observe_codex_attempts")
    analyzer = _support("analyze_codex_observer")
    projection = _support("project_full_offline_result")
    policy = _support("eval_offline_policy")
    modules = _load_runtime(source)
    backend_module = modules["backend"]
    if not hasattr(backend_module, "_run_codex_process"):
        raise ValueError("source lacks the required observer seam")

    output.mkdir(parents=True, exist_ok=False)
    output.chmod(0o700)
    state = output / "state"
    restricted_judger_cwd = None
    if plan["arm"] == "treatment":
        scratch = output / "typed-judger-scratch"
        os.mkdir(scratch, 0o500)
        restricted_judger_cwd = str(scratch)
    worker_id = uuid.uuid4().hex
    started_unix, started_mono = time.time_ns(), time.monotonic_ns()
    context = {str(state): {"arm": plan["arm"], "worker_id": worker_id,
                            "scope": "full-online-cold"}}
    start = {
        "schema": RESULT_SCHEMA, "status": "started", "plan": plan,
        "worker_id": worker_id, "runtime": RUNTIME,
        "started_at_unix_ns": started_unix,
        "scope": "full-online-service-commit-fresh-cold-arbitrary-tree",
        "live_world": True, "diagnostic_only": True,
        "automatic_outer_retries": 0, "contract_cache": False,
        "accepted_method_reuse": False,
        "input_contract": "bounded-sealed-arbitrary-tree-v2",
        "hook_feed_ingest": "production-Service.drain_hooks",
        "browser_runtime_pins": {
            "supported": False,
            "reason": "no source-pinned coordinator schema",
        },
    }
    _write_new(output / "worker-start.json", start)
    observation = overlay = chain = accounting = execution_audit = None
    transport_profile = None
    pre_inputs = post_inputs = artifacts = None
    failures, caught = [], None
    try:
        pre_inputs = _clone_inputs(inputs, state, plan["input_seal"])
        backend = policy.codex_class(backend_module)(
            cwd=str(state), **RUNTIME,
            compact_inspection=plan["arm"] == "treatment",
            judger_prefetch=plan["arm"] == "treatment",
            restricted_judger_cwd=restricted_judger_cwd,
        )
        transport_profile = _typed_transport_profile(
            modules, backend, plan["arm"])
        binding = policy.OfflinePolicyBinding(
            state_dir=str(state),
            runtime_read_roots=tuple(plan["runtime_read_roots"]),
        )
        with policy.eval_online_policy(
                backend_module, bindings=(binding,)) as overlay:
            with observer.observe_codex_attempts(
                    backend_module, output / "observer.jsonl",
                    seam=observer.RUN_CODEX_PROCESS_SEAM,
                    context_by_cwd=context) as observation:
                with backend_module.codex_usage_ledger(
                        str(output / "native-usage.jsonl"),
                        context_by_cwd=context):
                    chain = _execute_live_chain(
                        modules, state, backend, projection,
                        plan["source_fingerprint"])
        execution_audit = _audit_execution(
            output, state, plan["arm"], chain["certificate"],
            transport_profile)
        projection_identity = _verify_projection_identity(state, chain)
        if any(execution_audit[key] != projection_identity[key]
               for key in ("events_sha256", "events_rows")):
            raise ValueError("execution audit and projection events disagree")
        _verify_semantic_evidence(state, chain)
        artifacts = _artifact_inventory(state)
    except BaseException as error:
        caught = error
        failures.append(type(error).__name__)
    finally:
        if chain is not None:
            try:
                _verify_projection_identity(state, chain)
                _verify_semantic_evidence(state, chain)
            except BaseException:
                failures.append("QualityProjectionChanged")
        try:
            if observation is None:
                failures.append("ObserverUnavailable")
            else:
                accounting = analyzer.analyze_file(
                    output / "observer.jsonl", context_by_cwd=context,
                    **RUNTIME, session_summary=observation.summary(),
                    execution_cwds_by_state={str(state): (
                        (str(state), restricted_judger_cwd)
                        if restricted_judger_cwd is not None
                        else (str(state),))})
                if accounting.get("accounting_complete") is not True:
                    failures.append("AccountingIncomplete")
            post_inputs = input_snapshot(state, runtime_state=True)
            if (pre_inputs is None or pre_inputs["seal"] != plan["input_seal"]
                    or post_inputs["seal"] != plan["input_seal"]
                    or input_snapshot(inputs)["seal"] != plan["input_seal"]):
                failures.append("InputBytesChanged")
            if source_tools._source_fingerprint(source) != plan["source_fingerprint"]:
                failures.append("SourceChanged")
            if any(_sha(Path(__file__).with_name(name)) != digest
                   for name, digest in plan["support_sha256"].items()):
                failures.append("EvaluationSupportChanged")
        except BaseException as error:
            failures.append("Postflight" + type(error).__name__)
        result = {
            **start, "status": "terminal", "complete": not failures,
            "failures": list(dict.fromkeys(failures)),
            "chain": chain, "accounting": accounting,
            "execution_audit": execution_audit,
            "observer_summary": (observation.summary()
                                 if observation is not None else None),
            "online_policy_identity": (overlay.identity
                                       if overlay is not None else None),
            "input_before": pre_inputs, "input_after": post_inputs,
            "artifact_inventory": artifacts,
            "truth_eligible": (
                chain["quality_view"]["semantic_evidence"]["document"]
                ["truth_eligible"] if chain is not None else False),
            "finished_at_unix_ns": time.time_ns(),
            "worker_wall_seconds": (time.monotonic_ns() - started_mono) / 1e9,
            "quality_acceptance": None, "full_e2e_acceptance": None,
            "general_no_loss_claim": False, "general_10x_claim": False,
        }
        _write_new(output / "worker-result.json", result)
    if caught is not None:
        raise caught
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--plan-sha256", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--confirm-paid")
    args = parser.parse_args(argv)
    plan, digest = _read_json(args.plan)
    if digest != args.plan_sha256:
        parser.error("operator plan digest mismatch")
    validate_plan(plan)
    if args.preflight:
        if args.execute:
            parser.error("preflight and execute are mutually exclusive")
        print(json.dumps(preflight(plan), sort_keys=True))
        return 0
    if not args.execute:
        print(json.dumps({"plan": plan, "model_calls": 0}, sort_keys=True))
        return 0
    if args.confirm_paid != "GPT-5.6-Luna":
        parser.error("execution requires --confirm-paid GPT-5.6-Luna")
    result = run_worker(plan)
    print(json.dumps({"complete": result["complete"],
                      "worker_wall_seconds": result["worker_wall_seconds"]}))
    return 0 if result["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
