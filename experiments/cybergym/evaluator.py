"""Host-owned CyberGym online feedback adapter; it never issues a verdict.

Usage: python evaluator.py PRIVATE_CONFIG CANDIDATE
Only /submit-vul is reachable through this adapter. Hidden patched evaluation
belongs to the experiment controller after the system chooses its final input.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import os
from pathlib import Path
import secrets
import selectors
import stat
import subprocess
import sys
import tempfile
import time
import re
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

CONFIG_SCHEMA = "gv-cybergym-vul-evaluator-v1"
OBSERVATION_SCHEMA = "gv-cybergym-vul-observation-v1"
MAX_CONFIG_BYTES = 64 * 1024
MAX_CANDIDATE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
BASH_EXECUTABLE = "/usr/bin/bash"
CURL_EXECUTABLE = "/usr/bin/curl"
MAX_TOOL_BYTES = 64 * 1024 * 1024
# Exact upstream submit.template at the pinned official revision. No trailing
# newline is present upstream. Keeping this public template here avoids an
# ambient template dependency or trusting a host-configured arbitrary script.
SUBMISSION_TEMPLATE = r'''#!/bin/bash

# Display help message if no arguments provided or help flag used
if [ "$1" == "-h" ] || [ "$1" == "--help" ] || [ -z "$1" ]; then
    echo "Usage: $0 <poc_file_path>"
    echo ""
    echo "Submit a vulnerability proof-of-concept file to the CyberGym server"
    echo ""
    echo "Arguments:"
    echo "  <poc_file_path>    Path to the proof-of-concept file to submit"
    echo ""
    echo "Examples:"
    echo "  $0 /path/to/poc"
    exit 0
fi

POC_FILE="$1"

# Check if file exists
if [ ! -f "$POC_FILE" ]; then
    echo "Error: File '$POC_FILE' not found!"
    exit 1
fi

curl -X POST ##SERVER##/submit-vul \
  -F 'metadata={"task_id": "##TASK_ID##", "agent_id": "##AGENT_ID##", "checksum": "##CHECKSUM##", "require_flag": ##REQUIRE_FLAG##}' \
  -F "file=@${POC_FILE}"'''


class AdapterError(ValueError):
    """A static error code: never include private configuration or HTTP bodies."""

    def __init__(self, code: str, *, child_execution=None):
        super().__init__(code)
        self.child_execution = child_execution


def _json_object(raw: bytes, code: str) -> dict[str, Any]:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise AdapterError(code)
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                           parse_constant=lambda _: (_ for _ in ()).throw(AdapterError(code)))
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as exc:
        raise AdapterError(code) from exc
    if not isinstance(value, dict):
        raise AdapterError(code)
    return value


def _file_bytes(path: str, limit: int, code: str) -> bytes:
    try:
        fd = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        try:
            before = os.fstat(fd)
            if not stat.S_ISREG(before.st_mode) or before.st_size > limit:
                raise AdapterError(code)
            chunks, size = [], 0
            while chunk := os.read(fd, min(65536, limit + 1 - size)):
                size += len(chunk)
                if size > limit:
                    raise AdapterError(code)
                chunks.append(chunk)
            after = os.fstat(fd)
            identity = lambda entry: (entry.st_dev, entry.st_ino, entry.st_size,
                                       entry.st_mtime_ns, entry.st_ctime_ns)
            if identity(before) != identity(after):
                raise AdapterError(code)
            return b"".join(chunks)
        finally:
            os.close(fd)
    except OSError as exc:
        raise AdapterError(code) from exc


def load_config(path: str) -> tuple[dict[str, Any], str]:
    raw = _file_bytes(path, MAX_CONFIG_BYTES, "invalid_configuration")
    config = _json_object(raw, "invalid_configuration")
    required = {"schema", "server_url", "task_id", "agent_id", "checksum", "require_flag"}
    if (not required <= config.keys()
            or config.keys() - required - {"observations_dir", "submission_script"}
            or config["schema"] != CONFIG_SCHEMA
            or type(config["require_flag"]) is not bool):
        raise AdapterError("invalid_configuration")
    for key in ("server_url", "task_id", "agent_id", "checksum"):
        value = config[key]
        if not isinstance(value, str) or not value or len(value) > 4096 or "\x00" in value:
            raise AdapterError("invalid_configuration")
    try:
        origin = urllib.parse.urlsplit(config["server_url"])
        port = origin.port
    except ValueError as exc:
        raise AdapterError("invalid_configuration") from exc
    if (any(ord(char) < 32 or ord(char) == 127 for char in config["server_url"])
            or origin.scheme not in ("http", "https") or not origin.hostname
            or origin.username is not None or origin.password is not None
            or origin.path not in ("", "/") or origin.query or origin.fragment
            or (port is not None and not 0 < port <= 65535)):
        raise AdapterError("invalid_configuration")
    if "observations_dir" in config:
        directory = config["observations_dir"]
        if (not isinstance(directory, str) or not os.path.isabs(directory)
                or "\x00" in directory):
            raise AdapterError("invalid_configuration")
    if "submission_script" in config:
        script = config["submission_script"]
        if (type(script) is not str or not os.path.isabs(script) or "\x00" in script
                or os.path.normpath(script) != script or Path(script).name != "submit.sh"):
            raise AdapterError("invalid_submission_configuration")
    return config, hashlib.sha256(raw).hexdigest()


def render_submission_script(config: dict[str, Any]) -> bytes:
    """Exactly upstream rendering, but reject its unquoted shell/form hazards.

    These additional checks apply ONLY to script mode. The default HTTP ABI
    retains its original configuration grammar and multipart behavior.
    """
    if (not re.fullmatch(r'https?://(?:[A-Za-z0-9.-]+|\[[0-9A-Fa-f:.]+\])(?::[0-9]+)?', config["server_url"])
            or any(not re.fullmatch(r'[A-Za-z0-9_.:-]+', config[key])
                   for key in ("task_id", "agent_id", "checksum"))):
        raise AdapterError("invalid_submission_configuration")
    rendered = SUBMISSION_TEMPLATE
    for marker, value in (("TASK_ID", config["task_id"]), ("AGENT_ID", config["agent_id"]),
                          ("CHECKSUM", config["checksum"]), ("SERVER", config["server_url"]),
                          ("REQUIRE_FLAG", str(config["require_flag"]).lower())):
        rendered = rendered.replace("##" + marker + "##", value)
    return rendered.encode("ascii")


def _file_identity(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns,
            info.st_ctime_ns, info.st_mode, info.st_uid, info.st_nlink)


def _pinned_bytes(path: str, limit: int, code: str, *, readonly=False, readonly_parent=False, executable=False):
    """Nofollow all ancestors and bind bytes plus the actual directory entry.

    Read-only private script storage and generic evaluator dependency pins
    exclude worker writes. This does not claim to control a hostile host with
    the operator's uid or a root process modifying the executable filesystem.
    """
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise AdapterError(code)
    parent = fd = None
    try:
        parent = os.open("/", os.O_RDONLY | os.O_DIRECTORY)
        for part in Path(path).parts[1:-1]:
            following = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=parent)
            os.close(parent)
            parent = following
            info = os.fstat(parent)
            if info.st_uid not in (0, os.geteuid()) or (info.st_mode & 0o022 and not (info.st_uid == 0 and info.st_mode & stat.S_ISVTX)):
                raise AdapterError(code)
        owner = os.fstat(parent)
        if readonly_parent and owner.st_mode & 0o222:
            raise AdapterError(code)
        fd = os.open(Path(path).name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW, dir_fd=parent)
        before = os.fstat(fd)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1 or before.st_uid not in (0, os.geteuid())
                or before.st_size > limit or before.st_mode & 0o022
                or readonly and before.st_mode & 0o222 or executable and not before.st_mode & 0o111):
            raise AdapterError(code)
        raw = bytearray()
        while len(raw) < before.st_size:
            chunk = os.read(fd, min(65536, before.st_size - len(raw)))
            if not chunk:
                break
            raw.extend(chunk)
        current = os.stat(Path(path).name, dir_fd=parent, follow_symlinks=False)
        if len(raw) != before.st_size or _file_identity(before) != _file_identity(os.fstat(fd)) or _file_identity(before) != _file_identity(current):
            raise AdapterError(code)
        return bytes(raw), _file_identity(before)
    except OSError as exc:
        raise AdapterError(code) from exc
    finally:
        if fd is not None:
            os.close(fd)
        if parent is not None:
            os.close(parent)


def check_submission_configuration(config):
    """Pure readiness with local pinned-file reads; never submits a candidate."""
    script = config["submission_script"]
    expected = render_submission_script(config)
    raw, identity = _pinned_bytes(script, MAX_CONFIG_BYTES, "unsafe_submission_script", readonly=True, readonly_parent=True)
    if raw != expected:
        raise AdapterError("submission_script_mismatch")
    dependencies = {}
    for name, path in (("bash", BASH_EXECUTABLE), ("curl", CURL_EXECUTABLE)):
        value, fingerprint = _pinned_bytes(path, MAX_TOOL_BYTES, "submission_tool_unavailable", executable=True)
        dependencies[name] = {"path": path, "sha256": hashlib.sha256(value).hexdigest(),
                              "size": len(value), "identity": fingerprint}
    return {"script": {"path": script, "sha256": hashlib.sha256(raw).hexdigest(),
                       "size": len(raw), "identity": identity}, "tools": dependencies}


def _private_file(path, raw):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    try:
        view = memoryview(raw)
        while view:
            count = os.write(fd, view)
            if count <= 0:
                raise AdapterError("submission_storage_error")
            view = view[count:]
        os.fchmod(fd, 0o400)
    finally:
        os.close(fd)


def _child_output(argv, cwd, environment, limit, record):
    """No time deadline. Bounded stdout, streamed stderr hashes, no raw errors.

    Oversized stdout is closed (curl observes EPIPE); only this adapter's pipe
    is affected. There is no timeout, process-group signal or agent lifecycle
    action. A failed command is not a vulnerability/target verdict.
    """
    process = None
    captured, stdout_hash, stderr_hash = bytearray(), hashlib.sha256(), hashlib.sha256()
    counts, truncated = {"stdout": 0, "stderr": 0}, False
    eof = set()
    started = time.monotonic()
    record["started_unix"] = time.time()
    record["start_source"] = "host_pre_spawn_clock"
    record["child_started"] = False
    record["execution_state"] = "spawn_attempted"
    try:
        process = subprocess.Popen(argv, cwd=cwd, env=environment, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        record["child_pid"] = process.pid
        record.update(child_started=True, execution_state="process_started", spawned_unix=time.time())
        with selectors.DefaultSelector() as selector:
            for name, stream in (("stdout", process.stdout), ("stderr", process.stderr)):
                os.set_blocking(stream.fileno(), False)
                selector.register(stream, selectors.EVENT_READ, name)
            while selector.get_map():
                for key, _events in selector.select():
                    name, stream = key.data, key.fileobj
                    maximum = min(65536, max(1, limit + 1 - len(captured))) if name == "stdout" else 65536
                    chunk = os.read(stream.fileno(), maximum)
                    if not chunk:
                        eof.add(name)
                        selector.unregister(stream)
                        stream.close()
                        continue
                    counts[name] += len(chunk)
                    (stdout_hash if name == "stdout" else stderr_hash).update(chunk)
                    if name == "stdout":
                        captured.extend(chunk)
                        if len(captured) > limit:
                            truncated = True
                            selector.unregister(stream)
                            stream.close()
        record["exit_code"] = process.wait()  # Deliberately no timeout.
        record["execution_state"] = "process_completed"
    except OSError as exc:
        record["error_code"] = "submission_execution_error"
        record["execution_state"] = "spawn_failed" if process is None else "process_io_error"
        raise AdapterError("submission_execution_error", child_execution=record) from exc
    finally:
        if process is not None:
            for stream in (process.stdout, process.stderr):
                if not stream.closed:
                    stream.close()
            if process.poll() is None:
                record["exit_code"] = process.wait()
        record.update(completed_unix=time.time(), duration_seconds=max(0, time.monotonic() - started),
                      stdout_size=counts["stdout"], stdout_sha256=stdout_hash.hexdigest(),
                      stdout_capture_complete="stdout" in eof and not truncated, stderr_size=counts["stderr"],
                      stderr_sha256=stderr_hash.hexdigest(), stderr_capture_complete="stderr" in eof)
    if truncated:
        raise AdapterError("response_too_large", child_execution=record)
    return bytes(captured)


def _script_response(config, candidate_path, candidate):
    pins = check_submission_configuration(config)
    source, source_identity = _pinned_bytes(candidate_path, MAX_CANDIDATE_BYTES, "invalid_candidate", readonly=True)
    if source != candidate:
        raise AdapterError("candidate_changed")
    digest = hashlib.sha256(candidate).hexdigest()
    record = {"schema": "gv-cybergym-submission-execution-v1", "executor": "host_adapter",
              "delegation": "official_submission_script", "solver_executed": False,
              "source_candidate_sha256": digest, "source_candidate_size": len(candidate),
              "script_sha256": pins["script"]["sha256"],
              "tool_sha256": {name: value["sha256"] for name, value in pins["tools"].items()},
              "transport_hardening": {"inherited_environment": False, "proxy": "disabled",
                  "redirects": "disabled", "protocols": ["http", "https"], "http_status_required": 200,
                  "curl_config": "operator_private", "response_body_limit": MAX_RESPONSE_BYTES}}
    try:
        # Explicit /tmp ignores inherited TMPDIR. Its private resolved path and
        # fixed candidate basename exclude curl -F comma/semicolon grammar.
        with tempfile.TemporaryDirectory(prefix="gv-cybergym-submit-", dir="/tmp") as temporary:
            home = str(Path(temporary).resolve())
            if not re.fullmatch(r'[A-Za-z0-9_./-]+', home):
                raise AdapterError("unsafe_submission_storage")
            copied = os.path.join(home, "candidate.bin")
            _private_file(copied, candidate)
            copied_bytes, copied_identity = _pinned_bytes(copied, MAX_CANDIDATE_BYTES, "submission_candidate_copy_changed", readonly=True)
            if copied_bytes != candidate:
                raise AdapterError("submission_candidate_copy_changed")
            marker = "GVHTTP" + secrets.token_hex(16)
            trailer = ("\n" + marker).encode("ascii")
            curl_config = ('proxy = ""\nnoproxy = "*"\nno-location\nmax-redirs = 0\n'
                           'proto = "=http,https"\nproto-redir = "=http,https"\n'
                           'write-out = "\\n' + marker + '%{http_code}"\n').encode("ascii")
            _private_file(os.path.join(home, ".curlrc"), curl_config)
            argv = [BASH_EXECUTABLE, "./submit.sh", copied]
            cwd = str(Path(config["submission_script"]).parent)
            record.update(argv=argv, cwd=cwd, candidate_copy_sha256=digest,
                          candidate_copy_size=len(candidate), candidate_copy_readonly=True)
            environment = {"PATH": "/usr/bin:/bin", "HOME": home, "CURL_HOME": home, "LC_ALL": "C"}
            output = _child_output(argv, cwd, environment, MAX_RESPONSE_BYTES + len(trailer) + 3, record)
            current_copy, current_identity = _pinned_bytes(copied, MAX_CANDIDATE_BYTES, "submission_candidate_copy_changed", readonly=True)
            if current_copy != candidate or current_identity != copied_identity:
                raise AdapterError("submission_candidate_copy_changed")
            if record["exit_code"] != 0:
                raise AdapterError("transport_error")
            raw, separator, status = output.rpartition(trailer)
            if not separator or not re.fullmatch(rb'[0-9]{3}', status):
                raise AdapterError("invalid_submission_transport_record")
            record["http_status"] = int(status)
            record.update(response_sha256=hashlib.sha256(raw).hexdigest(), response_size=len(raw))
            if 300 <= record["http_status"] <= 399:
                raise AdapterError("unexpected_redirect")
            if record["http_status"] != 200:
                raise AdapterError("unexpected_http_status")
            if len(raw) > MAX_RESPONSE_BYTES:
                raise AdapterError("response_too_large")
        return raw, record
    except AdapterError as exc:
        record["error_code"] = str(exc)
        exc.child_execution = record
        raise
    except OSError as exc:
        record["error_code"] = "submission_storage_error"
        raise AdapterError("submission_storage_error", child_execution=record) from exc
    finally:
        # Candidate/script/tool pins are checked after success AND failure.
        try:
            current, identity = _pinned_bytes(candidate_path, MAX_CANDIDATE_BYTES, "candidate_changed", readonly=True)
            if current != candidate or identity != source_identity:
                raise AdapterError("candidate_changed")
            if check_submission_configuration(config) != pins:
                raise AdapterError("submission_dependency_changed")
        except AdapterError as exc:
            record["error_code"] = str(exc)
            exc.child_execution = record
            raise


def _multipart(config: dict[str, Any], candidate: bytes) -> tuple[bytes, str]:
    boundary = "gv-cybergym-" + secrets.token_hex(24)
    marker = boundary.encode("ascii")
    while marker in candidate:
        boundary = "gv-cybergym-" + secrets.token_hex(24)
        marker = boundary.encode("ascii")
    metadata = json.dumps({key: config[key] for key in
                           ("task_id", "agent_id", "checksum", "require_flag")},
                          separators=(",", ":")).encode("utf-8")
    body = (b"--" + marker + b'\r\nContent-Disposition: form-data; name="metadata"\r\n\r\n'
            + metadata + b"\r\n--" + marker
            + b'\r\nContent-Disposition: form-data; name="file"; filename="candidate.bin"\r\n'
            + b"Content-Type: application/octet-stream\r\n\r\n" + candidate
            + b"\r\n--" + marker + b"--\r\n")
    return body, "multipart/form-data; boundary=" + boundary


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise AdapterError("unexpected_redirect")


def _validate_response(raw: bytes, config: dict[str, Any]) -> dict[str, Any]:
    response = _json_object(raw, "invalid_response")
    required = {"task_id", "exit_code", "output", "poc_id"}
    if (not required <= response.keys() or response.keys() - required - {"flag"}
            or response["task_id"] != config["task_id"]
            or type(response["exit_code"]) is not int
            or not isinstance(response["output"], str)
            or not isinstance(response["poc_id"], str) or not response["poc_id"]
            or len(response["poc_id"]) > 4096 or "\x00" in response["poc_id"]
            or ("flag" in response and
                (not config["require_flag"] or not isinstance(response["flag"], str)))):
        raise AdapterError("invalid_response")
    return response


def _retain_response(directory: str, raw: bytes) -> dict[str, Any]:
    """Retain complete evidence when the generic tool's stdout cap truncates it."""
    digest = hashlib.sha256(raw).hexdigest()
    # This directory is host configured and private; do not follow substituted
    # ancestor symlinks even when the final directory itself looks ordinary.
    directory = os.path.normpath(directory)
    fd = None
    temporary = None
    try:
        fd = os.open(os.path.sep, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        for part in Path(directory).parts[1:]:
            next_fd = os.open(part, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
                              | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
            os.close(fd)
            fd = next_fd
        name = digest + ".vul-response.json"
        temporary = ".response-" + secrets.token_hex(16)
        output = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                         | getattr(os, "O_NOFOLLOW", 0), 0o400, dir_fd=fd)
        try:
            view = memoryview(raw)
            while view:
                view = view[os.write(output, view):]
            os.fsync(output)
        finally:
            os.close(output)
        try:
            os.link(temporary, name, src_dir_fd=fd, dst_dir_fd=fd, follow_symlinks=False)
        except FileExistsError:
            existing = os.open(name, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0), dir_fd=fd)
            try:
                before = os.fstat(existing)
                if not stat.S_ISREG(before.st_mode) or before.st_size != len(raw):
                    raise AdapterError("evidence_storage_error")
                chunks = []
                while chunk := os.read(existing, 65536):
                    chunks.append(chunk)
                after = os.fstat(existing)
                if (b"".join(chunks) != raw or before.st_mtime_ns != after.st_mtime_ns
                        or before.st_ctime_ns != after.st_ctime_ns):
                    raise AdapterError("evidence_storage_error")
            finally:
                os.close(existing)
        os.fsync(fd)
    except OSError as exc:
        raise AdapterError("evidence_storage_error") from exc
    finally:
        if fd is not None:
            try:
                if temporary is not None:
                    try:
                        os.unlink(temporary, dir_fd=fd)
                    except FileNotFoundError:
                        pass
            finally:
                os.close(fd)
    return {"path": os.path.join(directory, name), "sha256": digest, "size": len(raw)}


def evaluate(config_path: str, candidate_path: str) -> dict[str, Any]:
    config, config_digest = load_config(config_path)
    candidate = _file_bytes(candidate_path, MAX_CANDIDATE_BYTES, "invalid_candidate")
    child_execution = None
    if "submission_script" in config:
        raw, child_execution = _script_response(config, candidate_path, candidate)
    else:
        body, content_type = _multipart(config, candidate)
        request = urllib.request.Request(config["server_url"].rstrip("/") + "/submit-vul",
                                         data=body, headers={"Content-Type": content_type},
                                         method="POST")
        # Neither inherited proxy credentials nor server redirects can broaden
        # the endpoint. None explicitly disables inherited socket-level timeouts.
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(request, timeout=None) as result:
                if result.status != 200:
                    raise AdapterError("unexpected_http_status")
                raw = result.read(MAX_RESPONSE_BYTES + 1)
        except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
            raise AdapterError("transport_error") from exc
        if len(raw) > MAX_RESPONSE_BYTES:
            raise AdapterError("response_too_large")
    try:
        response = _validate_response(raw, config)
        retained = (_retain_response(config["observations_dir"], raw)
                    if "observations_dir" in config else None)
    except AdapterError as exc:
        if child_execution is not None:
            child_execution["error_code"] = str(exc)
            exc.child_execution = child_execution
        raise
    # Metadata goes before the full response so a bounded tool excerpt still
    # points the verifier at complete authenticated raw evidence. No server URL,
    # checksum, config contents, or private scoring authority is emitted.
    observation = {"schema": OBSERVATION_SCHEMA,
                   "candidate_sha256": hashlib.sha256(candidate).hexdigest(),
                   "candidate_size": len(candidate),
                   "environment": "vulnerable", "authority_scope": "candidate_signal",
                   "task_id": config["task_id"], "config_sha256": config_digest,
                   "response_sha256": hashlib.sha256(raw).hexdigest(),
                   "response_size": len(raw), "response_truncated": False}
    if retained is not None:
        observation["response_artifact"] = retained
    if child_execution is not None:
        observation["child_execution"] = child_execution
    observation["response"] = response
    return observation


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "--check-config":
        try:
            config, _digest = load_config(argv[1])
            if "submission_script" in config:
                check_submission_configuration(config)
        except AdapterError as exc:
            print(json.dumps({"schema": OBSERVATION_SCHEMA, "error": str(exc)}))
            return 1
        print(json.dumps({"schema": OBSERVATION_SCHEMA, "configuration_valid": True}))
        return 0
    if len(argv) != 2:
        print(json.dumps({"schema": OBSERVATION_SCHEMA, "error": "invalid_arguments"}))
        return 2
    try:
        observation = evaluate(argv[0], argv[1])
    except AdapterError as exc:
        print(json.dumps({"schema": OBSERVATION_SCHEMA, "error": str(exc),
                          **({"response_truncated": True} if str(exc) == "response_too_large" else {}),
                          **({"child_execution": exc.child_execution} if exc.child_execution is not None else {})}))
        return 1
    print(json.dumps(observation, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
