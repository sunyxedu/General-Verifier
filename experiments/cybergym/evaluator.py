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
import stat
import sys
from typing import Any
import urllib.error
import urllib.parse
import urllib.request

CONFIG_SCHEMA = "gv-cybergym-vul-evaluator-v1"
OBSERVATION_SCHEMA = "gv-cybergym-vul-observation-v1"
MAX_CONFIG_BYTES = 64 * 1024
MAX_CANDIDATE_BYTES = 10 * 1024 * 1024
MAX_RESPONSE_BYTES = 8 * 1024 * 1024


class AdapterError(ValueError):
    """A static error code: never include private configuration or HTTP bodies."""


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
            or config.keys() - required - {"observations_dir"}
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
    return config, hashlib.sha256(raw).hexdigest()


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
    body, content_type = _multipart(config, candidate)
    request = urllib.request.Request(config["server_url"].rstrip("/") + "/submit-vul",
                                     data=body, headers={"Content-Type": content_type},
                                     method="POST")
    # Neither inherited proxy credentials nor server redirects can broaden the
    # endpoint. None explicitly disables inherited socket-level timeouts.
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
    response = _validate_response(raw, config)
    retained = (_retain_response(config["observations_dir"], raw)
                if "observations_dir" in config else None)
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
    observation["response"] = response
    return observation


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 2:
        print(json.dumps({"schema": OBSERVATION_SCHEMA, "error": "invalid_arguments"}))
        return 2
    try:
        observation = evaluate(argv[0], argv[1])
    except AdapterError as exc:
        print(json.dumps({"schema": OBSERVATION_SCHEMA, "error": str(exc),
                          **({"response_truncated": True} if str(exc) == "response_too_large" else {})}))
        return 1
    print(json.dumps(observation, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
