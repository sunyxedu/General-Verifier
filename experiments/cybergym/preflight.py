"""Offline pinned-Codex typed configuration smoke; never starts a model turn.

Run ``python -m experiments.cybergym.preflight`` before an experiment. The
fresh, networkless container sees only this source tree, not any user's auth or
experiment homes. Every cyber role's real permission arguments reach the
pinned CLI. Acceptance is configuration validation, not a claim that sandbox
read/write enforcement or a model's usable context budget has been measured.
Only initialize, initialized and config/read JSON-RPC messages are permitted.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import sys
import time
import tomllib
import uuid

from .model_config import AUTO_COMPACT_TOKEN_LIMIT, CONTEXT_WINDOW, MODEL, EFFORT, SERVICE_TIER
from .run import IMAGE

LABEL = "gv.preflight"
PURPOSE = "cybergym-role-config-preflight"
METHODS = frozenset({"initialize", "initialized", "config/read"})
VALUES = {"model_context_window": CONTEXT_WINDOW,
          "model_auto_compact_token_limit": AUTO_COMPACT_TOKEN_LIMIT}
MAX_OUTPUT = 2 * 1024 * 1024


class PreflightError(ValueError):
    """Static error codes only; never echo a CLI response or configuration."""


def _remaining(deadline, maximum=5.0):
    left = deadline - time.monotonic()
    if left <= 0:
        raise PreflightError("preflight_deadline_exceeded")
    return min(left, maximum)


def _typed_error(raw, key):
    text = raw.decode("utf-8", errors="replace").lower()
    return key.lower() in text and any(marker in text for marker in
        ("invalid type", "expected i64", "expected u64", "expected integer",
         "expected an integer", "duplicate key", "duplicate field", "expected a map"))


class _Conversation:
    def __init__(self, process, deadline):
        self.process, self.deadline = process, deadline
        self.selector = selectors.DefaultSelector()
        self.buffers = {"stdout": b"", "stderr": b""}
        self.messages = []
        for stream, name in ((process.stdout, "stdout"), (process.stderr, "stderr")):
            os.set_blocking(stream.fileno(), False)
            self.selector.register(stream, selectors.EVENT_READ, name)

    def send(self, method, params, request_id=None):
        if method not in METHODS:
            raise PreflightError("forbidden_protocol_method")
        message = {"method": method, "params": params}
        if request_id is not None:
            message["id"] = request_id
        try:
            self.process.stdin.write((json.dumps(message) + "\n").encode())
            self.process.stdin.flush()
        except (OSError, BrokenPipeError):
            raise PreflightError("protocol_input_closed") from None

    def read(self, request_id):
        while True:
            for index, message in enumerate(self.messages):
                if message.get("id") == request_id:
                    return self.messages.pop(index)
            if not self.selector.get_map():
                return None
            for ready, _ in self.selector.select(_remaining(self.deadline, .2)):
                name = ready.data
                try:
                    chunk = os.read(ready.fileobj.fileno(), 65536)
                except BlockingIOError:
                    continue
                if not chunk:
                    self.selector.unregister(ready.fileobj)
                    continue
                self.buffers[name] += chunk
                if len(self.buffers[name]) > MAX_OUTPUT:
                    raise PreflightError("protocol_output_too_large")
                if name == "stdout":
                    while b"\n" in self.buffers[name]:
                        line, self.buffers[name] = self.buffers[name].split(b"\n", 1)
                        try:
                            message = json.loads(line)
                        except ValueError:
                            raise PreflightError("invalid_protocol_json") from None
                        if not isinstance(message, dict):
                            raise PreflightError("invalid_protocol_envelope")
                        self.messages.append(message)
                        if len(self.messages) > 100:
                            raise PreflightError("excess_protocol_notifications")

    def close(self):
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout, self.process.stderr):
            try:
                stream.close()
            except OSError:
                pass


def _permission_projection(args, role):
    """Inspect generated arguments, not a handwritten substitute for the seat."""
    assignments = [args[index + 1] for index, value in enumerate(args[:-1]) if value == "-c"]
    filesystem = [value for value in assignments if ".filesystem=" in value]
    if len(filesystem) != 1 or "--ignore-user-config" not in args:
        raise PreflightError("role_permission_arguments_missing")
    prefix, table = filesystem[0].split("=", 1)
    grants = tomllib.loads("grants=" + table)["grants"]
    profile = prefix.split(".")[1]
    if (f'default_permissions="{profile}"' not in assignments
            or f'permissions.{profile}.extends=":read-only"' not in assignments
            or f'permissions.{profile}.network.enabled=false' not in assignments
            or 'approval_policy="never"' not in assignments
            or "--dangerously-bypass-approvals-and-sandbox" in args):
        raise PreflightError("role_permission_boundary_mismatch")
    for path in ("/evaluator", "/codex-home", "/gv", "/controller", "/state/verifier-home"):
        if grants.get(path) != "deny":
            raise PreflightError("role_private_root_not_denied")
    envs = [value for value in assignments if value.startswith("shell_environment_policy.set=")]
    if len(envs) != 1:
        raise PreflightError("role_shell_environment_missing")
    env = tomllib.loads(envs[0])["shell_environment_policy"]["set"]
    blind = role in ("obligations", "critic")
    if blind:
        for path in ("/task", "/state/artifacts", "/state/cyber", "/state/work",
                     "/state/workspace", "/state/cybergym-observations", "/state/hook_feed.jsonl"):
            if grants.get(path) != "deny":
                raise PreflightError("answer_blind_root_not_denied")
        if env.get("HOME") != f"/state/_blind/{role}/home" or grants.get(f"/state/_blind/{role}") != "write":
            raise PreflightError("answer_blind_home_not_isolated")
    elif grants.get("/state/work") != "write":
        raise PreflightError("evidence_seat_scratch_not_writable")
    return {"profile": profile, "network_enabled": False, "answer_blind": blind,
            "generated_permission_arguments_sha256": hashlib.sha256(
                json.dumps(args, separators=(",", ":")).encode()).hexdigest()}


def _one_case(role, args, deadline, negative=None):
    command = ["codex", "app-server", "--stdio", "--strict-config", *args,
               "-c", 'check_for_update_on_startup=false', "-c", 'analytics.enabled=false',
               "-c", 'otel.exporter="none"', "-c", 'otel.trace_exporter="none"',
               "-c", f'model="{MODEL}"']
    error_key = None
    if negative in VALUES:
        error_key = negative
        command += ["-c", negative + '="preflight_invalid_type"']
    elif negative == "duplicate_grant":
        filesystem = next(value for value in args if ".filesystem=" in value)
        key, value = filesystem.split("=", 1)
        command += ["-c", key + "=" + value[:-1] + ',"/state/work"="deny"}']
        error_key = "filesystem"
    elif negative is not None:
        raise PreflightError("unknown_negative_control")
    case = role + ("-" + negative if negative else "")
    env = {"PATH": os.environ.get("PATH", os.defpath), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
           "HOME": "/preflight/home/" + case, "CODEX_HOME": "/preflight/codex-home/" + case,
           "TMPDIR": "/preflight/tmp/" + case, "PYTHONDONTWRITEBYTECODE": "1"}
    for key in ("HOME", "CODEX_HOME", "TMPDIR"):
        Path(env[key]).mkdir(parents=True, exist_ok=True)
    process = None
    conversation = None
    try:
        process = subprocess.Popen(command, cwd="/state", env=env, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        conversation = _Conversation(process, deadline)
        conversation.send("initialize", {"clientInfo": {"name": "gv_role_preflight",
                          "title": "GV Role Configuration Preflight", "version": "1.0.0"}}, 0)
        initialized = conversation.read(0)
        if initialized is None or "error" in initialized:
            error = (json.dumps(initialized["error"]).encode() if initialized else
                     conversation.buffers["stderr"])
            if error_key and _typed_error(error, error_key):
                return {"case": negative, "typed_rejected": True}
            raise PreflightError("initialize_not_accepted")
        if "result" not in initialized:
            raise PreflightError("initialize_result_missing")
        conversation.send("initialized", {})
        conversation.send("config/read", {"includeLayers": False}, 1)
        response = conversation.read(1)
        if response is None or "error" in response:
            error = (json.dumps(response["error"]).encode() if response else
                     conversation.buffers["stderr"])
            if error_key and _typed_error(error, error_key):
                return {"case": negative, "typed_rejected": True}
            raise PreflightError("config_read_not_accepted")
        if error_key:
            raise PreflightError("negative_control_was_accepted")
        config = response.get("result", {}).get("config")
        if not isinstance(config, dict):
            raise PreflightError("effective_config_missing")
        projection = {key: config.get(key) for key in VALUES}
        if projection != VALUES or any(type(value) is not int for value in projection.values()):
            raise PreflightError("effective_context_mismatch")
        for key, expected in (("model", MODEL), ("model_reasoning_effort", EFFORT),
                              ("service_tier", SERVICE_TIER)):
            if config.get(key) != expected:
                raise PreflightError("effective_" + key + "_mismatch")
        return {"role": role, "typed_config_accepted": True, "effective_context": projection,
                "generated_permissions": _permission_projection(args, role)}
    except OSError:
        raise PreflightError("protocol_process_io_failed") from None
    finally:
        if conversation is not None:
            conversation.close()
        if process is not None:
            try:
                process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                process.kill()  # Only this offline fixture's app-server child.
                process.wait(timeout=1)


def inside(timeout=15.0):
    from .sidecar import SandboxedVerifierBackend
    from verifier.llm.agents import ALL_AGENT_ROLES
    from verifier.profiles import get_profile

    for path in ("/state", "/task", "/preflight/home", "/preflight/codex-home", "/preflight/tmp"):
        Path(path).mkdir(parents=True, exist_ok=True)
    backend = SandboxedVerifierBackend(cwd="/state", project_root="/task")
    roles = get_profile("cyber").apply_roles(ALL_AGENT_ROLES)
    backend.roles = roles
    from verifier.llm.codex_service import service_tier_cli_args
    generated = {role: [*backend._execution_policy_args(role, {
        "world_mode": "live", "ledger_path": "/state/events.jsonl"}),
        "-c", "model_reasoning_effort=" + json.dumps(backend.effort),
        *service_tier_cli_args(backend.service_tier)] for role in roles}
    for role, args in generated.items():
        _permission_projection(args, role)
    deadline = time.monotonic() + timeout
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
        jobs = [(role, executor.submit(_one_case, role, args, deadline)) for role, args in generated.items()]
        records = []
        for role, job in jobs:
            try:
                records.append(job.result())
            except PreflightError as error:
                raise PreflightError("role_" + role + "_" + str(error)) from None
    controls = []
    for key in (*VALUES, "duplicate_grant"):
        try:
            controls.append(_one_case("discharge", generated["discharge"], deadline, key))
        except PreflightError as error:
            raise PreflightError("negative_" + key + "_" + str(error)) from None
    return {"schema": "gv-cybergym-preflight-v1", "image": IMAGE,
            "roles": sorted(records, key=lambda row: row["role"]), "negative_controls": controls,
            "sandbox_enforcement_tested": False, "model_calls": 0}


def _docker(args, timeout=5):
    try:
        return subprocess.run(["docker", *args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise PreflightError("docker_command_failed") from None


def _cleanup(container_id, marker):
    try:
        inspected = _docker(["inspect", "--format", "{{json .Config.Labels}}", container_id])
        if inspected.returncode:
            return False
        labels = json.loads(inspected.stdout)
        if not isinstance(labels, dict) or labels.get(LABEL) != marker or labels.get("gv.purpose") != PURPOSE:
            return False
        stopped = _docker(["stop", "--time", "1", container_id])
        return stopped.returncode == 0 and _docker(["rm", container_id]).returncode == 0
    except (PreflightError, ValueError, TypeError):
        return False


def _validate_report(report):
    from verifier.profiles.cyber.prompts import SYSTEMS
    from verifier.llm.codex_backend import _OFFLINE_PROFILE

    required = {"schema", "image", "roles", "negative_controls", "sandbox_enforcement_tested", "model_calls"}
    if (not isinstance(report, dict) or set(report) != required
            or report["schema"] != "gv-cybergym-preflight-v1" or report["image"] != IMAGE
            or type(report["model_calls"]) is not int or report["model_calls"] != 0
            or report["sandbox_enforcement_tested"] is not False
            or not isinstance(report["roles"], list) or len(report["roles"]) != len(SYSTEMS)
            or not isinstance(report["negative_controls"], list)
            or len(report["negative_controls"]) != 3):
        raise PreflightError("incomplete_preflight_report")
    seen = set()
    for row in report["roles"]:
        if (not isinstance(row, dict) or set(row) != {"role", "typed_config_accepted", "effective_context",
                                                    "generated_permissions"}
                or row["role"] not in SYSTEMS or row["role"] in seen
                or row["typed_config_accepted"] is not True
                or not isinstance(row["effective_context"], dict)
                or row["effective_context"] != VALUES
                or any(type(value) is not int for value in row["effective_context"].values())):
            raise PreflightError("invalid_role_config_report")
        seen.add(row["role"])
        permissions = row["generated_permissions"]
        if (not isinstance(permissions, dict) or set(permissions) != {
                "profile", "network_enabled", "answer_blind", "generated_permission_arguments_sha256"}
                or permissions["profile"] != _OFFLINE_PROFILE
                or permissions["network_enabled"] is not False
                or permissions["answer_blind"] is not (row["role"] in ("obligations", "critic"))
                or not isinstance(permissions["generated_permission_arguments_sha256"], str)
                or not re.fullmatch(r"[a-f0-9]{64}", permissions["generated_permission_arguments_sha256"])):
            raise PreflightError("invalid_permission_config_report")
    seen_controls = set()
    for row in report["negative_controls"]:
        if (not isinstance(row, dict) or set(row) != {"case", "typed_rejected"}
                or row["case"] not in (*VALUES, "duplicate_grant")
                or row["case"] in seen_controls or row["typed_rejected"] is not True):
            raise PreflightError("invalid_negative_control_report")
        seen_controls.add(row["case"])


def run(timeout=20.0):
    if type(timeout) not in (int, float) or not 5 <= timeout <= 60:
        raise PreflightError("invalid_preflight_timeout")
    marker = uuid.uuid4().hex
    source = str(Path(__file__).resolve().parents[2])
    command = ["create", "--name", "gv-role-preflight-" + marker, "--network", "none",
               "--user", "1000:1000", "--read-only", "--label", LABEL + "=" + marker,
               "--label", "gv.purpose=" + PURPOSE,
               "--mount", "type=bind,source=" + source + ",target=/gv,readonly",
               "--env", "HOME=/preflight/home", "--env", "CODEX_HOME=/preflight/codex-home",
               "--env", "TMPDIR=/preflight/tmp", "--env", "PYTHONPATH=/gv",
               "--env", "PYTHONDONTWRITEBYTECODE=1"]
    for path in ("/state", "/task", "/preflight", "/tmp"):
        command += ["--tmpfs", path + ":rw,uid=1000,gid=1000,mode=0700"]
    command += ["--entrypoint", "python3", IMAGE, "-m", "experiments.cybergym.preflight",
                "--inside", "--timeout", str(max(1.0, timeout - 5))]
    created = _docker(command)
    container_id = created.stdout.decode("ascii", errors="replace").strip()
    if created.returncode or not re.fullmatch(r"[a-f0-9]{64}", container_id):
        raise PreflightError("container_create_failed")
    try:
        result = _docker(["start", "-a", container_id], timeout=timeout)
        if result.returncode:
            match = re.search(rb"^cybergym-preflight ([a-z0-9_]{1,160})\r?$", result.stderr, re.MULTILINE)
            if match:
                raise PreflightError("pinned_" + match[1].decode("ascii"))
            raise PreflightError("pinned_role_config_preflight_failed")
        if len(result.stdout) > MAX_OUTPUT:
            raise PreflightError("preflight_report_too_large")
        try:
            report = json.loads(result.stdout)
        except ValueError:
            raise PreflightError("invalid_preflight_report") from None
        _validate_report(report)
        return report
    finally:
        if not _cleanup(container_id, marker):
            raise PreflightError("owned_preflight_container_cleanup_failed")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inside", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        report = inside(args.timeout) if args.inside else run(args.timeout)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, PreflightError) else "unexpected_preflight_error"
        print("cybergym-preflight " + code, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
