"""A private transport and execution boundary for the ordinary cyber GV service.

This adapter does not rank candidates, decide task success, or restart agents.
Only the host mounts /state, /evaluator, and verifier credentials. The solver
receives the standard MCP interface and a mechanically authenticated delivery
receipt. Hidden patched-build evaluation belongs to the host runner.
"""
from __future__ import annotations

import argparse
import asyncio
import hmac
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from verifier.llm.codex_backend import codex_usage_ledger
from verifier.mcp_server import Service, _default_kernel, build_server
from verifier.profiles.cyber.evaluator import EvaluatorSpec
from verifier.profiles.cyber.backend import CyberCodexBackend

from experiments.cybergym.model_config import (
    AUTO_COMPACT_TOKEN_LIMIT, CONTEXT_WINDOW, EFFORT, MODEL, SERVICE_TIER,
    context_cli_args,
)

MAX_HOOK_BODY = 4 * 1024 * 1024
PROFILE_ROOT = Path(__file__).resolve().parents[2]
ORIGIN = "solver-origin-untrusted"
AUTHORITY_SCOPE = "untrusted_solver_observation"


def strict_object(raw: bytes) -> dict | None:
    """Reject duplicate fields, nonfinite numbers, and non-object hook bodies."""
    duplicate = False

    def pairs(items):
        nonlocal duplicate
        result = {}
        for key, value in items:
            duplicate |= key in result
            result[key] = value
        return result

    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs,
                          parse_constant=lambda _: (_ for _ in ()).throw(ValueError()))
    except (UnicodeDecodeError, ValueError, TypeError):
        return None
    return value if isinstance(value, dict) and not duplicate else None


class SandboxedVerifierBackend(CyberCodexBackend):
    """Keep live prompts, but narrow every seat's shell to owned scratch."""

    _diagnostic_lock = threading.Lock()

    def __init__(self, *, cwd: str, project_root: str = "/task"):
        super().__init__(cwd=cwd, model=MODEL, effort=EFFORT,
                         service_tier=SERVICE_TIER, timeout=None,
                         phase_observer=self._diagnostic, project_root=project_root)
        self._diagnostic_failures = 0

    def _execution_policy_args(self, role_name, variables):
        return super()._execution_policy_args(role_name, variables) + context_cli_args()

    def _diagnostic(self, record):
        # The log is outside writable verifier scratch; it contains no prompt,
        # response, exception text, auth token, or evaluator configuration.
        path = Path(self.cwd) / "role_calls.jsonl"
        raw = (json.dumps(record, sort_keys=True) + "\n").encode()
        try:
            with self._diagnostic_lock:
                fd = os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT | os.O_NONBLOCK
                             | getattr(os, "O_NOFOLLOW", 0), 0o600)
                try:
                    info = os.fstat(fd)
                    if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                            or info.st_nlink != 1 or info.st_mode & 0o077):
                        raise OSError("unsafe role diagnostic log")
                    view = memoryview(raw)
                    while view:
                        written = os.write(fd, view)
                        if written <= 0:
                            raise OSError("role diagnostic write failed")
                        view = view[written:]
                finally:
                    os.close(fd)
        except OSError:
            self._diagnostic_failures += 1  # Telemetry must not decide a verdict.

    def complete(self, role_name, prompt, variables):
        call_id = uuid.uuid4().hex
        started = time.monotonic()
        common = {"call_id": call_id, "role": role_name, "model": MODEL,
                  "service_tier": SERVICE_TIER,
                  "model_context_window": CONTEXT_WINDOW,
                  "model_auto_compact_token_limit": AUTO_COMPACT_TOKEN_LIMIT}
        self._diagnostic({**common, "type": "role_call_started", "ts": time.time(),
                          "status": "running"})
        status = "failed"
        try:
            result = super().complete(role_name, prompt, variables)
            status = "completed"
            return result
        finally:
            self._diagnostic({**common, "type": "role_call_finished", "ts": time.time(),
                              "status": status,
                              "duration_seconds": time.monotonic() - started})


def kernel_factory(service):
    backend = SandboxedVerifierBackend(cwd=service.state_dir, project_root=service.project_root)
    return _default_kernel(service, model=MODEL, backend=backend)


def create_service(args) -> Service:
    state, project = args.state_dir, args.project_root
    for path in (state, project):
        info = path.lstat()
        if not path.is_absolute() or not stat.S_ISDIR(info.st_mode) or path.resolve() != path:
            raise ValueError("state and project must be real, absolute directories")
    if state.lstat().st_uid != os.geteuid():
        raise ValueError("private state must be owned by the sidecar process")
    if state == project or state.is_relative_to(project) or project.is_relative_to(state):
        raise ValueError("private state must be separate from the solver workspace")
    task = state / "task.md"
    if not stat.S_ISREG(task.lstat().st_mode):
        raise ValueError("create the authoritative task before starting the sidecar")
    command = json.loads(args.cyber_evaluator_command_json)
    if not isinstance(command, list) or not all(isinstance(item, str) for item in command):
        raise ValueError("evaluator command must be a JSON argv array")
    spec = EvaluatorSpec(tuple(command), environment=args.cyber_evaluator_environment,
                         authority_scope="candidate_signal", timeout=None,
                         pinned_files=tuple(args.cyber_evaluator_pin))
    from verifier.profiles.cyber.evidence import SeatCapabilities
    return Service(str(state), kernel_factory=kernel_factory, profile="cyber",
                   project_root=str(project), evaluator_spec=spec,
                   final_destination="final_poc", seats="codex", model=MODEL,
                   effort=EFFORT, service_tier=SERVICE_TIER,
                   seat_capabilities=SeatCapabilities(read_retained_artifacts=True,
                       read_candidate_blobs=True, execute_local_checks=True,
                       invoke_controller_evaluator=False))


class ReceiptBusy(ValueError):
    """A diagnostic poll must never wait for an in-flight verifier turn."""


def delivery_receipt(service: Service) -> dict:
    # Acquire both locks without waiting. A public commit can hold the event
    # feed lock during verification even when no candidate call is active.
    acquired = []
    try:
        for lock in (service._cyber_lock, service._event_feed_lock):
            if not lock.acquire(blocking=False):
                raise ReceiptBusy("verifier busy")
            acquired.append(lock)
        return _delivery_receipt_locked(service)
    finally:
        for lock in reversed(acquired):
            lock.release()


def _delivery_receipt_locked(service: Service) -> dict:
    """Authenticate an existing exact delivery without any model invocation."""
    with service._cyber_lock:
        service._check_task()
        service.drain_hooks()
        for event in reversed(list(service.log.replay())):
            if event["type"] != "cyber_materialized":
                continue
            payload = event["payload"]
            final_id = payload["final_claim_id"]
            candidate_id = payload["candidate_id"]
            certificate, full = service._authenticate_candidate_certificate(
                payload["certificate_id"], final_id, payload.get("full_certificate_id"))
            binding = service._cyber_bindings.get(final_id)
            if (binding is None or certificate.get("verdict") != "verified"
                    or certificate.get("final") != final_id
                    or service.store.status(final_id) != "verified"
                    or full.get("profile_metadata") != service._certificate_metadata(final_id)
                    or binding["subject"]["candidate_id"] != candidate_id
                    or payload["receipt"].get("candidate_id") != candidate_id
                    or payload["receipt"].get("destination") != service.final_destination):
                raise ValueError("delivery has no matching current verified certificate")
            if service.evaluator is not None:
                service.evaluator.spec.validate()
            with service.candidates.lease(candidate_id) as lease:
                service._validate_binding(binding)
                if binding["observations"] != service._observations(candidate_id):
                    raise ValueError("new observations require fresh adjudication")
                receipt = service.candidates.verify_delivery(payload["receipt"])
                service.candidates.validate(lease)
                candidate = lease.candidate
            return {"schema": "cybergym-delivery-receipt-v1",
                    "candidate_id": candidate_id, "sha256": candidate["sha256"],
                    "size": candidate["size"], "destination": receipt["destination"],
                    "certificate_id": payload["certificate_id"], "final_claim_id": final_id,
                    **({"full_certificate_id": payload["full_certificate_id"]}
                       if "full_certificate_id" in payload else {}),
                    "receipt_event_seq": event["seq"], "receipt": receipt,
                    "profile_id": service.profile.id,
                    "task_sha256": binding["task_sha256"]}
        raise ValueError("no certified delivery exists")


class ProtectedApplication:
    def __init__(self, app, *, token: str, service: Service,
                 profile_root: Path = PROFILE_ROOT):
        if not isinstance(token, str) or len(token) < 32 or any(ord(c) < 33 or ord(c) > 126 for c in token):
            raise ValueError("a strong per-attempt bearer token is required")
        self.app, self.token, self.service = app, token.encode("ascii"), service
        self.profile_root = profile_root
        self._hook_lock = asyncio.Lock()

    def authorized(self, scope):
        headers = [value for key, value in scope.get("headers", ())
                   if key.lower() == b"authorization"]
        return len(headers) == 1 and hmac.compare_digest(headers[0], b"Bearer " + self.token)

    async def response(self, send, status, payload):
        raw = json.dumps(payload, sort_keys=True).encode()
        await send({"type": "http.response.start", "status": status,
                    "headers": [(b"content-type", b"application/json"),
                                (b"content-length", str(len(raw)).encode()),
                                (b"cache-control", b"no-store")]})
        await send({"type": "http.response.body", "body": raw})

    async def body(self, receive):
        raw = bytearray()
        while True:
            message = await receive()
            if message["type"] == "http.disconnect":
                return None
            if message["type"] != "http.request":
                continue
            raw.extend(message.get("body", b""))
            if len(raw) > MAX_HOOK_BODY:
                return None
            if not message.get("more_body", False):
                return bytes(raw)

    def run_hook(self, kind, raw):
        # Original hooks normalize the raw tool envelope. An isolated temporary
        # feed lets this wrapper mark origin before the single canonical writer
        # ingests it; user-supplied type/authority fields cannot bypass this.
        script = self.profile_root / "hooks" / ("pre_tool_use.py" if kind == "pre" else "post_tool_use.py")
        with tempfile.TemporaryDirectory(prefix="hook-", dir=self.service.state_dir) as directory:
            feed = Path(directory) / "feed.jsonl"
            env = {"PATH": os.defpath, "LANG": "C", "LC_ALL": "C", "GV_HOOK_FEED": str(feed)}
            process = subprocess.run([sys.executable, str(script)], input=raw,
                                     capture_output=True, cwd=self.service.project_root,
                                     env=env, timeout=30, check=False)
            if process.returncode:
                raise ValueError("hook normalization failed")
            lines = feed.read_bytes().splitlines()
            event = strict_object(lines[0]) if len(lines) == 1 else None
            expected = "tool_call" if kind == "pre" else "tool_result"
            if event is None or event.get("type") != expected or not isinstance(event.get("payload"), dict):
                raise ValueError("hook did not produce one normal tool event")
            event["payload"]["origin"] = ORIGIN
            event["payload"]["authority_scope"] = AUTHORITY_SCOPE
            self.service._ingest_hook_event(event)
        return {}

    async def __call__(self, scope, receive, send):
        if scope["type"] == "lifespan":
            return await self.app(scope, receive, send)
        if scope["type"] != "http":
            return
        if not self.authorized(scope):
            return await self.response(send, 401, {"error": "unauthorized"})
        path, method = scope.get("path", ""), scope.get("method", "GET")
        if path == "/progress":
            if method != "GET":
                return await self.response(send, 405, {"error": "method not allowed"})
            # Operational metadata only, independent of graph/ledger locks.
            progress = await asyncio.to_thread(self.service.verification_jobs)
            return await self.response(send, 200, progress)
        if path in ("/health", "/receipt"):
            if method != "GET":
                return await self.response(send, 405, {"error": "method not allowed"})
            if path == "/health":
                return await self.response(send, 200, {"status": "ready"})
            try:
                receipt = await asyncio.to_thread(delivery_receipt, self.service)
            except ReceiptBusy:
                return await self.response(send, 409, {"error": "verifier busy"})
            except (OSError, ValueError, KeyError):
                return await self.response(send, 409, {"error": "no current certified delivery"})
            return await self.response(send, 200, receipt)
        if path in ("/hooks/pre", "/hooks/post"):
            if method != "POST":
                return await self.response(send, 405, {"error": "method not allowed"})
            raw = await self.body(receive)
            if raw is None or strict_object(raw) is None:
                return await self.response(send, 400, {"error": "invalid hook body"})
            async with self._hook_lock:
                try:
                    result = await asyncio.to_thread(self.run_hook, path.rsplit("/", 1)[-1], raw)
                except (OSError, ValueError, subprocess.TimeoutExpired):
                    return await self.response(send, 503, {"error": "hook relay unavailable"})
            return await self.response(send, 200, result)
        if path == "/mcp" or path.startswith("/mcp/"):
            return await self.app(scope, receive, send)
        return await self.response(send, 404, {"error": "not found"})


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", required=True, type=Path)
    parser.add_argument("--project-root", required=True, type=Path)
    parser.add_argument("--server-hostname", required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--token-env", default="GV_SIDECAR_TOKEN")
    parser.add_argument("--cyber-evaluator-command-json", required=True)
    parser.add_argument("--cyber-evaluator-pin", action="append", default=[])
    parser.add_argument("--cyber-evaluator-environment", default="vulnerable")
    return parser.parse_args(argv)


def main():
    args = parse_args()
    if os.geteuid() == 0:
        raise SystemExit("run the private verifier sidecar as a nonroot user")
    token = os.environ.get(args.token_env, "")
    # Validate auth before constructing the live service or exposing any port.
    if len(token) < 32 or any(ord(c) < 33 or ord(c) > 126 for c in token):
        raise SystemExit("missing strong per-attempt sidecar bearer token")
    service = create_service(args)
    server = build_server(service)
    server.settings.host, server.settings.port = "0.0.0.0", args.port
    server.settings.json_response = True
    server.settings.stateless_http = False
    from mcp.server.transport_security import TransportSecuritySettings
    server.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=[args.server_hostname + ":*", "127.0.0.1:*", "localhost:*"],
        allowed_origins=["http://" + args.server_hostname + ":*",
                         "http://127.0.0.1:*", "http://localhost:*"],
    )
    app = ProtectedApplication(server.streamable_http_app(), token=token, service=service)
    import uvicorn
    # Worker threads share this process-local capture. No usage context or
    # credentials are exported into a verifier seat's shell environment.
    with codex_usage_ledger(str(args.state_dir / "backend_usage.jsonl"), {
            str(args.state_dir): {"profile_id": service.profile.id,
                                 "task_sha256": service._profile_pin["task_sha256"]}}):
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
