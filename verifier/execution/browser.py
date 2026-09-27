"""Trusted browser-observation seam for typed execution adapters.

The Oracle may select observations, but it never supplies executable browser
code or a process command.  Deployments opt in a verifier-owned observer and
bind its executable identity by SHA-256.  With no configured observer the
browser adapter is simply absent from the catalog.
"""
from __future__ import annotations

import hashlib
import json
import os
import selectors
import signal
import stat
import subprocess
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol

from ..config import browser_observer_settings


BROWSER_REQUEST_SCHEMA = "gv-browser-observer-request-v1"
BROWSER_RESPONSE_SCHEMA = "gv-browser-observer-response-v1"
MAX_BROWSER_REQUEST_BYTES = 128 * 1024
MAX_BROWSER_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_BROWSER_RESPONSE_BYTES = 1024 * 1024
MAX_BROWSER_STDERR_BYTES = 64 * 1024
BROWSER_NETWORK_POLICY = "browser-request-blocking-v1"
BROWSER_DOCUMENT_POLICY = "single-document-v1"
BROWSER_OBSERVATION_PROFILE = {
    "viewport": {"width": 1280, "height": 720, "device_scale_factor": 1},
    "locale": "en-US",
    "timezone": "UTC",
    "dialog_policy": "dismiss",
    "favicon_policy": "empty",
    "color_scheme": "light",
    "reduced_motion": "no-preference",
}


class BrowserObserver(Protocol):
    """A verifier-owned browser observer with an explicit network policy."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def observe(self, document: bytes, request: Mapping[str, Any]) \
            -> Mapping[str, Any]: ...


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _pinned_executable(path: str, expected_digest: str, label: str) -> str:
    """Resolve and authenticate one verifier-configured executable."""
    absolute = os.path.abspath(path)
    try:
        info = os.stat(absolute, follow_symlinks=False)
    except OSError as error:
        raise ValueError(f"{label} is unavailable: {error}") from error
    if not stat.S_ISREG(info.st_mode) or not os.access(absolute, os.X_OK):
        raise ValueError(f"{label} must be an executable regular file")
    if _sha256_file(absolute) != expected_digest:
        raise ValueError(f"{label} SHA-256 does not match configuration")
    return absolute


def _kill_process_group(process: subprocess.Popen) -> None:
    """Stop the observer and every browser process it started."""
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    finally:
        if process.poll() is None:
            process.wait()


def _bounded_communicate(
    process: subprocess.Popen,
    body: bytes,
    timeout_seconds: float,
) -> tuple[bytes, bytes]:
    """Drain child pipes without ever retaining unbounded output."""
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise RuntimeError("browser observer pipes were not created")
    streams = selectors.DefaultSelector()
    buffers = {"stdout": bytearray(), "stderr": bytearray()}
    limits = {
        "stdout": MAX_BROWSER_RESPONSE_BYTES,
        "stderr": MAX_BROWSER_STDERR_BYTES,
    }
    os.set_blocking(process.stdin.fileno(), False)
    streams.register(process.stdin, selectors.EVENT_WRITE, "stdin")
    for name, stream in (("stdout", process.stdout),
                         ("stderr", process.stderr)):
        os.set_blocking(stream.fileno(), False)
        streams.register(stream, selectors.EVENT_READ, name)

    body_view = memoryview(body)
    body_offset = 0
    deadline = time.monotonic() + timeout_seconds
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise subprocess.TimeoutExpired(
                    process.args, timeout_seconds)
            ready = streams.select(min(remaining, 0.1))
            for key, _events in ready:
                name = key.data
                if name == "stdin":
                    try:
                        written = os.write(
                            key.fileobj.fileno(),
                            body_view[body_offset:body_offset + 64 * 1024],
                        )
                    except BrokenPipeError:
                        written = 0
                    body_offset += written
                    if written == 0 or body_offset == len(body_view):
                        streams.unregister(key.fileobj)
                        key.fileobj.close()
                    continue
                chunk = os.read(key.fileobj.fileno(), 64 * 1024)
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                buffers[name].extend(chunk)
                if len(buffers[name]) > limits[name]:
                    raise RuntimeError(
                        f"browser observer {name} exceeds the size limit")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(process.args, timeout_seconds)
        process.wait(timeout=remaining)
        # A correct observer reaps its browser.  This final group guard also
        # removes any descendant that ignored normal browser-parent cleanup.
        _kill_process_group(process)
    except Exception:
        _kill_process_group(process)
        raise
    finally:
        streams.close()
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None and not stream.closed:
                stream.close()
    return bytes(buffers["stdout"]), bytes(buffers["stderr"])


@dataclass(frozen=True)
class ProcessBrowserObserver:
    """Invoke one operator-pinned observer over a small JSON protocol.

    The executable is verifier configuration, never MethodIR.  It receives a
    fresh private copy of the HTML document and must perform its own browser
    under the exact, catalogued observation profile.  Request blocking is a
    browser-level policy, not a claim of OS-level network isolation.
    """

    executable: str
    executable_sha256: str
    browser_executable: str
    browser_executable_sha256: str
    timeout_seconds: float = 15.0

    def __post_init__(self) -> None:
        if os.name != "posix":
            raise ValueError(
                "the process browser observer requires POSIX process groups")
        if not isinstance(self.timeout_seconds, (int, float)) \
                or isinstance(self.timeout_seconds, bool) \
                or not 0 < self.timeout_seconds <= 60:
            raise ValueError("browser observer timeout must be in (0, 60]")
        observer = _pinned_executable(
            self.executable, self.executable_sha256, "browser observer")
        browser = _pinned_executable(
            self.browser_executable, self.browser_executable_sha256,
            "browser executable")
        object.__setattr__(self, "executable", observer)
        object.__setattr__(self, "browser_executable", browser)

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "kind": "pinned-process",
            "protocol": BROWSER_REQUEST_SCHEMA,
            "observer_sha256": self.executable_sha256,
            "browser_sha256": self.browser_executable_sha256,
            "network_policy": BROWSER_NETWORK_POLICY,
            "document_policy": BROWSER_DOCUMENT_POLICY,
            "observation_profile": json.loads(json.dumps(
                BROWSER_OBSERVATION_PROFILE, sort_keys=True)),
        }

    def observe(self, document: bytes, request: Mapping[str, Any]) \
            -> Mapping[str, Any]:
        if set(request) != {"wait_ms", "selectors"}:
            raise RuntimeError("browser observer request has the wrong shape")
        if not isinstance(document, bytes) \
                or len(document) > MAX_BROWSER_DOCUMENT_BYTES:
            raise RuntimeError("browser document exceeds the size limit")
        if _sha256_file(self.executable) != self.executable_sha256:
            raise RuntimeError("browser observer identity changed after startup")
        if _sha256_file(self.browser_executable) \
                != self.browser_executable_sha256:
            raise RuntimeError("browser executable identity changed after startup")
        with tempfile.TemporaryDirectory(prefix="gv-browser-") as root:
            document_path = os.path.join(root, "document.html")
            with open(document_path, "xb") as handle:
                handle.write(document)
            body = {
                **dict(request),
                "schema": BROWSER_REQUEST_SCHEMA,
                "document_path": document_path,
                "document_sha256": hashlib.sha256(document).hexdigest(),
                "browser_path": self.browser_executable,
                "browser_sha256": self.browser_executable_sha256,
                "observation_profile": BROWSER_OBSERVATION_PROFILE,
            }
            encoded_body = json.dumps(
                body, ensure_ascii=False, sort_keys=True).encode("utf-8")
            if len(encoded_body) > MAX_BROWSER_REQUEST_BYTES:
                raise RuntimeError("browser observer request exceeds the size limit")
            process = None
            try:
                process = subprocess.Popen(
                    [self.executable],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=root,
                    env={"LANG": "C.UTF-8", "LC_ALL": "C.UTF-8",
                         "PATH": os.defpath},
                    start_new_session=True,
                )
                from .resource_share import yield_process
                yield_process(process.pid)
                stdout, stderr = _bounded_communicate(
                    process, encoded_body, self.timeout_seconds,
                )
            except subprocess.TimeoutExpired as error:
                raise RuntimeError(f"browser observer failed: {error}") from error
            except (OSError, RuntimeError) as error:
                raise RuntimeError(f"browser observer failed: {error}") from error
        if process.returncode != 0:
            detail = stderr[:500].decode("utf-8", errors="replace").replace(
                "\n", " ")
            raise RuntimeError(
                f"browser observer exited {process.returncode}: {detail}")
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeDecodeError, TypeError, ValueError,
                json.JSONDecodeError) as error:
            raise RuntimeError("browser observer returned invalid JSON") from error
        if not isinstance(response, dict):
            raise RuntimeError("browser observer response must be an object")
        return response


def configured_browser_observer(
    environ: Optional[Mapping[str, str]] = None,
) -> Optional[BrowserObserver]:
    """Return an explicitly configured observer; never auto-discover one."""
    configured = browser_observer_settings(environ)
    if configured is None:
        return None
    return ProcessBrowserObserver(*configured)
