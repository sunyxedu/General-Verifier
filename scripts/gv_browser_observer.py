#!/usr/bin/env python3
"""Dependency-free Chromium observer for ``browser.html@1``.

This is a verifier-side instrument, not an agent tool.  It accepts exactly
one JSON request on stdin, serves only the supplied document on loopback,
blocks every other browser request through CDP, and emits objective browser
observations.  No caller-provided JavaScript is evaluated.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import secrets
import socket
import stat
import struct
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import urlopen


REQUEST_SCHEMA = "gv-browser-observer-request-v1"
RESPONSE_SCHEMA = "gv-browser-observer-response-v1"
MAX_REQUEST_BYTES = 128 * 1024
MAX_DOCUMENT_BYTES = 8 * 1024 * 1024
MAX_CDP_MESSAGE_BYTES = 2 * 1024 * 1024
MAX_DISCOVERY_BYTES = 64 * 1024
MAX_RESPONSE_BYTES = 512 * 1024
MAX_OBSERVATION_BYTES = 256 * 1024
MAX_ITEMS = 256
MAX_TEXT_CHARS = 4_096
MAX_SELECTOR_NODES = 1_000
MAX_SELECTOR_CHARS = 4_096
STARTUP_SECONDS = 8.0
LOAD_SECONDS = 8.0
NETWORK_POLICY = "browser-request-blocking-v1"
DOCUMENT_POLICY = "single-document-v1"
OBSERVATION_PROFILE = {
    "viewport": {"width": 1280, "height": 720, "device_scale_factor": 1},
    "locale": "en-US",
    "timezone": "UTC",
    "dialog_policy": "dismiss",
    "favicon_policy": "empty",
    "color_scheme": "light",
    "reduced_motion": "no-preference",
}


def _fail(message: str) -> "None":
    raise RuntimeError(message)


def _sha256_file(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


def _request() -> dict:
    raw = sys.stdin.buffer.read(MAX_REQUEST_BYTES + 1)
    if len(raw) > MAX_REQUEST_BYTES:
        _fail("request exceeds size limit")
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("request is not valid JSON") from error
    expected = {
        "schema", "document_path", "document_sha256", "wait_ms", "selectors",
        "browser_path", "browser_sha256", "observation_profile",
    }
    if not isinstance(value, dict) or set(value) != expected \
            or value.get("schema") != REQUEST_SCHEMA:
        _fail("request has the wrong schema")
    wait_ms = value["wait_ms"]
    selectors = value["selectors"]
    if isinstance(wait_ms, bool) or not isinstance(wait_ms, int) \
            or not 0 <= wait_ms <= 5_000:
        _fail("wait_ms is invalid")
    if not isinstance(selectors, list) or len(selectors) > 16 \
            or any(not isinstance(item, str) or not item
                   or len(item) > MAX_SELECTOR_CHARS
                   for item in selectors) \
            or selectors != sorted(set(selectors)):
        _fail("selectors are invalid")
    if value["observation_profile"] != OBSERVATION_PROFILE:
        _fail("observation profile is invalid")
    boundary = os.path.realpath(os.getcwd())
    path = os.path.realpath(value["document_path"])
    if os.path.commonpath((boundary, path)) != boundary \
            or os.path.islink(path) or not os.path.isfile(path):
        _fail("document is outside the observer boundary")
    with open(path, "rb") as handle:
        document = handle.read(MAX_DOCUMENT_BYTES + 1)
    if len(document) > MAX_DOCUMENT_BYTES:
        _fail("document exceeds size limit")
    if hashlib.sha256(document).hexdigest() != value["document_sha256"]:
        _fail("document digest mismatch")
    browser_path = value["browser_path"]
    browser_digest = value["browser_sha256"]
    if not isinstance(browser_path, str) or not isinstance(browser_digest, str):
        _fail("browser identity is invalid")
    browser_path = os.path.abspath(browser_path)
    try:
        browser_info = os.stat(browser_path, follow_symlinks=False)
    except OSError as error:
        raise RuntimeError(f"browser executable is unavailable: {error}") \
            from error
    if not stat.S_ISREG(browser_info.st_mode) \
            or not os.access(browser_path, os.X_OK) \
            or _sha256_file(browser_path) != browser_digest:
        _fail("browser executable identity mismatch")
    return {**value, "browser_path": browser_path, "document": document}


class _DocumentHandler(BaseHTTPRequestHandler):
    document = b""

    def do_GET(self) -> None:  # noqa: N802 (BaseHTTPRequestHandler API)
        path = urlsplit(self.path).path
        if path == "/favicon.ico":
            self.send_response(204)
            self.send_header("Content-Length", "0")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            return
        if path != "/document":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", str(len(self.document)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(self.document)

    def log_message(self, _format: str, *args) -> None:
        return


class _WebSocket:
    def __init__(self, url: str):
        parsed = urlsplit(url)
        if parsed.scheme != "ws" or parsed.hostname not in {
                "127.0.0.1", "localhost"}:
            _fail("CDP endpoint is not loopback WebSocket")
        self.socket = socket.create_connection(
            (parsed.hostname, parsed.port or 80), timeout=2.0)
        key = base64.b64encode(secrets.token_bytes(16)).decode("ascii")
        path = parsed.path + (("?" + parsed.query) if parsed.query else "")
        request = (
            f"GET {path} HTTP/1.1\r\nHost: {parsed.netloc}\r\n"
            "Upgrade: websocket\r\nConnection: Upgrade\r\n"
            f"Sec-WebSocket-Key: {key}\r\nSec-WebSocket-Version: 13\r\n"
            "Origin: http://127.0.0.1\r\n\r\n"
        )
        self.socket.sendall(request.encode("ascii"))
        response = b""
        while b"\r\n\r\n" not in response and len(response) < 64 * 1024:
            response += self.socket.recv(4096)
        header, separator, self.buffer = response.partition(b"\r\n\r\n")
        if not separator or not header.startswith(b"HTTP/1.1 101 "):
            _fail("CDP WebSocket handshake failed")
        expected = base64.b64encode(hashlib.sha1(
            (key + "258EAFA5-E914-47DA-95CA-C5AB0DC85B11").encode("ascii")
        ).digest()).decode("ascii")
        headers = {}
        for line in header.split(b"\r\n")[1:]:
            name, _, value = line.partition(b":")
            headers[name.strip().lower()] = value.strip()
        if headers.get(b"sec-websocket-accept", b"").decode("ascii") != expected:
            _fail("CDP WebSocket identity check failed")

    def close(self) -> None:
        try:
            self.socket.close()
        except OSError:
            pass

    def _read(self, count: int) -> bytes:
        while len(self.buffer) < count:
            chunk = self.socket.recv(max(4096, count - len(self.buffer)))
            if not chunk:
                _fail("CDP WebSocket closed")
            self.buffer += chunk
        value, self.buffer = self.buffer[:count], self.buffer[count:]
        return value

    def _send_payload(self, payload: bytes, opcode: int) -> None:
        mask = secrets.token_bytes(4)
        size = len(payload)
        if size < 126:
            header = bytes((0x80 | opcode, 0x80 | size))
        elif size < 65_536:
            header = bytes((0x80 | opcode, 0x80 | 126)) + struct.pack("!H", size)
        else:
            header = bytes((0x80 | opcode, 0x80 | 127)) + struct.pack("!Q", size)
        masked = bytes(byte ^ mask[index % 4]
                       for index, byte in enumerate(payload))
        self.socket.sendall(header + mask + masked)

    def send(self, value: dict) -> None:
        payload = json.dumps(value, separators=(",", ":")).encode("utf-8")
        self._send_payload(payload, 1)

    def receive(self, timeout: float) -> dict:
        self.socket.settimeout(timeout)
        fragments = bytearray()
        initial_opcode = None
        while True:
            first, second = self._read(2)
            final, opcode = bool(first & 0x80), first & 0x0F
            masked, size = bool(second & 0x80), second & 0x7F
            if size == 126:
                size = struct.unpack("!H", self._read(2))[0]
            elif size == 127:
                size = struct.unpack("!Q", self._read(8))[0]
            if size > MAX_CDP_MESSAGE_BYTES \
                    or len(fragments) + size > MAX_CDP_MESSAGE_BYTES:
                _fail("CDP message exceeds size limit")
            mask = self._read(4) if masked else b""
            payload = self._read(size)
            if masked:
                payload = bytes(byte ^ mask[index % 4]
                                for index, byte in enumerate(payload))
            if opcode == 8:
                _fail("CDP WebSocket sent a close frame")
            if opcode == 9:
                self._send_payload(payload, 10)
                continue
            if opcode in (1, 2):
                initial_opcode = opcode
                fragments = bytearray(payload)
            elif opcode == 0 and initial_opcode is not None:
                fragments.extend(payload)
            else:
                continue
            if not final:
                continue
            if initial_opcode != 1:
                _fail("CDP returned an unexpected binary frame")
            try:
                value = json.loads(fragments.decode("utf-8"))
            except (UnicodeDecodeError, ValueError, json.JSONDecodeError) as error:
                raise RuntimeError("CDP returned invalid JSON") from error
            if not isinstance(value, dict):
                _fail("CDP message is not an object")
            return value


class _CDP:
    def __init__(self, websocket: _WebSocket, document_url: str,
                 favicon_url: str):
        self.websocket = websocket
        self.document_url = document_url
        self.favicon_url = favicon_url
        self.next_id = 0
        self.dialogs: list[dict] = []
        self.console: list[dict] = []
        self.page_errors: list[str] = []
        self.blocked_requests: list[dict] = []
        self.loaded_loader_ids: set[str] = set()
        self.observation_bytes = 0
        self.capture_open = True

    def _send(self, method: str, params: dict | None = None) -> int:
        self.next_id += 1
        self.websocket.send({"id": self.next_id, "method": method,
                             "params": params or {}})
        return self.next_id

    @staticmethod
    def _bounded_text(value) -> str:
        if isinstance(value, str):
            text = value
        else:
            text = json.dumps(value, ensure_ascii=False, sort_keys=True,
                              separators=(",", ":"))
        if len(text) > MAX_TEXT_CHARS:
            _fail("browser observation contains oversized text")
        return text

    def _append(self, items: list, value) -> None:
        if len(items) >= MAX_ITEMS:
            _fail("browser observation contains too many events")
        size = len(json.dumps(
            value, ensure_ascii=False, sort_keys=True,
            separators=(",", ":")).encode("utf-8"))
        if self.observation_bytes + size > MAX_OBSERVATION_BYTES:
            _fail("browser observations exceed total size limit")
        self.observation_bytes += size
        items.append(value)

    def _event(self, message: dict) -> None:
        method, params = message.get("method"), message.get("params", {})
        if method == "Fetch.requestPaused":
            request_id = params.get("requestId")
            request = params.get("request", {})
            if request.get("url") == self.document_url \
                    and params.get("resourceType") == "Document":
                self._send("Fetch.continueRequest", {"requestId": request_id})
            elif request.get("url") == self.favicon_url \
                    and params.get("resourceType") == "Other":
                self._send("Fetch.continueRequest", {"requestId": request_id})
            else:
                if self.capture_open:
                    self._append(self.blocked_requests, {
                        "url": self._bounded_text(request.get("url", "")),
                        "resource_type": self._bounded_text(
                            params.get("resourceType", "")),
                    })
                self._send("Fetch.failRequest", {
                    "requestId": request_id, "errorReason": "BlockedByClient"})
            return
        if not self.capture_open:
            return
        if method == "Page.lifecycleEvent" and params.get("name") == "load":
            self.loaded_loader_ids.add(params.get("loaderId", ""))
        elif method == "Page.javascriptDialogOpening":
            self._append(self.dialogs, {
                "type": self._bounded_text(params.get("type", "")),
                "message": self._bounded_text(params.get("message", "")),
            })
            self._send("Page.handleJavaScriptDialog", {"accept": False})
        elif method == "Runtime.consoleAPICalled":
            parts = []
            for item in params.get("args", []):
                if "value" in item:
                    parts.append(self._bounded_text(item["value"]))
                else:
                    parts.append(self._bounded_text(item.get("description", "")))
            self._append(self.console, {
                "type": self._bounded_text(params.get("type", "")),
                "text": self._bounded_text(" ".join(parts)),
            })
        elif method == "Runtime.exceptionThrown":
            detail = params.get("exceptionDetails", {})
            self._append(self.page_errors, self._bounded_text(
                detail.get("text") or detail.get("exception", {}).get(
                    "description", "JavaScript exception")))

    def call(self, method: str, params: dict | None = None,
             timeout: float = 4.0) -> dict:
        call_id = self._send(method, params)
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail(f"CDP call timed out: {method}")
            message = self.websocket.receive(remaining)
            self._event(message)
            if message.get("id") != call_id:
                continue
            if "error" in message:
                _fail(f"CDP call failed: {method}: {message['error']}")
            result = message.get("result", {})
            return result if isinstance(result, dict) else {}

    def wait_for_load(self, loader_id: str, wait_ms: int) -> None:
        deadline = time.monotonic() + LOAD_SECONDS
        while loader_id not in self.loaded_loader_ids:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                _fail("document load timed out")
            try:
                self._event(self.websocket.receive(min(remaining, 0.25)))
            except socket.timeout:
                continue
        settle = time.monotonic() + wait_ms / 1_000
        while time.monotonic() < settle:
            try:
                self._event(self.websocket.receive(
                    min(settle - time.monotonic(), 0.1)))
            except socket.timeout:
                continue


def _page_websocket(port: int) -> str:
    deadline = time.monotonic() + STARTUP_SECONDS
    while time.monotonic() < deadline:
        try:
            with urlopen(f"http://127.0.0.1:{port}/json/list", timeout=0.25) \
                    as response:
                raw = response.read(MAX_DISCOVERY_BYTES + 1)
            if len(raw) > MAX_DISCOVERY_BYTES:
                _fail("Chromium target discovery response is oversized")
            targets = json.loads(raw)
            for target in targets:
                if target.get("type") == "page" \
                        and isinstance(target.get("webSocketDebuggerUrl"), str):
                    endpoint = urlsplit(target["webSocketDebuggerUrl"])
                    if endpoint.hostname in {"127.0.0.1", "localhost"} \
                            and endpoint.port == port:
                        return target["webSocketDebuggerUrl"]
        except Exception:
            time.sleep(0.05)
    _fail("Chromium did not expose a page target")


def _isolated_selector_observation(
    cdp: _CDP,
    context_id: int,
    selector: str,
) -> dict:
    """Observe DOM text in a pristine isolated world.

    The selector is a CDP value argument, never executable source.  Native
    DOM methods and built-ins come from the isolated world, so a subject page
    cannot monkeypatch the observer's query or collection operations.
    """
    function = """function(selector, maxNodes, maxChars) {
        "use strict";
        const query = Document.prototype.querySelectorAll;
        const item = NodeList.prototype.item;
        const textGetter = Object.getOwnPropertyDescriptor(
            Node.prototype, "textContent").get;
        const nodes = Reflect.apply(query, document, [selector]);
        const limit = nodes.length < maxNodes ? nodes.length : maxNodes;
        const texts = [];
        let complete = nodes.length <= maxNodes;
        for (let index = 0; index < limit; index += 1) {
            const node = Reflect.apply(item, nodes, [index]);
            const raw = Reflect.apply(textGetter, node, []);
            const text = raw === null ? "" : raw;
            if (text.length > maxChars) complete = false;
            texts.push(text.slice(0, maxChars));
        }
        return {count: nodes.length, texts: texts,
                texts_complete: complete};
    }"""
    result = cdp.call("Runtime.callFunctionOn", {
        "functionDeclaration": function,
        "executionContextId": context_id,
        "arguments": [
            {"value": selector},
            {"value": MAX_SELECTOR_NODES},
            {"value": MAX_TEXT_CHARS},
        ],
        "returnByValue": True,
        "awaitPromise": False,
    })
    if result.get("exceptionDetails"):
        _fail("isolated browser observation failed")
    remote = result.get("result", {})
    value = remote.get("value")
    if not isinstance(value, dict) or set(value) != {
            "count", "texts", "texts_complete"}:
        _fail("isolated browser observation is invalid")
    if isinstance(value["count"], bool) or not isinstance(value["count"], int) \
            or value["count"] < 0 \
            or not isinstance(value["texts_complete"], bool) \
            or not isinstance(value["texts"], list) \
            or len(value["texts"]) > MAX_SELECTOR_NODES \
            or any(not isinstance(text, str) or len(text) > MAX_TEXT_CHARS
                   for text in value["texts"]):
        _fail("isolated browser observation has invalid values")
    return value


def _observe(request: dict) -> dict:
    _DocumentHandler.document = request["document"]
    server = ThreadingHTTPServer(("127.0.0.1", 0), _DocumentHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]
    document_url = f"http://127.0.0.1:{port}/document"
    favicon_url = f"http://127.0.0.1:{port}/favicon.ico"
    profile = os.path.join(os.getcwd(), "chrome-profile")
    os.mkdir(profile)
    browser = None
    websocket = None
    try:
        if _sha256_file(request["browser_path"]) != request["browser_sha256"]:
            _fail("browser executable identity changed before launch")
        browser = subprocess.Popen([
            request["browser_path"],
            "--headless=new", "--disable-gpu", "--no-first-run",
            "--no-default-browser-check", "--disable-background-networking",
            "--disable-component-update", "--disable-sync",
            "--disable-quic", "--metrics-recording-only", "--mute-audio",
            "--disable-extensions", "--disable-default-apps", "--no-pings",
            "--disable-breakpad", "--disable-client-side-phishing-detection",
            "--proxy-server=socks5://127.0.0.1:9",
            f"--proxy-bypass-list=127.0.0.1:{port}",
            "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
            "--remote-allow-origins=*", "--remote-debugging-port=0",
            "--remote-debugging-address=127.0.0.1",
            "--host-resolver-rules=MAP * ~NOTFOUND, EXCLUDE 127.0.0.1",
            "--lang=en-US", "--force-device-scale-factor=1",
            "--window-size=1280,720", "--force-color-profile=srgb",
            f"--user-data-dir={profile}", "about:blank",
        ], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
           stderr=subprocess.DEVNULL, close_fds=True)
        active = os.path.join(profile, "DevToolsActivePort")
        deadline = time.monotonic() + STARTUP_SECONDS
        while not os.path.isfile(active) and time.monotonic() < deadline:
            if browser.poll() is not None:
                _fail(f"Chromium exited during startup ({browser.returncode})")
            time.sleep(0.05)
        if not os.path.isfile(active):
            _fail("Chromium startup timed out")
        with open(active, encoding="utf-8") as handle:
            debug_port = int(handle.readline().strip())
        websocket = _WebSocket(_page_websocket(debug_port))
        cdp = _CDP(websocket, document_url, favicon_url)
        cdp.call("Runtime.enable")
        cdp.call("Page.enable")
        cdp.call("Emulation.setDeviceMetricsOverride", {
            "width": 1280, "height": 720, "deviceScaleFactor": 1,
            "mobile": False, "screenWidth": 1280, "screenHeight": 720,
        })
        cdp.call("Emulation.setTimezoneOverride", {"timezoneId": "UTC"})
        cdp.call("Emulation.setLocaleOverride", {"locale": "en-US"})
        cdp.call("Emulation.setEmulatedMedia", {
            "media": "screen",
            "features": [
                {"name": "prefers-color-scheme", "value": "light"},
                {"name": "prefers-reduced-motion",
                 "value": "no-preference"},
            ],
        })
        cdp.call("Page.setLifecycleEventsEnabled", {"enabled": True})
        cdp.call("Fetch.enable", {"patterns": [{"urlPattern": "*",
                                                  "requestStage": "Request"}]})
        version = cdp.call("Browser.getVersion")
        navigation = cdp.call("Page.navigate", {"url": document_url})
        if navigation.get("errorText"):
            _fail(f"document navigation failed: {navigation['errorText']}")
        loader_id = navigation.get("loaderId")
        if not isinstance(loader_id, str) or not loader_id:
            _fail("document navigation omitted its loader identity")
        cdp.wait_for_load(loader_id, request["wait_ms"])
        cdp.call("Page.setWebLifecycleState", {"state": "frozen"})
        cdp.capture_open = False
        frame_tree = cdp.call("Page.getFrameTree").get("frameTree", {})
        frame = frame_tree.get("frame", {})
        final_url = frame.get("url")
        if final_url != document_url:
            _fail("document navigated outside its isolated origin")
        if frame.get("loaderId") != loader_id:
            _fail("document loader changed during the observation window")
        frame_id = frame.get("id")
        if not isinstance(frame_id, str) or not frame_id:
            _fail("document frame identity is missing")
        isolated = cdp.call("Page.createIsolatedWorld", {
            "frameId": frame_id,
            "worldName": "gv-browser-observer-v1",
            "grantUniveralAccess": False,
        })
        context_id = isolated.get("executionContextId")
        if isinstance(context_id, bool) or not isinstance(context_id, int):
            _fail("isolated browser world identity is missing")

        selector_results = {}
        for selector in request["selectors"]:
            value = _isolated_selector_observation(
                cdp, context_id, selector)
            encoded_size = len(json.dumps(
                value, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8"))
            if cdp.observation_bytes + encoded_size > MAX_OBSERVATION_BYTES:
                _fail("browser observations exceed total size limit")
            cdp.observation_bytes += encoded_size
            selector_results[selector] = value
        product = version.get("product", "Chromium/unknown")
        engine, _, engine_version = product.partition("/")
        result = {
            "schema": RESPONSE_SCHEMA,
            "document_sha256": request["document_sha256"],
            "browser_sha256": request["browser_sha256"],
            "network_policy": NETWORK_POLICY,
            "document_policy": DOCUMENT_POLICY,
            "observation_profile": OBSERVATION_PROFILE,
            "observation_window": {
                "start": "navigation",
                "end": "page-frozen-after-load+wait",
                "wait_ms": request["wait_ms"],
            },
            "engine": engine,
            "engine_version": engine_version,
            "dialogs": cdp.dialogs,
            "console": cdp.console,
            "page_errors": cdp.page_errors,
            "blocked_requests": cdp.blocked_requests,
            "selectors": selector_results,
        }
        if len(json.dumps(
                result, ensure_ascii=False, sort_keys=True,
                separators=(",", ":")).encode("utf-8")) > MAX_RESPONSE_BYTES:
            _fail("browser response exceeds size limit")
        return result
    finally:
        if websocket is not None:
            websocket.close()
        if browser is not None:
            browser.terminate()
            try:
                browser.wait(timeout=2)
            except subprocess.TimeoutExpired:
                browser.kill()
                browser.wait(timeout=2)
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def main() -> int:
    try:
        result = _observe(_request())
        sys.stdout.write(json.dumps(result, ensure_ascii=False, sort_keys=True))
        return 0
    except Exception as error:
        sys.stderr.write(f"{type(error).__name__}: {error}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
