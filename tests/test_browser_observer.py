"""The optional browser seam is explicit, pinned, and bounded."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import struct
import time

import pytest

from verifier.execution.browser import (
    BROWSER_DOCUMENT_POLICY,
    BROWSER_NETWORK_POLICY,
    BROWSER_OBSERVATION_PROFILE,
    MAX_BROWSER_RESPONSE_BYTES,
    ProcessBrowserObserver,
    configured_browser_observer,
)


def _executable(path, source):
    path.write_text(source)
    path.chmod(0o700)
    return path, hashlib.sha256(path.read_bytes()).hexdigest()


def _runner(tmp_path, source=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    observer = _executable(
        tmp_path / "observer",
        source or "#!/bin/sh\nprintf '{\"observed\":true}'\n",
    )
    browser = _executable(
        tmp_path / "browser",
        "#!/bin/sh\nexit 0\n",
    )
    return observer, browser


def _configured(observer, browser):
    path, digest = observer
    browser_path, browser_digest = browser
    return configured_browser_observer({
        "GV_BROWSER_OBSERVER": os.fspath(path),
        "GV_BROWSER_OBSERVER_SHA256": digest,
        "GV_BROWSER_EXECUTABLE": os.fspath(browser_path),
        "GV_BROWSER_EXECUTABLE_SHA256": browser_digest,
    })


def _observer_module():
    script = Path(__file__).parents[1] / "scripts" / "gv_browser_observer.py"
    spec = importlib.util.spec_from_file_location("gv_browser_observer", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_browser_observer_requires_an_explicit_complete_configuration():
    assert configured_browser_observer({}) is None
    with pytest.raises(ValueError, match="must be set together"):
        configured_browser_observer({"GV_BROWSER_OBSERVER": "/some/path"})
    with pytest.raises(ValueError, match="must be set together"):
        configured_browser_observer({"GV_BROWSER_OBSERVER_SHA256": "0" * 64})
    with pytest.raises(ValueError, match="must be set together"):
        configured_browser_observer({
            "GV_BROWSER_OBSERVER": "/observer",
            "GV_BROWSER_OBSERVER_SHA256": "0" * 64,
            "GV_BROWSER_EXECUTABLE": "/browser",
        })


def test_process_browser_observer_pins_identity_and_uses_json_protocol(tmp_path):
    runner, browser = _runner(tmp_path)
    observer = _configured(runner, browser)

    assert observer.identity == {
        "kind": "pinned-process",
        "protocol": "gv-browser-observer-request-v1",
        "observer_sha256": runner[1],
        "browser_sha256": browser[1],
        "network_policy": BROWSER_NETWORK_POLICY,
        "document_policy": BROWSER_DOCUMENT_POLICY,
        "observation_profile": BROWSER_OBSERVATION_PROFILE,
    }
    assert observer.observe(
        b"<p>portable</p>", {"wait_ms": 0, "selectors": []}) == {
        "observed": True}

    # Callers cannot mutate the catalogued nested profile through identity.
    observer.identity["observation_profile"]["locale"] = "changed"
    assert observer.identity["observation_profile"]["locale"] == "en-US"


def test_process_browser_observer_refuses_wrong_or_changed_identity(tmp_path):
    (path, digest), browser = _runner(tmp_path)
    with pytest.raises(ValueError, match="SHA-256"):
        ProcessBrowserObserver(
            os.fspath(path), "0" * 64,
            os.fspath(browser[0]), browser[1])

    observer = ProcessBrowserObserver(
        os.fspath(path), digest, os.fspath(browser[0]), browser[1])
    path.write_text("#!/bin/sh\nprintf '{\"changed\":true}'\n")
    with pytest.raises(RuntimeError, match="identity changed"):
        observer.observe(
            b"<p>portable</p>", {"wait_ms": 0, "selectors": []})

    runner, browser = _runner(tmp_path / "second")
    observer = _configured(runner, browser)
    browser[0].write_text("#!/bin/sh\nprintf changed\n")
    with pytest.raises(RuntimeError, match="browser executable identity changed"):
        observer.observe(
            b"<p>portable</p>", {"wait_ms": 0, "selectors": []})


def test_process_observer_rejects_unknown_request_surface(tmp_path):
    observer = _configured(*_runner(tmp_path))
    with pytest.raises(RuntimeError, match="wrong shape"):
        observer.observe(b"<p>x</p>", {
            "wait_ms": 0, "selectors": [], "script": "subject code",
        })


def test_process_observer_stops_before_buffering_unbounded_output(tmp_path):
    source = (
        "#!/usr/bin/env python3\n"
        "import sys\n"
        f"sys.stdout.write('x' * {MAX_BROWSER_RESPONSE_BYTES + 1})\n"
    )
    observer = _configured(*_runner(tmp_path, source))
    with pytest.raises(RuntimeError, match="stdout exceeds the size limit"):
        observer.observe(
            b"<p>bounded</p>", {"wait_ms": 0, "selectors": []})


def test_process_observer_timeout_covers_a_nonreading_child(tmp_path):
    runner, browser = _runner(
        tmp_path, "#!/bin/sh\nsleep 30 &\nwait\n")
    observer = ProcessBrowserObserver(
        os.fspath(runner[0]), runner[1],
        os.fspath(browser[0]), browser[1], timeout_seconds=0.1)

    started = time.monotonic()
    with pytest.raises(RuntimeError, match="timed out"):
        observer.observe(
            b"<p>timeout</p>", {"wait_ms": 0, "selectors": []})
    assert time.monotonic() - started < 3


def test_selector_is_a_value_argument_in_an_isolated_context():
    module = _observer_module()

    class FakeCDP:
        params = None

        def call(self, method, params):
            assert method == "Runtime.callFunctionOn"
            self.params = params
            return {"result": {"value": {
                "count": 1, "texts": ["safe"], "texts_complete": True,
            }}}

    cdp = FakeCDP()
    selector = "[data-value='not executable']"
    assert module._isolated_selector_observation(cdp, 17, selector) == {
        "count": 1, "texts": ["safe"], "texts_complete": True,
    }
    assert cdp.params["executionContextId"] == 17
    assert cdp.params["arguments"][0] == {"value": selector}
    assert selector not in cdp.params["functionDeclaration"]


def test_observer_script_matches_the_public_profile_and_bounds_cdp_frames():
    module = _observer_module()
    assert module.NETWORK_POLICY == BROWSER_NETWORK_POLICY
    assert module.DOCUMENT_POLICY == BROWSER_DOCUMENT_POLICY
    assert module.OBSERVATION_PROFILE == BROWSER_OBSERVATION_PROFILE

    class NoReadSocket:
        def settimeout(self, _timeout):
            pass

        def recv(self, _count):
            raise AssertionError("oversized frame body must not be read")

    websocket = object.__new__(module._WebSocket)
    websocket.socket = NoReadSocket()
    websocket.buffer = b"\x81\x7f" + struct.pack(
        "!Q", module.MAX_CDP_MESSAGE_BYTES + 1)
    with pytest.raises(RuntimeError, match="CDP message exceeds size limit"):
        websocket.receive(1)
