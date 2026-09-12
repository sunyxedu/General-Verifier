"""Opt-in integration tests for the pinned Chromium HTML adapter.

Set ``GV_TEST_CHROMIUM`` to an absolute Chromium-family executable path.
The fixtures are standalone synthetic documents and never use the network.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_WORLD, OUTCOME_HOLDS, Obligation
from verifier.execution.adapters import (
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.browser import ProcessBrowserObserver
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore


ROW = {"id": "browser-integration", "kind": "final",
       "text": "the synthetic browser observation is complete"}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                return digest.hexdigest()
            digest.update(chunk)


@pytest.fixture(scope="module")
def browser_observer():
    configured = os.environ.get("GV_TEST_CHROMIUM")
    if not configured:
        pytest.skip("set GV_TEST_CHROMIUM to run Chromium integration tests")
    browser = Path(configured)
    if not browser.is_absolute():
        pytest.fail("GV_TEST_CHROMIUM must be an absolute path")
    if not browser.is_file() or not os.access(browser, os.X_OK):
        pytest.fail("GV_TEST_CHROMIUM must name an executable regular file")

    observer = Path(__file__).parents[1] / "scripts" / "gv_browser_observer.py"
    return ProcessBrowserObserver(
        os.fspath(observer), _sha256(observer),
        os.fspath(browser), _sha256(browser),
        timeout_seconds=20,
    )


class _Fallback:
    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        artifact = self.artifacts.put({"instrument": "integration-fallback"})
        return DischargeResult(
            outcome=OUTCOME_HOLDS,
            witness=None,
            artifact=artifact,
            instrument="integration-fallback",
            witness_kind="execution",
        )


def _method(address, assertions, *, wait_ms=50):
    return {
        "procedure": "observe the standalone HTML with the registered browser",
        "artifact": "a content-addressed browser observation",
        "level": "deterministic",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": [],
        "adapter": "browser.html",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": address,
            "assertions": assertions,
            "wait_ms": wait_ms,
        }, ensure_ascii=False, sort_keys=True),
        "adapter_inputs": [{
            "ref": address,
            "provenance": "delivered",
            "role": "subject",
        }],
    }


def _routing(tmp_path, observer, html: bytes, *, live: bool):
    state = tmp_path / ("live-state" if live else "captured-state")
    state.mkdir()
    if live:
        target = tmp_path / "live-page.html"
        address = os.fspath(target)
    else:
        world = state / "world" / "app"
        world.mkdir(parents=True)
        (state / "world_manifest.json").write_text(json.dumps({
            "scheme": "filesystem",
            "captured_roots": ["/app"],
        }))
        target = world / "page.html"
        address = "/app/page.html"
    target.write_bytes(html)

    artifacts = ArtifactStore(str(state / "artifacts"))
    fallback = _Fallback(artifacts)
    routing = AdapterRoutingDischarger(
        fallback,
        default_adapter_registry(observer),
        artifacts,
        EventLog(str(state / "events.jsonl")),
        str(state),
        live_world=live,
    )
    return artifacts, fallback, routing, address


def _browser_evidence(artifacts, result):
    return artifacts.get(result.artifact)["execution"]["observation"]["browser"]


def test_unicode_attribute_selector_has_live_captured_parity(
        tmp_path, browser_observer):
    html = (
        '<!doctype html><meta charset="utf-8">'
        '<section data-label="café">你好 · café</section>'
    ).encode()
    observations = {}
    for live in (False, True):
        root = tmp_path / ("live" if live else "captured")
        root.mkdir()
        artifacts, fallback, routing, address = _routing(
            root, browser_observer, html, live=live)
        result = routing.discharge(
            ROW,
            Obligation("dom", "the frozen DOM contains the Unicode label",
                       BASIS_WORLD),
            _method(address, [{
                "kind": "selector.texts",
                "selector": '[data-label="café"]',
                "expected": ["你好 · café"],
            }]),
        )

        assert result.outcome == OUTCOME_HOLDS
        assert result.instrument == "adapter:browser.html@1"
        assert fallback.calls == []
        observations[live] = _browser_evidence(artifacts, result)

    assert observations[False] == observations[True]


def test_dialog_and_console_are_bound_to_the_finite_window(
        tmp_path, browser_observer):
    html = (
        "<!doctype html><script>"
        "console.info('finite-ready');"
        "alert('finite-dialog');"
        "</script><main>done</main>"
    ).encode()
    artifacts, fallback, routing, address = _routing(
        tmp_path, browser_observer, html, live=False)
    result = routing.discharge(
        ROW,
        Obligation("events", "the finite browser events are observed",
                   BASIS_WORLD),
        _method(address, [
            {"kind": "console.messages", "level": "info",
             "expected": ["finite-ready"]},
            {"kind": "dialog.messages", "expected": ["finite-dialog"]},
        ], wait_ms=100),
    )

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    browser = _browser_evidence(artifacts, result)
    assert browser["observation_window"] == {
        "start": "navigation",
        "end": "page-frozen-after-load+wait",
        "wait_ms": 100,
    }
    assert browser["blocked_requests"] == []


def test_linked_missing_resource_fails_closed_to_fallback(
        tmp_path, browser_observer):
    html = (
        '<!doctype html><script src="missing.js"></script>'
        '<main data-state="ready">fallback</main>'
    ).encode()
    _, fallback, routing, address = _routing(
        tmp_path, browser_observer, html, live=False)
    result = routing.discharge(
        ROW,
        Obligation("resource", "the linked document is fully observed",
                   BASIS_WORLD),
        _method(address, [{
            "kind": "selector.count",
            "selector": '[data-state="ready"]',
            "expected": 1,
        }]),
    )

    assert result.instrument == "integration-fallback"
    assert len(fallback.calls) == 1
    assert "undeclared external resources" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]
