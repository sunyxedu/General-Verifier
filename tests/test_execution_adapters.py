"""Typed adapter execution stays a fail-safe optimization of BindingGame."""
from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from verifier.core.events import EventLog
from verifier.core.model import (
    BASIS_RECORD,
    BASIS_WORLD,
    OUTCOME_HOLDS,
    OUTCOME_REFUTED,
    WITNESS_ABSENCE,
    WITNESS_NONE,
    WITNESS_RECORD_CONTRADICTION,
    WITNESS_WORLD_STATE,
    Obligation,
)
from verifier.engine.game import BindingGame
from verifier.engine.method_cache import AcceptedMethodCache
from verifier.execution.adapters import (
    EVIDENCE_SCHEMA,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRoutingDischarger,
    default_adapter_registry,
)
from verifier.execution.browser import (
    BROWSER_DOCUMENT_POLICY, BROWSER_NETWORK_POLICY,
    BROWSER_OBSERVATION_PROFILE, BROWSER_RESPONSE_SCHEMA,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service, _default_kernel


ROW = {"id": "c1", "kind": "final", "text": "the task is complete"}
JUDGE_YES = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "the adapter artifact matches the method",
}


def method(name, arguments, *, basis=BASIS_WORLD, inputs=None, version=1):
    return {
        "procedure": f"execute the exact registered method {name}",
        "artifact": "a content-addressed adapter evidence envelope",
        "level": "deterministic",
        "basis": basis,
        "strength": "decisive",
        "bridges": [],
        "adapter": name,
        "adapter_version": version,
        "adapter_arguments": json.dumps(arguments, sort_keys=True),
        "adapter_inputs": inputs or [],
    }


def subject(ref, provenance="delivered"):
    return {"ref": ref, "provenance": provenance, "role": "subject"}


class FallbackDischarger:
    def __init__(self, artifacts, outcome=OUTCOME_HOLDS):
        self.artifacts = artifacts
        self.outcome = outcome
        self.calls = []

    def discharge(self, row, obligation, method=None, tag=""):
        self.calls.append({"method": method, "tag": tag})
        digest = self.artifacts.put({"instrument": "fallback", "tag": tag})
        return DischargeResult(
            outcome=self.outcome, witness=None, artifact=digest,
            instrument="fallback", witness_kind="execution")


def captured(tmp_path, browser_observer=None):
    state = tmp_path / "gv"
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem",
        "captured_roots": ["/app"],
    }))
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(browser_observer), artifacts, log,
        str(state),
        live_world=False,
    )
    return state, world, artifacts, log, fallback, routing


def captured_candidate(tmp_path, name, *, status=None):
    state = tmp_path / name
    world = state / "world" / "app"
    world.mkdir(parents=True)
    (state / "task.md").write_text("Produce result.json with completed status")
    (state / "world_manifest.json").write_text(json.dumps({
        "scheme": "filesystem",
        "captured_roots": ["/app"],
    }))
    if status is not None:
        (world / "result.json").write_text(json.dumps({"status": status}))
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False,
    )
    return state, world, artifacts, log, fallback, routing


def test_method_ir_is_optional_but_never_partially_accepted():
    assert AdapterInvocation.from_method({"procedure": "legacy"}) is None
    with pytest.raises(AdapterNotApplicable, match="partial"):
        AdapterInvocation.from_method({"adapter": "filesystem.exists"})
    with pytest.raises(AdapterNotApplicable, match="duplicate JSON key"):
        AdapterInvocation.from_method(method(
            "filesystem.exists", {"address": "/app/x"},
            inputs=[subject("/app/x")],
        ) | {"adapter_arguments": '{"address":"/app/x","address":"/app/y"}'})


def test_catalog_is_versioned_narrow_and_has_no_arbitrary_executor():
    catalog = json.loads(default_adapter_registry().catalog())
    assert {(item["name"], item["version"]) for item in catalog} == {
        ("filesystem.exists", 1),
        ("filesystem.sha256", 1),
        ("document.json-predicate", 1),
        ("document.json-predicate", 2),
        ("document.json-value", 1),
        ("document.text", 1),
        ("ledger.event", 1),
        ("ledger.tool-call-string", 1),
        ("ledger.tool-call-string", 2),
        ("ledger.tool-call-string", 3),
    }
    command = next(item for item in catalog
                   if item["name"] == "ledger.tool-call-string")
    text = next(item for item in catalog if item["name"] == "document.text")
    assert text["accepted_method_reuse"] is True
    assert command["basis"] == BASIS_RECORD
    assert "never parses or executes" in command["summary"]
    assert not any(item["basis"] == BASIS_WORLD
                   and "command" in item["name"] for item in catalog)
    assert not any(item["name"] == "browser.html" for item in catalog)


def test_file_exists_adapter_emits_auditable_world_evidence(tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    target = world / "result.json"
    target.write_text('{"status":"completed"}')
    obligation = Obligation(
        "c1", "the result artifact exists", BASIS_WORLD,
        refutable_by_absence=True)
    result = routing.discharge(
        ROW, obligation,
        method("filesystem.exists", {"address": "/app/result.json",
                                      "kind": "file"},
               inputs=[subject("/app/result.json")]),
    )

    assert result.outcome == OUTCOME_HOLDS
    assert result.witness_kind == WITNESS_NONE
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    assert evidence["schema"] == EVIDENCE_SCHEMA
    assert evidence["instrument"] == "adapter:filesystem.exists@1"
    assert evidence["execution"]["observation"]["actual_kind"] == "file"
    events = list(log.replay())
    assert events[-1]["type"] == "adapter_discharge"


def test_absence_refutes_only_an_explicit_presence_requirement(tmp_path):
    _, _, _, _, fallback, routing = captured(tmp_path)
    typed = method(
        "filesystem.exists", {"address": "/app/missing", "kind": "file"},
        inputs=[subject("/app/missing")])

    refuted = routing.discharge(
        ROW, Obligation("c1", "the file exists", BASIS_WORLD,
                        refutable_by_absence=True), typed)
    fallback_result = routing.discharge(
        ROW, Obligation("c1", "the file has a property", BASIS_WORLD), typed)

    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert fallback_result.instrument == "fallback"
    assert len(fallback.calls) == 1


@pytest.mark.parametrize("actual_kind", ["file", "directory", "symlink", "absent"])
@pytest.mark.parametrize("expected_kind", ["any", "file", "directory"])
def test_file_exists_observation_has_the_same_kind_contract_for_every_outcome(
        tmp_path, actual_kind, expected_kind):
    _, world, artifacts, _, fallback, routing = captured(tmp_path)
    target = world / "subject"
    if actual_kind == "file":
        target.write_text("synthetic document")
    elif actual_kind == "directory":
        target.mkdir()
    elif actual_kind == "symlink":
        destination = world / "destination"
        destination.write_text("synthetic document")
        target.symlink_to(destination)
    result = routing.discharge(
        ROW,
        Obligation("c1", "the declared artifact is present", BASIS_WORLD,
                   refutable_by_absence=True),
        method("filesystem.exists", {"address": "/app/subject",
                                      "kind": expected_kind},
               inputs=[subject("/app/subject")]),
    )

    assert fallback.calls == []
    observation = artifacts.get(result.artifact)["execution"]["observation"]
    assert observation["exists"] is (actual_kind != "absent")
    assert observation["actual_kind"] == actual_kind
    assert observation["expected_kind"] == expected_kind
    assert observation["world_mode"] == "captured"


def test_captured_path_escape_and_wrong_basis_fall_back(tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    escape = method(
        "filesystem.exists", {"address": "/app/../../etc/passwd"},
        inputs=[subject("/app/../../etc/passwd")])
    wrong_basis = method(
        "filesystem.exists", {"address": "/app/result.json"},
        basis=BASIS_RECORD, inputs=[subject("/app/result.json")])
    obligation = Obligation("c1", "a condition", BASIS_RECORD)

    assert routing.discharge(ROW, obligation, escape).instrument == "fallback"
    assert routing.discharge(ROW, obligation, wrong_basis).instrument == "fallback"
    assert len(fallback.calls) == 2
    fallback_events = [e for e in log.replay()
                       if e["type"] == "adapter_fallback"]
    assert "escapes" in fallback_events[0]["payload"]["reason"]
    assert "basis=world" in fallback_events[1]["payload"]["reason"]


def test_unknown_adapter_bad_version_and_missing_subject_fall_back(tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    obligation = Obligation("c1", "the file exists", BASIS_WORLD,
                            refutable_by_absence=True)
    candidates = [
        method("unknown.adapter", {}, inputs=[subject("/app/x")]),
        method("filesystem.exists", {"address": "/app/x"}, version=2,
               inputs=[subject("/app/x")]),
        method("filesystem.exists", {"address": "/app/x"}),
    ]

    for candidate in candidates:
        assert routing.discharge(ROW, obligation, candidate).instrument \
            == "fallback"
    assert len(fallback.calls) == 3
    assert len([e for e in log.replay()
                if e["type"] == "adapter_fallback"]) == 3


def test_sha256_adapter_records_the_consumed_content_digest(tmp_path):
    _, world, artifacts, _, fallback, routing = captured(tmp_path)
    target = world / "answer.bin"
    target.write_bytes(b"answer\x00bytes")
    import hashlib
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    obligation = Obligation("c1", "the artifact has the required digest",
                            BASIS_WORLD)

    holds = routing.discharge(
        ROW, obligation,
        method("filesystem.sha256", {
            "address": "/app/answer.bin", "expected_sha256": expected,
        }, inputs=[subject("/app/answer.bin")]))
    refuted = routing.discharge(
        ROW, obligation,
        method("filesystem.sha256", {
            "address": "/app/answer.bin", "expected_sha256": "0" * 64,
        }, inputs=[subject("/app/answer.bin")]))

    assert holds.outcome == OUTCOME_HOLDS
    evidence = artifacts.get(holds.artifact)
    assert evidence["execution"]["consumed"][0]["sha256"] == expected
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_WORLD_STATE
    assert fallback.calls == []


def test_json_value_uses_strict_json_and_type_sensitive_equality(tmp_path):
    _, world, _, _, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text(
        '{"status":"completed","flag":true}')
    obligation = Obligation("c1", "the exact JSON value matches", BASIS_WORLD)

    holds = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")]))
    bool_is_not_one = routing.discharge(
        ROW, obligation,
        method("document.json-value", {
            "address": "/app/result.json", "pointer": "/flag",
            "expected": 1,
        }, inputs=[subject("/app/result.json")]))

    assert holds.outcome == OUTCOME_HOLDS
    assert bool_is_not_one.outcome == OUTCOME_REFUTED
    assert fallback.calls == []


@pytest.mark.parametrize("predicate,expected,matched", [
    ("equals", "alpha\nflag{general}\nomega\n", True),
    ("contains", "flag{general}", True),
    ("starts_with", "alpha\n", True),
    ("ends_with", "omega\n", True),
    ("contains", "flag{different}", False),
])
def test_text_adapter_applies_exact_utf8_literal_predicates(
    tmp_path, predicate, expected, matched,
):
    _, world, artifacts, _, fallback, routing = captured(tmp_path)
    target = world / "answer.txt"
    target.write_text("alpha\nflag{general}\nomega\n")
    result = routing.discharge(
        ROW, Obligation("c1", "the exact text predicate holds", BASIS_WORLD),
        method("document.text", {
            "address": "/app/answer.txt", "predicate": predicate,
            "expected": expected,
        }, inputs=[subject("/app/answer.txt")]))

    assert result.outcome == (OUTCOME_HOLDS if matched else OUTCOME_REFUTED)
    assert result.witness_kind == (
        WITNESS_NONE if matched else WITNESS_WORLD_STATE)
    evidence = artifacts.get(result.artifact)
    observation = evidence["execution"]["observation"]
    assert observation["matched"] is matched
    assert observation["identity"]["sha256"]
    assert "actual" not in observation
    assert fallback.calls == []


def test_text_adapter_is_strict_and_absence_aware(tmp_path):
    _, world, _, _, fallback, routing = captured(tmp_path)
    (world / "binary.txt").write_bytes(b"\xff")
    obligation = Obligation(
        "c1", "the required literal text exists", BASIS_WORLD,
        refutable_by_absence=True)

    invalid_utf8 = routing.discharge(
        ROW, obligation,
        method("document.text", {
            "address": "/app/binary.txt", "predicate": "contains",
            "expected": "required",
        }, inputs=[subject("/app/binary.txt")]))
    absent = routing.discharge(
        ROW, obligation,
        method("document.text", {
            "address": "/app/missing.txt", "predicate": "contains",
            "expected": "required",
        }, inputs=[subject("/app/missing.txt")]))

    assert invalid_utf8.outcome == OUTCOME_REFUTED
    assert invalid_utf8.witness_kind == WITNESS_WORLD_STATE
    assert absent.outcome == OUTCOME_REFUTED
    assert absent.witness_kind == WITNESS_ABSENCE
    assert fallback.calls == []


@pytest.mark.parametrize("arguments", [
    {"address": "/app/answer.txt", "predicate": "regex", "expected": ".*"},
    {"address": "/app/answer.txt", "predicate": "contains", "expected": 1},
    {"address": "/app/answer.txt", "predicate": "contains", "expected": ""},
    {"address": "/app/answer.txt", "predicate": "contains",
     "expected": "x" * 16_385},
])
def test_text_adapter_rejects_executable_or_unbounded_arguments(
    tmp_path, arguments,
):
    _, world, _, _, fallback, routing = captured(tmp_path)
    (world / "answer.txt").write_text("text")
    result = routing.discharge(
        ROW, Obligation("c1", "text check", BASIS_WORLD),
        method("document.text", arguments,
               inputs=[subject("/app/answer.txt")]))
    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_live_world_adapter_reads_the_original_absolute_address(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    target = tmp_path / "delivery" / "result.json"
    target.parent.mkdir()
    target.write_text('{"status":"completed"}')
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=True)
    typed = method(
        "document.json-value", {
            "address": str(target), "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject(str(target))])

    result = routing.discharge(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD), typed)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_ledger_adapter_matches_an_immutable_event_subset(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    event = log.append("tool_result", {
        "tool": "example", "result": {"exit": 0, "stdout": "ok"}})
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    obligation = Obligation("c1", "the retained action succeeded", BASIS_RECORD)
    declared = [subject(f"event:{event['seq']}", "ledger")]

    holds = routing.discharge(
        ROW, obligation,
        method("ledger.event", {
            "event_seq": event["seq"], "event_type": "tool_result",
            "payload_contains": {"result": {"exit": 0}},
        }, basis=BASIS_RECORD, inputs=declared))
    refuted = routing.discharge(
        ROW, obligation,
        method("ledger.event", {
            "event_seq": event["seq"],
            "payload_contains": {"result": {"exit": 1}},
        }, basis=BASIS_RECORD, inputs=declared))

    assert holds.outcome == OUTCOME_HOLDS
    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_RECORD_CONTRADICTION
    assert fallback.calls == []


@pytest.mark.parametrize("selector,expected,outcome", [
    ("first", "alpha --one", OUTCOME_HOLDS),
    ("last", "alpha --two", OUTCOME_HOLDS),
    ("any", "alpha --two", OUTCOME_HOLDS),
    ("all", "alpha --one", OUTCOME_REFUTED),
])
def test_ledger_string_uses_declared_dialect_and_exact_selection(
        tmp_path, selector, expected, outcome):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    log.append("tool_call", {
        "tool": "arbitrary-runner",
        "input": {"request": {"line": "alpha --one"}},
    })
    log.append("tool_call", {
        "tool": "ignored-runner",
        "input": {"request": {"line": "not selected"}},
    })
    log.append("tool_call", {
        "tool": "arbitrary-runner",
        "input": {"request": {"line": "alpha --two"}},
    })
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    typed = method(
        "ledger.tool-call-string", {
            "selector": selector,
            "value_pointer": "/input/request/line",
            "through_seq": 3,
            "tool_names": ["arbitrary-runner"],
            "expected": expected,
        }, basis=BASIS_RECORD, inputs=[subject("ledger:events", "ledger")])

    result = routing.discharge(
        ROW, Obligation("c1", "the selected command matches", BASIS_RECORD),
        typed)

    assert result.outcome == outcome
    assert fallback.calls == []
    evidence = artifacts.get(result.artifact)
    execution = evidence["execution"]
    assert execution["consumed"][0]["ref"] == "ledger:events"
    assert execution["observation"]["snapshot"]["candidate_string_count"] == 2
    if outcome == OUTCOME_REFUTED:
        assert result.witness_kind == WITNESS_RECORD_CONTRADICTION


def test_ledger_string_absence_only_refutes_a_presence_requirement(tmp_path):
    state = tmp_path / "gv"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    log.append("message", {"text": "no tool call"})
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=False)
    typed = method(
        "ledger.tool-call-string", {
            "selector": "any", "value_pointer": "/input/command",
            "through_seq": 1,
        }, basis=BASIS_RECORD, inputs=[subject("ledger:events", "ledger")])

    refuted = routing.discharge(
        ROW, Obligation("c1", "at least one command exists", BASIS_RECORD,
                        refutable_by_absence=True), typed)
    deferred = routing.discharge(
        ROW, Obligation("c2", "a command has some effect", BASIS_RECORD),
        typed)

    assert refuted.outcome == OUTCOME_REFUTED
    assert refuted.witness_kind == WITNESS_ABSENCE
    assert deferred.instrument == "fallback"
    assert len(fallback.calls) == 1


class FakeBrowserObserver:
    identity = {
        "kind": "fixture",
        "protocol": "gv-browser-observer-request-v1",
        "sha256": "f" * 64,
        "browser_sha256": "e" * 64,
        "observation_profile": BROWSER_OBSERVATION_PROFILE,
    }

    def __init__(self):
        self.calls = []

    def observe(self, document, request):
        import hashlib
        self.calls.append((document, dict(request)))
        return {
            "schema": BROWSER_RESPONSE_SCHEMA,
            "document_sha256": hashlib.sha256(document).hexdigest(),
            "network_policy": BROWSER_NETWORK_POLICY,
            "document_policy": BROWSER_DOCUMENT_POLICY,
            "browser_sha256": self.identity["browser_sha256"],
            "observation_profile": BROWSER_OBSERVATION_PROFILE,
            "observation_window": {"start": "navigation",
                                   "end": "page-frozen-after-load+wait",
                                   "wait_ms": request["wait_ms"]},
            "engine": "fixture-browser",
            "engine_version": "1",
            "dialogs": [{"type": "alert", "message": "hello"}],
            "console": [{"type": "log", "text": "ready"}],
            "page_errors": [],
            "blocked_requests": [],
            "selectors": {
                selector: {"count": 2, "texts": ["one", "two"],
                           "texts_complete": True}
                for selector in request["selectors"]
            },
        }


def test_browser_adapter_is_injected_and_has_no_oracle_script_surface(tmp_path):
    observer = FakeBrowserObserver()
    _, world, artifacts, _, fallback, routing = captured(
        tmp_path, browser_observer=observer)
    (world / "page.data").write_text(
        "<!doctype html><script>alert('hello')</script>")
    obligation = Obligation("c1", "the page has the expected observations",
                            BASIS_WORLD)
    assertions = [
        {"kind": "dialog.messages", "expected": ["hello"]},
        {"kind": "console.messages", "level": "log",
         "expected": ["ready"]},
        {"kind": "selector.count", "selector": ".item", "expected": 2},
        {"kind": "selector.texts", "selector": ".item",
         "expected": ["one", "two"]},
    ]
    typed = method(
        "browser.html", {"address": "/app/page.data",
                         "assertions": assertions, "wait_ms": 25},
        inputs=[subject("/app/page.data")])

    result = routing.discharge(ROW, obligation, typed)

    assert result.outcome == OUTCOME_HOLDS
    assert fallback.calls == []
    assert observer.calls[0][1] == {"wait_ms": 25,
                                    "selectors": [".item"]}
    catalog = json.loads(routing.adapter_catalog())
    browser = next(item for item in catalog if item["name"] == "browser.html")
    assert browser["backend"] == observer.identity
    assert "script" not in browser["arguments"]
    assert "command" not in browser["arguments"]
    evidence = artifacts.get(result.artifact)
    assert evidence["execution"]["observation"]["browser"]["engine"] \
        == "fixture-browser"


def test_browser_adapter_refutes_only_from_a_complete_bound_observation(
        tmp_path):
    observer = FakeBrowserObserver()
    _, world, _, _, fallback, routing = captured(
        tmp_path, browser_observer=observer)
    (world / "page.html").write_text("<!doctype html>")
    typed = method(
        "browser.html", {
            "address": "/app/page.html",
            "assertions": [{"kind": "dialog.messages", "expected": []}],
        }, inputs=[subject("/app/page.html")])

    result = routing.discharge(
        ROW, Obligation("c1", "the page opens no dialog", BASIS_WORLD), typed)

    assert result.outcome == OUTCOME_REFUTED
    assert result.witness_kind == "execution"
    assert fallback.calls == []


def test_browser_adapter_rejects_arbitrary_evaluation_arguments(tmp_path):
    observer = FakeBrowserObserver()
    _, world, _, _, fallback, routing = captured(
        tmp_path, browser_observer=observer)
    (world / "page.html").write_text("<!doctype html>")
    typed = method(
        "browser.html", {
            "address": "/app/page.html", "assertions": [
                {"kind": "dialog.messages", "expected": []}],
            "script": "globalThis.secret",
        }, inputs=[subject("/app/page.html")])

    result = routing.discharge(
        ROW, Obligation("c1", "an observation", BASIS_WORLD), typed)

    assert result.instrument == "fallback"
    assert observer.calls == []
    assert "unknown=script" in fallback.calls[0]["method"][
        "adapter_fallback_reason"]


@pytest.mark.parametrize("defect", [
    "resources", "document", "runtime", "profile", "window", "counts", "schema",
])
def test_browser_incomplete_or_misbound_observations_fall_back(tmp_path, defect):
    class DamagedObserver(FakeBrowserObserver):
        def observe(self, document, request):
            result = super().observe(document, request)
            if defect == "resources":
                result["blocked_requests"] = [
                    {"url": "https://assets.invalid/module.js",
                     "resource_type": "Script"}]
            elif defect == "document":
                result["document_sha256"] = "0" * 64
            elif defect == "runtime":
                result["browser_sha256"] = "0" * 64
            elif defect == "profile":
                result["observation_profile"] = {}
            elif defect == "window":
                result["observation_window"] = {
                    **result["observation_window"], "wait_ms": 1234}
            elif defect == "counts":
                result["selectors"]["p"]["count"] = 0
            else:
                result["extra"] = "unversioned extension"
            return result

    _, world, _, _, fallback, routing = captured(
        tmp_path, browser_observer=DamagedObserver())
    (world / "page.html").write_text("<p>one</p><p>two</p>")
    result = routing.discharge(
        ROW, Obligation("c1", "the snapshot contains two paragraphs", BASIS_WORLD),
        method("browser.html", {
            "address": "/app/page.html", "assertions": [
                {"kind": "selector.count", "selector": "p", "expected": 2}],
        }, inputs=[subject("/app/page.html")]))

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_non_reusable_adapter_keeps_fresh_judger_audits(tmp_path):
    observer = FakeBrowserObserver()
    state, world, artifacts, log, fallback, routing = captured(
        tmp_path, browser_observer=observer)
    (state / "task.md").write_text("Inspect the current document snapshot")
    (world / "page.html").write_text("<p>one</p><p>two</p>")
    typed = method("browser.html", {
        "address": "/app/page.html", "assertions": [
            {"kind": "selector.count", "selector": "p", "expected": 2}],
    }, inputs=[subject("/app/page.html")])
    cache = AcceptedMethodCache()
    backend = FixtureBackend({"oracles": {"methods": [typed]},
                              "judger": JUDGE_YES})
    game = BindingGame(backend, artifacts, log, routing, live_world=False,
                       method_cache=cache)
    obligation = Obligation("c1", "the snapshot contains two paragraphs", BASIS_WORLD)

    for _ in range(2):
        assert game.run(ROW, obligation).outcome == OUTCOME_HOLDS

    assert [call["role"] for call in backend.calls] == [
        "oracles", "judger", "oracles", "judger"]
    assert len(observer.calls) == 2
    assert len(cache) == 0
    assert fallback.calls == []


@pytest.mark.parametrize("assertion", [
    {"kind": "dialog.messages", "expected": ["hello", "later"]},
    {"kind": "console.messages", "expected": ["ready", "later"]},
    {"kind": "selector.count", "selector": "p", "expected": 3},
    {"kind": "selector.texts", "selector": "p", "expected": ["one", "two", "three"]},
])
@pytest.mark.parametrize("empty", [False, True])
def test_browser_missing_suffix_or_nodes_obey_absence_gate(tmp_path, assertion, empty):
    class Observer(FakeBrowserObserver):
        def observe(self, document, request):
            result = super().observe(document, request)
            if empty:
                result["dialogs"] = []
                result["console"] = []
                result["selectors"] = {
                    selector: {"count": 0, "texts": [], "texts_complete": True}
                    for selector in request["selectors"]}
            return result

    _, world, _, _, fallback, routing = captured(
        tmp_path, browser_observer=Observer())
    (world / "page.html").write_text("<p>one</p><p>two</p>")
    typed = method("browser.html", {
        "address": "/app/page.html", "assertions": [assertion],
    }, inputs=[subject("/app/page.html")])
    for allowed in (False, True):
        result = routing.discharge(
            ROW, Obligation("c1", "required observations exist", BASIS_WORLD,
                            refutable_by_absence=allowed), typed)
        if allowed:
            assert result.outcome == OUTCOME_REFUTED
            assert result.witness_kind == WITNESS_ABSENCE
        else:
            assert result.instrument == "fallback"
    assert len(fallback.calls) == 1


def test_binding_game_keeps_oracle_and_judger_but_skips_agent_discharge(
        tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    oracle_method = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    backend = FixtureBackend({
        "oracles": {"methods": [oracle_method]},
        "judger": JUDGE_YES,
    })
    game = BindingGame(
        backend, artifacts, log, routing, live_world=False)

    result = game.run(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD))

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]
    assert fallback.calls == []
    oracle_vars = backend.calls[0]["variables"]
    assert "document.json-value" in oracle_vars["adapter_catalog"]
    judge_vars = backend.calls[1]["variables"]
    assert "document.json-value" in judge_vars["adapter_method"]
    evidence = json.loads(judge_vars["evidence"])
    assert artifacts.get(evidence["artifact_digest"])["schema"] \
        == EVIDENCE_SCHEMA


def test_judger_redo_routes_to_agent_instead_of_repeating_adapter(tmp_path):
    _, world, artifacts, log, fallback, routing = captured(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    verdicts = iter([{
        "sufficient": "yes", "faithful": "no", "degradation": "n/a",
        "reason": "the execution needs an additional retained explanation",
    }, JUDGE_YES])
    backend = FixtureBackend({
        "oracles": {"methods": [typed]},
        "judger": lambda _: next(verdicts),
    })
    game = BindingGame(backend, artifacts, log, routing, live_world=False)

    result = game.run(
        ROW, Obligation("c1", "status equals completed", BASIS_WORLD))

    assert result.outcome == OUTCOME_HOLDS
    assert len(fallback.calls) == 1
    assert fallback.calls[0]["tag"] == "bound-redo"
    assert any(event["type"] == "adapter_fallback"
               for event in log.replay())


def test_legacy_oracle_method_is_byte_shape_compatible_and_uses_fallback(
        tmp_path):
    _, _, _, log, fallback, routing = captured(tmp_path)
    legacy = {
        "procedure": "perform the existing agent check",
        "artifact": "existing evidence", "level": "existing",
        "basis": BASIS_WORLD, "strength": "decisive", "bridges": [],
    }

    result = routing.discharge(
        ROW, Obligation("c1", "a semantic condition", BASIS_WORLD), legacy)

    assert result.instrument == "fallback"
    assert len(fallback.calls) == 1
    assert not any(event["type"] == "adapter_fallback"
                   for event in log.replay())


def test_default_online_and_offline_kernel_path_installs_adapter_router(
        tmp_path):
    service = Service(
        str(tmp_path / "gv"), kernel_factory=lambda _service: object())
    kernel = _default_kernel(
        service, model=None, backend=FixtureBackend({}), critic_rounds=0)

    assert isinstance(kernel.discharger, AdapterRoutingDischarger)
    assert kernel.binder.discharger is kernel.discharger
    assert "filesystem.exists" in kernel.discharger.adapter_catalog()
    # Adapter routing replaces only the executor. The production path must
    # still run a fresh Oracle and a fresh Judger for every obligation.
    assert kernel.binder.method_cache is None


def test_accepted_method_reuse_reexecutes_current_candidate_without_llm(
        tmp_path):
    cache = AcceptedMethodCache()
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])

    _, _, artifacts1, log1, fallback1, routing1 = captured_candidate(
        tmp_path, "candidate-1", status="completed")
    backend1 = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    game1 = BindingGame(
        backend1, artifacts1, log1, routing1, live_world=False,
        method_cache=cache)
    first = game1.run(
        ROW, Obligation("candidate-1", "status equals completed", BASIS_WORLD))

    row2 = {**ROW, "id": "c2", "text": "delivery finished successfully"}
    _, _, artifacts2, log2, fallback2, routing2 = captured_candidate(
        tmp_path, "candidate-2", status="wrong")
    backend2 = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    game2 = BindingGame(
        backend2, artifacts2, log2, routing2, live_world=False,
        method_cache=cache)
    second = game2.run(
        row2, Obligation("candidate-2", "status equals completed", BASIS_WORLD))

    assert first.outcome == OUTCOME_HOLDS
    assert second.outcome == OUTCOME_REFUTED
    assert len(cache) == 1
    assert [call["role"] for call in backend1.calls] == ["oracles", "judger"]
    assert [call["role"] for call in backend2.calls] == ["judger"]
    assert fallback1.calls == fallback2.calls == []
    first_bound = artifacts1.get(first.artifact)["bound"]
    second_bound = artifacts2.get(second.artifact)["bound"]
    assert first_bound["discharge_artifact"] \
        != second_bound["discharge_artifact"]
    assert artifacts2.get(second_bound["discharge_artifact"])["execution"][
        "observation"]["actual"] == "wrong"
    second_events = list(log2.replay())
    assert any(event["type"] == "method_cache_bypass"
               for event in second_events)
    assert not any(event["type"] == "method_cache_hit"
                   for event in second_events)

    # Each direction pays for one independent Judger signature. Once both
    # exist, every later candidate only re-executes the typed adapter.
    backends = [backend1, backend2]
    outcomes = [first.outcome, second.outcome]
    for index in range(3, 11):
        status = "completed" if index % 2 else "wrong"
        _, _, artifacts, log, fallback, routing = captured_candidate(
            tmp_path, f"candidate-{index}", status=status)
        backend = FixtureBackend({})
        game = BindingGame(
            backend, artifacts, log, routing, live_world=False,
            method_cache=cache)
        result = game.run(
            {**ROW, "id": f"c{index}", "text": f"candidate {index} done"},
            Obligation(f"candidate-{index}", "status equals completed",
                       BASIS_WORLD))
        backends.append(backend)
        outcomes.append(result.outcome)
        assert fallback.calls == []
    assert sum(len(backend.calls) for backend in backends) == 3
    assert outcomes.count(OUTCOME_HOLDS) == 5
    assert outcomes.count(OUTCOME_REFUTED) == 5


def test_text_method_reuse_rechecks_bytes_and_keeps_direction_signatures(
        tmp_path):
    cache = AcceptedMethodCache()
    typed = method(
        "document.text", {
            "address": "/app/result.json", "predicate": "contains",
            "expected": '"status": "completed"',
        }, inputs=[subject("/app/result.json")])
    cases = (("completed", OUTCOME_HOLDS), ("wrong", OUTCOME_REFUTED),
             ("completed", OUTCOME_HOLDS))
    observed = []
    for index, (status, expected) in enumerate(cases, 1):
        _, _, artifacts, log, fallback, routing = captured_candidate(
            tmp_path, f"text-candidate-{index}", status=status)
        backend = FixtureBackend(
            {"oracles": {"methods": [typed]}, "judger": JUDGE_YES}
            if index < 3 else {})
        game = BindingGame(
            backend, artifacts, log, routing, live_world=False,
            method_cache=cache)
        result = game.run(
            {**ROW, "id": f"text-c{index}"},
            Obligation(f"text-candidate-{index}",
                       "status text contains completed", BASIS_WORLD))
        assert result.outcome == expected
        assert fallback.calls == []
        observed.append((backend, list(log.replay())))

    assert [call["role"] for call in observed[0][0].calls] == [
        "oracles", "judger"]
    assert [call["role"] for call in observed[1][0].calls] == ["judger"]
    assert observed[2][0].calls == []
    assert any(event["type"] == "method_cache_hit"
               for event in observed[2][1])


def test_inapplicable_cached_method_falls_back_to_full_binding_game(tmp_path):
    cache = AcceptedMethodCache()
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    obligation1 = Obligation(
        "candidate-1", "status equals completed", BASIS_WORLD)
    _, _, artifacts1, log1, _, routing1 = captured_candidate(
        tmp_path, "candidate-1", status="completed")
    game1 = BindingGame(
        FixtureBackend({"oracles": {"methods": [typed]},
                        "judger": JUDGE_YES}),
        artifacts1, log1, routing1, live_world=False, method_cache=cache)
    assert game1.run(ROW, obligation1).outcome == OUTCOME_HOLDS

    legacy = {
        "procedure": "perform the existing agent check",
        "artifact": "existing evidence", "level": "existing",
        "basis": BASIS_WORLD, "strength": "decisive", "bridges": [],
    }
    _, _, artifacts2, log2, fallback2, routing2 = captured_candidate(
        tmp_path, "candidate-2")
    backend2 = FixtureBackend({
        "oracles": {"methods": [legacy]}, "judger": JUDGE_YES})
    game2 = BindingGame(
        backend2, artifacts2, log2, routing2, live_world=False,
        method_cache=cache)

    result = game2.run(
        {**ROW, "id": "c2"},
        Obligation("candidate-2", "status equals completed", BASIS_WORLD))

    assert result.outcome == OUTCOME_HOLDS
    assert [call["role"] for call in backend2.calls] == ["oracles", "judger"]
    assert len(fallback2.calls) == 1
    assert any(event["type"] == "method_cache_bypass"
               for event in log2.replay())


def test_accepted_method_cache_single_flights_parallel_candidates(tmp_path):
    cache = AcceptedMethodCache()
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    entered = threading.Event()
    release = threading.Event()

    def delayed_judger(_variables):
        entered.set()
        assert release.wait(5)
        return JUDGE_YES

    games = []
    for index in (1, 2):
        _, _, artifacts, log, _, routing = captured_candidate(
            tmp_path, f"candidate-{index}", status="completed")
        backend = FixtureBackend(
            {"oracles": {"methods": [typed]}, "judger": delayed_judger}
            if index == 1 else {})
        game = BindingGame(
            backend, artifacts, log, routing, live_world=False,
            method_cache=cache)
        games.append((game, backend))

    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(
            games[0][0].run, ROW,
            Obligation("candidate-1", "status equals completed", BASIS_WORLD))
        assert entered.wait(5)
        second = pool.submit(
            games[1][0].run, {**ROW, "id": "c2"},
            Obligation("candidate-2", "status equals completed", BASIS_WORLD))
        release.set()
        assert first.result(timeout=5).outcome == OUTCOME_HOLDS
        assert second.result(timeout=5).outcome == OUTCOME_HOLDS

    assert [call["role"] for call in games[0][1].calls] == [
        "oracles", "judger"]
    assert games[1][1].calls == []


def test_method_cache_key_keeps_different_task_authority_separate(tmp_path):
    cache = AcceptedMethodCache()
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    _, _, artifacts1, log1, _, routing1 = captured_candidate(
        tmp_path, "candidate-1", status="completed")
    first_backend = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    first_game = BindingGame(
        first_backend, artifacts1, log1, routing1, live_world=False,
        method_cache=cache)
    first_game.run(
        ROW, Obligation("candidate-1", "status equals completed", BASIS_WORLD))

    state2, _, artifacts2, log2, _, routing2 = captured_candidate(
        tmp_path, "candidate-2", status="completed")
    (state2 / "task.md").write_text("A different task with different authority")
    second_backend = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    second_game = BindingGame(
        second_backend, artifacts2, log2, routing2, live_world=False,
        method_cache=cache)

    second_game.run(
        {**ROW, "id": "c2"},
        Obligation("candidate-2", "status equals completed", BASIS_WORLD))

    assert [call["role"] for call in second_backend.calls] == [
        "oracles", "judger"]
    assert not any(event["type"] == "method_cache_hit"
                   for event in log2.replay())


def test_mechanical_adapter_audit_rejects_tampered_artifact(tmp_path):
    _, world, artifacts, _, _, routing = captured(tmp_path)
    (world / "result.json").write_text('{"status":"completed"}')
    typed = method(
        "document.json-value", {
            "address": "/app/result.json", "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject("/app/result.json")])
    obligation = Obligation(
        "c1", "status equals completed", BASIS_WORLD)
    result = routing.discharge(ROW, obligation, typed)
    path = tmp_path / "gv" / "artifacts" / f"{result.artifact}.json"
    payload = json.loads(path.read_text())
    payload["claim"] = "tampered"
    path.write_text(json.dumps(payload))

    valid, reason = routing.registry.validate_evidence(
        typed, ROW, obligation, result, artifacts)

    assert valid is False
    assert "digest" in reason


def test_online_method_reuse_is_state_scoped_and_reads_live_bytes(tmp_path):
    cache = AcceptedMethodCache()
    state = tmp_path / "gv"
    state.mkdir()
    (state / "task.md").write_text("Keep live result status completed")
    target = tmp_path / "delivery" / "result.json"
    target.parent.mkdir()
    target.write_text('{"status":"completed"}')
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, default_adapter_registry(), artifacts, log, str(state),
        live_world=True)
    typed = method(
        "document.json-value", {
            "address": str(target), "pointer": "/status",
            "expected": "completed",
        }, inputs=[subject(str(target))])
    first_backend = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    first_game = BindingGame(
        first_backend, artifacts, log, routing, live_world=True,
        method_cache=cache)
    obligation = Obligation("c1", "status equals completed", BASIS_WORLD)
    assert first_game.run(ROW, obligation).outcome == OUTCOME_HOLDS

    target.write_text('{"status":"wrong"}')
    second_backend = FixtureBackend({
        "oracles": {"methods": [typed]}, "judger": JUDGE_YES})
    second_game = BindingGame(
        second_backend, artifacts, log, routing, live_world=True,
        method_cache=cache)
    result = second_game.run({**ROW, "id": "c2"}, obligation)

    assert result.outcome == OUTCOME_REFUTED
    assert [call["role"] for call in second_backend.calls] == ["judger"]
    assert fallback.calls == []
    events = list(log.replay())
    assert any(event["type"] == "method_cache_bypass" for event in events)
    assert not any(event["type"] == "method_cache_hit" for event in events)
