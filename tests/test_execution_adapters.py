"""Typed adapter execution stays a fail-safe optimization of BindingGame."""
from __future__ import annotations

import hashlib
import json
import os
import stat
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
    ORACLE_CATALOG_INDEX_SCHEMA,
    ORACLE_CATALOG_INLINE_BYTES,
    ORACLE_CATALOG_MAX_DESCRIPTOR_BYTES,
    ORACLE_CATALOG_SLICE_SCHEMA,
    AdapterInvocation,
    AdapterNotApplicable,
    AdapterRegistry,
    AdapterRoutingDischarger,
    MethodAdapter,
    OracleCatalogView,
    default_adapter_registry,
)
from verifier.execution.browser import (
    BROWSER_DOCUMENT_POLICY, BROWSER_NETWORK_POLICY,
    BROWSER_OBSERVATION_PROFILE, BROWSER_RESPONSE_SCHEMA,
)
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import AgentBackend, AgentRole
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.schema import FixtureBackend
from verifier.llm.schema import RoleError
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
        ("evidence.program", 1),
        ("evidence.program", 2),
        ("record.program", 1),
        ("record.query", 1),
        ("ledger.event", 1),
        ("ledger.sequence", 1),
        ("ledger.sequence", 2),
        ("ledger.tool-call-string", 1),
        ("ledger.tool-call-string", 2),
        ("ledger.tool-call-string", 3),
    }
    assert all(item["identity_contract"] == {
        "selection": "exact-registered-name-version-only",
        "repair_version_inference": "forbidden",
    } for item in catalog)
    for item in catalog:
        evidence = item["evidence_contract"]
        assert evidence["schema"] == EVIDENCE_SCHEMA
        assert evidence["content_address"] == "sha256(canonical-json)"
        assert evidence["closed_fields"] == {
            "receipt": [
                "schema", "instrument", "claim", "obligation", "method",
                "execution",
            ],
            "execution": [
                "outcome", "witness", "witness_kind", "declared_inputs",
                "consumed", "observation",
            ],
        }
    command = next(item for item in catalog
                   if item["name"] == "ledger.tool-call-string")
    text = next(item for item in catalog if item["name"] == "document.text")
    assert text["accepted_method_reuse"] is True
    assert text["evidence_contract"]["retention_limits"] == {
        "source_file_bytes_read_max": 8 * 1024 * 1024,
        "expected_characters_max": 16_384,
        "complete_decoded_text_retained": False,
        "actual_text_sha256_retained_for_decoded_file": True,
        "first_match_character_retained": True,
    }
    decoded = next(
        variant for variant in text["evidence_contract"]["observation"][
            "variants"]
        if variant["name"] == "decoded-literal-predicate")
    assert decoded["required_fields"] == [
        "address", "identity", "encoding", "predicate", "expected",
        "text_characters", "matched", "first_match_character",
    ]
    assert command["basis"] == BASIS_RECORD
    assert "never parses or executes" in command["summary"]
    assert not any(item["basis"] == BASIS_WORLD
                   and "command" in item["name"] for item in catalog)
    assert not any(item["name"] == "browser.html" for item in catalog)


def test_oracle_catalog_slices_only_mechanically_ineligible_record_contracts():
    registry = default_adapter_registry()
    full = registry.catalog()

    world_view = registry.oracle_catalog_view(full, BASIS_WORLD)
    document = json.loads(world_view)
    assert document["schema"] == ORACLE_CATALOG_SLICE_SCHEMA
    assert document["selection"] == {
        "required_basis": BASIS_WORLD, "rule": "basis-floor-v1"}
    assert all(entry["basis"] == BASIS_WORLD
               for entry in document["adapters"])
    assert {(entry["name"], entry["version"])
            for entry in document["adapters"]} == {
        (entry["name"], entry["version"])
        for entry in json.loads(full) if entry["basis"] == BASIS_WORLD
    }
    assert {(entry["name"], entry["version"])
            for entry in document["index"]} == {
        (entry["name"], entry["version"]) for entry in json.loads(full)
    }
    assert all(
        entry["descriptor"] == (
            "included" if entry["basis"] == BASIS_WORLD
            else "excluded-below-floor")
        for entry in document["index"])
    assert len(world_view.encode("utf-8")) < len(full.encode("utf-8"))

    record_view = json.loads(
        registry.oracle_catalog_view(full, BASIS_RECORD))
    original = {
        (entry["name"], entry["version"]): entry
        for entry in json.loads(full)
    }
    assert {
        (entry["name"], entry["version"]): entry
        for entry in record_view
    } == original
    assert record_view[0]["name"] == "record.query"
    bases = [entry["basis"] for entry in record_view]
    assert bases == sorted(bases, key=lambda basis: basis != BASIS_RECORD)
    record_names = [
        entry["name"] for entry in record_view
        if entry["basis"] == BASIS_RECORD
    ]
    assert record_names[-1] == "record.program"
    lazy = registry.oracle_catalog_index(full, BASIS_RECORD)
    assert lazy is not None
    lazy_document, _pages = lazy
    assert lazy_document["index"][0]["name"] == "record.query"
    assert {
        (entry["name"], entry["version"])
        for entry in lazy_document["index"]
    } == set(original)


def test_oracle_discovery_priority_is_validated_and_snapshotted():
    class PreferredAdapter(MethodAdapter):
        name = "test.zpreferred"
        basis = BASIS_RECORD
        oracle_discovery_priority = 7

    class OrdinaryAdapter(MethodAdapter):
        name = "test.aordinary"
        basis = BASIS_RECORD

    preferred = PreferredAdapter()
    registry = AdapterRegistry([preferred, OrdinaryAdapter()])
    full = registry.catalog()
    assert [entry["name"] for entry in json.loads(full)] == [
        "test.aordinary", "test.zpreferred"]
    preferred.oracle_discovery_priority = 999
    assert [
        entry["name"]
        for entry in json.loads(
            registry.oracle_catalog_view(full, BASIS_RECORD))
    ] == ["test.zpreferred", "test.aordinary"]

    class InvalidAdapter(MethodAdapter):
        name = "test.invalid"
        oracle_discovery_priority = True

    with pytest.raises(ValueError, match="discovery priority"):
        AdapterRegistry([InvalidAdapter()])


def test_oracle_catalog_slice_fails_open_to_exact_supplied_catalog():
    registry = default_adapter_registry()
    full = registry.catalog()
    cases = [
        "not json",
        "{}",
        json.dumps(json.loads(full)[:-1], sort_keys=True, separators=(",", ":")),
        json.dumps(
            [json.loads(full)[0], *json.loads(full)],
            sort_keys=True, separators=(",", ":")),
    ]
    for malformed in cases:
        assert registry.oracle_catalog_view(malformed, BASIS_WORLD) == malformed
    assert registry.oracle_catalog_view(full, "unknown") == full

    class MisdeclaredCatalogAdapter(MethodAdapter):
        name = "test.misdeclared"
        basis = BASIS_WORLD

        def catalog_entry(self):
            return {**super().catalog_entry(), "basis": BASIS_RECORD}

    dishonest = AdapterRegistry([MisdeclaredCatalogAdapter()])
    dishonest_full = dishonest.catalog()
    assert dishonest.oracle_catalog_view(
        dishonest_full, BASIS_WORLD) == dishonest_full


def test_unknown_backend_uses_same_safe_world_baseline_for_initial_and_repair(
    tmp_path,
):
    _state, _world, artifacts, log, _fallback, routing = captured(tmp_path)
    full_catalog = routing.adapter_catalog()
    backend = FixtureBackend({"oracles": {"methods": []}})
    game = BindingGame(
        backend, artifacts, log, routing, live_world=False)
    obligation = Obligation(
        "c1", "a world fact needs an independent decision", BASIS_WORLD)

    result = game.run(ROW, obligation)

    assert result.outcome != OUTCOME_HOLDS
    assert [call["role"] for call in backend.calls] == ["oracles", "oracles"]
    views = [call["variables"]["adapter_catalog"] for call in backend.calls]
    assert views[0] == views[1]
    assert type(views[0]) is str
    assert json.loads(views[0])["schema"] == ORACLE_CATALOG_SLICE_SCHEMA
    assert routing.adapter_catalog() == full_catalog


@pytest.mark.parametrize(
    ("live_world", "expected_schema", "expected_calls"),
    [(False, ORACLE_CATALOG_SLICE_SCHEMA, 2),
     (True, ORACLE_CATALOG_SLICE_SCHEMA, 1)],
)
def test_codex_catalog_uses_inline_baseline_without_changing_call_count(
    tmp_path, live_world, expected_schema, expected_calls,
):
    if live_world:
        state = tmp_path / "live"
        state.mkdir()
        artifacts = ArtifactStore(str(state / "artifacts"))
        log = EventLog(str(state / "events.jsonl"))
        fallback = FallbackDischarger(artifacts)
        routing = AdapterRoutingDischarger(
            fallback, default_adapter_registry(), artifacts, log,
            str(state), live_world=True)
    else:
        state, _world, artifacts, log, _fallback, routing = captured(tmp_path)

    class RecordingCodexBackend(CodexBackend):
        def __init__(self, cwd):
            super().__init__(cwd=cwd)
            self.observed_calls = []

        def complete(self, role_name, _prompt, variables):
            self.observed_calls.append({
                "role": role_name,
                "catalog": variables["adapter_catalog"],
            })
            return {"methods": []}

    backend = RecordingCodexBackend(str(state))
    result = BindingGame(
        backend, artifacts, log, routing, live_world=live_world).run(
            ROW, Obligation(
                "c1", "a world fact needs an independent decision",
                BASIS_WORLD))

    assert result.outcome != OUTCOME_HOLDS
    assert [call["role"] for call in backend.observed_calls] \
        == ["oracles"] * expected_calls
    catalogs = [call["catalog"] for call in backend.observed_calls]
    assert len({str(catalog) for catalog in catalogs}) == 1
    assert json.loads(catalogs[0])["schema"] == expected_schema
    assert type(catalogs[0]) is str


def test_agent_backend_lazy_catalog_capability_checks_actual_tool_palette():
    backend = AgentBackend()
    assert backend.allows_lazy_oracle_catalog("oracles", {}) is True

    unsafe = AgentBackend(roles={
        "oracles": AgentRole(
            name="oracles", template="{adapter_catalog}",
            schema={"methods": [str]}, tools=["Read", "Write"],
        ),
    })
    assert unsafe.allows_lazy_oracle_catalog("oracles", {}) is False
    assert backend.allows_lazy_oracle_catalog("judger", {}) is False


def test_lazy_oracle_catalog_pages_reconstruct_the_exact_full_catalog(tmp_path):
    state, _world, _artifacts, _log, _fallback, routing = captured(tmp_path)
    full = routing.adapter_catalog()

    for basis in (BASIS_WORLD, BASIS_RECORD):
        document = json.loads(routing.oracle_adapter_catalog(full, basis))
        assert document["schema"] == ORACLE_CATALOG_INDEX_SCHEMA
        root = document["descriptor_root"]
        assert os.path.isabs(root)
        assert os.path.commonpath((str(state), root)) == str(state)
        assert stat.S_IMODE(os.lstat(root).st_mode) & 0o222 == 0
        entries = []
        for filename in os.listdir(root):
            path = os.path.join(root, filename)
            with open(path, "rb") as handle:
                raw = handle.read()
            assert filename == hashlib.sha256(raw).hexdigest() + ".json"
            assert stat.S_IMODE(os.lstat(path).st_mode) & 0o222 == 0
            entries.append(json.loads(raw))
        entries.sort(key=lambda entry: (entry["name"], entry["version"]))
        rebuilt = json.dumps(
            entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if basis == BASIS_RECORD:
            assert rebuilt == full
            assert hashlib.sha256(rebuilt.encode()).hexdigest() \
                == document["full_catalog_sha256"]
        else:
            expected_entries = [
                entry for entry in json.loads(full)
                if entry["basis"] == BASIS_WORLD]
            assert entries == expected_entries
        available = {
            (entry["name"], entry["version"])
            for entry in document["index"]
            if entry["descriptor"] != "excluded-below-floor"
        }
        expected = {
            (entry["name"], entry["version"])
            for entry in json.loads(full)
            if basis == BASIS_RECORD or entry["basis"] == BASIS_WORLD
        }
        assert available == expected
        assert len(json.dumps(document).encode()) < len(full.encode())


@pytest.mark.parametrize("basis", [BASIS_WORLD, BASIS_RECORD])
def test_hybrid_catalog_budget_and_selection_are_deterministic(tmp_path, basis):
    _state, _world, _artifacts, _log, _fallback, routing = captured(tmp_path)
    full = routing.adapter_catalog()
    first = routing.oracle_adapter_catalog(full, basis)
    second = routing.oracle_adapter_catalog(full, basis)
    assert type(first) is OracleCatalogView
    assert str(first) == str(second)
    document = json.loads(first)

    pages = {}
    for item in document["index"]:
        if item["descriptor"] == "excluded-below-floor":
            continue
        path = os.path.join(document["descriptor_root"], item["descriptor"])
        with open(path, "rb") as handle:
            pages[item["descriptor"][:-5]] = handle.read()
    ordered = sorted(pages.items(), key=lambda item: (len(item[1]), item[0]))
    expected = set()
    used = 0
    for digest, encoded in ordered:
        if used + len(encoded) > ORACLE_CATALOG_INLINE_BYTES:
            break
        expected.add(digest)
        used += len(encoded)
    actual = {
        item["descriptor"][:-5] for item in document["index"]
        if "inline_contract" in item}
    assert actual == expected
    assert document["inline_policy"]["used_bytes"] == used
    assert used <= ORACLE_CATALOG_INLINE_BYTES
    assert any(item["name"] == "filesystem.exists"
               and "inline_contract" in item for item in document["index"])


def test_lazy_catalog_preserves_unknown_extension_fields_losslessly(tmp_path):
    class ExtensionAdapter(MethodAdapter):
        name = "test.extension"
        summary = "A general extension adapter."

        def catalog_entry(self):
            return {
                **super().catalog_entry(),
                "future_extension": {
                    "opaque": "x" * 8_000,
                    "nested": [{"meaning": "retained exactly"}],
                },
            }

    state = tmp_path / "extension-state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, AdapterRegistry([ExtensionAdapter()]), artifacts, log,
        str(state), live_world=True)
    full = routing.adapter_catalog()

    document = json.loads(
        routing.oracle_adapter_catalog(full, BASIS_WORLD))
    assert document["schema"] == ORACLE_CATALOG_INDEX_SCHEMA
    descriptor = document["index"][0]["descriptor"]
    path = os.path.join(document["descriptor_root"], descriptor)
    with open(path, "rb") as handle:
        encoded = handle.read()
    assert b"[" + encoded + b"]" == full.encode("utf-8")
    assert json.loads(encoded)["future_extension"]["nested"] \
        == [{"meaning": "retained exactly"}]


def test_oversized_extension_fails_open_before_publication(tmp_path):
    class OversizedAdapter(MethodAdapter):
        name = "test.oversized"
        summary = "An oversized extension contract."

        def catalog_entry(self):
            return {
                **super().catalog_entry(),
                "future_extension": "x" * (
                    ORACLE_CATALOG_MAX_DESCRIPTOR_BYTES + 1),
            }

    state = tmp_path / "oversized-state"
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    fallback = FallbackDischarger(artifacts)
    routing = AdapterRoutingDischarger(
        fallback, AdapterRegistry([OversizedAdapter()]), artifacts, log,
        str(state), live_world=True)
    full = routing.adapter_catalog()

    baseline = routing.registry.oracle_catalog_view(full, BASIS_WORLD)
    assert type(routing.oracle_adapter_catalog(full, BASIS_WORLD)) is str
    assert routing.oracle_adapter_catalog(full, BASIS_WORLD) == baseline
    assert not (state / "oracle_catalog").exists()


def test_tampered_lazy_catalog_fails_open_to_lossless_inline_baseline(tmp_path):
    _state, _world, _artifacts, _log, _fallback, routing = captured(tmp_path)
    full = routing.adapter_catalog()
    document = json.loads(
        routing.oracle_adapter_catalog(full, BASIS_RECORD))
    descriptor = next(iter(os.listdir(document["descriptor_root"])))
    path = os.path.join(document["descriptor_root"], descriptor)
    os.chmod(path, 0o600)
    with open(path, "wb") as handle:
        handle.write(b"{}")
    os.chmod(path, 0o400)

    assert routing.oracle_adapter_catalog(
        full, BASIS_RECORD) == routing.registry.oracle_catalog_view(
            full, BASIS_RECORD)


@pytest.mark.parametrize("backend_raises", [False, True])
def test_call_time_catalog_tamper_discards_payload_and_retries_with_full_catalog(
    tmp_path, backend_raises,
):
    _state, _world, artifacts, log, _fallback, routing = captured(tmp_path)
    calls = []

    class TamperingBackend:
        def allows_lazy_oracle_catalog(self, role, _variables):
            return role == "oracles"

        def complete(self, role, _prompt, variables):
            assert role == "oracles"
            calls.append(str(variables["adapter_catalog"]))
            if len(calls) == 1:
                document = json.loads(variables["adapter_catalog"])
                filename = next(iter(os.listdir(document["descriptor_root"])))
                path = os.path.join(document["descriptor_root"], filename)
                os.chmod(path, 0o600)
                with open(path, "wb") as handle:
                    handle.write(b"{}")
                os.chmod(path, 0o400)
                if backend_raises:
                    raise RoleError("simulated backend failure")
            return {"methods": []}

    result = BindingGame(
        TamperingBackend(), artifacts, log, routing, live_world=False).run(
            ROW, Obligation("c1", "inspect one record fact", BASIS_RECORD))

    assert result.outcome != OUTCOME_HOLDS
    assert json.loads(calls[0])["schema"] == ORACLE_CATALOG_INDEX_SCHEMA
    assert all(isinstance(json.loads(value), list) for value in calls[1:])


def test_cleanup_does_not_remove_a_renamed_substitute_directory(
    tmp_path, monkeypatch,
):
    state, _world, _artifacts, _log, _fallback, routing = captured(tmp_path)
    full = routing.adapter_catalog()
    real_rename = os.rename
    raced = {}

    def rename_then_substitute(source, destination, *, src_dir_fd, dst_dir_fd):
        moved = source + ".moved"
        real_rename(
            source, moved, src_dir_fd=src_dir_fd, dst_dir_fd=dst_dir_fd)
        os.mkdir(source, 0o700, dir_fd=src_dir_fd)
        raced.update(source=source, moved=moved)
        raise OSError("simulated publication race")

    monkeypatch.setattr("verifier.execution.adapters.os.rename",
                        rename_then_substitute)
    assert routing.oracle_adapter_catalog(
        full, BASIS_RECORD) == routing.registry.oracle_catalog_view(
            full, BASIS_RECORD)

    parent = state / "oracle_catalog"
    assert (parent / raced["source"]).is_dir()
    assert (parent / raced["moved"]).is_dir()
    assert os.lstat(parent / raced["source"]).st_ino \
        != os.lstat(parent / raced["moved"]).st_ino


def test_concurrent_catalog_publication_has_one_complete_visible_tree(tmp_path):
    state, _world, _artifacts, _log, _fallback, routing = captured(tmp_path)
    full = routing.adapter_catalog()
    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(pool.map(
            lambda _index: routing.oracle_adapter_catalog(full, BASIS_RECORD),
            range(32)))

    assert len({str(view) for view in views}) == 1
    document = json.loads(views[0])
    assert set(os.listdir(state / "oracle_catalog")) == {
        document["descriptor_set_sha256"]}
    assert not any(name.startswith(".") for name in os.listdir(
        state / "oracle_catalog"))


def test_catalog_snapshot_rejects_registry_contract_drift(tmp_path):
    routing = captured(tmp_path / "default")[-1]
    view = routing.oracle_adapter_catalog(
        routing.adapter_catalog(), BASIS_WORLD)
    assert type(view) is OracleCatalogView
    original_baseline = view._baseline_catalog
    registered = next(iter(routing.registry._adapters.values()))
    registered.summary = registered.summary + " changed"
    assert view.snapshot() is None
    fresh_full = routing.registry.catalog()
    fallback = view.fallback_catalog()
    assert fallback != original_baseline
    assert json.loads(fallback)["schema"] == ORACLE_CATALOG_SLICE_SCHEMA
    assert json.loads(fallback)["full_catalog_sha256"] == hashlib.sha256(
        fresh_full.encode("utf-8")).hexdigest()


def test_online_and_offline_lazy_catalogs_have_same_identity_and_contracts(
    tmp_path,
):
    offline_state, _world, offline_artifacts, offline_log, offline_fallback, \
        offline = captured(tmp_path / "offline")
    online_state = tmp_path / "online"
    online_state.mkdir()
    online_artifacts = ArtifactStore(str(online_state / "artifacts"))
    online_log = EventLog(str(online_state / "events.jsonl"))
    online_fallback = FallbackDischarger(online_artifacts)
    online = AdapterRoutingDischarger(
        online_fallback, default_adapter_registry(), online_artifacts,
        online_log, str(online_state), live_world=True)

    offline_document = json.loads(offline.oracle_adapter_catalog(
        offline.adapter_catalog(), BASIS_RECORD))
    online_document = json.loads(online.oracle_adapter_catalog(
        online.adapter_catalog(), BASIS_RECORD))
    assert offline_document["full_catalog_sha256"] \
        == online_document["full_catalog_sha256"]
    assert offline_document["index"] == online_document["index"]
    assert offline_document["descriptor_root"] != online_document["descriptor_root"]
    assert os.path.commonpath((str(offline_state),
                               offline_document["descriptor_root"])) \
        == str(offline_state)


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
    catalog = json.loads(routing.adapter_catalog())
    text_contract = next(
        item["evidence_contract"] for item in catalog
        if (item["name"], item["version"]) == ("document.text", 1))
    decoded = next(
        variant for variant in text_contract["observation"]["variants"]
        if variant["name"] == "decoded-literal-predicate")
    assert set(observation) == set(decoded["required_fields"])
    assert set(observation["identity"]) == set(
        text_contract["observation"]["file_identity_fields"])
    assert observation["matched"] is matched
    assert observation["identity"]["sha256"]
    assert "actual" not in observation
    assert fallback.calls == []


def test_text_adapter_is_strict_and_absence_aware(tmp_path):
    _, world, artifacts, _, fallback, routing = captured(tmp_path)
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
    text_contract = next(
        item["evidence_contract"] for item in json.loads(
            routing.adapter_catalog())
        if (item["name"], item["version"]) == ("document.text", 1))
    variants = {
        item["name"]: set(item["required_fields"])
        for item in text_contract["observation"]["variants"]
    }
    invalid_observation = artifacts.get(invalid_utf8.artifact)["execution"][
        "observation"]
    absent_observation = artifacts.get(absent.artifact)["execution"][
        "observation"]
    assert set(invalid_observation) == variants["utf8-decode-error"]
    assert set(absent_observation) == variants["address-absent"]
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
    # the receipt carries the observed event itself, so the audit can check
    # entailment from the artifact alone
    observed = artifacts.get(holds.artifact)["execution"]["observation"][
        "observed"]
    assert observed == {"type": event["type"], "payload": event["payload"]}


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
    catalog_view = json.loads(oracle_vars["adapter_catalog"])
    ordered_names = [
        entry["name"] for entry in catalog_view["adapters"]
    ]
    assert ordered_names.index("document.json-value") > 0
    judge_vars = backend.calls[1]["variables"]
    assert "document.json-value" in judge_vars["adapter_method"]
    evidence = json.loads(judge_vars["evidence"])
    assert artifacts.get(evidence["artifact_digest"])["schema"] \
        == EVIDENCE_SCHEMA


def test_record_catalog_order_does_not_prevent_custom_oracle_choice(tmp_path):
    _, _, artifacts, log, fallback, routing = captured(tmp_path)
    custom = {
        "procedure": "audit the complete retained record with a custom check",
        "artifact": "a replayable custom audit",
        "level": "custom exact record audit",
        "basis": BASIS_RECORD,
        "strength": "decisive",
        "bridges": [],
    }
    backend = FixtureBackend({
        "oracles": {"methods": [custom]},
        "judger": JUDGE_YES,
    })

    result = BindingGame(
        backend, artifacts, log, routing, live_world=False).run(
            ROW, Obligation("c1", "inspect a retained record fact", BASIS_RECORD))

    assert result.outcome == OUTCOME_HOLDS
    assert len(fallback.calls) == 1
    catalog = json.loads(backend.calls[0]["variables"]["adapter_catalog"])
    assert catalog[0]["name"] == "record.query"
    assert "adapter" not in fallback.calls[0]["method"]
    assert [call["role"] for call in backend.calls] == ["oracles", "judger"]


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
