"""No-model tests of the explicit, diagnostic-only shared input boundary."""
from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import replace

import pytest

from verifier import offline
from verifier.contract_replay import (
    KernelBoundReplay, ReplaySetupError, SNAPSHOT_POLICY,
    attach_replay_contracts, clone_extracted_inputs,
    clone_sealed_inputs, compilation_config_sha256, compile_input_digests,
    fresh_graph_snapshot, load_replay_contracts, normalized_compile_inputs,
    replay_identity,
)
from verifier.core.events import EventLog
from verifier.core.model import Claim, content_hash
from verifier.core.store import TRIGGER_LAZY
from verifier.engine.kernel import Kernel
from verifier.engine.contract_bundle import (
    ContractReplayError, build_contract_bundle_bytes, failed_contract_entry,
    passed_contract_entry,
)
from verifier.llm.schema import FixtureBackend
from verifier.mcp_server import Service


COMMITS = [
    {"label": "unused", "kind": "assertion", "text": "unused context",
     "deps": [], "revises": ""},
    {"label": "A", "kind": "assertion", "text": "All records report 42",
     "deps": ["effect:2"], "revises": ""},
    {"label": "F", "kind": "final", "text": "The requested result is ready",
     "deps": ["A"], "revises": ""},
]
FEED = [
    {"type": "tool_call", "payload": {
        "tool": "Bash", "tool_use_id": "t1", "input": {"command": "echo 42"}}},
    {"type": "tool_result", "payload": {
        "tool": "Bash", "tool_use_id": "t1",
        "response_sha256": "a" * 64, "excerpt": "42"}},
]


def source_state(tmp_path, name="source", *, extract=True):
    state = tmp_path / name
    state.mkdir()
    (state / "world" / "app").mkdir(parents=True)
    (state / "world" / "app" / "answer.txt").write_text("42\n")
    (state / "task.md").write_text("Report every record's answer.")
    (state / "final.txt").write_text("42")
    (state / "world_manifest.json").write_text(json.dumps({
        "captured_roots": ["/app"], "artifacts": [
            {"address": "answer", "host": "app/answer.txt"}],
        "note": "Task addresses are semantic input, not host locators.",
    }))
    (state / "hook_feed.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in FEED))
    seal = offline.write_immutable_input_anchor(str(state))
    if extract:
        offline.extract(str(state), backend=FixtureBackend({
            "extract": lambda _: {"commits": copy.deepcopy(COMMITS)}}),
            stamp={"immutable_input_seal": seal})
    return state


def service(state, backend=None):
    backend = backend or FixtureBackend({})
    return Service(str(state), trigger=TRIGGER_LAZY, live_world=False,
                   kernel_factory=lambda sc: Kernel(
                       sc.store, sc.log, backend, sc.artifacts,
                       live_world=False))


def snapshot(sc):
    return fresh_graph_snapshot(sc.kernel, sc.builder.finals[-1])


def assertion(sc):
    return sc.store.get(sc.builder.labels["A"])


def test_clone_sealed_inputs_does_not_copy_derived_judgments(tmp_path):
    source = source_state(tmp_path)
    (source / "score.json").write_text('{"verdict":"verified"}')
    (source / "artifacts").mkdir()
    (source / "artifacts" / "proof").write_text("prior evidence")
    destination = tmp_path / "arm"
    seal = clone_sealed_inputs(source, destination)
    assert offline.verify_immutable_input_anchor(str(destination)) == seal
    assert set(path.name for path in destination.iterdir()) <= {
        *offline.IMMUTABLE_INPUT_PATHS, offline.IMMUTABLE_INPUT_SEAL_FILE}
    assert not (destination / "events.jsonl").exists()
    assert not (destination / "score.json").exists()
    original = source / "world" / "app" / "answer.txt"
    cloned = destination / "world" / "app" / "answer.txt"
    assert original.stat().st_ino != cloned.stat().st_ino


@pytest.mark.parametrize("target", ["same", "child", "parent", "existing", "link"])
def test_clone_refuses_overlap_or_existing_destination_without_deletion(tmp_path, target):
    source = source_state(tmp_path)
    destinations = {"same": source, "child": source / "child",
                    "parent": tmp_path, "existing": tmp_path / "existing",
                    "link": tmp_path / "link"}
    (tmp_path / "existing").mkdir()
    marker = tmp_path / "existing" / "marker"
    marker.write_text("keep me")
    (tmp_path / "link").symlink_to(source, target_is_directory=True)
    with pytest.raises(ReplaySetupError):
        clone_sealed_inputs(source, destinations[target])
    assert marker.read_text() == "keep me"
    assert (source / "task.md").exists()


def test_clone_does_not_bless_a_tampered_source(tmp_path):
    source = source_state(tmp_path)
    (source / "task.md").write_text("changed task")
    with pytest.raises(offline.ImmutableInputError):
        clone_sealed_inputs(source, tmp_path / "arm")
    assert not (tmp_path / "arm").exists()


def test_rebuild_graph_keeps_effect_identity_and_has_no_verification_state(tmp_path):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    stamp = clone_extracted_inputs(source, destination)
    first, second = service(source), service(destination)
    assert snapshot(first) == snapshot(second)
    assert stamp["diagnostic_only"] == SNAPSHOT_POLICY
    assert stamp["config"] != offline.config_fields()["config"]
    assert stamp["rounds"] == 1
    assert {e["type"] for e in second.log.replay()} == {
        "tool_call", "tool_result", "message", "extracted_commit"}
    assert not (destination / "verification.json").exists()


@pytest.mark.parametrize("event", ["decomposition_attacked", "judger_verdict",
                                   "certificate", "obligations_compiled"])
def test_rebuild_rejects_post_extraction_verifier_logs(tmp_path, event):
    source = source_state(tmp_path)
    EventLog(str(source / "events.jsonl")).append(event, {"claim": "x"})
    with pytest.raises(ReplaySetupError, match="post-extraction"):
        clone_extracted_inputs(source, tmp_path / "arm")
    assert not (tmp_path / "arm").exists()


def test_rebuild_rejects_unsealed_hook_context_changes(tmp_path):
    source = source_state(tmp_path)
    path = source / "events.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[1]["payload"]["excerpt"] = "untrusted replacement"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ReplaySetupError, match="sealed source hooks"):
        clone_extracted_inputs(source, tmp_path / "arm")
    assert not (tmp_path / "arm" / "extract.json").exists()


@pytest.mark.parametrize("change", ["live", "evidence", "adjudicated", "basis"])
def test_snapshot_refuses_verified_state(tmp_path, change):
    sc = service(source_state(tmp_path))
    cid = assertion(sc)["id"]
    if change == "live":
        sc.kernel.live_world = True
    elif change == "evidence":
        sc.store.record_evidence(cid, {"outcome": "holds"})
    else:
        column, value = ("status", "verified") if change == "adjudicated" else ("basis", "world")
        sc.store.db.execute(f"UPDATE claims SET {column}=? WHERE id=?", (value, cid))
    with pytest.raises(ReplaySetupError):
        snapshot(sc)


def test_snapshot_contains_off_slice_context_but_digests_cover_only_final_slice(tmp_path):
    sc = service(source_state(tmp_path))
    graph = snapshot(sc)
    unused = sc.builder.labels["unused"]
    assert unused in {row["id"] for row in graph["claims"]}
    assert unused not in graph["slice"]
    digests = compile_input_digests(sc.kernel, graph)
    assert set(digests) == {assertion(sc)["id"], sc.builder.finals[-1]}
    sc.store.add_claim(Claim("assertion", "new graph context"))
    with pytest.raises(ReplaySetupError, match="graph changed"):
        compile_input_digests(sc.kernel, graph)


def test_equivalent_clone_roots_have_exact_same_compile_inputs(tmp_path):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    first, second = service(source), service(destination)
    left = normalized_compile_inputs(first.kernel, assertion(first))
    right = normalized_compile_inputs(second.kernel, assertion(second))
    assert left == right
    assert str(source) not in json.dumps(left)
    assert left["dependency_referent_context"] == right["dependency_referent_context"]
    assert "42" in left["dependency_referent_context"]
    assert compile_input_digests(first.kernel, snapshot(first)) == compile_input_digests(
        second.kernel, snapshot(second))


@pytest.mark.parametrize("change", ["quantifier", "premise_status", "graph_text", "note"])
def test_semantic_context_is_not_normalized_away(tmp_path, change):
    sc = service(source_state(tmp_path))
    row = assertion(sc)
    before = normalized_compile_inputs(sc.kernel, row)
    if change == "quantifier":
        row = dict(row, text=row["text"].replace("All", "Some"))
    elif change == "premise_status":
        sc.store.db.execute("UPDATE claims SET status='verified' WHERE id=?",
                            (sc.builder.labels["effect:2"],))
    elif change == "graph_text":
        sc.store.add_claim(Claim("assertion", "additional semantic context"))
    else:
        path = tmp_path / "source" / "world_manifest.json"
        manifest = json.loads(path.read_text())
        manifest["note"] = f"Literal task text includes {tmp_path / 'source' / 'world'}"
        path.write_text(json.dumps(manifest))
    after = normalized_compile_inputs(sc.kernel, row)
    assert content_hash(before) != content_hash(after)
    if change == "note":
        assert str(tmp_path / "source" / "world") in after["world_map"]


def test_final_compile_inputs_are_authority_only(tmp_path):
    sc = service(source_state(tmp_path))
    variables = normalized_compile_inputs(sc.kernel, sc.store.get(sc.builder.finals[-1]))
    text = json.dumps(variables)
    assert "All records" not in text and "echo 42" not in text
    assert "Report every record" in text
    assert variables["graph"] == "[]"


def test_config_only_ignores_executor_switches_and_binds_effective_roles(tmp_path):
    sc = service(source_state(tmp_path))
    config = {"model": "fixture", "typed_adapters": False, "browser_observer": None,
              "contract_cache": True, "config": "derived"}
    expected = compilation_config_sha256(sc.kernel, config)
    assert compilation_config_sha256(sc.kernel, dict(
        config, typed_adapters=True, browser_observer={"version": "new"}, config="other")) == expected
    assert compilation_config_sha256(sc.kernel, dict(config, model="different")) != expected
    sc.kernel.critic_role = replace(sc.kernel.critic_role, system="different Critic semantics")
    assert compilation_config_sha256(sc.kernel, config) != expected


@pytest.mark.parametrize("change", ["memory", "critic", "rounds"])
def test_config_refuses_unclear_compilation_boundary(tmp_path, change):
    sc = service(source_state(tmp_path))
    if change == "memory":
        sc.kernel.memory_hits = '[{"prior":"advisory information"}]'
    elif change == "critic":
        sc.kernel.critic_role = None
    else:
        sc.kernel.critic_rounds = 0
    with pytest.raises(ReplaySetupError):
        compilation_config_sha256(sc.kernel, {})


def fixture_bundle(sc, config, *, failed=False):
    """Simulate an externally pinned operator bundle, not a real compiler run."""
    graph = snapshot(sc)
    inputs = compile_input_digests(sc.kernel, graph)
    payload = {"obligations": [{"text": "Check the stated requirement",
                                "required_basis": "world"}]}
    entries = []
    for claim_id, digest in inputs.items():
        if failed:
            entries.append(failed_contract_entry(
                claim_id=claim_id, compile_input_sha256=digest, stage="critic",
                critic_receipt_sha256=content_hash("synthetic failed Critic receipt"),
                detail_sha256=content_hash("synthetic failed setup")))
        else:
            entries.append(passed_contract_entry(
                claim_id=claim_id, compile_input_sha256=digest,
                critic_receipt_sha256=content_hash("synthetic operator receipt"),
                payload=payload, obligations=sc.kernel.contracts.materialize_obligations(
                    sc.store.get(claim_id), payload, record=False)))
    return build_contract_bundle_bytes(
        identity=replay_identity(sc.kernel, graph, config),
        claims=[{key: row[key] for key in ("id", "kind", "text", "deps", "revises")}
                for row in graph["claims"]], entries=entries)


def test_loader_derives_identity_from_fresh_arm_not_bundle_fields(tmp_path):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    backend = FixtureBackend({})
    first, second = service(source, backend), service(destination, backend)
    config = {"contract_cache": True, "typed_adapters": False}
    raw = fixture_bundle(first, config)
    provider = load_replay_contracts(
        second.kernel, second.builder.finals[-1], dict(config, typed_adapters=True),
        raw, expected_sha256=hashlib.sha256(raw).hexdigest())
    assert provider.replayable
    assert not backend.calls
    assert second.kernel.contract_replay is None  # no implicit activation
    attach_replay_contracts(second.kernel, provider)
    # Settling a premise is legitimate runtime work. Do not recompute the
    # original, status-sensitive compiler digest at contract lookup time.
    second.store.db.execute("UPDATE claims SET status='verified' WHERE id=?",
                            (second.builder.labels["effect:2"],))
    assert second.kernel.contracts.compile_obligations(assertion(second))
    assert not backend.calls


@pytest.mark.parametrize("change", ["pin", "model", "graph", "source"])
def test_loader_rejects_pinned_bundle_against_different_arm(tmp_path, change):
    sc = service(source_state(tmp_path))
    config = {"model": "fixture", "contract_cache": True}
    raw = fixture_bundle(sc, config)
    pin = hashlib.sha256(raw).hexdigest()
    if change == "pin":
        pin = "0" * 64
    elif change == "model":
        config = dict(config, model="changed")
    elif change == "graph":
        sc.store.add_claim(Claim("assertion", "new context"))
    else:
        (tmp_path / "source" / "final.txt").write_text("changed candidate")
    with pytest.raises((ContractReplayError, ReplaySetupError, offline.ImmutableInputError)):
        load_replay_contracts(sc.kernel, sc.builder.finals[-1], config, raw,
                              expected_sha256=pin)
    assert sc.kernel.contract_replay is None


def test_loader_refuses_failed_setup_without_retrying_or_compiling(tmp_path):
    sc = service(source_state(tmp_path))
    raw = fixture_bundle(sc, {}, failed=True)
    with pytest.raises(ContractReplayError, match="failed compilation"):
        load_replay_contracts(sc.kernel, sc.builder.finals[-1], {}, raw,
                              expected_sha256=hashlib.sha256(raw).hexdigest())
    assert not sc.kernel.backend.calls


@pytest.mark.parametrize("marker", ["valid", "invalid", "dangling_symlink"])
def test_ordinary_offline_score_refuses_diagnostic_artifacts_without_model_calls(tmp_path, marker):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    path = destination / offline.DIAGNOSTIC_ONLY_FILE
    if marker == "invalid":
        path.unlink()
        path.write_text("broken metadata")
    elif marker == "dangling_symlink":
        path.unlink()
        path.symlink_to(destination / "absent")
    artifact = destination / "artifacts" / "previous-proof"
    artifact.parent.mkdir()
    artifact.write_text("diagnostic evidence must remain private")
    before = (destination / "events.jsonl").read_bytes()
    backend = FixtureBackend({})
    with pytest.raises(offline.ImmutableInputError, match="diagnostic-only"):
        offline.score(str(destination), backend=backend)
    assert not backend.calls
    assert artifact.read_text() == "diagnostic evidence must remain private"
    assert (destination / "events.jsonl").read_bytes() == before
    assert not (destination / "score.json").exists()


@pytest.mark.parametrize("change", ["add", "remove", "text", "store", "log"])
def test_load_then_graph_or_kernel_mutation_cannot_be_attached(tmp_path, change):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    sc = service(destination)
    raw = fixture_bundle(sc, {})
    provider = load_replay_contracts(sc.kernel, sc.builder.finals[-1], {}, raw,
                                     expected_sha256=hashlib.sha256(raw).hexdigest())
    if change == "add":
        sc.store.add_claim(Claim("assertion", "extra off-slice claim"))
    elif change == "remove":
        sc.store.db.execute("DELETE FROM claims WHERE id=?", (sc.builder.labels["unused"],))
    elif change == "text":
        sc.store.db.execute("UPDATE claims SET text='changed' WHERE id=?",
                            (sc.builder.labels["unused"],))
    elif change == "store":
        sc.kernel.store = service(source).store
    else:
        sc.kernel.log = EventLog(str(destination / "another-log.jsonl"))
    with pytest.raises(ReplaySetupError):
        attach_replay_contracts(sc.kernel, provider)
    assert sc.kernel.contract_replay is None
    assert not sc.kernel.backend.calls


def test_bound_replay_allows_evidence_growth_but_not_later_structural_drift(tmp_path):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    sc = service(destination)
    raw = fixture_bundle(sc, {})
    provider = load_replay_contracts(sc.kernel, sc.builder.finals[-1], {}, raw,
                                     expected_sha256=hashlib.sha256(raw).hexdigest())
    attach_replay_contracts(sc.kernel, provider)
    cid = assertion(sc)["id"]
    obligation = sc.kernel.contracts.compile_obligations(assertion(sc))[0]
    sc.store.add_obligation(cid, obligation)
    sc.store.record_evidence(cid, {"outcome": "holds"})
    sc.store.db.execute("UPDATE claims SET status='verified' WHERE id=?", (cid,))
    provider.validate_inputs()
    assert provider.lookup(sc.builder.finals[-1])
    sc.store.add_claim(Claim("assertion", "new off-slice context after attach"))
    with pytest.raises(ReplaySetupError, match="structure changed"):
        sc.kernel.contracts.compile_obligations(sc.store.get(sc.builder.finals[-1]))
    assert not sc.kernel.backend.calls


def test_snapshot_refuses_compiler_context_without_any_judger_result(tmp_path):
    sc = service(source_state(tmp_path))
    sc.log.append("decomposition_attacked", {"claim": assertion(sc)["id"], "verdict": "pass"})
    with pytest.raises(ReplaySetupError, match="verifier-generated context"):
        snapshot(sc)


def test_attach_rechecks_exact_semantic_inputs_and_constructor_is_not_loader(tmp_path):
    source = source_state(tmp_path)
    destination = tmp_path / "arm"
    clone_extracted_inputs(source, destination)
    sc = service(destination)
    graph = snapshot(sc)
    raw = fixture_bundle(sc, {})
    provider = load_replay_contracts(sc.kernel, sc.builder.finals[-1], {}, raw,
                                     expected_sha256=hashlib.sha256(raw).hexdigest())
    with pytest.raises(ReplaySetupError, match="operator-pinned"):
        KernelBoundReplay(sc.kernel, provider._bundle, graph, {})
    original = sc.kernel.contracts.compile_variables

    def changed(row):
        return dict(original(row), dependency_referent_context="changed semantic context")

    sc.kernel.contracts.compile_variables = changed
    with pytest.raises(ReplaySetupError, match="compiler inputs changed"):
        attach_replay_contracts(sc.kernel, provider)
    assert sc.kernel.contract_replay is None
