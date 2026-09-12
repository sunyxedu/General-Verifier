"""Host-adapter isolation fixtures: no CLI, model, network or evaluator calls."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tomllib
from types import SimpleNamespace

import pytest

from experiments.cybergym.sidecar import SandboxedVerifierBackend
from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, BASIS_WORLD, Obligation, OUTCOME_FAILED, OUTCOME_HOLDS, OUTCOME_REFUTED
from verifier.execution import adapters
from verifier.execution.adapters import AdapterNotApplicable, AdapterRoutingDischarger, default_adapter_registry
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber.adapters import CyberAdapterContext


ROW = {"id": "fixture-claim", "kind": "final", "text": "the fixed subject satisfies its contract"}
PRIVATE = b"fixture private credential: must never be opened"


def subject(ref, provenance="delivered"):
    return {"ref": ref, "provenance": provenance, "role": "subject"}


def method(name, arguments, *, basis=BASIS_WORLD, inputs=None):
    return {"procedure": "execute only the registered fixture operation", "artifact": "exact receipt",
            "level": "deterministic", "basis": basis, "strength": "decisive", "bridges": [],
            "adapter": name, "adapter_version": 1, "adapter_arguments": json.dumps(arguments),
            "adapter_inputs": inputs or []}


def fixture_state(tmp_path, *, captured=False, freeze=True):
    state, project = tmp_path / "state", tmp_path / "project"
    state.mkdir(mode=0o700)
    project.mkdir(mode=0o700)
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    log.append("message", {"text": "public retained fixture event"})
    if freeze:
        log.bind_record(ROW["id"], log.freeze_record(1))
    if captured:
        (state / "world/app").mkdir(parents=True)
        (state / "world_manifest.json").write_text(json.dumps({
            "scheme": "filesystem", "captured_roots": ["/app"], "artifacts": []}))
    return SimpleNamespace(state=state, project=project, log=log, artifacts=artifacts, captured=captured)


def context(fixture, *, claim=ROW["id"], project=None):
    return CyberAdapterContext(str(fixture.state), not fixture.captured, fixture.log, claim,
                               str(fixture.project) if project is None else project)


def map_world_file(fixture, address, host):
    manifest_path = fixture.state / "world_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"].append({"address": address, "host": str(host)})
    manifest_path.write_text(json.dumps(manifest))


def write_fixture_file(path, data=b"public fixture bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


class PolicyOnlyFallback:
    """Exercise the real policy builder without constructing/running Codex."""
    def __init__(self, fixture):
        self.fixture, self.calls = fixture, []
        self.backend = object.__new__(SandboxedVerifierBackend)
        self.backend.cwd = str(fixture.state)
        self.backend.project_root = str(fixture.project)
        self.backend.offline_toolchain_dir = None
        self.backend.offline_shared_cache_dir = None
        self.backend.search_roles = frozenset()

    def discharge(self, row, obligation, method=None, tag=""):
        variables = {"world_mode": "live", "ledger_path": str(self.fixture.log.path)}
        args = self.backend._execution_policy_args("discharge", variables)
        filesystem = next(arg for arg in args if ".filesystem=" in arg)
        grants = tomllib.loads("grants=" + filesystem.split("=", 1)[1])["grants"]
        assert grants[":root"] == "deny"
        assert grants[str(self.fixture.state / "verifier-home")] == "deny"
        assert all(grants[root] == "deny" for root in self.backend.DENIED_ROOTS)
        assert any(".network.enabled=false" in arg for arg in args)
        assert "--dangerously-bypass-approvals-and-sandbox" not in args
        self.calls.append({"method": method, "grants": grants})
        return DischargeResult(OUTCOME_FAILED, None, self.fixture.artifacts.put({"boundary": "denied"}),
                               instrument="fixture:policy-only-fallback")


def router(fixture, fallback=None):
    fallback = fallback or PolicyOnlyFallback(fixture)
    routing = AdapterRoutingDischarger(fallback, default_adapter_registry(), fixture.artifacts,
        fixture.log, str(fixture.state), live_world=not fixture.captured,
        context_factory=lambda row: context(fixture, claim=row["id"]))
    return routing, fallback


def test_cyber_context_authenticates_an_immutable_prefix_excluding_later_events(tmp_path):
    fixture = fixture_state(tmp_path)
    first = context(fixture)
    later = fixture.log.append("message", {"text": "later event is outside the claim"})
    second = context(fixture)
    assert first.log.path == second.log.path == fixture.log.record_path_for(ROW["id"])
    assert first.log.get(1)["payload"]["text"] == "public retained fixture event"
    assert first.log.get(later["seq"]) is None and second.log.get(later["seq"]) is None
    assert fixture.log.get(later["seq"]) is not None


def test_typed_ledger_execution_cannot_promote_a_later_live_event_into_claim_evidence(tmp_path):
    fixture = fixture_state(tmp_path)
    fixture.log.append("message", {"text": "not in this claim's prefix"})
    routing, fallback = router(fixture)
    obligation = Obligation(ROW["id"], "required event exists in the claim record", BASIS_RECORD,
                            refutable_by_absence=True)
    result = routing.discharge(ROW, obligation, method("ledger.event", {"event_seq": 2},
        basis=BASIS_RECORD, inputs=[subject("event:2", "ledger")]))
    assert result.instrument == "adapter:ledger.event@1" and result.outcome == OUTCOME_REFUTED
    payload = fixture.artifacts.get(result.artifact)
    assert payload["execution"]["observation"] == {"event_seq": 2, "exists": False}
    assert fallback.calls == []


@pytest.mark.parametrize("location", ["project", "task.md", "artifacts", "cyber/blobs", "baseline", "world"])
@pytest.mark.parametrize("captured", [False, True])
def test_live_and_captured_contexts_allow_only_declared_public_evidence(tmp_path, location, captured):
    fixture = fixture_state(tmp_path, captured=captured)
    # Resolve the live view before a world/ fixture directory is introduced.
    live_context = context(fixture) if not captured else None
    host = (fixture.project / "subject.bin" if location == "project"
            else fixture.state / "task.md" if location == "task.md"
            else fixture.state / location / "subject.bin")
    write_fixture_file(host)
    address = "/declared/subject" if captured else str(host)
    if captured:
        map_world_file(fixture, address, host)
    selected_context = context(fixture) if captured else live_context
    resolved = selected_context.resolve(address)
    assert resolved.host_path == str(host)
    ir = method("filesystem.sha256", {"address": address,
        "expected_sha256": hashlib.sha256(host.read_bytes()).hexdigest()}, inputs=[subject(address)])
    result = default_adapter_registry().execute(ir, ROW,
        Obligation(ROW["id"], "declared public bytes match", BASIS_WORLD), selected_context, fixture.artifacts)
    assert result.outcome == OUTCOME_HOLDS


@pytest.mark.parametrize("relative", ["verifier-home/auth.json", "verification_jobs.jsonl",
    "backend_usage.jsonl", "role_calls.jsonl", "transport_calls.jsonl", "hook_feed.jsonl",
    "controller/runtime-secret", "evaluator/config.json", "codex-home/auth.json", "claims.sqlite"])
@pytest.mark.parametrize("captured", [False, True])
def test_private_state_credentials_journals_and_controller_paths_are_never_opened(tmp_path, monkeypatch, relative, captured):
    fixture = fixture_state(tmp_path, captured=captured)
    private = write_fixture_file(fixture.state / relative, PRIVATE)
    address = "/declared/private" if captured else str(private)
    if captured:
        map_world_file(fixture, address, private)
    opened = []
    original = os.open

    def audited_open(path, flags, *args, **kwargs):
        if os.path.abspath(os.fspath(path)) == str(private):
            opened.append(path)
        return original(path, flags, *args, **kwargs)

    monkeypatch.setattr(adapters.os, "open", audited_open)
    routing, fallback = router(fixture)
    ir = method("filesystem.sha256", {"address": address, "expected_sha256": "0" * 64}, inputs=[subject(address)])
    result = routing.discharge(ROW, Obligation(ROW["id"], "a private source is not evidence", BASIS_WORLD), ir)
    assert result.instrument == "fixture:policy-only-fallback" and result.outcome == OUTCOME_FAILED
    assert len(fallback.calls) == 1 and opened == []
    assert PRIVATE.decode() not in json.dumps(fixture.artifacts.get(result.artifact))


@pytest.mark.parametrize("private_root", SandboxedVerifierBackend.DENIED_ROOTS)
def test_actual_native_private_roots_are_denied_even_when_declared_as_project(tmp_path, private_root):
    fixture = fixture_state(tmp_path)
    with pytest.raises(AdapterNotApplicable, match="public evidence boundary"):
        context(fixture, project=private_root).resolve(private_root + "/auth.json")


@pytest.mark.parametrize("credential_root", [".codex", ".ssh", ".aws"])
def test_user_credential_roots_override_a_parent_project_allowance(tmp_path, monkeypatch, credential_root):
    fixture = fixture_state(tmp_path)
    private = write_fixture_file(fixture.project / credential_root / "credential", PRIVATE)
    original = os.path.expanduser
    monkeypatch.setattr(os.path, "expanduser", lambda value: str(fixture.project) if value == "~" else original(value))
    with pytest.raises(AdapterNotApplicable, match="public evidence boundary"):
        context(fixture).resolve(str(private))


@pytest.mark.parametrize("leaf", [False, True])
def test_symlink_leaf_and_parent_escapes_are_denied_before_private_open(tmp_path, leaf):
    fixture = fixture_state(tmp_path)
    private = write_fixture_file(fixture.state / "verifier-home/subject", PRIVATE)
    if leaf:
        link = fixture.project / "subject"
        link.symlink_to(private)
        address = str(link)
    else:
        link = fixture.project / "directory"
        link.symlink_to(private.parent, target_is_directory=True)
        address = str(link / private.name)
    with pytest.raises(AdapterNotApplicable, match="public evidence boundary"):
        context(fixture).resolve(address)


def test_unfrozen_claim_falls_back_fail_closed_and_cannot_enter_adapter_only_reuse(tmp_path):
    fixture = fixture_state(tmp_path, freeze=False)
    public = write_fixture_file(fixture.project / "subject")
    routing, fallback = router(fixture)
    ir = method("filesystem.sha256", {"address": str(public), "expected_sha256": "0" * 64}, inputs=[subject(str(public))])
    obligation = Obligation(ROW["id"], "fixed public subject was verified", BASIS_WORLD)
    result = routing.discharge(ROW, obligation, ir)
    assert result.outcome == OUTCOME_FAILED and len(fallback.calls) == 1
    cached_result, why = routing.discharge_adapter_only(ROW, obligation, ir)
    assert cached_result is None and "frozen record" in why and len(fallback.calls) == 1


def test_unregistered_or_tampered_frozen_record_is_not_authenticated(tmp_path):
    fixture = fixture_state(tmp_path)
    frozen = Path(fixture.log.record_path_for(ROW["id"]))
    with pytest.raises(AdapterNotApplicable, match="frozen record"):
        context(fixture, claim="unregistered-claim")
    frozen.chmod(0o644)
    frozen.write_text('{"seq":1,"type":"forged","payload":{}}\n')
    with pytest.raises((AdapterNotApplicable, ValueError), match="record"):
        context(fixture)


def test_context_factory_is_claim_scoped_not_the_router_initial_live_context(tmp_path):
    fixture = fixture_state(tmp_path)
    fixture.log.append("message", {"text": "visible only to the second claim"})
    fixture.log.bind_record("second-claim", fixture.log.freeze_record(2))
    routing, fallback = router(fixture)
    ir = method("ledger.event", {"event_seq": 2}, basis=BASIS_RECORD,
                inputs=[subject("event:2", "ledger")])
    first = routing.discharge(ROW, Obligation(ROW["id"], "retained event exists", BASIS_RECORD,
                             refutable_by_absence=True), ir)
    other_row = {**ROW, "id": "second-claim"}
    second = routing.discharge(other_row, Obligation(other_row["id"], "retained event exists", BASIS_RECORD,
                              refutable_by_absence=True), ir)
    assert first.outcome == OUTCOME_REFUTED and second.outcome == OUTCOME_HOLDS
    assert fallback.calls == []


def test_private_hardlink_alias_in_public_project_never_reads_private_bytes(tmp_path, monkeypatch):
    fixture = fixture_state(tmp_path)
    private = write_fixture_file(fixture.state / "verifier-home/subject", PRIVATE)
    public_alias = fixture.project / "alias"
    os.link(private, public_alias)
    identity = private.stat()
    private_reads = []
    original_read = os.read

    def audited_read(descriptor, size):
        info = os.fstat(descriptor)
        if (info.st_dev, info.st_ino) == (identity.st_dev, identity.st_ino):
            private_reads.append(True)
        return original_read(descriptor, size)

    monkeypatch.setattr(adapters.os, "read", audited_read)
    routing, fallback = router(fixture)
    ir = method("filesystem.sha256", {"address": str(public_alias),
        "expected_sha256": hashlib.sha256(PRIVATE).hexdigest()}, inputs=[subject(str(public_alias))])
    result = routing.discharge(ROW, Obligation(ROW["id"], "only public bytes may be read", BASIS_WORLD), ir)
    assert result.outcome == OUTCOME_FAILED and len(fallback.calls) == 1
    assert private_reads == []


def test_judger_redo_fallback_cannot_escape_the_existing_agent_policy(tmp_path):
    fixture = fixture_state(tmp_path)
    private = write_fixture_file(fixture.state / "verifier-home/subject", PRIVATE)
    routing, fallback = router(fixture)
    obligation = Obligation(ROW["id"], "private bytes are not a public witness", BASIS_WORLD)
    ir = method("filesystem.sha256", {"address": str(private), "expected_sha256": "0" * 64},
                inputs=[subject(str(private))])
    ir["redo_feedback"] = "fixture execution repair required"
    immediate = routing.discharge(ROW, obligation, ir)
    repair = routing.discharge_with_repair(ROW, obligation, ir, previous_artifact="0" * 64)
    assert immediate.outcome == repair.outcome == OUTCOME_FAILED
    assert len(fallback.calls) == 2
    assert all(call["grants"][str(fixture.state / "verifier-home")] == "deny" for call in fallback.calls)


def test_parent_symlink_replacement_after_admission_cannot_read_private_content(tmp_path, monkeypatch):
    fixture = fixture_state(tmp_path)
    public = write_fixture_file(fixture.project / "directory/subject", b"public subject")
    private = write_fixture_file(fixture.state / "verifier-home/subject", PRIVATE)
    parent_identity = public.parent.stat()
    original_open = os.open
    raced, private_opened = [], []

    def replace_parent_before_open(path, flags, *args, **kwargs):
        raw = os.fspath(path)
        parent_fd = kwargs.get("dir_fd")
        pinned_parent = (os.fstat(parent_fd) if parent_fd is not None else None)
        matches_public = raw == str(public) or (raw == public.name and pinned_parent is not None
            and (pinned_parent.st_dev, pinned_parent.st_ino) == (parent_identity.st_dev, parent_identity.st_ino))
        if matches_public and not raced:
            public.parent.rename(fixture.project / "original-directory")
            public.parent.symlink_to(private.parent, target_is_directory=True)
            raced.append(True)
        if raw == str(public) and Path(raw).resolve() == private:
            private_opened.append(True)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(adapters.os, "open", replace_parent_before_open)
    routing, fallback = router(fixture)
    ir = method("filesystem.sha256", {"address": str(public),
        "expected_sha256": hashlib.sha256(PRIVATE).hexdigest()}, inputs=[subject(str(public))])
    result = routing.discharge(ROW, Obligation(ROW["id"], "public subject must stay public", BASIS_WORLD), ir)
    assert raced, "fixture must replace the public parent after admission"
    assert private_opened == [], "host adapter opened a private file through a raced public parent"
    assert result.outcome != OUTCOME_HOLDS
