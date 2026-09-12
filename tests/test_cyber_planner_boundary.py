"""Inventory-only planner input fixtures; never run a CLI or model API."""
from __future__ import annotations

import json
import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, CLAIM_EFFECT, STATUS_VERIFIED
from verifier.engine import kernel as kernel_module
from verifier.engine.kernel import Kernel
from verifier.engine.source_evidence import SourceEvidenceView
from verifier.llm.agents import ALL_AGENT_ROLES
from verifier.llm.schema import FixtureBackend
from verifier.profiles import get_profile


RAW_INPUT = "fixture-raw-command-not-for-planners"
RAW_RESULT = "fixture-raw-result-not-for-planners"


class MemoryLog:
    path = "/nonexistent-planner-fixture/events.jsonl"

    def __init__(self, *, malformed=False):
        self.rows = [
            {"seq": 1, "type": "tool_call", "payload": {"tool": "fixture-tool", "tool_use_id": "fixture-1",
                "input": {"cmd": RAW_INPUT}}},
            {"seq": 2, "type": "tool_result", "payload": {"tool": "fixture-tool", "tool_use_id": "fixture-1",
                "excerpt": RAW_RESULT, "excerpt_truncated": False}}]
        self.malformed = malformed
        self.audit = []

    def replay(self):
        return iter(self.rows)

    def get(self, seq):
        return next((row for row in self.rows if row["seq"] == seq), None)

    def record_path_for(self, claim):
        return "/nonexistent-planner-fixture/record/events-through-2-fixture.jsonl"

    def append(self, kind, payload):
        self.audit.append((kind, payload))


class MemoryStore:
    def __init__(self, log):
        digest = EventLog.event_digest(log.get(2))
        self.effect = {"id": "effect-id", "kind": CLAIM_EFFECT, "text": json.dumps({
            "event_seq": 2, "tool": "fixture-tool", "digest": "0" * 64 if getattr(log, "malformed", False) else digest})}
        self.support = {"id": "support-id", "kind": "assertion", "text": "the selected logical subject has the named property"}

    def parents(self, claim):
        return {"assertion-id": ["support-id"], "support-id": ["effect-id"]}.get(claim, [])

    def get(self, claim):
        return self.effect if claim == "effect-id" else self.support

    def effective(self, claim):
        return {"effective_status": STATUS_VERIFIED, "effective_basis": BASIS_RECORD}

    def graph_snapshot(self, exclude):
        return [self.effect, self.support]


def kernel_fixture(*, profile="cyber", context_id="", malformed=False, with_profile_template=True):
    roles = get_profile(profile).apply_roles(ALL_AGENT_ROLES)
    kernel = object.__new__(Kernel)
    kernel.log = MemoryLog(malformed=malformed)
    kernel.store = MemoryStore(kernel.log)
    kernel.memory = None
    kernel.memory_hits = "[]"
    kernel.live_world = False
    kernel.critic_rounds = 1
    kernel.obligations_role = roles["obligations"]
    kernel.critic_role = roles["critic"]
    if not with_profile_template:
        kernel.obligations_role = ALL_AGENT_ROLES["obligations"]
        kernel.critic_role = ALL_AGENT_ROLES["critic"]
    kernel.context_provider = lambda role, claim: json.dumps({"inventory": "fixture", "role": role})
    kernel.context_id_provider = lambda claim: context_id
    return kernel


ROW = {"id": "assertion-id", "kind": "assertion", "text": "the fixed logical subject satisfies the assertion"}
PAYLOAD = {"obligations": [{"text": "the required property holds", "required_basis": BASIS_RECORD}]}
PASS = {"verdict": "pass", "counterexample_world": "n/a", "missing_condition": "n/a"}


@pytest.mark.parametrize("compact", [False, True])
def test_cyber_obligations_and_critic_prompts_never_inline_effect_inputs_or_results(compact):
    kernel = kernel_fixture()
    backend = FixtureBackend({"obligations": PAYLOAD, "critic": PASS})
    backend.compact_inspection = compact
    kernel.backend = backend
    payload = kernel._compile_obligation_payload(ROW)
    assert payload == PAYLOAD
    assert [call["role"] for call in backend.calls] == ["obligations", "critic"]
    for call in backend.calls:
        variables = call["variables"]
        role = kernel.obligations_role if call["role"] == "obligations" else kernel.critic_role
        rendered = role.render(variables)
        assert RAW_INPUT not in rendered and RAW_RESULT not in rendered
        assert RAW_INPUT not in json.dumps(variables) and RAW_RESULT not in json.dumps(variables)
        context = json.loads(variables["dependency_referent_context"])
        effect = next(node for node in context["nodes"] if node["kind"] == CLAIM_EFFECT)["effect_record"]
        assert effect == {"content_withheld": True, "identity_available": True,
            "record_locator": kernel.log.record_path_for(ROW["id"]), "effect_ref": "effect:2", "event_seq": 2,
            "event_digest": EventLog.event_digest(kernel.log.get(2)), "tool": "fixture-tool"}
        support = next(node for node in context["nodes"] if node["kind"] == "assertion")
        assert support["text"]["value"] == kernel.store.support["text"]
        assert support["parents"] == ["effect-id"]


def test_default_planner_preserves_the_existing_complete_referent_semantics():
    kernel = kernel_fixture(profile="default")
    variables = kernel._obligation_compile_variables(ROW)
    context = json.loads(variables["dependency_referent_context"])
    effect = next(node for node in context["nodes"] if node["kind"] == CLAIM_EFFECT)["effect_record"]
    assert effect["call"]["input"]["value"]["cmd"] == RAW_INPUT
    assert effect["result"]["excerpt"]["value"] == RAW_RESULT
    assert effect["complete"] is True


def test_authenticated_context_id_alone_keeps_custom_roles_inventory_only():
    kernel = kernel_fixture(profile="default", context_id="profile-binding-identity", with_profile_template=False)
    variables = kernel._obligation_compile_variables(ROW)
    assert RAW_INPUT not in json.dumps(variables) and RAW_RESULT not in json.dumps(variables)
    effect = next(node for node in json.loads(variables["dependency_referent_context"])["nodes"]
                  if node["kind"] == CLAIM_EFFECT)["effect_record"]
    assert effect["content_withheld"] is True and effect["identity_available"] is True


def test_invalid_effect_identity_exposes_neither_raw_content_nor_dynamic_error():
    kernel = kernel_fixture(malformed=True)
    context = kernel._dependency_referent_context(ROW["id"])
    effect = next(node for node in context["nodes"] if node["kind"] == CLAIM_EFFECT)["effect_record"]
    assert effect == {"content_withheld": True, "identity_available": False,
                      "record_locator": kernel.log.record_path_for(ROW["id"])}
    assert RAW_INPUT not in json.dumps(context) and RAW_RESULT not in json.dumps(context)


@pytest.mark.parametrize("identity", [
    "not-json", "[]", '{"event_seq":true,"digest":"' + "0" * 64 + '"}',
    '{"event_seq":2,"digest":"bad"}', '{"event_seq":99,"digest":"' + "0" * 64 + '"}',
])
def test_malformed_effect_identity_is_unknown_metadata_not_raw_evidence(identity):
    kernel = kernel_fixture()
    kernel.store.effect["text"] = identity
    context = kernel._dependency_referent_context(ROW["id"])
    effect = next(node for node in context["nodes"] if node["kind"] == CLAIM_EFFECT)["effect_record"]
    assert effect["identity_available"] is False and effect["content_withheld"] is True
    assert RAW_INPUT not in json.dumps(context) and RAW_RESULT not in json.dumps(context)


def test_metadata_route_does_not_invoke_the_raw_referent_collector(monkeypatch):
    kernel = kernel_fixture()
    def forbidden(*args):
        raise AssertionError("raw call/result collection is not a planner capability")
    monkeypatch.setattr(kernel, "_effect_referent_records", forbidden)
    assert kernel._dependency_referent_context(ROW["id"])["nodes"]


def test_authenticated_malformed_payload_is_unknown_metadata_not_a_bootstrap_error():
    kernel = kernel_fixture()
    event = kernel.log.get(2)
    event["payload"] = [RAW_RESULT]
    kernel.store.effect["text"] = json.dumps({
        "event_seq": 2, "digest": EventLog.event_digest(event)})
    context = kernel._dependency_referent_context(ROW["id"])
    effect = next(node for node in context["nodes"] if node["kind"] == CLAIM_EFFECT)["effect_record"]
    assert effect["identity_available"] is False
    assert RAW_RESULT not in json.dumps(context)


def test_minimal_bare_kernel_fixture_retains_default_referent_behavior():
    kernel = kernel_fixture(profile="default")
    del kernel.obligations_role, kernel.critic_role, kernel.context_id_provider
    context = kernel._dependency_referent_context(ROW["id"])
    assert RAW_RESULT in json.dumps(context)


@pytest.mark.parametrize("profile,context_id,inventory_only", [
    ("cyber", "", True),
    ("default", "authenticated-profile-context", True),
    ("default", "", False),
])
def test_real_frozen_compact_binding_and_captured_world_respect_planner_boundary(
        tmp_path, monkeypatch, profile, context_id, inventory_only):
    kernel = kernel_fixture(profile=profile, context_id=context_id)
    log = EventLog(str(tmp_path / "events.jsonl"))
    for event in kernel.log.rows:
        log.append(event["type"], event["payload"])
    frozen = log.freeze_record(2)
    log.bind_record(ROW["id"], frozen)
    kernel.log = log
    kernel.store = MemoryStore(log)
    (tmp_path / "world").mkdir()
    manifest = {"scheme": "filesystem", "captured_roots": ["/public-task"],
                "artifacts": [{"address": "/public-task/candidate", "host": "candidate"}],
                "note": RAW_RESULT}
    (tmp_path / "world_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    original_world = json.loads(kernel_module.world_map(str(tmp_path), live_world=False))
    original_bind = kernel_module.bind_source_evidence_variables
    binding_flags = []

    def capture_bind(log, variables, *, enabled):
        binding_flags.append(enabled)
        return original_bind(log, variables, enabled=enabled)

    monkeypatch.setattr(kernel_module, "bind_source_evidence_variables", capture_bind)
    backend = FixtureBackend({"obligations": PAYLOAD, "critic": PASS})
    backend.compact_inspection = True
    kernel.backend = backend
    assert kernel._compile_obligation_payload(ROW) == PAYLOAD
    assert binding_flags == [not inventory_only, not inventory_only]
    assert [call["role"] for call in backend.calls] == ["obligations", "critic"]
    expected_world = dict(original_world)
    if inventory_only:
        expected_world.pop("note")
    for call in backend.calls:
        variables = call["variables"]
        assert json.loads(variables["world_map"]) == expected_world
        assert variables["claim_text"] == ROW["text"]
        assert kernel.store.support["text"] in variables["verified_deps"]
        if inventory_only:
            assert "_source_evidence_view" not in variables
            assert "_mixed_ledger_path" not in variables
            assert variables["ledger_path"] == frozen
            assert RAW_INPUT not in json.dumps(variables)
            assert RAW_RESULT not in json.dumps(variables)
        else:
            assert type(variables["_source_evidence_view"]) is SourceEvidenceView
            assert variables["_mixed_ledger_path"] == frozen
