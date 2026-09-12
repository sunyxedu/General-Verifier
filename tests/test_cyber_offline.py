"""Offline and CLI cyber selection preserves the ordinary task contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from verifier import cli, offline
from verifier.core.events import EventLog
from verifier.core.model import content_hash
from verifier.llm.agents import ALL_AGENT_ROLES
from verifier.llm.schema import FixtureBackend
from verifier.profiles import get_profile


TASK = "Deliver an artifact demonstrating the stated in-scope security condition."
COMMITS = {"commits": [{
    "label": "F", "kind": "final", "text": "The task's complete deliverable is correct.",
    "deps": [], "revises": "",
}]}


class RecordingBackend(FixtureBackend):
    def __init__(self):
        def discharge(variables):
            evidence = Path(variables["workdir"]) / "observation.txt"
            evidence.write_text("independently observed task-defined condition")
            return {
                "outcome": "holds", "witness": "", "artifact": str(evidence),
                "assumed_premises": [], "witness_kind": "none",
                "inputs": [{"ref": variables["task_context"],
                            "provenance": "task_given", "role": "authority"}],
            }

        super().__init__({
            "extract": COMMITS,
            "obligations": {"obligations": [{
                "text": "The task's deliverable satisfies the stated in-scope security condition.",
                "required_basis": "world", "severity": "must", "clause": TASK,
                "refutable_by_absence": False,
            }]},
            "critic": {"verdict": "pass", "counterexample_world": "TERM AUDIT: complete", "missing_condition": "n/a"},
            "oracles": {"methods": [{
                "procedure": "Independently inspect the task-defined security property of the exact subject.",
                "artifact": "Retained task-bound observation", "level": "direct",
                "basis": "world", "strength": "decisive", "bridges": [],
            }]},
            "discharge": discharge,
            "judger": {"sufficient": "yes", "faithful": "yes", "degradation": "n/a", "reason": "Complete authenticated observations decide the exact obligation."},
        })
        self.prompts = []
        self.registries = []

    def complete(self, role_name, prompt, variables):
        self.prompts.append((role_name, prompt, dict(variables)))
        self.registries.append(dict(self.roles))
        return super().complete(role_name, prompt, variables)


def make_state(tmp_path):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text(TASK)
    (state / "final.txt").write_text("An exact task deliverable.")
    (state / "hook_feed.jsonl").write_text("")
    return str(state)


def fixture_factory(backend):
    def factory(service):
        from verifier.mcp_server import _default_kernel
        return _default_kernel(service, model="fixture", backend=backend,
                               parallel_claims=1, parallel_obligations=1)
    return factory


def test_default_config_uses_original_fields_and_hash_formula(monkeypatch):
    monkeypatch.setattr(offline, "pipeline_fingerprint", lambda: "frozen-baseline")
    fields = offline.config_fields(model="fixture", seats="codex", critic_rounds=3)
    original = {
        "model": "fixture", "seats": "codex", "effort": "medium",
        "service_tier": "standard", "max_rounds": 3, "critic_rounds": 3,
    }
    original["config"] = content_hash({"pipeline": "frozen-baseline", **original})[:16]
    assert fields == original
    assert fields == offline.config_fields(model="fixture", seats="codex", critic_rounds=3, profile="default")
    assert fields == offline.config_fields(model="fixture", seats="codex", critic_rounds=3, profile="plain")


def test_cyber_config_and_stamps_include_full_profile_identity(monkeypatch):
    monkeypatch.setattr(offline, "pipeline_fingerprint", lambda: "frozen-baseline")
    plain = offline.config_fields(model="fixture", seats="codex")
    cyber = offline.config_fields(model="fixture", seats="codex", profile="cyber")
    assert cyber["profile_id"] == get_profile("cyber").id
    assert cyber["config"] != plain["config"]
    assert offline.config_id(model="fixture", seats="codex", profile="cyber") == cyber["config"]
    assert set(cyber) == set(plain) | {"profile_id"}


def test_cyber_score_selects_all_six_agents_and_ordinary_full_task_scope(tmp_path):
    state = make_state(tmp_path)
    backend = RecordingBackend()
    features = offline.score(state, model="fixture", backend=backend,
                             kernel_factory=fixture_factory(backend), profile="cyber",
                             parallel_claims=1, parallel_obligations=1)
    assert features["verdict"] == "verified"
    assert features["profile_id"] == get_profile("cyber").id
    assert {role for role, _, _ in backend.prompts} == set(ALL_AGENT_ROLES)
    expected_roles = get_profile("cyber").apply_roles(ALL_AGENT_ROLES)
    for registry in backend.registries:
        assert all(registry[name].system == expected_roles[name].system for name in registry)
    for role, prompt, variables in backend.prompts:
        context = json.loads(variables["profile_context"])
        assert context["certificate_scope"] == "task"
        assert "subject" not in context
        assert variables["profile_context"] in prompt
    extraction = json.loads((Path(state) / "extract.json").read_text())
    verification = json.loads((Path(state) / "verification.json").read_text())
    assert extraction["profile_id"] == verification["profile_id"] == features["profile_id"]
    events = list(EventLog(str(Path(state) / "events.jsonl")).replay())
    assert len([event for event in events if event["type"] == "verification_profile"]) == 1
    certificate = next(event["payload"] for event in events if event["type"] == "certificate")
    from verifier.execution.instruments import ArtifactStore
    artifact = ArtifactStore(str(Path(state) / "artifacts")).get(certificate["digest"])
    assert artifact["profile_metadata"]["certificate_scope"] == "task"


def test_default_offline_stamps_and_extractor_variables_remain_original(tmp_path):
    state = make_state(tmp_path)
    backend = RecordingBackend()
    backend.roles = dict(ALL_AGENT_ROLES)
    features = offline.score(state, model="fixture", backend=backend,
                             kernel_factory=fixture_factory(backend), profile="default",
                             parallel_claims=1, parallel_obligations=1)
    assert features["verdict"] == "verified"
    assert "profile_id" not in features
    assert all("profile_context" not in variables for _, _, variables in backend.prompts)
    meta = json.loads((Path(state) / "extract.json").read_text())
    assert set(meta) == {"rounds", "commits", "fingerprint", "config", "immutable_input_seal"}
    assert not any(event["type"] == "verification_profile" for event in EventLog(str(Path(state) / "events.jsonl")).replay())


def test_cyber_pin_survives_extraction_cache_reset_and_rejects_cross_profile(tmp_path):
    state = make_state(tmp_path)
    backend = RecordingBackend()
    arguments = dict(model="fixture", backend=backend,
                     kernel_factory=fixture_factory(backend), profile="cyber",
                     parallel_claims=1, parallel_obligations=1)
    offline.score(state, **arguments)
    path = Path(state) / "extract.json"
    meta = json.loads(path.read_text())
    meta["config"] = "stale-derived-cache"
    path.write_text(json.dumps(meta))
    previous = len(backend.calls)
    assert offline.score(state, **arguments)["verdict"] == "verified"
    assert any(call["role"] == "extract" for call in backend.calls[previous:])
    events = list(EventLog(str(Path(state) / "events.jsonl")).replay())
    assert len([event for event in events if event["type"] == "verification_profile"]) == 1
    with pytest.raises(ValueError, match="different verification profile"):
        offline.score(state, model="fixture", backend=backend,
                      kernel_factory=fixture_factory(backend), profile="default")


def test_extractor_preserves_hook_effect_numbering_when_it_retains_profile_pin(tmp_path):
    state = make_state(tmp_path)
    pin = {"profile": "cyber", "profile_id": get_profile("cyber").id}
    EventLog(str(Path(state) / "events.jsonl")).append("verification_profile", pin)
    (Path(state) / "hook_feed.jsonl").write_text(json.dumps({
        "type": "tool_result", "payload": {"tool": "fixture", "excerpt": "result"}
    }) + "\n")
    offline.extract(state, backend=RecordingBackend(), profile="cyber")
    events = list(EventLog(str(Path(state) / "events.jsonl")).replay())
    assert events[0]["type"] == "tool_result" and events[0]["seq"] == 1
    assert events[1]["type"] == "verification_profile"


def test_cli_cyber_score_forwards_profile(monkeypatch, capsys):
    received = {}
    def score(state_dir, **kwargs):
        received.update(state_dir=state_dir, **kwargs)
        return {"verdict": "verified"}
    monkeypatch.setattr(offline, "score", score)
    assert cli.main(["score", "--state-dir", "/tmp/fixture", "--cyber"]) == 0
    assert received["profile"] == "cyber"
    capsys.readouterr()


def test_cli_cyber_certify_imports_trace_and_keeps_full_task_scope(tmp_path, capsys):
    source = tmp_path / "source.jsonl"
    EventLog(str(source)).append("commit", COMMITS["commits"][0])
    state = tmp_path / "certify-state"
    state.mkdir()
    (state / "task.md").write_text(TASK)
    backend = RecordingBackend()
    assert cli.certify(str(source), str(state), model="fixture", profile="cyber",
                       kernel_factory=fixture_factory(backend)) == 0
    assert json.loads(capsys.readouterr().out)["verdict"] == "verified"
    assert {role for role, _, _ in backend.prompts} == set(ALL_AGENT_ROLES) - {"extract"}
    assert all(json.loads(variables["profile_context"])["certificate_scope"] == "task"
               for _, _, variables in backend.prompts)


def test_cli_certify_rejects_cross_profile_before_copy_erases_state_pin(tmp_path):
    source = tmp_path / "source.jsonl"
    EventLog(str(source)).append("commit", COMMITS["commits"][0])
    state = tmp_path / "certify-state"
    state.mkdir()
    target = state / "events.jsonl"
    EventLog(str(target)).append("verification_profile", {
        "profile": "cyber", "profile_id": get_profile("cyber").id,
    })
    before = target.read_bytes()
    with pytest.raises(ValueError, match="different verification profile"):
        cli.certify(str(source), str(state), profile="default")
    assert target.read_bytes() == before


@pytest.mark.parametrize("profile,expected_timeout", [("default", 1800), ("cyber", None)])
def test_offline_codex_constructors_respect_profile_timeout_and_search_policy(tmp_path, monkeypatch, profile, expected_timeout):
    from verifier.llm import codex_backend
    instances = []

    class FakeCodexBackend(RecordingBackend):
        def __init__(self, **kwargs):
            super().__init__()
            self.timeout = kwargs.get("timeout", 1800)
            self.service_tier = kwargs.get("service_tier", "standard")
            self.search_roles = frozenset({"extract", "obligations", "critic", "oracles"})
            self.roles = dict(ALL_AGENT_ROLES)
            instances.append(self)

    monkeypatch.setattr(codex_backend, "CodexBackend", FakeCodexBackend)
    state = make_state(tmp_path)
    features = offline.score(state, model="fixture", seats="codex", profile=profile,
                             parallel_claims=1, parallel_obligations=1)
    assert features["verdict"] == "verified"
    assert len(instances) >= 2
    assert all(instance.timeout == expected_timeout for instance in instances)
    if profile == "cyber":
        assert all(not instance.search_roles for instance in instances)
    else:
        assert all(instance.search_roles for instance in instances)
