from __future__ import annotations

import os
from pathlib import Path

from verifier.core.events import EventLog
from verifier.core.model import Claim, Obligation, CLAIM_ASSERTION
from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
from verifier.engine.binding.game import BindingGame
from verifier.engine.kernel import Kernel
from verifier.engine.source_evidence import (
    SourceEvidenceView,
    bind_source_evidence_variables,
)
from verifier.execution.discharge import AgentDischarger, DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.agents import AgentRole, CRITIC_AGENT
from verifier.llm.schema import FixtureBackend


COMPILE = {"obligations": [{
    "text": "the retained action exists", "required_basis": "record",
}]}
PASS = {
    "verdict": "pass",
    "counterexample_world": "n/a",
    "missing_condition": "n/a",
}
ORACLE = {"methods": [{
    "procedure": "inspect the complete retained record",
    "artifact": "exact observation",
    "level": "deterministic",
    "basis": "record",
    "strength": "decisive",
    "bridges": [],
}]}
JUDGE = {
    "sufficient": "yes",
    "faithful": "yes",
    "degradation": "n/a",
    "reason": "all required audits pass",
}


class StubDischarger:
    def discharge(self, row, obligation, method=None, tag=""):
        return DischargeResult(
            outcome="holds", witness="retained", artifact="d1",
            instrument="stub", basis="record", witness_kind="execution",
        )


def _assert_bound(call, raw_ledger):
    variables = call["variables"]
    assert variables["_mixed_ledger_path"] == str(raw_ledger)
    assert type(variables["_source_evidence_view"]) is SourceEvidenceView
    assert variables["ledger_path"].endswith("/manifest.json")
    assert variables["_source_evidence_view"].revalidate()


def test_compiler_and_critic_use_fresh_lossless_transport(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({"obligations": COMPILE, "critic": PASS})
    backend.compact_inspection = True
    Kernel(
        store, log, backend, ArtifactStore(str(tmp_path / "artifacts")),
        critic_role=CRITIC_AGENT, discharger=StubDischarger(),
    )

    store.add_claim(Claim(CLAIM_ASSERTION, "the retained action exists"))

    compiler, critic = backend.calls[:2]
    assert compiler["role"] == "obligations"
    assert critic["role"] == "critic"
    _assert_bound(compiler, Path(log.path))
    _assert_bound(critic, Path(log.path))
    assert (compiler["variables"]["_source_evidence_view"].view_id
            == critic["variables"]["_source_evidence_view"].view_id)


def test_critic_rebinds_after_source_changes_during_compiler_attempt(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "first"})
    calls = {"compiler": 0}

    def compile_and_change_source(_variables):
        calls["compiler"] += 1
        if calls["compiler"] == 1:
            log.append("tool_result", {"output": "second"})
        return COMPILE

    backend = FixtureBackend({
        "obligations": compile_and_change_source,
        "critic": PASS,
    })
    backend.compact_inspection = True
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    Kernel(
        store, log, backend, ArtifactStore(str(tmp_path / "artifacts")),
        critic_role=CRITIC_AGENT, discharger=StubDischarger(),
    )

    store.add_claim(Claim(CLAIM_ASSERTION, "the retained action exists"))

    compiler_calls = [call for call in backend.calls
                      if call["role"] == "obligations"]
    critic_call = next(call for call in backend.calls
                       if call["role"] == "critic")
    assert len(compiler_calls) == 2
    assert (compiler_calls[-1]["variables"]["_source_evidence_view"].view_id
            == critic_call["variables"]["_source_evidence_view"].view_id)


def test_oracle_and_fresh_judger_receive_the_same_source_protocol(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("tool_call", {"tool": "shell", "command": "pwd"})
    backend = FixtureBackend({"oracles": ORACLE, "judger": JUDGE})
    backend.compact_inspection = True
    game = BindingGame(
        backend,
        ArtifactStore(str(tmp_path / "artifacts")),
        log,
        StubDischarger(),
    )

    result = game.run(
        {"id": "c1", "kind": "assertion", "text": "action exists"},
        Obligation("c1", "the retained action exists", "record"),
    )

    assert result.outcome == "holds"
    oracle = next(call for call in backend.calls if call["role"] == "oracles")
    judger = next(call for call in backend.calls if call["role"] == "judger")
    _assert_bound(oracle, Path(log.path))
    _assert_bound(judger, Path(log.path))
    assert (oracle["variables"]["_source_evidence_view"].view_id
            == judger["variables"]["_source_evidence_view"].view_id)


def test_agent_executor_gets_view_without_changing_its_role(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    backend = FixtureBackend({"discharge": {
        "outcome": "holds",
        "witness": "observed",
        "artifact": "",
        "assumed_premises": [],
    }})
    backend.compact_inspection = True
    discharger = AgentDischarger(
        backend,
        ArtifactStore(str(tmp_path / "artifacts")),
        state_dir=str(tmp_path),
        workdir_base=str(tmp_path / "work"),
        log=log,
    )

    result = discharger.discharge(
        {"id": "c1", "kind": "assertion", "text": "action exists"},
        Obligation("c1", "the retained action exists", "record"),
    )

    assert result.outcome == "holds"
    call, = backend.calls
    assert call["role"] == "discharge"
    _assert_bound(call, Path(log.path))


def test_unsafe_optional_projection_falls_back_to_full_ledger(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    os.link(log.path, tmp_path / "alias.jsonl")
    original = {"ledger_path": os.path.abspath(log.path), "value": 1}

    bound = bind_source_evidence_variables(log, original, enabled=True)

    assert bound == original


def test_source_view_change_uses_existing_role_retry_with_a_fresh_view(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "first"})
    variables = bind_source_evidence_variables(
        log, {"ledger_path": os.path.abspath(log.path)}, enabled=True)

    class ChangingBackend:
        def __init__(self):
            self.calls = []

        def complete(self, role_name, prompt, call_variables):
            self.calls.append((role_name, prompt, call_variables))
            if len(self.calls) == 1:
                log.append("tool_result", {"output": "second"})
            return {"ok": True}

    backend = ChangingBackend()
    role = AgentRole("fixture", "ledger={ledger_path}", {"ok": bool})

    assert role.call(backend, variables) == {"ok": True}
    assert len(backend.calls) == 2
    first = backend.calls[0][2]["_source_evidence_view"]
    second = backend.calls[1][2]["_source_evidence_view"]
    assert first.view_id != second.view_id
    assert not first.revalidate()
    assert second.revalidate()
    assert "previous attempt failed" in backend.calls[1][1]


def test_backend_explicit_opt_out_uses_full_ledger_before_call(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    variables = bind_source_evidence_variables(
        log, {"ledger_path": os.path.abspath(log.path)}, enabled=True)

    class OptOutBackend:
        def __init__(self):
            self.capability_calls = []
            self.calls = []

        def allows_compact_source_evidence(
                self, role_name, call_variables):
            self.capability_calls.append((role_name, call_variables))
            return False

        def complete(self, role_name, prompt, call_variables):
            self.calls.append((role_name, prompt, call_variables))
            return {"ok": True}

    backend = OptOutBackend()
    role = AgentRole("fixture", "ledger={ledger_path}", {"ok": bool})

    assert role.call(backend, variables) == {"ok": True}
    assert len(backend.capability_calls) == 1
    assert len(backend.calls) == 1
    call_variables = backend.calls[0][2]
    assert call_variables["ledger_path"] == os.path.abspath(log.path)
    assert "_source_evidence_view" not in call_variables
    assert variables["_source_evidence_view"].revalidate()


def test_non_explicit_source_capabilities_keep_compact_transport(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    variables = bind_source_evidence_variables(
        log, {"ledger_path": os.path.abspath(log.path)}, enabled=True)
    role = AgentRole("fixture", "ledger={ledger_path}", {"ok": bool})

    for behavior in (
            "absent", "true", "none", "unknown", "zero", "lookup-error",
            "call-error"):
        class CompatibleBackend:
            def __init__(self):
                self.calls = []

            @property
            def allows_compact_source_evidence(self):
                if behavior == "absent":
                    raise AttributeError
                if behavior == "lookup-error":
                    raise RuntimeError("optional capability unavailable")

                def capability(_role_name, _call_variables):
                    if behavior == "call-error":
                        raise RuntimeError("optional capability unavailable")
                    return {
                        "true": True,
                        "none": None,
                        "zero": 0,
                    }.get(behavior, "unknown")
                return capability

            def complete(self, role_name, prompt, call_variables):
                self.calls.append((role_name, prompt, call_variables))
                return {"ok": True}

        backend = CompatibleBackend()
        assert role.call(backend, variables) == {"ok": True}
        assert len(backend.calls) == 1
        call_variables = backend.calls[0][2]
        assert call_variables["ledger_path"] == \
            variables["ledger_path"]
        assert call_variables["_source_evidence_view"] is \
            variables["_source_evidence_view"]


def test_opt_out_requires_mixed_path_bound_to_the_source_view(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "candidate source"})
    original = bind_source_evidence_variables(
        log, {"ledger_path": os.path.abspath(log.path)}, enabled=True)
    role = AgentRole("fixture", "ledger={ledger_path}", {"ok": bool})

    for invalid in (
            None, "", "relative/events.jsonl",
            os.path.abspath(str(tmp_path / "different.jsonl"))):
        variables = dict(original)
        variables["_mixed_ledger_path"] = invalid

        class OptOutBackend:
            def __init__(self):
                self.calls = []

            def allows_compact_source_evidence(
                    self, _role_name, _call_variables):
                return False

            def complete(self, role_name, prompt, call_variables):
                self.calls.append((role_name, prompt, call_variables))
                return {"ok": True}

        backend = OptOutBackend()
        assert role.call(backend, variables) == {"ok": True}
        assert len(backend.calls) == 1
        call_variables = backend.calls[0][2]
        assert call_variables["ledger_path"] == original["ledger_path"]
        assert call_variables["_source_evidence_view"] is \
            original["_source_evidence_view"]
