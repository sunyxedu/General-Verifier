"""Complete Oracle plans are durable diagnostics, never accepted evidence.

All seats are deterministic fixtures. No models, target programs or Docker.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from verifier.core.events import EventLog
from verifier.core.model import BASIS_RECORD, BASIS_WORLD, CLAIM_FINAL, Claim, Obligation, content_hash
from verifier.core.store import ClaimStore
from verifier.engine.certificate import request_certificate
from verifier.engine.game import BindingGame, ORACLE_METHOD_PLAN_SCHEMA
from verifier.engine.kernel import Kernel
from verifier.execution.adapters import AdapterRoutingDischarger, default_adapter_registry
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, RoleError


ROW = {"id": "claim-plan", "kind": "assertion", "text": "the subject meets its contract"}
YES = {"sufficient": "yes", "faithful": "yes", "degradation": "n/a", "reason": "fresh fixture audit"}


def method(*, procedure=None, adapter=False):
    value = {
        "procedure": procedure or ("Inspect the complete independently retained observation. " * 60 + "LAST λ"),
        "artifact": "Complete inputs and output, including the final promised field. " * 35 + "ARTIFACT END",
        "level": "independent fixture inspection",
        "basis": BASIS_WORLD,
        "strength": "decisive",
        "bridges": ["A declared semantic bridge " * 20 + "BRIDGE END"],
    }
    if adapter:
        value.update(adapter="evidence.program", adapter_version=1, adapter_input_binding="fixed-v1",
                     adapter_arguments=json.dumps({"version": 1, "steps": [{"operation": "inspect", "field": "complete"}]}),
                     adapter_inputs=[{"ref": "task-given:fixture", "provenance": "task_given", "role": "input"}])
    return value


def fixture(tmp_path, *, payload=None, discharger=None, judger=None):
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("Check the subject's complete contract.", encoding="utf-8")
    artifacts = ArtifactStore(str(state / "artifacts"))
    log = EventLog(str(state / "events.jsonl"))
    backend = FixtureBackend({"oracles": payload if payload is not None else {"methods": [method()]},
                              "judger": judger if judger is not None else YES})
    game = BindingGame(backend, artifacts, log, discharger, live_world=True)
    return game, backend, artifacts, log


def plans(log, artifacts):
    result = []
    for event in log.replay():
        if event["type"] != "methods_enumerated":
            continue
        payload = event["payload"]
        assert payload["method_plan_schema"] == ORACLE_METHOD_PLAN_SCHEMA
        digest = payload["method_plan_artifact_id"]
        plan = artifacts.get(digest)
        assert content_hash(plan) == digest
        assert plan["schema"] == ORACLE_METHOD_PLAN_SCHEMA
        assert plan["instrument"] == "oracle-method-plan"
        assert plan["execution_state"] == "unexecuted"
        assert not {"outcome", "verdict", "holds", "audit", "judger_approved", "bound"} & plan.keys()
        result.append((event, digest, plan))
    return result


@pytest.mark.parametrize("adapter", [False, True])
def test_complete_prepared_plan_is_available_while_discharge_blocks(tmp_path, adapter):
    original = method(adapter=adapter)
    prepared = deepcopy(original)
    prepared["procedure"] += " PREPARED PROCEDURE"
    prepared["artifact"] += " PREPARED ARTIFACT"
    prepared["bridges"].append("Prepared bridge")
    entered, release = threading.Event(), threading.Event()

    class BlockingDischarger:
        def prepare_method(self, value):
            assert value == original
            return deepcopy(prepared)

        def discharge(self, row, obligation, method=None, tag=""):
            assert method == prepared
            entered.set()
            assert release.wait(3), "fixture release was not signaled"
            raise RuntimeError("fixture executor failed before settlement")

    game, backend, artifacts, log = fixture(tmp_path, payload={"methods": [original]}, discharger=BlockingDischarger())
    obligation = Obligation(ROW["id"], "the exact complete contract is true", BASIS_WORLD)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(game.run, ROW, obligation)
        try:
            assert entered.wait(2), "fixture executor did not start"
            retained = plans(log, artifacts)
            assert len(retained) == 1
            event, _, plan = retained[0]
            assert plan["methods"] == [prepared]
            assert plan["claim"] == ROW["id"] and plan["obligation"] == obligation.id
            assert plan["required_basis"] == BASIS_WORLD and plan["repair"] is False
            summary = event["payload"]["methods"][0]
            assert summary["procedure"] == prepared["procedure"][:80]
            assert summary["level"] == prepared["level"]
            assert summary["basis"] == prepared["basis"]
            assert summary["strength"] == prepared["strength"]
            if adapter:
                assert plan["methods"][0]["adapter_arguments"] == prepared["adapter_arguments"]
                assert summary["adapter"] == prepared["adapter"]
            assert not any(e["type"] == "method_bound" for e in log.replay())
            assert not any(c["role"] == "judger" for c in backend.calls)
        finally:
            release.set()
        with pytest.raises(RuntimeError, match="fixture executor failed"):
            running.result()
    assert len(plans(log, artifacts)) == 1
    assert not any(e["type"] == "method_bound" for e in log.replay())


def test_later_input_return_and_loaded_copy_mutation_cannot_change_plan(tmp_path):
    original = method(adapter=True)
    payload = {"methods": [original]}
    game, _, artifacts, log = fixture(tmp_path, payload=payload)
    returned = game._enumerate(ROW, Obligation(ROW["id"], "complete contract", BASIS_WORLD))
    _, digest, retained = plans(log, artifacts)[0]
    expected = deepcopy(retained)
    raw_before = Path(artifacts.root, digest + ".json").read_bytes()
    original["procedure"] = "caller mutation"
    original["bridges"].append("caller nested mutation")
    returned[0]["adapter_inputs"][0]["ref"] = "returned nested mutation"
    retained["methods"][0]["artifact"] = "loaded copy mutation"
    assert artifacts.get(digest) == expected
    assert Path(artifacts.root, digest + ".json").read_bytes() == raw_before
    assert content_hash(artifacts.get(digest)) == digest


def test_real_fixed_adapter_preparation_is_preserved_without_executing_recipe(tmp_path):
    original = method(procedure="Inspect the exact retained ledger event.")
    original.update(basis=BASIS_RECORD, bridges=[], adapter="ledger.event", adapter_version=1,
                    adapter_input_binding="fixed-v1", adapter_inputs=[],
                    adapter_arguments=json.dumps({"event_seq": 1, "event_type": "tool_call",
                                                  "payload_contains": {"input": {"value": "harmless"}}}))
    game, backend, artifacts, log = fixture(tmp_path, payload={"methods": [original]})
    registry = default_adapter_registry()
    expected = registry.bind_inputs(original)
    assert expected["adapter_inputs"] and original["adapter_inputs"] == []
    game.discharger = AdapterRoutingDischarger(None, registry, artifacts, log,
                                               str(tmp_path / "state"), live_world=True)
    returned = game._enumerate(ROW, Obligation(ROW["id"], "the complete retained event matches", BASIS_RECORD))
    assert returned == [expected]
    assert plans(log, artifacts)[0][2]["methods"] == [expected]
    assert original["adapter_inputs"] == []
    assert [call["role"] for call in backend.calls] == ["oracles"]
    assert not any(event["type"] in ("adapter_discharge", "discharge", "method_bound") for event in log.replay())


@pytest.mark.parametrize("payload", [{}, {"methods": [{"procedure": "incomplete"}]}, {"methods": "not a menu"}])
def test_failed_oracle_parse_never_publishes_a_plan(tmp_path, payload):
    game, _, artifacts, log = fixture(tmp_path, payload=payload)
    with pytest.raises(RoleError):
        game._enumerate(ROW, Obligation(ROW["id"], "complete contract"))
    assert plans(log, artifacts) == []
    assert list(Path(artifacts.root).glob("*.json")) == []


def test_failed_oracle_call_never_publishes_a_plan(tmp_path):
    def failed(_):
        raise RoleError("fixture generation failed")

    game, _, artifacts, log = fixture(tmp_path, payload=failed)
    with pytest.raises(RoleError):
        game._enumerate(ROW, Obligation(ROW["id"], "complete contract"))
    assert plans(log, artifacts) == []
    assert list(Path(artifacts.root).glob("*.json")) == []


def test_failed_prepare_does_not_publish_a_partially_prepared_menu(tmp_path):
    class FailedPrepare:
        def prepare_method(self, value):
            if value["procedure"] == "second":
                raise RoleError("fixture prepare failed")
            return value

    game, _, artifacts, log = fixture(tmp_path, payload={"methods": [method(procedure="first"), method(procedure="second")]},
                                       discharger=FailedPrepare())
    with pytest.raises(RoleError, match="fixture prepare failed"):
        game._enumerate(ROW, Obligation(ROW["id"], "complete contract"))
    assert plans(log, artifacts) == []
    assert list(Path(artifacts.root).glob("*.json")) == []


def test_claim_obligation_basis_and_repair_have_exact_separate_locators(tmp_path):
    game, _, artifacts, log = fixture(tmp_path)
    one = Obligation(ROW["id"], "first contract", BASIS_WORLD)
    two = Obligation(ROW["id"], "second contract", BASIS_RECORD)
    game._enumerate(ROW, one)
    game._enumerate(ROW, two)
    game._enumerate(ROW, one, method_feedback="structural repair", oracle_round=1)
    other_row = {**ROW, "id": "another-claim"}
    game._enumerate(other_row, Obligation(other_row["id"], "first contract", BASIS_WORLD))
    records = plans(log, artifacts)
    assert len({digest for _, digest, _ in records}) == 4
    assert [(p["claim"], p["obligation"], p["required_basis"], p["repair"]) for _, _, p in records] == [
        (ROW["id"], one.id, BASIS_WORLD, False),
        (ROW["id"], two.id, BASIS_RECORD, False),
        (ROW["id"], one.id, BASIS_WORLD, True),
        (other_row["id"], Obligation(other_row["id"], "first contract", BASIS_WORLD).id, BASIS_WORLD, False),
    ]


def test_empty_valid_menu_is_an_empty_unexecuted_plan(tmp_path):
    game, _, artifacts, log = fixture(tmp_path, payload={"methods": []})
    assert game._enumerate(ROW, Obligation(ROW["id"], "complete contract")) == []
    assert plans(log, artifacts)[0][2]["methods"] == []


@pytest.mark.parametrize("accepted", [False, True])
def test_normal_kernel_certificate_still_waits_for_fresh_judger(tmp_path, accepted):
    entered, release = threading.Event(), threading.Event()

    def judger(_):
        entered.set()
        assert release.wait(3), "fixture Judger was not released"
        return {**YES, "sufficient": "yes" if accepted else "no"}

    class ObservingDischarger:
        def discharge(self, row, obligation, method=None, tag=""):
            return DischargeResult(outcome="holds", witness="independent fixture observation", witness_kind="execution",
                                   artifact=observation_id, instrument="fixture", basis=BASIS_WORLD)

    game, backend, artifacts, log = fixture(tmp_path, discharger=ObservingDischarger(), judger=judger)
    observation_id = artifacts.put({"instrument": "fixture-observation", "observed": "harmless fixture fact"})
    backend.fixtures["obligations"] = {"obligations": [{"text": "the complete subject condition", "required_basis": BASIS_WORLD}]}
    store = ClaimStore()
    final_id = store.add_claim(Claim(CLAIM_FINAL, "the subject meets the complete task"))
    kernel = Kernel(store, log, backend, artifacts, critic_role=None, binder=game, discharger=game.discharger)
    with ThreadPoolExecutor(max_workers=1) as pool:
        running = pool.submit(request_certificate, store, kernel, log, artifacts, final_id)
        try:
            assert entered.wait(2), "fresh fixture Judger did not start"
            assert plans(log, artifacts)
            assert store.status(final_id) != "verified"
            assert not running.done()
            assert not any(e["type"] in ("method_bound", "certificate") for e in log.replay())
        finally:
            release.set()
        certificate = running.result()
    assert certificate["verdict"] == ("verified" if accepted else "unverified")
    assert any(c["role"] == "judger" for c in backend.calls)
    plan_ids = {digest for _, digest, _ in plans(log, artifacts)}
    for entry in certificate["slice"]:
        for evidence in entry["evidence"]:
            assert evidence.get("artifact") not in plan_ids
    if accepted:
        binding_ids = [e["artifact"] for child in certificate["slice"][0]["obligations"] for e in child["evidence"]]
        assert binding_ids and not plan_ids.intersection(binding_ids)
        for binding_id in binding_ids:
            binding = artifacts.get(binding_id)
            assert binding["instrument"] == "binding-game"
            assert binding["bound"]["discharge_artifact"] == observation_id
