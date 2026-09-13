"""Host runtime context/policy wiring, not sanitizer or pin-validation proof.

The already selected immutable host object is simulated here. Real file
validation and real runtime controls have separate tests and native gates.
No models, compilers, target programs, evaluators or Docker are invoked.
"""
import json
from types import SimpleNamespace
import tomllib

import pytest

from verifier.mcp_server import Service
from verifier.profiles.cyber import runtime
from verifier.profiles.cyber.backend import CyberCodexBackend, PLANNER_ROLES
from verifier.profiles.cyber.evidence import (
    EvidenceBindingError, SeatCapabilities, build_evidence_manifest,
)


@pytest.fixture
def selected_runtime(tmp_path, monkeypatch):
    tools = tmp_path / "public-runtime"
    tools.mkdir()
    # Isolate only the selected-object wiring: this test does NOT establish
    # root ownership, runtime validity, or successful sanitizer execution.
    selected = object.__new__(runtime.CyberRuntimeInventory)
    object.__setattr__(selected, "_inventory_json", json.dumps({
        "schema": "fixture-host-runtime",
        "files": {"runner": {"path": str(tools / "runner"), "sha256": "1" * 64},
                  "launcher": {"path": str(tools / "launcher"), "sha256": "2" * 64}},
        "capabilities": {"requested_leak_check_policy": "disabled_unchecked",
                         "future_target_compatibility_verified": False},
    }))
    object.__setattr__(selected, "_manifest", SimpleNamespace(path=str(tools / "manifest.json")))
    object.__setattr__(selected, "_files", ())
    checks = []
    monkeypatch.setattr(runtime.CyberRuntimeInventory, "validate_current",
                        lambda self: checks.append(self))
    return selected, tools, checks


def manifest(selected=None, *, blind=False):
    return build_evidence_manifest(evaluator_spec=None,
        capabilities=SeatCapabilities(execute_local_checks=True),
        answer_blind=blind, seat_runtime=selected)


def policy(backend, role, **variables):
    args = backend._execution_policy_args(role, {"world_mode": "live", **variables})
    table = next(value.split("=", 1)[1] for value in args if ".filesystem=" in value)
    return tomllib.loads("grants=" + table)["grants"], args


def test_none_preserves_actual_legacy_evidence_inventory():
    before = build_evidence_manifest(evaluator_spec=None,
                                    capabilities=SeatCapabilities(execute_local_checks=True))
    assert manifest() == before
    assert "seat_runtime" not in before["inventory"]


def test_host_runtime_is_revision_bound_not_an_observed_result(selected_runtime):
    selected, _, checks = selected_runtime
    executor, blind = manifest(selected), manifest(selected, blind=True)
    assert checks and all(item is selected for item in checks)
    assert executor["capability_revision"] == blind["capability_revision"]
    assert executor["capability_revision"] != manifest()["capability_revision"]
    assert executor["inventory"]["seat_runtime"] == selected.as_inventory()
    assert "executions" not in blind and executor["executions"] == []
    assert executor["inventory"]["retained_execution"]["automatic_candidate_verdict"] is False
    executor["inventory"]["seat_runtime"]["files"]["runner"]["sha256"] = "2" * 64
    assert manifest(selected)["capability_revision"] == blind["capability_revision"]


@pytest.mark.parametrize("value", [{}, True, "solver-provided-manifest", SimpleNamespace(as_inventory=lambda: {})])
def test_runtime_manifest_cannot_be_supplied_as_solver_json(value):
    with pytest.raises(EvidenceBindingError, match="host-selected"):
        manifest(value)


def test_default_cyber_kernel_and_context_share_the_host_runtime(tmp_path, selected_runtime):
    selected, _, checks = selected_runtime
    state, public = tmp_path / "state", tmp_path / "task"
    public.mkdir()
    service = Service(str(state), profile="cyber", seats="codex", model="fixture",
                      project_root=str(public), seat_runtime=selected)
    assert service.seat_runtime is selected
    assert service.kernel.backend.runtime_inventory is selected
    assert checks
    assert service._evidence_manifest("oracles")["inventory"]["seat_runtime"] == selected.as_inventory()
    assert service._evidence_manifest("oracles")["capability_revision"] == (
        service._evidence_manifest("judger")["capability_revision"])


def test_plain_service_cannot_advertise_a_cyber_runtime(tmp_path, selected_runtime):
    with pytest.raises(ValueError, match="host-configured cyber"):
        Service(str(tmp_path / "state"), profile="default", seat_runtime=selected_runtime[0])


@pytest.mark.parametrize("role", ["discharge", "judger"])
def test_execution_seats_get_only_a_read_grant_not_global_runtime_overrides(
        tmp_path, selected_runtime, role):
    selected, tools, _ = selected_runtime
    state, public = tmp_path / "state", tmp_path / "task"
    state.mkdir()
    public.mkdir()
    backend = CyberCodexBackend(cwd=str(state), project_root=str(public),
                               model="fixture", runtime_inventory=selected)
    grants, args = policy(backend, role)
    assert grants[str(tools)] == "read" and grants[str(public)] == "read"
    assert grants[str(tools / "launcher")] == "read"
    assert all(grants[path] == "deny" for path in backend.DENIED_ROOTS)
    assert not any(path == "/proc" or path.startswith("/proc/") for path in grants)
    assert any(value.endswith(".network.enabled=false") for value in args)
    environment = tomllib.loads(next(value for value in args
                    if value.startswith("shell_environment_policy.set=")))["shell_environment_policy"]["set"]
    assert "LD_PRELOAD" not in environment and "ASAN_OPTIONS" not in environment
    assert "--dangerously-bypass-approvals-and-sandbox" not in args


@pytest.mark.parametrize("role", sorted(PLANNER_ROLES))
def test_planners_do_not_receive_runtime_execution_or_result_access(tmp_path, selected_runtime, role):
    selected, tools, _ = selected_runtime
    state = tmp_path / "state"
    state.mkdir()
    backend = CyberCodexBackend(cwd=str(state), model="fixture", runtime_inventory=selected)
    grants, _ = policy(backend, role)
    assert grants[":root"] == "deny" and grants[str(tools)] == "deny"
    assert grants[str(tools / "launcher")] == "deny"
    assert grants[str(state)] == "deny"
    assert grants[str(state / "artifacts")] == "deny"


def test_method_variables_cannot_select_a_runtime_root(tmp_path, selected_runtime):
    selected, tools, _ = selected_runtime
    state = tmp_path / "state"
    state.mkdir()
    backend = CyberCodexBackend(cwd=str(state), model="fixture", runtime_inventory=selected)
    grants, _ = policy(backend, "discharge", runtime_inventory={"root": "/evaluator"},
                       seat_runtime="/proc", runtime_root="/codex-home")
    assert grants[str(tools)] == "read"
    assert grants["/evaluator"] == grants["/codex-home"] == "deny"
    assert "/proc" not in grants


def test_runtime_pin_change_fails_before_a_seat_can_run(tmp_path, selected_runtime, monkeypatch):
    selected, _, _ = selected_runtime
    state = tmp_path / "state"
    state.mkdir()
    backend = CyberCodexBackend(cwd=str(state), model="fixture", runtime_inventory=selected)
    def changed(_self):
        raise runtime.RuntimeInventoryError("runtime_file_changed")
    monkeypatch.setattr(runtime.CyberRuntimeInventory, "validate_current", changed)
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_file_changed"):
        policy(backend, "discharge")
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_file_changed"):
        manifest(selected, blind=True)
