"""General cyber seat policy, independent of any benchmark or model call."""
import os
import tomllib

import pytest

from verifier.mcp_server import Service
from verifier.profiles.cyber.backend import CyberCodexBackend, PLANNER_ROLES


def grants_for(backend, role):
    args = backend._execution_policy_args(role, {"world_mode": "live"})
    table = next(value.split("=", 1)[1] for value in args if ".filesystem=" in value)
    return tomllib.loads("grants=" + table)["grants"], args


@pytest.mark.parametrize("role", sorted(PLANNER_ROLES))
def test_general_planners_only_receive_authority_and_own_scratch(tmp_path, role):
    state, public = tmp_path / "state", tmp_path / "public"
    state.mkdir()
    public.mkdir()
    backend = CyberCodexBackend(cwd=str(state), project_root=str(public), model="fixture")
    grants, args = grants_for(backend, role)
    assert grants[str(state)] == grants[str(public)] == "deny"
    assert grants[str(state / "record")] == grants[str(state / "source_evidence")] == "deny"
    for name in ("task.md", "baseline", "oracle_catalog"):
        assert grants[str(state / name)] == "read"
    assert grants[str(state / "_blind" / role)] == "write"
    assert backend._execution_cwd(role, {"world_mode": "live"}) == str(state / "_blind" / role)
    assert any(value.endswith(".network.enabled=false") for value in args)
    assert "--dangerously-bypass-approvals-and-sandbox" not in args


@pytest.mark.parametrize("role", ["discharge", "judger"])
def test_public_source_is_readable_not_writable_by_execution_seats(tmp_path, role):
    state, public = tmp_path / "state", tmp_path / "public"
    state.mkdir()
    public.mkdir()
    backend = CyberCodexBackend(cwd=str(state), project_root=str(public), model="fixture")
    grants, _ = grants_for(backend, role)
    assert grants[str(public)] == "read"
    assert grants[str(state / "verifier-home")] == "deny"
    assert all(grants[path] == "deny" for path in backend.DENIED_ROOTS)


@pytest.mark.parametrize("private", CyberCodexBackend.DENIED_ROOTS)
def test_declaring_private_project_cannot_grant_execution_access(tmp_path, private):
    backend = CyberCodexBackend(cwd=str(tmp_path), project_root=private, model="fixture")
    grants, _ = grants_for(backend, "discharge")
    assert grants[private] == "deny"


def test_default_cyber_service_uses_shared_seat_policy_without_sidecar(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("policy construction must not run any agent")

    monkeypatch.setattr(CyberCodexBackend, "complete", forbidden)
    state, public = tmp_path / "state", tmp_path / "public"
    public.mkdir()
    service = Service(str(state), profile="cyber", seats="codex", project_root=str(public),
                      model="fixture", effort="max", service_tier="fast")
    assert type(service.kernel.backend) is CyberCodexBackend
    assert service.kernel.backend.project_root == os.fspath(public)
    assert service.kernel.backend.timeout is None
    assert not service.kernel.backend.search_roles
    from verifier.profiles.cyber.feedback import filter_oracle_feedback
    assert service.kernel.binder.oracle_feedback_filter is filter_oracle_feedback


def test_cyber_profile_pins_executable_boundaries_but_plain_does_not(monkeypatch):
    from verifier import profiles
    cyber = profiles.get_profile("cyber")
    plain = profiles.get_profile("plain")
    before_cyber, before_plain = cyber.id, plain.id
    assert set(cyber.identity["cyber_boundary"]) == {"adapters.py", "backend.py", "feedback.py"}
    assert "cyber_boundary" not in plain.identity
    monkeypatch.setattr(profiles, "_cyber_boundary_identity", lambda: {"policy": "changed-fixture"})
    assert cyber.id != before_cyber
    assert plain.id == before_plain


@pytest.mark.parametrize("role,tool_grant", [("discharge", "write"), ("judger", "read")])
def test_general_cyber_preserves_explicit_operator_toolchain_scope(tmp_path, role, tool_grant):
    state = tmp_path / "state"
    state.mkdir()
    tools, cache = tmp_path / "operator-tools", tmp_path / "operator-cache"
    backend = CyberCodexBackend(cwd=str(state), model="fixture",
        offline_toolchain_dir=str(tools), offline_shared_cache_dir=str(cache))
    args = backend._execution_policy_args(role, {
        "world_mode": "live", "toolchain_dir": str(tools), "shared_cache_dir": str(cache)})
    table = next(value.split("=", 1)[1] for value in args if ".filesystem=" in value)
    grants = tomllib.loads("grants=" + table)["grants"]
    assert grants[str(tools)] == tool_grant
    assert grants.get(str(cache)) == ("write" if role == "discharge" else None)


@pytest.mark.parametrize("private", CyberCodexBackend.DENIED_ROOTS)
@pytest.mark.parametrize("role", ["discharge", "judger"])
def test_operator_child_grants_cannot_override_private_root_denies(tmp_path, monkeypatch, private, role):
    backend = CyberCodexBackend(cwd=str(tmp_path), model="fixture")
    monkeypatch.setattr(backend, "_offline_tool_roots", lambda _: (private + "/tools", None))
    from verifier.llm.schema import RoleError
    with pytest.raises(RoleError, match="private evidence boundary"):
        backend._execution_policy_args(role, {"world_mode": "live"})
