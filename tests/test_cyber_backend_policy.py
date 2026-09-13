"""General cyber seat policy, independent of any benchmark or model call."""
import os
import tomllib

import pytest

from verifier.mcp_server import Service
from verifier.profiles.cyber.backend import (
    CyberCodexBackend, PLANNER_ROLES, _prune_redundant_state_denies,
    _planner_state_allowlist, _PLANNER_RUNTIME_BOUNDARIES,
)


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
    assert grants[":root"] == grants[str(public)] == "deny"
    assert str(state) not in grants
    for name in ("record", "source_evidence", "artifacts", "verifier-home",
                 "events.jsonl", "future-private-transport", "_blind/another-role"):
        assert effective_state_grant(grants, state / name) == "deny"
        assert str(state / name) not in grants
    for name in ("task.md", "baseline", "oracle_catalog"):
        assert grants[str(state / name)] == "read"
    assert grants[str(state / "_blind" / role)] == "write"
    assert backend._execution_cwd(role, {"world_mode": "live"}) == str(state / "_blind" / role)
    assert any(value.endswith(".network.enabled=false") for value in args)
    assert "--dangerously-bypass-approvals-and-sandbox" not in args


def effective_state_grant(grants, path):
    """Independent explicit-rule lookup; fixture paths never use macros."""
    return next((grants[str(parent)] for parent in (path, *path.parents)
                 if str(parent) in grants), grants[":root"])


def test_planner_allowlist_keeps_private_exceptions_and_removes_parent_masks(tmp_path):
    state = tmp_path / "state"
    rules = {":root": "deny", ":minimal": "read", str(state): "read",
             str(state / "work"): "write", str(state / "artifacts"): "deny",
             str(state / "baseline"): "read", str(state / "baseline/private"): "deny",
             str(state / "_blind/oracles/private"): "deny",
             "/usr/local/lib/gv-sanitizer": "deny"}
    _planner_state_allowlist(rules, state, "oracles")
    assert str(state) not in rules and str(state / "work") not in rules
    assert effective_state_grant(rules, state / "artifacts/new-file") == "deny"
    assert effective_state_grant(rules, state / "future-private") == "deny"
    assert effective_state_grant(rules, state / "baseline/public") == "read"
    assert effective_state_grant(rules, state / "baseline/private/file") == "deny"
    assert effective_state_grant(rules, state / "_blind/oracles/private/file") == "deny"
    assert effective_state_grant(rules, state / "_blind/oracles/file") == "write"
    assert effective_state_grant(rules, state / "_blind/critic/file") == "deny"
    assert rules["/usr/local/lib/gv-sanitizer"] == "deny"


@pytest.mark.parametrize("name", ["task.md", "baseline", "oracle_catalog", "_blind/oracles"])
def test_planner_allowlist_never_overrides_an_exact_private_boundary(tmp_path, name):
    state = tmp_path / "state"
    rules = {":root": "deny", str(state): "deny", str(state / name): "deny"}
    original = rules.copy()
    from verifier.llm.schema import RoleError
    with pytest.raises(RoleError, match="explicit private boundary"):
        _planner_state_allowlist(rules, state, "oracles")
    assert rules == original


@pytest.mark.parametrize("relative", [".", "baseline", ".."])
@pytest.mark.parametrize("role", sorted(PLANNER_ROLES))
def test_planner_project_and_state_must_remain_disjoint(tmp_path, role, relative):
    state = tmp_path / "state"
    state.mkdir()
    project = os.path.abspath(str(state / relative))
    backend = CyberCodexBackend(cwd=str(state), project_root=project, model="fixture")
    from verifier.llm.schema import RoleError
    with pytest.raises(RoleError, match="separate from the public project"):
        backend._execution_policy_args(role, {})


@pytest.mark.parametrize("role", sorted(PLANNER_ROLES))
def test_planner_project_alias_cannot_reopen_private_state(tmp_path, role):
    state = tmp_path / "state"
    (state / "baseline").mkdir(parents=True)
    alias = tmp_path / "project-alias"
    alias.symlink_to(state / "baseline", target_is_directory=True)
    backend = CyberCodexBackend(cwd=str(state), project_root=str(alias), model="fixture")
    from verifier.llm.schema import RoleError
    with pytest.raises(RoleError, match="separate from the public project"):
        backend._execution_policy_args(role, {})


@pytest.mark.parametrize("root", _PLANNER_RUNTIME_BOUNDARIES)
@pytest.mark.parametrize("role", sorted(PLANNER_ROLES))
def test_planner_state_cannot_overlap_platform_read_mounts(tmp_path, monkeypatch, root, role):
    backend = CyberCodexBackend(cwd=str(tmp_path), model="fixture")
    monkeypatch.setattr(backend, "_offline_state_dir", lambda _: root + "/private-state")
    monkeypatch.setattr(backend, "_offline_tool_roots",
                        lambda _: pytest.fail("must reject before scratch creation"))
    from verifier.llm.schema import RoleError
    with pytest.raises(RoleError, match="platform runtime mount"):
        backend._execution_policy_args(role, {})


def test_deny_normalization_preserves_reopened_subtree_exceptions(tmp_path):
    state = tmp_path / "state"
    rules = {
        ":root": "deny", ":minimal": "read", str(state): "deny",
        str(state / "hidden"): "deny", str(state / "hidden/nested"): "deny",
        str(state / "baseline"): "read", str(state / "baseline/private"): "deny",
        str(state / "baseline/private/nested"): "deny",
        str(state / "baseline/private/public"): "read",
        str(state / "baseline/private/public/secret"): "deny",
        str(state / "_blind/oracles"): "write",
        str(state / "_blind/oracles/secret"): "deny",
        "/usr/local/lib/gv-sanitizer": "deny",
        "/usr/local/lib/gv-sanitizer/runner": "deny",
        "~/.codex": "deny",
    }
    original = rules.copy()
    _prune_redundant_state_denies(rules, state)
    assert set(original) - set(rules) == {
        str(state / "hidden"), str(state / "hidden/nested"),
        str(state / "baseline/private/nested"),
    }
    for path in (state, state / "hidden/nested/file", state / "new-transport/file",
                 state / "baseline/file", state / "baseline/private/nested/file",
                 state / "baseline/private/public/file",
                 state / "baseline/private/public/secret/file",
                 state / "_blind/oracles/file", state / "_blind/oracles/secret/file",
                 state / "_blind/critic/file"):
        assert effective_state_grant(rules, path) == effective_state_grant(original, path)
    normalized = rules.copy()
    _prune_redundant_state_denies(rules, state)
    assert rules == normalized


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
    assert set(cyber.identity["cyber_boundary"]) == {"adapters.py", "backend.py", "feedback.py", "phases.py", "runtime.py"}
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
