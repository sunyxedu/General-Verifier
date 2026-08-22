"""CodexBackend: the pure parts (JSON recovery, role lookup, sandbox
split). The subprocess round-trip is exercised by live smokes, not here."""
import json
import os

import pytest

from verifier.llm.codex_backend import CodexBackend, extract_json
from verifier.llm.schema import RoleError


def test_extract_json_tolerates_fences_and_prose():
    text = 'Sure! Here it is:\n```json\n{"verdict": "pass", "n": [1, 2]}\n```'
    assert extract_json(text) == {"verdict": "pass", "n": [1, 2]}


def test_extract_json_skips_broken_braces():
    assert extract_json('{oops} then {"ok": true}') == {"ok": True}


def test_extract_json_fails_closed():
    with pytest.raises(RoleError):
        extract_json("no json here")


def test_unknown_role_is_a_role_error():
    with pytest.raises(RoleError, match="unknown agent role"):
        CodexBackend().complete("nope", "prompt", {})


@pytest.mark.parametrize("role", ["discharge", "judger"])
@pytest.mark.parametrize("world_mode", ["captured", "unavailable"])
def test_offline_evidence_seats_get_isolated_execution(
    tmp_path, role, world_mode,
):
    backend = CodexBackend(cwd=str(tmp_path))
    args = backend._execution_policy_args(
        role, {"world_mode": world_mode})

    assert "--dangerously-bypass-approvals-and-sandbox" not in args
    assert "--sandbox" not in args  # a permission profile replaces it
    assert "--ignore-user-config" in args
    assert 'default_permissions="gv-offline"' in args
    assert 'approval_policy="never"' in args
    assert 'permissions.gv-offline.extends=":workspace"' in args
    assert "permissions.gv-offline.network.enabled=false" in args
    assert 'shell_environment_policy.inherit="core"' in args

    filesystem = next(
        arg for arg in args
        if arg.startswith("permissions.gv-offline.filesystem=")
    )
    for denied in (
        "~/.cache", "~/Library/Caches", "~/.huggingface", "~/.codex",
    ):
        assert f'"{denied}"="deny"' in filesystem

    home = os.path.join(str(tmp_path), "work", "_offline_home")
    env = next(
        arg for arg in args
        if arg.startswith("shell_environment_policy.set=")
    )
    assert f'HOME={json.dumps(home)}' in env
    assert f'XDG_CACHE_HOME={json.dumps(os.path.join(home, ".cache"))}' in env
    assert f'TMPDIR={json.dumps(os.path.join(home, "tmp"))}' in env
    for key in (
        "HF_HOME", "HF_DATASETS_CACHE", "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE", "TRANSFORMERS_CACHE", "HF_XET_CACHE",
        "HF_ASSETS_CACHE",
    ):
        assert f"{key}=" in env
    assert os.path.isdir(home)
    assert os.path.isdir(os.path.join(home, "tmp"))
    assert os.path.isdir(os.path.join(home, ".cache", "huggingface", "hub"))


@pytest.mark.parametrize(
    ("role", "world_mode"),
    [
        ("discharge", "live"),
        ("judger", "live"),
        ("obligations", "captured"),
        ("critic", "unavailable"),
        ("oracles", "captured"),
        ("extract", "unavailable"),
    ],
)
def test_live_and_semantic_seats_retain_original_yolo_policy(
    tmp_path, role, world_mode,
):
    args = CodexBackend(cwd=str(tmp_path))._execution_policy_args(
        role, {"world_mode": world_mode})

    assert args == ["--dangerously-bypass-approvals-and-sandbox"]


def test_offline_evidence_can_derive_state_from_ledger(tmp_path):
    ledger = tmp_path / "events.jsonl"
    args = CodexBackend()._execution_policy_args(
        "judger",
        {"world_mode": "unavailable", "ledger_path": str(ledger)},
    )

    env = next(
        arg for arg in args
        if arg.startswith("shell_environment_policy.set=")
    )
    expected = os.path.join(str(tmp_path), "work", "_offline_home")
    assert f'HOME={json.dumps(expected)}' in env


def test_offline_evidence_without_a_state_fails_closed():
    with pytest.raises(RoleError, match="needs a state cwd"):
        CodexBackend()._execution_policy_args(
            "discharge", {"world_mode": "captured"})
