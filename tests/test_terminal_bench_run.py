"""Terminal-Bench solver service-tier command construction."""
import sys
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run


@pytest.mark.parametrize(
    ("tier", "expected"),
    [
        ("standard", [
            "--ak", "service_tier=default", "--ak", "fast_mode=disable",
        ]),
        ("fast", [
            "--ak", "service_tier=fast", "--ak", "fast_mode=enable",
        ]),
    ],
)
def test_codex_service_agent_kwargs_are_explicit(tier, expected):
    assert run.codex_service_agent_kwargs(tier) == expected


def test_codex_service_agent_kwargs_reject_unknown_tier():
    with pytest.raises(ValueError, match="unknown Codex service tier"):
        run.codex_service_agent_kwargs("turbo")


@pytest.mark.parametrize(
    ("extra", "tier_value", "feature_value"),
    [
        ([], "service_tier=default", "fast_mode=disable"),
        (["--service-tier", "fast"], "service_tier=fast", "fast_mode=enable"),
    ],
)
def test_dry_run_uses_custom_agent_and_requested_tier(
    tmp_path, monkeypatch, capsys, extra, tier_value, feature_value,
):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("task-a\n", encoding="utf-8")
    monkeypatch.setattr(run.provider, "claude_env", lambda: {})
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--arm", "codex", "--tasks-file", str(tasks),
        "--dry-run", *extra,
    ])

    assert run.main() is None

    command = capsys.readouterr().out
    assert "-a codex_agent:TieredCodex" in command
    assert f"--ak {tier_value}" in command
    assert f"--ak {feature_value}" in command


def test_rec_codex_always_collects_post_run_runtime_snapshot(
    tmp_path, monkeypatch, capsys,
):
    tasks = tmp_path / "tasks.txt"
    tasks.write_text("task-a\n", encoding="utf-8")
    monkeypatch.setattr(run.provider, "claude_env", lambda: {})
    monkeypatch.setattr(
        run.harbor_task_artifacts,
        "resolve_harbor_dataset_artifacts",
        lambda *_args, **_kwargs: SimpleNamespace(
            dataset="terminal-bench/terminal-bench-2-1@sha256:abc",
            artifacts=("/app",),
            referenced_by_task={"task-a": ()},
            audit_by_task={"task-a": ()},
        ),
    )
    monkeypatch.setattr(sys, "argv", [
        "run.py", "--arm", "rec-codex", "--tasks-file", str(tasks),
        "--dry-run",
    ])

    assert run.main() is None

    command = capsys.readouterr().out
    assert command.count(f"--artifact {run.RUNTIME_SNAPSHOT_ROOT}") == 1
