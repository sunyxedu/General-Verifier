"""Candidate-independent, fail-closed authority compilation views."""
from __future__ import annotations

import json
import os

import pytest

from verifier.engine import authority_view
from verifier.engine.authority_view import (
    AUTHORITY_VIEW_SCHEMA,
    AuthorityViewUnavailable,
    build_authority_compilation_view,
)


SHA_A = "a" * 64
SHA_B = "b" * 64


def _state(tmp_path, name="state", *, baseline=True):
    state = tmp_path / name
    state.mkdir()
    (state / "task.md").write_text("Build every required output. ✓")
    if baseline:
        base = state / "baseline"
        base.mkdir()
        (base / "git.json").write_text(json.dumps({
            "head": "c" * 40, "refs": "c" * 40 + " refs/heads/main",
            "status": "",
        }))
        (base / "manifest.json").write_text(json.dumps({
            "root": f"/private/candidate/{name}",
            "files": {
                "small.txt": SHA_A,
                "large.bin": "size:70000000",
                "service/current": "link:/private/host/service-v1",
            },
        }))
    return state


def test_view_is_canonical_candidate_independent_and_has_no_host_paths(
        tmp_path):
    first = build_authority_compilation_view(_state(tmp_path, "one"))
    second = build_authority_compilation_view(_state(tmp_path, "two"))

    assert first.schema == AUTHORITY_VIEW_SCHEMA
    assert first.identity == second.identity
    assert first.context_json == second.context_json
    assert "/private/candidate" not in first.context_json
    assert "/private/host/service-v1" not in first.context_json
    context = first.context()
    assert context["task"] == {
        "logical_source": "task.md",
        "text": "Build every required output. ✓",
    }
    assert context["baseline"]["status"] == "present"
    assert context["baseline"]["git"]["head"] == "c" * 40
    assert [item["path"] for item in context["baseline"]["files"]] == [
        "large.bin", "service/current", "small.txt",
    ]
    link = context["baseline"]["files"][1]["identity"]
    assert link["kind"] == "symlink"
    assert set(link) == {"kind", "target_sha256"}


def test_candidate_outputs_ledger_and_world_never_enter_the_view(tmp_path):
    state = _state(tmp_path)
    original = build_authority_compilation_view(state)
    (state / "final.txt").write_text("solver answer")
    (state / "events.jsonl").write_text('{"candidate":"ledger"}\n')
    (state / "world").mkdir()
    (state / "world" / "delivered.txt").write_text("candidate bytes")

    current = build_authority_compilation_view(state)

    assert current.identity == original.identity
    assert current.context_json == original.context_json
    for forbidden in ("solver answer", "ledger", "candidate bytes"):
        assert forbidden not in current.context_json


def test_absent_baseline_is_explicit_and_stable(tmp_path):
    state = _state(tmp_path, baseline=False)

    view = build_authority_compilation_view(state)

    baseline = view.context()["baseline"]
    assert baseline["status"] == "absent"
    assert "do not generate baseline-preservation" \
        in baseline["minimal_change_authority"]
    assert view.revalidate(state).identity == view.identity


def test_context_method_returns_a_detached_copy(tmp_path):
    view = build_authority_compilation_view(_state(tmp_path))
    decoded = view.context()
    decoded["task"]["text"] = "mutated"

    assert view.context()["task"]["text"] \
        == "Build every required output. ✓"


def test_revalidate_fails_after_authority_changes(tmp_path):
    state = _state(tmp_path)
    view = build_authority_compilation_view(state)
    (state / "task.md").write_text("A different authoritative task")

    with pytest.raises(AuthorityViewUnavailable, match="authority changed"):
        view.revalidate(state)


@pytest.mark.parametrize("missing", ["task.md", "git.json", "manifest.json"])
def test_required_authority_files_fail_closed(tmp_path, missing):
    state = _state(tmp_path)
    target = state / missing
    if missing != "task.md":
        target = state / "baseline" / missing
    target.unlink()

    with pytest.raises(AuthorityViewUnavailable):
        build_authority_compilation_view(state)


@pytest.mark.parametrize("target", ["task.md", "baseline", "baseline/git.json"])
def test_authority_symlinks_fail_closed(tmp_path, target):
    state = _state(tmp_path)
    path = state / target
    outside = tmp_path / ("outside-dir" if path.is_dir() else "outside-file")
    if path.is_dir():
        outside.mkdir()
        for child in path.iterdir():
            child.rename(outside / child.name)
        path.rmdir()
        path.symlink_to(outside, target_is_directory=True)
    else:
        outside.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(outside)

    with pytest.raises(AuthorityViewUnavailable):
        build_authority_compilation_view(state)


def test_state_path_with_symlinked_component_fails_closed(tmp_path):
    state = _state(tmp_path)
    alias = tmp_path / "state-alias"
    alias.symlink_to(state, target_is_directory=True)

    with pytest.raises(AuthorityViewUnavailable, match="traverse symlinks"):
        build_authority_compilation_view(alias)


def test_hard_linked_authority_file_fails_closed(tmp_path):
    state = _state(tmp_path)
    alias = tmp_path / "task-alias"
    os.link(state / "task.md", alias)

    with pytest.raises(AuthorityViewUnavailable, match="unlinked regular"):
        build_authority_compilation_view(state)


@pytest.mark.parametrize(
    "filename,payload",
    [
        ("git.json", '{"head":"a","head":"b"}'),
        ("git.json", '{"unknown":"value"}'),
        ("manifest.json", '{"root":"/x","files":{},"extra":1}'),
        ("manifest.json", '{"root":"/x","files":{"../x":"' + SHA_A + '"}}'),
        ("manifest.json", '{"root":"/x","files":{"x":"opaque"}}'),
    ],
)
def test_unknown_ambiguous_or_noncanonical_baseline_data_fails_closed(
        tmp_path, filename, payload):
    state = _state(tmp_path)
    (state / "baseline" / filename).write_text(payload)

    with pytest.raises(AuthorityViewUnavailable):
        build_authority_compilation_view(state)


def test_partial_or_extended_baseline_directory_fails_closed(tmp_path):
    state = _state(tmp_path)
    (state / "baseline" / "extra.json").write_text("{}")

    with pytest.raises(AuthorityViewUnavailable, match="exactly"):
        build_authority_compilation_view(state)


def test_size_and_entry_caps_fail_closed(tmp_path, monkeypatch):
    state = _state(tmp_path)
    monkeypatch.setattr(authority_view, "MAX_TASK_BYTES", 8)
    with pytest.raises(AuthorityViewUnavailable, match="exceeds"):
        build_authority_compilation_view(state)

    monkeypatch.setattr(authority_view, "MAX_TASK_BYTES", 1024)
    monkeypatch.setattr(authority_view, "MAX_BASELINE_FILES", 1)
    with pytest.raises(AuthorityViewUnavailable, match="manifest exceeds"):
        build_authority_compilation_view(state)


def test_mutation_between_double_reads_fails_closed(tmp_path, monkeypatch):
    state = _state(tmp_path)
    original = authority_view._capture
    calls = 0

    def mutate_after_first(path):
        nonlocal calls
        calls += 1
        captured = original(path)
        if calls == 1:
            (state / "task.md").write_text("changed between snapshots")
        return captured

    monkeypatch.setattr(authority_view, "_capture", mutate_after_first)

    with pytest.raises(AuthorityViewUnavailable, match="while.*built"):
        build_authority_compilation_view(state)
