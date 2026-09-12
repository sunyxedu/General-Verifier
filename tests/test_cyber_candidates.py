from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import pytest

from verifier.core.events import EventLog
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber import candidates as module
from verifier.profiles.cyber.candidates import CandidateError, CandidateStore


@pytest.fixture
def store(tmp_path):
    project, state = tmp_path / "project", tmp_path / "state"
    project.mkdir()
    state.mkdir()
    artifacts = ArtifactStore(str(state / "artifacts"))
    return CandidateStore(str(state), str(project), artifacts,
                          EventLog(str(state / "events.jsonl")))


def register(store, data=b"\x00arbitrary report\xff"):
    source = Path(store.project_root) / "input"
    source.write_bytes(data)
    return source, store.register(str(source), label="candidate")


def test_exact_bytes_canonical_events_and_restart(store):
    source, candidate = register(store)
    assert candidate["candidate_id"] == hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"later solver revision")
    assert Path(candidate["blob_path"]).read_bytes() == b"\x00arbitrary report\xff"
    assert next(store.log.replay())["type"] == "cyber_candidate_registered"
    recovered = CandidateStore(store.state_dir, store.project_root,
                               store.artifacts, EventLog(store.log.path))
    assert recovered.get(candidate["candidate_id"]) == candidate
    assert recovered.candidates() == [candidate]


def test_identical_bytes_deduplicate_identity_not_registration(store):
    _, first = register(store, b"report")
    _, second = register(store, b"report")
    assert first["candidate_id"] == second["candidate_id"]
    assert len(store.candidates()) == 1
    assert len(list(store.log.replay())) == 2


@pytest.mark.parametrize("kind", ["symlink", "directory", "fifo", "outside", "parent_symlink"])
def test_registration_rejects_unsafe_paths(store, tmp_path, kind):
    root = Path(store.project_root)
    outside = tmp_path / "outside"
    outside.write_bytes(b"external")
    path = root / "unsafe"
    if kind == "symlink":
        path.symlink_to(outside)
    elif kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "outside":
        path = outside
    else:
        directory = tmp_path / "external_directory"
        directory.mkdir()
        (directory / "file").write_bytes(b"external")
        path.symlink_to(directory, target_is_directory=True)
        path /= "file"
    with pytest.raises(CandidateError):
        store.register(str(path))


def test_oversize_is_rejected_before_read(store):
    path = Path(store.project_root) / "large"
    with path.open("wb") as handle:
        handle.truncate(module.MAX_CANDIDATE_BYTES + 1)
    with pytest.raises(CandidateError, match="16 MiB"):
        store.register(str(path))


def test_source_mutation_during_registration_rejected(store, monkeypatch):
    source = Path(store.project_root) / "changing"
    source.write_bytes(b"first")
    original = os.read
    mutated = False

    def changing_read(fd, size):
        nonlocal mutated
        chunk = original(fd, size)
        if chunk and not mutated:
            mutated = True
            source.write_bytes(b"other")
        return chunk

    monkeypatch.setattr(module.os, "read", changing_read)
    with pytest.raises(CandidateError, match="changed"):
        store.register(str(source))
    assert not store.candidates()


def test_blob_mutation_and_path_swap_break_lease(store):
    _, candidate = register(store)
    blob = Path(candidate["blob_path"])
    with pytest.raises(CandidateError, match="identity changed"):
        with store.lease(candidate["candidate_id"]) as lease:
            backup = blob.with_suffix(".old")
            blob.rename(backup)
            blob.write_bytes(backup.read_bytes())
            store.validate(lease)


def test_lease_rejects_content_change_even_without_path_swap(store):
    _, candidate = register(store)
    with pytest.raises(CandidateError):
        with store.lease(candidate["candidate_id"]):
            blob = Path(candidate["blob_path"])
            blob.chmod(0o600)
            blob.write_bytes(b"tampered")


def test_metadata_artifact_tampering_rejected(store):
    _, candidate = register(store)
    artifact = Path(store.artifacts.root) / (candidate["artifact_id"] + ".json")
    payload = json.loads(artifact.read_text())
    payload["size"] += 1
    artifact.write_text(json.dumps(payload))
    with pytest.raises(CandidateError, match="artifact identity"):
        store.get(candidate["candidate_id"])


def test_solver_tool_result_cannot_register_candidate(store):
    _, candidate = register(store)
    other = {**candidate, "candidate_id": "0" * 64, "sha256": "0" * 64}
    store.log.append("tool_result", {"origin": "candidate_store",
                                    "tool": "cyber_register_candidate", "candidate": other})
    assert store.candidates() == [candidate]


def test_materialization_exact_receipt_and_inactive_lease(store):
    _, candidate = register(store)
    output = Path(store.project_root) / "final"
    with store.lease(candidate["candidate_id"]) as lease:
        receipt = store.materialize(lease, str(output))
    assert output.read_bytes() == b"\x00arbitrary report\xff"
    assert receipt["destination"] == str(output)
    assert receipt["sha256"] == candidate["sha256"]
    assert store.verify_delivery(receipt) == receipt
    assert "verified" not in receipt and "score" not in receipt
    with pytest.raises(CandidateError, match="inactive"):
        store.validate(lease)


@pytest.mark.parametrize("kind", ["source", "task", "symlink", "directory", "fifo", "outside", "parent_symlink"])
def test_delivery_never_overwrites_or_escapes(store, tmp_path, kind):
    source, candidate = register(store)
    output = Path(store.project_root) / "final"
    if kind == "source":
        output = source
    elif kind == "task":
        output = Path(store.project_root) / "task.md"
        output.write_bytes(b"authoritative task")
    elif kind == "symlink":
        output.symlink_to(source)
    elif kind == "directory":
        output.mkdir()
    elif kind == "fifo":
        os.mkfifo(output)
    elif kind == "outside":
        output = tmp_path / "escaped"
    else:
        external = tmp_path / "external"
        external.mkdir()
        output.symlink_to(external, target_is_directory=True)
        output /= "escaped"
    with store.lease(candidate["candidate_id"]) as lease:
        with pytest.raises(CandidateError):
            store.materialize(lease, str(output))
    assert source.read_bytes() == b"\x00arbitrary report\xff"
    assert not (tmp_path / "escaped").exists()


def test_destination_swap_after_link_cannot_receive_receipt(store, monkeypatch):
    _, candidate = register(store)
    output = Path(store.project_root) / "final"
    original = os.link

    def swapping_link(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if dst == "final":
            output.unlink()
            output.write_bytes(b"attacker replacement")
        return result

    monkeypatch.setattr(module.os, "link", swapping_link)
    with store.lease(candidate["candidate_id"]) as lease:
        with pytest.raises(CandidateError, match="canonical output changed"):
            store.materialize(lease, str(output))
    assert output.read_bytes() == b"attacker replacement"


def test_parent_directory_swap_is_detected(store, tmp_path, monkeypatch):
    _, candidate = register(store)
    parent = Path(store.project_root) / "outputs"
    parent.mkdir()
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original = os.link

    def moving_link(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if dst == "final":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
        return result

    monkeypatch.setattr(module.os, "link", moving_link)
    with store.lease(candidate["candidate_id"]) as lease:
        with pytest.raises(CandidateError):
            store.materialize(lease, str(parent / "final"))
    assert not (moved / "final").exists()
    assert not (replacement / "final").exists()


def test_preflight_checks_before_any_write(store):
    destination = Path(store.project_root) / "new"
    assert store.preflight_destination(str(destination)) == str(destination)
    assert not destination.exists()
    destination.symlink_to(Path(store.project_root) / "missing")
    with pytest.raises(CandidateError, match="already exists"):
        store.preflight_destination(str(destination))


def delivered(store, parent=""):
    _, candidate = register(store)
    root = Path(store.project_root)
    if parent:
        (root / parent).mkdir()
    output = root / parent / "final"
    with store.lease(candidate["candidate_id"]) as lease:
        receipt = store.materialize(lease, str(output))
    return output, receipt


@pytest.mark.parametrize("kind", ["replacement", "symlink", "content"])
def test_delivery_verification_and_rollback_preserve_modified_outputs(store, kind):
    output, receipt = delivered(store)
    if kind == "replacement":
        output.unlink()
        output.write_bytes(b"user replacement")
    elif kind == "symlink":
        target = Path(store.project_root) / "target"
        target.write_bytes(b"user target")
        output.unlink()
        output.symlink_to(target)
    else:
        output.chmod(0o600)
        output.write_bytes(b"user changed content")
    with pytest.raises(CandidateError):
        store.verify_delivery(receipt)
    assert store.rollback(receipt) is False
    if kind == "symlink":
        assert output.is_symlink()
        assert output.read_bytes() == b"user target"
    else:
        assert output.read_bytes().startswith(b"user")


def test_successful_rollback_removes_only_exact_created_output(store):
    output, receipt = delivered(store)
    assert store.rollback(receipt) is True
    assert not output.exists()
    assert store.rollback(receipt) is False
    assert Path(store.get(receipt["candidate_id"])["blob_path"]).exists()


def test_verify_and_rollback_recheck_ancestors(store, tmp_path):
    output, receipt = delivered(store, parent="outputs")
    parent = output.parent
    moved = tmp_path / "moved"
    parent.rename(moved)
    parent.symlink_to(moved, target_is_directory=True)
    with pytest.raises(CandidateError):
        store.verify_delivery(receipt)
    assert store.rollback(receipt) is False
    assert (moved / "final").exists()


def test_rollback_checks_atomically_captured_leaf_not_earlier_stat(store, monkeypatch):
    output, receipt = delivered(store)
    original = os.rename

    def swapping_rename(src, dst, **kwargs):
        if src == "final":
            output.unlink()
            output.write_bytes(b"replacement during rollback")
        return original(src, dst, **kwargs)

    monkeypatch.setattr(module.os, "rename", swapping_rename)
    assert store.rollback(receipt) is False
    assert output.read_bytes() == b"replacement during rollback"


def test_rollback_parent_swap_after_capture_preserves_file(store, tmp_path, monkeypatch):
    output, receipt = delivered(store, parent="outputs")
    parent = output.parent
    moved = tmp_path / "moved"
    replacement = tmp_path / "replacement"
    replacement.mkdir()
    original = os.rename

    def moving_rename(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if src == "final":
            parent.rename(moved)
            parent.symlink_to(replacement, target_is_directory=True)
        return result

    monkeypatch.setattr(module.os, "rename", moving_rename)
    assert store.rollback(receipt) is False
    assert (moved / "final").exists()


def test_aborted_delivery_preserves_same_inode_modified_content(store, monkeypatch):
    _, candidate = register(store)
    output = Path(store.project_root) / "final"
    original = os.link

    def changing_link(src, dst, **kwargs):
        result = original(src, dst, **kwargs)
        if dst == "final":
            output.chmod(0o600)
            output.write_bytes(b"user changed the newly created inode")
        return result

    monkeypatch.setattr(module.os, "link", changing_link)
    with store.lease(candidate["candidate_id"]) as lease:
        with pytest.raises(CandidateError, match="output digest mismatch"):
            store.materialize(lease, str(output))
    assert output.read_bytes() == b"user changed the newly created inode"


def test_abort_cleanup_captures_entry_before_inspecting_or_deleting(store, monkeypatch):
    _, candidate = register(store)
    output = Path(store.project_root) / "final"
    read = module._read_fd
    rename = os.rename
    failed = False

    def fail_after_link(fd):
        nonlocal failed
        if output.exists() and os.fstat(fd).st_ino == output.stat().st_ino and not failed:
            failed = True
            raise CandidateError("fixture post-copy failure")
        return read(fd)

    def replace_before_capture(src, dst, **kwargs):
        if src == "final":
            output.unlink()
            output.write_bytes(b"replacement while abort cleanup starts")
        return rename(src, dst, **kwargs)

    monkeypatch.setattr(module, "_read_fd", fail_after_link)
    monkeypatch.setattr(module.os, "rename", replace_before_capture)
    with store.lease(candidate["candidate_id"]) as lease:
        with pytest.raises(CandidateError, match="fixture post-copy"):
            store.materialize(lease, str(output))
    assert failed
    assert output.read_bytes() == b"replacement while abort cleanup starts"
