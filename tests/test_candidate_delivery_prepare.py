"""Offline durable intent ordering and exact publication identity tests."""
from __future__ import annotations

import os
import stat
import threading
from pathlib import Path

import pytest

from verifier.core import events as event_module
from verifier.core.events import EVENT_TYPES, SOURCE_EVENT_TYPES, EventLog, SourceProjectionUnavailable
from verifier.execution.instruments import ArtifactStore
from verifier.profiles.cyber import candidates as candidate_module
from verifier.profiles.cyber.candidates import CandidateError, CandidateStore


def fixture(tmp_path, *, hook=None):
    project, state = tmp_path / "project", tmp_path / "state"
    project.mkdir()
    state.mkdir()
    log = EventLog(str(state / "events.jsonl"))
    store = CandidateStore(str(state), str(project), ArtifactStore(str(state / "artifacts")),
                           log, prepare_delivery=hook)
    source = project / "proposed"
    source.write_bytes(b"synthetic generic artifact\x00\xff")
    candidate = store.register(str(source))
    return store, candidate, project / "canonical"


def publish(store, candidate, output):
    with store.lease(candidate["candidate_id"]) as lease:
        return store.materialize(lease, str(output))


def assert_no_publication(store, output):
    assert not output.exists()
    assert list(Path(store.project_root).glob(".delivery-*")) == []


def test_hook_is_after_temp_fsync_before_canonical_link_and_durably_synced(tmp_path, monkeypatch):
    fsynced, prepared = [], []
    real_fsync, real_link = os.fsync, os.link
    def sync(fd):
        info = os.fstat(fd)
        result = real_fsync(fd)
        fsynced.append((info.st_dev, info.st_ino, stat.S_ISDIR(info.st_mode)))
        return result
    monkeypatch.setattr(candidate_module.os, "fsync", sync)
    store, candidate, output = fixture(tmp_path)
    fsynced.clear()
    def prepare(receipt):
        assert not output.exists()
        assert (receipt["device"], receipt["inode"], False) in fsynced
        temporary, = Path(store.project_root).glob(".delivery-*")
        info = temporary.stat()
        assert (info.st_dev, info.st_ino) == (receipt["device"], receipt["inode"])
        assert info.st_nlink == 1
        store.log.append("cyber_candidate_delivery_prepared", {"receipt": receipt})
        store.log.sync()
        prepared.append(dict(receipt))
    def link(src, dst, **kwargs):
        if dst == output.name:
            assert prepared
            assert list(store.log.replay())[-1]["type"] == "cyber_candidate_delivery_prepared"
            log_info = os.stat(store.log.path)
            parent_info = os.stat(Path(store.log.path).parent)
            assert (log_info.st_dev, log_info.st_ino, False) in fsynced
            assert (parent_info.st_dev, parent_info.st_ino, True) in fsynced
        return real_link(src, dst, **kwargs)
    store._prepare_delivery = prepare
    monkeypatch.setattr(candidate_module.os, "link", link)
    receipt = publish(store, candidate, output)
    assert receipt == prepared[0]
    assert all(type(receipt[key]) is type(value) for key, value in prepared[0].items())
    assert store.verify_delivery(receipt) == receipt
    info = output.stat()
    assert (info.st_dev, info.st_ino) == (receipt["device"], receipt["inode"])
    assert list(Path(store.project_root).glob(".delivery-*")) == []


@pytest.mark.parametrize("error", [RuntimeError("synthetic prepare failure"),
                                  OSError("synthetic sync failure")])
def test_hook_failure_propagates_original_error_without_publication(tmp_path, error):
    def prepare(receipt):
        raise error
    store, candidate, output = fixture(tmp_path, hook=prepare)
    with pytest.raises(type(error)) as caught:
        publish(store, candidate, output)
    assert caught.value is error
    assert_no_publication(store, output)


@pytest.mark.parametrize("mutation", ["destination", "inode", "candidate", "size_bool", "remove", "add"])
def test_hook_cannot_mutate_or_extend_expected_receipt(tmp_path, mutation):
    def prepare(receipt):
        if mutation == "destination":
            receipt["destination"] += ".other"
        elif mutation == "inode":
            receipt["inode"] += 1
        elif mutation == "candidate":
            receipt["candidate_id"] = "0" * 64
        elif mutation == "size_bool":
            receipt["size"] = True
        elif mutation == "remove":
            del receipt["parent_inode"]
        else:
            receipt["verified"] = True
    store, candidate, output = fixture(tmp_path, hook=prepare)
    with pytest.raises(CandidateError, match="changed the receipt"):
        publish(store, candidate, output)
    assert_no_publication(store, output)


@pytest.mark.parametrize("result", [False, True, {}, "success"])
def test_hook_return_value_cannot_replace_durable_prepare_protocol(tmp_path, result):
    store, candidate, output = fixture(tmp_path, hook=lambda receipt: result)
    with pytest.raises(CandidateError, match="must return None"):
        publish(store, candidate, output)
    assert_no_publication(store, output)


def test_returned_receipt_is_not_the_hooks_mutable_object(tmp_path):
    held = []
    def prepare(receipt):
        held.append(receipt)
    store, candidate, output = fixture(tmp_path, hook=prepare)
    receipt = publish(store, candidate, output)
    assert receipt is not held[0] and receipt == held[0]
    held[0]["inode"] += 1
    assert store.verify_delivery(receipt) == receipt


def test_hook_cannot_change_staged_bytes_before_publication(tmp_path):
    store, candidate, output = fixture(tmp_path)
    def prepare(receipt):
        temporary, = Path(store.project_root).glob(".delivery-*")
        temporary.chmod(0o600)
        temporary.write_bytes(b"synthetic changed staged content")
    store._prepare_delivery = prepare
    with pytest.raises(CandidateError, match="changed before publication"):
        publish(store, candidate, output)
    assert_no_publication(store, output)


def test_default_materialize_interface_and_events_are_unchanged(tmp_path):
    store, candidate, output = fixture(tmp_path)
    before = list(store.log.replay())
    receipt = publish(store, candidate, output)
    assert receipt["schema"] == "cyber-materialization-v1"
    assert list(store.log.replay()) == before
    assert store.verify_delivery(receipt) == receipt


def test_constructor_rejects_noncallable_prepare_hook(tmp_path):
    with pytest.raises(CandidateError, match="hook must be callable"):
        CandidateStore(str(tmp_path), str(tmp_path), ArtifactStore(str(tmp_path / "art")),
                       EventLog(str(tmp_path / "events.jsonl")), prepare_delivery=True)


def test_prepared_events_are_host_only_not_source_evidence():
    for verb in ("cyber_candidate_delivery_prepared", "cyber_candidate_delivery_aborted"):
        assert verb in EVENT_TYPES and verb not in SOURCE_EVENT_TYPES
    assert "independent_certificate" in EVENT_TYPES
    assert "cyber_candidate_delivered" in EVENT_TYPES
    assert "cyber_candidate_delivery_rolled_back" in EVENT_TYPES


def test_sync_uses_no_follow_regular_descriptor_and_real_file_parent_fsync(tmp_path, monkeypatch):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "synthetic event"})
    before = Path(log.path).read_bytes()
    opened, synced = [], []
    real_open, real_fsync = os.open, os.fsync
    def open_fd(path, flags, *args, **kwargs):
        opened.append((os.fspath(path), flags))
        return real_open(path, flags, *args, **kwargs)
    def fsync(fd):
        info = os.fstat(fd)
        synced.append((info.st_dev, info.st_ino, stat.S_ISDIR(info.st_mode)))
        return real_fsync(fd)
    monkeypatch.setattr(event_module.os, "open", open_fd)
    monkeypatch.setattr(event_module.os, "fsync", fsync)
    assert log.sync() is None
    assert Path(log.path).read_bytes() == before
    assert log.last_seq() == 1
    assert len(synced) == 2 and [row[2] for row in synced] == [False, True]
    assert all(flags & os.O_NOFOLLOW for _, flags in opened)
    assert list(log.replay())[0]["type"] == "message"


@pytest.mark.parametrize("kind", ["symlink", "fifo", "directory", "hardlink", "owner"])
def test_sync_rejects_unsafe_current_ledger_without_fsync(tmp_path, monkeypatch, kind):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append("message", {"text": "synthetic event"})
    if kind == "symlink":
        path.rename(tmp_path / "retained")
        path.symlink_to(tmp_path / "retained")
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "hardlink":
        os.link(path, tmp_path / "alias")
    else:
        monkeypatch.setattr(event_module.os, "geteuid", lambda: os.stat(path).st_uid + 1)
    calls = []
    monkeypatch.setattr(event_module.os, "fsync", lambda fd: calls.append(fd))
    with pytest.raises(SourceProjectionUnavailable):
        log.sync()
    assert calls == []


def test_sync_failure_rejects_prepare_and_does_not_publish(tmp_path, monkeypatch):
    store, candidate, output = fixture(tmp_path)
    real_fsync = os.fsync
    def fail_ledger_sync(fd):
        if os.fstat(fd).st_ino == os.stat(store.log.path).st_ino:
            raise OSError("synthetic ledger fsync failure")
        return real_fsync(fd)
    def prepare(receipt):
        store.log.append("cyber_candidate_delivery_prepared", {"receipt": receipt})
        store.log.sync()
    store._prepare_delivery = prepare
    monkeypatch.setattr(event_module.os, "fsync", fail_ledger_sync)
    with pytest.raises(SourceProjectionUnavailable, match="cannot be safely synced"):
        publish(store, candidate, output)
    assert_no_publication(store, output)


def test_sync_does_not_fsync_a_replacement_of_the_just_appended_ledger(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append("message", {"text": "synthetic prepared intent"})
    path.rename(tmp_path / "retained")
    path.write_bytes(b"synthetic unrelated owned regular file")
    calls = []
    monkeypatch.setattr(event_module.os, "fsync", lambda fd: calls.append(fd))
    with pytest.raises(SourceProjectionUnavailable, match="changed before sync"):
        log.sync()
    assert calls == []


def test_sync_does_not_accept_a_different_log_path_after_append(tmp_path, monkeypatch):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "synthetic prepared intent"})
    alternative = tmp_path / "other.jsonl"
    alternative.write_bytes(b"synthetic unrelated owned regular file")
    log.path = str(alternative)
    calls = []
    monkeypatch.setattr(event_module.os, "fsync", lambda fd: calls.append(fd))
    with pytest.raises(SourceProjectionUnavailable, match="path changed before sync"):
        log.sync()
    assert calls == []


def test_sync_detects_leaf_replacement_during_fsync(tmp_path, monkeypatch):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append("message", {"text": "synthetic event"})
    real_fsync = os.fsync
    replaced = False
    def replace(fd):
        nonlocal replaced
        result = real_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode) and not replaced:
            replaced = True
            path.rename(tmp_path / "retained")
            path.write_bytes(b"synthetic replacement")
        return result
    monkeypatch.setattr(event_module.os, "fsync", replace)
    with pytest.raises(SourceProjectionUnavailable, match="changed during sync"):
        log.sync()
    assert path.read_bytes() == b"synthetic replacement"


def test_sync_detects_parent_replacement_during_fsync(tmp_path, monkeypatch):
    parent = tmp_path / "state"
    parent.mkdir()
    log = EventLog(str(parent / "events.jsonl"))
    log.append("message", {"text": "synthetic event"})
    real_fsync = os.fsync
    replaced = False
    def replace(fd):
        nonlocal replaced
        result = real_fsync(fd)
        if stat.S_ISREG(os.fstat(fd).st_mode) and not replaced:
            replaced = True
            parent.rename(tmp_path / "retained-state")
            parent.mkdir()
            (parent / "events.jsonl").write_bytes(b"synthetic replacement")
        return result
    monkeypatch.setattr(event_module.os, "fsync", replace)
    with pytest.raises(SourceProjectionUnavailable, match="changed during sync"):
        log.sync()
    assert (parent / "events.jsonl").read_bytes() == b"synthetic replacement"


def test_sync_holds_writer_lock_through_file_and_parent_fsync(tmp_path, monkeypatch):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "synthetic first event"})
    entered, release, attempting, appended = (threading.Event() for _ in range(4))
    real_fsync = os.fsync
    failures = []
    def fsync(fd):
        if stat.S_ISREG(os.fstat(fd).st_mode):
            entered.set()
            if not release.wait(2):
                raise RuntimeError("synthetic test barrier deadline")
        return real_fsync(fd)
    def sync():
        try:
            log.sync()
        except Exception as error:
            failures.append(error)
    def append():
        attempting.set()
        log.append("message", {"text": "synthetic next event"})
        appended.set()
    monkeypatch.setattr(event_module.os, "fsync", fsync)
    sync_thread = threading.Thread(target=sync)
    append_thread = threading.Thread(target=append)
    sync_thread.start()
    try:
        assert entered.wait(2)
        append_thread.start()
        assert attempting.wait(2)
        assert not appended.wait(0.05)
    finally:
        release.set()
        sync_thread.join(2)
        if append_thread.ident is not None:
            append_thread.join(2)
    assert failures == []
    assert not sync_thread.is_alive() and not append_thread.is_alive()
    assert appended.is_set() and log.last_seq() == 2
