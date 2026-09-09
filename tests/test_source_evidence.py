from __future__ import annotations

import json
import os
import shutil
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from verifier.core.events import EventLog, SourceProjectionUnavailable
from verifier.engine import source_evidence
from verifier.engine.source_evidence import (
    SourceEvidenceInvalid,
    build_source_evidence_view,
)


def _log(tmp_path: Path) -> EventLog:
    return EventLog(str(tmp_path / "events.jsonl"))


def _json(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _first_event(view):
    manifest = _json(view.manifest_path)
    page = _json(Path(view._root) / manifest["pages"][0]["path"])
    return manifest, page["events"][0]


def test_view_is_deterministic_and_preserves_complete_source_record(tmp_path):
    log = _log(tmp_path)
    written = log.append("message", {"role": "assistant", "text": "hello"})

    first = build_source_evidence_view(log)
    second = build_source_evidence_view(log)

    assert first.view_id == second.view_id
    assert first.manifest_path == second.manifest_path
    assert first._files == second._files
    assert first.revalidate()
    manifest, event = _first_event(first)
    assert manifest["event_count"] == 1
    assert event["original_seq"] == written["seq"]
    assert event["original_ts"] == written["ts"]
    assert event["type"] == written["type"]
    assert event["payload"] == written["payload"]
    assert event["event_digest"] == EventLog.event_digest(written)


def test_same_writer_bytes_make_same_tree_in_different_state_roots(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr("verifier.core.events.time.time", lambda: 17.25)
    roots = [tmp_path / "one", tmp_path / "two"]
    for root in roots:
        root.mkdir()
    first_log = EventLog(str(roots[0] / "events.jsonl"))
    second_log = EventLog(str(roots[1] / "events.jsonl"))
    first_log.append("message", {"z": "last", "a": {"y": 2, "x": 1}})
    second_log.append("message", {"a": {"x": 1, "y": 2}, "z": "last"})

    first = build_source_evidence_view(first_log)
    second = build_source_evidence_view(second_log)

    assert Path(first_log.path).read_bytes() == Path(second_log.path).read_bytes()
    assert first.view_id == second.view_id
    assert first._files == second._files


def test_non_writer_float_spelling_cannot_alias_a_writer_view(tmp_path):
    log = _log(tmp_path)
    Path(log.path).write_bytes(
        b'{"payload": {"n": 0.100000000000000005551}, "seq": 1, '
        b'"ts": 1.0, "type": "message"}\n'
    )

    with pytest.raises(SourceProjectionUnavailable, match="writer encoding"):
        log.source_projection()


@pytest.mark.parametrize("bad", [float("nan"), "\ud800"])
def test_failed_event_serialization_does_not_leave_a_sequence_gap(tmp_path, bad):
    log = _log(tmp_path)
    with pytest.raises((ValueError, UnicodeError)):
        log.append("message", {"bad": bad})

    event = log.append("message", {"ok": True})
    assert event["seq"] == 1
    assert log.source_projection()[0]["seq"] == 1


@pytest.mark.parametrize(
    "payload",
    [[], {"tuple": (1, 2)}, {1: "integer key"}],
)
def test_writer_rejects_non_json_object_identity_before_touching_log(
    tmp_path, payload,
):
    log = _log(tmp_path)
    with pytest.raises(ValueError):
        log.append("message", payload)

    assert not Path(log.path).exists()
    assert log.append("message", {"ok": [1, 2]})["seq"] == 1


def test_benign_parent_symlink_resolves_to_one_safe_ledger(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(real, target_is_directory=True)
    log = EventLog(str(alias / "events.jsonl"))
    log.append("message", {"text": "source"})

    view = build_source_evidence_view(log)
    assert view.revalidate()
    assert Path(view.manifest_path).is_relative_to(real)


def test_verifier_append_reuses_view_but_source_append_invalidates_it(tmp_path):
    log = _log(tmp_path)
    log.append("tool_call", {"tool": "shell", "command": "pwd"})
    view = build_source_evidence_view(log)

    log.append("status_change", {"claim": "c", "status": "verified"})
    unchanged = build_source_evidence_view(log)
    assert unchanged.view_id == view.view_id
    assert view.revalidate()

    log.append("tool_result", {"output": "/work"})
    assert not view.revalidate()
    assert build_source_evidence_view(log).view_id != view.view_id


def test_large_utf8_leaf_is_exactly_reconstructable(tmp_path):
    log = _log(tmp_path)
    exact = "世界/🙂\n" * 500
    log.append("tool_result", {"nested": [{"exact": exact}], "small": "ok"})

    view = build_source_evidence_view(log)
    manifest, event = _first_event(view)
    descriptor = event["externalized_utf8_fields"][0]
    marker = event["payload"]["nested"][0]["exact"]

    assert descriptor["json_pointer"] == "/payload/nested/0/exact"
    assert marker["$gv_source_utf8_blob"] == descriptor
    blob_path = Path(view._root) / descriptor["blob"]
    assert blob_path.read_bytes() == exact.encode("utf-8")
    assert event["payload"]["small"] == "ok"
    assert manifest["blobs"] == [{
        "bytes": len(exact.encode("utf-8")),
        "path": descriptor["blob"],
        "sha256": descriptor["sha256"],
    }]


def test_paging_is_complete_and_indexed(tmp_path, monkeypatch):
    monkeypatch.setattr(source_evidence, "TARGET_PAGE_BYTES", 350)
    monkeypatch.setattr(source_evidence, "MAX_PAGE_BYTES", 4096)
    log = _log(tmp_path)
    written = [
        log.append("message", {"role": "assistant", "text": str(index) * 80})
        for index in range(12)
    ]

    view = build_source_evidence_view(log)
    manifest = _json(view.manifest_path)
    assert len(manifest["pages"]) > 1
    assert sum(page["event_count"] for page in manifest["pages"]) == 12
    lines = Path(view._root, "index.jsonl").read_text(
        encoding="utf-8").splitlines()
    header, *entries = [json.loads(line) for line in lines]
    assert header["event_count"] == 12
    assert [entry["source_ordinal"] for entry in entries] == list(range(1, 13))
    assert [entry["original_seq"] for entry in entries] == [
        event["seq"] for event in written]


def test_oversized_projected_event_uses_one_exact_blob(tmp_path, monkeypatch):
    monkeypatch.setattr(source_evidence, "TARGET_PAGE_BYTES", 300)
    monkeypatch.setattr(source_evidence, "MAX_PAGE_BYTES", 900)
    log = _log(tmp_path)
    payload = {f"field-{index}": "abcdefgh" for index in range(180)}
    log.append("commit", payload)

    view = build_source_evidence_view(log)
    _manifest, event = _first_event(view)
    descriptor = event["exact_event_blob"]
    exact = Path(view._root, descriptor["blob"]).read_bytes()
    projected = json.loads(exact.decode("utf-8"))

    assert projected["payload"] == payload
    assert projected["original_ts"] == event["original_ts"]
    assert descriptor["bytes"] == len(exact)


@pytest.mark.parametrize("mutation", ["content", "extra_file", "extra_dir"])
def test_any_publication_mutation_is_rejected(tmp_path, mutation):
    log = _log(tmp_path)
    log.append("message", {"text": "source"})
    view = build_source_evidence_view(log)
    Path(view._root).chmod(0o700)

    if mutation == "content":
        page = next(Path(view._root, "pages").iterdir())
        page.chmod(0o600)
        page.write_bytes(page.read_bytes() + b" ")
    elif mutation == "extra_file":
        extra = Path(view._root, "unexpected")
        extra.write_text("x", encoding="utf-8")
    else:
        Path(view._root, "unexpected-directory").mkdir()

    assert not view.revalidate()
    with pytest.raises(SourceEvidenceInvalid):
        build_source_evidence_view(log)


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_source_projection_rejects_aliased_or_special_ledger(tmp_path, kind):
    log = _log(tmp_path)
    log.append("message", {"text": "source"})
    ledger = Path(log.path)
    retained = tmp_path / "retained.jsonl"

    if kind == "symlink":
        ledger.rename(retained)
        ledger.symlink_to(retained)
    elif kind == "hardlink":
        os.link(ledger, retained)
    else:
        ledger.unlink()
        os.mkfifo(ledger)

    with pytest.raises(SourceProjectionUnavailable):
        log.source_projection()


@pytest.mark.parametrize(
    "raw",
    [
        b'{"seq":1,"seq":1,"ts":1,"type":"message","payload":{}}\n',
        b'{"seq":1,"ts":NaN,"type":"message","payload":{}}\n',
        b'{"seq":1,"ts":1,"type":"message","payload":{}}',
        (
            b'{"seq":1,"ts":1,"type":"message","payload":{}}\n\n'
            b'{"seq":2,"ts":2,"type":"stop","payload":{}}\n'
        ),
        b'{"seq":1,"ts":1,"type":"unknown","payload":{}}\n',
        b'{"seq":1,"ts":1,"type":"message","payload":{},"extra":1}\n',
    ],
)
def test_source_projection_rejects_noncanonical_records(tmp_path, raw):
    log = _log(tmp_path)
    Path(log.path).write_bytes(raw)

    with pytest.raises(SourceProjectionUnavailable):
        log.source_projection()


def test_unicode_next_line_is_payload_data_not_a_record_boundary(tmp_path):
    log = _log(tmp_path)
    log.append("message", {"text": "before\u0085after"})

    projection = log.source_projection()
    assert projection[0]["payload"]["text"] == "before\u0085after"
    assert build_source_evidence_view(log).revalidate()


def test_blob_cap_failure_publishes_no_partial_view(tmp_path, monkeypatch):
    monkeypatch.setattr(source_evidence, "MAX_TOTAL_BLOB_BYTES", 1100)
    log = _log(tmp_path)
    log.append("tool_result", {"one": "a" * 1500, "two": "b" * 1500})

    with pytest.raises(SourceEvidenceInvalid, match="total byte limit"):
        build_source_evidence_view(log)

    assert not (tmp_path / source_evidence.VIEW_DIRECTORY).exists()


def test_pointer_amplification_cap_publishes_no_partial_view(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(source_evidence, "MAX_JSON_POINTER_BYTES", 32)
    log = _log(tmp_path)
    log.append("tool_result", {"long-key-" * 10: "x" * 1500})

    with pytest.raises(SourceEvidenceInvalid, match="pointer"):
        build_source_evidence_view(log)

    assert not (tmp_path / source_evidence.VIEW_DIRECTORY).exists()


def test_publication_permissions_and_inventory_are_closed(tmp_path):
    log = _log(tmp_path)
    log.append("tool_result", {"output": "x" * 1500})
    view = build_source_evidence_view(log)

    assert stat.S_IMODE(os.lstat(Path(view._root).parent).st_mode) == 0o700
    assert stat.S_IMODE(os.lstat(view._root).st_mode) == 0o500
    for directory, dirnames, filenames in os.walk(view._root):
        for name in dirnames:
            assert stat.S_IMODE(os.lstat(Path(directory, name)).st_mode) == 0o500
        for name in filenames:
            assert stat.S_IMODE(os.lstat(Path(directory, name)).st_mode) == 0o400
    assert not [name for name in os.listdir(Path(view._root).parent)
                if name.startswith(".building-")]


def test_concurrent_builders_publish_one_identical_view(tmp_path):
    log = _log(tmp_path)
    log.append("message", {"text": "source"})

    with ThreadPoolExecutor(max_workers=8) as pool:
        views = list(pool.map(
            lambda _index: build_source_evidence_view(log), range(24)))

    assert len({view.view_id for view in views}) == 1
    assert len({view.manifest_path for view in views}) == 1
    assert all(view.revalidate() for view in views)
    evidence_root = Path(views[0]._root).parent
    assert sorted(path.name for path in evidence_root.iterdir()) == [
        views[0].view_id]


def test_preplanted_writable_publication_root_is_rejected(tmp_path):
    evidence_root = tmp_path / source_evidence.VIEW_DIRECTORY
    evidence_root.mkdir(mode=0o755)
    log = _log(tmp_path)
    log.append("message", {"text": "source"})

    with pytest.raises(SourceEvidenceInvalid, match="identity is unsafe"):
        build_source_evidence_view(log)


def test_member_name_swap_during_validation_is_detected(tmp_path, monkeypatch):
    log = _log(tmp_path)
    log.append("message", {"text": "source"})
    view = build_source_evidence_view(log)
    page = next(Path(view._root, "pages").iterdir())
    page_inode = page.stat().st_ino
    original_read = os.read
    swapped = False

    def swap_after_read(descriptor, size):
        nonlocal swapped
        data = original_read(descriptor, size)
        if data and not swapped and os.fstat(descriptor).st_ino == page_inode:
            swapped = True
            page.parent.chmod(0o700)
            retained = page.with_suffix(".retained")
            page.rename(retained)
            page.write_bytes(data + b" ")
            page.chmod(0o400)
        return data

    monkeypatch.setattr(source_evidence.os, "read", swap_after_read)
    assert not view.revalidate()
    assert swapped


def test_evidence_root_swap_during_validation_is_detected(
    tmp_path, monkeypatch,
):
    log = _log(tmp_path)
    log.append("message", {"text": "source"})
    view = build_source_evidence_view(log)
    evidence_root = Path(view._root).parent
    retained = tmp_path / "retained-evidence"
    original_open = source_evidence._open_directory_at
    swapped = False

    def swap_after_view_open(parent_fd, name, *, mode):
        nonlocal swapped
        descriptor = original_open(parent_fd, name, mode=mode)
        if name == view.view_id and not swapped:
            swapped = True
            evidence_root.rename(retained)
            shutil.copytree(retained, evidence_root, copy_function=shutil.copy2)
        return descriptor

    monkeypatch.setattr(
        source_evidence, "_open_directory_at", swap_after_view_open)
    assert not view.revalidate()
    assert swapped
