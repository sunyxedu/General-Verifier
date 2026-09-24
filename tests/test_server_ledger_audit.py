"""The server's ledger survives what an audit found breaking it: a commit
whose reference failed no longer kills every restart, the all-labels status
counts effects instead of listing each, delivered news survives a restart
and a claim refuted again is news again, a second server on a served state
exits without touching it, a bad hook-feed line is skipped (and a restart
resumes exactly where the ledger stops), a torn ledger tail is cut off, and
a lone surrogate in a tool call never drops the call."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from verifier.core.events import EventLog, TORN_SUFFIX
from verifier.core.store import TRIGGER_LAZY
from verifier.mcp_server import (
    EFFECTS_LISTED, FEED_OFFSET, HOOK_FEED, SERVER_LOCK, SERVER_PIDFILE,
    Service, StateBusy, hold_server_lock, launch_identity)
from verifier.stream.tail import Tail

ROOT = Path(__file__).resolve().parent.parent


class Idle:
    def __init__(self, service):
        self.service = service

    def verify(self, claim_id, force=False):
        return self.service.store.status(claim_id)


def service(state, **kw):
    return Service(str(state), trigger=TRIGGER_LAZY,
                   kernel_factory=lambda sc: Idle(sc), **kw)


def feed(state, *lines):
    with open(os.path.join(state, HOOK_FEED), "a", encoding="utf-8") as f:
        for line in lines:
            f.write((line if isinstance(line, str) else json.dumps(line)) + "\n")


def bash(uid, command="ls"):
    return ({"type": "tool_call", "payload": {
                "tool": "Bash", "tool_use_id": uid, "input": {"command": command}}},
            {"type": "tool_result", "payload": {
                "tool": "Bash", "tool_use_id": uid, "excerpt": "\"ok\""}})


def events(svc, etype):
    return [e for e in svc.log.replay() if e["type"] == etype]


# 1. a commit whose reference failed

def test_a_failed_commit_is_marked_and_a_restart_skips_it(tmp_path):
    svc = service(tmp_path)
    out = svc.commit("A1", "assertion", "x holds", ["nope"])
    assert "unknown claim reference" in out["error"]
    commit, = events(svc, "commit")
    failed, = events(svc, "commit_failed")
    assert failed["payload"]["commit"] == commit["seq"]
    again = service(tmp_path)                    # used to raise TraceError
    assert "A1" not in again.builder.labels
    assert again.commit("A2", "assertion", "y holds", [])["status"]
    assert len(events(again, "commit_failed")) == 1


def test_a_ledger_with_an_unmarked_failed_commit_still_starts(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("commit", {"label": "A1", "kind": "assertion",
                          "text": "x holds", "deps": ["effect:99"]})
    log.append("commit", {"label": "A2", "kind": "assertion",
                          "text": "y holds", "deps": []})
    svc = service(tmp_path)
    assert "A2" in svc.builder.labels and "A1" not in svc.builder.labels
    failed, = events(svc, "commit_failed")
    assert failed["payload"]["commit"] == 1
    service(tmp_path)                            # and again: marked once
    assert len(events(svc, "commit_failed")) == 1


def test_a_failed_recommit_keeps_the_label_on_its_claim(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "x holds", [])
    first = svc.builder.labels["A1"]
    svc.commit("A1", "assertion", "x holds twice", ["nope"])
    assert svc.builder.labels["A1"] == first
    assert service(tmp_path).builder.labels["A1"] == first


# 2. the all-labels status

def test_the_status_map_counts_effects_instead_of_listing_each(tmp_path):
    svc = service(tmp_path)
    for i in range(EFFECTS_LISTED + 10):
        feed(tmp_path, *bash(f"u{i}"))
    svc.commit("A1", "assertion", "x holds", [])
    out = svc.status()
    assert out["A1"] == "unverified"
    assert not any(k.startswith("effect:") for k in out)
    assert out["effects"]["recorded"] == EFFECTS_LISTED + 10
    assert len(out["effects"]["latest"]) == EFFECTS_LISTED
    assert out["effects"]["latest"][-1] in svc.builder.labels


# 3. news

def refute(svc, label, log_it=True):
    cid = svc.builder.labels[label]
    if svc.store.status(cid) != "unverified":
        svc.store.set_stale(cid)
    svc.store.set_refuted(cid, reason="counterexample")
    if log_it:
        svc.log.append("status_change", {"claim": cid, "status": "refuted"})
    return cid


def news_labels(result):
    return [n["label"] for n in result.get("news", [])]


def test_delivered_news_survives_a_restart(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "x holds", [])
    refute(svc, "A1")
    assert news_labels(svc.commit("A2", "assertion", "y", [])) == ["A1"]
    assert news_labels(svc.commit("A3", "assertion", "z", [])) == []
    again = service(tmp_path)
    assert news_labels(again.commit("A4", "assertion", "w", [])) == []


def test_a_claim_refuted_again_is_news_again(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "x holds", [])
    refute(svc, "A1")
    assert news_labels(svc.status("A1")) == []     # told by the residue
    assert news_labels(svc.commit("A2", "assertion", "y", [])) == []
    refute(svc, "A1")                                # stale, rechecked
    assert news_labels(svc.commit("A3", "assertion", "z", [])) == ["A1"]
    assert news_labels(service(tmp_path).commit("A4", "assertion", "w", [])) == []


def test_news_told_before_its_event_landed_is_told_once(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "x holds", [])
    cid = refute(svc, "A1", log_it=False)            # the store moves first
    assert news_labels(svc.commit("A2", "assertion", "y", [])) == ["A1"]
    svc.log.append("status_change", {"claim": cid, "status": "refuted"})
    assert news_labels(svc.commit("A3", "assertion", "z", [])) == []
    assert news_labels(service(tmp_path).commit("A4", "assertion", "w", [])) == []


def test_a_restart_does_not_list_effects_a_response_already_listed(tmp_path):
    svc = service(tmp_path)
    feed(tmp_path, *bash("u1", "make"))
    assert svc.status()["recent_effects"]
    feed(tmp_path,
         {"type": "tool_call", "payload": {"tool": "mcp__gv__status",
                                           "tool_use_id": "g1", "input": {}}},
         {"type": "tool_result", "payload": {"tool": "mcp__gv__status",
                                             "tool_use_id": "g1", "excerpt": "{}"}})
    svc.drain_hooks()
    feed(tmp_path, *bash("u2", "make test"))
    svc.drain_hooks()
    again = service(tmp_path)
    assert [e["input"] for e in again.status()["recent_effects"]] == [
        "command=make test"]


# 4. one server per state

def test_the_state_lock_admits_one_holder(tmp_path):
    held = hold_server_lock(str(tmp_path))
    assert held is not None
    with pytest.raises(StateBusy):
        hold_server_lock(str(tmp_path))
    held.close()                                     # the holder died
    hold_server_lock(str(tmp_path)).close()


def test_a_second_server_on_a_served_state_exits_untouched(tmp_path):
    held = hold_server_lock(str(tmp_path))
    try:
        done = subprocess.run(
            [sys.executable, "-m", "verifier.mcp_server",
             "--state-dir", str(tmp_path)],
            cwd=ROOT, capture_output=True, text=True, timeout=120,
            stdin=subprocess.DEVNULL)
    finally:
        held.close()
    assert done.returncode != 0 and SERVER_LOCK in done.stderr
    assert not (tmp_path / "events.jsonl").exists()
    assert not (tmp_path / SERVER_PIDFILE).exists()


def test_a_lock_holder_is_not_a_concurrent_start_whatever_the_pidfile_says(tmp_path):
    from verifier.engine.certificate import verifier_integrity
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    held = hold_server_lock(str(tmp_path))
    try:
        # a stale pidfile naming a live (or zombie, or reused) pid
        (tmp_path / SERVER_PIDFILE).write_text(str(alive.pid))
        svc = service(tmp_path, server_lock=held)
        import argparse
        svc.record_start(launch_identity(argparse.Namespace(state_dir=str(tmp_path))))
        assert verifier_integrity(svc.log) == {"server_starts": 1}
        assert (tmp_path / SERVER_PIDFILE).read_text() == str(os.getpid())
    finally:
        held.close()
        alive.kill()
        alive.wait()


# 5./6. the hook feed

def test_bad_feed_lines_are_skipped_and_the_rest_ingested(tmp_path):
    svc = service(tmp_path)
    call, result = bash("u1")
    feed(tmp_path, call, "not json {", "[1, 2]",
         {"type": "tool_call", "payload": "a string"},
         '{"type": "tool_call", "payload": {"tool": "Bash", "x": NaN}}',
         result)
    svc.drain_hooks()
    assert [e["payload"]["tool_use_id"] for e in events(svc, "tool_call")] == ["u1"]
    assert len(events(svc, "tool_result")) == 1
    rejected = events(svc, "hook_feed_rejected")
    assert len(rejected) == 4
    assert "not json" in rejected[0]["payload"]["excerpt"]
    size = os.path.getsize(tmp_path / HOOK_FEED)
    assert int((tmp_path / FEED_OFFSET).read_text()) == size
    again = service(tmp_path)
    again.drain_hooks()
    assert len(events(again, "tool_call")) == 1
    assert len(events(again, "hook_feed_rejected")) == 4


def test_a_drain_error_never_fails_the_verb(tmp_path):
    svc = service(tmp_path)

    def broken():
        raise RuntimeError("disk gone")
    svc._feed.poll = broken
    out = svc.commit("A1", "assertion", "x holds", [])
    assert "error" not in out and out["status"] == "unverified"


def test_a_line_whose_handler_raises_stays_unconsumed(tmp_path):
    path = tmp_path / "feed.jsonl"
    path.write_text('{"n": 1}\n{"n": 2}\n{"n": 3}\n')
    seen, fail = [], {2}

    def on_event(event):
        if event["n"] in fail:
            raise RuntimeError("once")
        seen.append(event["n"])
    tail = Tail(str(path), on_event, raw=True)
    with pytest.raises(RuntimeError):
        tail.poll()
    assert seen == [1] and tail.consumed == len('{"n": 1}\n')
    fail.clear()
    assert tail.poll() == 2 and seen == [1, 2, 3]
    assert tail.consumed == path.stat().st_size


def test_a_restart_resumes_from_the_ledger_when_the_offset_file_lags(tmp_path):
    svc = service(tmp_path)
    feed(tmp_path, *bash("u1"), *bash("u2"))
    svc.drain_hooks()
    offsets = [e["payload"]["feed_offset"] for e in events(svc, "tool_result")]
    assert offsets[-1] == os.path.getsize(tmp_path / HOOK_FEED)
    # killed after the events were appended, before the offset was saved
    (tmp_path / FEED_OFFSET).write_text("0")
    again = service(tmp_path)
    again.drain_hooks()
    assert [e["payload"]["tool_use_id"] for e in events(again, "tool_call")] == [
        "u1", "u2"]


# 7. a torn ledger tail

def test_a_torn_last_line_is_skipped_then_cut_off(tmp_path):
    path = tmp_path / "events.jsonl"
    log = EventLog(str(path))
    log.append("commit", {"label": "A1", "kind": "assertion",
                          "text": "x holds", "deps": []})
    with open(path, "ab") as f:
        f.write(b'{"payload": {"label": "A2", "kind": "ass')   # disk full
    svc = service(tmp_path)                           # used to never start
    assert "A1" in svc.builder.labels
    assert svc.log.last_seq() == 1
    svc.commit("A2", "assertion", "y holds", [])
    seqs = [e["seq"] for e in EventLog(str(path)).replay()]
    assert seqs == list(range(1, len(seqs) + 1)) and len(seqs) >= 2
    assert b"kind\": \"ass" in (tmp_path / ("events.jsonl" + TORN_SUFFIX)).read_bytes()
    assert path.read_bytes().endswith(b"\n")


def test_a_last_record_larger_than_the_scan_window_keeps_its_seq(tmp_path):
    path = str(tmp_path / "events.jsonl")
    first, second = EventLog(path), EventLog(path)
    first.append("message", {"text": "x" * 200_000})
    assert second.append("message", {"text": "y"})["seq"] == 2
    assert EventLog(path).last_seq() == 2
    assert [e["seq"] for e in EventLog(path).replay()] == [1, 2]


# 8. labels inserted while a status map is built

def test_the_status_map_tolerates_labels_added_meanwhile(tmp_path):
    svc = service(tmp_path)
    svc.commit("A1", "assertion", "x holds", [])
    svc.commit("A2", "assertion", "y holds", [])
    get = svc.store.get

    def get_while_committing(cid):
        svc.builder.labels.setdefault(f"late-{len(svc.builder.labels)}", cid)
        return get(cid)
    svc.store.get = get_while_committing
    out = svc.status()
    assert "A1" in out and "A2" in out


# 9. lone surrogates in the hooks

@pytest.mark.parametrize("hook,key", [("pre_tool_use.py", "tool_input"),
                                      ("post_tool_use.py", "tool_response")])
def test_a_lone_surrogate_never_drops_the_call(tmp_path, hook, key):
    event = {"tool_name": "Bash", "tool_use_id": "u1", "session_id": "s",
             key: {"command": "cat \udcff.bin", "stdout": "\ud800"}}
    stdin = json.dumps(event)            # ASCII, with \\udcff escapes
    feed_path = tmp_path / HOOK_FEED
    done = subprocess.run(
        [sys.executable, str(ROOT / "hooks" / hook)], input=stdin,
        capture_output=True, text=True, timeout=60,
        env=dict(os.environ, GV_HOOK_FEED=str(feed_path)))
    assert done.returncode == 0, done.stderr
    line, = feed_path.read_bytes().decode("utf-8").splitlines()
    assert json.loads(line)["payload"]["tool_use_id"] == "u1"
    svc = service(tmp_path)
    svc.drain_hooks()
    assert not events(svc, "hook_feed_rejected")
    assert len(events(svc, json.loads(line)["type"])) == 1
