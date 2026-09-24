"""A restarted server resumes the hook feed where the ledger stops (it used
to re-read the feed from its start: 8164 tool-call rows for 1169 calls after
eleven restarts), and a restart with other settings or code, or a second
server beside a live one, voids the session's certificates."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from verifier.core.store import TRIGGER_LAZY
from verifier.engine.certificate import verifier_integrity
from verifier.mcp_server import (
    FEED_OFFSET, HOOK_FEED, SERVER_PIDFILE, Service, launch_identity)


class Idle:
    def __init__(self, service):
        self.service = service

    def verify(self, claim_id, force=False):
        return self.service.store.status(claim_id)


def service(state):
    return Service(str(state), trigger=TRIGGER_LAZY,
                   kernel_factory=lambda sc: Idle(sc))


def bash(state, uid, command):
    with open(os.path.join(state, HOOK_FEED), "a") as f:
        f.write(json.dumps({"type": "tool_call", "payload": {
            "tool": "Bash", "tool_use_id": uid, "input": {"command": command}}}) + "\n")
        f.write(json.dumps({"type": "tool_result", "payload": {
            "tool": "Bash", "tool_use_id": uid, "excerpt": "\"ok\""}}) + "\n")


def calls(svc):
    return [e["payload"]["tool_use_id"] for e in svc.log.replay()
            if e["type"] == "tool_call"]


def test_a_restarted_server_does_not_ingest_the_feed_again(tmp_path):
    first = service(tmp_path)
    bash(tmp_path, "u1", "make")
    bash(tmp_path, "u2", "make test")
    first.drain_hooks()
    assert calls(first) == ["u1", "u2"]
    assert (tmp_path / FEED_OFFSET).exists()
    again = service(tmp_path)          # the restart
    again.drain_hooks()
    assert calls(again) == ["u1", "u2"]
    bash(tmp_path, "u3", "ls")
    again.drain_hooks()
    assert calls(again) == ["u1", "u2", "u3"]


def test_a_torn_last_line_is_read_whole_after_a_restart(tmp_path):
    first = service(tmp_path)
    bash(tmp_path, "u1", "make")
    with open(tmp_path / HOOK_FEED, "a") as f:
        f.write('{"type": "tool_call", "payload": {"tool": "Bash", "tool_use_id": "u2", ')
    first.drain_hooks()
    again = service(tmp_path)
    with open(tmp_path / HOOK_FEED, "a") as f:
        f.write('"input": {"command": "ls"}}}\n')
    again.drain_hooks()
    assert calls(again) == ["u1", "u2"]


def launch(tmp_path, **env):
    args = argparse.Namespace(state_dir=str(tmp_path), seats="codex",
                              model="gpt-5.6-luna", effort="max")
    return launch_identity(args, environ={"GV_SEAT_EFFORT": "max", **env})


def test_a_restart_with_the_same_settings_is_recorded_not_voided(tmp_path):
    svc = service(tmp_path)
    svc.record_start(launch(tmp_path))
    os.remove(tmp_path / SERVER_PIDFILE)          # the first server died
    service(tmp_path).record_start(launch(tmp_path))
    integrity = verifier_integrity(svc.log)
    assert integrity == {"server_starts": 2}


def test_a_restart_with_other_settings_voids_the_certificate(tmp_path):
    svc = service(tmp_path)
    svc.record_start(launch(tmp_path))
    os.remove(tmp_path / SERVER_PIDFILE)
    service(tmp_path).record_start(launch(tmp_path, GV_SEAT_EFFORT="low",
                                          GV_DIRECT_SEATS="0"))
    integrity = verifier_integrity(svc.log)
    assert integrity["changed"] == ["env.GV_DIRECT_SEATS", "env.GV_SEAT_EFFORT"]
    assert "other settings" in integrity["void"]
    svc.commit("F", "final", "the task is complete", [])
    cert = svc.certificate("F")
    assert cert["verdict"] == "unverified" and "void" in cert


def test_credentials_never_enter_the_launch_record(tmp_path):
    identity = launch(tmp_path, GV_API_KEY="sk-secret", GV_AUTH_TOKEN="t")
    assert "GV_API_KEY" not in identity["env"] and "GV_AUTH_TOKEN" not in identity["env"]


def test_a_second_server_beside_a_live_one_voids(tmp_path):
    svc = service(tmp_path)
    alive = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    try:
        (tmp_path / SERVER_PIDFILE).write_text(str(alive.pid))
        svc.record_start(launch(tmp_path))
        integrity = verifier_integrity(svc.log)
        assert integrity["concurrent_starts"] == 1 and "second verifier" in integrity["void"]
    finally:
        alive.kill()
        alive.wait()
