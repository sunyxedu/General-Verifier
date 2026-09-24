"""The solver cannot see the ledger's sequence numbers: every gv response
lists the tool results recorded since the previous response, each under the
id a claim cites it by, with a glimpse of its input and output."""
from __future__ import annotations

import json
import os

import pytest

from verifier.core.store import TRIGGER_LAZY
from verifier.mcp_server import HOOK_FEED, RECENT_EFFECTS, Service, _excerpt


class Idle:
    def __init__(self, service):
        pass

    def verify(self, claim_id, force=False):
        return "unverified"


def service(state):
    return Service(str(state), trigger=TRIGGER_LAZY,
                   kernel_factory=lambda sc: Idle(sc))


def hook(state, *events):
    with open(os.path.join(state, HOOK_FEED), "a") as f:
        for etype, payload in events:
            f.write(json.dumps({"type": etype, "payload": payload}) + "\n")


def bash(state, uid, command, output):
    hook(state,
         ("tool_call", {"tool": "Bash", "tool_use_id": uid,
                        "input": {"command": command}}),
         ("tool_result", {"tool": "Bash", "tool_use_id": uid,
                          "excerpt": json.dumps(output)}))


def test_a_response_lists_the_effects_recorded_since_the_last_one(tmp_path):
    svc = service(tmp_path)
    bash(tmp_path, "u1", "pytest -q tests/", "5 passed in 0.3s\n")
    hook(tmp_path,
         ("tool_call", {"tool": "mcp__gv__status", "tool_use_id": "g1", "input": {}}),
         ("tool_result", {"tool": "mcp__gv__status", "tool_use_id": "g1",
                          "excerpt": "{}"}))
    first = svc.status()
    listed = first["recent_effects"]
    assert len(listed) == 1
    entry, = listed
    assert entry["tool"] == "Bash"
    assert entry["input"] == "command=pytest -q tests/"
    assert entry["output"] == "5 passed in 0.3s"
    # the id resolves to that very result
    assert entry["ref"] in svc.builder.labels
    seq = int(entry["ref"].split(":")[1])
    result = [e for e in svc.log.replay() if e["seq"] == seq][0]
    assert result["type"] == "tool_result" and result["payload"]["tool_use_id"] == "u1"
    # already listed: the next response does not repeat it
    assert "recent_effects" not in svc.status()
    bash(tmp_path, "u2", "cat out.txt", "done\n")
    out = svc.commit("A1", "assertion", "out.txt says done", [])
    assert [e["input"] for e in out["recent_effects"]] == ["command=cat out.txt"]


def test_only_the_newest_are_listed(tmp_path):
    svc = service(tmp_path)
    for i in range(RECENT_EFFECTS + 3):
        bash(tmp_path, f"u{i}", f"echo {i}", f"{i}\n")
    listed = svc.status()["recent_effects"]
    assert len(listed) == RECENT_EFFECTS
    assert listed[-1]["output"] == str(RECENT_EFFECTS + 2)


def test_a_restarted_server_lists_the_record_it_rebuilt(tmp_path):
    svc = service(tmp_path)
    bash(tmp_path, "u1", "make", "ok\n")
    svc.drain_hooks()
    again = service(tmp_path)
    assert [e["input"] for e in again.status()["recent_effects"]] == ["command=make"]


@pytest.mark.parametrize("value,expected", [
    ('"line one\\nline two"', "line one line two"),
    ({"command": "ls -la", "timeout": 5}, "command=ls -la timeout=5"),
    ("not json", "not json"),
])
def test_excerpts_are_one_line(value, expected):
    assert _excerpt(value) == expected


def test_a_long_excerpt_keeps_both_ends(tmp_path):
    text = "start " + "x" * 500 + " end"
    out = _excerpt(json.dumps(text))
    assert out.startswith("start") and out.endswith("end") and "…" in out
    assert len(out) <= 101
