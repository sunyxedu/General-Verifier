"""Tail + incremental builder: online mode is the same stream, lower latency."""
import json

import pytest

from verifier.builder import Builder
from verifier.events import EventLog
from verifier.instruments import ArtifactStore
from verifier.kernel import Kernel
from verifier.roles import FixtureBackend
from verifier.store import ClaimStore
from verifier.tail import Tail, TailError


def test_poll_consumes_incrementally_and_tolerates_torn_lines(tmp_path):
    path = str(tmp_path / "e.jsonl")
    seen = []
    tail = Tail(path, seen.append)

    assert tail.poll() == 0                      # no file yet

    line1 = json.dumps({"seq": 1, "type": "session_start", "payload": {}})
    line2 = json.dumps({"seq": 2, "type": "message", "payload": {"t": "x"}})
    with open(path, "w") as f:
        f.write(line1 + "\n" + line2[:10])       # second line torn mid-record
    assert tail.poll() == 1
    assert seen[0]["seq"] == 1

    with open(path, "a") as f:
        f.write(line2[10:] + "\n")               # torn line completed
    assert tail.poll() == 1
    assert seen[1]["seq"] == 2 and seen[1]["payload"] == {"t": "x"}
    assert tail.poll() == 0                      # nothing new


def test_seq_gap_raises_instead_of_absorbing(tmp_path):
    path = str(tmp_path / "e.jsonl")
    tail = Tail(path, lambda e: None)
    with open(path, "w") as f:
        f.write(json.dumps({"seq": 1, "type": "message", "payload": {}}) + "\n")
        f.write(json.dumps({"seq": 3, "type": "message", "payload": {}}) + "\n")
    with pytest.raises(TailError, match="expected seq 2"):
        tail.poll()


def test_online_builder_fires_kernel_with_zero_changes(tmp_path):
    """Events appended after attach still drive verification lazily."""
    log = EventLog(str(tmp_path / "e.jsonl"))
    store = ClaimStore()
    backend = FixtureBackend({
        "compile_obligations": {"obligations": [
            {"text": "O1", "failure_looks_like": "反例"}]},
        "write_check": {"check_source":
                        'import json\nprint(json.dumps({"holds": True}))'},
    })
    kernel = Kernel(store, log, backend,
                    ArtifactStore(str(tmp_path / "a")))
    builder = Builder(log, store)
    tail = Tail(log.path, builder.feed)

    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "某 claim", "deps": []})
    tail.poll()
    a = builder.labels["A"]
    assert store.status(a) == "unverified"       # no dependent yet: not a cent

    log.append("commit", {"label": "F", "kind": "final",
                          "text": "结论", "deps": ["A"]})
    tail.poll()                                   # F arrives -> A gains first dependent
    assert store.status(a) == "verified"


def test_replay_and_tail_agree(tmp_path):
    log = EventLog(str(tmp_path / "e.jsonl"))
    log.append("tool_call", {"tool": "python"})
    log.append("tool_result", {"exit": 0})
    log.append("commit", {"label": "A", "kind": "assertion",
                          "text": "t", "deps": ["effect:2"]})

    replayed = Builder(log, ClaimStore()).replay()
    tailed = Builder(log, ClaimStore())
    Tail(log.path, tailed.feed).poll()
    assert replayed.labels == tailed.labels
