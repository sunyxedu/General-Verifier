"""A replay is a published finite prefix, not a moving append target."""
from verifier.core.events import EventLog


def test_replay_snapshot_does_not_chase_later_append_and_releases_writer_lock(tmp_path):
    log = EventLog(str(tmp_path / "events.jsonl"))
    log.append("message", {"text": "first"})
    log.append("message", {"text": "second"})
    replay = log.replay()
    assert next(replay)["seq"] == 1
    log.append("message", {"text": "published after snapshot"})
    assert [event["seq"] for event in replay] == [2]
    assert [event["seq"] for event in log.replay()] == [1, 2, 3]
