"""The candidate response is a harness fact supplied to every seat as data."""
import json
import os

from verifier.execution.instruments import authoritative_task_context


def test_online_state_names_the_final_claim_as_the_candidate_response(tmp_path):
    (tmp_path / "task.md").write_text("Answer the question. End your reply "
                                      "with a line FINAL ANSWER: <n>\n")
    ctx = json.loads(authoritative_task_context(str(tmp_path)))
    cr = ctx["candidate_response"]
    assert cr["reply_written"] is False
    assert cr["path"] == "candidate_response.txt"
    assert "final claim" in cr["note"]
    assert "never refute the response's presence" in cr["note"]
    assert "never write one there" in cr["note"]


def test_offline_state_points_at_the_delivered_reply(tmp_path):
    (tmp_path / "task.md").write_text("Answer.\n")
    (tmp_path / "final.txt").write_text("FINAL ANSWER: 7\n")
    ctx = json.loads(authoritative_task_context(str(tmp_path)))
    cr = ctx["candidate_response"]
    assert cr["reply_written"] is True
    assert cr["path"] == "final.txt"


def test_no_task_statement_means_no_context(tmp_path):
    assert authoritative_task_context(str(tmp_path)) == "(none supplied)"


def test_the_statement_is_marked_as_supplied_data_not_a_world_file(tmp_path):
    (tmp_path / "task.md").write_text("Compute the value.\n")
    ctx = json.loads(authoritative_task_context(str(tmp_path)))
    assert ctx["text"] == "Compute the value.\n"
    note = ctx["statement_note"]
    assert "not an artifact of the delivered world" in note
    assert "never write a copy" in note


def test_publishing_writes_the_final_text_unless_a_reply_is_delivered(tmp_path):
    from verifier.execution.instruments import publish_candidate_response
    path = publish_candidate_response(str(tmp_path), "The answer is 42.")
    assert path and open(path).read() == "The answer is 42."
    (tmp_path / "final.txt").write_text("FINAL ANSWER: 42\n")
    assert publish_candidate_response(str(tmp_path), "other") is None


def test_the_kernel_publishes_the_final_under_check(tmp_path):
    from verifier.core.events import EventLog
    from verifier.core.model import Claim, CLAIM_FINAL
    from verifier.core.store import ClaimStore, TRIGGER_ON_COMMIT
    from verifier.engine.kernel import Kernel
    from verifier.execution.discharge import DischargeResult
    from verifier.execution.instruments import ArtifactStore
    from verifier.llm.schema import FixtureBackend

    class _Discharger:
        def discharge(self, row, obligation, method=None, tag=""):
            return DischargeResult(witness_kind="execution", outcome="holds",
                                   witness=None, artifact="", instrument="stub",
                                   basis="world")

    log = EventLog(os.path.join(tmp_path, "events.jsonl"))
    store = ClaimStore(trigger=TRIGGER_ON_COMMIT)
    backend = FixtureBackend({
        "obligations": {"obligations": [
            {"text": "the stated answer is an integer", "required_basis": "world"}]},
        "critic": {"verdict": "pass", "counterexample_world": "n/a",
                   "missing_condition": "n/a"}})
    Kernel(store, log, backend, ArtifactStore(os.path.join(tmp_path, "art")),
           discharger=_Discharger())
    store.add_claim(Claim(CLAIM_FINAL, "The answer is 42."))
    published = tmp_path / "candidate_response.txt"
    assert published.read_text() == "The answer is 42."
    assert any(e["type"] == "candidate_response_published" for e in log.replay())
