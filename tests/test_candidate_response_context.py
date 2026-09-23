"""The candidate response is a harness fact supplied to every seat as data."""
import json
import os

from verifier.execution.instruments import authoritative_task_context


def test_online_state_names_the_final_claim_as_the_candidate_response(tmp_path):
    (tmp_path / "task.md").write_text("Answer the question. End your reply "
                                      "with a line FINAL ANSWER: <n>\n")
    ctx = json.loads(authoritative_task_context(str(tmp_path)))
    cr = ctx["candidate_response"]
    assert cr["delivered"] is False
    assert "final claim" in cr["note"]
    assert "not to the claim" in cr["note"]


def test_offline_state_points_at_the_delivered_reply(tmp_path):
    (tmp_path / "task.md").write_text("Answer.\n")
    (tmp_path / "final.txt").write_text("FINAL ANSWER: 7\n")
    ctx = json.loads(authoritative_task_context(str(tmp_path)))
    cr = ctx["candidate_response"]
    assert cr["delivered"] is True
    assert cr["path"] == os.path.join(str(tmp_path), "final.txt")


def test_no_task_statement_means_no_context(tmp_path):
    assert authoritative_task_context(str(tmp_path)) == "(none supplied)"
