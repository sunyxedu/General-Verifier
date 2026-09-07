import json
from pathlib import Path

from experiments.terminal_bench import analyze_typed_adapter_eval as analyze
from experiments.terminal_bench import run_typed_adapter_eval as runner


MANIFEST = (
    Path(__file__).parents[1]
    / "experiments/terminal_bench/typed_adapter_eval_manifest.json"
)


def _attempt(**updates):
    row = {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": "call-1",
        "attempt_id": "attempt-1",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "input_tokens": 10,
        "uncached_input_tokens": 6,
        "cached_input_tokens": 4,
        "output_tokens": 3,
    }
    row.update(updates)
    return row


def test_canary_warm_is_one_process_per_task_and_arm():
    manifest = runner._manifest(MANIFEST)
    selected = runner._selected(manifest, "canary")
    groups = runner._groups("warm", selected)

    assert len(groups) == 3
    assert all(len(group.tasks) == 1 for group in groups)
    assert all(len(group.trials) == 2 for group in groups)

    schedule = runner._schedule("warm", groups, ("control", "treatment"))
    assert len(schedule) == 6
    for group in groups:
        assert {arm for arm, item in schedule if item == group} == {
            "control", "treatment",
        }


def test_canary_all_plan_has_every_phase_in_one_ledger():
    manifest = runner._manifest(MANIFEST)
    rows = runner._plan_rows(
        manifest, MANIFEST.parents[2], "full-e2e", "canary",
        runner.PHASES, ("control", "treatment"), Path("/tmp/eval"),
    )

    assert len(rows) == 22
    assert {row["phase"] for row in rows} == set(runner.PHASES)


def test_batch_schedule_is_globally_serial_and_each_arm_contiguous():
    manifest = runner._manifest(MANIFEST)
    groups = runner._groups(
        "batch", runner._selected(manifest, "acceptance"))
    schedule = runner._schedule("batch", groups, ("control", "treatment"))

    arms = [arm for arm, _group in schedule]
    split = len(groups)
    assert len(schedule) == 2 * split
    assert len(set(arms[:split])) == 1
    assert len(set(arms[split:])) == 1
    assert arms[0] != arms[-1]
    assert [group.key for _arm, group in schedule[:split]] == [
        group.key for group in groups
    ]
    assert [group.key for _arm, group in schedule[split:]] == [
        group.key for group in groups
    ]


def test_wrapper_attempt_audit_rejects_unknown_or_inconsistent_cost(tmp_path):
    usage = tmp_path / "usage.jsonl"
    usage.write_text(json.dumps(_attempt()) + "\n", encoding="utf-8")
    assert runner._attempt_audit(usage) == (True, [])

    usage.write_text(json.dumps(_attempt(
        usage_complete=False,
        usage_status="unknown",
        input_tokens=None,
        uncached_input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
    )) + "\n", encoding="utf-8")
    valid, errors = runner._attempt_audit(usage)
    assert not valid
    assert "incomplete/invalid token usage" in " ".join(errors)

    usage.write_text(json.dumps(_attempt(
        uncached_input_tokens=7,
    )) + "\n", encoding="utf-8")
    valid, errors = runner._attempt_audit(usage)
    assert not valid
    assert "inconsistent input token usage" in " ".join(errors)


def test_analyzer_rejects_paths_outside_a_run_root(tmp_path):
    run = {"run_id": "run-1"}
    inside = tmp_path / "runs/run-1/results.jsonl"
    outside = tmp_path / "other/results.jsonl"

    assert analyze._private_path(
        run, tmp_path, str(inside), "results")[1] is None
    assert analyze._private_path(
        run, tmp_path, str(outside), "results")[1] is not None
