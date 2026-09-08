import copy
import hashlib
import json
from pathlib import Path

from experiments.terminal_bench import run_typed_adapter_eval
from experiments.terminal_bench import shared_setup
from verifier.core.model import CLAIM_FINAL, Claim, Obligation
from verifier.engine.contract_bundle import (
    ContractBundleIdentity,
    build_contract_bundle_bytes,
    claim_graph_sha256,
    contract_record_sha256,
    passed_contract_entry,
)


_FINAL = Claim(CLAIM_FINAL, "the delivered result satisfies the task")
_CLAIMS = [_FINAL]
_COMPILE_INPUTS = {_FINAL.id: "4" * 64}
_CONTRACT_IDENTITY = ContractBundleIdentity(
    pipeline_sha256="5" * 64,
    config_sha256="6" * 64,
    source_sha256="7" * 64,
    graph_sha256=claim_graph_sha256(_CLAIMS),
)


def _contract_bundle_bytes():
    payload = {"obligations": [{
        "text": "the result satisfies every requested condition",
        "required_basis": "world",
        "severity": "must",
        "clause": "satisfies the task",
        "refutable_by_absence": False,
    }]}
    obligations = [Obligation(
        _FINAL.id,
        payload["obligations"][0]["text"],
        "world",
        severity="must",
        clause="satisfies the task",
    )]
    entry = passed_contract_entry(
        claim_id=_FINAL.id,
        compile_input_sha256=_COMPILE_INPUTS[_FINAL.id],
        critic_receipt_sha256=contract_record_sha256({"verdict": "pass"}),
        payload=payload,
        obligations=obligations,
    )
    return build_contract_bundle_bytes(
        identity=_CONTRACT_IDENTITY, claims=_CLAIMS, entries=[entry])


def _write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _scope(role):
    if role == "extract":
        return {
            "locus": "candidate_extract", "stage": "extract", "round": 1,
        }
    stage = {
        "obligations": "contract_compile",
        "critic": "contract_critic",
        "oracles": "oracle_enumerate",
        "discharge": "discharge",
        "judger": "judger",
    }[role]
    value = {
        "claim_id": "a" * 64,
        "claim_kind": "final",
        "locus": "final_task",
        "stage": stage,
        "round": 0,
    }
    if role in {"oracles", "discharge", "judger"}:
        value.update({
            "obligation_id": "b" * 64,
            "method_attempt": 1,
            "redo": 0,
        })
    if role == "judger":
        value.update({
            "judger_retry_index": 1,
            "independent_judger": False,
        })
    return value


def _attempt(call_number, role):
    return {
        "schema_version": 1,
        "record_type": "codex_cli_attempt",
        "attempt_schema_version": 1,
        "call_id": f"{call_number:032x}",
        "attempt_id": f"{call_number + 100:032x}",
        "attempt_index": 1,
        "started": True,
        "attempt_status": "succeeded",
        "usage_status": "complete",
        "usage_complete": True,
        "role": role,
        "model": "gpt-5.6-luna",
        "effort": "medium",
        "service_tier": "standard",
        "duration_seconds": 2.0,
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "uncached_input_tokens": 40,
        "output_tokens": 10,
        "reasoning_tokens": None,
    }


def _spans(call_number, role):
    call_id = f"{call_number:032x}"
    role_span_id = f"{call_number + 200:032x}"
    role_attempt_id = f"{call_number + 300:032x}"
    scope = _scope(role)
    base = {"span_schema_version": 1, **scope}
    return [
        {**base, "record_type": "role_span_started",
         "role_span_id": role_span_id, "role": role,
         "monotonic_ns": 1, "unix_ns": 1},
        {**base, "record_type": "role_attempt_started",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "monotonic_ns": 2, "unix_ns": 2},
        {**base, "record_type": "seat_acquired",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "queue_duration_ns": 3,
         "monotonic_ns": 3, "unix_ns": 3},
        {**base, "record_type": "model_call_started", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "model": "gpt-5.6-luna",
         "attribution": "attributed", "monotonic_ns": 4, "unix_ns": 4},
        {**base, "record_type": "model_call_finished", "call_id": call_id,
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "model": "gpt-5.6-luna",
         "attribution": "attributed", "duration_ns": 5,
         "status": "succeeded", "monotonic_ns": 9, "unix_ns": 9},
        {**base, "record_type": "seat_released",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "role": role, "held_duration_ns": 7,
         "monotonic_ns": 10, "unix_ns": 10},
        {**base, "record_type": "role_attempt_finished",
         "role_span_id": role_span_id, "role_attempt_id": role_attempt_id,
         "role_attempt_index": 1, "duration_ns": 9,
         "status": "succeeded", "monotonic_ns": 11, "unix_ns": 11},
        {**base, "record_type": "role_span_finished",
         "role_span_id": role_span_id, "role": role, "duration_ns": 11,
         "status": "succeeded", "monotonic_ns": 12, "unix_ns": 12},
    ]


def _fixture(tmp_path):
    output = tmp_path / "output"
    setup_id = "cold-job-task-trial-r1"
    setup_root = output / "setups" / setup_id
    setup_root.mkdir(parents=True)
    source = tmp_path / "source"
    (source / "verifier").mkdir(parents=True)
    (source / "verifier" / "sample.py").write_text("VALUE = 1\n")
    fingerprint = run_typed_adapter_eval._source_fingerprint(source)

    bundle = setup_root / "bundle.json"
    bundle.write_bytes(_contract_bundle_bytes())
    bundle_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    usage = setup_root / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    roles = ("extract", "obligations", "critic")
    _write_jsonl(usage, [_attempt(index, role)
                         for index, role in enumerate(roles, 1)])
    _write_jsonl(spans, [row for index, role in enumerate(roles, 1)
                         for row in _spans(index, role)])
    record = {
        "schema": shared_setup.SETUP_SCHEMA,
        "setup_id": setup_id,
        "status": "terminal",
        "comparison": "adapter-isolation",
        "tier": "canary",
        "phase": "cold",
        "rep": "r1",
        "group_key": "cold-job-task-trial",
        "job": "job",
        "tasks": ["task"],
        "trials": ["task__trial"],
        "source_root": str(source.resolve()),
        "source_commit": "1" * 40,
        "verifier_fingerprint": fingerprint,
        "verifier_fingerprint_after": fingerprint,
        "source_fingerprint_unchanged": True,
        "manifest_sha256": "2" * 64,
        "bundle_path": str(bundle.relative_to(output)),
        "bundle_sha256": bundle_sha,
        "bundle_expected_sha256": bundle_sha,
        "usage_path": str(usage.relative_to(output)),
        "spans_path": str(spans.relative_to(output)),
        "started_at_monotonic_ns": 1_000_000_000,
        "finished_at_monotonic_ns": 3_000_000_000,
        "wall_seconds": 2.0,
        "exit_code": 0,
        "usage_audit_complete": True,
        "span_attribution_complete": True,
        "allowed_model_work_only": True,
        "fresh_setup_root": True,
        "fresh_process_cache": True,
        "model": "gpt-5.6-luna",
        "seats": "codex",
        "effort": "medium",
        "service_tier": "standard",
    }
    return output, record, usage, spans, bundle


def _record_identity(record):
    return {field: record[field]
            for field in shared_setup.EXPECTED_IDENTITY_FIELDS}


def _validate(record, output, **updates):
    arguments = {
        "expected_bundle_sha256": record["bundle_sha256"],
        "expected_identity": _record_identity(record),
        "expected_contract_identity": _CONTRACT_IDENTITY,
        "expected_compile_inputs": _COMPILE_INPUTS,
        **updates,
    }
    return shared_setup.validate_setup_record(record, output, **arguments)


def test_valid_setup_record_recomputes_private_usage_and_spans(tmp_path):
    output, record, _usage, _spans_path, _bundle = _fixture(tmp_path)

    audit = _validate(record, output)

    assert audit["complete"] is True
    assert audit["findings"] == []
    assert audit["bundle_validation"] == {
        "complete": True,
        "bundle_sha256": record["bundle_sha256"],
    }
    assert audit["metrics"] == {
        "complete": True,
        "cost_complete": True,
        "wall_complete": True,
        "started_attempts": 3,
        "uncached_input_tokens": 120,
        "cached_input_tokens": 180,
        "output_tokens": 30,
        "wall_seconds": 2.0,
    }


def test_setup_record_requires_independent_complete_operator_pins(tmp_path):
    output, record, _usage, _spans_path, _bundle = _fixture(tmp_path)

    unpinned = shared_setup.validate_setup_record(record, output)
    partial = shared_setup.validate_setup_record(
        record, output,
        expected_bundle_sha256=record["bundle_sha256"],
        expected_identity={"source_commit": record["source_commit"]},
        expected_contract_identity=_CONTRACT_IDENTITY,
        expected_compile_inputs=_COMPILE_INPUTS,
    )

    assert unpinned["complete"] is False
    assert "operator_bundle_sha_not_pinned" in " ".join(
        unpinned["findings"])
    assert "replay_validation_inputs_not_pinned" in " ".join(
        unpinned["findings"])
    assert partial["complete"] is False
    assert "operator_identity_not_fully_pinned" in " ".join(
        partial["findings"])


def test_self_pinned_arbitrary_bundle_is_not_a_valid_replay_bundle(tmp_path):
    output, record, _usage, _spans_path, bundle = _fixture(tmp_path)
    record = copy.deepcopy(record)
    bundle.write_bytes(b'{"schema":"not-a-contract-bundle"}')
    changed_sha = hashlib.sha256(bundle.read_bytes()).hexdigest()
    record["bundle_sha256"] = changed_sha
    record["bundle_expected_sha256"] = changed_sha

    audit = _validate(record, output)

    assert audit["complete"] is False
    assert audit["bundle_validation"]["complete"] is False
    assert "bundle_replay_validation_failed" in " ".join(audit["findings"])


def test_setup_record_fails_closed_on_bundle_source_wall_and_terminal_drift(
        tmp_path):
    output, record, _usage, _spans_path, bundle = _fixture(tmp_path)
    record = copy.deepcopy(record)
    bundle.write_text("changed\n", encoding="utf-8")
    (Path(record["source_root"]) / "verifier" / "sample.py").write_text(
        "VALUE = 2\n")
    record["wall_seconds"] = 9.0
    record["exit_code"] = 1

    audit = shared_setup.validate_setup_record(record, output)

    assert audit["complete"] is False
    assert audit["metrics"]["started_attempts"] is None
    findings = " ".join(audit["findings"])
    assert "bundle_content_sha_mismatch" in findings
    assert "source_fingerprint_mismatch" in findings
    assert "invalid_monotonic_wall" in findings
    assert "nonzero_exit" in findings


def test_setup_record_rejects_short_commit_and_huge_wall_without_crashing(
        tmp_path):
    output, record, _usage, _spans_path, _bundle = _fixture(tmp_path)
    record = copy.deepcopy(record)
    record["source_commit"] = "1" * 7
    record["wall_seconds"] = 10 ** 10_000

    audit = shared_setup.validate_setup_record(record, output)

    assert audit["complete"] is False
    findings = " ".join(audit["findings"])
    assert "invalid_source_commit" in findings
    assert "invalid_monotonic_wall" in findings

    record["wall_seconds"] = 2.0
    record["exit_code"] = False
    record["finished_at_monotonic_ns"] = 10 ** 400
    audit = shared_setup.validate_setup_record(record, output)
    findings = " ".join(audit["findings"])
    assert "nonzero_exit" in findings
    assert "invalid_monotonic_wall" in findings


def test_unknown_schema_shapes_fail_closed_without_type_errors(tmp_path):
    output, record, _usage, _spans_path, _bundle = _fixture(tmp_path)
    record = copy.deepcopy(record)
    record["schema"] = {"unexpected": True}
    record["phase"] = ["cold"]
    record["tasks"] = [{"task": "not-a-string"}]
    record["bundle_expected_sha256"] = [record["bundle_sha256"]]

    audit = shared_setup.validate_setup_record(record, output)

    assert audit["complete"] is False
    findings = " ".join(audit["findings"])
    assert "wrong_schema" in findings
    assert "invalid_phase" in findings
    assert "invalid_tasks" in findings
    assert "invalid_bundle_expected_sha256" in findings


def test_setup_paths_must_be_private_distinct_and_adjacent(tmp_path):
    output, record, _usage, _spans_path, _bundle = _fixture(tmp_path)
    outside = output / "outside.jsonl"
    outside.write_text("{}\n", encoding="utf-8")
    escaped = copy.deepcopy(record)
    escaped["usage_path"] = str(outside.relative_to(output))

    audit = shared_setup.validate_setup_record(escaped, output)

    assert audit["complete"] is False
    assert "wrong_private_setup_prefix" in " ".join(audit["findings"])

    reused = copy.deepcopy(record)
    reused["spans_path"] = reused["bundle_path"]
    audit = shared_setup.validate_setup_record(reused, output)
    findings = " ".join(audit["findings"])
    assert "private_paths_reused" in findings
    assert "spans_not_adjacent_to_usage" in findings


def test_setup_rejects_replay_role_missing_span_and_missing_attempt(tmp_path):
    output, record, usage, spans, _bundle = _fixture(tmp_path)
    _write_jsonl(usage, [_attempt(1, "judger")])
    _write_jsonl(spans, _spans(1, "judger"))

    wrong_role = shared_setup.validate_setup_record(record, output)
    assert wrong_role["complete"] is False
    assert "forbidden_role_stage" in " ".join(
        wrong_role["model_work"]["findings"])

    spans.unlink()
    missing_span = shared_setup.audit_model_work(
        usage, spans, scope="replay", wall_seconds=1.0)
    assert missing_span["complete"] is False
    assert "span_sidecar_missing" in " ".join(missing_span["findings"])

    _write_jsonl(spans, _spans(1, "judger"))
    usage.write_text("", encoding="utf-8")
    missing_attempt = shared_setup.audit_model_work(
        usage, spans, scope="replay", wall_seconds=1.0)
    assert missing_attempt["complete"] is False
    assert "span_call_without_usage" in " ".join(
        missing_attempt["findings"])


def test_usage_cap_is_applied_before_any_unbounded_attempt_audit(
        tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    usage.write_bytes(b"x" * 65)
    monkeypatch.setattr(
        shared_setup.analyze_model_spans, "MAX_LEDGER_BYTES", 64)

    audit = shared_setup.audit_model_work(
        usage, spans, scope="replay", wall_seconds=1.0)

    assert audit["complete"] is False
    assert "ledger_too_large" in " ".join(audit["findings"])


def test_replay_scope_and_explicit_mechanical_zero_model_work(tmp_path):
    usage = tmp_path / "usage.jsonl"
    spans = Path(str(usage) + ".spans.jsonl")
    _write_jsonl(usage, [_attempt(1, "oracles"), _attempt(2, "judger")])
    _write_jsonl(spans, [*_spans(1, "oracles"), *_spans(2, "judger")])

    replay = shared_setup.audit_model_work(
        usage, spans, scope="replay", wall_seconds=4.0,
        expected_runtime={
            "model": "gpt-5.6-luna",
            "effort": "medium",
            "service_tier": "standard",
        },
    )
    assert replay["complete"] is True
    assert replay["metrics"]["started_attempts"] == 2

    empty_usage = tmp_path / "effect-only.jsonl"
    empty_spans = Path(str(empty_usage) + ".spans.jsonl")
    mechanical = shared_setup.audit_model_work(
        empty_usage, empty_spans, scope="replay", wall_seconds=0.25,
        model_calls_expected=False,
    )
    assert mechanical["complete"] is True
    assert mechanical["metrics"]["started_attempts"] == 0
    assert mechanical["metrics"]["wall_seconds"] == 0.25
    ordinary = shared_setup.audit_model_work(
        empty_usage, empty_spans, scope="replay", wall_seconds=0.25)
    assert ordinary["complete"] is False
    assert ordinary["metrics"]["started_attempts"] is None


def _metrics(started, uncached, cached, output, wall):
    return {
        "complete": True,
        "cost_complete": True,
        "wall_complete": True,
        "started_attempts": started,
        "uncached_input_tokens": uncached,
        "cached_input_tokens": cached,
        "output_tokens": output,
        "wall_seconds": wall,
    }


def test_charge_pair_charges_common_setup_to_each_arm_but_physical_once():
    setup = _metrics(3, 120, 180, 30, 2.0)
    control = _metrics(5, 200, 300, 50, 7.0)
    treatment = _metrics(2, 80, 120, 20, 3.0)

    report = shared_setup.charge_pair(setup, control, treatment)

    assert report["complete"] is True
    assert report["charged"]["control"] == _metrics(
        8, 320, 480, 80, 9.0)
    assert report["charged"]["treatment"] == _metrics(
        5, 200, 300, 50, 5.0)
    assert report["physical"] == _metrics(10, 400, 600, 100, 12.0)
    # Physical spend contains 3 setup calls once, not once per charged arm.
    assert report["physical"]["started_attempts"] \
        != (report["charged"]["control"]["started_attempts"]
            + report["charged"]["treatment"]["started_attempts"])


def test_charge_pair_never_coerces_unknown_cost_to_zero():
    setup = _metrics(3, 120, 180, 30, 2.0)
    incomplete = {
        "complete": False,
        "cost_complete": False,
        "wall_complete": True,
        "started_attempts": None,
        "uncached_input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "wall_seconds": 4.0,
    }

    report = shared_setup.charge_pair(setup, incomplete, _metrics(
        1, 10, 20, 3, 1.0))

    assert report["complete"] is False
    assert report["charged"]["control"]["cost_complete"] is False
    assert report["charged"]["control"]["started_attempts"] is None
    assert report["charged"]["control"]["wall_seconds"] == 6.0
    assert report["physical"]["started_attempts"] is None
    assert report["physical"]["wall_seconds"] == 7.0


def test_charged_metrics_rejects_nonfinite_sum():
    left = _metrics(1, 1, 1, 1, 1e308)
    right = _metrics(1, 1, 1, 1, 1e308)

    result = shared_setup.charged_metrics(left, right)

    assert result["complete"] is False
    assert result["cost_complete"] is True
    assert result["started_attempts"] == 2
    assert result["wall_complete"] is False
    assert result["wall_seconds"] is None
