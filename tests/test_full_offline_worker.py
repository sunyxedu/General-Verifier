"""No-model admission and lifecycle checks for the full offline worker."""
from contextlib import contextmanager
from copy import deepcopy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench import run_full_offline_worker as worker


def _plan(tmp_path, monkeypatch, arm="treatment"):
    source, inputs = tmp_path / "source", tmp_path / "inputs"
    source.mkdir()
    inputs.mkdir()
    monkeypatch.setattr(worker, "_sha", lambda _: "a" * 64)
    return {
        "schema": worker.SCHEMA, "source_root": str(source.resolve()),
        "input_source": str(inputs.resolve()),
        "output_root": "/Users/fixture-gv-eval/new-" + tmp_path.name,
        "source_commit": worker.BASELINE if arm == "control" else "c" * 40,
        "source_fingerprint": "b" * 64, "input_seal": "d" * 64,
        "arm": arm, "judger_prefetch": arm == "treatment",
        "runtime_read_roots": [],
        "support_sha256": {key: "a" * 64 for key in worker.SUPPORT_FILES},
    }


@pytest.mark.parametrize("arm", ["control", "treatment"])
def test_plan_is_read_only_and_requires_pinned_support(tmp_path, monkeypatch, arm):
    plan = _plan(tmp_path, monkeypatch, arm)
    assert worker.validate_plan(plan) == plan
    assert not Path(plan["output_root"]).exists()
    del plan["support_sha256"]["run_full_offline_worker.py"]
    with pytest.raises(ValueError, match="every common support"):
        worker.validate_plan(plan)


@pytest.mark.parametrize("field,bad", [
    ("judger_prefetch", False), ("judger_prefetch", 1), ("arm", "other"),
    ("input_seal", "not-a-hash"), ("source_commit", "main"),
    ("runtime_read_roots", None),
])
def test_invalid_or_non_fresh_profile_fails(tmp_path, monkeypatch, field, bad):
    plan = _plan(tmp_path, monkeypatch)
    plan[field] = bad
    with pytest.raises(ValueError):
        worker.validate_plan(plan)


def test_baseline_cannot_silently_become_a_current_version_ablation(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch, "control")
    plan["source_commit"] = "c" * 40
    with pytest.raises(ValueError, match="original-main"):
        worker.validate_plan(plan)


def test_no_resume_or_output_inside_source(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    plan["output_root"] = plan["input_source"]
    with pytest.raises(ValueError, match="new"):
        worker.validate_plan(plan)
    plan["output_root"] = plan["source_root"] + "/child"
    with pytest.raises(ValueError, match="disjoint"):
        worker.validate_plan(plan)


def test_paid_output_cannot_be_inside_host_temp(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    plan["output_root"] = "/private/tmp/gv-worker-must-not-run"
    with pytest.raises(ValueError, match="host temporary"):
        worker.validate_plan(plan)


def test_existing_repo_is_not_a_frozen_source(tmp_path, monkeypatch):
    plan = _plan(tmp_path, monkeypatch)
    (Path(plan["source_root"]) / ".git").write_text("gitdir: elsewhere")
    with pytest.raises(ValueError, match="frozen source"):
        worker.validate_plan(plan)


def test_runtime_environment_keeps_auth_but_clears_ambient_evaluation_flags(monkeypatch):
    env = {"HOME": "/auth-home", "GV_SHARED_DIR": "/unrelated",
           "GV_CONTRACT_CACHE": "1", "PYTHONPATH": "/wrong-source",
           "UNCHANGED": "ordinary"}
    monkeypatch.setattr(worker.os, "environ", env)
    monkeypatch.setattr(worker.sys, "dont_write_bytecode", False)
    worker._environment({"arm": "treatment", "runtime_read_roots": []})
    assert env["HOME"] == "/auth-home"
    assert env["UNCHANGED"] == "ordinary"
    assert env["GV_CONTRACT_CACHE"] == "0"
    assert env["GV_TYPED_ADAPTERS"] == "1"
    assert env["GV_TYPED_JUDGER_PACKET"] == "1"
    assert env["GV_CERTIFICATE_SCOPE"] == "final"
    assert "GV_SHARED_DIR" not in env and "PYTHONPATH" not in env
    worker._environment({"arm": "control", "runtime_read_roots": []})
    assert env["GV_TYPED_ADAPTERS"] == "0"
    assert env["GV_TYPED_JUDGER_PACKET"] == "0"
    assert "GV_CERTIFICATE_SCOPE" not in env


def test_certificate_is_read_from_content_address_not_just_feature_label(tmp_path):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    cert = {"final": "f", "verdict": "verified", "verdict_strength": "decisive",
            "slice": [{"id": "f", "obligations": [{"id": "o"}]}]}
    raw = json.dumps(cert, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    digest = hashlib.sha256(raw.encode()).hexdigest()
    path = artifact / (digest + ".json")
    path.write_text(raw)
    features = {"unscoreable": False, "certificate_id": digest,
                "verdict": "verified", "verdict_strength": "decisive"}
    assert worker._certificate(tmp_path, features)["certificate"] == cert
    path.write_text(raw.replace("decisive", "bounded"))
    with pytest.raises(ValueError, match="content address"):
        worker._certificate(tmp_path, features)


@pytest.mark.parametrize("arm", ["control", "treatment"])
@pytest.mark.parametrize("failure", [False, True])
def test_full_score_called_once_and_failure_keeps_accounting(
        tmp_path, monkeypatch, arm, failure):
    plan = _plan(tmp_path, monkeypatch, arm)
    plan["output_root"] = str(tmp_path / "output")
    calls = []
    monkeypatch.setattr(worker, "validate_plan", lambda p: p)
    monkeypatch.setattr(worker, "_environment", lambda p: None)
    monkeypatch.setattr(worker, "_clone", lambda offline, source, dest, expected: dest.mkdir())
    monkeypatch.setattr(worker, "_certificate", lambda *args: {"fixture": True})
    monkeypatch.setattr(worker, "_audit_role_and_cache_records", lambda *args: {"fixture": True})

    def score(state, **kwargs):
        calls.append((state, kwargs))
        if failure:
            raise ValueError("synthetic scoring failure")
        return {"unscoreable": False, "verdict": "verified"}

    offline = SimpleNamespace(score=score, verify_immutable_input_anchor=lambda _: plan["input_seal"])
    summary = {"closed": True, "fixture": True}

    @contextmanager
    def observer(*args, **kwargs):
        yield SimpleNamespace(summary=lambda: deepcopy(summary))

    @contextmanager
    def policy(*args, **kwargs):
        yield SimpleNamespace(identity={"schema": "fixture-policy"})

    @contextmanager
    def native(*args, **kwargs):
        yield

    backend = SimpleNamespace(codex_usage_ledger=native)
    if arm == "treatment":
        backend._run_codex_process = lambda: None
    monkeypatch.setattr(worker, "_load_runtime", lambda _: (offline, backend))
    support = {
        "run_typed_adapter_eval": SimpleNamespace(_source_fingerprint=lambda _: plan["source_fingerprint"]),
        "observe_codex_attempts": SimpleNamespace(observe_codex_attempts=observer),
        "analyze_codex_observer": SimpleNamespace(analyze_file=lambda *a, **k: {
            "accounting_complete": True, "fixture_charged_attempts": 9}),
        "eval_offline_policy": SimpleNamespace(eval_offline_policy=policy,
                                               OfflinePolicyBinding=lambda **k: k),
    }
    monkeypatch.setattr(worker, "_support", lambda name: support[name])
    if failure:
        with pytest.raises(ValueError, match="synthetic scoring failure"):
            worker.run_worker(plan)
    else:
        assert worker.run_worker(plan)["complete"] is True
    assert len(calls) == 1
    kwargs = calls[0][1]
    assert kwargs["seats"] == "codex" and kwargs["model"] == "gpt-5.6-luna"
    assert kwargs["shared_dir"] is None
    assert kwargs["parallel_claims"] == kwargs["parallel_obligations"] == 4
    assert kwargs["max_rounds"] == 3
    assert ("judger_prefetch" in kwargs) is (arm == "treatment")
    terminal = json.loads((Path(plan["output_root"]) / "worker-result.json").read_text())
    assert terminal["complete"] is not failure
    assert terminal["accounting"]["fixture_charged_attempts"] == 9
    assert terminal["quality_acceptance"] is None
    assert terminal["original_project_10x_acceptance"] is False


def _audit_fixture(tmp_path, *, attached=True):
    roles = ["extract", "obligations", "critic", "oracles", "judger"]
    observer = [{"record_type": "codex_logical_call_started", "role": role}
                for role in roles]
    observer.append({"record_type": "codex_cli_attempt_finished", "thread_id": "thread-1"})
    native = [{"record_type": "codex_cli_attempt", "thread_id": "thread-1",
               "role": "judger", "judger_prefetch": True,
               "compact_inspection": False, "judger_prefetch_protocol": "fixture-v1",
               "judger_prefetch_attached": attached,
               "judger_prefetch_packet_sha256": "a" * 64 if attached else None}]
    (tmp_path / "state").mkdir()
    _jsonl(tmp_path / "observer.jsonl", observer)
    _jsonl(tmp_path / "native-usage.jsonl", native)
    _jsonl(tmp_path / "state/events.jsonl", [{"type": "obligations_compiled"}])
    return observer, native


def _jsonl(path, rows):
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


@pytest.mark.parametrize("attached", [False, True])
def test_actual_prefetch_use_is_distinct_from_enabled_profile(tmp_path, attached):
    _audit_fixture(tmp_path, attached=attached)
    audit = worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")
    assert audit["prefetch_exercised"] is attached
    assert audit["prefetch_attached_attempts"] == int(attached)
    assert audit["typed_judger_exercised"] is False


def _typed_row(**changes):
    row = {"record_type": "codex_cli_attempt", "thread_id": "thread-2",
           "role": "judger", "typed_judger_transport": True,
           "typed_judger_protocol": "typed-v1",
           "typed_judger_packet_sha256": "b" * 64,
           "judger_prefetch": False, "compact_inspection": False,
           "judger_prefetch_protocol": None,
           "judger_prefetch_attached": False,
           "judger_prefetch_packet_sha256": None}
    row.update(changes)
    return row


def test_typed_judger_rows_count_as_the_exercised_profile(tmp_path):
    observer, native = _audit_fixture(tmp_path, attached=False)
    observer.append({"record_type": "codex_cli_attempt_finished",
                     "thread_id": "thread-2"})
    native.append(_typed_row())
    _jsonl(tmp_path / "observer.jsonl", observer)
    _jsonl(tmp_path / "native-usage.jsonl", native)
    audit = worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")
    assert audit["prefetch_exercised"] is False
    assert audit["typed_judger_exercised"] is True
    assert audit["typed_judger_attempts"] == 1


@pytest.mark.parametrize("field,value", [
    ("role", "discharge"), ("typed_judger_protocol", ""),
    ("typed_judger_packet_sha256", "bad"), ("judger_prefetch", True),
    ("judger_prefetch_attached", True), ("compact_inspection", True),
    ("judger_prefetch_packet_sha256", "a" * 64),
])
def test_typed_judger_rows_must_be_internally_consistent(tmp_path, field, value):
    observer, native = _audit_fixture(tmp_path, attached=False)
    observer.append({"record_type": "codex_cli_attempt_finished",
                     "thread_id": "thread-2"})
    native.append(_typed_row(**{field: value}))
    _jsonl(tmp_path / "observer.jsonl", observer)
    _jsonl(tmp_path / "native-usage.jsonl", native)
    with pytest.raises(ValueError, match="typed Judger profile"):
        worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")


@pytest.mark.parametrize("field,value", [
    ("judger_prefetch", False), ("judger_prefetch", 1),
    ("compact_inspection", True), ("judger_prefetch_protocol", "other"),
    ("judger_prefetch_attached", 1), ("judger_prefetch_packet_sha256", "bad"),
    ("role", "oracles"), ("thread_id", "unknown"), ("thread_id", []),
    ("record_type", "other"),
])
def test_native_attempt_profile_drift_is_rejected(tmp_path, field, value):
    _, native = _audit_fixture(tmp_path)
    native[0][field] = value
    _jsonl(tmp_path / "native-usage.jsonl", native)
    with pytest.raises(ValueError):
        worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")


@pytest.mark.parametrize("missing", ["judger_prefetch_packet_sha256", "judger_prefetch_protocol"])
def test_missing_native_profile_field_is_not_assumed(tmp_path, missing):
    _, native = _audit_fixture(tmp_path)
    del native[0][missing]
    _jsonl(tmp_path / "native-usage.jsonl", native)
    with pytest.raises(ValueError, match="profile mismatch"):
        worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")


def test_unattached_packet_cannot_claim_digest(tmp_path):
    _, native = _audit_fixture(tmp_path, attached=False)
    native[0]["judger_prefetch_packet_sha256"] = "a" * 64
    _jsonl(tmp_path / "native-usage.jsonl", native)
    with pytest.raises(ValueError, match="unattached"):
        worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")


@pytest.mark.parametrize("protocol", [None, "", 1])
def test_treatment_requires_selected_source_protocol(tmp_path, protocol):
    _audit_fixture(tmp_path)
    with pytest.raises(ValueError, match="protocol was not pinned"):
        worker._audit_role_and_cache_records(tmp_path, "treatment", protocol)


@pytest.mark.parametrize("change", ["missing", "duplicate", "invalid"])
def test_attempt_inventories_must_match(tmp_path, change):
    observer, native = _audit_fixture(tmp_path)
    if change == "missing":
        observer.pop()
    elif change == "duplicate":
        native.append(dict(native[0]))
    else:
        observer[-1]["thread_id"] = []
    _jsonl(tmp_path / "observer.jsonl", observer)
    _jsonl(tmp_path / "native-usage.jsonl", native)
    with pytest.raises(ValueError, match="ledger disagree"):
        worker._audit_role_and_cache_records(tmp_path, "treatment", "fixture-v1")


@pytest.mark.parametrize("role", ["extract", "obligations", "critic", "oracles", "judger"])
def test_full_stage_coverage_cannot_skip_a_role(tmp_path, role):
    observer, _ = _audit_fixture(tmp_path)
    _jsonl(tmp_path / "observer.jsonl", [row for row in observer if row.get("role") != role])
    with pytest.raises(ValueError, match="role coverage"):
        worker._audit_role_and_cache_records(tmp_path, "control")


@pytest.mark.parametrize("event", ["method_cache_hit", "method_cache_direction_audit", "contract_cache_hit"])
def test_fresh_review_rejects_reuse_events(tmp_path, event):
    _audit_fixture(tmp_path)
    _jsonl(tmp_path / "state/events.jsonl", [{"type": event}])
    with pytest.raises(ValueError, match="unexpectedly reused"):
        worker._audit_role_and_cache_records(tmp_path, "control")


@pytest.mark.parametrize("raw", [b'{}', b'[]\n', b'{"x":1,"x":2}\n', b'{"x":NaN}\n', b'\xff\n'])
def test_malformed_audit_records_fail_closed(tmp_path, raw):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(raw)
    with pytest.raises(ValueError):
        worker._read_jsonl(path)


def test_audit_record_size_is_bounded(tmp_path):
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b'{}\n')
    with pytest.raises(ValueError, match="oversized"):
        worker._read_jsonl(path, cap=2)
