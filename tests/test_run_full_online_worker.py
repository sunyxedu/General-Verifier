"""Local-only tests for the source-pinned online evaluation worker."""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import tempfile
import pytest

from experiments.terminal_bench import run_full_online_worker as worker
from experiments.terminal_bench import project_full_offline_result as projection
from verifier.core.model import content_hash
from verifier.engine import certificate
from verifier.llm.schema import FixtureBackend
from verifier import mcp_server, offline
from verifier.core import store


PASS = {
    "verdict": "pass", "counterexample_world": "n/a",
    "missing_condition": "n/a",
}
JUDGE = {
    "sufficient": "yes", "faithful": "yes", "degradation": "n/a",
    "reason": "the fresh execution directly establishes this obligation",
}
METHOD = {
    "procedure": "read the live JSON subject and compare the required field",
    "artifact": "a retained complete read and comparison",
    "level": "exact", "basis": "world", "strength": "decisive",
    "bridges": [],
}


def _write_inputs(path: Path, *, value="alpha") -> dict:
    path.mkdir()
    (path / "task.md").write_text(
        "Inspect live-world.json and require that its value is alpha.\n",
        encoding="utf-8",
    )
    (path / "live-world.json").write_text(
        json.dumps({"value": value}) + "\n", encoding="utf-8",
    )
    return worker.input_snapshot(path)


@pytest.fixture
def repository_tmp():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="gv-online-worker-test.", dir=root) as name:
        yield Path(name)


def _plan(root: Path):
    source = root / "source"
    source.mkdir()
    (source / "verifier").mkdir()
    inputs = root / "inputs"
    snapshot = _write_inputs(inputs)
    support = {
        name: worker._sha(Path(worker.__file__).with_name(name))
        for name in worker.SUPPORT_FILES
    }
    return {
        "schema": worker.SCHEMA,
        "source_root": str(source.resolve()),
        "source_commit": "a" * 40,
        "source_fingerprint": "b" * 64,
        "input_source": str(inputs.resolve()),
        "input_seal": snapshot["seal"],
        "output_root": str((root / "output").resolve()),
        "arm": "control",
        "runtime_read_roots": [str(Path("/usr/bin").resolve())],
        "support_sha256": support,
    }


def _modules():
    return {
        "offline": offline, "mcp": mcp_server, "certificate": certificate,
        "store": store,
    }


def _fixtures():
    return {
        "obligations": {"obligations": [{
            "text": "live-world.json has value alpha",
            "required_basis": "world", "severity": "must",
            "clause": "The value must be alpha.",
            "refutable_by_absence": False,
        }]},
        "critic": copy.deepcopy(PASS),
        "oracles": {"methods": [copy.deepcopy(METHOD)]},
        "discharge": {
            "outcome": "holds", "witness": "value is alpha",
            "artifact": "complete local read", "assumed_premises": [],
            "witness_kind": "none",
            "inputs": [{"ref": "live-world.json", "provenance": "delivered",
                        "role": "subject"}],
        },
        "judger": copy.deepcopy(JUDGE),
    }


def _put_artifact(state: Path, payload: dict) -> str:
    digest = content_hash(payload)
    root = state / "artifacts"
    root.mkdir(parents=True, exist_ok=True)
    (root / f"{digest}.json").write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=1),
        encoding="utf-8",
    )
    return digest


def _adapter_method(state: Path) -> dict:
    return {
        "procedure": "execute the registered exact JSON value comparison",
        "artifact": "content-addressed typed evidence",
        "level": "exact", "basis": "world", "strength": "decisive",
        "bridges": [], "adapter": "document.json-value",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": str(state / "live-world.json"),
            "pointer": "/value", "expected": "alpha",
        }, sort_keys=True),
        "adapter_inputs": [{
            "ref": str(state / "live-world.json"),
            "provenance": "delivered", "role": "subject",
        }],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows),
                    encoding="utf-8")


def _audit_files(root: Path, monkeypatch, *, treatment: bool,
                 final_adapter: bool = True, cache=False,
                 missing_profile=False, parallel_obligations=False,
                 no_methods=False):
    output, state = root / "out", root / "state"
    output.mkdir()
    _write_inputs(state)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1" if treatment else "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1" if treatment else "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    fixtures = _fixtures()
    if parallel_obligations:
        fixtures["obligations"]["obligations"].append({
            "text": "live-world.json remains a valid JSON object",
            "required_basis": "world", "severity": "must",
            "clause": "The delivered subject must be valid JSON.",
            "refutable_by_absence": False,
        })
    if no_methods:
        fixtures["oracles"] = {"methods": []}
    if treatment and not no_methods:
        if final_adapter:
            fixtures["oracles"] = {"methods": [_adapter_method(state)]}
            fixtures.pop("discharge")
    backend = FixtureBackend(fixtures)
    backend.judger_prefetch = treatment
    result = worker._execute_live_chain(
        _modules(), state, backend, projection, "f" * 64)
    if treatment and not final_adapter and not no_methods:
        final = next(entry for entry in result["certificate"]["slice"]
                     if entry["id"] == result["certificate"]["final"])
        adapter_digest = _put_artifact(
            state, {"instrument": "adapter:document.json-value",
                    "outcome": "holds", "fixture": "rejected"})
        events = worker._read_jsonl(state / "events.jsonl")
        events.insert(-1, {
            "type": "adapter_discharge", "payload": {
                "claim": result["certificate"]["final"],
                "obligation": final["obligations"][0]["id"],
                "adapter": "document.json-value", "outcome": "holds",
                "artifact": adapter_digest,
            },
        })
        for seq, event in enumerate(events, 1):
            event["seq"] = seq
        _write_jsonl(state / "events.jsonl", events)

    logical, attempts, native_rows = [], [], []
    ordinary_identity = "e" * 64
    restricted_identity = "f" * 64 if treatment else None
    transport_profile = {
        "enabled": treatment,
        "protocol": "test-typed-packet-v1" if treatment else None,
        "boundary": "test-no-tools-v1" if treatment else None,
        "raw_prefetch_enabled": treatment,
        "raw_prefetch_protocol": "test-raw-prefetch-v1" if treatment else None,
        "ordinary_runtime_instruction_identity": ordinary_identity,
        "restricted_runtime_instruction_identity": restricted_identity,
        "eligible_adapter_identities": (
            ["document.json-value@1"] if treatment else []),
    }
    counters = {
        "input_tokens": 11, "cached_input_tokens": 3,
        "uncached_input_tokens": 8, "cache_write_input_tokens": None,
        "output_tokens": 5, "reasoning_tokens": 2,
    }
    for index, call in enumerate(backend.calls):
        role = call["role"]
        thread = f"thread-{index}"
        observer_attempt_identity = {
            "attempt_id": f"{index + 1:032x}",
            "call_id": f"{index + 101:032x}",
            "attempt_index": 1,
        }
        native_attempt_identity = {
            "attempt_id": f"{index + 1001:032x}",
            "call_id": f"{index + 1101:032x}",
            "attempt_index": 1,
        }
        timing = {
            "started_at_unix_ns": 1_000_000_000 + index * 1_000_000,
            "finished_at_unix_ns": 1_000_100_000 + index * 1_000_000,
            "duration_ns": 100_000,
        }
        restricted = treatment and final_adapter and role == "judger"
        logical.append({"record_type": "codex_logical_call_started",
                        "role": role})
        attempts.append({
            "record_type": "codex_cli_attempt_finished", "role": role,
            **worker.RUNTIME,
            "command_model": worker.RUNTIME["model"],
            "command_effort": worker.RUNTIME["effort"],
            "command_service_tier": worker.RUNTIME["service_tier"],
            "runtime_instruction_identity": (
                restricted_identity if restricted else ordinary_identity),
            "typed_judger_transport": restricted,
            "typed_judger_protocol": (
                transport_profile["protocol"] if restricted else None),
            "typed_judger_packet_sha256": "a" * 64 if restricted else None,
            "terminal_status": "returned", "usage_complete": True,
            "usage_status": "complete", "thread_id": thread,
            "cwd": str(output / "typed-judger-scratch")
            if restricted else str(state),
            **observer_attempt_identity, **timing, **counters,
        })
        raw_attached = treatment and not restricted and role == "judger"
        native_rows.append({
            "record_type": "codex_cli_attempt", "backend": "codex-cli",
            **worker.RUNTIME, "compact_inspection": treatment,
            "judger_prefetch": treatment and not restricted,
            "judger_prefetch_protocol": (
                transport_profile["raw_prefetch_protocol"]
                if treatment and not restricted else None),
            "judger_prefetch_attached": raw_attached,
            "judger_prefetch_packet_sha256": (
                "d" * 64 if raw_attached else None),
            "typed_judger_transport": restricted,
            "typed_judger_protocol": (
                transport_profile["protocol"] if restricted else None),
            "typed_judger_packet_sha256": "a" * 64 if restricted else None,
            "runtime_instruction_identity": (
                restricted_identity if restricted else ordinary_identity),
            "attempt_status": "success", "usage_complete": True,
            "usage_status": "complete", "thread_id": thread, "role": role,
            **native_attempt_identity, **timing, **counters,
        })
    _write_jsonl(output / "observer.jsonl", logical + attempts)
    if cache:
        events = worker._read_jsonl(state / "events.jsonl")
        events.insert(-1, {"type": "method_cache_hit", "payload": {}})
        for seq, event in enumerate(events, 1):
            event["seq"] = seq
        _write_jsonl(state / "events.jsonl", events)
    if missing_profile:
        native_rows[0].pop("judger_prefetch_protocol")
    _write_jsonl(output / "native-usage.jsonl", native_rows)
    return output, state, result["certificate"], transport_profile


def test_validate_plan_and_input_seal(repository_tmp):
    plan = _plan(repository_tmp)
    assert worker.validate_plan(plan) is plan
    original = (Path(plan["input_source"]) / "live-world.json").read_bytes()
    (Path(plan["input_source"]) / "live-world.json").write_bytes(
        original.replace(b"alpha", b"theta"))
    with pytest.raises(ValueError, match="seal"):
        worker.validate_plan(plan)


@pytest.mark.parametrize("arm,typed", [("control", "0"), ("treatment", "1")])
def test_environment_clears_ambient_and_changes_only_arm_flags(
        repository_tmp, monkeypatch, arm, typed):
    plan = _plan(repository_tmp)
    plan["arm"] = arm
    monkeypatch.setattr(worker.os, "environ", dict(worker.os.environ))
    monkeypatch.setenv("GV_SHARED_DIR", "/untrusted")
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "wrong")
    monkeypatch.setenv("PYTHONSTARTUP", "/untrusted/start.py")
    worker._environment(plan)
    assert worker.os.environ["GV_TYPED_ADAPTERS"] == typed
    assert worker.os.environ["GV_TYPED_JUDGER_PACKET"] == typed
    assert worker.os.environ["GV_CONTRACT_CACHE"] == "0"
    assert worker.os.environ["GV_APPEND_ONLY_DELTAS"] == "0"
    assert "GV_SHARED_DIR" not in worker.os.environ
    assert "PYTHONSTARTUP" not in worker.os.environ


def test_input_snapshot_seals_arbitrary_nested_and_binary_inputs(tmp_path):
    inputs = tmp_path / "inputs"
    _write_inputs(inputs)
    (inputs / "documents").mkdir()
    (inputs / "documents" / "brief.pdf").write_bytes(b"%PDF\x00fixture")
    (inputs / "table.csv").write_text("a,b\n1,2\n")
    snapshot = worker.input_snapshot(inputs)
    assert set(snapshot["files"]) == {
        "task.md", "live-world.json", "documents/brief.pdf", "table.csv"}
    assert snapshot["directory_count"] == 1
    (inputs / "table.csv").write_text("a,b\n1,3\n")
    assert worker.input_snapshot(inputs)["seal"] != snapshot["seal"]


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_input_snapshot_rejects_aliases_and_special_files(kind, tmp_path):
    inputs = tmp_path / "inputs"
    _write_inputs(inputs)
    target = inputs / "unsafe"
    if kind == "symlink":
        target.symlink_to(inputs / "task.md")
    elif kind == "hardlink":
        worker.os.link(inputs / "task.md", target)
    else:
        worker.os.mkfifo(target)
    with pytest.raises(ValueError, match="symlink|regular|private"):
        worker.input_snapshot(inputs)


def test_clone_preserves_complete_tree_and_production_ingests_hook_feed(
        tmp_path, monkeypatch):
    source = tmp_path / "source-input"
    _write_inputs(source)
    (source / "documents").mkdir()
    (source / "documents" / "subject.bin").write_bytes(b"\x00\xffsubject")
    hook = ({"type": "tool_call",
             "payload": {"tool": "fixture", "input": {"x": 1}}})
    (source / "hook_feed.jsonl").write_text(json.dumps(hook) + "\n")
    before = worker.input_snapshot(source)
    state = tmp_path / "state"
    assert worker._clone_inputs(source, state, before["seal"])["seal"] \
        == before["seal"]
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    result = worker._execute_live_chain(
        _modules(), state, FixtureBackend(_fixtures()), projection, "f" * 64)
    events = worker._read_jsonl(state / "events.jsonl")
    assert events[0]["type"] == "tool_call"
    assert worker.input_snapshot(state, runtime_state=True)["seal"] \
        == before["seal"]
    assert result["quality_view"]["semantic_evidence"]["document"][
        "input_tree_seal"] == before["seal"]


def test_live_chain_uses_service_commit_and_all_five_stages(tmp_path,
                                                            monkeypatch):
    state_path = tmp_path / "state"
    _write_inputs(state_path)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    backend = FixtureBackend(_fixtures())

    result = worker._execute_live_chain(
        _modules(), state_path, backend, projection, "f" * 64)

    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "oracles", "discharge", "judger"]
    assert [call["variables"]["world_mode"] for call in backend.calls] == [
        "authority-only", "authority-only", "live", "live", "live"]
    assert result["commit"]["status"] == "verified"
    assert result["quality_view"]["ordered_contract"]
    assert result["quality_view"]["ordered_outcomes"][0]["status"] == "verified"
    assert result["quality_view"]["score"]["rank_key"] == [4, 0]
    assert result["quality_view"]["semantic_evidence"]["document"][
        "obligations"][0]["evidence"][0]["mechanism"]["kind"] == "agent"


def test_live_chain_retains_raw_certificate_and_artifacts(tmp_path,
                                                          monkeypatch):
    state_path = tmp_path / "state"
    _write_inputs(state_path)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    backend = FixtureBackend(_fixtures())
    result = worker._execute_live_chain(
        _modules(), state_path, backend, projection, "f" * 64)
    digest = result["certificate"]["certificate_id"]
    assert (state_path / "artifacts" / f"{digest}.json").is_file()
    assert worker._artifact_inventory(state_path)
    assert "evidence" not in result["quality_view"]["certificate"]["slice"][0]
    assert result["certificate"]["slice"][0]["obligations"][0]["evidence"]


def test_live_chain_treatment_keeps_oracle_and_fresh_judger_around_adapter(
        tmp_path, monkeypatch):
    state_path = tmp_path / "state"
    _write_inputs(state_path)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    method = {
        "procedure": "execute the registered exact JSON value comparison",
        "artifact": "content-addressed typed evidence",
        "level": "exact", "basis": "world", "strength": "decisive",
        "bridges": [], "adapter": "document.json-value",
        "adapter_version": 1,
        "adapter_arguments": json.dumps({
            "address": str(state_path / "live-world.json"),
            "pointer": "/value", "expected": "alpha",
        }, sort_keys=True),
        "adapter_inputs": [{
            "ref": str(state_path / "live-world.json"),
            "provenance": "delivered", "role": "subject",
        }],
    }
    fixtures = _fixtures()
    fixtures["oracles"] = {"methods": [method]}
    fixtures.pop("discharge")
    backend = FixtureBackend(fixtures)
    backend.judger_prefetch = True

    result = worker._execute_live_chain(
        _modules(), state_path, backend, projection, "f" * 64)

    assert [call["role"] for call in backend.calls] == [
        "obligations", "critic", "oracles", "judger"]
    assert any(event["type"] == "adapter_discharge"
               for event in mcp_server.EventLog(
                   str(state_path / "events.jsonl")).replay())
    assert worker._bound_adapter_digests(
        state_path, result["certificate"])
    assert result["quality_view"]["semantic_evidence"]["document"][
        "obligations"][0]["evidence"][0]["mechanism"]["kind"] \
        == "typed_adapter"


def test_artifact_inventory_rejects_content_address_mismatch(tmp_path):
    root = tmp_path / "state" / "artifacts"
    root.mkdir(parents=True)
    (root / (("a" * 64) + ".json")).write_text('{"payload":"wrong"}\n')
    with pytest.raises(ValueError, match="canonical content"):
        worker._artifact_inventory(tmp_path / "state")


def test_control_execution_audit_requires_agent_and_no_cache(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    report = worker._audit_execution(
        output, state_path, "control", cert, protocol)
    assert report["typed_adapter_exercised"] is False
    assert report["typed_judger_transport"]["enabled"] is False
    assert report["typed_judger_transport"]["packet_attached_attempts"] == 0
    assert report["cache_events"] == {
        "method_cache_hit": 0, "method_cache_direction_audit": 0,
        "contract_cache_hit": 0, "contract_cache_published": 0,
    }


def test_treatment_execution_audit_requires_bound_adapter_and_prefetch(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=True)
    report = worker._audit_execution(
        output, state_path, "treatment", cert, protocol)
    assert report["typed_adapter_exercised"] is True
    assert report["typed_judger_transport"]["packet_eligible_bound_methods"] == 1
    assert report["typed_judger_transport"]["restricted_attempted"] is True
    assert len(report["bound_adapter_artifacts"]) == 1


@pytest.mark.parametrize("problem", ["cache", "missing_profile"])
def test_treatment_audit_fails_closed(problem, tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=True, cache=problem == "cache",
        missing_profile=problem == "missing_profile")
    with pytest.raises(ValueError):
        worker._audit_execution(output, state_path, "treatment", cert, protocol)


def test_rejected_adapter_exercise_is_not_mislabeled_as_final_binding(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=True, final_adapter=False)
    report = worker._audit_execution(
        output, state_path, "treatment", cert, protocol)
    assert report["typed_adapter_exercised"] is True
    assert report["typed_adapter_bound"] is False


@pytest.mark.parametrize("bad_role", ["extract", "unregistered-role"])
def test_execution_audit_rejects_extractor_and_unknown_roles(
        bad_role, tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    rows = worker._read_jsonl(output / "observer.jsonl")
    rows[0]["role"] = bad_role
    _write_jsonl(output / "observer.jsonl", rows)
    with pytest.raises(ValueError, match="Extractor or unknown"):
        worker._audit_execution(output, state_path, "control", cert, protocol)


@pytest.mark.parametrize("problem", ["thread", "token", "role", "typed"])
def test_execution_audit_requires_exact_native_observer_bijection(
        problem, tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    rows = worker._read_jsonl(output / "native-usage.jsonl")
    if problem == "thread":
        rows[0]["thread_id"] = "not-an-observed-thread"
    elif problem == "token":
        rows[0]["input_tokens"] += 1
    elif problem == "role":
        rows[0]["role"] = "judger"
    else:
        rows[0]["typed_judger_packet_sha256"] = "a" * 64
    _write_jsonl(output / "native-usage.jsonl", rows)
    with pytest.raises(ValueError, match="disagree|restricted telemetry"):
        worker._audit_execution(output, state_path, "control", cert, protocol)


def test_execution_audit_pairs_threadless_timeout_across_independent_ledgers(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    observer = worker._read_jsonl(output / "observer.jsonl")
    observed = next(row for row in reversed(observer)
                    if row.get("record_type") == "codex_cli_attempt_finished")
    native = worker._read_jsonl(output / "native-usage.jsonl")
    native_attempt = native[-1]
    observed.update({
        "thread_id": None, "terminal_status": "timeout",
        "usage_complete": False, "usage_status": "partial",
    })
    native_attempt.update({
        "attempt_status": "timeout", "usage_complete": False,
        "usage_status": "partial",
    })
    for key in worker._ATTEMPT_COUNTER_FIELDS:
        observed[key] = None
        native_attempt[key] = None
    _write_jsonl(output / "observer.jsonl", observer)
    _write_jsonl(output / "native-usage.jsonl", native)

    report = worker._audit_execution(
        output, state_path, "control", cert, protocol)

    assert report["native_attempts"] == report["observer_finished_attempts"]


def test_execution_audit_rejects_unbound_threadless_timeout(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    observer = worker._read_jsonl(output / "observer.jsonl")
    observed = next(row for row in reversed(observer)
                    if row.get("record_type") == "codex_cli_attempt_finished")
    native = worker._read_jsonl(output / "native-usage.jsonl")
    native_attempt = native[-1]
    observed.update({
        "thread_id": None, "terminal_status": "timeout",
        "usage_complete": False, "usage_status": "partial",
        "finished_at_unix_ns": native_attempt["finished_at_unix_ns"]
        + 50_000_001,
    })
    native_attempt.update({
        "attempt_status": "timeout", "usage_complete": False,
        "usage_status": "partial",
    })
    for key in worker._ATTEMPT_COUNTER_FIELDS:
        observed[key] = None
        native_attempt[key] = None
    _write_jsonl(output / "observer.jsonl", observer)
    _write_jsonl(output / "native-usage.jsonl", native)

    with pytest.raises(ValueError, match="thread identities disagree"):
        worker._audit_execution(output, state_path, "control", cert, protocol)


def test_execution_audit_rejects_observer_selected_restricted_cwd(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=True)
    rows = worker._read_jsonl(output / "observer.jsonl")
    restricted = next(row for row in rows
                      if row.get("typed_judger_transport") is True)
    restricted["cwd"] = "/private/tmp/gv-typed-judger-a1_b2c3d"
    _write_jsonl(output / "observer.jsonl", rows)

    with pytest.raises(ValueError, match="execution cwd disagree"):
        worker._audit_execution(
            output, state_path, "treatment", cert, protocol)


def test_event_audit_accepts_retry_noise_and_parallel_obligations(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False,
        parallel_obligations=True)
    events = worker._read_jsonl(state_path / "events.jsonl")
    final = next(entry for entry in cert["slice"] if entry["id"] == cert["final"])
    oid = final["obligations"][0]["id"]
    enumeration = next(event for event in events
                       if event["type"] == "methods_enumerated"
                       and event["payload"]["obligation"] == oid)
    insert_at = next(index for index, event in enumerate(events)
                     if event["type"] == "judger_verdict"
                     and event["payload"]["obligation"] == oid)
    retry = [
        {"type": "method_redo", "payload": {
            "claim": cert["final"], "obligation": oid,
            "procedure": enumeration["payload"]["methods"][0]["procedure"],
            "round": 1, "reason": "retry fixture",
        }},
        {"type": "judger_verdict", "payload": {
            "obligation": oid, "redo": 1,
            "procedure": enumeration["payload"]["methods"][0]["procedure"],
            "sufficient": "no", "faithful": "yes", "degradation": "n/a",
            "reason": "retry fixture",
        }},
    ]
    events[insert_at:insert_at] = retry
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    _write_jsonl(state_path / "events.jsonl", events)
    report = worker._audit_execution(
        output, state_path, "control", cert, protocol)
    assert len(report["final_obligation_lineage"]) == 2


def test_event_audit_rejects_missing_bound_stage(tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    events = [event for event in worker._read_jsonl(state_path / "events.jsonl")
              if event["type"] != "method_bound"]
    for seq, event in enumerate(events, 1):
        event["seq"] = seq
    _write_jsonl(state_path / "events.jsonl", events)
    with pytest.raises(ValueError, match="complete method lineage"):
        worker._audit_execution(output, state_path, "control", cert, protocol)


def test_event_audit_requires_adjacent_payload_bound_critic_pass(
        tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    events = worker._read_jsonl(state_path / "events.jsonl")
    critic = next(event for event in events
                  if event["type"] == "decomposition_attacked")
    critic["payload"]["claim"] = "0" * 64
    _write_jsonl(state_path / "events.jsonl", events)
    with pytest.raises(ValueError, match="Critic pass"):
        worker._audit_execution(output, state_path, "control", cert, protocol)


@pytest.mark.parametrize("treatment", [False, True])
def test_unverified_no_method_is_a_complete_fail_closed_result(
        treatment, tmp_path, monkeypatch):
    output, state_path, cert, protocol = _audit_files(
        tmp_path, monkeypatch, treatment=treatment, final_adapter=False,
        no_methods=True)
    report = worker._audit_execution(
        output, state_path, "treatment" if treatment else "control",
        cert, protocol)
    assert cert["verdict"] == "unverified"
    assert "judger" not in report["roles_observed"]
    lineage = report["final_obligation_lineage"][0]
    assert lineage["status"] == "unverified"
    assert "skipped fail-closed" in lineage["stages_skipped_reason"]
    assert report["typed_adapter_exercised"] is False
    assert report["typed_judger_transport"]["restricted_attempted"] is False
    semantic = worker._semantic_evidence_projection(state_path, cert)
    assert semantic["document"]["truth_eligible"] is False


@pytest.mark.parametrize(
    ("verdict", "eligible"),
    [(None, False), ("future-verdict", False), ("partial", False),
     ("unverified", False), ("verified", True), ("supported", True),
     ("refuted", True)],
)
def test_semantic_truth_eligibility_is_an_explicit_allowlist(
        verdict, eligible, tmp_path, monkeypatch):
    _output, state_path, cert, _protocol = _audit_files(
        tmp_path, monkeypatch, treatment=False, final_adapter=False)
    cert["verdict"] = verdict
    semantic = worker._semantic_evidence_projection(state_path, cert)
    assert semantic["document"]["truth_eligible"] is eligible


def test_semantic_evidence_is_content_addressed_and_path_independent(
        tmp_path, monkeypatch):
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "1")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    views = []
    for name in ("arm-a", "arm-b"):
        state = tmp_path / name
        _write_inputs(state)
        fixtures = _fixtures()
        fixtures["oracles"] = {"methods": [_adapter_method(state)]}
        fixtures.pop("discharge")
        result = worker._execute_live_chain(
            _modules(), state, FixtureBackend(fixtures), projection, "f" * 64)
        view = result["quality_view"]["semantic_evidence"]
        assert view["content_sha256"] == worker._content_digest(view["document"])
        assert str(tmp_path) not in json.dumps(view, sort_keys=True)
        views.append(view)
    assert views[0] == views[1]


def test_semantic_projection_preserves_nested_business_identity_keys(tmp_path):
    state = tmp_path / "state"
    inputs = _write_inputs(state)
    business = {
        "reason": "business reason", "digest": "business digest",
        "ts": "business timestamp", "seq": 17,
        "path": "/business/value/is/not/an/input-address",
        "nested": {"reason": "nested", "path": "$.payload.reason"},
    }
    arguments = {
        "scope": "prefix", "payload_equals": {"/payload": business},
    }
    assert worker._semantic_adapter_arguments(
        arguments, state, inputs)["payload_equals"]["/payload"] == business

    observation = {
        "payload_equals": {"/payload": business},
        "snapshot": {
            "ref": "ledger:events", "prefix_events_digest": "a" * 64,
        },
    }
    projected = worker._semantic_observation(observation, state, inputs)
    assert projected["payload_equals"]["/payload"] == business
    assert projected["snapshot"]["logical_ref"] == "ledger:events"


def test_semantic_input_refs_reject_external_same_basename_collisions(tmp_path):
    state = tmp_path / "state"
    inputs = _write_inputs(state)
    left = (tmp_path / "outside-a" / "shared.json").resolve()
    right = (tmp_path / "outside-b" / "shared.json").resolve()
    for address in (left, right):
        with pytest.raises(ValueError, match="bound to the sealed tree"):
            worker._logical_ref(str(address), state, inputs)


def test_agent_snapshot_uses_canonical_json_content_digest():
    left = {"proof.json": {
        "reason": "retained", "nested": {"b": 2, "a": 1},
    }}
    right = {"proof.json": {
        "nested": {"a": 1, "b": 2}, "reason": "retained",
    }}
    projected_left = worker._semantic_agent_snapshot(left)
    projected_right = worker._semantic_agent_snapshot(right)
    assert projected_left == projected_right
    assert projected_left[0]["content_sha256"] == worker._content_digest(
        left["proof.json"])


def test_projection_identity_and_result_views_fail_closed(tmp_path, monkeypatch):
    state_path = tmp_path / "state"
    _write_inputs(state_path)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "0")
    monkeypatch.setenv("GV_TYPED_JUDGER_PACKET", "0")
    monkeypatch.setenv("GV_CONTRACT_CACHE", "0")
    monkeypatch.setenv("GV_APPEND_ONLY_DELTAS", "0")
    result = worker._execute_live_chain(
        _modules(), state_path, FixtureBackend(_fixtures()), projection,
        "f" * 64)
    assert worker._verify_projection_identity(state_path, result)

    commit, cert = result["commit"], result["certificate"]
    features = offline._features(cert, 0, "f" * 64)
    rank = offline.rank_key(features)
    quality = projection.project_full_offline_result(
        str(state_path / "events.jsonl"), cert)
    bad_commit = {**commit, "status": "refuted"}
    with pytest.raises(ValueError, match="commit and final"):
        worker._validate_result_views(
            bad_commit, cert, features, rank, quality, "f" * 64,
            offline.FEATURE_SCHEMA_VERSION)
    bad_features = {**features, "verdict": "refuted"}
    with pytest.raises(ValueError, match="features disagree"):
        worker._validate_result_views(
            commit, cert, bad_features, rank, quality, "f" * 64,
            offline.FEATURE_SCHEMA_VERSION)
    bad_quality = copy.deepcopy(quality)
    bad_quality["ordered_outcomes"][0]["status"] = "refuted"
    with pytest.raises(ValueError, match="projection disagrees"):
        worker._validate_result_views(
            commit, cert, features, rank, bad_quality, "f" * 64,
            offline.FEATURE_SCHEMA_VERSION)

    events = worker._read_jsonl(state_path / "events.jsonl")
    events.append({"seq": len(events) + 1, "type": "postflight-mutation",
                   "payload": {}})
    _write_jsonl(state_path / "events.jsonl", events)
    with pytest.raises(ValueError, match="changed after projection"):
        worker._verify_projection_identity(state_path, result)


def test_semantic_certificate_view_rejects_missing_fields_and_excludes_evidence():
    malformed = {"final": "x", "verdict": "verified", "slice": [{"id": "x"}]}
    with pytest.raises(ValueError, match="incomplete"):
        worker._certificate_semantics(malformed)

    entry = {key: None for key in worker._ENTRY_FIELDS}
    entry.update({"id": "x", "open_premises": [], "obligations": []})
    view = worker._certificate_semantics({
        "final": "x", "verdict": "unverified", "verdict_strength": None,
        "slice": [{**entry, "evidence": [{"secret": "retained only raw"}]}],
    })
    assert "evidence" not in view["slice"][0]


def test_plan_mode_never_runs_worker(repository_tmp, monkeypatch, capsys):
    plan = _plan(repository_tmp)
    plan_path = repository_tmp / "plan.json"
    raw = json.dumps(plan).encode()
    plan_path.write_bytes(raw)
    monkeypatch.setattr(worker, "run_worker",
                        lambda _plan: pytest.fail("plan mode ran paid work"))
    assert worker.main([
        "--plan", str(plan_path), "--plan-sha256",
        hashlib.sha256(raw).hexdigest(),
    ]) == 0
    assert json.loads(capsys.readouterr().out)["model_calls"] == 0
