"""Causal model-span telemetry; every test is local and model-free."""
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.llm.agents import AgentRole, JUDGER_AGENT
from verifier.llm.codex_backend import CodexBackend, codex_usage_ledger
from verifier.core.events import EventLog
from verifier.core.model import Obligation
from verifier.engine.game import Attempt, BindingGame
from verifier.execution.discharge import DischargeResult
from verifier.execution.instruments import ArtifactStore
from verifier.llm.model_spans import (
    model_scope,
    span_sidecar_path,
    submit_with_context,
)
from verifier.llm.schema import RoleError


def _rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _usage_stream(thread_id="thread-1"):
    events = [
        {"type": "thread.started", "thread_id": thread_id},
        {"type": "item.completed", "item": {
            "type": "agent_message", "text": '{"ok":true}',
        }},
        {"type": "turn.completed", "usage": {
            "input_tokens": 10,
            "cached_input_tokens": 4,
            "output_tokens": 2,
            "reasoning_output_tokens": 1,
        }},
    ]
    return "\n".join(json.dumps(event) for event in events) + "\n"


def test_role_span_covers_gate_and_separates_schema_attempts(
        tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    sidecar = tmp_path / "usage.jsonl.spans.jsonl"
    calls = []

    class Backend:
        def complete(self, role, prompt, variables):
            calls.append((role, prompt, dict(variables)))
            if len(calls) == 1:
                return {"ok": "not-a-bool"}
            return {"ok": True}

    @contextmanager
    def checked_slot():
        records = _rows(sidecar)
        assert records[-1]["record_type"] == "role_attempt_started"
        assert any(r["record_type"] == "role_span_started" for r in records)
        assert not any(r["record_type"] == "role_span_finished" for r in records)
        yield
        records = _rows(sidecar)
        assert records[-1]["record_type"] == "seat_released"
        assert records[-2]["record_type"] == "seat_acquired"
        assert not any(r["record_type"] == "role_span_finished" for r in records)

    monkeypatch.setattr("verifier.llm.agents.SEAT_GATE.slot", checked_slot)
    role = AgentRole("critic", "payload={payload}", {"ok": bool})
    with codex_usage_ledger(str(usage), {}):
        with model_scope(
                claim_id="claim-a", stage="contract_critic",
                round=3, private_feedback="must-not-appear"):
            assert role.call(Backend(), {"payload": "candidate-secret"}) == {
                "ok": True}

    rows = _rows(sidecar)
    role_started = [r for r in rows
                    if r["record_type"] == "role_span_started"]
    role_finished = [r for r in rows
                     if r["record_type"] == "role_span_finished"]
    attempts = [r for r in rows
                if r["record_type"] == "role_attempt_finished"]
    assert len(role_started) == len(role_finished) == 1
    assert role_finished[0]["status"] == "succeeded"
    assert [r["role_attempt_index"] for r in attempts] == [1, 2]
    assert [r["status"] for r in attempts] == ["error", "succeeded"]
    assert attempts[0]["error_class"] == "RoleError"
    assert len({r["role_span_id"] for r in attempts}) == 1
    encoded = sidecar.read_text()
    assert "candidate-secret" not in encoded
    assert "private_feedback" not in encoded
    assert all(r.get("claim_id") == "claim-a" for r in rows)
    seats = [r for r in rows if r["record_type"] == "seat_acquired"]
    releases = [r for r in rows if r["record_type"] == "seat_released"]
    assert len(seats) == len(releases) == 2
    assert all(r["queue_duration_ns"] >= 0 for r in seats)
    assert all(r["held_duration_ns"] >= 0 for r in releases)


def test_codex_provider_retries_share_one_call_span(tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    role = AgentRole("critic", "prompt", {"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))
    calls = []

    def fake_run(cmd, **_kwargs):
        calls.append(list(cmd))
        if len(calls) == 1:
            return SimpleNamespace(
                returncode=1, stdout="", stderr="HTTP status 503")
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stdout=_usage_stream(), stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    monkeypatch.setattr("verifier.llm.codex_backend.time.sleep", lambda _n: None)
    with codex_usage_ledger(
            str(usage), {str(tmp_path): {"trial": "t1"}}):
        with model_scope(
                claim_id="claim-1", obligation_id="ob-1",
                stage="judger", redo=2, judger_retry_index=1):
            assert role.call(backend, {}) == {"ok": True}

    usage_rows = _rows(usage)
    span_rows = _rows(tmp_path / "usage.jsonl.spans.jsonl")
    assert [row["attempt_index"] for row in usage_rows] == [1, 2]
    assert len({row["call_id"] for row in usage_rows}) == 1
    started, = [r for r in span_rows
                if r["record_type"] == "model_call_started"]
    finished, = [r for r in span_rows
                 if r["record_type"] == "model_call_finished"]
    assert started["call_id"] == finished["call_id"] \
        == usage_rows[0]["call_id"]
    assert started["attribution"] == "attributed"
    assert started["role_attempt_index"] == 1
    assert started["claim_id"] == "claim-1"
    assert started["obligation_id"] == "ob-1"
    assert started["redo"] == 2
    assert all("role_span_id" not in row for row in usage_rows)
    assert all("span_schema_version" not in row for row in usage_rows)
    assert not any("claim-1" in str(part) for cmd in calls for part in cmd)


def test_direct_codex_call_is_explicitly_unattributed(tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    backend = CodexBackend(
        roles={"critic": SimpleNamespace(system="", schema={"ok": bool})},
        cwd=str(tmp_path),
    )

    def fake_run(cmd, **_kwargs):
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stdout=_usage_stream(), stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
            str(usage), {str(tmp_path): {"trial": "direct"}}):
        assert backend.complete("critic", "opaque prompt", {
            "sensitive": "opaque variable"}) == {"ok": True}

    spans = _rows(tmp_path / "usage.jsonl.spans.jsonl")
    started, = [r for r in spans
                if r["record_type"] == "model_call_started"]
    assert started["attribution"] == "unattributed"
    assert "role_span_id" not in started
    encoded = json.dumps(spans)
    assert "opaque prompt" not in encoded
    assert "opaque variable" not in encoded


def test_concurrent_submits_use_isolated_context_copies(tmp_path):
    usage = tmp_path / "usage.jsonl"
    barrier = threading.Barrier(2)
    role = AgentRole("critic", "fixed", {"ok": bool})

    class Backend:
        def complete(self, _role, _prompt, _variables):
            barrier.wait(timeout=2)
            return {"ok": True}

    with codex_usage_ledger(str(usage), {}):
        with ThreadPoolExecutor(max_workers=2) as pool:
            with model_scope(claim_id="claim-a", stage="contract_critic"):
                first = submit_with_context(pool, role.call, Backend(), {})
            with model_scope(claim_id="claim-b", stage="contract_critic"):
                second = submit_with_context(pool, role.call, Backend(), {})
            assert first.result() == {"ok": True}
            assert second.result() == {"ok": True}

    starts = [row for row in _rows(
        tmp_path / "usage.jsonl.spans.jsonl")
        if row["record_type"] == "role_span_started"]
    assert sorted(row["claim_id"] for row in starts) == ["claim-a", "claim-b"]
    assert len({row["role_span_id"] for row in starts}) == 2


def test_outer_judger_retry_is_distinct_from_role_schema_retry(tmp_path):
    usage = tmp_path / "usage.jsonl"
    state = tmp_path / "state"
    state.mkdir()
    (state / "task.md").write_text("fixed authority", encoding="utf-8")
    calls = 0

    class Backend:
        def complete(self, _role, _prompt, _variables):
            nonlocal calls
            calls += 1
            if calls <= 2:
                raise RoleError("malformed role output")
            return {
                "sufficient": "yes", "faithful": "yes",
                "degradation": "n/a", "reason": "accepted",
            }

    artifacts = ArtifactStore(str(state / "artifacts"))
    game = BindingGame(
        Backend(), artifacts, EventLog(str(state / "events.jsonl")),
        SimpleNamespace(), judger_role=JUDGER_AGENT,
        judger_error_retries=1, live_world=True,
    )
    obligation = Obligation("claim", "fixed obligation")
    attempt = Attempt(method={"procedure": "fixed"})
    result = DischargeResult(
        outcome="holds", witness="fixed", artifact="digest",
        instrument="fixed", witness_kind="execution",
    )
    with codex_usage_ledger(str(usage), {}):
        with model_scope(claim_id="claim", obligation_id=obligation.id):
            verdict = game._judge(
                obligation, attempt.method, result, attempt,
                method_attempt=4)
    assert verdict["faithful"] == "yes"

    rows = _rows(tmp_path / "usage.jsonl.spans.jsonl")
    spans = [r for r in rows if r["record_type"] == "role_span_started"]
    attempts = [r for r in rows
                if r["record_type"] == "role_attempt_started"]
    assert [r["judger_retry_index"] for r in spans] == [1, 2]
    assert [r["role_attempt_index"] for r in attempts] == [1, 2, 1]
    assert [r["judger_retry_index"] for r in attempts] == [1, 1, 2]
    assert all(r["method_attempt"] == 4 for r in spans)


def test_binding_game_stamps_claim_obligation_and_stage(tmp_path):
    usage = tmp_path / "usage.jsonl"
    state = tmp_path / "game"
    state.mkdir()
    (state / "task.md").write_text("fixed task", encoding="utf-8")
    responses = {
        "oracles": {"methods": [{
            "procedure": "fixed procedure",
            "artifact": "fixed artifact",
            "level": "independent recomputation",
            "basis": "world",
            "strength": "decisive",
            "bridges": [],
        }]},
        "discharge": {"ok": True},
        "judger": {
            "sufficient": "yes", "faithful": "yes",
            "degradation": "n/a", "reason": "accepted",
        },
    }

    class Backend:
        def complete(self, role, _prompt, _variables):
            return responses[role]

    backend = Backend()
    discharge_role = AgentRole("discharge", "fixed", {"ok": bool})

    class Discharger:
        def discharge(self, _row, _obligation, method=None, tag=""):
            discharge_role.call(backend, {})
            return DischargeResult(
                outcome="holds", witness="fixed", artifact="digest",
                instrument="fixed", witness_kind="execution",
            )

    game = BindingGame(
        backend, ArtifactStore(str(state / "artifacts")),
        EventLog(str(state / "events.jsonl")), Discharger(), live_world=True,
    )
    row = {"id": "support-claim", "kind": "assertion", "text": "holds"}
    obligation = Obligation("support-claim", "fixed obligation")
    with codex_usage_ledger(str(usage), {}):
        assert game.run(row, obligation).outcome == "holds"

    starts = [r for r in _rows(tmp_path / "usage.jsonl.spans.jsonl")
              if r["record_type"] == "role_span_started"]
    assert [r["stage"] for r in starts] == [
        "oracle_enumerate", "discharge", "judger"]
    assert all(r["claim_id"] == "support-claim" for r in starts)
    assert all(r["obligation_id"] == obligation.id for r in starts)
    assert all(r["locus"] == "support" for r in starts)
    assert starts[1]["method_attempt"] == 1
    assert starts[1]["redo"] == 0
    assert starts[2]["judger_retry_index"] == 1


def test_base_exception_closes_role_and_model_call_without_raw_error(
        tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    role = AgentRole("critic", "fixed", {"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))

    def interrupted(_cmd, **_kwargs):
        error = KeyboardInterrupt("sensitive interrupt body")
        error._gv_process_started = True
        raise error

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", interrupted)
    with codex_usage_ledger(
            str(usage), {str(tmp_path): {"trial": "interrupt"}}):
        with pytest.raises(KeyboardInterrupt):
            with model_scope(stage="discharge", method_attempt=2):
                role.call(backend, {})

    spans = _rows(tmp_path / "usage.jsonl.spans.jsonl")
    finished = [r for r in spans if r["record_type"].endswith("_finished")]
    assert {r["record_type"] for r in finished} == {
        "model_call_finished", "role_attempt_finished", "role_span_finished",
    }
    assert all(r["status"] == "interrupted" for r in finished)
    assert all(r["error_class"] == "KeyboardInterrupt" for r in finished)
    assert "sensitive interrupt body" not in json.dumps(spans)


def test_interrupted_scopes_reset_before_following_direct_call(
        tmp_path, monkeypatch):
    usage = tmp_path / "usage.jsonl"
    role = AgentRole("critic", "fixed", {"ok": bool})
    backend = CodexBackend(roles={"critic": role}, cwd=str(tmp_path))
    calls = 0

    def fake_run(cmd, **_kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            error = KeyboardInterrupt()
            error._gv_process_started = True
            raise error
        with open(cmd[cmd.index("-o") + 1], "w", encoding="utf-8") as handle:
            json.dump({"ok": True}, handle)
        return SimpleNamespace(
            returncode=0, stdout=_usage_stream("second"), stderr="")

    monkeypatch.setattr(
        "verifier.llm.codex_backend._run_codex_process", fake_run)
    with codex_usage_ledger(
            str(usage), {str(tmp_path): {"trial": "reset"}}):
        with pytest.raises(KeyboardInterrupt):
            with model_scope(claim_id="old-claim", stage="discharge"):
                role.call(backend, {})
        assert backend.complete("critic", "fixed", {}) == {"ok": True}

    starts = [r for r in _rows(tmp_path / "usage.jsonl.spans.jsonl")
              if r["record_type"] == "model_call_started"]
    assert starts[0]["attribution"] == "attributed"
    assert starts[0]["claim_id"] == "old-claim"
    assert starts[1]["attribution"] == "unattributed"
    assert "claim_id" not in starts[1]


def test_no_usage_capture_keeps_span_telemetry_inert(tmp_path):
    role = AgentRole("critic", "fixed", {"ok": bool})
    with model_scope(claim_id="claim"):
        assert role.call(
            SimpleNamespace(complete=lambda *_args: {"ok": True}), {}) \
            == {"ok": True}
    assert not Path(
        span_sidecar_path(str(tmp_path / "usage.jsonl"))).exists()
