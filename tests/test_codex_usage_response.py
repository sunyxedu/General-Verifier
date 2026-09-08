"""Billing gaps never fabricate cost or repeat an independently valid answer."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from experiments.terminal_bench.analyze_model_spans import analyze
from verifier.llm.agents import AgentRole
from verifier.llm.authority_backend import (
    _AuthorityCodexBackend, AuthorityBackendUnavailable, _DISABLED_CODE_MODE_NOTICE,
)
from verifier.llm.codex_backend import (
    CodexBackend, _attempt_usage, _completed_response_without_usage,
    codex_usage_ledger,
)
from verifier.llm.model_spans import model_scope
from verifier.llm.schema import RoleError


def events(usage=None, text='{"ok": true}'):
    return [
        {"type": "thread.started", "thread_id": "exact-current-thread"},
        {"type": "turn.started"},
        {"type": "item.completed", "item": {"id": "answer", "type": "agent_message", "text": text}},
        {"type": "turn.completed", "usage": {} if usage is None else usage},
    ]


def encoded(rows):
    return "\n".join(json.dumps(row) for row in rows) + "\n"


def accounting_stream(extra_line, *, usage=None):
    counts = usage or {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "output_tokens": 12,
        "reasoning_output_tokens": 7,
    }
    return "\n".join([
        json.dumps({
            "type": "thread.started", "thread_id": "exact-current-thread",
        }),
        extra_line,
        json.dumps({"type": "turn.completed", "usage": counts}),
    ]) + "\n"


@pytest.mark.parametrize("extra_line", [
    '{"type":"item.completed","item":{"type":"command_execution",'
    '"payload":{"same":1,"same":2}}}',
    '{"type":"item.completed","item":{"payload":{"broken":,}}}',
    '{"type":"item.completed","item":{"payload":[',
    '{"type":"item.completed","item":{"type":"command_execution",'
    '"payload":' + "[" * 2000 + "0" + "]" * 2000 + "}}",
    '{"type":"item.completed","item":{"payload":' + "[" * 50_000,
    json.dumps({
        "type": "item.completed",
        "item": {
            "type": "command_execution",
            "payload": "x" * (2 * 1024 * 1024),
        },
    }),
])
def test_billing_ignores_unrelated_nested_deep_or_large_item_payload(extra_line):
    attempt = _attempt_usage(accounting_stream(extra_line))
    assert attempt.status == "complete"
    assert attempt.thread_id == "exact-current-thread"
    assert attempt.telemetry is not None
    assert attempt.telemetry.usage == {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "output_tokens": 12,
        "reasoning_output_tokens": 7,
    }


@pytest.mark.parametrize("extra_line", [
    '{"type":"turn.completed","type":"item.completed",'
    '"usage":{"input_tokens":1,"cached_input_tokens":0,'
    '"output_tokens":1,"reasoning_output_tokens":0}}',
    '{"type":"turn.completed",'
    '"usage":{"input_tokens":1,"cached_input_tokens":0,'
    '"output_tokens":1,"reasoning_output_tokens":0},'
    '"usage":{"input_tokens":1,"cached_input_tokens":0,'
    '"output_tokens":1,"reasoning_output_tokens":0}}',
    '{"type":"turn.completed","usage":',
])
def test_billing_rejects_malformed_or_duplicate_accounting(extra_line):
    attempt = _attempt_usage(accounting_stream(extra_line))
    assert attempt.status == "invalid"
    assert attempt.telemetry is None


@pytest.mark.parametrize("extra_line", [
    '[]',
    '{"item":{},"type":"item.completed"}',
    '{"type" "item.completed","item":{}}',
    '{"type":',
    '{"type":true,"item":{}}',
])
def test_billing_rejects_ambiguous_non_accounting_prefix(extra_line):
    attempt = _attempt_usage(accounting_stream(extra_line))
    assert attempt.status == "invalid"
    assert attempt.telemetry is None


def test_billing_rejects_tampered_counter_after_ignoring_item_payload():
    attempt = _attempt_usage(accounting_stream(
        '{"type":"item.completed","item":{"payload":{"same":1,"same":2}}}',
        usage={
            "input_tokens": 10,
            "cached_input_tokens": 11,
            "output_tokens": 2,
            "reasoning_output_tokens": 1,
        },
    ))
    assert attempt.status == "invalid"
    assert attempt.telemetry is None
    assert attempt.error == (
        "codex exec reported cached_input_tokens greater than input_tokens")


def setup_call(tmp_path, monkeypatch, stream, file_text='{"ok": true}', *, authority=False, role_name="critic"):
    role = AgentRole(role_name, "Verify the requirement.", {"ok": bool})
    backend_class = _AuthorityCodexBackend if authority else CodexBackend
    backend = backend_class(roles={role_name: role}, cwd=str(tmp_path))
    if authority:
        backend._authority_scratch = str(tmp_path)
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        path = Path(cmd[cmd.index("-o") + 1])
        if file_text is not None:
            path.write_text(file_text)
        return SimpleNamespace(returncode=0, stdout=stream, stderr="")

    monkeypatch.setattr("verifier.llm.codex_backend._run_codex_process", run)
    return role, backend, calls


def test_scoped_billing_does_not_relax_full_response_stream_validation(
    tmp_path, monkeypatch,
):
    usage = {
        "input_tokens": 100,
        "cached_input_tokens": 60,
        "output_tokens": 12,
        "reasoning_output_tokens": 7,
    }
    rows = events(usage)
    malformed_item = (
        '{"type":"item.completed","item":{"type":"command_execution",'
        '"payload":{"broken":,}}}'
    )
    stream = "\n".join([
        json.dumps(rows[0]), json.dumps(rows[1]), malformed_item,
        json.dumps(rows[2]), json.dumps(rows[3]),
    ]) + "\n"
    _, backend, calls = setup_call(tmp_path, monkeypatch, stream)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError, match="malformed line"):
            backend.complete("critic", "prompt", {})
    assert len(calls) == 1
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] == "output_error"
    assert row["usage_status"] == "complete"
    assert row["usage_complete"] is True
    assert row["input_tokens"] == 100


@pytest.mark.parametrize("usage,code", [
    ({}, "invalid_counter:input_tokens"),
    ({"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2},
     "invalid_counter:reasoning_output_tokens"),
    ({"input_tokens": 10, "cached_input_tokens": 11, "output_tokens": 2,
      "reasoning_output_tokens": 1}, "cached_exceeds_input"),
    ({"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2,
      "reasoning_output_tokens": "usage-secret"}, "invalid_counter:reasoning_output_tokens"),
])
def test_complete_schema_valid_response_needs_one_call_but_cost_stays_incomplete(
    tmp_path, monkeypatch, usage, code,
):
    role, backend, calls = setup_call(
        tmp_path, monkeypatch, encoded(events(usage)))
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {"trial": "fixture"}}):
        with model_scope(claim_id="a" * 64, claim_kind="assertion", locus="support",
                         stage="contract_critic", round=0):
            assert role.call(backend, {}) == {"ok": True}
    assert len(calls) == 1
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 1
    row = rows[0]
    assert row["attempt_status"] == "succeeded"
    assert row["usage_status"] == "invalid" and row["usage_complete"] is False
    assert row["usage_error_code"] == code
    assert row["thread_id"] == "exact-current-thread"
    for field in ("input_tokens", "cached_input_tokens", "uncached_input_tokens",
                  "output_tokens", "reasoning_tokens", "cache_write_input_tokens"):
        assert row[field] is None
    assert "usage-secret" not in ledger.read_text()
    report = analyze(ledger, Path(str(ledger) + ".spans.jsonl"))
    assert report["cost_complete"] is False
    assert report["totals"]["started_attempt_count"] == 1


@pytest.mark.parametrize("change", [
    "no_thread", "no_turn", "no_completion", "no_message", "duplicate_thread",
    "duplicate_completion", "message_after_completion", "event_after_completion",
    "turn_failed", "error", "item_error", "malformed", "non_object", "duplicate_key",
    "huge_number", "deep_json", "unfinished_item", "duplicate_item",
    "update_without_start", "type_change", "second_message", "item_after_message",
])
def test_billing_fallback_never_accepts_an_ambiguous_or_incomplete_stream(
    tmp_path, monkeypatch, change,
):
    rows = events()
    if change == "no_thread":
        rows.pop(0)
    elif change == "no_turn":
        rows.pop(1)
    elif change == "no_completion":
        rows.pop()
    elif change == "no_message":
        rows.pop(2)
    elif change == "duplicate_thread":
        rows.insert(1, rows[0])
    elif change == "duplicate_completion":
        rows.append(rows[-1])
    elif change == "message_after_completion":
        rows[2], rows[3] = rows[3], rows[2]
    elif change == "event_after_completion":
        rows.append({"type": "turn.started"})
    elif change in {"turn_failed", "error"}:
        rows.insert(2, {"type": "turn.failed" if change == "turn_failed" else "error"})
    elif change == "item_error":
        rows.insert(2, {"type": "item.completed", "item": {"type": "error"}})
    elif change == "non_object":
        rows.insert(2, [])
    elif change == "unfinished_item":
        rows.insert(2, {"type": "item.started", "item": {"id": "cmd", "type": "command_execution"}})
    elif change == "duplicate_item":
        item = {"type": "item.completed", "item": {"id": "cmd", "type": "command_execution"}}
        rows[2:2] = [item, item]
    elif change == "update_without_start":
        rows.insert(2, {"type": "item.updated", "item": {"id": "cmd", "type": "command_execution"}})
    elif change == "type_change":
        rows[2:2] = [
            {"type": "item.started", "item": {"id": "cmd", "type": "command_execution"}},
            {"type": "item.completed", "item": {"id": "cmd", "type": "reasoning"}},
        ]
    elif change == "second_message":
        rows.insert(2, {"type": "item.completed", "item": {"id": "another", "type": "agent_message", "text": "another message"}})
    elif change == "item_after_message":
        rows.insert(3, {"type": "item.completed", "item": {"id": "later", "type": "reasoning"}})
    stream = encoded(rows)
    if change == "malformed":
        stream += "not-json\n"
    elif change == "duplicate_key":
        stream = stream.replace('"usage": {}', '"usage": {}, "usage": {}')
    elif change == "huge_number":
        stream += '{"value":' + "9" * 5000 + '}\n'
    elif change == "deep_json":
        stream += "[" * 2000 + "0" + "]" * 2000 + "\n"
    assert _completed_response_without_usage(stream) is None
    _, backend, calls = setup_call(tmp_path, monkeypatch, stream)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError):
            backend.complete("critic", "prompt", {})
    assert len(calls) == 1
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] == "telemetry_error"
    assert row["usage_complete"] is False


@pytest.mark.parametrize("file_text", [None, "", "  \n", '{"ok": false}'])
def test_completed_stream_and_output_file_must_agree(tmp_path, monkeypatch, file_text):
    _, backend, _ = setup_call(tmp_path, monkeypatch, encoded(events()),
                               file_text=file_text)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError, match="disagrees"):
            backend.complete("critic", "prompt", {})
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] == "output_error"
    assert row["error_kind"] == "output_stream_mismatch"
    assert row["usage_complete"] is False


def test_complete_response_still_obeys_role_schema_and_normal_retry_budget(tmp_path, monkeypatch):
    text = '{"ok": "not-a-boolean"}'
    role, backend, calls = setup_call(tmp_path, monkeypatch, encoded(events(text=text)), text)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError, match="failed twice"):
            role.call(backend, {})
    assert len(calls) == 2
    rows = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert len(rows) == 2
    assert all(row["attempt_status"] == "schema_error" and not row["usage_complete"]
               for row in rows)


@pytest.mark.parametrize("tool_present", [False, True])
@pytest.mark.parametrize("startup_notice", [False, True])
def test_authority_capability_policy_is_independent_of_billing_fallback(tmp_path, monkeypatch, tool_present, startup_notice):
    rows = events()
    if tool_present:
        rows.insert(2, {"type": "item.completed", "item": {"id": "cmd", "type": "command_execution"}})
    if startup_notice:
        rows.insert(1, {"type": "item.completed", "item": {
            "id": "notice", "type": "error", "message": _DISABLED_CODE_MODE_NOTICE}})
    _, backend, _ = setup_call(tmp_path, monkeypatch, encoded(rows), authority=True)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        if tool_present:
            with pytest.raises(AuthorityBackendUnavailable):
                backend.complete("critic", "prompt", {})
        else:
            assert backend.complete("critic", "prompt", {}) == {"ok": True}
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] == ("output_error" if tool_present else "succeeded")
    assert row["usage_complete"] is False


@pytest.mark.parametrize("usage", [None, [], "not-an-object"])
def test_missing_usage_object_is_not_a_counter_only_failure(tmp_path, monkeypatch, usage):
    rows = events()
    rows[-1]["usage"] = usage
    _, backend, _ = setup_call(tmp_path, monkeypatch, encoded(rows))
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError):
            backend.complete("critic", "prompt", {})
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] == "telemetry_error"
    assert row["usage_error_code"] == "completion_or_usage_object"


def test_complete_item_lifecycles_can_recover_without_relaxing_counters():
    rows = events()
    rows[2:2] = [
        {"type": "item.started", "item": {"id": "cmd", "type": "command_execution"}},
        {"type": "item.updated", "item": {"id": "cmd", "type": "command_execution"}},
        {"type": "item.completed", "item": {"id": "cmd", "type": "command_execution"}},
    ]
    assert _completed_response_without_usage(encoded(rows)) == '{"ok": true}'


@pytest.mark.parametrize("role_name", ["extract", "obligations", "critic", "oracles", "discharge", "judger"])
@pytest.mark.parametrize("world_mode", ["live", "captured"])
def test_counter_only_recovery_is_role_and_world_mode_independent(tmp_path, monkeypatch, role_name, world_mode):
    world = tmp_path / "world"
    world.mkdir()
    role, backend, calls = setup_call(tmp_path, monkeypatch, encoded(events()), role_name=role_name)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        assert role.call(backend, {
            "world_mode": world_mode, "world_path": str(world),
            "ledger_path": str(tmp_path / "events.jsonl"),
        }) == {"ok": True}
    assert len(calls) == 1
    row = json.loads(ledger.read_text())
    assert row["role"] == role_name and row["attempt_status"] == "succeeded"
    assert row["usage_complete"] is False and row["input_tokens"] is None


@pytest.mark.parametrize("field", ["input_tokens", "cached_input_tokens", "output_tokens",
                                   "reasoning_output_tokens", "cache_write_input_tokens"])
@pytest.mark.parametrize("invalid", [True, -1, "private-counter-value"])
def test_every_counter_error_is_bounded_and_never_zero_filled(tmp_path, monkeypatch, field, invalid):
    usage = {"input_tokens": 10, "cached_input_tokens": 3, "output_tokens": 2,
             "reasoning_output_tokens": 1, "cache_write_input_tokens": 4}
    usage[field] = invalid
    _, backend, calls = setup_call(tmp_path, monkeypatch, encoded(events(usage)))
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        assert backend.complete("critic", "prompt", {}) == {"ok": True}
    assert len(calls) == 1
    row = json.loads(ledger.read_text())
    assert row["usage_error_code"] == "invalid_counter:" + field
    assert not row["usage_complete"]
    assert all(row[key] is None for key in (
        "input_tokens", "cached_input_tokens", "uncached_input_tokens",
        "output_tokens", "reasoning_tokens", "cache_write_input_tokens"))
    assert "private-counter-value" not in ledger.read_text()


def test_recovery_preserves_a_valid_negative_role_response(tmp_path, monkeypatch):
    text = '{"ok": false}'
    role, backend, calls = setup_call(tmp_path, monkeypatch, encoded(events(text=text)), text)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        assert role.call(backend, {}) == {"ok": False}
    assert len(calls) == 1
    assert not json.loads(ledger.read_text())["usage_complete"]


@pytest.mark.parametrize("change", ["duplicate", "duplicate_new_id", "missing_id", "changed_message"])
def test_authority_recovery_accepts_only_one_exact_identified_startup_notice(tmp_path, monkeypatch, change):
    rows = events()
    item = {"id": "notice", "type": "error", "message": _DISABLED_CODE_MODE_NOTICE}
    if change == "missing_id":
        item.pop("id")
    elif change == "changed_message":
        item["message"] += "not the permitted notice"
    notice = {"type": "item.completed", "item": item}
    rows.insert(1, notice)
    if change in {"duplicate", "duplicate_new_id"}:
        rows.insert(2, {"type": "item.completed", "item": dict(
            item, id="different" if change == "duplicate_new_id" else "notice")})
    _, backend, _ = setup_call(tmp_path, monkeypatch, encoded(rows), authority=True)
    ledger = tmp_path / "usage.jsonl"
    with codex_usage_ledger(str(ledger), {str(tmp_path): {}}):
        with pytest.raises(RoleError):
            backend.complete("critic", "prompt", {})
    row = json.loads(ledger.read_text())
    assert row["attempt_status"] in {"telemetry_error", "output_error"}
    assert not row["usage_complete"]
