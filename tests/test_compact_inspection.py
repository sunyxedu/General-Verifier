"""An opt-in output preference must not change role or capability contracts."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from verifier.engine.method_cache import _backend_identity
from verifier.llm import codex_backend as cb


def _runner(calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        Path(command[command.index("-o") + 1]).write_text('{"ok":true}')
        events = [
            {"type": "thread.started", "thread_id": "local-stub"},
            {"type": "turn.started"},
            {"type": "item.completed", "item": {
                "id": "answer", "type": "agent_message", "text": '{"ok":true}'}},
            {"type": "turn.completed", "usage": {
                "input_tokens": 100, "cached_input_tokens": 20,
                "output_tokens": 5, "reasoning_output_tokens": 2}},
        ]
        return SimpleNamespace(returncode=0, stderr="",
                               stdout="\n".join(map(json.dumps, events)))
    return run


@pytest.mark.parametrize("role_name", [
    "extract", "obligations", "critic", "oracles", "discharge", "judger"])
@pytest.mark.parametrize("world_mode", ["live", "captured", "unavailable"])
def test_opt_in_compacts_reports_and_all_post_extract_roles_get_inspection_note(
        tmp_path, monkeypatch, role_name, world_mode):
    role = SimpleNamespace(system="fixed system", schema={"ok": bool})
    calls = []
    monkeypatch.setattr(cb, "_run_codex_process", _runner(calls))
    backends = [cb.CodexBackend(
        roles={role_name: role}, cwd=str(tmp_path), model="gpt-5.6-luna",
        effort="medium", service_tier="standard", compact_inspection=enabled)
        for enabled in (False, True)]
    for backend in backends:
        assert backend.complete(role_name, "exact role input", {
            "world_mode": world_mode, "basis": "record"}) == {"ok": True}
    (base, base_args), (opt, opt_args) = calls
    assert cb._COMPACT_INSPECTION_NOTE not in base[-1]
    assert cb._COMPACT_REPORT_NOTE not in base[-1]
    assert (cb._ORACLE_COMPILE_NOTE in base[-1]) is (role_name == "oracles")
    assert (cb._ORACLE_COMPILE_NOTE in opt[-1]) is (role_name == "oracles")
    expected = role_name not in {"extract", "oracles"}
    assert (cb._COMPACT_INSPECTION_NOTE in opt[-1]) is expected
    assert cb._COMPACT_REPORT_NOTE in opt[-1]
    assert opt[-1].replace(cb._COMPACT_INSPECTION_NOTE, "").replace(
        cb._COMPACT_REPORT_NOTE, "") == base[-1]
    # The only command difference is prompt text and the fresh output path.
    def command_contract(command):
        command = list(command[:-1])
        command[command.index("-o") + 1] = "FRESH_OUTPUT"
        return command
    assert command_contract(opt) == command_contract(base)
    assert base_args["cwd"] == opt_args["cwd"]
    assert base_args["timeout"] == opt_args["timeout"] == 1800
    assert role.system == "fixed system" and role.schema == {"ok": bool}


def test_default_keeps_previous_identity_and_profiles_never_share(monkeypatch):
    base = cb.CodexBackend()
    opt = cb.CodexBackend(compact_inspection=True)
    assert base.compact_inspection is False
    assert base.runtime_instruction_identity == hashlib.sha256(
        (cb._ARTIFACT_ADDRESS_NOTE + cb._ORACLE_COMPILE_NOTE).encode()
    ).hexdigest()
    assert _backend_identity(base) != _backend_identity(opt)
    before = _backend_identity(opt)
    monkeypatch.setattr(cb, "_COMPACT_INSPECTION_NOTE", "changed runtime note")
    assert _backend_identity(opt) == before
    assert _backend_identity(cb.CodexBackend(compact_inspection=True)) != before


@pytest.mark.parametrize("invalid", [None, 0, 1, "1", [], {}])
def test_profile_requires_explicit_bool(invalid):
    with pytest.raises(ValueError, match="must be a bool"):
        cb.CodexBackend(compact_inspection=invalid)


@pytest.mark.parametrize("enabled", [False, True])
def test_attempt_bill_records_actual_profile_without_changing_usage(
        tmp_path, monkeypatch, enabled):
    calls = []
    monkeypatch.setattr(cb, "_run_codex_process", _runner(calls))
    role = SimpleNamespace(system="fixed", schema={"ok": bool})
    backend = cb.CodexBackend(roles={"judger": role}, cwd=str(tmp_path),
                              compact_inspection=enabled)
    usage = tmp_path / "usage.jsonl"
    with cb.codex_usage_ledger(str(usage), {str(tmp_path): {"fixture": True}}):
        assert backend.complete("judger", "fixed", {}) == {"ok": True}
    row, = [json.loads(line) for line in usage.read_text().splitlines()]
    assert row["compact_inspection"] is enabled
    assert row["runtime_instruction_identity"] == backend.runtime_instruction_identity
    assert row["usage_complete"] is True
    assert row["input_tokens"] == 100
    assert row["cached_input_tokens"] == 20
    assert row["uncached_input_tokens"] == 80
    assert row["output_tokens"] == 5
    assert row["execution_diagnostic"]["complete"] is True


def test_note_preserves_full_evidence_and_no_historical_reexecution():
    note = cb._COMPACT_INSPECTION_NOTE
    for invariant in (
        "existing role and tool permissions",
        "required input must still be inspected completely",
        "not proof of uninspected content",
        "independent semantic audit",
        "not permission to omit evidence",
        "re-execute a historical action",
        "manifest is lossless source-event transport",
    ):
        assert invariant in note


def test_compact_report_note_preserves_every_role_gate_and_failure():
    note = cb._COMPACT_REPORT_NOTE
    for invariant in (
        "perform every analysis",
        "Preserve every required output item",
        "never merge or omit claims, obligations, methods, defects",
        "retain each required label",
        "<REQUIRED LABEL>: pass",
        "sufficient=yes, faithful=yes, degradation=n/a",
        "every independently visible defect",
        "not permission to weaken a contract, skip a gate",
    ):
        assert invariant in note
