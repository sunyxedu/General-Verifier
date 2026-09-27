"""An insufficient agent method is rejected before the executor runs."""
from __future__ import annotations

import copy
import json

import pytest

from verifier import offline
from verifier.config import sufficiency_preaudit_enabled
from verifier.llm.agents import SUFFICIENCY_AGENT
from verifier.llm.schema import FixtureBackend
from verifier.app.assembly import build_kernel
from verifier.mcp_server import Service
from test_certificate_scope import _fixtures, _service as _scoped_service


def _service(tmp_path, backend, monkeypatch, enabled=True):
    monkeypatch.setenv("GV_SUFFICIENCY_PREAUDIT", "1" if enabled else "0")
    return _scoped_service(tmp_path, backend, monkeypatch)


def _roles(backend):
    return [call["role"] for call in backend.calls]


def test_default_is_off_and_flag_is_strict(monkeypatch):
    monkeypatch.delenv("GV_SUFFICIENCY_PREAUDIT", raising=False)
    assert sufficiency_preaudit_enabled() is False
    assert sufficiency_preaudit_enabled({"GV_SUFFICIENCY_PREAUDIT": "1"}) is True
    with pytest.raises(ValueError):
        sufficiency_preaudit_enabled({"GV_SUFFICIENCY_PREAUDIT": "yes"})
    assert SUFFICIENCY_AGENT.tools == [] and SUFFICIENCY_AGENT.max_turns == 1
    on = offline.config_fields(model="m", seats="codex")
    monkeypatch.setenv("GV_SUFFICIENCY_PREAUDIT", "1")
    assert offline.config_fields(model="m", seats="codex")["config"] != on["config"]


def test_sufficient_method_runs_and_is_still_judged(tmp_path, monkeypatch):
    fixtures = _fixtures()
    fixtures["sufficiency"] = {"sufficient": "yes",
                               "reason": "the read decides the exact term"}
    backend = FixtureBackend(fixtures)
    service = _service(tmp_path, backend, monkeypatch)
    result = service.verify("f")
    assert result["status"] == "verified"
    roles = _roles(backend)
    assert roles.index("sufficiency") < roles.index("discharge") < roles.index("judger")
    audits = [e["payload"] for e in service.log.replay()
              if e["type"] == "sufficiency_audited"]
    assert audits and audits[0]["sufficient"] == "yes"
    variables = next(c["variables"] for c in backend.calls if c["role"] == "sufficiency")
    assert variables["procedure"] == fixtures["oracles"]["methods"][0]["procedure"]
    assert variables["declared_strength"] == "decisive"
    assert "evidence" not in variables and "artifact_root" not in variables
    # a first attempt has no prior attempts to be briefed on
    discharge = next(c["variables"] for c in backend.calls if c["role"] == "discharge")
    assert discharge["redo_feedback"] == "(none)"
    assert not [e for e in service.log.replay() if e["type"] == "executor_briefed"]


def test_insufficient_method_is_rejected_without_execution(tmp_path, monkeypatch):
    fixtures = _fixtures()
    fixtures["sufficiency"] = {"sufficient": "no",
                               "reason": "the mapping from bytes to the term is unstated"}
    backend = FixtureBackend(fixtures)
    service = _service(tmp_path, backend, monkeypatch)
    result = service.verify("f")
    assert result["status"] != "verified"
    roles = _roles(backend)
    assert "discharge" not in roles and "judger" not in roles
    # the Oracle got its one repair pass with the pre-audit's reason
    assert roles.count("oracles") == 2
    repair = [c for c in backend.calls if c["role"] == "oracles"][1]
    assert "bytes to the term" in repair["variables"]["method_feedback"]
    rejected = [e["payload"] for e in service.log.replay()
                if e["type"] == "method_rejected"]
    assert rejected and rejected[0]["reason"].startswith("sufficiency pre-audit:")


def test_seat_failure_never_blocks_execution(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())  # no sufficiency fixture: RoleError
    service = _service(tmp_path, backend, monkeypatch)
    assert service.verify("f")["status"] == "verified"
    roles = _roles(backend)
    assert "discharge" in roles and "judger" in roles
    audits = [e["payload"] for e in service.log.replay()
              if e["type"] == "sufficiency_audited"]
    assert audits and "error" in audits[0]


def test_disabled_flag_keeps_the_established_sequence(tmp_path, monkeypatch):
    fixtures = _fixtures()
    fixtures["sufficiency"] = {"sufficient": "no", "reason": "never asked"}
    backend = FixtureBackend(fixtures)
    service = _service(tmp_path, backend, monkeypatch, enabled=False)
    assert service.verify("f")["status"] == "verified"
    assert "sufficiency" not in _roles(backend)


def test_adapter_methods_skip_the_preaudit(tmp_path, monkeypatch):
    fixtures = _fixtures()
    fixtures["sufficiency"] = {"sufficient": "no", "reason": "must not be asked"}
    fixtures["oracles"] = {"methods": [{
        "procedure": "check the file exists via the registered adapter",
        "artifact": "typed evidence", "level": "exact", "basis": "world",
        "strength": "decisive", "bridges": [],
        "adapter": "filesystem.exists", "adapter_version": 1,
        "adapter_arguments": json.dumps({"address": "/app/answer.txt"}),
        "adapter_inputs": [{"ref": "/app/answer.txt", "provenance": "delivered",
                            "role": "subject"}],
    }]}
    backend = FixtureBackend(fixtures)
    monkeypatch.setenv("GV_TYPED_ADAPTERS", "1")
    monkeypatch.setenv("GV_SUFFICIENCY_PREAUDIT", "1")
    service = _scoped_service(tmp_path, backend, monkeypatch)
    service.verify("f")
    assert "sufficiency" not in _roles(backend)


def test_replacement_method_executor_is_briefed_on_the_rejected_attempt(
        tmp_path, monkeypatch):
    fixtures = _fixtures()
    first = copy.deepcopy(fixtures["oracles"]["methods"][0])
    second = {**first, "procedure": first["procedure"] + " over every page"}
    calls = {"oracle": 0, "audit": 0}

    def oracles(_variables):
        calls["oracle"] += 1
        return {"methods": [first if calls["oracle"] == 1 else second]}

    def sufficiency(_variables):
        calls["audit"] += 1
        if calls["audit"] == 1:
            return {"sufficient": "no", "reason": "one page is not every page"}
        return {"sufficient": "yes", "reason": "the complete scan decides it"}

    fixtures["oracles"] = oracles
    fixtures["sufficiency"] = sufficiency
    backend = FixtureBackend(fixtures)
    service = _service(tmp_path, backend, monkeypatch)
    assert service.verify("f")["status"] == "verified"
    discharge = [c for c in backend.calls if c["role"] == "discharge"]
    assert len(discharge) == 1
    feedback = discharge[0]["variables"]["redo_feedback"]
    assert feedback.startswith("Prior attempt 1 on this obligation used a different method")
    assert "rejected before execution" in feedback
    assert "one page is not every page" in feedback
    # the bound procedure itself is the replacement's, unchanged
    assert discharge[0]["variables"]["procedure"] == second["procedure"]
    briefed = [e["payload"] for e in service.log.replay()
               if e["type"] == "executor_briefed"]
    assert briefed == [{**briefed[0], "method_attempt": 2, "prior_attempts": 1}]
    # the Judger audited the replacement's execution as before
    assert [c["role"] for c in backend.calls].count("judger") >= 1
