"""The compiler's exclusion ledger is reviewed by the Critic and retained."""
from __future__ import annotations

import json

from verifier.engine.certificate import CERTIFICATE_SCOPE_FINAL, request_certificate
from verifier.llm.agents import COMPILE_OBLIGATIONS_AGENT
from verifier.llm.schema import FixtureBackend, validate
from test_certificate_scope import _fixtures, _service


EXCLUDED = [{"clause": "Your response MUST include AT LEAST ONE bash tool call",
             "reason": "runner interface: varying the call history changes no "
                       "task-owned property or acceptance consequence"}]


def test_ledger_is_optional_in_the_schema():
    validate(COMPILE_OBLIGATIONS_AGENT.schema, _fixtures()["obligations"])
    validate(COMPILE_OBLIGATIONS_AGENT.schema,
             {**_fixtures()["obligations"], "excluded": EXCLUDED})


def test_critic_reviews_the_ledger_and_the_event_retains_it(tmp_path, monkeypatch):
    fixtures = _fixtures()
    fixtures["obligations"] = {**fixtures["obligations"], "excluded": EXCLUDED}
    backend = FixtureBackend(fixtures)
    service = _service(tmp_path, backend, monkeypatch)
    certificate = request_certificate(
        service.store, service.kernel, service.log, service.artifacts,
        service.builder.finals[-1], scope=CERTIFICATE_SCOPE_FINAL)
    assert certificate["verdict"] == "verified"
    critic = [c for c in backend.calls if c["role"] == "critic"]
    assert critic and json.loads(critic[0]["variables"]["excluded_json"]) == EXCLUDED
    compiled = [e["payload"] for e in service.log.replay()
                if e["type"] == "obligations_compiled"]
    assert compiled and compiled[-1]["excluded"] == EXCLUDED
    # the ledger never becomes an obligation
    assert all("bash" not in o["text"] for o in compiled[-1]["obligations"])


def test_absent_ledger_reads_as_empty(tmp_path, monkeypatch):
    backend = FixtureBackend(_fixtures())
    service = _service(tmp_path, backend, monkeypatch)
    request_certificate(service.store, service.kernel, service.log,
                        service.artifacts, service.builder.finals[-1],
                        scope=CERTIFICATE_SCOPE_FINAL)
    critic = [c for c in backend.calls if c["role"] == "critic"]
    assert critic and critic[0]["variables"]["excluded_json"] == "[]"
    compiled = [e["payload"] for e in service.log.replay()
                if e["type"] == "obligations_compiled"]
    assert compiled and "excluded" not in compiled[-1]
