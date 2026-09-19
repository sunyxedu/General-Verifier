"""Claude comparator wiring; no model calls or benchmark-specific fixtures."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import pytest

from experiments.cybergym.sidecar import ClaudeVerifierBackend
from verifier.llm import provider
from verifier.llm.agents import AgentBackend, AgentRole, ENUMERATE_ORACLES_AGENT, _ATTEMPT_SINK
from verifier.llm.schema import RoleError, RoleTransportError


def test_claude_verifier_disallows_hosted_search_after_role_expansion(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "claude_env", lambda base=None: {})
    backend = ClaudeVerifierBackend(cwd=str(tmp_path))
    backend.roles["oracles"] = ENUMERATE_ORACLES_AGENT
    options = backend._build_options(backend.roles["oracles"], lambda _line: None)
    assert "Read" in options.tools
    assert "WebSearch" not in options.tools
    assert "WebFetch" not in options.tools
    assert {"WebSearch", "WebFetch"} <= set(options.disallowed_tools)


def test_claude_verifier_keeps_transient_stream_break_recoverable(tmp_path):
    backend = ClaudeVerifierBackend(cwd=str(tmp_path))
    backend._observe_message({"stream_error": True, "observed_messages": 2})
    assert backend.search_policy_guard.snapshot()["evidence_complete"] is True


def test_claude_verifier_latches_only_whole_call_transport_failure(monkeypatch, tmp_path):
    for failure, incomplete in ((RoleError("invalid output"), False),
                                (RoleTransportError("connection lost"), True)):
        backend = ClaudeVerifierBackend(cwd=str(tmp_path))
        def fail(_self, _role_name, _prompt, _variables):
            raise failure
        monkeypatch.setattr(AgentBackend, "complete", fail)
        with pytest.raises(RoleError):
            backend.complete("oracles", "fixture", {})
        assert backend.search_policy_guard.snapshot()["evidence_complete"] is (not incomplete)


def test_concurrent_sdk_roles_share_loop_and_keep_separate_receipts(monkeypatch):
    backend = AgentBackend(roles={"fixture": AgentRole(name="fixture", template="",
                         schema={"answer": str})})
    loops, receipts = [], []

    async def fake_run(_role, prompt):
        loops.append(id(asyncio.get_running_loop()))
        _ATTEMPT_SINK.get().append({"num_turns": 1})
        await asyncio.sleep(0.02)
        return {"answer": prompt}

    monkeypatch.setattr(backend, "run", fake_run)
    monkeypatch.setattr(backend, "_receipt", lambda _role, _started, attempts,
                        _error, variables: receipts.append((variables["id"], len(attempts))))
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = [pool.submit(backend.complete, "fixture", str(index), {"id": index})
                   for index in range(3)]
        assert [future.result() for future in futures] == [
            {"answer": str(index)} for index in range(3)]
    assert len(set(loops)) == 1
    assert sorted(receipts) == [(0, 1), (1, 1), (2, 1)]
