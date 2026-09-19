"""Claude comparator wiring; no model calls or benchmark-specific fixtures."""
import asyncio
from concurrent.futures import ThreadPoolExecutor

from experiments.cybergym.sidecar import ClaudeVerifierBackend
from verifier.llm import provider
from verifier.llm.agents import AgentBackend, AgentRole, ENUMERATE_ORACLES_AGENT, _ATTEMPT_SINK


def test_claude_verifier_disallows_hosted_search_after_role_expansion(monkeypatch, tmp_path):
    monkeypatch.setattr(provider, "claude_env", lambda base=None: {})
    backend = ClaudeVerifierBackend(cwd=str(tmp_path))
    backend.roles["oracles"] = ENUMERATE_ORACLES_AGENT
    options = backend._build_options(backend.roles["oracles"], lambda _line: None)
    assert "Read" in options.tools
    assert "WebSearch" not in options.tools
    assert "WebFetch" not in options.tools
    assert {"WebSearch", "WebFetch"} <= set(options.disallowed_tools)


def test_claude_verifier_marks_broken_stream_as_incomplete(tmp_path):
    backend = ClaudeVerifierBackend(cwd=str(tmp_path))
    backend._observe_message({"stream_error": True, "observed_messages": 2})
    assert backend.search_policy_guard.snapshot()["evidence_complete"] is False


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
