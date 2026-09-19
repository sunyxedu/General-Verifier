"""Claude comparator wiring; no model calls or benchmark-specific fixtures."""
from experiments.cybergym.sidecar import ClaudeVerifierBackend
from verifier.llm import provider
from verifier.llm.agents import ENUMERATE_ORACLES_AGENT


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
