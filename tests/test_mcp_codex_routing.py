"""Online Codex seats honor the direct transport configured in .env."""

import sys

from verifier import mcp_server
from verifier.llm import provider
from verifier.llm.codex_backend import CodexBackend
from verifier.llm.direct_backend import OpenAIChatTransport, RoutingBackend


def test_codex_mcp_routes_tool_free_roles_to_dotenv_openrouter(
    monkeypatch, tmp_path,
):
    captured = {}

    class FakeService:
        def __init__(self, state_dir, *, kernel_factory, **kwargs):
            captured["backend"] = kernel_factory(self)

        def start_background(self):
            return False

        def record_start(self, launch):
            captured["launch"] = launch

    class FakeServer:
        def run(self):
            pass

    for name in ("GV_DIRECT_PROVIDER", "GV_DIRECT_MODEL", "GV_DIRECT_EFFORT",
                 "GV_DIRECT_SEATS", "OPENROUTER_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(provider, "_DOTENV", {
        "GV_DIRECT_PROVIDER": "openrouter",
        "GV_DIRECT_MODEL": "openai/gpt-5.6-luna",
        "GV_DIRECT_EFFORT": "max",
        "OPENROUTER_API_KEY": "fixture-key",
    })
    monkeypatch.setattr(mcp_server, "Service", FakeService)
    monkeypatch.setattr(mcp_server, "build_server", lambda service: FakeServer())
    monkeypatch.setattr(mcp_server, "_default_kernel",
                        lambda sc, model, *, backend: backend)
    # the real one ends the process once the server returns
    monkeypatch.setattr(mcp_server, "terminate", lambda service: None)
    monkeypatch.setattr(sys, "argv", [
        "verifier.mcp_server", "--state-dir", str(tmp_path),
        "--seats", "codex", "--model", "gpt-5.6-luna",
        "--effort", "max", "--service-tier", "fast",
    ])

    mcp_server.main()

    backend = captured["backend"]
    assert isinstance(backend, RoutingBackend)
    assert isinstance(backend.agent, CodexBackend)
    assert backend.agent.model == "gpt-5.6-luna"
    assert backend.agent.effort == "max"
    assert backend.agent.service_tier == "fast"
    assert backend.routes_direct("obligations")
    assert backend.routes_direct("critic")
    assert not backend.routes_direct("oracles")
    assert isinstance(backend.direct.transport, OpenAIChatTransport)
    assert backend.direct.transport.model == "openai/gpt-5.6-luna"
    assert backend.direct.transport.reasoning_effort == "max"
    assert backend.direct.transport.api_key == "fixture-key"
