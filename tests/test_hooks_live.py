"""Live hooks e2e: a REAL Claude Code session (SDK harness) with the repo's
command hooks configured in project settings — tool events flow through the
scripts into the raw feed, and the sidecar ingests them as effect claims.
Opt-in via GV_LIVE=1."""
import json
import os

import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("GV_LIVE") != "1", reason="live SDK test; set GV_LIVE=1"
)

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_live_settings_hooks_feed_the_sidecar(tmp_path):
    import asyncio

    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    feed = str(tmp_path / "state" / "hook_feed.jsonl")
    hook_env = f"GV_HOOK_FEED='{feed}'"
    settings_dir = tmp_path / "proj" / ".claude"
    settings_dir.mkdir(parents=True)
    (settings_dir.parent / "hello.txt").write_text("hello hooks\n")
    settings = {
        "hooks": {
            "PreToolUse": [{"matcher": "*", "hooks": [{
                "type": "command",
                "command": f"{hook_env} python3 "
                           f"'{PROJECT}/hooks/pre_tool_use.py'",
            }]}],
            "PostToolUse": [{"matcher": "*", "hooks": [{
                "type": "command",
                "command": f"{hook_env} python3 "
                           f"'{PROJECT}/hooks/post_tool_use.py'",
            }]}],
        }
    }
    (settings_dir / "settings.json").write_text(json.dumps(settings))

    options = ClaudeAgentOptions(
        model="claude-haiku-4-5",
        cwd=str(settings_dir.parent),
        setting_sources=["project"],       # load the hooks from settings.json
        tools=["Bash", "Read"],
        allowed_tools=["Bash", "Read"],
        permission_mode="bypassPermissions",
        max_turns=6,
        output_format={
            "type": "json_schema",
            "schema": {"type": "object",
                       "properties": {"content": {"type": "string"}},
                       "required": ["content"], "additionalProperties": False},
        },
        max_buffer_size=64 * 1024 * 1024,
    )

    async def run():
        result = None
        async for message in query(
            prompt="用 Bash 运行 `cat hello.txt`，把它的输出原样填入 content。",
            options=options,
        ):
            if isinstance(message, ResultMessage):
                result = message
        return result

    result = asyncio.run(run())
    assert result is not None and result.subtype == "success"
    assert "hello hooks" in result.structured_output["content"]

    # the scripts really fired: raw feed carries call+result lines
    lines = [json.loads(line) for line in open(feed, encoding="utf-8")]
    types = [l["type"] for l in lines]
    assert "tool_call" in types and "tool_result" in types
    bash_calls = [l for l in lines
                  if l["type"] == "tool_call" and l["payload"]["tool"] == "Bash"]
    assert bash_calls, types

    # and the sidecar ingests them into the canonical stream as effects
    from verifier.instruments import ArtifactStore  # noqa: F401 (import check)
    from verifier.kernel import Kernel
    from verifier.roles import FixtureBackend
    from verifier.sidecar import Sidecar

    def factory(sc):
        backend = FixtureBackend({
            "compile_obligations": {"obligations": [
                {"text": "O1", "failure_looks_like": "反例"}]},
            "write_check": {"check_source":
                            'import json\nprint(json.dumps({"holds": True}))'},
        })
        return Kernel(sc.store, sc.log, backend, sc.artifacts)

    sc = Sidecar(str(tmp_path / "state"), kernel_factory=factory)
    ingested = sc.drain_hooks()
    assert ingested >= 2
    effects = [lbl for lbl in sc.status() if lbl.startswith("effect:")]
    assert effects, sc.status()
    out = sc.commit("A", "assertion", "基于真实工具输出", [effects[-1]])
    assert "claim_id" in out
