"""One model and context policy for the solver and all cyber GV roles.

BACKEND selects the CLI family for both the solver and the verifier seats.
The claude backend runs Claude Code on the first-party OAuth login; the codex
backend runs the Codex CLI on the pinned Luna configuration. These are runner
settings, not prompts.
"""

BACKEND = "claude"

if BACKEND == "claude":
    # Claude Code on the first-party login. Observed claude-sonnet-5 window:
    # 1,000,000 tokens. Claude Code auto-compacts and manages its own
    # reasoning budget, so effort/tier/compaction pins do not apply.
    MODEL = "claude-sonnet-5"
    EFFORT = None
    SERVICE_TIER = None
    CONTEXT_WINDOW = 1_000_000
    AUTO_COMPACT_TOKEN_LIMIT = None
    EGRESS_DOMAINS = (".anthropic.com", ".claude.ai")
    MODEL_CONFIG = None
elif BACKEND == "codex":
    # Luna supports a 1,050,000-token total context and at most 922,000 input
    # tokens. Compact before that input ceiling, retaining headroom for a new
    # tool result and reasoning/output.
    MODEL = "gpt-5.6-luna"
    EFFORT = "max"
    SERVICE_TIER = "fast"
    CONTEXT_WINDOW = 1_050_000
    # Pinned Codex reports an 828,400-token usable budget in real Luna turns.
    # Leave 48,400 tokens below that observed budget, not only the API ceiling.
    AUTO_COMPACT_TOKEN_LIMIT = 780_000
    EGRESS_DOMAINS = (".openai.com", ".chatgpt.com", ".oaistatic.com", ".oaiusercontent.com")
    MODEL_CONFIG = f'''web_search = "disabled"
check_for_update_on_startup = false
model = "{MODEL}"
model_reasoning_effort = "{EFFORT}"
service_tier = "{SERVICE_TIER}"
model_context_window = {CONTEXT_WINDOW}
model_auto_compact_token_limit = {AUTO_COMPACT_TOKEN_LIMIT}
[analytics]
enabled = false
[otel]
exporter = "none"
trace_exporter = "none"
metrics_exporter = "none"
'''
else:
    raise RuntimeError(f"unknown cyber model backend: {BACKEND!r} (expected claude | codex)")


def context_cli_args() -> list[str]:
    # Codex CLI overrides also pin continuations and guard against local
    # config drift. Claude Code manages its context window itself.
    if BACKEND != "codex":
        return []
    return ["-c", f"model_context_window={CONTEXT_WINDOW}",
            "-c", f"model_auto_compact_token_limit={AUTO_COMPACT_TOKEN_LIMIT}"]
