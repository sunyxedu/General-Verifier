"""Shared, task-independent model policy for the solver and verifier roles."""

import os

BACKEND = os.environ.get("CYBERGYM_BACKEND", "codex")
if BACKEND == "claude":
    MODEL = "claude-sonnet-5"
    EFFORT = None
    SERVICE_TIER = None
    CONTEXT_WINDOW = 1_000_000
    AUTO_COMPACT_TOKEN_LIMIT = None
    EGRESS_DOMAINS = (".anthropic.com", ".claude.ai", ".claude.com")
    MODEL_CONFIG = None
elif BACKEND == "codex":
    MODEL = "gpt-5.6-luna"
    EFFORT = "max"
    SERVICE_TIER = "fast"
    CONTEXT_WINDOW = 1_050_000
    # Pinned Codex reports an 828,400-token usable budget in real Luna turns.
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
    raise ValueError("CYBERGYM_BACKEND must be codex or claude")


def context_cli_args() -> list[str]:
    # CLI overrides also pin continuations and guard against local config drift.
    if BACKEND != "codex":
        return []
    return ["-c", f"model_context_window={CONTEXT_WINDOW}",
            "-c", f"model_auto_compact_token_limit={AUTO_COMPACT_TOKEN_LIMIT}"]
