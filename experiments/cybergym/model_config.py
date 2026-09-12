"""One model and context policy for the solver and all cyber GV roles.

Luna supports a 1,050,000-token total context and at most 922,000 input
tokens. Compact before that input ceiling, retaining headroom for a new
tool result and reasoning/output. These are runner settings, not prompts.
"""

MODEL = "gpt-5.6-luna"
EFFORT = "max"
SERVICE_TIER = "fast"
CONTEXT_WINDOW = 1_050_000
# Pinned Codex reports an 828,400-token usable budget in real Luna turns.
# Leave 48,400 tokens below that observed budget, not only the API ceiling.
AUTO_COMPACT_TOKEN_LIMIT = 780_000


def context_cli_args() -> list[str]:
    # CLI overrides also pin continuations and guard against local config drift.
    return ["-c", f"model_context_window={CONTEXT_WINDOW}",
            "-c", f"model_auto_compact_token_limit={AUTO_COMPACT_TOKEN_LIMIT}"]


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
