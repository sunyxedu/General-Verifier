"""Provider wiring: every Claude Code invocation in this repo runs on the
provider configured in `.env` at the repo root (default: DeepSeek-V4-Flash).

One source of truth. The CLI spawns (experiments/common/solver.py), the
Agent SDK seats (verifier/llm/agents.py), the TurboAgent proxy backend
(experiments/common/turbo.py) and the Harbor container agents
(experiments/terminal_bench/) all take their provider env from here —
experiments import it through the installed `verifier` package; gv_agent.py
(which runs under Harbor's own Python) file-loads it from GV_ROOT.

Configuration (.env, or real environment variables — the real environment
wins; see .env.example for the full reference):

  GV_PROVIDER = deepseek (default) | openrouter | custom
  GV_MODEL    = model id on that provider (defaults below)
  DEEPSEEK_API_KEY / OPENROUTER_API_KEY = the key for the chosen provider
  GV_ANTHROPIC_BASE_URL / GV_OPENAI_BASE_URL / GV_API_KEY / GV_LITELLM_MODEL
              = everything explicit, for GV_PROVIDER=custom

Verified live (2026-08):
  - DeepSeek's Anthropic-compat endpoint accepts both `Authorization:
    Bearer` and `x-api-key`, and maps any claude-* model id server-side to
    deepseek-v4-flash, so alias invocations (`--model sonnet`, Harbor's
    `-m anthropic/sonnet`) land on DeepSeek too. `deepseek-chat` was
    retired 2026-07-24; valid ids are deepseek-v4-flash / deepseek-v4-pro.
  - OpenRouter's Anthropic-compat endpoint is `https://openrouter.ai/api`
    (native, no proxy) — but it authenticates ONLY via Bearer token, so
    ANTHROPIC_API_KEY must be explicitly blank, and Harbor's key
    forwarding (x-api-key) does not carry over: the Terminal-Bench claude
    arms are only guaranteed on deepseek.
"""
import os

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _load_dotenv(path):
    """Minimal .env parser: KEY=VALUE lines, #-comments, optional quotes."""
    conf = {}
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                v = v.strip().strip("'\"")
                conf[k.strip()] = v
    except OSError:
        pass
    return conf


_DOTENV = _load_dotenv(os.path.join(ROOT, ".env"))


def cfg(name, default=None):
    """Real environment > .env file > baked-in default."""
    v = os.environ.get(name)
    if v is None:
        v = _DOTENV.get(name)
    return v if v not in (None, "") else default


# Baked-in fallback so a fresh clone runs with zero setup.
_BAKED_DEEPSEEK_KEY = "***REMOVED-LEAKED-KEY***"

PROVIDER = (cfg("GV_PROVIDER", "deepseek") or "deepseek").lower()

if PROVIDER == "deepseek":
    MODEL = cfg("GV_MODEL", "deepseek-v4-flash")
    ANTHROPIC_BASE_URL = "https://api.deepseek.com/anthropic"
    OPENAI_BASE_URL = "https://api.deepseek.com"
    API_KEY = cfg("DEEPSEEK_API_KEY", _BAKED_DEEPSEEK_KEY)
    KEY_ENV = "DEEPSEEK_API_KEY"          # litellm's env var for this provider
    LITELLM_MODEL = f"deepseek/{MODEL}"
    _X_API_KEY = API_KEY                  # endpoint accepts x-api-key too
elif PROVIDER == "openrouter":
    # model ids are OpenRouter ids, e.g. deepseek/deepseek-v4-flash,
    # qwen/qwen3-coder, moonshotai/kimi-k2 ...
    MODEL = cfg("GV_MODEL", "deepseek/deepseek-v4-flash")
    ANTHROPIC_BASE_URL = "https://openrouter.ai/api"
    OPENAI_BASE_URL = "https://openrouter.ai/api/v1"
    API_KEY = cfg("OPENROUTER_API_KEY")
    if not API_KEY:
        raise RuntimeError("GV_PROVIDER=openrouter needs OPENROUTER_API_KEY "
                           "(in .env or the environment)")
    KEY_ENV = "OPENROUTER_API_KEY"
    LITELLM_MODEL = f"openrouter/{MODEL}"
    _X_API_KEY = ""    # MUST be explicitly blank: Bearer-only auth, and a
                       # non-empty x-api-key makes claude fall back to it
elif PROVIDER == "custom":
    MODEL = cfg("GV_MODEL")
    ANTHROPIC_BASE_URL = cfg("GV_ANTHROPIC_BASE_URL")
    OPENAI_BASE_URL = cfg("GV_OPENAI_BASE_URL")
    API_KEY = cfg("GV_API_KEY")
    if not (MODEL and ANTHROPIC_BASE_URL and API_KEY):
        raise RuntimeError("GV_PROVIDER=custom needs GV_MODEL, "
                           "GV_ANTHROPIC_BASE_URL and GV_API_KEY")
    KEY_ENV = "GV_API_KEY"
    LITELLM_MODEL = cfg("GV_LITELLM_MODEL", MODEL)
    _X_API_KEY = API_KEY
else:
    raise RuntimeError(f"unknown GV_PROVIDER: {PROVIDER!r} "
                       "(expected deepseek | openrouter | custom)")

# Kept for backward reference; prefer API_KEY.
DEEPSEEK_API_KEY = API_KEY if PROVIDER == "deepseek" else \
    cfg("DEEPSEEK_API_KEY", _BAKED_DEEPSEEK_KEY)

# A dedicated config dir for host-side spawns: the user's own ~/.claude may
# hold an Anthropic OAuth login, which the CLI prefers over env credentials —
# with a login present the OAuth token (not the provider key) gets sent to
# the provider endpoint and 401s. A clean config dir has no stored
# credentials, so ANTHROPIC_AUTH_TOKEN is the only one left. (Containers are
# born clean and Harbor sets its own CLAUDE_CONFIG_DIR, so claude_env() adds
# this on the host only — it is deliberately NOT part of CLAUDE_ENV.)
CONFIG_DIR = os.path.expanduser(f"~/.claude-gv-{PROVIDER}")

# Everything a `claude` process needs to run on the provider. The DEFAULT_*
# pins cover alias resolution (sonnet/opus/haiku and the background small
# model) so no code path falls back to an Anthropic model id.
CLAUDE_ENV = {
    "ANTHROPIC_BASE_URL": ANTHROPIC_BASE_URL,
    "ANTHROPIC_AUTH_TOKEN": API_KEY,
    "ANTHROPIC_API_KEY": _X_API_KEY,
    "ANTHROPIC_MODEL": MODEL,
    "ANTHROPIC_SMALL_FAST_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_OPUS_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_SONNET_MODEL": MODEL,
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": MODEL,
    "CLAUDE_CODE_SUBAGENT_MODEL": MODEL,
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",  # no stray Anthropic calls
    "API_TIMEOUT_MS": "3600000",   # effectively no per-request time limit
                                   # (this EXTENDS claude's own default)
    "MCP_TIMEOUT": "600000",       # MCP server startup handshake: claude's
                                   # ~30s default dropped the gv tools on 2/16
                                   # terminal-bench trials under heavy load
    "MCP_TOOL_TIMEOUT": "14400000",  # gv.certificate legitimately runs long
    KEY_ENV: API_KEY,                                 # litellm-style consumers
}


def claude_env(base=None):
    """Full process env for spawning `claude` (or anything that spawns it)
    against the configured provider: the current environment overlaid with
    CLAUDE_ENV, plus a login-free CLAUDE_CONFIG_DIR (see CONFIG_DIR above)."""
    env = dict(os.environ if base is None else base)
    env.update(CLAUDE_ENV)
    os.makedirs(CONFIG_DIR, exist_ok=True)
    env["CLAUDE_CONFIG_DIR"] = CONFIG_DIR
    return env
