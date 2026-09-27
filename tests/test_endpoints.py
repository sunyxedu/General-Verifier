"""The one provider preset table and the resolvers every consumer uses."""
import pytest

from verifier import config
from verifier.config import DirectSettings, ProviderSettings
from verifier.llm import endpoints
from verifier.llm.codex.runtime import CodexRuntime
from verifier.llm.endpoints import presets
from verifier.llm.endpoints.presets import (
    AGENT_PROVIDERS, PROVIDERS, AgentProvider, default_model,
    resolve_agent_provider, resolve_direct_endpoint,
)


def _agent(env):
    return resolve_agent_provider(ProviderSettings.from_env(env))


def test_every_variable_a_preset_names_is_parsed_by_config():
    parsed = set(config.PROVIDER_VALUE_ENVS) | set(config.DIRECT_VALUE_ENVS)
    for preset in PROVIDERS.values():
        for surface in (preset.agent, preset.direct):
            if surface is None:
                continue
            for name in (*surface.key_envs, surface.base_url_env):
                assert name is None or name in parsed, (preset.name, name)
        assert preset.litellm_env is None or preset.litellm_env in parsed


def test_agent_providers_are_the_messages_surfaces():
    assert set(AGENT_PROVIDERS) == {"anthropic", "deepseek", "openrouter",
                                    "custom"}
    for name in ("openai", "codex", "nonsense"):
        with pytest.raises(RuntimeError, match="unknown GV_PROVIDER"):
            _agent({"GV_PROVIDER": name})


def test_agent_defaults_come_from_the_table():
    deepseek = _agent({"DEEPSEEK_API_KEY": "dk"})
    assert (deepseek.name, deepseek.model, deepseek.base_url,
            deepseek.api_key, deepseek.key_env, deepseek.litellm_model,
            deepseek.x_api_key, deepseek.oauth_token) == (
        "deepseek", "deepseek-v4-flash", "https://api.deepseek.com/anthropic",
        "dk", "DEEPSEEK_API_KEY", "deepseek/deepseek-v4-flash", "dk", None)
    openrouter = _agent({"GV_PROVIDER": "OpenRouter",
                         "OPENROUTER_API_KEY": "ok", "GV_MODEL": "x/y"})
    assert openrouter.model == "x/y"
    assert openrouter.litellm_model == "openrouter/x/y"
    assert openrouter.x_api_key == ""       # Bearer-only: blank x-api-key
    anthropic = _agent({"GV_PROVIDER": "anthropic", "ANTHROPIC_API_KEY": "ak",
                        "GV_API_KEY": "gk", "CLAUDE_CODE_OAUTH_TOKEN": "t"})
    assert (anthropic.base_url, anthropic.api_key, anthropic.oauth_token,
            anthropic.model) == (None, "gk", "t", "claude-sonnet-5")
    custom = _agent({"GV_PROVIDER": "custom", "GV_API_KEY": "ck",
                     "GV_ANTHROPIC_BASE_URL": "https://c", "GV_MODEL": "cm"})
    assert (custom.base_url, custom.api_key, custom.litellm_model,
            custom.oauth_token) == ("https://c", "ck", "cm", None)
    assert _agent({"GV_PROVIDER": "custom", "GV_MODEL": "cm",
                   "GV_LITELLM_MODEL": "lm"}).litellm_model == "lm"


def _direct(env, agent_name="deepseek", key="agent-key", model="agent-model",
            base_url="https://agent"):
    agent = AgentProvider(agent_name, model, base_url, key, None, None, key,
                          None)
    return resolve_direct_endpoint(DirectSettings.from_env(env), agent)


def test_direct_endpoint_choice():
    chat = _direct({"GV_DIRECT_PROVIDER": " OpenRouter ",
                    "OPENROUTER_API_KEY": "ork", "GV_DIRECT_EFFORT": "max"})
    assert (chat.protocol, chat.api_key, chat.model, chat.base_url,
            chat.reasoning_effort) == (
        presets.OPENAI_CHAT, "ork", "openai/gpt-5.6-luna", None, "max")
    assert dict(chat.extra_headers)["X-Title"] == "General Verifier"
    # an openrouter agent lends its key to an openrouter direct seat
    assert _direct({}, agent_name="openrouter").api_key == "agent-key"
    assert _direct({"GV_DIRECT_PROVIDER": "openrouter"}) is None
    responses = _direct({"GV_DIRECT_PROVIDER": "openai",
                         "OPENAI_API_KEY": "oak", "GV_DIRECT_MODEL": "dm",
                         "OPENAI_BASE_URL": "https://oa"})
    assert (responses.protocol, responses.model, responses.base_url) == (
        presets.OPENAI_RESPONSES, "dm", "https://oa")
    assert _direct({"GV_DIRECT_PROVIDER": "openai"}) is None
    # otherwise the agent provider's own Messages surface
    messages = _direct({}, agent_name="deepseek")
    assert (messages.protocol, messages.api_key, messages.model,
            messages.base_url, messages.auth_header) == (
        presets.ANTHROPIC_MESSAGES, "agent-key", "agent-model",
        "https://agent", presets.AUTH_X_API_KEY)
    assert _direct({"GV_DIRECT_PROVIDER": "anthropic"},
                   agent_name="openrouter").auth_header == presets.AUTH_BEARER
    assert _direct({}, agent_name="anthropic", key=None) is None  # login only
    assert _direct({}, agent_name="unknown") is None


def test_resolved_specs_open_the_matching_wire():
    wires = {
        presets.ANTHROPIC_MESSAGES: endpoints.AnthropicMessagesEndpoint,
        presets.OPENAI_RESPONSES: endpoints.OpenAIResponsesEndpoint,
        presets.OPENAI_CHAT: endpoints.OpenAIChatEndpoint,
    }
    for protocol, wire in wires.items():
        endpoint = endpoints.open_endpoint(presets.EndpointSpec(
            protocol=protocol, api_key="k", model="m"))
        assert type(endpoint) is wire
        assert endpoint.model == "m" and endpoint.api_key == "k"
        assert endpoint.name.startswith(protocol + "@")


def test_direct_seat_roles_are_parsed_by_config():
    default = frozenset({"a", "b"})
    assert DirectSettings.from_env({}).seat_roles(default) == {"a", "b"}
    assert DirectSettings.from_env(
        {"GV_DIRECT_SEATS": " OFF "}).seat_roles(default) == set()
    assert DirectSettings.from_env(
        {"GV_DIRECT_SEATS": "x, y,"}).seat_roles(default) == {"x", "y"}


def test_process_settings_fall_back_to_the_shipped_dotenv(monkeypatch):
    monkeypatch.delenv("GV_DIRECT_MODEL", raising=False)
    monkeypatch.setattr(config, "_DOTENV", {"GV_DIRECT_MODEL": "from-dotenv"})
    assert DirectSettings.from_env().model == "from-dotenv"
    monkeypatch.setenv("GV_DIRECT_MODEL", "from-env")
    assert DirectSettings.from_env().model == "from-env"
    # an injected mapping stays self-contained
    assert DirectSettings.from_env({}).model is None


def test_codex_consumers_share_the_table_default():
    assert default_model("codex") == PROVIDERS["codex"].agent.default_model
    assert CodexRuntime().model == default_model("codex")
    from verifier import offline
    assert offline.config_fields(seats="codex")["model"] == \
        default_model("codex")
