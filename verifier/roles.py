"""Stateless LLM roles (report: "角色即无状态子调用").

Every role = a generic prompt template + an output schema. Backends are
injectable: FixtureBackend gives deterministic outputs for tests and replay;
OpenAIBackend does real single-shot calls (OpenAI SDK, provider-agnostic via
base_url — works against OpenAI, Anthropic's compat endpoint, local vLLM…).
Roles that must ACT (probe, execute, open artifacts) live in verifier.agents
on the Claude Agent SDK instead; this module is for pure text-in/JSON-out.
Schema validation failures retry once with the error attached, then raise —
a role that cannot produce well-formed output is a FAILED generation, and per
the master principle that must surface, not be papered over.
"""
from __future__ import annotations

import json
from typing import Any, Callable, Dict, List


class RoleError(RuntimeError):
    pass


# ---- minimal schema validation (no external deps) ----------------------------
# schema: {"field": type} where type in {str, bool, int, float, list, dict},
# {"field": [element_schema]} for a list of dicts, or {"field": ("a", "b")}
# for a closed string enum.

def validate(schema: Dict[str, Any], payload: Any, path: str = "$") -> None:
    if not isinstance(payload, dict):
        raise RoleError(f"{path}: expected object, got {type(payload).__name__}")
    for key, spec in schema.items():
        if key not in payload:
            raise RoleError(f"{path}.{key}: missing")
        value = payload[key]
        if isinstance(spec, tuple):
            if value not in spec:
                raise RoleError(f"{path}.{key}: {value!r} not in {spec}")
        elif isinstance(spec, list):
            if not isinstance(value, list):
                raise RoleError(f"{path}.{key}: expected list")
            for i, item in enumerate(value):
                validate(spec[0], item, f"{path}.{key}[{i}]")
        elif isinstance(spec, dict):
            validate(spec, value, f"{path}.{key}")
        else:
            if not isinstance(value, spec):
                raise RoleError(
                    f"{path}.{key}: expected {spec.__name__}, "
                    f"got {type(value).__name__}"
                )


# ---- backends ----------------------------------------------------------------

class FixtureBackend:
    """Deterministic backend: role name -> payload or callable(vars)->payload."""

    def __init__(self, fixtures: Dict[str, Any]):
        self.fixtures = fixtures
        self.calls: List[Dict[str, Any]] = []

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        self.calls.append({"role": role_name, "variables": variables})
        if role_name not in self.fixtures:
            raise RoleError(f"no fixture for role {role_name}")
        fixture = self.fixtures[role_name]
        return fixture(variables) if callable(fixture) else fixture


class OpenAIBackend:
    """Single-shot backend on the OpenAI SDK — provider-agnostic.

    base_url/api_key select the provider (OpenAI, Anthropic's OpenAI-compat
    endpoint, DeepSeek, a local vLLM, …); the SDK also honors OPENAI_BASE_URL
    and OPENAI_API_KEY from the environment. No response_format is sent, for
    maximum provider compatibility — the JSON is extracted from the text and
    the role-level validate()+retry is the gate, as everywhere else.
    """

    def __init__(self, model: str, base_url: str = None, api_key: str = None,
                 client=None, max_tokens: int = 4096):
        if client is not None:
            self.client = client  # injectable for tests
        else:
            import openai  # lazy: not a test-time dependency

            self.client = openai.OpenAI(
                **({"base_url": base_url} if base_url else {}),
                **({"api_key": api_key} if api_key else {}),
            )
        self.model = model
        self.max_tokens = max_tokens

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=self.max_tokens,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.choices[0].message.content or ""
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise RoleError(f"{role_name}: no JSON object in model output")
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError as err:
            raise RoleError(f"{role_name}: malformed JSON in model output: {err}")


# ---- role --------------------------------------------------------------------

class Role:
    def __init__(self, name: str, template: str, schema: Dict[str, Any]):
        self.name = name
        self.template = template  # generic — never contains domain knowledge
        self.schema = schema

    def render(self, variables: Dict[str, Any]) -> str:
        return self.template.format(**variables)

    def call(self, backend, variables: Dict[str, Any]) -> Dict[str, Any]:
        prompt = self.render(variables)
        last_error = None
        for attempt in range(2):
            payload = backend.complete(self.name, prompt, variables)
            try:
                validate(self.schema, payload)
                return payload
            except RoleError as err:
                last_error = err
                prompt = prompt + f"\n\n[schema error, fix and retry] {err}"
        raise RoleError(f"{self.name}: schema still invalid after retry: {last_error}")


# ---- M0 role definitions (templates are fully generic) -----------------------

COMPILE_OBLIGATIONS = Role(
    name="compile_obligations",
    template=(
        "任务：验证一个 claim（answer-blind：你只看到 claim 的陈述，看不到其论证）。\n"
        "claim 陈述：{claim_text}\n"
        "历史义务模式（检索命中，可为空）：{memory_hits}\n\n"
        "列出\"任何正确答案都必须满足\"的必要条件。每条给出：\n"
        "  text: 义务内容\n"
        "  failure_looks_like: 若义务不成立，失败证据具体长什么样（给不出就留空字符串）\n"
        '输出 JSON: {{"obligations": [{{"text": ..., "failure_looks_like": ...}}]}}'
    ),
    schema={"obligations": [{"text": str, "failure_looks_like": str}]},
)

COVERAGE_PATCH = Role(
    name="coverage_patch",
    template=(
        "一个已确认为假的变体没有被任何现有义务拦下——这是义务集的覆盖洞。\n"
        "原 claim：{claim_text}\n漏网变体：{variant_text}\n"
        "该变体为假的证据：{disproof}\n\n"
        "写出能拦下这一类错误的新义务：\n"
        "  text: 义务内容（必要条件，任何正确答案都必须满足）\n"
        "  failure_looks_like: 该义务不成立时，失败证据具体长什么样\n"
        '输出 JSON: {{"text": ..., "failure_looks_like": ...}}'
    ),
    schema={"text": str, "failure_looks_like": str},
)

WRITE_CHECK = Role(
    name="write_check",
    template=(
        "为下面的义务写一个可执行检查（python）。\n"
        "claim：{claim_text}\n义务：{obligation_text}\n\n"
        "程序必须在最后一行 print 一个 JSON：{{\"holds\": bool, \"witness\": ...}}，\n"
        "holds=false 时 witness 给出具体反例。\n"
        '输出 JSON: {{"check_source": "<python source>"}}'
    ),
    schema={"check_source": str},
)
