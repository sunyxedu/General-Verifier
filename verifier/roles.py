"""Stateless LLM roles (report: "角色即无状态子调用").

Every role = a generic prompt template + an output schema. Backends are
injectable: FixtureBackend gives deterministic outputs for tests and replay;
AnthropicBackend does real structured-output calls when an API key exists.
Schema validation failures retry once with the error attached, then raise —
a role that cannot produce well-formed output is a FAILED generation, and per
the master principle that must surface, not be papered over.
"""
from __future__ import annotations

import json
import os
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


class AnthropicBackend:
    """Real backend. Imported lazily; requires ANTHROPIC_API_KEY."""

    def __init__(self, model: str = "claude-sonnet-5"):
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise RoleError("ANTHROPIC_API_KEY not set")
        import anthropic  # lazy: not a test-time dependency

        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        message = self.client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in message.content if b.type == "text")
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end < 0:
            raise RoleError(f"{role_name}: no JSON object in model output")
        return json.loads(text[start : end + 1])


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
