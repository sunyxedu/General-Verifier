"""Verifier roles as Claude Agent SDK agents.

Why agents rather than single-shot completions: most verifier roles genuinely
need to act, not just emit text. The oracle enumerator must probe what is
installed; the discharge executor must run the check; the entailment auditor
must open the artifact it is judging. A role that cannot touch the world can
only give an opinion — which is the very thing this system refuses to trust.

Structured output uses the SDK's official mechanism (`output_format` with a
JSON schema): the harness validates the agent's final answer against the
schema and re-prompts on mismatch; the validated object arrives on
`ResultMessage.structured_output`. A run that ends without one is a FAILED
generation and surfaces as RoleError — never papered over. We still run our
own `validate()` on what comes back: the SDK's gate is convenience, ours is
the contract.

Capability boundaries are enforced with `tools` (the tool palette), not
`allowed_tools`: per the SDK docs, allowed_tools only pre-approves calls and
does not restrict the palette — under bypassPermissions an agent could still
reach Bash. The palette is part of a role's identity (the obligation compiler
being unable to execute anything is what makes it answer-blind), so it must
be a hard boundary.
"""
from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .roles import RoleError, validate

try:  # the SDK is optional at import time so the offline kernel still runs
    from claude_agent_sdk import ClaudeAgentOptions, ResultMessage, query

    SDK_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without the SDK
    SDK_AVAILABLE = False


class _TransientSDKError(RuntimeError):
    """Infrastructure failure worth backing off and retrying (internal)."""


def to_json_schema(spec: Any) -> Dict[str, Any]:
    """Mini-DSL (roles.validate) -> JSON Schema draft-07.

    One schema source of truth: the same spec drives the SDK's output gate
    and our own post-hoc validate().  str -> string; (a, b, ...) -> string
    enum; [x] -> array of x; {k: v} -> object with all keys required.
    """
    if spec is str:
        return {"type": "string"}
    if spec is bool:
        return {"type": "boolean"}
    if spec is int:
        return {"type": "integer"}
    if spec is float:
        return {"type": "number"}
    if isinstance(spec, tuple):
        return {"type": "string", "enum": list(spec)}
    if isinstance(spec, list):
        return {"type": "array", "items": to_json_schema(spec[0])}
    if isinstance(spec, dict):
        return {
            "type": "object",
            "properties": {k: to_json_schema(v) for k, v in spec.items()},
            "required": list(spec.keys()),
            "additionalProperties": False,
        }
    raise ValueError(f"unsupported schema spec: {spec!r}")


@dataclass
class AgentRole:
    """A verifier role: generic prompt + output contract + permitted palette."""

    name: str
    template: str                      # generic — never contains domain knowledge
    schema: Dict[str, Any]             # output contract (mini-DSL, see roles.py)
    tools: List[str] = field(default_factory=list)   # hard palette, e.g. ["Bash"]
    model: Optional[str] = None
    max_turns: int = 20
    system: str = ""

    def render(self, variables: Dict[str, Any]) -> str:
        return self.template.format(**variables)

    def call(self, backend, variables: Dict[str, Any]) -> Dict[str, Any]:
        """Same contract as roles.Role.call: two attempts, then RoleError."""
        prompt = self.render(variables)
        last_error: Optional[RoleError] = None
        for _attempt in range(2):
            try:
                payload = backend.complete(self.name, prompt, variables)
                validate(self.schema, payload)
                return payload
            except RoleError as err:
                last_error = err
                prompt = prompt + f"\n\n[上次尝试失败，修正后重来] {err}"
        raise RoleError(f"{self.name}: failed twice: {last_error}")


class AgentBackend:
    """Runs AgentRoles through the Claude Agent SDK.

    One SDK session per call: roles are stateless by construction, which is
    what keeps the auditors independent of the things they audit.

    permission_mode defaults to bypassPermissions because roles run headless
    (a permission prompt would silently deny). This is safe to the extent the
    palette is: restriction lives in `tools`, not in permissions.
    """

    def __init__(
        self,
        roles: Optional[Dict[str, AgentRole]] = None,
        cwd: Optional[str] = None,
        model: Optional[str] = None,
        permission_mode: str = "bypassPermissions",
    ):
        if not SDK_AVAILABLE:
            raise RoleError("claude-agent-sdk is not installed")
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.cwd = cwd
        self.model = model
        self.permission_mode = permission_mode
        self.transcripts: List[Dict[str, Any]] = []

    def complete(self, role_name: str, prompt: str, variables: Dict[str, Any]) -> Any:
        """Backend entry point (same signature as FixtureBackend.complete).

        Loop-safe sync wrapper: callable from plain sync code AND from code
        executing inside a running event loop (e.g. an MCP tool handler in the
        sidecar) — in the latter case the coroutine runs on a private loop in
        a worker thread, because asyncio.run would raise on a running loop.
        """
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run(role, prompt))
        from concurrent.futures import ThreadPoolExecutor

        with ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self.run(role, prompt)).result()

    # statuses worth waiting out (rate limit / overload / gateway); the CLI
    # sometimes reports these only as the opaque "error result: success"
    TRANSIENT_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504, 529}
    transient_delays = (0, 20, 60, 120)   # seconds before each attempt

    async def run(self, role: AgentRole, prompt: str) -> Dict[str, Any]:
        """One role run with backoff on transient infrastructure failures.
        Immediate re-prompts against a rate limit are useless (the Role-level
        retry exists for GENERATION failures); infrastructure failures back
        off and retry here, and only then surface as RoleError."""
        last: Optional[str] = None
        for delay in self.transient_delays:
            if delay:
                await asyncio.sleep(delay)
            try:
                return await self._attempt(role, prompt)
            except _TransientSDKError as err:
                last = str(err)
        raise RoleError(f"{role.name}: transient sdk failure persisted: {last}")

    async def _attempt(self, role: AgentRole, prompt: str) -> Dict[str, Any]:
        stderr_tail: List[str] = []

        def keep_stderr(line: str) -> None:
            stderr_tail.append(line)
            del stderr_tail[:-20]

        options = self._build_options(role, keep_stderr)

        result: Optional[ResultMessage] = None
        error: Optional[Exception] = None
        try:
            async for message in query(prompt=prompt, options=options):
                if isinstance(message, ResultMessage):
                    result = message
        except Exception as err:  # noqa: BLE001 - classified below
            error = err

        self.transcripts.append(
            {
                "role": role.name,
                "session_id": getattr(result, "session_id", None),
                "subtype": getattr(result, "subtype", None),
                "num_turns": getattr(result, "num_turns", None),
                "total_cost_usd": getattr(result, "total_cost_usd", None),
                "usage": getattr(result, "usage", None),
                "api_error_status": getattr(result, "api_error_status", None),
                "stderr_tail": list(stderr_tail),
            }
        )

        if result is not None and result.structured_output is not None \
                and result.subtype == "success":
            payload = result.structured_output
            validate(role.schema, payload)  # our gate, independent of the SDK's
            return payload

        status = getattr(result, "api_error_status", None)
        detail = (
            f"subtype={getattr(result, 'subtype', None)}, "
            f"api_error_status={status}, "
            f"errors={getattr(result, 'errors', None)}, exception={error}"
        )
        if status in self.TRANSIENT_STATUSES:
            raise _TransientSDKError(f"{role.name}: {detail}")
        if status is None and error is not None \
                and "error result: success" in str(error):
            # the CLI's opaque placeholder — usually a rate limit; retrying
            # then failing also covers the non-transient (e.g. auth) case
            raise _TransientSDKError(f"{role.name}: {detail}")
        if error is not None:
            raise RoleError(f"{role.name}: sdk failure: {detail}") from error
        raise RoleError(f"{role.name}: no structured output ({detail})")

    def _build_options(self, role: AgentRole, stderr_cb) -> "ClaudeAgentOptions":
        system_prompt: Any = {"type": "preset", "preset": "claude_code"}
        if role.system:
            system_prompt["append"] = role.system
        return ClaudeAgentOptions(
            system_prompt=system_prompt,
            model=role.model or self.model,
            tools=list(role.tools),            # hard palette (capability boundary)
            allowed_tools=list(role.tools),    # and pre-approved within it
            permission_mode=self.permission_mode,
            max_turns=role.max_turns,
            cwd=self.cwd,
            setting_sources=[],                # hermetic: no ambient project settings
            output_format={
                "type": "json_schema",
                "schema": to_json_schema(role.schema),
            },
            max_buffer_size=64 * 1024 * 1024,  # default 1MB truncates long tool output
            stderr=stderr_cb,
        )


# ---- role definitions --------------------------------------------------------
# Templates stay generic. Domain content arrives at runtime: from the environment
# the agent can probe, from the model's latent knowledge, and from retrieved
# memory — never from text written here.

EVIDENCE_KINDS = ("text", "rederive", "execute", "search", "formal")
OUTCOMES = ("holds", "refuted", "failed")
GRADES_OR_EMPTY = ("proven", "demonstrated", "corroborated", "")

COMPILE_OBLIGATIONS_AGENT = AgentRole(
    name="obligations",
    system=(
        "你是义务编译器。你只看到 claim 的陈述，看不到它的论证（answer-blind）。"
        "你可以读取环境中的文件来了解任务背景，但你不能执行任何东西。"
    ),
    template=(
        "claim 陈述：{claim_text}\n"
        "历史义务模式（检索命中，可为空）：{memory_hits}\n\n"
        "列出\"任何正确答案都必须满足\"的必要条件（义务）。每条义务给出：\n"
        "  text: 义务内容\n"
        "  failure_looks_like: 若该义务不成立，失败证据具体长什么样"
        "（必须指向一个可拿出手的具体 artifact；给不出就留空字符串）。"
    ),
    schema={"obligations": [{"text": str, "failure_looks_like": str}]},
    tools=["Read", "Grep", "Glob"],
)

ENUMERATE_ORACLES_AGENT = AgentRole(
    name="oracles",
    system=(
        "你是 oracle 枚举器。方法想象归你，存在性判断归探测：一种仪器是否就绪，"
        "必须实际探测环境确认（比如用 Bash 检查命令与库是否存在），不许凭印象。"
        "仪器未就绪但可以安装时，直接安装并确认——安装是一次性成本，不构成不可达。"
    ),
    template=(
        "claim：{claim_text}\n"
        "义务：{obligation_text}\n"
        "claim 特征：{features}\n"
        "已知就绪仪器（可为空）：{registry}\n\n"
        "枚举所有可用于检查该义务的方法，按证据强度从强到弱排列。每个方法给出：\n"
        "  procedure: 具体怎么做\n"
        "  instrument_id: 依赖的仪器（探测确认过的）\n"
        "  evidence_kind: 证据类型\n"
        "  artifact: 该方法产出的可复核证据是什么\n"
        "  failure_looks_like: 该方法发现问题时，失败证据长什么样"
    ),
    schema={
        "methods": [
            {
                "procedure": str,
                "instrument_id": str,
                "evidence_kind": EVIDENCE_KINDS,
                "artifact": str,
                "failure_looks_like": str,
            }
        ]
    },
    tools=["Bash", "Read"],
)

GENERATE_VARIANTS_AGENT = AgentRole(
    name="variants",
    system=(
        "你是变体生成器。你生成\"似是而非的错误变体\"，并且必须机械确认每个变体"
        "确实为假（实际运行一个最小检查），否则会冤枉验证手段。确认不了就不要提交它。"
    ),
    template=(
        "原 claim：{claim_text}\n义务：{obligation_text}\n"
        "历史攻击模式（检索命中，可为空）：{memory_hits}\n\n"
        "生成 {n} 个错误变体：与原 claim 结构相似、乍看合理、但确实为假。\n"
        "对每个变体给出：\n"
        "  text: 变体陈述\n"
        "  disproof: 你实际运行检查后确认它为假的证据（命令与输出摘要）"
    ),
    schema={"variants": [{"text": str, "disproof": str}]},
    tools=["Bash", "Write", "Read"],
)

VALIDATE_VARIANT_AGENT = AgentRole(
    name="validate_variant",
    system=(
        "你是考题质检员。一道\"错误变体\"考题把所有候选方法都放行了——要么全体方法"
        "共享盲区，要么这个变体根本不假（生成器错了）。你必须实际重跑检查来确认"
        "它的真假，不许只看文字。拿不准就判 actually_false=yes（宁可冤枉方法，"
        "不可放过盲区）。"
    ),
    template=(
        "原 claim：{claim_text}\n针对的义务：{obligation_text}\n"
        "变体：{variant_text}\n生成器给的假性证据（disproof）：{disproof}\n\n"
        "实际运行检查，确认该变体是否真的为假、且假在违反这条义务上。\n"
        "  actually_false: yes | no\n"
        "  reason: 你的检查过程与结论"
    ),
    schema={"actually_false": ("yes", "no"), "reason": str},
    tools=["Bash", "Read", "Write"],
)

DISCHARGE_AGENT = AgentRole(
    name="discharge",
    system=(
        "你是 verifier 执行器。你按给定的方法执行检查并产出可复核的 artifact。"
        "崩溃、超时、输出不可解析一律如实记为 failed——绝不折算成通过。"
    ),
    template=(
        "claim：{claim_text}\n义务：{obligation_text}\n"
        "绑定的方法：{procedure}\n仪器：{instrument_id}\n工作目录：{workdir}\n"
        "工具链目录：{toolchain_dir}（重量级依赖/项目脚手架装这里，跨检查共享复用；"
        "工作目录只放本次检查的脚本与输出）\n\n"
        "执行该检查。把检查脚本与完整输出落盘到工作目录（它们就是 artifact）。\n"
        "  outcome: holds（义务成立）| refuted（找到反例）| failed（检查本身没跑成）\n"
        "  witness: refuted 时给具体反例，failed 时给失败原因，否则空字符串\n"
        "  artifact: 落盘文件的路径列表（JSON 数组之外用逗号分隔的字符串）"
    ),
    schema={"outcome": OUTCOMES, "witness": str, "artifact": str},
    tools=["Bash", "Read", "Write", "Grep"],
)

ENTAILMENT_AUDIT_AGENT = AgentRole(
    name="entailment",
    system=(
        "你是蕴含审计器（闸二）。你不判断 claim 真假，只判断这份证据在逻辑上"
        "是否解除了这条义务。抽样证据蕴含不了全称结论——这类缺口必须指出。"
        "你必须打开 artifact 亲自核对（必要时重跑校验），不许只读摘要。"
    ),
    template=(
        "义务：{obligation_text}\n声称的失败样例形态：{failure_looks_like}\n"
        "提交的证据（含 artifact 路径）：{evidence}\n\n"
        "核对证据本身，判断它是否在逻辑上蕴含该义务成立。\n"
        "  entails: yes | no\n"
        "  grade: 证据等级（entails=no 时留空字符串）\n"
        "  reason: 理由，指出证据链上任何缺口"
    ),
    schema={"entails": ("yes", "no"), "grade": GRADES_OR_EMPTY, "reason": str},
    tools=["Read", "Bash", "Grep"],
)

ALL_AGENT_ROLES = {
    role.name: role
    for role in (
        COMPILE_OBLIGATIONS_AGENT,
        ENUMERATE_ORACLES_AGENT,
        GENERATE_VARIANTS_AGENT,
        VALIDATE_VARIANT_AGENT,
        DISCHARGE_AGENT,
        ENTAILMENT_AUDIT_AGENT,
    )
}
