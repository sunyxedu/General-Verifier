"""The Codex runtime: verifier seats on the Codex CLI (ChatGPT-subscription
models).

Same runtime interface as ClaudeCodeRuntime / FixtureBackend —
`complete(role_name, prompt, variables) -> payload` — so every seam that
accepts a runtime (kernel seats, the extractor, tests) takes this one
unchanged. Each call is one `codex exec`: the role's system text and
rendered template become the instruction (``command``), and the final
assistant message must be a JSON object matching the role's schema; it is
parsed and re-validated by the same mini-DSL gate as always, so a malformed
generation surfaces as RoleError and retries through AgentRole.call exactly
like the SDK path.

The runtime composes: the run's instruction notes (``command``), the seat's
CLI boundary (``policy``), its image inputs (``images``), the attempts of one
call with their usage ledger rows (``attempts``, ``usage``) and the process
launcher (``process``). ``_run_codex_process`` and ``codex_usage_ledger``
are read from this module by the experiments' attempt observer and scoring
workers, which is why this module names them.

The SDK path enforces the non-executing seats' no-execution palette as a hard
boundary; for the Codex backend it holds only at the prompt level ("you cannot
execute anything"), which is a fidelity gap to keep in mind when comparing
backends.

Fidelity note: codex drives its own agent loop; per-turn budgets and the
exact tool set differ from the SDK seats. Scores from the two backends
are comparable to each other only within a backend.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
import time
import uuid
import weakref
from typing import Any, Dict, Mapping, Optional

from ...config import RunOptions
from ..agents import ALL_AGENT_ROLES
from ..endpoints.presets import default_model
from ..model_spans import model_call_span
from ..policy import NoToolsSession, ToolPolicy
from ..receipts import seat_call_receipt
from ..runtime import RuntimeCapabilities
from ..schema import AuthorityBackendUnavailable, RoleError
from .attempts import CodexInvocation
from .command import (
    RunInstructions, codex_command, image_note, render_instruction,
)
from .images import CapturedImage, CapturedImages
from .paths import (
    _BoundExecutionDirectory,
    _bind_private_empty_directory,
    _canonical_policy_path,
    _path_is_within,
    _revalidate_private_empty_directory,
)
from .policy import (
    LIVE_POLICY_ARGS,
    OfflineProfile,
    is_offline_execution,
    no_tools_policy_args,
    seat_grants,
)
from .process import _run_codex_process
from .service_tier import DEFAULT_SERVICE_TIER, normalize_service_tier
from .stream import _completed_response_without_usage
from .usage import _typed_judger_attempt_fields, _usage_target
# named here for the experiments' scoring workers, which read the usage
# ledger from the module that defines the runtime
from .usage import codex_usage_ledger  # noqa: F401


class _JudgerScratch:
    """The runtime's operator-bound restricted Judger directory."""

    def __init__(self, runtime: "CodexRuntime"):
        self._runtime = runtime
        self.path = runtime.restricted_judger_cwd

    def execution_dir(self) -> str:
        return self._runtime._restricted_judger_execution_cwd()

    def revalidate(self) -> str:
        return self._runtime._revalidate_restricted_judger_cwd()


class CodexRuntime(RuntimeCapabilities):
    """Runs verifier roles through `codex exec` on a subscription model.

    With a ``session`` the seats run under its no-tools policy instead of
    their ordinary palette: the empty scratch directory as cwd, the
    root-deny no-tools profile, no web search or image inputs, a forced
    JSON event stream that the session validates before any answer is
    accepted."""

    RUNTIME_IDENTITY = "verifier.llm.codex_backend.CodexBackend"

    def __init__(self, roles: Optional[Dict[str, Any]] = None,
                 cwd: Optional[str] = None, model: Optional[str] = None,
                 effort: str = "medium", timeout: int = 1800,
                 service_tier: str = DEFAULT_SERVICE_TIER,
                 offline_toolchain_dir: Optional[str] = None,
                 offline_shared_cache_dir: Optional[str] = None,
                 compact_inspection: bool = False,
                 judger_prefetch: bool = False,
                 seat_search: bool = True,
                 restricted_judger_cwd: Optional[str] = None,
                 session: Optional[NoToolsSession] = None):
        RunOptions(compact_inspection=compact_inspection,
                   judger_prefetch=judger_prefetch)   # both must be bools
        self._session = session
        self.roles = dict(roles) if roles is not None else dict(ALL_AGENT_ROLES)
        self.uses_role_wire_schema = True
        self.cwd = cwd
        # Compound model spec: "base[;role=model...]" — e.g.
        # "gpt-5.6-luna;oracles=gpt-5.6-terra" runs every seat on luna
        # except method synthesis. The full spec string is what config
        # fingerprints hash, so mixed-seat runs never share records or
        # derived state with single-model runs.
        spec = model or default_model("codex")
        parts = spec.split(";")
        self.model = parts[0].strip()
        self.model_overrides: Dict[str, str] = {}
        for part in parts[1:]:
            role_name, _, role_model = part.partition("=")
            if role_name.strip() and role_model.strip():
                self.model_overrides[role_name.strip()] = role_model.strip()
        self.effort = effort
        # The knowledge channel is a property of the RUN, not of the task: a
        # benchmark whose problems and answers are published cannot let any
        # seat reach them, so the whole set closes at once rather than per
        # role. Part of the config identity, like every other seat setting.
        self.seat_search = bool(seat_search)
        self.service_tier = normalize_service_tier(service_tier)
        self.timeout = timeout
        self._instructions = RunInstructions(compact_inspection,
                                             judger_prefetch)
        self._offline = OfflineProfile(offline_toolchain_dir,
                                       offline_shared_cache_dir)
        self._restricted_judger_cwd_binding = (
            _bind_private_empty_directory(
                restricted_judger_cwd, "restricted Judger cwd")
            if restricted_judger_cwd is not None else None
        )
        binding = self._restricted_judger_cwd_binding
        self._restricted_judger_cwd_finalizer = (
            weakref.finalize(self, os.close, binding.descriptor)
            if binding is not None else None)
        if binding is not None:
            for candidate in (
                    self.cwd, self.offline_toolchain_dir,
                    self.offline_shared_cache_dir):
                if candidate is None:
                    continue
                candidate_path = _canonical_policy_path(
                    candidate, "restricted Judger disjoint root")
                if (_path_is_within(candidate_path, binding.path)
                        or _path_is_within(binding.path, candidate_path)):
                    raise RoleError(
                        "restricted Judger cwd must be disjoint from every "
                        "candidate or shared writable root")
        self._images = CapturedImages()

    @property
    def offline_toolchain_dir(self) -> Optional[str]:
        return self._offline.toolchain_dir

    @property
    def offline_shared_cache_dir(self) -> Optional[str]:
        return self._offline.shared_cache_dir

    @property
    def restricted_judger_cwd(self) -> Optional[str]:
        binding = self._restricted_judger_cwd_binding
        return binding.path if binding is not None else None

    def _revalidate_restricted_judger_cwd(self) -> Optional[str]:
        binding = self._restricted_judger_cwd_binding
        return (_revalidate_private_empty_directory(
            binding, "restricted Judger cwd") if binding is not None else None)

    def _restricted_judger_execution_cwd(
            self) -> Optional[_BoundExecutionDirectory]:
        binding = self._restricted_judger_cwd_binding
        if binding is None:
            return None
        _revalidate_private_empty_directory(binding, "restricted Judger cwd")
        return _BoundExecutionDirectory(binding.path, binding.descriptor)

    @property
    def session(self) -> Optional[NoToolsSession]:
        """The no-tools session this runtime's seats run under, if any."""
        return self._session

    def judger_scratch(self) -> Optional[_JudgerScratch]:
        if self._restricted_judger_cwd_binding is None:
            return None
        return _JudgerScratch(self)

    @property
    def runtime_instruction_identity(self) -> str:
        identity = self._instructions.identity
        session = self._session
        if session is not None and session.instruction_identity is not None:
            return session.instruction_identity(identity)
        return identity

    @property
    def compact_inspection(self) -> bool:
        return self._instructions.compact_inspection

    @property
    def _judger_prefetch_protocol(self) -> Optional[str]:
        return self._instructions.judger_prefetch_protocol

    @property
    def run_options(self) -> RunOptions:
        return RunOptions(compact_inspection=self.compact_inspection,
                          judger_prefetch=self.judger_prefetch)

    def reuse_identity(self) -> Dict[str, Any]:
        return {"model": self.model, "model_overrides": self.model_overrides,
                "effort": self.effort, "service_tier": self.service_tier,
                "runtime_instruction_identity":
                    self.runtime_instruction_identity}

    def check_service_tier(self, service_tier: str) -> None:
        if self.service_tier != service_tier:
            raise ValueError(
                "provided CodexBackend service_tier does not match the "
                "offline scoring config")

    def supports_policy(self, policy: ToolPolicy) -> bool:
        # the JSON event stream lets a session prove that no tool ran
        return policy.no_tools

    def no_tools_runtime(self, roles: Dict[str, Any],
                         session: NoToolsSession) -> "CodexRuntime":
        """A fresh runtime with this one's seat settings and captured run
        instructions (not potentially changed module defaults), whose seats
        run under ``session``."""
        model_spec = self.model + "".join(
            f";{name}={model}" for name, model in
            sorted(self.model_overrides.items()))
        prefetch = session.evidence_prefetch
        isolated = CodexRuntime(
            roles=roles, cwd=self.cwd, model=model_spec,
            effort=self.effort, timeout=self.timeout,
            service_tier=self.service_tier,
            compact_inspection=self.compact_inspection,
            judger_prefetch=self.judger_prefetch if prefetch else False,
            session=session)
        self._instructions.for_session(isolated._instructions, prefetch)
        return isolated

    def no_tools_identity(self, policy: ToolPolicy,
                          scratch_placeholder: str) -> Dict[str, Any]:
        try:
            version = subprocess.run(
                ["codex", "--version"], check=True, capture_output=True,
                text=True, timeout=10).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise AuthorityBackendUnavailable(
                "Codex runtime identity unavailable") from error
        if not version.startswith("codex-cli ") or len(version) > 100:
            raise AuthorityBackendUnavailable("invalid Codex runtime identity")
        return {"runtime": version,
                "policy": no_tools_policy_args(scratch_placeholder, policy)}

    def allows_compact_source_evidence(
            self, role_name: str, variables: Dict[str, Any]) -> bool:
        """Codex filesystem reads can mutate sealed descriptor metadata."""
        return False

    @property
    def judger_prefetch(self) -> bool:
        return self._instructions.judger_prefetch

    @property
    def typed_judger_transport(self) -> bool:
        return _typed_judger_attempt_fields()["typed_judger_transport"]

    @property
    def typed_judger_protocol(self) -> Optional[str]:
        return _typed_judger_attempt_fields()["typed_judger_protocol"]

    @property
    def typed_judger_packet_sha256(self) -> Optional[str]:
        return _typed_judger_attempt_fields()[
            "typed_judger_packet_sha256"]

    def _execution_cwd(self, role_name: str,
                       variables: Mapping[str, Any]) -> Optional[str]:
        if self._session is not None:
            return self._session.scratch()
        if self._is_offline_execution(role_name, variables):
            return self._offline.state_dir(self.cwd, variables)
        return self.cwd

    def _search_enabled(self, role_name: str) -> bool:
        if self._session is not None:
            return self._session.policy.web_search
        return self.seat_search and seat_grants(role_name).web_search

    def _requires_json(self) -> bool:
        # a session's policy is checked on its event stream
        return self._session is not None

    def _validate_attempt_stream(self, stdout: str) -> None:
        """The session's stream policy (none for ordinary seats)."""
        if self._session is not None:
            self._session.validate_stream(stdout)

    def _validated_stream_response(self, stdout: str) -> Optional[str]:
        """Return a policy-bound final response, if this session has one."""
        session = self._session
        if session is not None and session.bound_response is not None:
            return session.bound_response(stdout)
        return None

    def _recover_completed_response(self, stdout: str) -> Optional[str]:
        """Called only after the role's stream policy has passed."""
        if self._session is not None:
            return self._session.recover_response(stdout)
        return _completed_response_without_usage(stdout)

    def validate_offline_tool_scope(
        self,
        state_dir: str,
        toolchain_dir: Optional[str],
        shared_cache_dir: Optional[str],
    ) -> None:
        """Require Service and backend to share one operator-owned scope."""
        self._offline.validate_scope(
            self.cwd, state_dir, toolchain_dir, shared_cache_dir)

    def captured_images(self, role_name: str,
                        variables: Dict[str, Any]) -> tuple[CapturedImage, ...]:
        if self._session is not None and not self._session.policy.images:
            return ()
        return self._images.collect(role_name, variables, self.cwd)

    def _is_offline_execution(self, role_name: str,
                              variables: Dict[str, Any]) -> bool:
        return is_offline_execution(role_name, variables, self.cwd)

    def _execution_policy_args(self, role_name: str,
                               variables: Dict[str, Any]) -> list[str]:
        """Return the CLI boundary for this seat and evidence context."""
        if self._session is not None:
            return self._session_policy_args()
        if not self._is_offline_execution(role_name, variables):
            return list(LIVE_POLICY_ARGS)
        state_dir = self._offline.state_dir(self.cwd, variables)
        return self._offline.args(role_name, variables, state_dir,
                                  search=self._search_enabled(role_name))

    def _session_policy_args(self) -> list[str]:
        """A session's CLI boundary: its no-tools policy in its scratch."""
        return no_tools_policy_args(
            self._session.scratch(), self._session.policy)

    def _model_for(self, role_name: str) -> str:
        return self.model_overrides.get(role_name, self.model)

    def complete(self, role_name: str, prompt: str,
                 variables: Dict[str, Any]) -> Any:
        role = self.roles.get(role_name)
        if role is None:
            raise RoleError(f"unknown agent role: {role_name}")
        role_model = self._model_for(role_name)
        usage_target = _usage_target(self.cwd)
        call_id = uuid.uuid4().hex
        started = time.time()
        from ...execution.resource_share import take_room_wait
        take_room_wait()
        with seat_call_receipt(
                self.cwd,
                lambda error: self._receipt(
                    role_name, role_model, started, error, variables),
                errors=(BaseException,),
                describe=lambda err: str(err)[:300] or type(err).__name__):
            with model_call_span(
                    call_id, role=role_name, model=role_model):
                return self._complete(
                    role_name, prompt, variables,
                    role=role, role_model=role_model,
                    usage_target=usage_target, call_id=call_id)

    @staticmethod
    def _scratch_watch(role_name: str, variables: Dict[str, Any]) -> Dict[str, Any]:
        """The executor's working directory and its size budget while it
        runs (scratch_budget); other seats write no scratch of their own."""
        from ...execution.scratch_budget import scratch_cap_bytes
        workdir = variables.get("workdir") if role_name == "discharge" else None
        cap = scratch_cap_bytes()
        if not workdir or not cap:
            return {}
        return {"watch_dir": str(workdir), "watch_cap": cap}

    def _receipt(self, role_name: str, model: str, started: float,
                 error: Optional[str], variables: Dict[str, Any]) -> Dict[str, Any]:
        """The receipt row of one codex seat call, beside the direct seats'
        rows: which seat, how long, on which claim and obligation. The
        oracle, executor and judger runs were otherwise untimed, so a
        latency breakdown could only infer them from ledger gaps."""
        from ...execution.resource_share import take_room_wait_detail
        finished = time.time()
        room_wait = take_room_wait_detail()
        return {"role": role_name, "backend": "codex", "model": model,
                "started": started, "finished": finished,
                "seconds": round(finished - started, 3), "error": error,
                # of those, waiting for the envelope's memory (resource_share)
                "room_wait": room_wait["seconds"],
                "room_wait_reason": room_wait["reason"],
                "claim_text": str(variables.get("claim_text", ""))[:120],
                "obligation_text": str(variables.get("obligation_text", ""))[:160]}

    def _complete(self, role_name: str, prompt: str,
                  variables: Dict[str, Any], *, role, role_model: str,
                  usage_target, call_id: str) -> Any:
        images = self.captured_images(role_name, variables)
        packet = None
        image_packet_digest = None
        prefetch_body = ""
        if role_name == "judger" and self.judger_prefetch:
            # Only the trusted BindingGame constructs this private transport
            # object. Never open paths supplied through role/model variables.
            images, packet, image_packet_digest = \
                self._images.attach_judger_packet(
                    variables.get("_judger_evidence_packet"), images)
            if packet is not None:
                prefetch_body = packet.render()
        search = self._search_enabled(role_name)
        instruction = render_instruction(
            role_name, role, prompt, self._instructions, search=search,
            images_note=image_note(
                images, role_name=role_name, role_model=role_model,
                effort=self.effort, service_tier=self.service_tier),
            prefetch_body=prefetch_body)
        attempt_base = {
            "backend": "codex-cli",
            "call_id": call_id,
            "role": role_name,
            "model": role_model,
            "effort": self.effort,
            "service_tier": self.service_tier,
            "seat_search": search,
            "compact_inspection": self.compact_inspection,
            "judger_prefetch": self.judger_prefetch,
            "judger_prefetch_protocol": self._instructions.judger_prefetch_protocol,
            "judger_prefetch_attached": bool(prefetch_body),
            "judger_prefetch_packet_sha256": (
                hashlib.sha256(prefetch_body.encode("utf-8")).hexdigest()
                if prefetch_body else None),
            **_typed_judger_attempt_fields(),
            "runtime_instruction_identity": self.runtime_instruction_identity,
            "state_dir": (
                os.path.realpath(os.path.abspath(self.cwd))
                if self.cwd else None
            ),
            "prompt_sha256": hashlib.sha256(
                instruction.encode("utf-8")).hexdigest(),
        }
        # A session's boundary is its own: an overlay that replaces the
        # ordinary seats' policy method never reaches it.
        policy_args = (self._session_policy_args()
                       if self._session is not None
                       else self._execution_policy_args(role_name, variables))
        # The default offline hook keeps the state root as cwd: the harness
        # contract deliberately exposes ./task.md and ./baseline/ as read-only
        # authoritative inputs. A no-tools session narrows cwd to its scratch.
        execution_cwd = self._execution_cwd(role_name, variables)
        out = tempfile.NamedTemporaryFile(suffix=".txt", delete=False)
        out.close()
        image_paths = []
        call_succeeded = False
        try:
            for item in images:
                suffix = (".jpg" if item.mime == "image/jpeg"
                          else "." + item.mime[6:])
                handle = tempfile.NamedTemporaryFile(
                    prefix="gv-evidence-image-", suffix=suffix, delete=False)
                image_paths.append(handle.name)
                try:
                    handle.write(item.data)
                finally:
                    handle.close()
                os.chmod(handle.name, 0o400)
            cmd, via_stdin = codex_command(
                role_model=role_model, search=search,
                json_stream=(usage_target is not None
                             or self._requires_json()),
                service_tier=self.service_tier, policy_args=policy_args,
                image_paths=image_paths, effort=self.effort,
                output_path=out.name, instruction=instruction)
            invocation = CodexInvocation(
                role_name=role_name, cmd=cmd, cwd=execution_cwd,
                instruction=instruction, via_stdin=via_stdin,
                timeout=self.timeout,
                watch=self._scratch_watch(role_name, variables),
                output_path=out.name, usage_target=usage_target,
                attempt_base=attempt_base,
                validate_stream=self._validate_attempt_stream,
                bound_response=self._validated_stream_response,
                recover_response=self._recover_completed_response)
            # both read here, at call time: the observer seam and a
            # patched sleep take effect on a runtime built before them
            completed = invocation.run(launch=_run_codex_process,
                                       sleep=time.sleep)
            payload = invocation.payload(completed, role)
            call_succeeded = True
            return payload
        finally:
            if image_packet_digest is not None and not call_succeeded:
                self._images.disable_packet(image_packet_digest)
            for path in [out.name, *image_paths]:
                try:
                    os.remove(path)
                except OSError:
                    pass
