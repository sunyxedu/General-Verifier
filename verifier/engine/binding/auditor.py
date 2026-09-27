"""The judging seats of the binding game: the fresh Judger that audits
every substantive result (with the judger evidence packets as pluggable
transports), the script reader, and the optional sufficiency pre-audit."""
from __future__ import annotations

import json
import os
from contextlib import nullcontext
from typing import Dict, Optional

from ...core.model import OUTCOME_FAILED
from ...execution.adapters import adapter_method_view
from ...execution.discharge import ScriptRun
from ...execution.instruments import (
    authoritative_task_context, world_evidence_context, world_map,
)
from ...execution.judger_packets.prefetch import build_judger_evidence_packet
from ...execution.judger_packets.typed import (
    build_typed_adapter_judger_packet,
)
from ...llm.model_spans import model_scope
from ...llm.runtime import capabilities_of
from ...llm.schema import RoleError
from ...llm.typed_judger_backend import restricted_typed_judger_backend
from ..source_evidence import bind_source_evidence_variables
from .methods import method_strength


class _RecordLedger:
    """The ledger a fresh Judger reads: the claim's frozen record."""

    __slots__ = ("path",)

    def __init__(self, path: str):
        self.path = path


class Auditor:
    """The judging seats of one binding game (the ``host``, whose seats,
    runtime, and ledger they use)."""

    def __init__(self, host):
        self.host = host

    def judge(self, obligation, method, result, attempt,
               redo: int = 0,
               degradation: str = "(none — decisive method, no degradation "
                                  "to audit)",
               method_attempt: int = 0,
               independent_judger: bool = False,
               source: Optional[str] = None) -> Dict[str, str]:
        evidence = {
            "outcome": result.outcome,
            "witness": result.witness,
            # the executor's own classification of its refutation witness
            # and of every input it consumed — the judger audits both:
            # a mislabelled absence or a delivered artifact hidden under
            # another provenance class is unfaithful
            "witness_kind": result.witness_kind,
            "inputs": list(result.inputs),
            "artifact_digest": result.artifact,
            "artifact_path": os.path.join(
                self.host.artifacts.root, result.artifact + ".json"
            ),
            # premise claim ids whose content the executor declared it
            # ASSUMED (workspace borrowing without re-establishment) —
            # part of the fidelity audit: undeclared reliance on prior
            # artifacts or premise content is unfaithful
            "assumed_premises": list(result.assumed_premises),
        }
        # same state-dir derivation as the kernel: the judger audits
        # refutations against the delivered world and its capture
        # manifest, so it needs the same paths the executor had
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        world_mode, world, manifest = world_evidence_context(
            state, live_world=self.host.live_world)
        from ...llm import prompts
        variables = {
            "countersign": (prompts.COUNTERSIGN_REFUTATION
                            if source == "refutation countersign" else ""),
            "obligation_text": obligation.text,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "procedure": method["procedure"],
            "adapter_method": adapter_method_view(method),
            "promised_artifact": method.get("artifact", ""),
            # the oracle's declared evidence basis — the judger audits it as
            # part of fidelity, and it is what the settled verdict will carry
            # downstream
            "basis": method.get("basis", ""),
            # the strength ladder's three inputs: sufficiency is judged AT the
            # declared tier, bridges must close the gap, and a non-decisive
            # binding needs the third signature over the degradation record
            "declared_strength": method_strength(method),
            "bridges": json.dumps(method.get("bridges", []),
                                  ensure_ascii=False),
            "degradation_record": degradation,
            "task_context": authoritative_task_context(state),
            "ledger_path": self.host.log.record_path_for(obligation.claim_id),
            "artifact_root": os.path.abspath(self.host.artifacts.root),
            "world_mode": world_mode,
            "world_path": world,
            "world_manifest": manifest,
            "world_map": world_map(state, live_world=self.host.live_world),
            "evidence": json.dumps(evidence, ensure_ascii=False),
            # the artifact's file snapshots are bounded; the complete bytes
            # live in the run's working directory (the artifact's `workdir`)
            # and the shared workspace — a judger that read a cut snapshot
            # as the file itself declared a valid certificate invalid
            "snapshot_per_file_cap": snapshot_per_file_cap(),
            "workspace_dir": self.host.executor.workspace_dir,
        }
        variables = bind_source_evidence_variables(
            self.host.log,
            variables,
            enabled=self.host.run_options.compact_inspection,
        )
        typed_packet = None
        if self.host.typed_adapter_judger_packet:
            # Codex attaches captured images independently of prompt text.
            # A text packet may not stand in for those raw inputs, so probe
            # the same deterministic selector the backend will use and stay
            # on the original path whenever an image is present or the probe
            # itself cannot be established.
            try:
                captured_images_present = bool(capabilities_of(
                    self.host.backend).captured_images("judger", variables))
            except Exception:
                captured_images_present = True
            typed_packet = build_typed_adapter_judger_packet(
                self.host.discharger,
                self.host.artifacts,
                obligation,
                method,
                result,
                degradation,
                captured_images_present=captured_images_present,
            )
        total_calls = self.host.judger_error_retries + 1
        for call_no in range(1, total_calls + 1):
            try:
                # Bind a fresh immutable view for every ordinary Judger call.
                # A failed call may have overlapped a source change, and a
                # large-artifact transport failure must not force later calls
                # back into an avoidable full-artifact dump.
                packet = None
                if self.host.run_options.judger_prefetch:
                    packet = build_judger_evidence_packet(
                        self.host.artifacts,
                        _RecordLedger(
                            self.host.log.record_path_for(obligation.claim_id)),
                        method.get("basis", ""),
                        result.artifact,
                    )
                call_variables = variables
                if packet is not None:
                    call_variables = dict(variables)
                    call_variables["_judger_evidence_packet"] = packet
                if typed_packet is not None and call_no == 1:
                    if call_variables is variables:
                        call_variables = dict(variables)
                    call_variables["_typed_adapter_judger_packet"] = typed_packet
                with model_scope(
                        stage="judger", method_attempt=method_attempt,
                        redo=redo, judger_retry_index=call_no,
                        independent_judger=independent_judger):
                    # For an eligible typed packet, only the exact shipped
                    # Codex backend changes transport: the same independent
                    # Judger runs in a fresh no-tools session. Unsupported
                    # backends keep their established call path. A transport
                    # failure enters this existing retry loop; a semantic
                    # ``no`` verdict returns normally and never falls back.
                    transport = (
                        restricted_typed_judger_backend(self.host.backend)
                        if typed_packet is not None and call_no == 1
                        else nullcontext(self.host.backend)
                    )
                    with transport as judger_backend:
                        try:
                            verdict = self.host.judger_role.call(
                                judger_backend, call_variables)
                        finally:
                            packet_valid = (
                                packet is None or packet.revalidate())
                    if not packet_valid:
                        raise RoleError(
                            "judger evidence prefetch changed during audit")
                    if (typed_packet is not None and call_no == 1
                            and not typed_packet.revalidate()):
                        raise RoleError(
                            "typed-adapter Judger packet changed during audit")
                break
            except RoleError as err:
                will_retry = call_no < total_calls
                # This is the error audit.  Backends retain ownership of usage
                # accounting (CodexRuntime records exact usage before parsing
                # its final JSON), so do not synthesize or duplicate token
                # rows here.
                self.host.log.append(
                    "judger_call_failed",
                    {
                        "obligation": obligation.id,
                        "redo": redo,
                        "procedure": method["procedure"][:80],
                        "artifact_digest": result.artifact,
                        "call": call_no,
                        "max_calls": total_calls,
                        "will_retry": will_retry,
                        "error": str(err)[:500],
                    },
                )
                if will_retry:
                    continue
                # fail-closed; the _infra marker keeps a persistent Judger
                # infrastructure failure from awarding itself an executor
                # redo.  The original digest remains in Attempt and in the
                # final binding-game artifact for later audit/recovery.
                verdict = {
                    # Internal fail-closed sentinel; public role output permits
                    # only yes/no, so neither signature can accidentally pass.
                    "sufficient": "unknown",
                    "faithful": "no",
                    "degradation": "n/a",
                    "reason": f"judger failed: {err}",
                    "_infra": True,
                }
        if redo:
            attempt.redos[-1]["audit"] = verdict
        else:
            attempt.audit = verdict
        public = {k: v for k, v in verdict.items() if not k.startswith("_")}
        self.host.log.append(
            "judger_verdict",
            {"obligation": obligation.id, "redo": redo,
             "procedure": method["procedure"][:80],
             **({"source": source} if source else {}), **public},
        )
        return verdict

    def judge_script(self, obligation, method, run: ScriptRun, attempt,
                      degradation: str) -> Dict[str, str]:
        """The script judger: one no-tools call over the script and its
        retained output. Fail-closed on infrastructure failure, mirroring
        the agent-path judger (no redo is ever awarded on this path)."""
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        world_mode, world, _manifest = world_evidence_context(
            state, live_world=self.host.live_world)
        from ...execution.discharge import SCRIPT_OUTPUT_CAP
        variables = {
            "obligation_text": obligation.text,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "required_basis": obligation.required_basis,
            "task_context": authoritative_task_context(state),
            "procedure": method["procedure"],
            "basis": method.get("basis", ""),
            "declared_strength": method_strength(method),
            "bridges": json.dumps(method.get("bridges", []),
                                  ensure_ascii=False),
            "degradation_record": degradation,
            "world_mode": world_mode,
            "world_path": world,
            "world_map": world_map(state, live_world=self.host.live_world),
            "script": run.script,
            "exit_code": run.exit_code,
            "seconds": run.seconds,
            "timed_out": run.timed_out,
            "output_cap": SCRIPT_OUTPUT_CAP,
            "stdout": run.stdout,
            "stdout_truncated": run.stdout_truncated,
            "stderr": run.stderr,
            "stderr_truncated": run.stderr_truncated,
        }
        total_calls = self.host.judger_error_retries + 1
        for call_no in range(1, total_calls + 1):
            try:
                verdict = dict(self.host.script_judger_role.call(
                    self.host.backend, variables))
                break
            except RoleError as err:
                self.host.log.append(
                    "judger_call_failed",
                    {"obligation": obligation.id, "redo": 0,
                     "procedure": method["procedure"][:80],
                     "artifact_digest": run.artifact,
                     "call": call_no, "max_calls": total_calls,
                     "will_retry": call_no < total_calls,
                     "source": "script", "error": str(err)[:500]})
                if call_no < total_calls:
                    continue
                verdict = {"sufficient": "unknown", "outcome": OUTCOME_FAILED,
                           "witness": "", "degradation": "n/a",
                           "reason": f"script judger failed: {err}",
                           "_infra": True}
        # fidelity is by construction on this path: the script ran verbatim
        verdict["faithful"] = "yes" if verdict.get("sufficient") == "yes" \
            else verdict.get("faithful", "yes")
        attempt.audit = verdict
        public = {k: v for k, v in verdict.items() if not k.startswith("_")}
        self.host.log.append(
            "judger_verdict",
            {"obligation": obligation.id, "redo": 0, "source": "script",
             "procedure": method["procedure"][:80], **public})
        return verdict

    def preaudit(self, row, obligation, method, method_attempt: int):
        """Audit a proposed agent method's sufficiency before execution.

        Typed adapter methods execute in milliseconds and are audited by
        the fresh Judger afterwards, so only agent-executed methods are
        asked.  A seat failure never blocks execution: the established
        post-execution audit still runs.
        """
        if self.host.sufficiency_role is None or "adapter" in method:
            return None
        state = os.path.dirname(os.path.abspath(self.host.log.path))
        world_mode, _world, _manifest = world_evidence_context(
            state, live_world=self.host.live_world)
        variables = {
            "obligation_text": obligation.text,
            "required_basis": obligation.required_basis,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "task_context": authoritative_task_context(state),
            "world_mode": world_mode,
            "world_map": world_map(state, live_world=self.host.live_world),
            "procedure": method.get("procedure", ""),
            "promised_artifact": method.get("artifact", ""),
            "basis": method.get("basis", ""),
            "declared_strength": method_strength(method),
            "bridges": json.dumps(method.get("bridges", []),
                                  ensure_ascii=False),
        }
        try:
            with model_scope(stage="sufficiency_preaudit",
                             method_attempt=method_attempt):
                verdict = self.host.sufficiency_role.call(self.host.backend, variables)
        except RoleError as error:
            self.host.log.append("sufficiency_audited", {
                "claim": row["id"], "obligation": obligation.id,
                "procedure": method.get("procedure", "")[:80],
                "method_attempt": method_attempt,
                "error": str(error)[:300]})
            return None
        self.host.log.append("sufficiency_audited", {
            "claim": row["id"], "obligation": obligation.id,
            "procedure": method.get("procedure", "")[:80],
            "method_attempt": method_attempt,
            "sufficient": verdict["sufficient"],
            "reason": verdict["reason"][:500]})
        return verdict


def snapshot_per_file_cap() -> int:
    from ...execution.discharge import SNAPSHOT_PER_FILE_CAP
    return SNAPSHOT_PER_FILE_CAP
