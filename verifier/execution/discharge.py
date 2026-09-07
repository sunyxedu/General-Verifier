"""
Discharge: how an obligation actually gets checked.

One seat: AgentDischarger. A Claude Agent SDK agent executes the check
itself and reports outcome/witness plus on-disk artifacts, which we snapshot
into the content-addressed store. The reported outcome is the agent's own
attestation; what audits it is the binding game's judger (which opens the
artifact and may rerun the check), not a mechanical gate.
"""
from __future__ import annotations

import json
import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..llm.agents import DISCHARGE_AGENT, AgentRole
from .instruments import (
    ArtifactStore, authoritative_task_context, world_evidence_context,
    world_map,
)
from ..core.model import (
    BASIS_RECORD, Obligation, OUTCOME_FAILED, STRENGTH_ANALYTIC, STRENGTHS,
)
from ..llm.schema import RoleError


# Text evidence is embedded in ArtifactStore JSON, so this is a decoded
# character budget rather than an on-disk byte budget.  Keep the executor's
# contract mechanically tied to the value used by _snapshot().
SNAPSHOT_PER_FILE_CAP = 65_536


@dataclass(frozen=True)
class DischargeResult:
    outcome: str                 # holds | refuted | failed | unobtainable
    witness: Optional[Any]
    artifact: str                # digest in the artifact store
    instrument: str
    # evidence basis (world | record): what a 'holds' entitles downstream
    # reliance to. Declared by the bound method and audited by the judger;
    # defaults to record — the fail-closed direction — for paths with no
    # declaration (direct discharge, ledger checks).
    basis: str = BASIS_RECORD
    # evidence strength (decisive | bounded | analytic): how decisive the
    # bound method was. None = the ladder was never in play (direct
    # discharge, ledger checks) and reads as decisive downstream —
    # pre-ladder semantics. Where a declaration EXISTS but is absent from
    # a method dict, the binding game defaults it to analytic (fail-closed
    # at the oracle boundary, mirroring basis -> record).
    strength: Optional[str] = None
    # the bound method's declared bridging assumptions (empty for
    # decisive): a supported verdict is only honest with its unverified
    # bridges disclosed, so they ride the evidence onto the certificate
    bridges: tuple = ()
    # premise claim ids whose CONTENT this evidence assumed (e.g. a Lean
    # lemma imported as an axiom instead of re-proved). Legitimate only
    # inside the claim's own dependency closure — the conditional fold
    # already charges those premises to every consumer; anything outside
    # the closure fails closed at the kernel. Empty = the check stands
    # on its own.
    assumed_premises: tuple = ()
    # what a refutation is made of (model.WITNESS_*). None = undeclared,
    # which the kernel reads as NONE: a refutation with no admissible
    # witness settles nothing. Positive verdicts ignore it.
    witness_kind: Optional[str] = None
    # the substantive sources the check consumed, each
    # {"ref", "provenance", "role"} (model.PROVENANCE_* / ROLE_*). The
    # independence axiom is enforced on this declaration: a delivered
    # artifact in the authority role makes a holds circular.
    inputs: tuple = ()


class AgentDischarger:
    """
    Discharge agent executes the check itself; we snapshot its workdir.

    Fail-closed: a RoleError (agent produced no conforming answer even after retries) is a FAILED discharge, on record. Never a pass.
    """

    instrument = "agent:discharge"

    def __init__(self, backend, artifacts: ArtifactStore,
                 role: AgentRole = DISCHARGE_AGENT,
                 workdir_base: Optional[str] = None,
                 cleanup: bool = False,
                 toolchain_dir: Optional[str] = None,
                 state_dir: Optional[str] = None,
                 live_world: bool = True,
                 store=None,
                 shared_cache_dir: Optional[str] = None):
        self.backend = backend
        # verifier-owned install caches shared across trials (the codex
        # backend points the executor's package caches at it and grants it
        # write access); None keeps caches state-local
        self.shared_cache_dir = (os.path.abspath(shared_cache_dir)
                                 if shared_cache_dir else None)
        self.artifacts = artifacts
        self.role = role
        # ClaimStore (duck-typed, optional): lets each discharge see the
        # claim's premises and the prior settled evidence in its dependency
        # closure — the evidence-workspace index. Without it the executor
        # runs contextless, exactly as before.
        self.store = store
        # The verifier's state dir anchors the check's evidence sources:
        # events.jsonl is the ledger, and world/ (when the harness captured
        # one) mirrors the finished run's filesystem root. Obligations are
        # phrased in the run's own paths, so the executor must be handed
        # the mapping explicitly — without it, a path like /app/x gets
        # checked against the scoring machine and refutes spuriously.
        # Resolved per discharge, not here: world/ may be linked later.
        self.state_dir = os.path.abspath(state_dir) if state_dir else None
        self.live_world = live_world
        self.workdir_base = workdir_base or tempfile.mkdtemp(prefix="gv-discharge-")
        # Heavyweight installs (a Lean project with mathlib, a node_modules…)
        # live here and survive across discharges, claims, and runs when the
        # caller passes a stable path. Cache warming without a warming step.
        self.toolchain_dir = toolchain_dir or os.path.join(
            self.workdir_base, "_toolchains")
        # The evidence WORKSPACE: durable building material, full bytes,
        # per task/candidate. Where the Lean project grows, where parsed
        # inputs stay parsed. It survives across discharges and claims and
        # is never cleaned up — the immutable artifact snapshots remain
        # the evidence of record; this layer only makes later checks
        # cheaper, never sound.
        self.workspace_dir = os.path.join(
            self.state_dir if self.state_dir else self.workdir_base,
            "workspace")
        self.cleanup = cleanup  # Remove workdir after snapshot (scratch, not evidence).

    # caps for the evidence-workspace index injected per discharge: the
    # index is a table of contents, not the evidence itself
    INDEX_MAX_PREMISES = 32
    INDEX_MAX_EVIDENCE = 40

    def _workspace_index(self, claim_id: str) -> Dict[str, Any]:
        """Premises assumable by this claim, and prior settled evidence in
        its dependency closure (plus the claim's own already-settled
        sibling obligations), with artifact digests for retrieval.

        Verdicts do NOT transfer through this index — reuse of a verdict
        goes through the kernel's inheritance gates. What transfers here
        is raw material (the artifact files, now durably in the
        workspace) and the RIGHT to assume a listed premise's content by
        declaring its id in assumed_premises.
        """
        empty = {"premises": [], "prior_evidence": []}
        if self.store is None:
            return empty
        try:
            closure = self.store.backward_slice(claim_id)
            final = self.store.get(claim_id)["kind"] == "final"
        except Exception:
            return empty
        premises = []
        if final:
            # a completion claim's contract is adjudicated independently
            # of the solver-derived graph and is exempt from the premise
            # fold — nothing is assumable; prior evidence below remains
            # usable as raw material under re-establishment
            premises = [{"note": "completion claim: no premise is "
                                 "assumable; declare nothing"}]
        for pid in [] if final else closure[1:]:
            if len(premises) >= self.INDEX_MAX_PREMISES:
                premises.append({"note": "...truncated"})
                break
            prow = self.store.get(pid)
            premises.append({
                "id": pid,
                "text": ("[a tool-execution record]"
                         if prow["kind"] == "effect"
                         else prow["text"][:300]),
                "status": prow["status"],
            })
        prior = []
        for cid in closure:  # closure[0] is the claim itself: its settled
            for node in self.store.obligation_children(cid):  # siblings count
                if node["status"] not in ("verified", "refuted"):
                    continue
                ev = self.store.evidence_for(node["id"])
                last = ev[-1] if ev else {}
                prior.append({
                    "obligation": node["text"][:300],
                    "status": node["status"],
                    "basis": node["basis"],
                    "strength": node["strength"],
                    "artifact": last.get("artifact", ""),
                })
                if len(prior) >= self.INDEX_MAX_EVIDENCE:
                    break
            if len(prior) >= self.INDEX_MAX_EVIDENCE:
                prior.append({"note": "...truncated"})
                break
        return {"premises": premises, "prior_evidence": prior}

    def discharge(self, row: Dict, obligation: Obligation,
                  method: Optional[Dict[str, str]] = None,
                  tag: str = "") -> DischargeResult:
        subdir = obligation.id[:12] + (f"-{tag}" if tag else "")
        workdir = os.path.join(self.workdir_base, subdir)
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(self.toolchain_dir, exist_ok=True)
        os.makedirs(self.workspace_dir, exist_ok=True)
        if self.shared_cache_dir:
            os.makedirs(self.shared_cache_dir, exist_ok=True)
        method = method or {
            "procedure": "no bound method (direct discharge, no binding game): "
                         "choose the strongest reachable check yourself, and "
                         "write the check script and its complete output to disk",
        }
        ledger = ""
        if self.state_dir:
            path = os.path.join(self.state_dir, "events.jsonl")
            ledger = path if os.path.exists(path) else ""
        world_mode, world, manifest = world_evidence_context(
            self.state_dir, live_world=self.live_world)
        # Match the binding game's fail-closed interpretation. The executor
        # must see the same tier and bridging assumptions that the independent
        # Judger will audit, including on a fidelity redo or adapter fallback.
        strength = method.get("strength", STRENGTH_ANALYTIC)
        if strength not in STRENGTHS:
            strength = STRENGTH_ANALYTIC
        variables = {
            "claim_text": row["text"],
            "obligation_text": obligation.text,
            "procedure": method["procedure"],
            "adapter_method": self._adapter_method(method),
            # A redo diagnostic is not part of the bound contract. Keeping it
            # separate prevents feedback from silently becoming a new method
            # or an authority for the proposition under test.
            "redo_feedback": method.get("redo_feedback", "(none)"),
            # The executor must see the whole contract the oracle bound,
            # not just its procedure.  Otherwise it cannot know what
            # artifact it promised to produce or which provenance floor its
            # execution has to preserve; the judger would discover those
            # omissions only after an avoidable redo.
            "promised_artifact": method.get("artifact", ""),
            "declared_basis": method.get("basis", BASIS_RECORD),
            "declared_strength": strength,
            "bridges": json.dumps(method.get("bridges", []),
                                  ensure_ascii=False),
            "required_basis": obligation.required_basis,
            "severity": obligation.severity,
            "clause": obligation.clause,
            "refutable_by_absence": obligation.refutable_by_absence,
            "task_context": authoritative_task_context(self.state_dir),
            "workdir": workdir,
            "toolchain_dir": self.toolchain_dir,
            "shared_cache_dir": self.shared_cache_dir or "",
            "workspace_dir": self.workspace_dir,
            "workspace_index": json.dumps(
                self._workspace_index(row["id"]), ensure_ascii=False),
            "ledger_path": ledger,
            "artifact_root": os.path.abspath(self.artifacts.root),
            "world_mode": world_mode,
            "world_path": world,
            "world_manifest": manifest,
            "world_map": world_map(self.state_dir,
                                   live_world=self.live_world),
            "snapshot_per_file_cap": SNAPSHOT_PER_FILE_CAP,
        }
        try:
            payload = self.role.call(self.backend, variables)
        except RoleError as err:
            digest = self.artifacts.put(
                {"instrument": self.instrument, "error": str(err),
                 "files": _snapshot(workdir)}
            )
            self._maybe_cleanup(workdir)
            return DischargeResult(OUTCOME_FAILED, str(err), digest,
                                   self.instrument)

        digest = self.artifacts.put(
            {
                "instrument": self.instrument,
                "reported": payload,
                "files": _snapshot(workdir),
            }
        )
        self._maybe_cleanup(workdir)
        return DischargeResult(
            outcome=payload["outcome"],
            witness=payload["witness"] or None,
            artifact=digest,
            instrument=self.instrument,
            assumed_premises=tuple(payload.get("assumed_premises") or ()),
            witness_kind=payload.get("witness_kind"),
            inputs=tuple(payload.get("inputs") or ()),
        )

    @staticmethod
    def _adapter_method(method: Dict[str, Any]) -> str:
        # Lazy import keeps the legacy executor usable on its own without a
        # module cycle at import time (adapters imports DischargeResult).
        from .adapters import adapter_method_view
        rendered = adapter_method_view(method)
        reason = method.get("adapter_fallback_reason")
        if reason:
            rendered += "\nfast-path unavailable; execute normally: " + str(reason)
        return rendered

    def _maybe_cleanup(self, workdir: str) -> None:
        if self.cleanup:
            import shutil
            shutil.rmtree(workdir, ignore_errors=True)


# Mechanical infra exclusions. Vendored trees and VCS internals are toolchain plumbing, never evidence. 
# (A Lean discharge once vendored all of mathlib into its workdir; the snapshot must survive that.)
SNAPSHOT_SKIP_DIRS = {
    ".git", ".lake", "node_modules", "__pycache__", ".venv", ".cache",
    "build", "dist", "target",
}


def _snapshot(workdir: str, per_file_cap: int = SNAPSHOT_PER_FILE_CAP,
              max_files: int = 200, max_total: int = 2_000_000) -> Dict[str, str]:
    """
    Bounded content snapshot of what the agent left on disk. 
    Caps are mechanical: decoded characters per file, file count, and total
    decoded characters; truncation is marked.
    """
    files: Dict[str, str] = {}
    total = 0
    skipped = 0
    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = sorted(d for d in dirnames if d not in SNAPSHOT_SKIP_DIRS)
        for filename in sorted(filenames):
            if len(files) >= max_files or total >= max_total:
                skipped += 1
                continue
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, workdir)
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as f:
                    content = f.read(per_file_cap + 1)
            except OSError as err:
                files[rel] = f"<unreadable: {err}>"
                continue
            if len(content) > per_file_cap:
                content = content[:per_file_cap] + "\n<truncated>"
            total += len(content)
            files[rel] = content
    if skipped:
        files["<snapshot>"] = f"truncated: {skipped} files beyond caps"
    return files
