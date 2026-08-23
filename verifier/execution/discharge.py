"""
Discharge: how an obligation actually gets checked.

One seat: AgentDischarger. A Claude Agent SDK agent executes the check
itself and reports outcome/witness plus on-disk artifacts, which we snapshot
into the content-addressed store. The reported outcome is the agent's own
attestation; what audits it is the binding game's judger (which opens the
artifact and may rerun the check), not a mechanical gate.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..llm.agents import DISCHARGE_AGENT, AgentRole
from .instruments import (
    ArtifactStore, authoritative_task_context, world_evidence_context,
)
from ..core.model import BASIS_RECORD, Obligation, OUTCOME_FAILED
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
                 live_world: bool = True):
        self.backend = backend
        self.artifacts = artifacts
        self.role = role
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
        self.cleanup = cleanup  # Remove workdir after snapshot (scratch, not evidence).

    def discharge(self, row: Dict, obligation: Obligation,
                  method: Optional[Dict[str, str]] = None,
                  tag: str = "") -> DischargeResult:
        subdir = obligation.id[:12] + (f"-{tag}" if tag else "")
        workdir = os.path.join(self.workdir_base, subdir)
        os.makedirs(workdir, exist_ok=True)
        os.makedirs(self.toolchain_dir, exist_ok=True)
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
        variables = {
            "claim_text": row["text"],
            "obligation_text": obligation.text,
            "procedure": method["procedure"],
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
            "required_basis": obligation.required_basis,
            "task_context": authoritative_task_context(self.state_dir),
            "workdir": workdir,
            "toolchain_dir": self.toolchain_dir,
            "ledger_path": ledger,
            "artifact_root": os.path.abspath(self.artifacts.root),
            "world_mode": world_mode,
            "world_path": world,
            "world_manifest": manifest,
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
        )

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
