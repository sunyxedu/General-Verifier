"""Discharge: how an obligation actually gets checked.

One seat: AgentDischarger — a Claude Agent SDK agent executes the check
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
from .instruments import ArtifactStore
from ..core.model import Obligation, OUTCOME_FAILED
from ..llm.roles import RoleError


@dataclass(frozen=True)
class DischargeResult:
    outcome: str                 # holds | refuted | failed | unobtainable
    witness: Optional[Any]
    artifact: str                # digest in the artifact store
    instrument: str


class AgentDischarger:
    """discharge agent executes the check itself; we snapshot its workdir.

    Fail-closed: a RoleError (agent produced no conforming answer even after
    retries) is a FAILED discharge, on record — never a pass.
    """

    instrument = "agent:discharge"

    def __init__(self, backend, artifacts: ArtifactStore,
                 role: AgentRole = DISCHARGE_AGENT,
                 workdir_base: Optional[str] = None,
                 cleanup: bool = False,
                 toolchain_dir: Optional[str] = None):
        self.backend = backend
        self.artifacts = artifacts
        self.role = role
        self.workdir_base = workdir_base or tempfile.mkdtemp(prefix="gv-discharge-")
        # heavyweight installs (a Lean project with mathlib, a node_modules…)
        # live here and survive across discharges, claims, and runs when the
        # caller passes a stable path — cache warming without a warming step
        self.toolchain_dir = toolchain_dir or os.path.join(
            self.workdir_base, "_toolchains")
        self.cleanup = cleanup  # remove workdir after snapshot (scratch, not evidence)

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
        variables = {
            "claim_text": row["text"],
            "obligation_text": obligation.text,
            "procedure": method["procedure"],
            "workdir": workdir,
            "toolchain_dir": self.toolchain_dir,
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


# Mechanical infra exclusions — vendored trees and VCS internals are toolchain
# plumbing, never evidence. (A Lean discharge once vendored all of mathlib into
# its workdir; the snapshot must survive that.)
SNAPSHOT_SKIP_DIRS = {
    ".git", ".lake", "node_modules", "__pycache__", ".venv", ".cache",
    "build", "dist", "target",
}


def _snapshot(workdir: str, per_file_cap: int = 65536,
              max_files: int = 200, max_total: int = 2_000_000) -> Dict[str, str]:
    """Bounded content snapshot of what the agent left on disk. Caps are
    mechanical: per file, file count, and total bytes; truncation is marked,
    never silent."""
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
