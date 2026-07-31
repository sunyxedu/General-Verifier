"""Discharge strategies: how an obligation actually gets checked.

Two seats for the same contract:

- SandboxDischarger (M0): a role writes a check program, the kernel runs it
  mechanically in the python sandbox.

- AgentDischarger: a Claude Agent SDK agent executes the check itself and
  reports outcome/witness plus on-disk artifacts, which we snapshot into the
  content-addressed store. Re-verification (strength.py) independently re-runs
  any extractable check script as a fabrication detector.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ..llm.agents import DISCHARGE_AGENT, AgentRole
from .instruments import ArtifactStore, PythonSandbox
from ..core.model import Obligation, OUTCOME_FAILED
from ..llm.roles import Role, RoleError, WRITE_CHECK


@dataclass(frozen=True)
class DischargeResult:
    outcome: str                 # holds | refuted | failed
    witness: Optional[Any]
    artifact: str                # digest in the artifact store
    instrument: str


class SandboxDischarger:
    """write_check role -> python sandbox -> mechanical verdict."""

    def __init__(self, backend, sandbox: PythonSandbox, artifacts: ArtifactStore,
                 role: Role = WRITE_CHECK):
        self.backend = backend
        self.sandbox = sandbox
        self.artifacts = artifacts
        self.role = role

    def discharge(self, row: Dict, obligation: Obligation,
                  method: Optional[Dict[str, str]] = None,
                  tag: str = "") -> DischargeResult:
        # method/tag are part of the shared discharger contract (used by the
        # binding game); the sandbox path has a fixed instrument, so both are
        # accepted and ignored.
        check = self.role.call(
            self.backend,
            {"claim_text": row["text"], "obligation_text": obligation.text},
        )
        result = self.sandbox.run(check["check_source"])
        digest = self.artifacts.put(result.artifacts)
        return DischargeResult(
            outcome=result.outcome,
            witness=result.witness,
            artifact=digest,
            instrument=self.sandbox.name,
        )


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
            "procedure": "no bound method (oracle binding arrives in M1): "
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
