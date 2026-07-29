"""Discharge strategies: how an obligation actually gets checked.

Two seats for the same contract:

- SandboxDischarger (M0): a role writes a check program, the kernel runs it
  mechanically in the python sandbox. The verdict comes from deterministic
  execution, so it grades `demonstrated`.

- AgentDischarger: a Claude Agent SDK agent executes the check itself and
  reports outcome/witness plus on-disk artifacts, which we snapshot into the
  content-addressed store. The outcome is agent-REPORTED — generation, not
  mechanics — so it honestly grades `corroborated` until an independent audit
  (M1 闸二) re-checks the artifacts. Grades never overstate provenance.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .agents import DISCHARGE_AGENT, AgentRole
from .instruments import ArtifactStore, PythonSandbox
from .model import Obligation, OUTCOME_FAILED
from .roles import Role, RoleError, WRITE_CHECK


@dataclass(frozen=True)
class DischargeResult:
    outcome: str                 # holds | refuted | failed
    witness: Optional[Any]
    artifact: str                # digest in the artifact store
    instrument: str
    grade: str                   # demonstrated | corroborated


class SandboxDischarger:
    """write_check role -> python sandbox -> mechanical verdict."""

    grade = "demonstrated"

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
            grade=self.grade,
        )


class AgentDischarger:
    """discharge agent executes the check itself; we snapshot its workdir.

    Fail-closed: a RoleError (agent produced no conforming answer even after
    retries) is a FAILED discharge, on record — never a pass.
    """

    grade = "corroborated"
    instrument = "agent:discharge"

    def __init__(self, backend, artifacts: ArtifactStore,
                 role: AgentRole = DISCHARGE_AGENT,
                 workdir_base: Optional[str] = None):
        self.backend = backend
        self.artifacts = artifacts
        self.role = role
        self.workdir_base = workdir_base or tempfile.mkdtemp(prefix="gv-discharge-")

    def discharge(self, row: Dict, obligation: Obligation,
                  method: Optional[Dict[str, str]] = None,
                  tag: str = "") -> DischargeResult:
        subdir = obligation.id[:12] + (f"-{tag}" if tag else "")
        workdir = os.path.join(self.workdir_base, subdir)
        os.makedirs(workdir, exist_ok=True)
        method = method or {
            "procedure": "未绑定方法（oracle 绑定在 M1 接入）：自行选择最强可达的检查手段，"
                         "并把检查脚本与完整输出落盘",
            "instrument_id": "self-selected",
        }
        variables = {
            "claim_text": row["text"],
            "obligation_text": obligation.text,
            "procedure": method["procedure"],
            "instrument_id": method["instrument_id"],
            "workdir": workdir,
        }
        try:
            payload = self.role.call(self.backend, variables)
        except RoleError as err:
            digest = self.artifacts.put(
                {"instrument": self.instrument, "error": str(err),
                 "files": _snapshot(workdir)}
            )
            return DischargeResult(OUTCOME_FAILED, str(err), digest,
                                   self.instrument, self.grade)

        digest = self.artifacts.put(
            {
                "instrument": self.instrument,
                "reported": payload,
                "files": _snapshot(workdir),
            }
        )
        return DischargeResult(
            outcome=payload["outcome"],
            witness=payload["witness"] or None,
            artifact=digest,
            instrument=self.instrument,
            grade=self.grade,
        )


def _snapshot(workdir: str, per_file_cap: int = 65536) -> Dict[str, str]:
    """Content snapshot of everything the agent left on disk (its artifacts)."""
    files: Dict[str, str] = {}
    for dirpath, _dirnames, filenames in os.walk(workdir):
        for filename in sorted(filenames):
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
            files[rel] = content
    return files
