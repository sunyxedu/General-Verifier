"""Grade ceiling: earned from artifacts, not declared by category.

The ceiling is the maximum grade a discharge result can receive, determined
by the mechanical properties of how the verdict was produced — not by the
method's self-declared evidence type. A method earns its ceiling by
producing an artifact the system can independently verify.

Re-verification bridges agent discharge (untrusted) to sandbox discharge
(trusted): extract the check script the agent wrote, re-run it in a
sandbox, and compare outcomes. A match upgrades the ceiling.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from .discharge import DischargeResult
from .instruments import PythonSandbox
from .model import OUTCOME_FAILED


def ceiling(result: DischargeResult,
            reverified: Optional[DischargeResult] = None) -> str:
    if reverified is not None:
        return reverified.grade
    return result.grade


def reverify(result: DischargeResult,
             artifact_content: dict,
             sandbox: PythonSandbox) -> Optional[DischargeResult]:
    """Try to independently re-run an agent's check script in a sandbox.

    Returns a new DischargeResult with the sandbox's grade if the re-run
    outcome matches the agent's report; None if re-verification could not
    be performed or the outcomes diverge.
    """
    if result.outcome == OUTCOME_FAILED:
        return None
    if result.grade != "corroborated":
        return None

    files = artifact_content.get("files", {})
    script = _find_check_script(files)
    if script is None:
        return None

    try:
        sandbox_result = sandbox.run(script)
    except Exception:
        return None

    if sandbox_result.outcome != result.outcome:
        return None

    return DischargeResult(
        outcome=sandbox_result.outcome,
        witness=sandbox_result.witness,
        artifact=result.artifact,
        instrument="reverify:sandbox",
        grade="demonstrated",
    )


_VERDICT_RE = re.compile(r'["\']holds["\']')


def _find_check_script(files: dict) -> Optional[str]:
    """Find the most likely check script among artifact files."""
    candidates = []
    for filename, content in files.items():
        if not filename.endswith(".py"):
            continue
        if not isinstance(content, str):
            continue
        if _VERDICT_RE.search(content) or "holds" in content:
            candidates.append((filename, content))

    if not candidates:
        return None
    for name, content in candidates:
        if "check" in name.lower():
            return content
    return candidates[0][1]
