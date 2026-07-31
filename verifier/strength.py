"""Fabrication detector: independent re-run of agent-produced check scripts.

When an agent discharge produces artifacts containing a check script, we
extract and re-run it in a sandbox. If the re-run outcome disagrees with
the agent's report, the discharge is rejected (fabrication detected).
Agreement means the script is at least reproducible — correctness of the
script itself is the entailment audit's job (gate 2).
"""
from __future__ import annotations

import re
from typing import Optional

from .instruments import PythonSandbox


def check_fabrication(artifact_content: dict,
                      reported_outcome: str,
                      sandbox: PythonSandbox) -> Optional[str]:
    """Try to re-run the agent's check script independently.

    Returns None if no fabrication detected (or re-verification not possible).
    Returns a reason string if the re-run outcome disagrees with the agent's
    reported outcome (fabrication detected).
    """
    files = artifact_content.get("files", {})
    script = _find_check_script(files)
    if script is None:
        return None

    try:
        sandbox_result = sandbox.run(script)
    except Exception:
        return None

    if sandbox_result.outcome != reported_outcome:
        return (f"fabrication: agent reported '{reported_outcome}' "
                f"but re-run produced '{sandbox_result.outcome}'")
    return None


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
