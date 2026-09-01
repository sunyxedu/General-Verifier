"""
World view: the delivered world as a set of ADDRESSABLE ARTIFACTS, not as
a filesystem.

A task addresses its deliverables in its own scheme — a path on a machine,
a section of a document, a named answer, a cell in a notebook. The
verifier never assumes which. The harness that captured the world tells
it, through world_manifest.json:

    {"scheme": "filesystem",            # how the task addresses artifacts
     "captured_roots": ["/app"],        # addresses whose contents were
                                         # captured (absence outside them
                                         # proves nothing)
     "artifacts": [                     # optional explicit address map
        {"address": "final answer", "host": "final.txt"}],
     "note": "..."}                     # optional free text for the seats

Every seat receives the same rendered description (``describe()``) and
phrases its checks in the task's own addresses; the description says
where each address lives on the scoring host. A legacy manifest carrying
only ``captured_roots`` is a filesystem world whose root is the world/
directory, exactly the previous behaviour.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

MODE_LIVE = "live"
MODE_CAPTURED = "captured"
MODE_UNAVAILABLE = "unavailable"

SCHEME_FILESYSTEM = "filesystem"
SCHEME_DOCUMENT = "document"
SCHEME_ANSWER = "answer"
SCHEME_CUSTOM = "custom"


@dataclass(frozen=True)
class WorldView:
    mode: str                        # live | captured | unavailable
    scheme: str = SCHEME_FILESYSTEM  # the task's own addressing scheme
    host_root: str = ""              # where captured artifacts live
    captured_roots: Tuple[str, ...] = ()
    artifacts: Tuple[Dict[str, str], ...] = ()   # explicit address map
    manifest_path: str = ""
    note: str = ""

    def resolve(self, address: str) -> Optional[str]:
        """Host location of a task address, or None when the address is
        outside every captured root (its absence then proves nothing)."""
        for item in self.artifacts:
            if item.get("address") == address:
                host = item.get("host", "")
                return (host if os.path.isabs(host)
                        else os.path.join(self.host_root, host))
        if self.scheme == SCHEME_FILESYSTEM and self.host_root:
            if not any(address == r or address.startswith(r.rstrip("/") + "/")
                       for r in self.captured_roots):
                return None
            return os.path.join(self.host_root, address.lstrip("/"))
        return None

    def describe(self) -> Dict[str, Any]:
        """The scheme-agnostic description every seat receives."""
        out: Dict[str, Any] = {
            "mode": self.mode,
            "scheme": self.scheme,
            "captured_roots": list(self.captured_roots),
        }
        if self.mode == MODE_CAPTURED:
            out["host_root"] = self.host_root
            if self.scheme == SCHEME_FILESYSTEM:
                out["address_mapping"] = (
                    "a task address A under a captured root lives at "
                    f"{self.host_root}/A on this host; addresses outside "
                    "every captured root were not captured and their "
                    "absence proves nothing")
            if self.artifacts:
                out["artifacts"] = [
                    {"address": a.get("address", ""),
                     "host": self.resolve(a.get("address", "")) or ""}
                    for a in self.artifacts]
        elif self.mode == MODE_LIVE:
            out["address_mapping"] = (
                "the original world is still live: task addresses are "
                "reachable as the task states them")
        else:
            out["address_mapping"] = (
                "the delivered world is gone and was not captured: no "
                "current-state check of any address is possible; only "
                "retained records and verifier-owned computation remain")
        if self.note:
            out["note"] = self.note
        return out

    def render(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, sort_keys=True)


def load_world_view(state_dir: Optional[str],
                    live_world: bool = True) -> WorldView:
    """Build the view for a verifier state dir. A ``world/`` directory
    beside the ledger means captured; its manifest (when present) names
    the scheme and captured addresses. Otherwise live or unavailable."""
    if state_dir:
        state = os.path.abspath(state_dir)
        world = os.path.join(state, "world")
        if os.path.isdir(world):
            manifest_path = os.path.join(state, "world_manifest.json")
            manifest: Dict[str, Any] = {}
            if os.path.exists(manifest_path):
                try:
                    with open(manifest_path, encoding="utf-8") as handle:
                        manifest = json.load(handle) or {}
                except (OSError, ValueError):
                    manifest = {}
            roots = manifest.get("captured_roots")
            if not isinstance(roots, list):
                roots = []
            artifacts = manifest.get("artifacts")
            if not isinstance(artifacts, list):
                artifacts = []
            return WorldView(
                mode=MODE_CAPTURED,
                scheme=str(manifest.get("scheme") or SCHEME_FILESYSTEM),
                host_root=world,
                captured_roots=tuple(str(r) for r in roots),
                artifacts=tuple(a for a in artifacts if isinstance(a, dict)),
                manifest_path=(manifest_path
                               if os.path.exists(manifest_path) else ""),
                note=str(manifest.get("note") or ""),
            )
    return WorldView(mode=MODE_LIVE if live_world else MODE_UNAVAILABLE)
