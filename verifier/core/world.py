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
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

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
    # the task's working root as the pre-task baseline recorded it: a bare
    # file name in the task statement is an address relative to it
    task_root: str = ""
    # repositories the pre-task baseline found under the task root, as task
    # addresses: "the repository" in an obligation is one of these
    repositories: Tuple[str, ...] = ()

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
        if self.task_root:
            host = (self.resolve(self.task_root) or self.task_root
                    if self.mode == MODE_CAPTURED else self.task_root)
            out["task_root"] = self.task_root
            out["relative_addresses"] = (
                "a task address given without a directory (a bare file "
                f"name) lives under the task root {self.task_root}"
                + (f", which is {host} on this host" if host != self.task_root
                   else "")
                + "; the verifier's own state directory is never a task "
                "address")
        if self.repositories:
            out["repositories"] = list(self.repositories)
            out["repositories_note"] = (
                "the version-controlled repositories the pre-task baseline "
                "found under the task root; an obligation that speaks of a "
                "repository without an address means one of these — never "
                "assume the task root itself is one")
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
        # every seat and every script runs from a verifier scratch
        # directory, never from a task address: a relative path or a bare
        # `git` invocation there reads the verifier's own state, not the
        # world, and a check written that way "finds" nothing
        out["working_directory"] = (
            "the checking process runs in a verifier scratch directory, "
            "not at any task address; address every world location "
            "absolutely, as the task states it (mapped as above when "
            "captured), never relative to the working directory")
        if self.note:
            out["note"] = self.note
        return out

    def render(self) -> str:
        return json.dumps(self.describe(), ensure_ascii=False, sort_keys=True)


def _task_root(state: str) -> str:
    """The delivered root the pre-task baseline recorded (the task's
    working directory), as a task address; "" when no baseline exists."""
    try:
        with open(os.path.join(state, "baseline", "manifest.json"),
                  encoding="utf-8") as handle:
            root = json.load(handle).get("root") or ""
    except (OSError, ValueError):
        return ""
    return str(root) if root else ""


def _repositories(state: str, task_root: str) -> Tuple[str, ...]:
    try:
        with open(os.path.join(state, "baseline", "repositories.json"),
                  encoding="utf-8") as handle:
            found = json.load(handle) or {}
    except (OSError, ValueError):
        return ()
    if not isinstance(found, dict) or not task_root:
        return ()
    out = []
    for rel in sorted(found):
        if not isinstance(rel, str) or rel.startswith("/") or ".." in rel.split("/"):
            continue
        out.append(task_root if rel in ("", ".")
                   else task_root.rstrip("/") + "/" + rel)
    return tuple(out)


def load_world_view(state_dir: Optional[str],
                    live_world: bool = True) -> WorldView:
    """Build the view for a verifier state dir. A ``world/`` directory
    beside the ledger means captured; its manifest (when present) names
    the scheme and captured addresses. Otherwise live or unavailable."""
    task_root = ""
    repos: Tuple[str, ...] = ()
    if state_dir:
        state = os.path.abspath(state_dir)
        task_root = _task_root(state)
        repos = _repositories(state, task_root)
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
                task_root=task_root,
                repositories=repos,
            )
    return WorldView(mode=MODE_LIVE if live_world else MODE_UNAVAILABLE,
                     task_root=task_root, repositories=repos)


# WORLD EPOCH — a world-basis verdict is about the world as it was when
# the check ran. Every solver action that can change the world (anything
# but a read-only tool) advances the epoch; a world-basis verdict settled
# in an earlier epoch is stale for the current one and must be re-checked
# rather than reused. Record-basis verdicts (about the immutable ledger)
# never go stale this way. The classification is by tool KIND, not task:
# reading, searching, fetching, and the verifier's own verbs leave the
# world alone; execution and editing may not.
READ_ONLY_TOOLS = frozenset({
    "Read", "Grep", "Glob", "LS", "WebSearch", "WebFetch", "ToolSearch",
    "TodoWrite", "TodoRead", "AskUserQuestion",
})


def tool_mutates_world(tool) -> bool:
    name = str(tool or "")
    if name in READ_ONLY_TOOLS:
        return False
    if name.startswith("mcp__gv__"):
        return False
    return True


def world_epoch(events) -> int:
    """The number of world-mutating solver actions in the ledger so far."""
    n = 0
    for event in events:
        if event.get("type") == "tool_call" and tool_mutates_world(
                (event.get("payload") or {}).get("tool")):
            n += 1
    return n
