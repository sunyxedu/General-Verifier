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
import re
import stat
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

from ..llm.agents import DISCHARGE_AGENT, AgentRole
from ..llm.prompts.discharge import DISCHARGE_REPAIR_CONTEXT
from .instruments import (
    ArtifactStore, authoritative_task_context, image_media_type,
    world_evidence_context, world_map,
)
from ..core.model import (
    BASIS_RECORD, Obligation, OUTCOME_FAILED, STRENGTH_ANALYTIC, STRENGTHS,
    content_hash,
)
from ..core.events import EventLog
from ..engine.source_evidence import bind_source_evidence_variables
from ..llm.schema import RoleError


# Text evidence is embedded in ArtifactStore JSON, so this is a decoded
# character budget rather than an on-disk byte budget.  Keep the executor's
# contract mechanically tied to the value used by _snapshot().
SNAPSHOT_PER_FILE_CAP = 65_536
SNAPSHOT_MAX_FILES = 200
SNAPSHOT_MAX_TOTAL_CHARS = 2_000_000
SNAPSHOT_IMAGE_PER_FILE_BYTES = 20 * 1024 * 1024
SNAPSHOT_IMAGE_MAX_TOTAL_BYTES = 40 * 1024 * 1024
# Repair material is optional. Oversized or invalid retained artifacts simply
# leave the executor on its existing context-free redo path.
REPAIR_ARTIFACT_MAX_BYTES = 16 * 1024 * 1024
SNAPSHOT_MAX_PREFERRED_PATHS = 32
SNAPSHOT_MAX_PREFERRED_PATH_CHARS = 4096


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
                 shared_cache_dir: Optional[str] = None,
                 log: Optional[EventLog] = None):
        self.backend = backend
        self.log = log if type(log) is EventLog else None
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
        return self._discharge(row, obligation, method, tag)

    def discharge_with_repair(self, row: Dict, obligation: Obligation,
                              method=None, tag="", *,
                              previous_artifact=None) -> DischargeResult:
        """Harness-selected previous attempt, not an Oracle method field.

        Nothing is restored automatically and no previous verdict transfers.
        Each call still runs the executor and produces new evidence for the
        independent Judger. Custom dischargers need not support this capability.
        """
        # Inheriting this optional capability must not bypass a downstream
        # subclass/instance that customizes the historical discharge method.
        if getattr(self.discharge, "__func__", None) \
                is not AgentDischarger.discharge:
            return self.discharge(row, obligation, method=method, tag=tag)
        return self._discharge(
            row, obligation, method, tag,
            previous_artifact=previous_artifact)

    def _repair_material(self, digest) -> Optional[Dict[str, str]]:
        """Bounded, local locator for untrusted building material only.

        The binding game selects the digest from this same method attempt's
        execution history. Never interpret model-supplied paths or discover a
        different artifact when that exact artifact is unavailable. Validate
        the canonical JSON digest, not the pretty-printed file's byte hash.
        """
        unavailable = None
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            return unavailable
        if not all(hasattr(os, flag) for flag in (
                "O_DIRECTORY", "O_NOFOLLOW", "O_NONBLOCK")):
            return unavailable
        root = os.path.abspath(self.artifacts.root)
        root_fd = None
        try:
            root_fd = os.open(
                root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            name = digest + ".json"
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            with os.fdopen(os.open(name, flags, dir_fd=root_fd), "rb") as handle:
                before = os.fstat(handle.fileno())
                if (not stat.S_ISREG(before.st_mode)
                        or before.st_size > REPAIR_ARTIFACT_MAX_BYTES):
                    return unavailable
                raw = handle.read(REPAIR_ARTIFACT_MAX_BYTES + 1)
                after = os.fstat(handle.fileno())
                identity = lambda s: (s.st_dev, s.st_ino, s.st_size,
                                      s.st_mtime_ns, s.st_ctime_ns)
                if (len(raw) > REPAIR_ARTIFACT_MAX_BYTES
                        or identity(before) != identity(after)):
                    return unavailable
                payload = json.loads(raw)
                if (not isinstance(payload, dict)
                        or content_hash(payload) != digest):
                    return unavailable
            return {"artifact": digest, "path": os.path.join(root, name)}
        except (OSError, ValueError, TypeError, RecursionError, OverflowError):
            return unavailable
        finally:
            if root_fd is not None:
                os.close(root_fd)

    def _discharge(self, row: Dict, obligation: Obligation,
                   method=None, tag="", *,
                   previous_artifact=None) -> DischargeResult:
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
        if self.log is not None:
            ledger = self.log.record_path_for(row.get("id"))
            if not os.path.exists(ledger):
                ledger = ""
        elif self.state_dir:
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
        repair_material = self._repair_material(previous_artifact)
        variables = {
            "claim_text": row["text"],
            "obligation_text": obligation.text,
            "procedure": method["procedure"],
            "adapter_method": self._adapter_method(method),
            # A redo diagnostic is not part of the bound contract. Keeping it
            # separate prevents feedback from silently becoming a new method
            # or an authority for the proposition under test.
            "redo_feedback": method.get("redo_feedback", "(none)"),
            # Separate harness capability: a method field with this name
            # cannot select an artifact or make an initial execution a redo.
            "redo_repair_context": (DISCHARGE_REPAIR_CONTEXT.format(
                locator=json.dumps(repair_material)) if repair_material else ""),
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
            "snapshot_max_files": SNAPSHOT_MAX_FILES,
            "snapshot_max_total_chars": SNAPSHOT_MAX_TOTAL_CHARS,
        }
        if self.log is not None:
            variables = bind_source_evidence_variables(
                self.log,
                variables,
                enabled=(getattr(self.backend, "compact_inspection", False)
                         is True),
            )
        try:
            payload = self.role.call(self.backend, variables)
        except RoleError as err:
            snapshot = _snapshot_envelope(workdir, self.artifacts)
            digest = self.artifacts.put(
                {"instrument": self.instrument, "error": str(err),
                 **snapshot}
            )
            self._maybe_cleanup(workdir)
            return DischargeResult(OUTCOME_FAILED, str(err), digest,
                                   self.instrument)

        snapshot = _snapshot_envelope(
            workdir, self.artifacts,
            preferred_paths=_reported_artifact_paths(payload, workdir))
        digest = self.artifacts.put({
            "instrument": self.instrument,
            "reported": payload,
            **snapshot,
        })
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


def _safe_preferred_paths(paths: Iterable[str]) -> tuple[str, ...]:
    """Normalize a bounded untrusted path hint list to relative paths."""
    if isinstance(paths, (str, bytes)):
        return ()
    result = []
    seen = set()
    try:
        iterator = iter(paths)
    except TypeError:
        return ()
    for raw in iterator:
        if len(result) >= SNAPSHOT_MAX_PREFERRED_PATHS:
            break
        if (not isinstance(raw, str) or not raw
                or len(raw) > SNAPSHOT_MAX_PREFERRED_PATH_CHARS
                or "\x00" in raw or os.path.isabs(raw)):
            continue
        normalized = os.path.normpath(raw)
        if (normalized in {"", ".", ".."}
                or normalized.startswith(".." + os.sep)
                or normalized in seen):
            continue
        seen.add(normalized)
        result.append(normalized)
    return tuple(result)


def _reported_artifact_paths(payload: Any, workdir: str) -> tuple[str, ...]:
    """Extract optional snapshot-order hints from the executor's report.

    These hints affect retention order only. They never make a path evidence,
    bypass byte/file caps, or change the Judger's independent audit.
    """
    if not isinstance(payload, dict) or not isinstance(
            payload.get("artifact"), str):
        return ()
    root = os.path.abspath(workdir)
    candidates = []
    for item in payload["artifact"].split(","):
        path = item.strip()
        if not path:
            continue
        if os.path.isabs(path):
            try:
                if os.path.commonpath((root, os.path.abspath(path))) != root:
                    continue
                path = os.path.relpath(path, root)
            except ValueError:
                continue
        candidates.append(path)
    return _safe_preferred_paths(candidates)


def _snapshot(
    workdir: str,
    per_file_cap: int = SNAPSHOT_PER_FILE_CAP,
    max_files: int = SNAPSHOT_MAX_FILES,
    max_total: int = SNAPSHOT_MAX_TOTAL_CHARS,
    preferred_paths: Iterable[str] = (),
) -> Dict[str, str]:
    """
    Bounded content snapshot of what the agent left on disk. 
    Caps are mechanical: decoded characters per file, file count, and total
    decoded characters; truncation is marked.
    """
    files: Dict[str, str] = {}
    total = 0
    skipped = 0

    def capture(path: str, rel: str) -> None:
        nonlocal total
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                content = f.read(per_file_cap + 1)
        except OSError as err:
            files[rel] = f"<unreadable: {err}>"
            return
        if len(content) > per_file_cap:
            content = content[:per_file_cap] + "\n<truncated>"
        total += len(content)
        files[rel] = content

    root = os.path.realpath(workdir)
    for rel in _safe_preferred_paths(preferred_paths):
        if len(files) >= max_files or total >= max_total:
            break
        path = os.path.join(workdir, rel)
        try:
            if os.path.commonpath((root, os.path.realpath(path))) != root \
                    or not os.path.isfile(path):
                continue
        except (OSError, ValueError):
            continue
        capture(path, rel)

    for dirpath, dirnames, filenames in os.walk(workdir):
        dirnames[:] = sorted(d for d in dirnames if d not in SNAPSHOT_SKIP_DIRS)
        for filename in sorted(filenames):
            path = os.path.join(dirpath, filename)
            rel = os.path.relpath(path, workdir)
            if rel in files:
                continue
            if len(files) >= max_files or total >= max_total:
                skipped += 1
                continue
            capture(path, rel)
    if skipped:
        files["<snapshot>"] = f"truncated: {skipped} files beyond caps"
    return files


def _snapshot_envelope(
    workdir: str,
    artifacts: Optional[ArtifactStore],
    per_file_cap: int = SNAPSHOT_PER_FILE_CAP,
    max_files: int = SNAPSHOT_MAX_FILES,
    max_total: int = SNAPSHOT_MAX_TOTAL_CHARS,
    preferred_paths: Iterable[str] = (),
) -> Dict[str, Any]:
    """Capture text plus exact, content-addressed raster evidence.

    Binary retention is additive and mechanical.  Header validation controls
    only automatic model attachment: within the byte caps, even an ineligible
    raster is retained losslessly for the ordinary toolful Judger.  A body
    that cannot be retained is recorded as incomplete so no downstream seat
    can mistake a partial body or its digest for semantic evidence.  The
    public ``_snapshot`` wrapper preserves the established text-only helper
    contract.
    """
    # Preserve the established snapshot byte-for-byte, including its handling
    # of text symlinks, hardlinks, replacement decoding, and truncation. Raw
    # image retention is a strictly additive second pass.
    files = _snapshot(
        workdir, per_file_cap=per_file_cap,
        max_files=max_files, max_total=max_total,
        preferred_paths=preferred_paths)
    binary_files: Dict[str, Dict[str, Any]] = {}
    image_total = 0
    for dirpath, dirnames, filenames, directory_fd in os.fwalk(
            workdir, follow_symlinks=False):
        dirnames[:] = sorted(d for d in dirnames if d not in SNAPSHOT_SKIP_DIRS)
        for filename in sorted(filenames):
            rel = os.path.relpath(os.path.join(dirpath, filename), workdir)
            # The additive pass inherits the exact legacy file/count/total
            # boundary: a path omitted by the text snapshot is not promoted
            # back into the envelope merely because it looks like an image.
            if rel not in files:
                continue
            try:
                expected = os.stat(
                    filename, dir_fd=directory_fd, follow_symlinks=False)
                if (not stat.S_ISREG(expected.st_mode)
                        or expected.st_nlink != 1):
                    raise OSError("not a private regular file")
                flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
                         | getattr(os, "O_NONBLOCK", 0))
                descriptor = os.open(filename, flags, dir_fd=directory_fd)
                try:
                    before = os.fstat(descriptor)
                    if ((before.st_dev, before.st_ino)
                            != (expected.st_dev, expected.st_ino)
                            or not stat.S_ISREG(before.st_mode)
                            or before.st_nlink != 1):
                        raise OSError("file changed before snapshot")
                    prefix = os.read(descriptor, 12)
                    media_type = image_media_type(prefix)
                    if media_type is None:
                        continue
                    os.lseek(descriptor, 0, os.SEEK_SET)
                    if (before.st_size > SNAPSHOT_IMAGE_PER_FILE_BYTES
                            or image_total + before.st_size
                            > SNAPSHOT_IMAGE_MAX_TOTAL_BYTES):
                        binary_files[rel] = {
                            "schema": "gv-raw-image-blob-ref-v1",
                            "complete": False,
                            "media_type": media_type,
                            "byte_count": before.st_size,
                            "attachment_eligible": False,
                            "reason": "snapshot image capacity exceeded",
                        }
                        continue
                    chunks = []
                    read_total = 0
                    while True:
                        block = os.read(descriptor, 1024 * 1024)
                        if not block:
                            break
                        chunks.append(block)
                        read_total += len(block)
                        if read_total > SNAPSHOT_IMAGE_PER_FILE_BYTES:
                            raise OSError("snapshot image exceeded its byte cap")
                    raw = b"".join(chunks)
                    after = os.fstat(descriptor)
                    current = os.stat(
                        filename, dir_fd=directory_fd, follow_symlinks=False)
                    identity = lambda value: (
                        value.st_dev, value.st_ino, value.st_mode,
                        value.st_nlink, value.st_size, value.st_mtime_ns,
                        value.st_ctime_ns)
                    if identity(before) != identity(after) \
                            or identity(before) != identity(current):
                        raise OSError("file changed during snapshot")
                finally:
                    os.close(descriptor)
            except (OSError, UnicodeError, ValueError):
                # Binary admission is optional. Never overwrite the legacy
                # text snapshot when stricter no-link/stability checks fail.
                continue
            if artifacts is None:
                continue
            try:
                binary_files[rel] = artifacts.put_image_blob(raw, media_type)
            except ValueError as err:
                # Header/dimension admission governs automatic model
                # decoding, not evidence retention. Keep the exact raw
                # body for the original toolful Judger.
                try:
                    identity = artifacts.put_blob(raw)
                except (OSError, TypeError, ValueError) as store_err:
                    binary_files[rel] = {
                        "schema": "gv-raw-image-blob-ref-v1",
                        "complete": False,
                        "media_type": media_type,
                        "byte_count": len(raw),
                        "attachment_eligible": False,
                        "reason": (
                            "snapshot image retention failed: "
                            + str(store_err)),
                    }
                else:
                    binary_files[rel] = {
                        "schema": "gv-raw-image-blob-ref-v1",
                        "complete": True,
                        **identity,
                        "media_type": media_type,
                        "attachment_eligible": False,
                        "attachment_reason": str(err),
                    }
                    image_total += len(raw)
            except (OSError, TypeError) as err:
                binary_files[rel] = {
                    "schema": "gv-raw-image-blob-ref-v1",
                    "complete": False,
                    "media_type": media_type,
                    "byte_count": len(raw),
                    "attachment_eligible": False,
                    "reason": f"snapshot image retention failed: {err}",
                }
            else:
                image_total += len(raw)
    result: Dict[str, Any] = {"files": files}
    if binary_files:
        result["binary_files"] = binary_files
    return result
