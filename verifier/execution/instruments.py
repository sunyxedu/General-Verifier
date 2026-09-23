"""
Instruments: the mechanical components of the verifier.

Contract (report, fail-closed iron rule): any crash / timeout / unparseable output is a FAILED discharge. Never converted into a pass.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

from ..core.events import EventLog
from ..core.model import content_hash, OUTCOME_HOLDS, OUTCOME_REFUTED, OUTCOME_FAILED


@dataclass(frozen=True)
class RunResult:
    outcome: str                 # holds | refuted | failed
    artifacts: Dict[str, Any]
    witness: Optional[Any] = None


class ArtifactStore:
    """Content-addressed artifact directory."""

    def __init__(self, root: str):
        self.root = root
        os.makedirs(root, exist_ok=True)

    def put(self, payload: Dict[str, Any]) -> str:
        digest = content_hash(payload)
        path = os.path.join(self.root, digest + ".json")
        if not os.path.exists(path):
            # write-then-rename: a concurrent reader (the judger opening an
            # artifact while another seat stores one) never sees a partial
            # file, and two writers of the same digest are idempotent
            tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, sort_keys=True,
                          indent=1)
            os.replace(tmp, path)
        return digest

    def get(self, digest: str) -> Dict[str, Any]:
        with open(os.path.join(self.root, digest + ".json"), "r", encoding="utf-8") as f:
            return json.load(f)

    def put_image_blob(self, raw: bytes, media_type: str) -> Dict[str, Any]:
        """Retain exact raster bytes beside the JSON evidence envelope.

        This is transport only: the raw digest authenticates bytes and never
        asserts what an image depicts.  ``media_type`` must have been derived
        mechanically from a bounded header parsed by
        :func:`validated_image_info`.
        """
        if not isinstance(raw, bytes):
            raise TypeError("image blob must be bytes")
        detected, _width, _height = validated_image_info(raw)
        if detected is None or detected != media_type:
            raise ValueError("image media type does not match its bytes")
        identity = self.put_blob(raw)
        return {
            "schema": "gv-raw-image-blob-ref-v1",
            "complete": True,
            **identity,
            "media_type": media_type,
            "attachment_eligible": True,
        }

    def put_blob(self, raw: bytes) -> Dict[str, Any]:
        """Store opaque bytes without interpreting their semantic content."""
        if not isinstance(raw, bytes):
            raise TypeError("artifact blob must be bytes")
        digest = hashlib.sha256(raw).hexdigest()
        relative = f"blobs/{digest}.blob"
        root_fd, blob_fd, root_identity, blob_identity = \
            _open_blob_directory(self.root)
        name = digest + ".blob"
        temporary = (
            f".{digest}.{os.getpid()}.{threading.get_ident()}."
            f"{time.time_ns()}.tmp")
        temporary_created = False
        try:
            try:
                descriptor = os.open(
                    temporary,
                    os.O_WRONLY | os.O_CREAT | os.O_EXCL
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_CLOEXEC", 0),
                    0o600,
                    dir_fd=blob_fd,
                )
                temporary_created = True
                try:
                    view = memoryview(raw)
                    while view:
                        written = os.write(descriptor, view)
                        if written <= 0:
                            raise OSError("artifact blob write made no progress")
                        view = view[written:]
                    os.fsync(descriptor)
                    os.fchmod(descriptor, 0o400)
                finally:
                    os.close(descriptor)
                try:
                    # link(), unlike rename(), is an atomic publish-if-absent
                    # primitive. Honest concurrent writers have identical
                    # bytes by construction and never replace one another.
                    os.link(
                        temporary, name,
                        src_dir_fd=blob_fd, dst_dir_fd=blob_fd,
                        follow_symlinks=False,
                    )
                except FileExistsError:
                    pass
            finally:
                if temporary_created:
                    try:
                        os.unlink(temporary, dir_fd=blob_fd)
                    except FileNotFoundError:
                        pass
            _validate_blob_at(blob_fd, name, raw)
            _revalidate_blob_directories(
                self.root, root_fd, blob_fd, root_identity, blob_identity)
        finally:
            os.close(blob_fd)
            os.close(root_fd)
        return {
            "sha256": digest,
            "byte_count": len(raw),
            "store_relpath": relative,
        }


_IMAGE_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
MAX_IMAGE_SIDE = 32_768
MAX_IMAGE_PIXELS = 50_000_000


def image_media_type(raw: bytes) -> Optional[str]:
    """Return an allowlisted raster type derived only from magic bytes."""
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw[:4] == b"RIFF" and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def validated_image_info(raw: bytes) -> tuple[str, int, int]:
    """Return a bounded raster header identity or reject the attachment.

    Parsing only fixed-format headers keeps untrusted compressed bodies out of
    in-process decoders.  It is a resource gate, not validation of image
    content and never a semantic observation.
    """
    if not isinstance(raw, bytes):
        raise TypeError("image input must be bytes")
    media_type = image_media_type(raw)
    if media_type is None:
        raise ValueError("unsupported image signature")
    dimensions = {
        "image/png": _png_dimensions,
        "image/jpeg": _jpeg_dimensions,
        "image/gif": _gif_dimensions,
        "image/webp": _webp_dimensions,
    }[media_type](raw)
    if dimensions is None:
        raise ValueError("image header is malformed or incomplete")
    width, height = dimensions
    if (width <= 0 or height <= 0
            or width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE
            or width * height > MAX_IMAGE_PIXELS):
        raise ValueError("image dimensions exceed the attachment cap")
    return media_type, width, height


def _png_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    if (len(raw) < 33 or raw[8:12] != b"\x00\x00\x00\r"
            or raw[12:16] != b"IHDR"):
        return None
    width = int.from_bytes(raw[16:20], "big")
    height = int.from_bytes(raw[20:24], "big")
    bit_depth, color_type = raw[24], raw[25]
    allowed_depths = {
        0: {1, 2, 4, 8, 16},
        2: {8, 16},
        3: {1, 2, 4, 8},
        4: {8, 16},
        6: {8, 16},
    }
    if (color_type not in allowed_depths
            or bit_depth not in allowed_depths[color_type]
            or raw[26] != 0 or raw[27] != 0 or raw[28] not in {0, 1}):
        return None
    return width, height


_JPEG_SOF_MARKERS = frozenset({
    0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
    0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF,
})


def _jpeg_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    position = 2
    while position < len(raw):
        if raw[position] != 0xFF:
            return None
        while position < len(raw) and raw[position] == 0xFF:
            position += 1
        if position >= len(raw):
            return None
        marker = raw[position]
        position += 1
        if marker in {0x01, *range(0xD0, 0xDA)}:
            if marker in {0xD8, 0xD9}:
                return None
            continue
        if position + 2 > len(raw):
            return None
        segment_length = int.from_bytes(raw[position:position + 2], "big")
        if segment_length < 2 or position + segment_length > len(raw):
            return None
        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 7:
                return None
            height = int.from_bytes(raw[position + 3:position + 5], "big")
            width = int.from_bytes(raw[position + 5:position + 7], "big")
            return width, height
        if marker == 0xDA:
            return None
        position += segment_length
    return None


def _gif_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    if len(raw) < 13:
        return None
    return (int.from_bytes(raw[6:8], "little"),
            int.from_bytes(raw[8:10], "little"))


def _webp_dimensions(raw: bytes) -> Optional[tuple[int, int]]:
    if len(raw) < 20:
        return None
    riff_size = int.from_bytes(raw[4:8], "little")
    chunk_type = raw[12:16]
    chunk_size = int.from_bytes(raw[16:20], "little")
    if riff_size + 8 > len(raw) or 20 + chunk_size > len(raw):
        return None
    payload = raw[20:20 + chunk_size]
    if chunk_type == b"VP8X":
        if len(payload) < 10:
            return None
        return (1 + int.from_bytes(payload[4:7], "little"),
                1 + int.from_bytes(payload[7:10], "little"))
    if chunk_type == b"VP8L":
        if len(payload) < 5 or payload[0] != 0x2F:
            return None
        return (1 + payload[1] + ((payload[2] & 0x3F) << 8),
                1 + (payload[2] >> 6) + (payload[3] << 2)
                + ((payload[4] & 0x0F) << 10))
    if chunk_type == b"VP8 ":
        if len(payload) < 10 or payload[3:6] != b"\x9d\x01\x2a":
            return None
        return (int.from_bytes(payload[6:8], "little") & 0x3FFF,
                int.from_bytes(payload[8:10], "little") & 0x3FFF)
    return None


def image_blob_path(root: str, digest: str) -> str:
    """Derive a blob locator without trusting an envelope-supplied path."""
    if _IMAGE_DIGEST.fullmatch(digest or "") is None:
        raise ValueError("invalid image blob identity")
    return os.path.join(os.path.abspath(root), "blobs", digest + ".blob")


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (
        value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid,
    )


def _open_blob_directory(
    root: str,
) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
    """Pin the artifact root and blob directory for one publication."""
    root_path = os.path.abspath(os.fspath(root))
    if os.path.realpath(root_path) != root_path:
        raise ValueError("artifact root is not a real directory")
    expected_root = os.lstat(root_path)
    if not stat.S_ISDIR(expected_root.st_mode):
        raise ValueError("artifact root is not a real directory")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    root_fd = os.open(root_path, directory_flags)
    blob_fd = -1
    try:
        opened_root = os.fstat(root_fd)
        if (_directory_identity(opened_root)
                != _directory_identity(expected_root)):
            raise ValueError("artifact root changed while opening")
        try:
            os.mkdir("blobs", 0o700, dir_fd=root_fd)
        except FileExistsError:
            pass
        expected_blob = os.stat(
            "blobs", dir_fd=root_fd, follow_symlinks=False)
        if (not stat.S_ISDIR(expected_blob.st_mode)
                or stat.S_IMODE(expected_blob.st_mode) & 0o077):
            raise ValueError("artifact blob directory is not private")
        blob_fd = os.open("blobs", directory_flags, dir_fd=root_fd)
        opened_blob = os.fstat(blob_fd)
        if (_directory_identity(opened_blob)
                != _directory_identity(expected_blob)):
            raise ValueError("artifact blob directory changed while opening")
        return (
            root_fd, blob_fd,
            _directory_identity(os.fstat(root_fd)),
            _directory_identity(opened_blob),
        )
    except BaseException:
        if blob_fd >= 0:
            os.close(blob_fd)
        os.close(root_fd)
        raise


def _revalidate_blob_directories(
    root: str,
    root_fd: int,
    blob_fd: int,
    root_identity: tuple[int, ...],
    blob_identity: tuple[int, ...],
) -> None:
    root_path = os.path.abspath(os.fspath(root))
    current_root = os.lstat(root_path)
    current_blob = os.stat("blobs", dir_fd=root_fd, follow_symlinks=False)
    if (not stat.S_ISDIR(current_root.st_mode)
            or not stat.S_ISDIR(current_blob.st_mode)
            or _directory_identity(os.fstat(root_fd)) != root_identity
            or _directory_identity(current_root) != root_identity
            or _directory_identity(os.fstat(blob_fd)) != blob_identity
            or _directory_identity(current_blob) != blob_identity):
        raise ValueError("artifact blob directories changed during publication")


def _validate_blob_at(directory_fd: int, name: str, expected: bytes) -> None:
    flags = (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
             | getattr(os, "O_CLOEXEC", 0))
    path_before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_nlink != 1
                or _directory_identity(before) != _directory_identity(path_before)):
            raise ValueError("artifact blob is not a private regular file")
        chunks = []
        total = 0
        while total <= len(expected):
            block = os.read(
                descriptor, min(64 * 1024, len(expected) + 1 - total))
            if not block:
                break
            chunks.append(block)
            total += len(block)
        actual = b"".join(chunks)
        after = os.fstat(descriptor)
        path_after = os.stat(
            name, dir_fd=directory_fd, follow_symlinks=False)
        file_identity = lambda value: (
            value.st_dev, value.st_ino, value.st_mode, value.st_nlink,
            value.st_uid, value.st_gid, value.st_size,
            value.st_mtime_ns, value.st_ctime_ns)
        if (file_identity(before) != file_identity(after)
                or file_identity(before) != file_identity(path_after)
                or actual != expected
                or hashlib.sha256(actual).hexdigest()
                != hashlib.sha256(expected).hexdigest()):
            raise ValueError("artifact blob differs from its content address")
    finally:
        os.close(descriptor)


def world_evidence_context(
    state_dir: Optional[str], live_world: bool = True,
) -> Tuple[str, str, str]:
    """Return ``(mode, world_path, manifest_path)`` for a verifier seat.

    An empty captured-world path is ambiguous on its own: during an online
    run it means the original world is still live, while in offline replay it
    means that world is gone and was not captured.  Keep that distinction
    explicit at every semantic seat instead of asking an LLM to infer it from
    a missing path.
    """
    if state_dir:
        state = os.path.abspath(state_dir)
        world = os.path.join(state, "world")
        if os.path.isdir(world):
            manifest = os.path.join(state, "world_manifest.json")
            return (
                "captured",
                world,
                manifest if os.path.exists(manifest) else "",
            )
    return ("live", "", "") if live_world else ("unavailable", "", "")


def world_map(state_dir: Optional[str], live_world: bool = True) -> str:
    """The scheme-agnostic world description every seat receives (see
    core.world.WorldView): the task's own addressing scheme, which
    addresses were captured, and where they live on this host."""
    from ..core.world import load_world_view
    return load_world_view(state_dir, live_world=live_world).render()


def authoritative_task_context(state_dir: Optional[str]) -> str:
    """Inline harness task truth so semantic seats cannot skip a Read."""
    if not state_dir:
        return "(none supplied)"
    task = os.path.join(os.path.abspath(state_dir), "task.md")
    if not os.path.isfile(task):
        return "(none supplied)"
    with open(task, encoding="utf-8") as handle:
        text = handle.read()
    context = {"path": os.path.abspath(task), "text": text,
               # An executor that met the statement's path inside the
               # verifier's state went looking for the statement at a world
               # address, declared the authority absent there, and then
               # wrote a copy into the delivery. The text is supplied here
               # in full; the path is only its identity.
               "statement_note": (
                   "The text above is the complete authoritative task "
                   "statement, supplied by the harness; the path is its "
                   "identity, not a location to read. It is the authority "
                   "wherever the task is referenced. It is not an artifact "
                   "of the delivered world: never look for it there, and "
                   "never write a copy of it anywhere.")}
    # The candidate response is a fact about the harness, supplied as data
    # so every seat reads the same one. Offline the solver's reply is on
    # disk (final.txt) and is the carrier of a response-borne answer.
    # Online the solver is certified BEFORE it writes its reply, so the
    # final claim's text is the candidate response; without this every
    # seat that read the task's reply-layout instruction went looking for
    # a reply in the world, found the verifier's own claim store, and held
    # the claim to the reply's grammar.
    final_txt = os.path.join(os.path.abspath(state_dir), "final.txt")
    if os.path.isfile(final_txt):
        # named relative to the state directory: a replay clone of the same
        # sealed inputs must produce the same compile input
        context["candidate_response"] = {
            "delivered": True, "path": "final.txt",
            "note": "the solver's reply as delivered; a response-borne "
                    "answer is read from it"}
    else:
        context["candidate_response"] = {
            "delivered": False,
            "note": "the solver's reply has not been written at "
                    "certification: the text of the final claim is the "
                    "candidate response, and a response-borne answer is "
                    "the answer that text states. Instructions in the task "
                    "about the reply's layout (a closing line, a label, a "
                    "marker) apply to the reply the runner collects "
                    "afterwards, not to the claim; they are not "
                    "conditions on this candidate. The verifier's own "
                    "state directory and stores are not the delivered "
                    "world."}
    baseline = os.path.join(os.path.abspath(state_dir), "baseline")
    if os.path.isdir(baseline):
        from ..baseline import environment_delta
        try:
            with open(os.path.join(baseline, "environment.json"),
                      encoding="utf-8") as handle:
                recorded = json.load(handle)
        except (OSError, ValueError):
            recorded = {}
        if recorded.get("path"):
            # the task environment's standard invocation: a check that
            # runs the delivered program does so on THIS path, not on a
            # path of its own choosing
            context["environment_invocation"] = {
                "PATH": recorded["path"],
                "interpreters": recorded.get("interpreters") or {},
            }
        try:
            delta = environment_delta(baseline)
        except Exception:  # noqa: BLE001 - context, never a verdict
            delta = {}
        if delta:
            # the environment as the task supplied it is the one a
            # delivery must run in; what the solver added for its own
            # session is disclosed so no seat mistakes it for the task's
            context["environment_changes_since_task_start"] = delta
            context["environment_note"] = (
                "the task's environment is the one recorded at task start; "
                "executables and packages listed here were added by the "
                "solver during its session and are not part of the "
                "delivered environment")
    return json.dumps(context, ensure_ascii=False)


class LedgerInstrument:
    """
    Verifies effect claims against the event stream.

    An effect claim's text is the canonical JSON produced by the mechanical
    parser: {"event_seq": n, "tool": ..., "digest": <event_digest>}. The check
    recomputes the digest from the log. The ledger, not the narration, is the
    source of truth for "this happened".
    """

    name = "ledger"

    def __init__(self, log: EventLog):
        self.log = log

    def check(self, effect_text: str) -> RunResult:
        base = {"instrument": self.name}
        try:
            ref = json.loads(effect_text)
            seq, claimed_digest = ref["event_seq"], ref["digest"]
        except Exception:
            return RunResult(OUTCOME_FAILED, {**base, "error": "malformed effect text"})

        event = self.log.get(seq)
        if event is None:
            return RunResult(
                OUTCOME_REFUTED, {**base, "event_seq": seq},
                witness={"reason": "no such event in ledger"},
            )
        actual = EventLog.event_digest(event)
        if actual != claimed_digest:
            return RunResult(
                OUTCOME_REFUTED,
                {**base, "event_seq": seq, "actual_digest": actual},
                witness={"reason": "ledger digest mismatch"},
            )
        return RunResult(OUTCOME_HOLDS, {**base, "event_seq": seq, "digest": actual})
