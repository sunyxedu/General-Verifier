"""Image inputs attached to an evidence seat's call.

Captured-world images an obligation's contract names, read once as fixed
raw bytes and reused unchanged by both evidence signers, and the images a
Judger evidence packet transports from the discharge artifact.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import stat
import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional

from ...execution.instruments import validated_image_info
from ..schema import RoleError
from .policy import seat_grants

_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".webp")
_MAX_IMAGE_COUNT = 4
_MAX_IMAGE_BYTES = 20 * 1024 * 1024
_MAX_IMAGE_TOTAL = 40 * 1024 * 1024
_MAX_IMAGE_CACHE_TOTAL = 128 * 1024 * 1024
_MAX_FAILED_IMAGE_PACKETS = 1024


@dataclass(frozen=True)
class CapturedImage:
    logical_path: str
    mime: str
    data: bytes
    sha256: str
    source: str = "captured_world"
    artifact_digest: Optional[str] = None


def image_mime(data: bytes) -> Optional[str]:
    try:
        media_type, _width, _height = validated_image_info(data)
        return media_type
    except (TypeError, ValueError):
        return None


def _mentions_path(text: str, path: str) -> bool:
    """Match a contract path/name as a token, not inside another name."""
    return re.search(
        r"(?<![A-Za-z0-9_./-])" + re.escape(path)
        + r"(?![A-Za-z0-9_./-])",
        text,
    ) is not None


def _read_fixed_image(path: str) -> tuple[bytes, str]:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as err:
        raise RoleError(f"cannot safely open captured image {path}: {err}")
    try:
        opened = os.fstat(fd)
        if not stat.S_ISREG(opened.st_mode):
            raise RoleError(f"captured image is not a regular file: {path}")
        if opened.st_size > _MAX_IMAGE_BYTES:
            raise RoleError(
                f"captured image exceeds {_MAX_IMAGE_BYTES} bytes: {path}")
        chunks = []
        total = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > _MAX_IMAGE_BYTES:
                raise RoleError(
                    f"captured image exceeds {_MAX_IMAGE_BYTES} bytes: {path}")
    finally:
        os.close(fd)
    data = b"".join(chunks)
    mime = image_mime(data)
    if mime is None:
        raise RoleError(f"unsupported or spoofed captured image: {path}")
    return data, mime


class CapturedImages:
    """The image inputs of one runtime's seats.

    A discharge and its independent judger must inspect exactly the same
    captured bytes, even if an executor later mutates its writable state;
    kernel construction shares one runtime (and so one cache) across both
    seats. A packet whose raw attachment made a Judger call fail is not
    retried through the same optional transport. AgentRole/BindingGame
    retain their existing retry loops; later attempts use the original
    toolful artifact path instead. The negative cache is bounded, and the
    optimization is conservatively disabled if it ever fills."""

    def __init__(self):
        self._image_cache: Dict[tuple[str, ...], tuple[CapturedImage, ...]] = {}
        self._image_blobs: Dict[tuple[str, str], CapturedImage] = {}
        self._image_blob_bytes = 0
        self._failed_image_packet_lock = threading.Lock()
        self._failed_image_packets: set[str] = set()
        self._image_packet_transport_disabled = False

    def collect(self, role_name: str, variables: Dict[str, Any],
                state_cwd: Optional[str]) -> tuple[CapturedImage, ...]:
        """The captured-world images an evidence seat's contract names."""
        basis_key = "declared_basis" if role_name == "discharge" else "basis"
        if (not seat_grants(role_name).images
                or variables.get("world_mode") != "captured"
                or str(variables.get(basis_key, "")).strip().lower() != "world"):
            return ()

        world = os.path.abspath(str(variables.get("world_path", "")))
        manifest = os.path.abspath(str(variables.get("world_manifest", "")))
        if not state_cwd:
            raise RoleError(f"{role_name}: captured image input needs a state cwd")
        state = os.path.realpath(os.path.abspath(state_cwd))
        expected_world = os.path.join(state, "world")
        expected_manifest = os.path.join(state, "world_manifest.json")
        if world != expected_world or manifest != expected_manifest:
            raise RoleError(f"{role_name}: invalid captured world paths")

        key = (
            world,
            str(variables.get("obligation_text", "")),
            str(variables.get("procedure", "")),
            str(variables.get("promised_artifact", "")),
            "world",
        )
        cached = self._image_cache.get(key)
        if cached is not None:
            return cached

        if (os.path.realpath(world) != world or os.path.islink(world)
                or not os.path.isdir(world)):
            raise RoleError(f"{role_name}: invalid captured world path")
        if (os.path.realpath(manifest) != manifest
                or os.path.islink(manifest) or not os.path.isfile(manifest)):
            raise RoleError(f"{role_name}: invalid captured world manifest")
        if os.path.getsize(manifest) > 65_536:
            raise RoleError(f"{role_name}: captured world manifest is too large")
        try:
            with open(manifest, encoding="utf-8") as handle:
                roots = json.load(handle).get("captured_roots")
        except (OSError, ValueError, AttributeError) as err:
            raise RoleError(f"{role_name}: invalid captured world manifest: {err}")
        if not isinstance(roots, list) or len(roots) > 32:
            raise RoleError(f"{role_name}: invalid captured_roots")

        contract = "\n".join(str(variables.get(key, "")) for key in (
            "obligation_text", "procedure", "promised_artifact"))

        inventory: Dict[str, str] = {}
        for root in roots:
            if (not isinstance(root, str) or not root.startswith("/")
                    or root.startswith("//")
                    or posixpath.normpath(root) != root):
                raise RoleError(f"{role_name}: non-canonical captured root")
            root_path = os.path.join(world, root.lstrip("/"))
            if not os.path.exists(root_path):
                continue
            if (os.path.islink(root_path)
                    or os.path.commonpath((world, os.path.realpath(root_path))) != world):
                raise RoleError(f"{role_name}: unsafe captured root {root}")
            for dirpath, dirnames, filenames in os.walk(
                root_path, followlinks=False
            ):
                dirnames[:] = sorted(
                    name for name in dirnames
                    if not os.path.islink(os.path.join(dirpath, name)))
                for filename in sorted(filenames):
                    if not filename.lower().endswith(_IMAGE_EXTENSIONS):
                        continue
                    path = os.path.join(dirpath, filename)
                    if os.path.islink(path):
                        continue
                    logical = "/" + os.path.relpath(path, world).replace(os.sep, "/")
                    inventory[logical] = path

        groups: Dict[str, list[str]] = {}
        selected: set[str] = set()
        for logical in inventory:
            basename = posixpath.basename(logical)
            groups.setdefault(basename, []).append(logical)
            aliases = (
                logical,
                logical.lstrip("/"),
                "world" + logical,
                inventory[logical],
            )
            if any(_mentions_path(contract, alias) for alias in aliases):
                selected.add(logical)
        for basename, logicals in groups.items():
            if any(logical in selected for logical in logicals):
                continue
            if _mentions_path(contract, basename):
                if len(logicals) != 1:
                    raise RoleError(
                        f"{role_name}: ambiguous captured image {basename!r}")
                selected.add(logicals[0])
        if len(selected) > _MAX_IMAGE_COUNT:
            raise RoleError(
                f"{role_name}: more than {_MAX_IMAGE_COUNT} captured images referenced")

        images = []
        total = 0
        for logical in sorted(selected):
            blob_key = (world, logical)
            item = self._image_blobs.get(blob_key)
            if item is None:
                data, mime = _read_fixed_image(inventory[logical])
                if self._image_blob_bytes + len(data) > _MAX_IMAGE_CACHE_TOTAL:
                    raise RoleError(
                        f"{role_name}: captured image cache exceeds "
                        f"{_MAX_IMAGE_CACHE_TOTAL} bytes")
                item = CapturedImage(
                    logical_path=logical,
                    mime=mime,
                    data=data,
                    sha256=hashlib.sha256(data).hexdigest(),
                )
                self._image_blobs[blob_key] = item
                self._image_blob_bytes += len(data)
            total += len(item.data)
            if total > _MAX_IMAGE_TOTAL:
                raise RoleError(
                    f"{role_name}: captured images exceed {_MAX_IMAGE_TOTAL} bytes")
            images.append(item)
        bundle = tuple(images)
        self._image_cache[key] = bundle
        return bundle

    def attach_judger_packet(self, candidate: Any,
                             images: tuple[CapturedImage, ...]):
        """Admit a Judger evidence packet and the images it transports:
        ``(images, packet, image_packet_digest)``, the packet None when it
        is not attached."""
        from ...engine.judger_prefetch import JudgerEvidencePacket
        packet = None
        image_packet_digest = None
        if type(candidate) is JudgerEvidencePacket:
            if self.packet_failed(candidate.artifact_digest):
                candidate = None
        if type(candidate) is JudgerEvidencePacket:
            transported = []
            try:
                for item in candidate.image_attachments():
                    if (image_mime(item.data) != item.media_type
                            or hashlib.sha256(item.data).hexdigest()
                            != item.sha256):
                        raise ValueError("prefetched image identity mismatch")
                    transported.append(CapturedImage(
                        logical_path=item.logical_path,
                        mime=item.media_type,
                        data=item.data,
                        sha256=item.sha256,
                        source="discharge_artifact",
                        artifact_digest=candidate.artifact_digest,
                    ))
            except (AttributeError, TypeError, ValueError):
                transported = []
                candidate = None
            if images and transported:
                # Captured-world image indexes and discharge-packet image
                # indexes are separate contracts. Until a unified index is
                # specified, preserve the original captured inputs and
                # drop the optional packet wholesale.
                packet = None
            elif (len(transported) <= _MAX_IMAGE_COUNT
                    and sum(len(item.data) for item in transported)
                    <= _MAX_IMAGE_TOTAL):
                images = (*images, *transported)
                packet = candidate
                if transported:
                    image_packet_digest = candidate.artifact_digest
            elif transported:
                # The ordinary Judger still has its original tools and
                # artifact path. Never attach an arbitrary subset.
                packet = None
            else:
                packet = candidate
        return images, packet, image_packet_digest

    def packet_failed(self, artifact_digest: str) -> bool:
        with self._failed_image_packet_lock:
            return (self._image_packet_transport_disabled
                    or artifact_digest in self._failed_image_packets)

    def disable_packet(self, artifact_digest: str) -> None:
        with self._failed_image_packet_lock:
            if len(self._failed_image_packets) >= _MAX_FAILED_IMAGE_PACKETS:
                self._failed_image_packets.clear()
                self._image_packet_transport_disabled = True
            else:
                self._failed_image_packets.add(artifact_digest)
