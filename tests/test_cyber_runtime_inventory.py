"""Offline runtime metadata fixtures; no root privileges, commands or models."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from verifier.core.model import content_hash
from verifier.profiles.cyber import runtime


@pytest.fixture
def host_runtime(tmp_path, monkeypatch):
    # Real descriptor/nofollow/streaming operations, simulated UID0 ownership.
    # Only test-temp ancestors have their unrelated writable directory bits
    # masked; files and the fixture subtree keep their actual permission bits.
    base = tmp_path.resolve()
    public = base / "usr"
    public.mkdir()
    paths = {role: (base / role if role in runtime._FIXED_PATHS else public / role)
             for role in runtime._ROLES}
    for role, path in paths.items():
        raw = ("public pinned " + role + "\n").encode()
        if role == "policy_source":
            raw = runtime._POLICY_SOURCE
        elif role in ("shim", "gcc_runtime", "gcc_compiler", "clang_compiler"):
            raw = b"\x7fELF\x02\x01\x01" + b"\x00" * 11 + (183).to_bytes(2, "little") + raw
        path.write_bytes(raw)
        path.chmod(0o644)
    manifest = base / "manifest.json"
    monkeypatch.setattr(runtime, "_MANIFEST_PATH", str(manifest))
    monkeypatch.setattr(runtime, "_FIXED_PATHS",
                        {role: str(paths[role]) for role in runtime._FIXED_PATHS})
    monkeypatch.setattr(runtime, "_COMPILER_ROOT", str(public) + "/")
    monkeypatch.setattr(runtime.os, "uname", lambda: SimpleNamespace(sysname="Linux", machine="aarch64"))
    real_fstat = os.fstat
    invalid_owners = set()

    def fixture_fstat(fd):
        info = real_fstat(fd)
        values = {name: getattr(info, name) for name in
                  ("st_dev", "st_ino", "st_mode", "st_uid", "st_nlink",
                   "st_size", "st_mtime_ns", "st_ctime_ns")}
        values["st_uid"] = 1000 if info.st_ino in invalid_owners else 0
        if stat.S_ISDIR(info.st_mode) and info.st_ino not in subtree_inodes():
            values["st_mode"] &= ~0o022
        return SimpleNamespace(**values)

    def subtree_inodes():
        return {base.stat().st_ino, public.stat().st_ino}

    monkeypatch.setattr(runtime.os, "fstat", fixture_fstat)

    def value():
        files = {role: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                 for role, path in paths.items()}
        return {
            "schema": runtime.RUNTIME_SCHEMA,
            "platform": {"system": "Linux", "machine": "aarch64"},
            "source_revision": content_hash({
                "schema": "gv-sanitizer-source-revision-v1",
                "files": {role: files[role]["sha256"]
                          for role in ("runner", "launcher", "policy_source")},
            }),
            "files": files,
            "compiler_versions": {"gcc": "gcc (public fixture) 12", "clang": "clang version 14"},
            "profiles": copy.deepcopy(runtime._PROFILES),
            "capabilities": copy.deepcopy(runtime._CAPABILITIES),
        }

    def write(data=None, raw=None):
        manifest.write_bytes(raw if raw is not None else json.dumps(value() if data is None else data).encode())
        manifest.chmod(0o644)
        return str(manifest)

    return SimpleNamespace(base=base, paths=paths, manifest=manifest, value=value,
                           write=write, invalid_owners=invalid_owners)


def test_none_preserves_legacy_inventory_and_revision(monkeypatch):
    def forbidden(*_args, **_kwargs):
        raise AssertionError("None must not touch filesystem")
    monkeypatch.setattr(runtime, "_read_pin", forbidden)
    original = {"schema": "legacy", "inventory": {"seat": {"execute": False}}}
    before = content_hash(original)
    selected = runtime.load_runtime_inventory(None)
    if selected is not None:
        original["inventory"]["runtime"] = selected.as_inventory()
    assert selected is None and content_hash(original) == before
    assert "runtime" not in original["inventory"]


def test_inventory_is_data_only_complete_and_immutable(host_runtime):
    selected = runtime.load_runtime_inventory(host_runtime.write())
    data = selected.as_inventory()
    assert data["schema"] == runtime.INVENTORY_SCHEMA
    assert data["capabilities"]["requested_leak_check_policy"] == "disabled_unchecked"
    assert selected.tool_root == str(host_runtime.base)
    assert data["child_leak_policy_validation"] == {"state": "unchecked", "hook_override_possible": True}
    assert data["command_interface"]["mode"] == "explicit_runtime_profile"
    assert data["command_interface"]["implicit_command_interception"] is False
    assert data["capabilities"]["compiler_or_target_rewrite"] is False
    assert data["capabilities"]["proc_read_grant"] is False
    assert data["asan_control_validation"] == {
        "state": "not_established_by_runtime_manifest",
        "scope": "declared_compilation_runtime", "automatic_safety_verdict": False,
    }
    for role, path in host_runtime.paths.items():
        assert data["files"][role] == {
            "path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "size": path.stat().st_size,
        }
    assert selected.revision == content_hash(data)
    data["capabilities"]["scope"] = "solver-selected"
    assert selected.as_inventory()["capabilities"]["scope"] == "explicit_runner_only"
    with pytest.raises(FrozenInstanceError):
        selected._inventory_json = "{}"
    assert not any(key in selected.as_inventory() for key in
                   ("result", "source", "task", "candidate", "score", "verified"))


@pytest.mark.parametrize("mutate", [
    lambda x: x.update(extra=True),
    lambda x: x.update(schema="foreign"),
    lambda x: x.update(source_revision="0" * 64),
    lambda x: x["platform"].update(system="Darwin"),
    lambda x: x["platform"].update(machine="unknown"),
    lambda x: x["platform"].update(extra="data"),
    lambda x: x["files"].pop("shim"),
    lambda x: x["files"]["runner"].update(size=1),
    lambda x: x["files"]["runner"].update(sha256="A" * 64),
    lambda x: x["files"]["shim"].update(path="/task/target"),
    lambda x: x["files"]["gcc_runtime"].update(path="/tmp/runtime"),
    lambda x: x["compiler_versions"].update(gcc=""),
    lambda x: x["compiler_versions"].update(clang="clang\nprivate"),
    lambda x: x["compiler_versions"].update(gcc=True),
    lambda x: x["compiler_versions"].update(gcc="gcc\ud800"),
    lambda x: x["profiles"]["gcc-shared"].update(preload=["shim", "gcc_runtime"]),
    lambda x: x["profiles"]["clang-static"].update(scope="all_commands"),
    lambda x: x["profiles"].update(extra={}),
    lambda x: x["capabilities"].update(proc_read_grant=0),
    lambda x: x["capabilities"].update(proc_read_grant=True),
    lambda x: x["capabilities"].update(future_target_compatibility_verified=True),
    lambda x: x["capabilities"].update(requested_leak_check_policy="passed"),
    lambda x: x["capabilities"].update(asan_memory_checks="disabled"),
    lambda x: x["capabilities"].update(link_order_check_disabled=True),
    lambda x: x["capabilities"].update(control_passed=True),
])
def test_manifest_rejects_extras_types_and_authority_expansion(host_runtime, mutate):
    data = host_runtime.value()
    mutate(data)
    with pytest.raises(runtime.RuntimeInventoryError):
        runtime.load_runtime_inventory(host_runtime.write(data))


@pytest.mark.parametrize("raw", [
    b'{"schema":1,"schema":1}', b'{"x":NaN}', b'{"x":Infinity}',
    b'{"x":-Infinity}', b'{"x":1.0}', b'[]', b'null', b'\xff', b'{',
    b'{"x":9223372036854775808}',
    b'{"x":"' + b'a' * 4097 + b'"}',
    b'{"x":' + b'[' * 14 + b'0' + b']' * 14 + b'}',
    b'{"x":[' + b'0,' * 2048 + b'0]}',
    b' ' * (runtime.MAX_MANIFEST_BYTES + 1),
])
def test_strict_json_bounds_and_invalid_values(host_runtime, raw):
    with pytest.raises(runtime.RuntimeInventoryError):
        runtime.load_runtime_inventory(host_runtime.write(raw=raw))


@pytest.mark.parametrize("path", [False, 1, "", "/", "/tmp/../manifest", "//tmp/manifest", "relative", "/x\x00", "/x\ud800"])
def test_host_path_is_fixed_and_canonical(path):
    with pytest.raises(runtime.RuntimeInventoryError):
        runtime.load_runtime_inventory(path)


def test_foreign_valid_absolute_manifest_refused_before_io(monkeypatch):
    monkeypatch.setattr(runtime, "_read_pin", lambda *_a, **_k: pytest.fail("foreign path opened"))
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_manifest_path_invalid"):
        runtime.load_runtime_inventory("/task/manifest.json")


def test_file_digest_mismatch(host_runtime):
    data = host_runtime.value()
    data["files"]["shim"]["sha256"] = "0" * 64
    with pytest.raises(runtime.RuntimeInventoryError, match="digest_mismatch"):
        runtime.load_runtime_inventory(host_runtime.write(data))


@pytest.mark.parametrize("kind", ["symlink", "fifo", "hardlink", "writable", "owner"])
def test_unsafe_file_kinds_fail_closed(host_runtime, kind):
    path = host_runtime.paths["shim"]
    data = host_runtime.value()
    if kind == "symlink":
        actual = host_runtime.base / "actual"
        path.rename(actual)
        path.symlink_to(actual)
    elif kind == "fifo":
        path.unlink()
        os.mkfifo(path)
    elif kind == "hardlink":
        os.link(path, host_runtime.base / "alias")
    elif kind == "writable":
        path.chmod(0o666)
    else:
        host_runtime.invalid_owners.add(path.stat().st_ino)
    with pytest.raises(runtime.RuntimeInventoryError):
        runtime.load_runtime_inventory(host_runtime.write(data))


def test_writable_parent_rejected(host_runtime):
    data = host_runtime.value()
    host_runtime.base.chmod(0o777)
    with pytest.raises(runtime.RuntimeInventoryError, match="untrusted"):
        runtime.load_runtime_inventory(host_runtime.write(data))


def test_symlink_parent_rejected(host_runtime):
    data = host_runtime.value()
    directory = host_runtime.base / "usr"
    moved = host_runtime.base / "moved"
    directory.rename(moved)
    directory.symlink_to(moved, target_is_directory=True)
    with pytest.raises(runtime.RuntimeInventoryError):
        runtime.load_runtime_inventory(host_runtime.write(data))


@pytest.mark.parametrize("mutation", ["bytes", "replace", "mode", "manifest"])
def test_immutable_snapshot_rechecked_before_publication(host_runtime, mutation):
    selected = runtime.load_runtime_inventory(host_runtime.write())
    path = host_runtime.paths["runner"]
    if mutation == "bytes":
        path.write_bytes(b"changed public bytes")
    elif mutation == "replace":
        old = path.read_bytes()
        path.unlink()
        path.write_bytes(old)
        path.chmod(0o644)
    elif mutation == "mode":
        path.chmod(0o444)
    else:
        host_runtime.write()
    with pytest.raises(runtime.RuntimeInventoryError, match="changed"):
        selected.as_inventory()
    with pytest.raises(runtime.RuntimeInventoryError, match="changed"):
        _ = selected.revision


def test_bytes_change_during_read_rejected(host_runtime, monkeypatch):
    host_runtime.write()
    original = os.read
    changed = False
    target = host_runtime.paths["shim"]

    def read(fd, count):
        nonlocal changed
        data = original(fd, count)
        if data and os.fstat(fd).st_ino == target.stat().st_ino and not changed:
            changed = True
            target.write_bytes(target.read_bytes() + b"changed")
        return data

    monkeypatch.setattr(runtime.os, "read", read)
    with pytest.raises(runtime.RuntimeInventoryError, match="changed"):
        runtime.load_runtime_inventory(str(host_runtime.manifest))


def test_runtime_file_stream_limit_is_bounded(host_runtime, monkeypatch):
    monkeypatch.setattr(runtime, "MAX_RUNTIME_FILE_BYTES", 8)
    with pytest.raises(runtime.RuntimeInventoryError, match="oversize"):
        runtime.load_runtime_inventory(host_runtime.write())


def test_revision_covers_entire_host_inventory(host_runtime):
    first = runtime.load_runtime_inventory(host_runtime.write()).revision
    data = host_runtime.value()
    data["compiler_versions"]["gcc"] += " public metadata revision"
    second = runtime.load_runtime_inventory(host_runtime.write(data)).revision
    assert second != first


def test_source_revision_canonical_algorithm_matches_runner(host_runtime):
    data = host_runtime.value()
    canonical = json.dumps({
        "schema": "gv-sanitizer-source-revision-v1",
        "files": {role: data["files"][role]["sha256"]
                  for role in ("runner", "launcher", "policy_source")},
    }, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")
    assert data["source_revision"] == hashlib.sha256(canonical).hexdigest()


@pytest.mark.parametrize("role", ["shim", "gcc_runtime", "gcc_compiler", "clang_compiler"])
def test_actual_pinned_binary_must_match_declared_runtime_abi(host_runtime, role):
    path = host_runtime.paths[role]
    raw = bytearray(path.read_bytes())
    raw[18:20] = (62).to_bytes(2, "little")
    path.write_bytes(raw)
    with pytest.raises(runtime.RuntimeInventoryError, match="binary_abi_invalid"):
        runtime.load_runtime_inventory(host_runtime.write())


@pytest.mark.parametrize("source", [
    b"int __lsan_is_turned_off(void) { return 0; }\n",
    runtime._POLICY_SOURCE + b"const char *__asan_default_options(void) { return \"detect_leaks=0\"; }\n",
    b"__attribute__((constructor)) static void changed(void) {}\n",
])
def test_policy_source_only_constant_hook_no_defaultoptions_or_behavior(host_runtime, source):
    host_runtime.paths["policy_source"].write_bytes(source)
    with pytest.raises(runtime.RuntimeInventoryError, match="policy_source_invalid"):
        runtime.load_runtime_inventory(host_runtime.write())


def test_actual_platform_mismatch_does_not_advertise_foreign_runtime(host_runtime, monkeypatch):
    path = host_runtime.write()
    monkeypatch.setattr(runtime.os, "uname", lambda: SimpleNamespace(sysname="Linux", machine="x86_64"))
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_platform_mismatch"):
        runtime.load_runtime_inventory(path)


def test_platform_is_runtime_metadata_not_requested_target_platform(host_runtime):
    selected = runtime.load_runtime_inventory(host_runtime.write())
    assert selected.as_inventory()["platform"] == {"system": "Linux", "machine": "aarch64"}
    assert "target_platform" not in selected.as_inventory()


def test_file_path_alias_and_manifest_alias_rejected(host_runtime):
    data = host_runtime.value()
    data["files"]["gcc_runtime"]["path"] = data["files"]["gcc_compiler"]["path"]
    with pytest.raises(runtime.RuntimeInventoryError, match="aliased"):
        runtime.load_runtime_inventory(host_runtime.write(data))


def test_named_path_replacement_during_read_rejected(host_runtime, monkeypatch):
    host_runtime.write()
    original = os.read
    target = host_runtime.paths["shim"]
    inode = target.stat().st_ino
    changed = False

    def read(fd, count):
        nonlocal changed
        data = original(fd, count)
        if data and os.fstat(fd).st_ino == inode and not changed:
            changed = True
            old = target.read_bytes()
            target.unlink()
            target.write_bytes(old)
            target.chmod(0o644)
        return data

    monkeypatch.setattr(runtime.os, "read", read)
    with pytest.raises(runtime.RuntimeInventoryError, match="changed"):
        runtime.load_runtime_inventory(str(host_runtime.manifest))


def test_fixed_runtime_paths_match_readonly_usr_tool_inventory():
    assert runtime._MANIFEST_PATH == "/usr/local/lib/gv-sanitizer/manifest.json"
    assert runtime._FIXED_PATHS == {
        "runner": "/usr/local/lib/gv-sanitizer/sanitizer_runner.py",
        "launcher": "/usr/local/bin/gv-sanitizer-run",
        "policy_source": "/usr/local/lib/gv-sanitizer/lsan_policy.c",
        "shim": "/usr/local/lib/gv-sanitizer/lsan_policy.so",
    }


def test_old_opt_manifest_refused_before_any_filesystem_io(monkeypatch):
    monkeypatch.setattr(runtime, "_read_pin", lambda *_a, **_k: pytest.fail("old manifest opened"))
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_manifest_path_invalid"):
        runtime.load_runtime_inventory("/opt/gv-sanitizer/manifest.json")


@pytest.mark.parametrize("role", ["runner", "policy_source", "shim"])
def test_old_opt_role_path_refused_before_any_tool_io(host_runtime, monkeypatch, role):
    data = host_runtime.value()
    data["files"][role]["path"] = "/opt/gv-sanitizer/" + {
        "runner": "sanitizer_runner.py", "policy_source": "lsan_policy.c", "shim": "lsan_policy.so",
    }[role]
    path = host_runtime.write(data)
    original = runtime._read_pin

    def manifest_only(candidate, **kwargs):
        if candidate != path:
            pytest.fail("invalid role path reached tool IO")
        return original(candidate, **kwargs)

    monkeypatch.setattr(runtime, "_read_pin", manifest_only)
    with pytest.raises(runtime.RuntimeInventoryError, match="runtime_file_role_path_invalid"):
        runtime.load_runtime_inventory(path)
