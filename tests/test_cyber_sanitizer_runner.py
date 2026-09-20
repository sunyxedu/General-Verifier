"""Offline runner/manifest safety tests, not sanitizer or task-score proofs."""
from __future__ import annotations

import copy
import errno
import hashlib
import json
import os
import stat
import struct
from types import SimpleNamespace

import pytest

from experiments.cybergym import sanitizer_runner as runner
from experiments.cybergym import sanitizer_manifest as generator


def elf():
    data = bytearray(20)
    data[:6] = b"\x7fELF\x02\x01"
    struct.pack_into("<H", data, 18, 183)
    return bytes(data)


@pytest.fixture
def manifest_fixture(monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Linux")
    monkeypatch.setattr(runner.platform, "machine", lambda: "aarch64")
    paths = {**runner.FIXED_FILES, "gcc_runtime": "/usr/lib/libasan.so.8",
             "gcc_compiler": "/usr/bin/gcc-real", "clang_compiler": "/usr/bin/clang-real"}
    contents = {path: elf() for path in paths.values()}
    contents[paths["runner"]] = b"trusted standalone runner fixture\n"
    contents[paths["launcher"]] = runner.LAUNCHER_SOURCE
    contents[paths["policy_source"]] = runner.POLICY_SOURCE
    files = {role: {"path": path, "sha256": hashlib.sha256(contents[path]).hexdigest()}
             for role, path in paths.items()}
    data = {"schema": runner.SCHEMA, "platform": {"system": "Linux", "machine": "aarch64"},
            "source_revision": runner.source_revision(files), "files": files,
            "compiler_versions": {"gcc": "gcc fixture", "clang": "clang fixture"},
            "profiles": copy.deepcopy(runner.PROFILES), "capabilities": dict(runner.CAPABILITIES)}
    contents[runner.MANIFEST_PATH] = runner.canonical_json(data)
    identities = {path: (1, i, stat.S_IFREG | 0o444, 0, 1, len(raw), 1, 1)
                  for i, (path, raw) in enumerate(contents.items(), 1)}
    def reader(path, *, limit=runner.MAX_FILE_BYTES):
        raw = contents[path]
        if len(raw) > limit:
            raise runner.SanitizerError("runtime_file_too_large")
        return raw, identities[path]
    monkeypatch.setattr(runner, "read_trusted_file", reader)
    return SimpleNamespace(data=data, contents=contents, identities=identities,
                           refresh=lambda: contents.__setitem__(runner.MANIFEST_PATH,
                                                               runner.canonical_json(data)))


def test_valid_manifest_is_rechecked_and_only_declares_profiles(manifest_fixture):
    manifest = runner.load_manifest()
    manifest.recheck()
    assert manifest.data["capabilities"]["requested_leak_check_policy"] == "disabled_unchecked"
    assert manifest.data["capabilities"]["future_target_compatibility_verified"] is False
    assert "leak_checking" not in manifest.data["capabilities"]


@pytest.mark.parametrize("raw", [b'{"a":1,"a":2}', b'{"a":NaN}', b'{"a":Infinity}',
                                 b'{"a":1e309}', b'{"a":0}', b'\xff', b'{'])
def test_malformed_numeric_and_duplicate_json_fail_closed(raw):
    with pytest.raises(runner.SanitizerError):
        runner._strict_json(raw)


def test_json_budget_fail_closed():
    with pytest.raises(runner.SanitizerError):
        runner._strict_json(b"[" * 14 + b"true" + b"]" * 14)
    with pytest.raises(runner.SanitizerError):
        runner._strict_json(runner.canonical_json([True] * 2050))


@pytest.mark.parametrize("change", [
    lambda d: d["capabilities"].__setitem__("proc_read_grant", 0),
    lambda d: d["capabilities"].__setitem__("link_order_check_disabled", True),
    lambda d: d["capabilities"].__setitem__("future_target_compatibility_verified", True),
    lambda d: d["capabilities"].__setitem__("leak_checking", "disabled"),
    lambda d: d["profiles"]["gcc-shared"].__setitem__("preload", ["shim", "gcc_runtime"]),
    lambda d: d["profiles"].__setitem__("auto", {"preload": ["shim"]}),
    lambda d: d["files"]["shim"].__setitem__("path", "/tmp/shim.so"),
    lambda d: d["files"]["gcc_runtime"].__setitem__("path", "/task/libasan.so"),
    lambda d: d["compiler_versions"].__setitem__("gcc", ""),
    lambda d: d.__setitem__("source_revision", "0" * 64),
    lambda d: d.__setitem__("solver_override", True),
])
def test_manifest_scope_and_typed_drift_rejected(manifest_fixture, change):
    change(manifest_fixture.data)
    manifest_fixture.refresh()
    with pytest.raises(runner.SanitizerError):
        runner.load_manifest()


@pytest.mark.parametrize("role", ["shim", "gcc_runtime", "gcc_compiler", "clang_compiler"])
def test_non_native_elf_rejected_even_if_hash_matches(manifest_fixture, role):
    path = manifest_fixture.data["files"][role]["path"]
    raw = b"non ELF trusted-looking fixture"
    manifest_fixture.contents[path] = raw
    manifest_fixture.data["files"][role]["sha256"] = hashlib.sha256(raw).hexdigest()
    manifest_fixture.refresh()
    with pytest.raises(runner.SanitizerError, match="runtime_native_elf_invalid"):
        runner.load_manifest()


def test_policy_and_launcher_cannot_be_changed_by_repinning(manifest_fixture):
    for role, raw, expected in (("policy_source", b"int __lsan_is_turned_off(void){return 0;}\n",
                                  "runtime_policy_source_mismatch"),
                                 ("launcher", b'#!/bin/sh\nexec python "$@"\n',
                                  "runtime_launcher_source_mismatch")):
        data = copy.deepcopy(manifest_fixture.data)
        path = data["files"][role]["path"]
        original = manifest_fixture.contents[path]
        data["files"][role]["sha256"] = hashlib.sha256(raw).hexdigest()
        data["source_revision"] = runner.source_revision(data["files"])
        manifest_fixture.contents[path] = raw
        manifest_fixture.contents[runner.MANIFEST_PATH] = runner.canonical_json(data)
        with pytest.raises(runner.SanitizerError, match=expected):
            runner.load_manifest()
        manifest_fixture.contents[path] = original


def test_platform_not_linux_is_rejected(manifest_fixture, monkeypatch):
    monkeypatch.setattr(runner.platform, "system", lambda: "Darwin")
    with pytest.raises(runner.SanitizerError, match="runtime_platform_mismatch"):
        runner.load_manifest()


def test_manifest_content_and_descriptor_replay_rejected(manifest_fixture):
    manifest = runner.load_manifest()
    path = manifest.data["files"]["shim"]["path"]
    manifest_fixture.contents[path] += b"changed"
    with pytest.raises(runner.SanitizerError, match="runtime_pin_mismatch"):
        manifest.recheck()
    manifest_fixture.contents[path] = manifest_fixture.contents[path][:-7]
    manifest_fixture.identities[path] = (2, 999, stat.S_IFREG | 0o444, 0, 1, 20, 1, 1)
    with pytest.raises(runner.SanitizerError, match="runtime_pins_changed"):
        manifest.recheck()


def test_mutable_python_manifest_cannot_change_authority(manifest_fixture):
    manifest = runner.load_manifest()
    manifest.data["profiles"]["gcc-shared"]["preload"].reverse()
    with pytest.raises(runner.SanitizerError, match="runtime_pins_changed"):
        manifest.recheck()


@pytest.mark.parametrize("key", ["LD_PRELOAD", "LD_AUDIT", "LD_DYNAMIC_WEAK", "ASAN_OPTIONS",
                                "LSAN_OPTIONS", "UBSAN_OPTIONS"])
def test_environment_conflicts_fail_without_echoing_value(manifest_fixture, key):
    manifest = runner.load_manifest()
    with pytest.raises(runner.SanitizerError) as error:
        runner.scoped_environment(manifest, "gcc-shared", {key: "do-not-print-secret-value"})
    assert str(error.value) == "runtime_environment_conflict"
    assert "do-not-print" not in str(error.value)


def test_scoped_preload_preserves_runtime_order_and_other_environment(manifest_fixture):
    manifest = runner.load_manifest()
    source = {"PATH": "/trusted/bin", "CFLAGS": "-O2"}
    gcc = runner.scoped_environment(manifest, "gcc-shared", source)
    clang = runner.scoped_environment(manifest, "clang-static", source)
    assert gcc["LD_PRELOAD"].split(":") == [manifest.data["files"][role]["path"]
                                           for role in ("gcc_runtime", "shim")]
    assert clang["LD_PRELOAD"] == manifest.data["files"]["shim"]["path"]
    assert gcc["CFLAGS"] == source["CFLAGS"] and "LD_PRELOAD" not in source


@pytest.mark.parametrize("returncode,exitcode,signal", [(0, 0, None), (1, 1, None),
                                                        (77, 77, None), (125, 125, None), (-11, 139, 11)])
def test_real_child_status_never_becomes_a_verdict(manifest_fixture, monkeypatch, returncode, exitcode, signal):
    manifest = runner.load_manifest()
    calls, records = [], []
    class Child:
        pid = 991
        def wait(self, **kwargs):
            assert kwargs == {}  # No independent total deadline.
            return returncode
    def popen(command, **kwargs):
        calls.append((command, kwargs))
        assert "shell" not in kwargs and "cwd" not in kwargs
        return Child()
    monkeypatch.setattr(runner.subprocess, "Popen", popen)
    command = ["./configure", "--enable-checks"]
    actual = runner.run_command(manifest, "gcc-shared", command, environment={"PATH": "/bin"}, emit=records.append)
    assert actual == exitcode and calls[0][0] == command
    assert [r["event"] for r in records] == ["started", "completed"]
    final = records[-1]
    assert final["child_started"] is True and final["child_returncode"] == returncode
    assert final["child_signal"] == signal
    assert final["requested_leak_check_policy"] == "disabled_unchecked"
    assert final["absence_of_leaks_proven"] is False
    assert final["policy_effective_for_all_targets"] is False
    assert final["future_target_compatibility_verified"] is False
    assert not {"verdict", "verified", "score", "certificate"} & final.keys()


def test_spawn_failure_is_distinct_infra_and_not_a_candidate_exit(manifest_fixture, monkeypatch):
    records = []
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: (_ for _ in ()).throw(PermissionError()))
    assert runner.run_command(runner.load_manifest(), "clang-static", ["./program"],
                              environment={}, emit=records.append) == 125
    assert records[-1]["event"] == "infra_error" and records[-1]["child_started"] is False
    assert "child_returncode" not in records[-1]


def test_post_child_pin_drift_keeps_actual_exit_in_diagnostic(manifest_fixture, monkeypatch):
    manifest = runner.load_manifest()
    records = []
    class Child:
        pid = 992
        def wait(self):
            manifest_fixture.identities[runner.MANIFEST_PATH] = (9,) * 8
            return 77
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Child())
    assert runner.run_command(manifest, "clang-static", ["./program"], environment={}, emit=records.append) == 125
    assert records[-1]["event"] == "infra_error" and records[-1]["child_started"] is True
    assert records[-1]["child_returncode"] == 77


@pytest.mark.parametrize("argv", [["--runtime-profile", "auto", "--", "./x"],
                                  ["--runtime-profile", "gcc-shared", "./x"],
                                  ["--manifest", "do-not-print-secret-value"], []])
def test_cli_requires_profile_and_no_solver_manifest_entry(argv, capsys):
    assert runner.main(argv) == 125
    record = json.loads(capsys.readouterr().err.removeprefix(runner.METADATA_PREFIX))
    assert record["event"] == "infra_error" and record["child_started"] is False
    assert "do-not-print-secret-value" not in json.dumps(record)


def fake_descriptor_tree(monkeypatch, *, mode=stat.S_IFREG | 0o444, uid=0, nlink=1, changed=False):
    raw, reads, closed = b"safe", [], []
    def info(fd, mutation=0):
        return SimpleNamespace(st_dev=1, st_ino=fd, st_mode=(stat.S_IFDIR | 0o755) if fd != 12 else mode,
                               st_uid=uid if fd == 12 else 0, st_nlink=nlink if fd == 12 else 2,
                               st_size=4 if fd == 12 else 0, st_mtime_ns=mutation, st_ctime_ns=0)
    calls = {12: 0}
    def fstat(fd):
        if fd == 12:
            calls[12] += 1
        return info(fd, int(changed and fd == 12 and calls[12] > 1))
    def open_(path, flags, *, dir_fd=None):
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK and flags & os.O_CLOEXEC
        return {"/": 10, "opt": 11, "file": 12}[path]
    def read(fd, size):
        reads.append(fd)
        return raw if len(reads) == 1 else b""
    monkeypatch.setattr(runner.os, "open", open_)
    monkeypatch.setattr(runner.os, "fstat", fstat)
    monkeypatch.setattr(runner.os, "read", read)
    monkeypatch.setattr(runner.os, "close", closed.append)
    return reads, closed


def test_descriptor_reader_is_bounded_nofollow_and_closes(monkeypatch):
    _, closed = fake_descriptor_tree(monkeypatch)
    raw, _ = runner.read_trusted_file("/opt/file", limit=4)
    assert raw == b"safe" and set(closed) == {10, 11, 12}


@pytest.mark.parametrize("kwargs", [{"mode": stat.S_IFIFO | 0o444}, {"mode": stat.S_IFREG | 0o666},
                                    {"uid": 1000}, {"nlink": 2}])
def test_fifo_mutable_nonroot_or_hardlink_rejected_before_read(monkeypatch, kwargs):
    reads, closed = fake_descriptor_tree(monkeypatch, **kwargs)
    with pytest.raises(runner.SanitizerError):
        runner.read_trusted_file("/opt/file")
    assert not reads and set(closed) == {10, 11, 12}


def test_mutation_during_descriptor_read_rejected(monkeypatch):
    fake_descriptor_tree(monkeypatch, changed=True)
    with pytest.raises(runner.SanitizerError, match="runtime_file_changed"):
        runner.read_trusted_file("/opt/file")


@pytest.mark.parametrize("path", ["relative", "//opt/file", "/opt/../file", "/opt//file", "/opt/file\0x"])
def test_noncanonical_paths_rejected(path):
    with pytest.raises(runner.SanitizerError, match="runtime_path_invalid"):
        runner.read_trusted_file(path)


def test_build_generator_pins_actual_compilers_and_gcc_runtime(manifest_fixture, monkeypatch):
    monkeypatch.setattr(generator, "runner", runner)
    commands = []
    paths = manifest_fixture.data["files"]
    monkeypatch.setattr(generator.shutil, "which", lambda name, *, path: paths[name + "_compiler"]["path"])
    def public_command(argv, **kwargs):
        commands.append((argv, kwargs))
        assert kwargs == {"env": {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C",
                                  "HOME": "/nonexistent"}, "check": True,
                          "capture_output": True, "text": True}
        if argv[1] == "-print-file-name=libasan.so":
            return SimpleNamespace(stdout=paths["gcc_runtime"]["path"] + "\n")
        name = "gcc" if argv[0] == paths["gcc_compiler"]["path"] else "clang"
        return SimpleNamespace(stdout=name + " fixture\npublic version detail\n")
    monkeypatch.setattr(generator.subprocess, "run", public_command)
    actual = generator.build_manifest()
    assert actual == manifest_fixture.data
    assert [argv[1] for argv, _ in commands] == ["--version", "--version", "-print-file-name=libasan.so"]
    assert actual["capabilities"]["requested_leak_check_policy"] == "disabled_unchecked"
    assert "leak_checking" not in actual["capabilities"]


def test_build_generator_refuses_missing_or_task_compiler(manifest_fixture, monkeypatch):
    monkeypatch.setattr(generator, "runner", runner)
    for path, reason in ((None, "runtime_compiler_missing"), ("/tmp/task-gcc", "runtime_compiler_path_invalid")):
        monkeypatch.setattr(generator.shutil, "which", lambda *a, **k: path)
        with pytest.raises(runner.SanitizerError, match=reason):
            generator.build_manifest()


def test_build_generator_only_creates_fixed_manifest_and_handles_short_writes(manifest_fixture, monkeypatch):
    monkeypatch.setattr(generator, "runner", runner)
    monkeypatch.setattr(generator.sys, "argv", ["/usr/local/lib/gv-sanitizer/sanitizer_manifest.py"])
    monkeypatch.setattr(generator, "build_manifest", lambda: manifest_fixture.data)
    operations, written = [], bytearray()
    def create(path, flags, mode):
        assert path == runner.MANIFEST_PATH and mode == 0o444
        assert flags == os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW | os.O_CLOEXEC
        operations.append("create")
        return 123
    def write(fd, data):
        assert fd == 123
        count = min(11, len(data))
        written.extend(data[:count])
        return count
    monkeypatch.setattr(generator.os, "open", create)
    monkeypatch.setattr(generator.os, "write", write)
    monkeypatch.setattr(generator.os, "fsync", lambda fd: operations.append("fsync"))
    monkeypatch.setattr(generator.os, "close", lambda fd: operations.append("close"))
    monkeypatch.setattr(runner, "load_manifest", lambda: operations.append("validate"))
    generator.main()
    assert bytes(written) == runner.canonical_json(manifest_fixture.data) + b"\n"
    assert operations == ["create", "fsync", "close", "validate"]


def test_build_generator_has_no_path_config_or_overwrite_fallback(monkeypatch):
    monkeypatch.setattr(generator, "runner", runner)
    monkeypatch.setattr(generator.sys, "argv", ["generator", "--manifest", "/tmp/solver.json"])
    with pytest.raises(runner.SanitizerError, match="runtime_generator_no_arguments"):
        generator.main()


@pytest.mark.parametrize("diagnostic_case", ["mixed_static_shared_runtimes", "strong_target_callback_zero"])
def test_incompatible_or_overriding_child_failure_is_not_reclassified(manifest_fixture, monkeypatch, diagnostic_case):
    # Offline readout regression only: the separate native matrix establishes the
    # actual sanitizer failures. Neither this fixture nor the manifest proves them.
    class Child:
        pid = 993
        def wait(self):
            return 1
    monkeypatch.setattr(runner.subprocess, "Popen", lambda *a, **k: Child())
    records = []
    assert runner.run_command(runner.load_manifest(), "gcc-shared", ["./dummy-control"],
                              environment={}, emit=records.append) == 1
    assert records[-1]["event"] == "completed"
    assert records[-1]["child_returncode"] == 1
    assert records[-1]["future_target_compatibility_verified"] is False
    assert records[-1]["absence_of_leaks_proven"] is False


@pytest.fixture
def readonly_tree(manifest_fixture, monkeypatch):
    """Descriptor/mount simulator; not a claim of actual sandbox enforcement."""
    tree = SimpleNamespace(contents=manifest_fixture.contents, overrides={}, fds={},
                           opened=[], read_paths=[], closed=[], next_fd=100,
                           root_device=1, usr_device=2, root_readonly=False,
                           usr_readonly=True, symlinks=set(), replace_on_read=None)
    directories = {"/", "/usr"}
    for path in tree.contents:
        parent = os.path.dirname(path)
        while parent != "/":
            directories.add(parent)
            parent = os.path.dirname(parent)
    tree.directories = directories
    def info(path):
        directory = path in directories
        value = dict(st_dev=tree.root_device if path == "/" else tree.usr_device,
                     st_ino=sorted(directories | set(tree.contents)).index(path) + 1,
                     st_mode=(stat.S_IFDIR | (0o777 if path == "/" else 0o755))
                     if directory else stat.S_IFREG | 0o444,
                     st_uid=1000 if path == "/" else 65534,
                     st_nlink=2 if directory else 1,
                     st_size=0 if directory else len(tree.contents[path]),
                     st_mtime_ns=1, st_ctime_ns=1)
        value.update(tree.overrides.get(path, {}))
        return SimpleNamespace(**value)
    def open_(path, flags, *, dir_fd=None):
        assert flags & os.O_NOFOLLOW and flags & os.O_NONBLOCK and flags & os.O_CLOEXEC
        named = path if dir_fd is None else os.path.join(tree.fds[dir_fd]["path"], path)
        tree.opened.append(named)
        if named in tree.symlinks:
            raise OSError(errno.ELOOP, "fixture symlink")
        if named not in directories and named not in tree.contents:
            raise FileNotFoundError(named)
        if flags & os.O_DIRECTORY and named not in directories:
            raise NotADirectoryError(named)
        fd = tree.next_fd
        tree.next_fd += 1
        tree.fds[fd] = {"path": named, "info": info(named), "position": 0}
        return fd
    def read(fd, size):
        entry = tree.fds[fd]
        path = entry["path"]
        tree.read_paths.append(path)
        position = entry["position"]
        raw = tree.contents[path][position:position + size]
        entry["position"] += len(raw)
        if raw and tree.replace_on_read is not None:
            changed = tree.replace_on_read
            tree.overrides[changed] = {"st_ino": info(changed).st_ino + 999}
            tree.replace_on_read = None
        return raw
    def fstatvfs(fd):
        path = tree.fds[fd]["path"]
        ro = tree.root_readonly if path == "/" else tree.usr_readonly
        ro = tree.overrides.get(path, {}).get("readonly", ro)
        return SimpleNamespace(f_flag=os.ST_RDONLY if ro else 0)
    monkeypatch.setattr(runner.os, "open", open_)
    monkeypatch.setattr(runner.os, "fstat", lambda fd: tree.fds[fd]["info"])
    monkeypatch.setattr(runner.os, "fstatvfs", fstatvfs)
    monkeypatch.setattr(runner.os, "read", read)
    monkeypatch.setattr(runner.os, "close", tree.closed.append)
    return tree


def test_synthetic_mutable_root_and_overflow_uid_require_readonly_anchor(readonly_tree):
    assert runner._detect_trust_mode() == runner.READONLY_TRUST
    manifest = runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert manifest.trust_mode == runner.READONLY_TRUST
    assert all(identity[3] == 65534 for identity in manifest.identities.values())
    manifest.recheck()
    assert all(path.startswith("/usr/") for path in readonly_tree.read_paths)
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


def test_same_device_is_safe_only_with_readonly_root(readonly_tree):
    readonly_tree.usr_device = readonly_tree.root_device
    readonly_tree.root_readonly = True
    assert runner._detect_trust_mode() == runner.READONLY_TRUST
    assert runner.load_manifest(_trust_mode=runner.READONLY_TRUST).trust_mode == runner.READONLY_TRUST


def test_readonly_authority_is_mount_proof_not_a_specific_mapped_uid(readonly_tree):
    for path in readonly_tree.directories | set(readonly_tree.contents):
        readonly_tree.overrides[path] = {"st_uid": 1000}
    manifest = runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert all(identity[3] == 1000 for identity in manifest.identities.values())
    manifest.recheck()


@pytest.mark.parametrize("change", ["writable_usr", "same_device_mutable_root", "writable_usr_mode"])
def test_unsafe_anchor_never_reinterprets_uid_or_reads_bytes(readonly_tree, change):
    if change == "writable_usr":
        readonly_tree.usr_readonly = False
    elif change == "same_device_mutable_root":
        readonly_tree.usr_device = readonly_tree.root_device
    else:
        readonly_tree.overrides["/usr"] = {"st_mode": stat.S_IFDIR | 0o777}
    assert runner._detect_trust_mode() == runner.HOST_TRUST
    with pytest.raises(runner.SanitizerError, match="runtime_readonly_anchor_untrusted"):
        runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert not readonly_tree.read_paths
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


@pytest.mark.parametrize("uid", [1000, 65534])
def test_default_host_rule_does_not_accept_namespace_uids(monkeypatch, uid):
    reads, _ = fake_descriptor_tree(monkeypatch, uid=uid)
    with pytest.raises(runner.SanitizerError, match="runtime_file_untrusted"):
        runner.read_trusted_file("/opt/file")
    assert not reads


@pytest.mark.parametrize("path", ["/opt/gv-sanitizer/manifest.json", "/usr/../tmp/x", "/usr//lib/x",
                                 "relative", "/usr", "/usr/local/x\0secret"])
def test_readonly_mode_rejects_outside_or_noncanonical_paths_before_io(readonly_tree, path):
    with pytest.raises(runner.SanitizerError, match="runtime_readonly_path_invalid"):
        runner._read_readonly_file(path)
    assert not readonly_tree.opened and not readonly_tree.read_paths


@pytest.mark.parametrize("path", ["/usr", "/usr/local", runner.MANIFEST_PATH])
def test_readonly_anchor_ancestors_and_leaf_cannot_be_symlinks(readonly_tree, path):
    readonly_tree.symlinks.add(path)
    with pytest.raises(runner.SanitizerError):
        runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert not readonly_tree.read_paths
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


@pytest.mark.parametrize("path,change", [
    ("/usr/local", {"readonly": False}),
    ("/usr/local", {"st_mode": stat.S_IFDIR | 0o777}),
    (runner.MANIFEST_PATH, {"readonly": False}),
    (runner.MANIFEST_PATH, {"st_mode": stat.S_IFREG | 0o666}),
    (runner.MANIFEST_PATH, {"st_nlink": 2}),
    (runner.MANIFEST_PATH, {"st_mode": stat.S_IFIFO | 0o444}),
])
def test_readonly_descendants_require_immutable_regular_singlelink_files(readonly_tree, path, change):
    readonly_tree.overrides[path] = change
    with pytest.raises(runner.SanitizerError):
        runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert not readonly_tree.read_paths
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


@pytest.mark.parametrize("changed", [runner.MANIFEST_PATH, "/usr/local/lib"])
def test_readonly_named_entry_rewalk_detects_leaf_or_ancestor_replacement(readonly_tree, changed):
    readonly_tree.replace_on_read = changed
    with pytest.raises(runner.SanitizerError, match="runtime_file_changed"):
        runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


def test_readonly_mode_still_requires_full_content_pins(readonly_tree, manifest_fixture):
    path = runner.FIXED_FILES["shim"]
    readonly_tree.contents[path] += b"different content"
    with pytest.raises(runner.SanitizerError, match="runtime_pin_mismatch"):
        runner.load_manifest(_trust_mode=runner.READONLY_TRUST)


def test_readonly_reader_enforces_size_budget_before_any_read(readonly_tree):
    with pytest.raises(runner.SanitizerError, match="runtime_file_too_large"):
        runner._read_readonly_file(runner.MANIFEST_PATH, limit=1)
    assert not readonly_tree.read_paths
    assert set(readonly_tree.fds) == set(readonly_tree.closed)


def test_invalid_internal_trust_mode_cannot_fallback_to_readonly(readonly_tree):
    with pytest.raises(runner.SanitizerError, match="runtime_trust_mode_invalid"):
        runner.load_manifest(_trust_mode="solver_selected_mode")
    assert not readonly_tree.opened and not readonly_tree.read_paths


def test_readonly_manifest_recheck_keeps_original_mode_and_mount_proof(readonly_tree):
    manifest = runner.load_manifest(_trust_mode=runner.READONLY_TRUST)
    with pytest.raises(AttributeError):
        manifest.trust_mode = runner.HOST_TRUST
    readonly_tree.usr_readonly = False
    with pytest.raises(runner.SanitizerError, match="runtime_readonly_anchor_untrusted"):
        manifest.recheck()


def test_cli_selects_proven_anchor_without_user_trust_switch(readonly_tree, monkeypatch):
    monkeypatch.setattr(runner, "__file__", runner.FIXED_FILES["runner"])
    calls = []
    monkeypatch.setattr(runner, "run_command", lambda manifest, profile, command:
                        calls.append((manifest.trust_mode, profile, command)) or 77)
    assert runner.main(["--runtime-profile", "clang-static", "--", "./configure"]) == 77
    assert calls == [(runner.READONLY_TRUST, "clang-static", ["./configure"])]


def test_cli_rejects_user_supplied_trust_mode(capsys):
    assert runner.main(["--trust-mode", "readonly_usr_anchor", "--runtime-profile", "clang-static",
                        "--", "./configure"]) == 125
    record = json.loads(capsys.readouterr().err.removeprefix(runner.METADATA_PREFIX))
    assert record["event"] == "infra_error" and record["child_started"] is False
