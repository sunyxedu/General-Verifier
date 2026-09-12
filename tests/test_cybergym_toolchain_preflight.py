"""Pure toolchain metadata fixtures: no API, Docker, compiler or target runs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from experiments.cybergym import toolchain_preflight as probe


def fixture(monkeypatch, *, missing=(), machine="aarch64", overrides=None, absent=()):
    calls = []
    target = "x86_64-linux-gnu" if machine in ("x86_64", "amd64") else "aarch64-linux-gnu"
    outputs = {("gcc", "-dumpfullversion"): "12.2.0", ("gcc", "-dumpmachine"): target,
               ("gcc", "-print-file-name=libasan.so"): "/gcc/libasan.so",
               ("gcc", "-print-file-name=libubsan.so"): "/gcc/libubsan.so",
               ("clang", "-dumpversion"): "14.0.6", ("clang", "-dumpmachine"): target,
               ("clang", "-print-resource-dir"): "/llvm/lib/clang/14.0.6"}
    outputs.update(overrides or {})

    def metadata(command):
        calls.append(command)
        value = outputs[tuple(command)]
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(probe, "_metadata", metadata)
    monkeypatch.setattr(probe.shutil, "which", lambda name: None if name in missing else "/tools/" + name)
    monkeypatch.setattr(probe.platform, "machine", lambda: machine)
    monkeypatch.setattr(probe.platform, "system", lambda: "Linux")
    monkeypatch.setattr(probe.os.path, "isfile", lambda path: path not in absent)
    return calls


def test_complete_metadata_and_declared_mismatch_remains_ready(monkeypatch):
    calls = fixture(monkeypatch)
    report = probe.inspect_toolchain("linux/amd64")
    assert report["ready"] is True and report["errors"] == []
    assert report["platform"] == {"host": "linux/arm64", "requested_target": "linux/amd64", "mismatch": True}
    assert report["warnings"] == ["target_platform_mismatch"]
    assert report["runtimes"] == {"gcc": {"asan": True, "ubsan": True},
                                  "clang": {"asan": True, "ubsan": True, "libfuzzer": True}}
    assert report["model_calls"] == 0 and report["network_used"] is False
    assert report["task_data_used"] is False and report["smoke_tested"] is False
    assert report["metadata_only"] is True
    assert all(command[0] in ("gcc", "clang") and len(command) == 2 for command in calls)
    assert "/tools/" not in json.dumps(report) and "/llvm/" not in json.dumps(report)


@pytest.mark.parametrize("machine", ["x86_64", "amd64"])
def test_native_platform_and_unrequested_platform(monkeypatch, machine):
    fixture(monkeypatch, machine=machine)
    report = probe.inspect_toolchain("linux/amd64")
    assert report["ready"] and report["platform"]["mismatch"] is False
    assert report["warnings"] == []
    assert probe.inspect_toolchain()["platform"]["mismatch"] is None


def test_gcc_sanitizers_not_confused_with_absent_llvm(monkeypatch):
    calls = fixture(monkeypatch, missing=("clang", "clang++", "cmake", "llvm-symbolizer"))
    report = probe.inspect_toolchain()
    assert not report["ready"]
    assert report["runtimes"]["gcc"] == {"asan": True, "ubsan": True}
    assert "command_missing_cmake" in report["errors"]
    assert not any(command[0] == "clang" for command in calls)


def test_absent_runtime_and_unresolved_compiler_library_name(monkeypatch):
    fixture(monkeypatch, overrides={("gcc", "-print-file-name=libasan.so"): "libasan.so"},
            absent=("/llvm/lib/clang/14.0.6/lib/linux/libclang_rt.fuzzer-aarch64.a",))
    report = probe.inspect_toolchain()
    assert report["runtimes"]["gcc"]["asan"] is False
    assert report["runtimes"]["clang"]["libfuzzer"] is False
    assert report["errors"] == ["clang_runtime_missing_libfuzzer", "gcc_runtime_missing_asan"]


def test_cross_compiler_runtime_uses_compiler_target_not_host(monkeypatch):
    fixture(monkeypatch, overrides={("clang", "-dumpmachine"): "x86_64-pc-linux-gnu"},
            absent=("/llvm/lib/clang/14.0.6/lib/linux/libclang_rt.fuzzer-aarch64.a",))
    report = probe.inspect_toolchain()
    assert report["ready"] and report["runtimes"]["clang"]["libfuzzer"]
    assert report["compilers"]["clang"]["platform"] == "linux/amd64"
    assert report["warnings"] == ["clang_host_platform_mismatch"]


@pytest.mark.parametrize("command,value", [(("gcc", "-dumpfullversion"), "secret output\n"),
                                         (("gcc", "-dumpmachine"), "bad target;private"),
                                         (("clang", "-print-resource-dir"), "relative/private"),
                                         (("clang", "-print-resource-dir"), "/private\nsecret"),
                                         (("gcc", "-dumpmachine"), subprocess.TimeoutExpired("private", 5))])
def test_bad_metadata_is_static_failure_without_output_dump(monkeypatch, command, value):
    fixture(monkeypatch, overrides={command: value})
    report = probe.inspect_toolchain()
    assert not report["ready"] and command[0] + "_metadata_failed" in report["errors"]
    assert "secret" not in json.dumps(report) and "private" not in json.dumps(report)


def test_unknown_machine_and_invalid_requested_target_not_echoed(monkeypatch):
    fixture(monkeypatch, machine="private-secret")
    report = probe.inspect_toolchain("private-secret")
    assert "host_platform_unknown" in report["errors"] and "requested_target_invalid" in report["errors"]
    assert report["platform"]["mismatch"] is None
    assert "private-secret" not in json.dumps(report)


@pytest.mark.parametrize("stdout,code,ready,error", [(b"private" * 1000, 0, True, "metadata_command_failed"),
                                                    (b"\xff", 0, True, "metadata_invalid"),
                                                    (b"private", 1, True, "metadata_command_failed"),
                                                    (b"", 0, False, "metadata_command_failed"),
                                                    (b"aarch64-linux-gnu\n", 0, True, None)])
def test_metadata_subprocess_limits_and_never_dump_stderr(monkeypatch, stdout, code, ready, error):
    calls = []
    reads = []
    killed = []
    position = 0
    status = None

    def wait(**kwargs):
        nonlocal status
        status = code
        return code

    def kill():
        nonlocal status
        killed.append(True)
        status = -9

    process = SimpleNamespace(stdout=SimpleNamespace(fileno=lambda: 42, close=lambda: None),
                              wait=wait, poll=lambda: status, kill=kill)

    def popen(command, **kwargs):
        calls.append((command, kwargs))
        return process

    def read(fd, size):
        nonlocal position
        reads.append(size)
        data = stdout[position:position + size]
        position += len(data)
        return data

    class Selector:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def register(self, *args):
            return None

        def select(self, timeout):
            return [True] if ready else []

    monkeypatch.setattr(probe.subprocess, "Popen", popen)
    monkeypatch.setattr(probe.os, "read", read)
    monkeypatch.setattr(probe.os, "set_blocking", lambda *args: None)
    monkeypatch.setattr(probe.selectors, "DefaultSelector", Selector)
    if error:
        with pytest.raises(probe.ToolchainError, match="^" + error + "$"):
            probe._metadata(["gcc", "-dumpmachine"])
    else:
        assert probe._metadata(["gcc", "-dumpmachine"]) == "aarch64-linux-gnu"
    assert calls[0] == (["/usr/bin/gcc", "-dumpmachine"],
                       {"stdout": probe.subprocess.PIPE, "stderr": probe.subprocess.DEVNULL})
    assert all(size <= probe.MAX_METADATA + 1 for size in reads)
    assert bool(killed) == (not ready or len(stdout) > probe.MAX_METADATA)


def test_metadata_command_whitelist_rejects_arbitrary_executable_or_flags():
    for command in (["sh", "-c", "private"], ["gcc", "private"], ["gcc", "-o", "private"]):
        with pytest.raises(probe.ToolchainError, match="^metadata_command_invalid$"):
            probe._metadata(command)


def test_cli_success_mismatch_and_static_invalid_arguments(monkeypatch, capsys):
    fixture(monkeypatch)
    assert probe.main(["--target-platform", "linux/amd64"]) == 0
    assert json.loads(capsys.readouterr().out)["warnings"] == ["target_platform_mismatch"]
    assert probe.main(["--private-secret"]) == 1
    captured = capsys.readouterr()
    assert "private-secret" not in captured.out + captured.err
    assert json.loads(captured.out)["errors"] == ["toolchain_probe_failed"]


def test_dockerfile_extends_explicit_base_only_generic_packages_restores_node():
    dockerfile = Path(probe.__file__).with_name("Dockerfile.agent-tools").read_text()
    assert "ARG BASE_IMAGE\nFROM ${BASE_IMAGE}" in dockerfile
    assert "USER root" in dockerfile and dockerfile.rstrip().endswith("USER node")
    assert "clang libclang-rt-dev cmake ninja-build llvm lld gdb" in dockerfile
    assert "--no-install-recommends" in dockerfile and "apt-get upgrade" not in dockerfile
    assert "COPY " not in dockerfile and "ADD " not in dockerfile
    assert not any(text in dockerfile for text in ("repo-vul", "repo-fix", "/tmp/poc", "@openai/"))
