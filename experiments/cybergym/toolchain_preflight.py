"""Model-independent, metadata-only generic Linux toolchain capability probe.

Run inside the future solver image with ``--target-platform linux/amd64``.
No Docker, network, model, task input, compiler build or target execution occurs.
Presence does not prove runtime execution: this probe deliberately has no smoke
test. Platform mismatch is declared, not an automatic source-only rejection.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import selectors
import shutil
import subprocess
import time

TOOLS = ("python3", "gcc", "g++", "make", "pkg-config", "clang", "clang++",
         "cmake", "ninja", "llvm-ar", "llvm-symbolizer", "ld.lld", "gdb")
TARGETS = frozenset({"linux/amd64", "linux/arm64"})
ARCHES = {"x86_64": "amd64", "amd64": "amd64", "aarch64": "arm64", "arm64": "arm64"}
RUNTIME_ARCHES = {"linux/amd64": "x86_64", "linux/arm64": "aarch64"}
MAX_METADATA = 4096
METADATA_FLAGS = {"gcc": frozenset({"-dumpfullversion", "-dumpmachine", "-print-file-name=libasan.so",
                                   "-print-file-name=libubsan.so"}),
                  "clang": frozenset({"-dumpversion", "-dumpmachine", "-print-resource-dir"})}


class ToolchainError(ValueError):
    """Static probe codes only; never include command output or environment."""


def _metadata(command: list[str]) -> str:
    if len(command) != 2 or command[0] not in METADATA_FLAGS or command[1] not in METADATA_FLAGS[command[0]]:
        raise ToolchainError("metadata_command_invalid")
    process = None
    try:
        # Apt-based image contract: avoid source-controlled PATH executables.
        process = subprocess.Popen(["/usr/bin/" + command[0], command[1]],
                                   stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        deadline = time.monotonic() + 5
        output = bytearray()
        os.set_blocking(process.stdout.fileno(), False)
        with selectors.DefaultSelector() as selector:
            selector.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise ToolchainError("metadata_command_failed")
                chunk = os.read(process.stdout.fileno(), MAX_METADATA + 1 - len(output))
                if not chunk:
                    if process.wait(timeout=max(0.001, deadline - time.monotonic())):
                        raise ToolchainError("metadata_command_failed")
                    break
                output.extend(chunk)
                if len(output) > MAX_METADATA:
                    raise ToolchainError("metadata_command_failed")
        return output.decode("utf-8", errors="strict").strip()
    except (OSError, subprocess.SubprocessError) as exc:
        raise ToolchainError("metadata_command_failed") from exc
    except UnicodeError as exc:
        raise ToolchainError("metadata_invalid") from exc
    finally:
        if process is not None:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=1)
            process.stdout.close()


def _absolute_path(value: str) -> bool:
    return bool(value) and os.path.isabs(value) and not any(
        character in value for character in ("\x00", "\n", "\r"))


def inspect_toolchain(target_platform: str | None = None) -> dict:
    commands = {name: shutil.which(name) is not None for name in TOOLS}
    host = "linux/" + ARCHES.get(platform.machine().lower(), "unknown") if (
        platform.system() == "Linux") else "unknown"
    errors = ["command_missing_" + name.replace("+", "p").replace("-", "_").replace(".", "_")
              for name, present in commands.items() if not present]
    requested = target_platform if target_platform in TARGETS else None
    if target_platform is not None and requested is None:
        errors.append("requested_target_invalid")
    if host not in TARGETS:
        errors.append("host_platform_unknown")
    compilers = {}
    warnings = []
    runtimes = {"gcc": {"asan": False, "ubsan": False},
                "clang": {"asan": False, "ubsan": False, "libfuzzer": False}}
    for name in ("gcc", "clang"):
        if not commands[name]:
            continue
        try:
            version = _metadata([name, "-dumpfullversion" if name == "gcc" else "-dumpversion"])
            target = _metadata([name, "-dumpmachine"])
            if not re.fullmatch(r"[0-9]+(?:\.[0-9]+){0,3}", version) or not re.fullmatch(
                    r"[A-Za-z0-9_+.-]{1,128}", target):
                raise ToolchainError("metadata_invalid")
            parts = target.lower().split("-")
            compiler_arch = ARCHES.get(parts[0])
            if compiler_arch is None or "linux" not in parts:
                raise ToolchainError("metadata_invalid")
            compiler_platform = "linux/" + compiler_arch
            compilers[name] = {"version": version, "target": target, "platform": compiler_platform}
            if host in TARGETS and compiler_platform != host:
                warnings.append(name + "_host_platform_mismatch")
            if name == "gcc":
                for sanitizer, library in (("asan", "libasan.so"), ("ubsan", "libubsan.so")):
                    path = _metadata([name, "-print-file-name=" + library])
                    runtimes[name][sanitizer] = _absolute_path(path) and os.path.isfile(path)
            else:
                resource = _metadata([name, "-print-resource-dir"])
                if not _absolute_path(resource):
                    raise ToolchainError("metadata_invalid")
                arch = RUNTIME_ARCHES.get(compiler_platform)
                if arch:
                    directory = os.path.join(resource, "lib", "linux")
                    for runtime, suffixes in (("asan", ("a", "so")), ("ubsan_standalone", ("a", "so")),
                                              ("fuzzer", ("a",))):
                        key = {"ubsan_standalone": "ubsan", "fuzzer": "libfuzzer"}.get(runtime, runtime)
                        runtimes[name][key] = any(os.path.isfile(os.path.join(
                            directory, f"libclang_rt.{runtime}-{arch}.{suffix}")) for suffix in suffixes)
        except (ToolchainError, OSError, subprocess.SubprocessError):
            errors.append(name + "_metadata_failed")
        for runtime, present in runtimes[name].items():
            if not present:
                errors.append(name + "_runtime_missing_" + runtime)
    mismatch = requested is not None and host in TARGETS and requested != host
    if mismatch:
        warnings.append("target_platform_mismatch")
    return {"schema": "gv-cybergym-toolchain-preflight-v1", "ready": not errors,
            "platform": {"host": host, "requested_target": requested,
                         "mismatch": mismatch if requested and host in TARGETS else None},
            "commands": commands, "compilers": compilers, "runtimes": runtimes,
            "errors": sorted(errors), "warnings": sorted(warnings),
            "metadata_only": True, "smoke_tested": False,
            "model_calls": 0, "task_data_used": False, "network_used": False}


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ToolchainError("invalid_arguments")


def main(argv: list[str] | None = None) -> int:
    parser = _Parser(description=__doc__)
    parser.add_argument("--target-platform")
    try:
        args = parser.parse_args(argv)
        report = inspect_toolchain(args.target_platform)
    except (ToolchainError, OSError, subprocess.SubprocessError):
        print(json.dumps({"schema": "gv-cybergym-toolchain-preflight-v1", "ready": False,
                          "errors": ["toolchain_probe_failed"]}))
        return 1
    print(json.dumps(report, sort_keys=True))
    return 0 if report["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
