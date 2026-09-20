"""Host build-only manifest generator; no runtime-config input in the runner."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sanitizer_runner as runner


def build_manifest():
    files = dict(runner.FIXED_FILES)
    versions = {}
    build_env = {"PATH": "/usr/bin:/bin", "LANG": "C", "LC_ALL": "C", "HOME": "/nonexistent"}
    for name in ("gcc", "clang"):
        executable = shutil.which(name, path=build_env["PATH"])
        if executable is None:
            raise runner.SanitizerError("runtime_compiler_missing")
        executable = str(Path(executable).resolve())
        if not executable.startswith("/usr/"):
            raise runner.SanitizerError("runtime_compiler_path_invalid")
        runner.read_trusted_file(executable)
        files[name + "_compiler"] = executable
        output = subprocess.run([executable, "--version"], env=build_env,
                                check=True, capture_output=True, text=True).stdout
        versions[name] = output.splitlines()[0]
    output = subprocess.run([files["gcc_compiler"], "-print-file-name=libasan.so"],
                            env=build_env, check=True, capture_output=True, text=True).stdout
    files["gcc_runtime"] = str(Path(output.strip()).resolve())
    if not files["gcc_runtime"].startswith("/usr/"):
        raise runner.SanitizerError("runtime_gcc_library_missing")
    entries = {role: {"path": path, "sha256": hashlib.sha256(
        runner.read_trusted_file(path)[0]).hexdigest()} for role, path in files.items()}
    data = {"schema": runner.SCHEMA,
            "platform": {"system": "Linux", "machine": platform.machine()},
            "source_revision": runner.source_revision(entries), "files": entries,
            "compiler_versions": versions, "profiles": runner.PROFILES,
            "capabilities": runner.CAPABILITIES}
    runner.validate_manifest(data)
    return data


def main():
    if len(sys.argv) != 1:
        raise runner.SanitizerError("runtime_generator_no_arguments")
    data = build_manifest()
    raw = runner.canonical_json(data) + b"\n"
    fd = os.open(runner.MANIFEST_PATH, os.O_WRONLY | os.O_CREAT | os.O_EXCL
                 | os.O_NOFOLLOW | os.O_CLOEXEC, 0o444)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(fd, view)
            if written <= 0:
                raise OSError("manifest write failed")
            view = view[written:]
        os.fsync(fd)
    finally:
        os.close(fd)
    runner.load_manifest()


if __name__ == "__main__":
    main()
