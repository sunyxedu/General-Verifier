"""Offline pinned-Claude configuration smoke; never starts a model turn.

Run ``python -m experiments.cybergym.preflight`` before an experiment. Every
check is a fresh networkless container started straight from the pinned image
(no source, auth, or experiment home mounts) plus one host-side stat of the
Claude Code OAuth login. Acceptance is configuration validation, not a claim
that sandbox enforcement or a model's behavior has been measured. Only
``claude --version`` and fixed import/manifest probes are permitted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys

from .model_config import MODEL
from .run import CLAUDE_CREDENTIALS, IMAGE

PURPOSE = "cybergym-role-config-preflight"
EXPECTED_CLAUDE_VERSION = "2.1.257"
SDK_IMPORTS = ("claude_agent_sdk", "mcp", "uvicorn", "pydantic")
SANITIZER_MANIFEST = "/usr/local/lib/gv-sanitizer/manifest.json"
MAX_OUTPUT = 2 * 1024 * 1024


class PreflightError(ValueError):
    """Static error codes only; never echo a CLI response or configuration."""


def _docker(args, timeout=5):
    try:
        return subprocess.run(["docker", *args], capture_output=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        raise PreflightError("docker_command_failed") from None


def _image_check(entrypoint, argv, timeout):
    """One disposable networkless container per check; nothing is mounted."""
    result = _docker(["run", "--rm", "--network", "none", "--user", "1000:1000",
                      "--read-only", "--label", "gv.purpose=" + PURPOSE,
                      "--entrypoint", entrypoint, IMAGE, *argv], timeout=timeout)
    if len(result.stdout) > MAX_OUTPUT or len(result.stderr) > MAX_OUTPUT:
        raise PreflightError("pinned_image_check_output_too_large")
    return result


def _claude_version(timeout):
    result = _image_check("claude", ["--version"], timeout)
    if result.returncode:
        raise PreflightError("pinned_claude_version_unavailable")
    try:
        version = result.stdout.decode("utf-8").strip()
    except UnicodeDecodeError:
        raise PreflightError("pinned_claude_version_mismatch") from None
    if len(version) > 200 or not re.fullmatch(
            re.escape(EXPECTED_CLAUDE_VERSION) + r"(\s.*)?", version, flags=re.DOTALL):
        raise PreflightError("pinned_claude_version_mismatch")
    return version


def _sdk_imports(timeout):
    probe = "import " + ", ".join(SDK_IMPORTS)
    result = _image_check("python3", ["-c", probe], timeout)
    if result.returncode:
        raise PreflightError("pinned_sdk_imports_unavailable")


def _sanitizer_manifest(timeout):
    probe = ("import json; "
             "json.load(open('" + SANITIZER_MANIFEST + "', encoding='utf-8'))")
    result = _image_check("python3", ["-c", probe], timeout)
    if result.returncode:
        raise PreflightError("pinned_sanitizer_manifest_unavailable")


def _host_credentials():
    # Existence and integrity fingerprint only; OAuth contents never enter the
    # report.
    try:
        raw = CLAUDE_CREDENTIALS.read_bytes()
    except OSError:
        raise PreflightError("claude_credentials_missing") from None
    if not raw:
        raise PreflightError("claude_credentials_missing")
    return {"present": True, "sha256": hashlib.sha256(raw).hexdigest()}


def run(timeout=20.0):
    if type(timeout) not in (int, float) or not 5 <= timeout <= 60:
        raise PreflightError("invalid_preflight_timeout")
    per_check = max(1.0, (timeout - 5) / 3)
    version = _claude_version(per_check)
    _sdk_imports(per_check)
    _sanitizer_manifest(per_check)
    return {"schema": "gv-cybergym-preflight-v1", "image": IMAGE, "model": MODEL,
            "model_calls": 0, "claude_version": version,
            "claude_version_expected": EXPECTED_CLAUDE_VERSION,
            "sdk_imports": list(SDK_IMPORTS),
            "sanitizer_manifest": SANITIZER_MANIFEST,
            "credentials": _host_credentials()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=20.0)
    args = parser.parse_args(argv)
    try:
        report = run(args.timeout)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as error:
        code = str(error) if isinstance(error, PreflightError) else "unexpected_preflight_error"
        print("cybergym-preflight " + code, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
