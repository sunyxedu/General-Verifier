"""Offline preflight fixtures; Docker/model calls and host credentials are mocked."""
from __future__ import annotations

import hashlib
import json
import subprocess

import pytest

from experiments.cybergym import preflight, run


@pytest.fixture
def credentials(tmp_path, monkeypatch):
    path = tmp_path / ".credentials.json"
    path.write_bytes(b'{"fixture":"oauth-sentinel"}')
    monkeypatch.setattr(preflight, "CLAUDE_CREDENTIALS", path)
    return path


def docker_fixture(monkeypatch, *, version=None, fail=None):
    calls = []

    def docker(args, timeout=5):
        calls.append((args, timeout))
        check = args[args.index("--entrypoint") + 1]
        if fail == check or fail == "all":
            return subprocess.CompletedProcess(args, 1, b"", b"private raw noise")
        if check == "claude":
            text = "2.1.257 (Claude Code)\n" if version is None else version
            return subprocess.CompletedProcess(args, 0, text.encode(), b"")
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr(preflight, "_docker", docker)
    return calls


def test_report_is_typed_networkless_and_never_echoes_credentials(credentials, monkeypatch):
    calls = docker_fixture(monkeypatch)
    report = preflight.run()
    assert report["schema"] == "gv-cybergym-preflight-v1"
    assert report["image"] == run.IMAGE == preflight.IMAGE
    assert report["model_calls"] == 0
    assert report["claude_version"] == "2.1.257 (Claude Code)"
    assert report["claude_version_expected"] == preflight.EXPECTED_CLAUDE_VERSION
    assert report["sdk_imports"] == list(preflight.SDK_IMPORTS)
    assert report["credentials"] == {"present": True,
        "sha256": hashlib.sha256(credentials.read_bytes()).hexdigest()}
    assert "oauth-sentinel" not in json.dumps(report)
    assert len(calls) == 3
    for args, _timeout in calls:
        assert args[:2] == ["run", "--rm"]
        assert args[args.index("--network") + 1] == "none"
        assert args[args.index("--user") + 1] == "1000:1000"
        assert "--read-only" in args and "--mount" not in args and "-v" not in args
        assert args[args.index("--entrypoint") + 2] == run.IMAGE
    assert calls[0][0][calls[0][0].index("--entrypoint") + 1:] == ["claude", run.IMAGE, "--version"]
    probes = [args[-1] for args, _timeout in calls
              if args[args.index("--entrypoint") + 1] == "python3"]
    assert any("claude_agent_sdk" in probe and "uvicorn" in probe for probe in probes)
    assert any(preflight.SANITIZER_MANIFEST in probe for probe in probes)


@pytest.mark.parametrize("version", ["", "2.1.256 (Claude Code)", "2.1.2570", "x" * 201])
def test_pinned_version_mismatch_fails_closed(credentials, monkeypatch, version):
    docker_fixture(monkeypatch, version=version)
    with pytest.raises(preflight.PreflightError, match="version"):
        preflight.run()


@pytest.mark.parametrize("fail,code", [
    ("claude", "pinned_claude_version_unavailable"),
    ("python3", "pinned_sdk_imports_unavailable"),
])
def test_failed_image_checks_expose_only_fixed_codes(credentials, monkeypatch, fail, code):
    docker_fixture(monkeypatch, fail=fail)
    with pytest.raises(preflight.PreflightError, match="^" + code + "$") as caught:
        preflight.run()
    assert "private raw noise" not in str(caught.value)


def test_sanitizer_manifest_check_failure_fails_closed(credentials, monkeypatch):
    calls = []

    def docker(args, timeout=5):
        calls.append(args)
        probe = args[-1]
        failed = args[args.index("--entrypoint") + 1] == "python3" \
            and preflight.SANITIZER_MANIFEST in probe
        return subprocess.CompletedProcess(args, 1 if failed else 0,
            b"2.1.257 (Claude Code)\n" if "--version" in args else b"", b"")

    monkeypatch.setattr(preflight, "_docker", docker)
    with pytest.raises(preflight.PreflightError, match="^pinned_sanitizer_manifest_unavailable$"):
        preflight.run()


def test_missing_or_empty_host_credentials_fail_before_any_container(monkeypatch, tmp_path):
    docker_fixture(monkeypatch)
    monkeypatch.setattr(preflight, "CLAUDE_CREDENTIALS", tmp_path / "absent")
    with pytest.raises(preflight.PreflightError, match="^claude_credentials_missing$"):
        preflight.run()
    empty = tmp_path / "empty"
    empty.write_bytes(b"")
    monkeypatch.setattr(preflight, "CLAUDE_CREDENTIALS", empty)
    with pytest.raises(preflight.PreflightError, match="^claude_credentials_missing$"):
        preflight.run()


def test_docker_transport_failure_is_a_static_code(credentials, monkeypatch):
    def unavailable(_args, timeout=5):
        raise preflight.PreflightError("docker_command_failed")
    monkeypatch.setattr(preflight, "_docker", unavailable)
    with pytest.raises(preflight.PreflightError, match="^docker_command_failed$"):
        preflight.run()


@pytest.mark.parametrize("timeout", [None, "20", 4.9, 61, True])
def test_invalid_timeouts_are_rejected(credentials, monkeypatch, timeout):
    docker_fixture(monkeypatch)
    with pytest.raises(preflight.PreflightError, match="^invalid_preflight_timeout$"):
        preflight.run(timeout)


def test_oversized_check_output_is_rejected(credentials, monkeypatch):
    docker_fixture(monkeypatch)
    def loud(args, timeout=5):
        return subprocess.CompletedProcess(args, 0, b"2.1.257 " + b"x" * preflight.MAX_OUTPUT, b"")
    monkeypatch.setattr(preflight, "_docker", loud)
    with pytest.raises(preflight.PreflightError, match="^pinned_image_check_output_too_large$"):
        preflight.run()


def test_main_prints_sorted_report_or_static_code(credentials, monkeypatch, capsys):
    docker_fixture(monkeypatch)
    assert preflight.main([]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["schema"] == "gv-cybergym-preflight-v1" and report["model_calls"] == 0
    monkeypatch.setattr(preflight, "CLAUDE_CREDENTIALS", credentials.with_name("absent"))
    assert preflight.main([]) == 1
    assert capsys.readouterr().err.strip() == "cybergym-preflight claude_credentials_missing"
