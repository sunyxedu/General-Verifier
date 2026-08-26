import hashlib
import json
import subprocess
from pathlib import Path

from experiments.terminal_bench import runtime_snapshot


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fake_proc(root: Path, pid: int = 123) -> None:
    (root / "net").mkdir(parents=True)
    for name in runtime_snapshot.PROC_NET_FILES:
        (root / "net" / name).write_text("header\n", encoding="utf-8")
    (root / "sys/kernel/random").mkdir(parents=True)
    (root / "sys/kernel/random/boot_id").write_text("boot-id\n")
    (root / "uptime").write_text("1.0 2.0\n")
    for name in ("1", "self", str(pid)):
        process = root / name
        (process / "ns").mkdir(parents=True)
        (process / "fd").mkdir()
        for namespace in runtime_snapshot.NAMESPACE_NAMES:
            (process / "ns" / namespace).symlink_to(f"{namespace}:[42]")
    process = root / str(pid)
    (process / "stat").write_text(f"{pid} (server) S 1 1 1\n")
    (process / "status").write_text("Name:\tserver\n")
    (process / "cmdline").write_bytes(b"server\x00--port\x005328\x00")
    (process / "comm").write_text("server\n")
    (process / "exe").symlink_to("/usr/bin/server")
    (process / "cwd").symlink_to("/app")
    (process / "root").symlink_to("/")
    (process / "fd" / "3").symlink_to("socket:[98765]")


def test_runtime_snapshot_captures_proc_relationships_without_environ(tmp_path):
    output = tmp_path / "snapshot"
    proc_root = tmp_path / "proc"
    _fake_proc(proc_root)

    manifest = runtime_snapshot.capture(output, proc_root=proc_root)

    assert manifest["schema"] == runtime_snapshot.SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["capture_boundary"] == (
        "after solver return, before official verifier")
    assert manifest["privacy"]["process_environments_captured"] is False
    assert manifest["process_count"] == len(manifest["processes"])
    assert manifest["process_count"] > 0

    by_pid = {record["pid"]: record for record in manifest["processes"]}
    assert 123 in by_pid
    own = output / by_pid[123]["path"]
    assert by_pid[123]["sha256"] == _sha(own)
    process = json.loads(own.read_text(encoding="utf-8"))
    assert "environ" not in process
    assert process["pid"] == 123
    assert process["cmdline"]["status"] == "ok"
    assert process["namespaces"]["net"]["status"] == "ok"
    assert process["fds"] == [{
        "fd": "3", "status": "ok", "target": "socket:[98765]"}]

    for phase in ("before", "after"):
        records = manifest["network"][phase]
        assert {record["name"] for record in records} == set(
            runtime_snapshot.PROC_NET_FILES)
        for record in records:
            if record["status"] != "ok":
                continue
            path = output / record["path"]
            assert path.is_file()
            assert record["sha256"] == _sha(path)

    collector = output / manifest["collector"]["path"]
    assert manifest["collector"]["sha256"] == _sha(collector)
    persisted = json.loads((output / "manifest.json").read_text())
    assert persisted["schema"] == runtime_snapshot.SCHEMA


def test_runtime_snapshot_replaces_candidate_supplied_output(tmp_path):
    output = tmp_path / "snapshot"
    proc_root = tmp_path / "proc"
    _fake_proc(proc_root)
    output.mkdir()
    (output / "spoofed.json").write_text("candidate data", encoding="utf-8")

    runtime_snapshot.capture(output, proc_root=proc_root)

    assert not (output / "spoofed.json").exists()
    assert (output / "manifest.json").is_file()


def test_posix_shell_fallback_is_syntax_valid_and_never_mentions_environ():
    collector = Path(runtime_snapshot.__file__).with_name("runtime_snapshot.sh")

    subprocess.run(["/bin/sh", "-n", collector], check=True)
    source = collector.read_text(encoding="utf-8")
    assert '"collector_runtime": "posix-sh-fallback"' in source
    assert "/proc/net/" in source
    assert '"$proc_dir"/fd/*' in source
    assert "/proc/self/ns/net" in source
    assert "/proc/1/ns/net" in source
    assert "/proc/*/environ" not in source


def test_tiered_codex_uploads_and_invokes_shell_fallback():
    source = Path(runtime_snapshot.__file__).with_name(
        "codex_agent.py").read_text(encoding="utf-8")

    assert 'Path(__file__).with_name("runtime_snapshot.sh")' in source
    assert "_REMOTE_SHELL_COLLECTOR" in source
    assert 'f"  /bin/sh {shlex.quote(_REMOTE_SHELL_COLLECTOR)} "' in source
    assert "no python interpreter found" not in source
