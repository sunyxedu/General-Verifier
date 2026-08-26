from pathlib import Path

import pytest

from experiments.common import harbor_task_artifacts as task_artifacts


def _task_package(tmp_path: Path, instruction: str, config: str = "artifacts = []\n") -> Path:
    task = tmp_path / "task"
    task.mkdir()
    (task / "instruction.md").write_text(instruction, encoding="utf-8")
    (task / "task.toml").write_text(config, encoding="utf-8")
    # These sentinels make it observable if discovery ever starts traversing
    # private benchmark material: discovery succeeds without opening them.
    (task / "tests").mkdir()
    (task / "solution").mkdir()
    (task / "tests/forbidden").write_text("not toml: [", encoding="utf-8")
    (task / "solution/forbidden").write_text("not utf8: \udcff", errors="surrogatepass")
    return task


def test_instruction_discovery_finds_specific_system_paths_not_urls():
    text = """
    Write `/app/result.txt` and /app/.result-cache; configure /etc/nginx/nginx.conf.
    Serve https://localhost:8443/index.html from /var/www/html.
    Host git@localhost:/git/project, never collect /etc or /var/log.
    Ignore /proc/self/environ, /root/.ssh/id_rsa, and /etc/shadow.
    """

    assert task_artifacts.extract_instruction_paths(text) == [
        "/app/result.txt",
        "/app/.result-cache",
        "/etc/nginx/nginx.conf",
        "/var/www/html",
        "/git/project",
    ]


def test_instruction_audit_explains_url_glob_and_broad_root_rejections():
    records = task_artifacts.audit_instruction_paths(
        "Fetch https://host:8443/dev/index.html; don't collect /etc; "
        "write /tmp/*.json and /etc/tool/out.conf."
    )

    assert [(record.raw, record.decision, record.reason) for record in records] == [
        ("/dev/index.html", "rejected", "url_path"),
        ("/etc;", "rejected", "broad_root"),
        ("/tmp/*.json", "rejected", "shell_or_glob"),
        ("/etc/tool/out.conf.", "accepted", "specific_absolute_path"),
    ]


def test_instruction_audit_rejects_routes_math_home_and_runtime_socket():
    records = task_artifacts.audit_instruction_paths(
        'Expose an endpoint at "/sentiment"; compute argv[1]/2; source ~/.bashrc; '
        "use /tmp/monitor.sock; store at /build/."
    )

    assert [(record.normalized, record.reason) for record in records] == [
        ("/sentiment", "uri_route"),
        ("/2", "numeric_expression"),
        ("/.bashrc", "home_relative"),
        ("/tmp/monitor.sock", "runtime_socket_unsupported"),
        ("/build", "broad_top_level"),
    ]


def test_markdown_emphasis_is_not_mistaken_for_a_glob():
    records = task_artifacts.audit_instruction_paths(
        "**Output `/app/compress.py`** and /app/decompress.py**:"
    )

    assert [record.normalized for record in records] == [
        "/app/compress.py",
        "/app/decompress.py",
    ]
    assert all(record.decision == "accepted" for record in records)


def test_task_package_reads_only_public_instruction_and_artifact_fields(tmp_path):
    task = _task_package(
        tmp_path,
        "Read /app/input.png and write /app/output.txt; configure /etc/x/y.conf.",
        """
artifacts = ["/declared/result.json", {source = "/var/log/tool/result.log"}]

[environment]
docker_image = "example/image:latest"
""",
    )

    assert task_artifacts.discover_task_package(task) == (
        "/declared/result.json",
        "/var/log/tool/result.log",
        "/app/input.png",
        "/app/output.txt",
        "/etc/x/y.conf",
    )


def test_final_environment_workdir_captures_relative_deliverables(tmp_path):
    task = _task_package(
        tmp_path,
        "Complete proof.v and compile proof.vo.",
    )
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nWORKDIR /workspace\nCOPY proof.v .\n",
        encoding="utf-8",
    )

    accepted, audit = task_artifacts.discover_task_package_audit(task)

    assert accepted == ("/workspace",)
    workdir = [
        record for record in audit
        if record.source == "environment/Dockerfile:WORKDIR"
    ]
    assert len(workdir) == 1
    assert workdir[0].decision == "accepted"
    assert workdir[0].reason == "approved_root"


def test_only_final_docker_stage_workdir_is_captured(tmp_path):
    task = _task_package(tmp_path, "Write result.txt.")
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM ubuntu AS build\nWORKDIR /workspace\n"
        "FROM ubuntu\nWORKDIR /app\nWORKDIR output\n",
        encoding="utf-8",
    )

    assert task_artifacts.discover_task_package(task) == ("/app/output",)


def test_dynamic_environment_workdir_is_audited_not_guessed(tmp_path):
    task = _task_package(tmp_path, "Write result.txt.")
    environment = task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM ubuntu\nARG ROOT=/workspace\nWORKDIR $ROOT\n",
        encoding="utf-8",
    )

    accepted, audit = task_artifacts.discover_task_package_audit(task)

    assert accepted == ()
    workdir = [
        record for record in audit
        if record.source == "environment/Dockerfile:WORKDIR"
    ]
    assert len(workdir) == 1
    assert workdir[0].decision == "rejected"
    assert workdir[0].reason == "dynamic_workdir"


def test_multistep_public_instructions_are_included(tmp_path):
    task = _task_package(
        tmp_path,
        "Prepare /app/first.txt.",
        """
artifacts = []
steps = [{name = "finish", artifacts = ["/srv/output/final.json"]}]
""",
    )
    step = task / "steps/finish"
    step.mkdir(parents=True)
    (step / "instruction.md").write_text(
        "Also write /etc/tool/final.conf.", encoding="utf-8"
    )

    assert task_artifacts.discover_task_package(task) == (
        "/srv/output/final.json",
        "/app/first.txt",
        "/etc/tool/final.conf",
    )


def test_multistep_package_does_not_require_unused_root_instruction(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    (task / "task.toml").write_text(
        'artifacts = []\nsteps = [{name = "only", artifacts = []}]\n',
        encoding="utf-8",
    )
    step = task / "steps/only"
    step.mkdir(parents=True)
    (step / "instruction.md").write_text(
        "Write /srv/result/final.txt.", encoding="utf-8"
    )

    assert task_artifacts.discover_task_package(task) == ("/srv/result/final.txt",)


def test_public_metadata_symlink_is_rejected(tmp_path):
    task = tmp_path / "task"
    task.mkdir()
    outside = tmp_path / "outside.md"
    outside.write_text("write /root/secret", encoding="utf-8")
    (task / "instruction.md").symlink_to(outside)
    (task / "task.toml").write_text("artifacts = []\n", encoding="utf-8")

    with pytest.raises(task_artifacts.ArtifactDiscoveryError, match="symlink"):
        task_artifacts.discover_task_package(task)


def test_merge_keeps_explicit_root_and_safe_nonoverlapping_references():
    referenced = {
        "one": ["/app/a.txt", "/etc/nginx/nginx.conf", "/var/www/html/index.html"],
        "two": ["/var/www/html", "/etc/nginx/nginx.conf"],
    }

    assert task_artifacts.merge_artifact_paths(["/app"], referenced) == (
        "/app",
        "/etc/nginx/nginx.conf",
        "/var/www/html/index.html",
    )


@pytest.mark.parametrize(
    "path",
    ["/", "/etc", "/var/log", "/proc/cpuinfo", "/root/x", "/usr/local/toolchain"],
)
def test_unsafe_explicit_root_fails_closed(path):
    with pytest.raises(task_artifacts.ArtifactDiscoveryError, match="unsafe explicit"):
        task_artifacts.merge_artifact_paths([path], {})


def test_dry_resolution_invokes_harbor_interpreter_without_download(monkeypatch, tmp_path):
    harbor = tmp_path / "harbor"
    interpreter = tmp_path / "python"
    interpreter.write_text("", encoding="utf-8")
    harbor.write_text(f"#!{interpreter}\n", encoding="utf-8")
    harbor.chmod(0o755)

    def fake_run(command, **_kwargs):
        assert "--download" not in command
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            '{"dataset":"org/data@sha256:abc",'
            '"task_packages":{"org/t":"/cache/t"},'
            '"referenced_by_task":{"org/t":["/etc/tool/out.conf"]},'
            '"audit_by_task":{"org/t":[{"source":"instruction.md",'
            '"raw":"/etc/tool/out.conf","normalized":"/etc/tool/out.conf",'
            '"decision":"accepted","reason":"specific_absolute_path"}]}}',
            encoding="utf-8",
        )
        return type("Result", (), {"returncode": 0, "stdout": "", "stderr": ""})()

    monkeypatch.setattr(task_artifacts.subprocess, "run", fake_run)
    result = task_artifacts.resolve_harbor_dataset_artifacts(
        "org/data@sha256:abc",
        ["org/t"],
        ["/app"],
        download=False,
        harbor_executable=str(harbor),
    )

    assert result.dataset == "org/data@sha256:abc"
    assert result.artifacts == ("/app", "/etc/tool/out.conf")
    assert result.audit_by_task["org/t"][0].reason == "specific_absolute_path"
