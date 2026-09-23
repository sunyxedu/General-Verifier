"""A repository below the task root is as visible as one at the root, and
the verifier's instruments read repositories whoever owns them."""
import json
import os
import subprocess

from verifier import baseline
from verifier.core.world import load_world_view

ENV = {**os.environ, "GIT_AUTHOR_NAME": "a", "GIT_AUTHOR_EMAIL": "a@b",
       "GIT_COMMITTER_NAME": "a", "GIT_COMMITTER_EMAIL": "a@b"}


def _repo(path):
    os.makedirs(path, exist_ok=True)
    subprocess.run(["git", "init", "-q", path], check=True, env=ENV)
    subprocess.run(["git", "-C", path, "commit", "-q", "--allow-empty",
                    "-m", "x"], check=True, env=ENV)


def test_nested_repositories_are_recorded_and_described(tmp_path):
    root = tmp_path / "app"
    _repo(str(root / "site"))
    _repo(str(root / "tools" / "lib"))
    state = tmp_path / ".gv"
    out = baseline.snapshot(str(root), str(state / "baseline"))
    assert out["repositories"] == ["site", "tools/lib"]
    recorded = json.loads((state / "baseline" / "repositories.json").read_text())
    assert set(recorded) == {"site", "tools/lib"}
    assert recorded["site"]["head"]
    view = load_world_view(str(state)).describe()
    assert view["repositories"] == [f"{root}/site", f"{root}/tools/lib"]
    assert "never assume the task root itself is one" in view["repositories_note"]


def test_a_repository_at_the_root_is_the_root(tmp_path):
    root = tmp_path / "app"
    _repo(str(root))
    state = tmp_path / ".gv"
    baseline.snapshot(str(root), str(state / "baseline"))
    assert load_world_view(str(state)).describe()["repositories"] == [str(root)]


def test_no_repository_means_no_listing(tmp_path):
    root = tmp_path / "app"
    root.mkdir()
    state = tmp_path / ".gv"
    baseline.snapshot(str(root), str(state / "baseline"))
    assert "repositories" not in load_world_view(str(state)).describe()


def test_instruments_read_any_repository_without_overriding_a_config(monkeypatch):
    from verifier import mcp_server
    env = {}
    mcp_server._instruments_read_any_repository(env)
    assert env == {"GIT_CONFIG_COUNT": "1", "GIT_CONFIG_KEY_0": "safe.directory",
                   "GIT_CONFIG_VALUE_0": "*"}
    preset = {"GIT_CONFIG_COUNT": "2"}
    mcp_server._instruments_read_any_repository(preset)
    assert preset == {"GIT_CONFIG_COUNT": "2"}
