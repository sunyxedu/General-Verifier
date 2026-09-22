"""The codex GV mount keeps the harness's own state outside the world."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "experiments"))
from common import solver  # noqa: E402


def test_codex_home_lives_outside_the_delivered_world(tmp_path):
    work = tmp_path / "s0"
    work.mkdir()
    home = solver.mount_gv_codex(str(work), model="m", web=False,
                                 seat_search=False)
    assert not os.path.abspath(home).startswith(str(work) + os.sep)
    assert os.path.isfile(os.path.join(home, "config.toml"))
    assert os.path.isfile(os.path.join(home, "hooks.json"))
    assert os.path.isfile(os.path.join(home, "AGENTS.md"))


def test_gv_calls_are_serialized_and_preapproved(tmp_path):
    import tomllib
    work = tmp_path / "s0"
    work.mkdir()
    home = solver.mount_gv_codex(str(work), model="m", web=False,
                                 seat_search=False)
    gv = tomllib.loads(open(os.path.join(home, "config.toml")).read())["mcp_servers"]["gv"]
    assert gv["supports_parallel_tool_calls"] is False
    assert gv["default_tools_approval_mode"] == "approve"
    assert gv["tool_timeout_sec"] >= 3600
