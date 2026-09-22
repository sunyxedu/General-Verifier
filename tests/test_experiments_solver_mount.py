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
