"""A running check cannot fill the disk: its scratch directory has a size
budget, and a check that outgrows it is stopped and fails like any other
execution failure."""
from __future__ import annotations

import os
import subprocess
import sys
import time

import pytest

import verifier.execution.scratch_budget as sb
import verifier.llm.codex_backend as cb
from verifier.core.model import Obligation
from verifier.execution.discharge import AgentDischarger
from verifier.execution.instruments import ArtifactStore
from verifier.llm.schema import FixtureBackend, SeatScratchExceeded, SeatTimeout

WRITER = ("import os,time\n"
          "with open('blob','wb') as f:\n"
          "    for _ in range(400):\n"
          "        f.write(os.urandom(1 << 18)); f.flush(); time.sleep(0.01)\n")


def test_the_walk_stops_as_soon_as_the_budget_is_passed(tmp_path):
    (tmp_path / "a").write_bytes(b"x" * 1000)
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "b").write_bytes(b"x" * 1000)
    assert sb.exceeds(str(tmp_path), 1500)
    assert not sb.exceeds(str(tmp_path), 5000)


def test_a_seat_that_outgrows_its_scratch_is_stopped(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "POLL_SECONDS", 0.2)
    t0 = time.time()
    with pytest.raises(SeatScratchExceeded):
        cb._run_codex_process([sys.executable, "-c", WRITER], cwd=str(tmp_path),
                              stdin=subprocess.DEVNULL, timeout=60,
                              watch_dir=str(tmp_path), watch_cap=2 << 20)
    assert time.time() - t0 < 10
    # a budget, like time: never retried with the same prompt
    assert issubclass(SeatScratchExceeded, SeatTimeout)


def test_a_script_that_outgrows_its_scratch_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(sb, "POLL_SECONDS", 0.2)
    monkeypatch.setenv("GV_SCRATCH_CAP_GB", str(2 / 1024))     # 2 MB
    discharger = AgentDischarger(FixtureBackend({}), ArtifactStore(str(tmp_path / "art")),
                                 workdir_base=str(tmp_path / "work"), state_dir=str(tmp_path))
    method = {"procedure": "write a lot", "script": "#!/usr/bin/env python3\n" + WRITER,
              "basis": "record", "strength": "decisive"}
    run = discharger.run_script({"id": "c1", "text": "claim"},
                                Obligation("c1", "the blob is small"), method, tag="t")
    assert "outgrew" in run.launch_error
