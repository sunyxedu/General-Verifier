"""The rolling TB scheduler admits up to --max-concurrent tasks, one at a
time, skips tasks that already have a result, and records each outcome."""
from __future__ import annotations

import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ROLLING = os.path.join(ROOT, "experiments", "terminal_bench", "rolling.py")

FAKE_RUN = r'''
import argparse, json, os, sys, time
ap = argparse.ArgumentParser()
ap.add_argument("--tasks-file"); ap.add_argument("--job-name")
ap.add_argument("--jobs-dir"); ap.add_argument("--n-concurrent")
ap.add_argument("--arm")
a = ap.parse_args()
trial = os.path.join(a.jobs_dir, a.job_name, a.job_name + "__Abc123")
os.makedirs(os.path.join(trial, "verifier"), exist_ok=True)
with open(os.path.join(a.jobs_dir, "concurrency.log"), "a") as f:
    f.write("start %s\n" % a.job_name)
time.sleep(0.6)
open(os.path.join(trial, "verifier", "reward.txt"), "w").write("1")
json.dump({}, open(os.path.join(trial, "result.json"), "w"))
with open(os.path.join(a.jobs_dir, "concurrency.log"), "a") as f:
    f.write("end %s\n" % a.job_name)
'''


def run(tmp_path, tasks, max_concurrent):
    fake = tmp_path / "fake_run.py"
    fake.write_text(FAKE_RUN)
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("# header\n" + "\n".join(tasks) + "\n")
    batch = tmp_path / "batch"
    done = subprocess.run(
        [sys.executable, ROLLING, "--tasks-file", str(tasks_file),
         "--batch-dir", str(batch), "--max-concurrent", str(max_concurrent),
         "--min-free-gb", "0", "--hard-free-gb", "0", "--pull-timeout", "0",
         "--poll-seconds", "0.05", "--run-py", str(fake), "--keep-images",
         "--", "--arm", "gv-codex"],
        capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr
    return batch


def test_all_tasks_run_within_the_concurrency_cap(tmp_path):
    batch = run(tmp_path, ["a", "b", "c", "d", "e"], 2)
    live = peak = 0
    for line in (batch / "concurrency.log").read_text().splitlines():
        live += 1 if line.startswith("start") else -1
        peak = max(peak, live)
    assert peak == 2
    log = (batch / "rolling.log").read_text()
    assert all(f"done {t} reward=1" in log for t in "abcde")


def test_a_rerun_skips_tasks_that_already_have_a_result(tmp_path):
    batch = run(tmp_path, ["a", "b"], 2)
    batch = run(tmp_path, ["a", "b", "c"], 2)
    starts = [l for l in (batch / "concurrency.log").read_text().splitlines()
              if l.startswith("start")]
    assert starts.count("start a") == 1 and starts.count("start c") == 1


def test_a_restarted_scheduler_adopts_a_task_still_running(tmp_path):
    import time
    fake = tmp_path / "fake_run.py"
    fake.write_text(FAKE_RUN)
    batch = tmp_path / "batch"
    batch.mkdir()
    # a task a previous scheduler started: its run.py outlives the scheduler
    slow = tmp_path / "terminal_bench" / "run.py"
    slow.parent.mkdir()
    slow.write_text(FAKE_RUN.replace("time.sleep(0.6)", "time.sleep(2)"))
    orphan = subprocess.Popen(
        [sys.executable, str(slow), "--tasks-file", "x", "--job-name", "a",
         "--jobs-dir", str(batch), "--n-concurrent", "1", "--arm", "gv-codex"],
        start_new_session=True)
    time.sleep(0.3)
    tasks_file = tmp_path / "tasks.txt"
    tasks_file.write_text("a\nb\n")
    done = subprocess.run(
        [sys.executable, ROLLING, "--tasks-file", str(tasks_file),
         "--batch-dir", str(batch), "--max-concurrent", "2",
         "--min-free-gb", "0", "--hard-free-gb", "0", "--pull-timeout", "0",
         "--poll-seconds", "0.05", "--run-py", str(fake), "--keep-images",
         "--", "--arm", "gv-codex"],
        capture_output=True, text=True, timeout=120)
    orphan.wait(timeout=30)
    assert done.returncode == 0, done.stderr
    log = (batch / "rolling.log").read_text()
    assert "adopted running a" in log and "start a " not in log
    starts = [l for l in (batch / "concurrency.log").read_text().splitlines()
              if l.startswith("start")]
    assert starts.count("start a") == 1
    assert "a\t1" in (batch / "summary.tsv").read_text()
