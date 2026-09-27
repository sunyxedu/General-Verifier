"""Run many Terminal-Bench tasks through run.py, one Harbor job per task,
admitting a new task only while the Docker disk has room.

A single `harbor run -n N` pulls every image and starts N containers at
once; on a shared server whose Docker root is small that fills the disk
before any task finishes. This scheduler instead:

  - admits a task when fewer than --max-concurrent are running and the
    Docker root has at least --min-free-gb free, one image pull at a time
    (the next admission waits until the last one's container is up, so the
    free-space reading includes it);
  - removes a finished task's image (Harbor already removed its container
    after copying its artifacts back), so the images of all tasks never
    sit on the disk together;
  - below --hard-free-gb, saves the /gv-state of the task whose container
    holds the most disk, stops it,
    moves its job directory aside and queues it again (an infrastructure
    stop, not a result: every task runs until it has one);
  - copies each running task's /gv-state into its trial directory every
    --backup-seconds, so a lost container loses minutes, not the run;
  - skips tasks whose job already has a result, and adopts tasks a previous
    scheduler left running, so a restart resumes without duplicates;
  - writes <batch-dir>/summary.tsv (task, reward) when every task is done.

Everything after `--` is passed to run.py unchanged (arm, model, effort,
service tier, ...). Example:

  uv run python experiments/terminal_bench/rolling.py \\
      --tasks-file experiments/terminal_bench/all89_tasks.txt \\
      --batch-dir /data/jobs/gv-all89-r1 --max-concurrent 16 \\
      -- --arm gv-codex --model gpt-5.6-luna --effort max --service-tier standard
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import traceback
from typing import Dict, List, Optional

HERE = os.path.dirname(os.path.abspath(__file__))
MAX_NO_RESULT = 3   # run.py exits without a trial result this often: give up
RUN_PY = os.path.join(HERE, "run.py")


def log(batch_dir: str, msg: str) -> None:
    line = time.strftime("%F %T ") + msg
    print(line, flush=True)
    with open(os.path.join(batch_dir, "rolling.log"), "a", encoding="utf-8") as f:
        f.write(line + "\n")


def read_tasks(path: str) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [l.strip() for l in f if l.strip() and not l.startswith("#")]


def free_gb(path: str) -> float:
    return shutil.disk_usage(path).free / 1e9


def docker(*args: str, timeout: float = 120) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True,
                          timeout=timeout)


def trial_dirs(job_dir: str) -> List[str]:
    if not os.path.isdir(job_dir):
        return []
    return [os.path.join(job_dir, d) for d in sorted(os.listdir(job_dir))
            if "__" in d and "." not in d
            and os.path.isdir(os.path.join(job_dir, d))]


def container_of(trial: str) -> str:
    # Harbor's compose project is the lower-cased trial name + "__env"
    return os.path.basename(trial).lower() + "__env-main-1"


def done(job_dir: str) -> bool:
    return any(os.path.exists(os.path.join(t, "result.json"))
               for t in trial_dirs(job_dir))


def reward(job_dir: str) -> str:
    for t in trial_dirs(job_dir):
        p = os.path.join(t, "verifier", "reward.txt")
        if os.path.exists(p):
            return open(p, encoding="utf-8").read().strip()
        r = os.path.join(t, "result.json")
        if os.path.exists(r):
            info = (json.load(open(r, encoding="utf-8")).get("exception_info")
                    or {})
            return "ERR:" + str(info.get("exception_type", "?"))
    return "-"


def container_sizes() -> Dict[str, float]:
    """Writable-layer size of each running container, in bytes."""
    r = docker("ps", "-s", "--format", "{{.Names}}\t{{.Size}}", timeout=300)
    units = {"B": 1, "kB": 1e3, "KB": 1e3, "MB": 1e6, "GB": 1e9, "TB": 1e12}
    sizes: Dict[str, float] = {}
    for line in r.stdout.splitlines():
        name, _, size = line.partition("\t")
        head = size.split(" ")[0]          # "10.5GB (virtual 11.1GB)"
        num = head.rstrip("kKMGTB")
        unit = head[len(num):]
        try:
            sizes[name] = float(num) * units.get(unit, 1)
        except ValueError:
            continue
    return sizes


def task_to_stop(running, sizes: Dict[str, float]):
    """The running task the disk guard stops: the one whose container's
    writable layer is largest (newest first among equals, and when no size
    is known) — stopping small tasks in turn frees little while the one
    that grew keeps growing."""
    return max(running, key=lambda x: (sizes.get(x.container() or "", 0.0),
                                       x.started))


def save_state(container: str, trial: str, name: str) -> bool:
    """/gv-state -> <trial>/artifacts/<name>, replacing the previous copy.
    A tar pipe, not docker cp: the state holds read-only directories that
    docker cp cannot fill."""
    dest = os.path.join(trial, "artifacts", name)
    tmp = dest + ".new"
    shutil.rmtree(tmp, ignore_errors=True)
    os.makedirs(tmp, exist_ok=True)
    src = subprocess.Popen(["docker", "exec", container, "tar", "-C", "/",
                            "-cf", "-", "gv-state"], stdout=subprocess.PIPE,
                           stderr=subprocess.DEVNULL)
    ok = subprocess.run(["tar", "-C", tmp, "-xf", "-"], stdin=src.stdout,
                        stderr=subprocess.DEVNULL).returncode == 0
    src.wait()
    if not ok or src.returncode != 0:
        subprocess.run(["chmod", "-R", "u+w", tmp], stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)
        return False
    if os.path.exists(dest):
        subprocess.run(["chmod", "-R", "u+w", dest], stderr=subprocess.DEVNULL)
        shutil.rmtree(dest, ignore_errors=True)
    os.replace(tmp, dest)
    return True


class Running:
    def __init__(self, task: str, proc: Optional[subprocess.Popen],
                 job_dir: str, pid: Optional[int] = None):
        self.task, self.proc, self.job_dir = task, proc, job_dir
        self.pid = proc.pid if proc is not None else pid
        self.started = time.time()
        self.image: Optional[str] = None

    def alive(self) -> bool:
        if self.proc is not None:
            return self.proc.poll() is None
        try:
            os.kill(self.pid, 0)   # adopted: re-parented to init
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        # a zombie answers the signal probe; it is still gone
        stat = subprocess.run(["ps", "-o", "stat=", "-p", str(self.pid)],
                              capture_output=True, text=True).stdout.strip()
        return bool(stat) and not stat.startswith("Z")

    def stop(self) -> None:
        for sig, wait in ((signal.SIGTERM, 60), (signal.SIGKILL, 10)):
            try:
                os.killpg(self.pid, sig)
            except ProcessLookupError:
                return
            deadline = time.time() + wait
            while self.alive() and time.time() < deadline:
                time.sleep(1)
            if not self.alive():
                return

    def trial(self) -> Optional[str]:
        trials = trial_dirs(self.job_dir)
        return trials[-1] if trials else None

    def container(self) -> Optional[str]:
        t = self.trial()
        return container_of(t) if t else None

    def container_up(self) -> bool:
        c = self.container()
        if not c:
            return False
        r = docker("inspect", "-f", "{{.State.Running}} {{.Config.Image}}", c)
        if r.returncode != 0:
            return False
        running, image = (r.stdout.split() + ["", ""])[:2]
        if image:
            self.image = image
        return running == "true"


def move_aside(path: str, tag: str) -> str:
    """Rename a job directory out of the way under the first unused
    <path>.<tag>-<n>; never fails the scheduler for a name already taken."""
    n = 1
    while os.path.exists(f"{path}.{tag}-{n}"):
        n += 1
    dest = f"{path}.{tag}-{n}"
    os.rename(path, dest)
    return dest


def adopt(batch_dir: str, tasks: List[str]) -> Dict[str, int]:
    """run.py processes a previous scheduler started for this batch and
    that still run (their sessions outlive it): task -> pid."""
    out = subprocess.run(["ps", "-eo", "pid=,args="], capture_output=True,
                         text=True).stdout
    found: Dict[str, int] = {}
    for line in out.splitlines():
        pid, _, cmd = line.strip().partition(" ")
        parts = cmd.split()
        if not any(p.endswith("terminal_bench/run.py") for p in parts[:3]):
            continue
        try:
            task = parts[parts.index("--job-name") + 1]
            jobs = parts[parts.index("--jobs-dir") + 1]
        except (ValueError, IndexError):
            continue
        if os.path.abspath(jobs) == os.path.abspath(batch_dir) and task in tasks:
            found[task] = int(pid)
    return found


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--exclude-file", default=None,
                    help="tasks (one per line) not to run here")
    ap.add_argument("--batch-dir", required=True,
                    help="parent of the per-task job directories; the Docker "
                         "daemon bind-mounts trial directories from here")
    ap.add_argument("--max-concurrent", type=int, default=8)
    ap.add_argument("--min-free-gb", type=float, default=18.0,
                    help="admit a task only with this much free on the "
                         "Docker root")
    ap.add_argument("--hard-free-gb", type=float, default=10.0,
                    help="below this, stop the newest task and queue it again")
    ap.add_argument("--docker-root", default="/var/lib/docker")
    ap.add_argument("--pull-timeout", type=float, default=900,
                    help="seconds to wait for an admitted task's container "
                         "before admitting the next")
    ap.add_argument("--backup-seconds", type=float, default=180)
    ap.add_argument("--run-py", default=RUN_PY, help=argparse.SUPPRESS)
    ap.add_argument("--poll-seconds", type=float, default=10,
                    help=argparse.SUPPRESS)
    ap.add_argument("--keep-images", action="store_true",
                    help="do not remove a finished task's image")
    ap.add_argument("run_args", nargs=argparse.REMAINDER,
                    help="-- then arguments for run.py")
    args = ap.parse_args()
    run_args = [a for a in args.run_args if a != "--"]
    os.makedirs(args.batch_dir, exist_ok=True)
    disk_path = args.docker_root if os.path.exists(args.docker_root) else "/"

    excluded = set(read_tasks(args.exclude_file)) if args.exclude_file else set()
    queue = [t for t in read_tasks(args.tasks_file) if t not in excluded]
    all_tasks = list(queue)
    queue = [t for t in queue if not done(os.path.join(args.batch_dir, t))]
    requeued: Dict[str, int] = {}
    failures: Dict[str, int] = {}
    running: List[Running] = []
    for task, pid in adopt(args.batch_dir, queue).items():
        running.append(Running(task, None, os.path.join(args.batch_dir, task),
                               pid=pid))
        queue.remove(task)
        log(args.batch_dir, f"adopted running {task} (pid {pid})")
    pulling: Optional[Running] = None
    last_backup = 0.0
    stopping = False

    def on_signal(signum, _frame):
        nonlocal stopping
        stopping = True
    signal.signal(signal.SIGTERM, on_signal)
    signal.signal(signal.SIGINT, on_signal)

    log(args.batch_dir, f"start: {len(queue)} tasks queued, max "
        f"{args.max_concurrent}, admit at >= {args.min_free_gb} GB free, "
        f"hard stop < {args.hard_free_gb} GB; run.py {' '.join(run_args)}")

    while (queue or running) and not stopping:
        try:
            # finished tasks: record, drop the image
            for r in list(running):
                if r.alive():
                    continue
                running.remove(r)
                if pulling is r:
                    pulling = None
                got = reward(r.job_dir)
                if got == "-":
                    # run.py ended without a trial result (setup failed, a
                    # stale job directory, ...): that is no result, run it again
                    failures[r.task] = failures.get(r.task, 0) + 1
                    n = failures[r.task]
                    if os.path.isdir(r.job_dir):
                        move_aside(r.job_dir, "noresult")
                    if n < MAX_NO_RESULT:
                        queue.append(r.task)
                    log(args.batch_dir, f"no result {r.task} after "
                        f"{(time.time() - r.started) / 60:.0f} min (#{n}); "
                        + ("queued again" if n < MAX_NO_RESULT else
                           f"given up after {n}, see {r.task}.log"))
                else:
                    log(args.batch_dir, f"done {r.task} reward={got} "
                        f"after {(time.time() - r.started) / 60:.0f} min")
                if r.image and not args.keep_images:
                    busy = {x.image for x in running}
                    if r.image not in busy and docker("rmi", r.image).returncode == 0:
                        log(args.batch_dir, f"removed image {r.image}")

            free = free_gb(disk_path)

            # out of disk: stop the task whose container holds the most,
            # queue it again
            if free < args.hard_free_gb and running:
                r = task_to_stop(running, container_sizes())
                trial, cont = r.trial(), r.container()
                if trial and cont:
                    save_state(cont, trial, "gv-state-at-diskguard")
                r.stop()
                if cont:
                    docker("rm", "-f", cont)
                running.remove(r)
                if pulling is r:
                    pulling = None
                n = requeued.get(r.task, 0) + 1
                requeued[r.task] = n
                if os.path.isdir(r.job_dir):
                    # the whole job aside: Harbor wrote a result for the stopped
                    # trial, and the rerun starts a fresh job under the name
                    move_aside(r.job_dir, "diskguard-stopped")
                queue.append(r.task)
                log(args.batch_dir, f"disk {free:.1f} GB < {args.hard_free_gb}: "
                    f"stopped {r.task} (state saved, job moved aside), queued "
                    f"again (stop #{n})")
                continue

            # the last admission's container is up (or it gave up): admit again
            if pulling and (pulling.container_up() or not pulling.alive()
                            or time.time() - pulling.started > args.pull_timeout):
                pulling = None

            if (queue and pulling is None and len(running) < args.max_concurrent
                    and free >= args.min_free_gb):
                task = queue.pop(0)
                job_dir = os.path.join(args.batch_dir, task)
                fd, tasks_file = tempfile.mkstemp(prefix=f"rolling-{task}-",
                                                  suffix=".txt")
                with os.fdopen(fd, "w") as f:
                    f.write(task + "\n")
                out = open(os.path.join(args.batch_dir, f"{task}.log"), "a")
                proc = subprocess.Popen(
                    [sys.executable, args.run_py, *run_args, "--tasks-file", tasks_file,
                     "--job-name", task, "--jobs-dir", args.batch_dir,
                     "--n-concurrent", "1"],
                    stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                    start_new_session=True)
                r = Running(task, proc, job_dir)
                running.append(r)
                pulling = r
                log(args.batch_dir, f"start {task} ({len(running)} running, "
                    f"{len(queue)} queued, {free:.1f} GB free)")

            if time.time() - last_backup >= args.backup_seconds:
                last_backup = time.time()
                saved = 0
                for r in running:
                    trial, cont = r.trial(), r.container()
                    if trial and cont and r.container_up():
                        saved += save_state(cont, trial, "gv-state-live")
                status = {"time": time.strftime("%F %T"), "free_gb": round(free, 1),
                          "running": [r.task for r in running], "queued": queue,
                          "backed_up": saved}
                with open(os.path.join(args.batch_dir, "rolling_status.json"),
                          "w", encoding="utf-8") as f:
                    json.dump(status, f, indent=1)
                log(args.batch_dir, f"heartbeat: {len(running)} running, "
                    f"{len(queue)} queued, {free:.1f} GB free, backed up {saved}")
            time.sleep(args.poll_seconds)

        except Exception:  # noqa: BLE001 - the batch outlives one bad pass
            log(args.batch_dir, "scheduler pass failed, continuing:\n"
                + traceback.format_exc())
            time.sleep(args.poll_seconds)

    if stopping:
        log(args.batch_dir, f"signal: stopping {len(running)} running tasks "
            "(state saved first)")
        for r in running:
            trial, cont = r.trial(), r.container()
            if trial and cont:
                save_state(cont, trial, "gv-state-at-stop")
            r.stop()
            cont = r.container()
            if cont:
                docker("rm", "-f", cont)
    else:
        with open(os.path.join(args.batch_dir, "summary.tsv"), "w",
                  encoding="utf-8") as f:
            f.write("task\treward\n")
            for t in all_tasks:
                f.write(f"{t}\t{reward(os.path.join(args.batch_dir, t))}\n")
    log(args.batch_dir, "finished")


if __name__ == "__main__":
    main()
