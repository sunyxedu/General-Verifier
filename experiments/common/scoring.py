"""Parallel driver for offline trace scoring.

Traces are scored independently by construction — each has its own state
dir, its own extraction, its own claim graph, its own certificate, and
shares nothing with the others (not even memory). So the only reason to
score them one at a time is bookkeeping, and this removes it: a thread
pool over the traces, with the results file the single shared resource
under a lock. Appends stay one-record-per-line, so a killed run resumes
from what already landed.
"""
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import progress
from . import results as res
from verifier import offline

_write_lock = threading.Lock()


def scalar(features) -> float:
    """Flatten offline.rank_key's lexicographic tuple into ONE absolute
    score in [0, 1]: five equal verdict bands that can never cross —
    verified [0.8, 1) · partial [0.6, 0.8) · unverified [0.4, 0.6) ·
    refuted [0.2, 0.4) · unscoreable exactly 0.0 — with the fraction
    inside a band packing verified-fraction then verified-obligation
    count at weights that reproduce rank_key's exact order for
    certificate slices up to ~44 claims (two fractions with denominators
    <= S differ by at least 1/S^2, and 0.0005 sits below that until
    S ~ 44). Selection keeps comparing the exact tuple; this is a view,
    not the decision variable.

    Lives in the experiment layer on purpose: verifier/ is fingerprinted,
    and a derived display value must not invalidate scored caches."""
    rank, frac, obligations = offline.rank_key(features)
    return (rank + 1 + 0.999 * frac
            + 0.0005 * (obligations / (obligations + 1))) / 5


def score_many(jobs, out_path, workers=4, model=None, max_rounds=3,
               seats=None, effort=None, critic_rounds=None,
               label=lambda rec: str(rec), tag="score", on_record=None,
               heartbeat=60):
    """jobs: list of (record, state_dir). Scores every state dir, merges
    the features into its record, appends to out_path. Returns the
    records that were scored. `on_record`, when given, fires once per
    scored record under the write lock (serialized) — callers use it to
    emit grouped output the moment a group completes. Every `heartbeat`
    seconds a liveness line (done/total plus each in-flight trace and its
    age) goes to stderr — trials are silent for tens of minutes, and a
    piped/tee'd run has no TTY for a live bar (0 disables)."""
    if not jobs:
        return []
    done = []
    active = {}
    stop = threading.Event()

    # harbor-style live view: overall bar (Mean = verified rate) plus one
    # spinner row per in-flight trace. On a tee'd/piped run it draws on
    # /dev/tty while the pipes keep receiving plain lines; on a bare tty
    # rich owns stderr and the raw prints below are routed through it.
    console, separate = progress.tty_console()
    board = progress.Board(len(jobs), console=console) if console else None

    def say(line):
        if board is not None and not separate:
            board.log(line)     # rich owns the terminal: print through it
            return
        print(line, file=sys.stderr, flush=True)
        if board is not None:
            board.log(line)     # piped log AND the live terminal view

    def one(job):
        record, state = job
        name = label(record)
        handle = board.begin(name) if board is not None else None
        with _write_lock:
            active[name] = time.time()
        try:
            features = offline.score(state, model=model,
                                     max_rounds=max_rounds,
                                     seats=seats, effort=effort,
                                     critic_rounds=critic_rounds)
        except BaseException:
            if board is not None:
                board.advance(handle, 0.0)
            raise
        finally:
            with _write_lock:
                active.pop(name, None)
        record = dict(record, features=features)
        if board is not None:
            board.advance(handle,
                          1.0 if features.get("verdict") == "verified"
                          else 0.0)
        key = offline.rank_key(features)
        with _write_lock:
            res.append(out_path, record)
            done.append(record)
            say(f"[{tag} {len(done)}/{len(jobs)}] {label(record)} -> "
                f"{features.get('verdict') or 'unscoreable'} "
                f"score={scalar(features):.4f} "
                f"frac={key[1]:.2f} obligations={key[2]}")
            if on_record is not None:
                on_record(record)
        return record

    def beat():
        while not stop.wait(heartbeat):
            with _write_lock:
                now = time.time()
                flight = ", ".join(
                    f"{name} {int((now - t0) // 60)}m"
                    for name, t0 in sorted(active.items(),
                                           key=lambda kv: kv[1]))
                print(f"[{tag}] {len(done)}/{len(jobs)} done · "
                      f"{len(active)} running: {flight}",
                      file=sys.stderr, flush=True)

    # the heartbeat is a liveness record for a FULLY headless run (no tty
    # anywhere). Whenever a live board exists it stays off: with `| tee`
    # the heartbeat would bounce back onto the same screen and displace
    # rich's in-place redraw into stacked stale frames
    if heartbeat and board is None:
        threading.Thread(target=beat, daemon=True).start()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:            # noqa: BLE001 - reported
                say(f"[{tag}] FAILED: {type(err).__name__}: {err}")
    stop.set()
    if board is not None:
        board.close()
    return done
