"""Parallel driver for offline trace scoring.

Traces are scored independently by construction — each has its own state
dir, its own extraction, its own claim graph, its own certificate, and
shares nothing with the others (not even memory). So the only reason to
score them one at a time is bookkeeping, and this removes it: a thread
pool over the traces, with the results file the single shared resource
under a lock. Appends stay one-record-per-line, so a killed run resumes
from what already landed.
"""
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

from . import results as res
from verifier import offline

_write_lock = threading.Lock()


def score_many(jobs, out_path, workers=4, model=None, max_rounds=3,
               seats=None, label=lambda rec: str(rec), tag="score"):
    """jobs: list of (record, state_dir). Scores every state dir, merges
    the features into its record, appends to out_path. Returns the
    records that were scored."""
    if not jobs:
        return []
    done = []

    def one(job):
        record, state = job
        features = offline.score(state, model=model, max_rounds=max_rounds,
                                 seats=seats)
        record = dict(record, features=features)
        with _write_lock:
            res.append(out_path, record)
            done.append(record)
            print(f"[{tag} {len(done)}/{len(jobs)}] {label(record)} "
                  f"verdict={features.get('verdict')} "
                  f"rank={offline.rank_key(features)}", flush=True)
        return record

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(one, job) for job in jobs]
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as err:            # noqa: BLE001 - reported
                print(f"[{tag}] FAILED: {type(err).__name__}: {err}",
                      flush=True)
    return done
