"""Recoverable scheduling for the existing verifier, not another verifier.

The journal is host-owned operational state, never claim evidence. Disconnecting
an MCP waiter cannot cancel an adjudication or discard its result. A single
worker preserves the Service's ledger/graph discipline; the kernel retains its
normal parallelism inside an adjudication. Completed results are explicitly
historical: callers must authenticate current certificates/delivery separately.
"""
from __future__ import annotations

import contextvars
import copy
import json
import os
import queue
import stat
import threading
import time
import uuid
from typing import Any, Callable

from .core.model import canonical, content_hash


PENDING = frozenset({"queued", "running"})
TERMINAL = frozenset({"completed", "failed", "interrupted"})


class VerificationJobs:
    """Durable, idempotent jobs with an independent, cheap status surface.

An interrupted job is not silently restarted on process startup. Submitting the
same request explicitly recovers the same job ID. An explicit retry is allowed
only after completion, and is distinct from reconnecting to an existing job.
The supplied operation remains responsible for domain identity and fail-closed
certificate validation; this class never grants a verdict itself.
"""

    def __init__(self, state_dir: str):
        self.path = os.path.join(state_dir, "verification_jobs.jsonl")
        self._lock = threading.RLock()
        self._jobs: dict[str, dict] = {}
        self._keys: dict[str, str] = {}
        self._operations: dict[str, Callable[[bool], dict]] = {}
        self._done: dict[str, threading.Event] = {}
        self._queue: queue.Queue = queue.Queue()
        self._worker: threading.Thread | None = None
        self._chain = ""
        self._journal_error = None
        self._public_jobs = {}
        self._snapshot_unix = time.time()
        self._load()

    def _open(self, flags: int) -> int:
        fd = os.open(self.path, flags | os.O_NONBLOCK
                     | getattr(os, "O_NOFOLLOW", 0)
                     | getattr(os, "O_CLOEXEC", 0), 0o600)
        info = os.fstat(fd)
        if (not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid()
                or info.st_nlink != 1 or info.st_mode & 0o077):
            os.close(fd)
            raise ValueError("unsafe verification job journal")
        return fd

    def _load(self):
        try:
            fd = self._open(os.O_RDONLY)
        except FileNotFoundError:
            return
        with os.fdopen(fd, "r", encoding="utf-8") as stream:
            for line in stream:
                record = json.loads(line)
                digest = record.pop("digest", None)
                if (record.get("schema") != "gv-verification-job-v1"
                        or record.get("previous_digest") != self._chain
                        or content_hash(record) != digest):
                    raise ValueError("verification job journal integrity failure")
                job = record["job"]
                if (job.get("state") not in PENDING | TERMINAL
                        or job.get("request_key") != content_hash(job["request"])
                        or not isinstance(job.get("job_id"), str)):
                    raise ValueError("invalid verification job record")
                self._jobs[job["job_id"]] = job
                self._keys[job["request_key"]] = job["job_id"]
                self._chain = digest
        for job in tuple(self._jobs.values()):
            self._done[job["job_id"]] = threading.Event()
            if job["state"] in PENDING:
                job["state"] = "interrupted"
                job["updated_unix"] = time.time()
                job["error_type"] = "ServerRestart"
                self._persist(job)
            self._done[job["job_id"]].set()
        self._publish_snapshot()

    def _persist(self, job):
        if self._journal_error is not None:
            raise OSError("verification job journal unavailable")
        record = {"schema": "gv-verification-job-v1",
                  "previous_digest": self._chain, "job": job}
        digest = content_hash(record)
        raw = (canonical({**record, "digest": digest}) + "\n").encode("utf-8")
        try:
            fd = self._open(os.O_WRONLY | os.O_APPEND | os.O_CREAT)
            try:
                while raw:
                    written = os.write(fd, raw)
                    if written <= 0:
                        raise OSError("verification journal write failed")
                    raw = raw[written:]
                os.fsync(fd)
            finally:
                os.close(fd)
        except Exception as exc:
            # A partial write or failed fsync leaves durability/chain position
            # unknown. Never append using the old chain or launch further jobs.
            self._journal_error = type(exc).__name__
            raise OSError("verification job journal unavailable") from None
        self._chain = digest

    @staticmethod
    def _summary(job):
        return {key: copy.deepcopy(job[key]) for key in (
            "job_id", "tool", "state", "created_unix", "started_unix",
            "updated_unix", "attempts", "reused_count", "error_type")
            if key in job}

    def _publish_snapshot(self):
        # Published dictionaries are immutable internally and replaced as one
        # object. Readers need no lock that could be held across journal fsync.
        self._public_jobs = {job_id: {"summary": self._summary(job),
            **({"result": copy.deepcopy(job["result"])} if job["state"] == "completed" else {})}
            for job_id, job in self._jobs.items()}
        self._snapshot_unix = time.time()

    def progress(self):
        published = self._public_jobs
        jobs = [copy.deepcopy(item["summary"]) for item in published.values()]
        return {"schema": "gv-verification-progress-v1",
                "checked_unix": time.time(), "jobs": jobs,
                "journal_available": self._journal_error is None,
                "metadata_snapshot_unix": self._snapshot_unix,
                "active_job_ids": [job["job_id"] for job in jobs
                                   if job["state"] in PENDING]}

    def status(self, job_id: str):
        published = self._public_jobs
        retained = published.get(job_id)
        if retained is None:
            return {"error": "unknown verification job", "verified": False}
        job = retained["summary"]
        result = {**copy.deepcopy(job), "pending": job["state"] in PENDING,
                  "verified": False, "official_score": None,
                  "journal_available": self._journal_error is None}
        if job["state"] == "completed":
            result["result"] = copy.deepcopy(retained["result"])
            result["result_is_retained"] = True
        return result

    def request(self, job_id: str):
        with self._lock:
            if job_id not in self._jobs:
                raise ValueError("unknown verification job")
            return copy.deepcopy(self._jobs[job_id]["request"])

    def submit(self, request: dict, operation: Callable[[bool], dict]):
        # Freeze caller-owned lists/maps before deriving an identity or running.
        request = json.loads(canonical(request))
        key = content_hash(request)
        with self._lock:
            if self._journal_error is not None:
                raise OSError("verification job journal unavailable")
            prior = self._keys.get(key)
            if prior is not None:
                job = self._jobs[prior]
                staged = copy.deepcopy(job)
                staged["reused_count"] += 1
                if job["state"] != "interrupted":
                    self._persist(staged)
                    self._jobs[prior] = staged
                    self._publish_snapshot()
                    self._operations[prior] = operation
                    return self.status(prior)
                self._enqueue(staged, operation, retry=False)
                return self.status(prior)
            now = time.time()
            job_id = "verification:" + uuid.uuid4().hex
            job = {"job_id": job_id, "tool": request["tool"],
                   "request": request, "request_key": key,
                   "state": "interrupted", "created_unix": now,
                   "started_unix": None, "updated_unix": now,
                   "attempts": 0, "reused_count": 0}
            self._enqueue(job, operation, retry=False)
            return self.status(job_id)

    def retry(self, job_id: str, operation: Callable[[bool], dict] | None = None):
        with self._lock:
            if self._journal_error is not None:
                raise OSError("verification job journal unavailable")
            job = self._jobs.get(job_id)
            if job is None:
                return {"error": "unknown verification job", "verified": False}
            if job["state"] in PENDING:
                return self.status(job_id)
            result = job.get("result") or {}
            delivery_retry = (job["tool"] == "cyber_finalize_candidate"
                              and result.get("materialized") is False)
            if result.get("materialized") or (not delivery_retry and (
                    result.get("verdict") == "verified" or result.get("status") == "verified")):
                return {**self.status(job_id), "error": "verified job is not retryable"}
            operation = operation or self._operations.get(job_id)
            if operation is None:
                return {**self.status(job_id), "error": "resubmit the original request after restart"}
            self._enqueue(job, operation, retry=True)
            return self.status(job_id)

    def _enqueue(self, job, operation, *, retry):
        staged = copy.deepcopy(job)
        staged.update(state="queued", updated_unix=time.time(), started_unix=None,
                      attempts=job["attempts"] + 1)
        staged.pop("result", None)
        staged.pop("error_type", None)
        self._persist(staged)  # Publish only after successful durable retention.
        job_id = staged["job_id"]
        self._jobs[job_id] = staged
        self._keys[staged["request_key"]] = job_id
        self._operations[job_id] = operation
        self._done.setdefault(job_id, threading.Event()).clear()
        self._publish_snapshot()
        self._queue.put((job_id, operation, retry, contextvars.copy_context()))
        if self._worker is None or not self._worker.is_alive():
            self._worker = threading.Thread(target=self._run,
                name="gv-verification-jobs", daemon=True)
            self._worker.start()

    def _run(self):
        while True:
            job_id, operation, retry, context = self._queue.get()
            try:
                with self._lock:
                    job = copy.deepcopy(self._jobs[job_id])
                    job.update(state="running", started_unix=time.time(), updated_unix=time.time())
                    self._persist(job)
                    self._jobs[job_id] = job
                    self._publish_snapshot()
                result = context.run(operation, retry)
                # Serialization is also a boundary check against mutable or
                # non-JSON backend return values entering operational state.
                result = json.loads(canonical(result))
                with self._lock:
                    job = copy.deepcopy(self._jobs[job_id])
                    job.update(state="completed", result=result, updated_unix=time.time())
                    self._persist(job)
                    self._jobs[job_id] = job
                    self._publish_snapshot()
            except BaseException as exc:
                with self._lock:
                    job = copy.deepcopy(self._jobs[job_id])
                    job.update(state="failed", error_type=type(exc).__name__,
                               updated_unix=time.time())
                    job.pop("result", None)
                    try:
                        self._persist(job)
                    except (OSError, ValueError):
                        pass  # An unretained result must never be reported verified.
                    self._jobs[job_id] = job
                    self._publish_snapshot()
            finally:
                self._done[job_id].set()
                self._queue.task_done()

    def wait(self, job_id: str, seconds: float = 0.1):
        """Bound the response wait only, never the verifier's execution time."""
        with self._lock:
            done = self._done.get(job_id)
        if done is not None:
            done.wait(max(0.0, min(seconds, 1.0)))
        return self.status(job_id)
