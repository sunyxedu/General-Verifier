"""Seat-call receipts: one row per seat call in ``<state>/role_calls.jsonl``.

Every runtime writes the same kind of row for each call it runs (which
seat, how long, whether it failed, on which claim and obligation), so a
latency or cost breakdown sees every seat call whatever ran it. Each
runtime builds its own row (the SDK's turns and cost, a direct call's
attempts, a Codex call's room wait); the call's outcome is captured and
the row appended here. Telemetry only: never read by any seat, and a
receipt that cannot be written never fails the call.
"""
from __future__ import annotations

import json
import os
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, Optional, Tuple

ROLE_CALLS_FILE = "role_calls.jsonl"


def append_role_call(state_dir: str, row: Dict[str, Any]) -> None:
    """Append one receipt row; a failed write is dropped."""
    try:
        with open(os.path.join(state_dir, ROLE_CALLS_FILE), "a",
                  encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    except OSError:
        pass


def _error_text(error: BaseException) -> str:
    return str(error)[:300]


@contextmanager
def seat_call_receipt(
    state_dir: Optional[str],
    row: Callable[[Optional[str]], Dict[str, Any]],
    *,
    errors: Tuple[type, ...] = (Exception,),
    describe: Callable[[BaseException], str] = _error_text,
) -> Iterator[None]:
    """Receipt the call made in this block: ``row(error)`` is built when it
    ends (only with a state directory to write in), ``error`` being
    ``describe`` of a failure among ``errors`` it raised, else None."""
    error: Optional[str] = None
    try:
        yield
    except errors as failure:
        error = describe(failure)
        raise
    finally:
        if state_dir:
            append_role_call(state_dir, row(error))
