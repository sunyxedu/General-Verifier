"""Log tail (component 3, online attach point): follow an append-only JSONL.

The stream may be mid-append when we read it, so a torn final line is normal:
read by byte offset, publish only up to the last newline, keep the remainder
for the next poll. Sequence continuity is validated exactly as in replay —
a gap means the file was truncated or reordered, which is corruption, not
latency, and must raise rather than be absorbed.
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict


class TailError(ValueError):
    pass


class Tail:
    def __init__(self, path: str, on_event: Callable[[Dict], None],
                 start_seq: int = 0, raw: bool = False):
        """raw=True: a feed of bare JSON lines with no seq discipline (e.g.
        hook processes appending concurrently); torn-line handling only."""
        self.path = path
        self.on_event = on_event
        self.raw = raw
        self._offset = 0
        self._buffer = b""
        self._expected = start_seq  # last seq already consumed

    def poll(self) -> int:
        """Consume every complete new line; returns how many events fired."""
        if not os.path.exists(self.path):
            return 0
        size = os.path.getsize(self.path)
        if size <= self._offset:
            return 0
        with open(self.path, "rb") as f:
            f.seek(self._offset)
            chunk = f.read(size - self._offset)
        self._offset = size

        data = self._buffer + chunk
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            self._buffer = data          # still torn: wait for more
            return 0
        publishable, self._buffer = data[: last_newline + 1], data[last_newline + 1:]

        fired = 0
        for raw in publishable.split(b"\n"):
            raw = raw.strip()
            if not raw:
                continue
            event = json.loads(raw.decode("utf-8"))
            if not self.raw:
                self._expected += 1
                if event["seq"] != self._expected:
                    raise TailError(
                        f"stream corrupted: expected seq {self._expected}, "
                        f"got {event['seq']}"
                    )
            self.on_event(event)
            fired += 1
        return fired

    def follow(self, interval: float = 0.2,
               should_stop: Callable[[], bool] = lambda: False) -> None:
        while not should_stop():
            self.poll()
            time.sleep(interval)
        self.poll()  # drain once after stop
