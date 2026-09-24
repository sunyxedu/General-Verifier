"""
Follow an append-only JSONL log (similar to `tail -f`).

When reading mid-append, incomplete trailing lines are expected. The tailer reads
by byte offset, publishes only complete lines up to the final newline, and buffers
the remainder for the next poll.
Sequence continuity is validated strictly against expected `seq` numbers: gaps or
reordering represent data corruption rather than latency, and immediately raise
`TailError` instead of being silently absorbed.
"""

from __future__ import annotations

import json
import os
import time
from typing import Callable, Dict, Optional


class TailError(ValueError):
    pass


class Tail:
    def __init__(self, path: str, on_event: Callable[[Dict], None],
                 start_seq: int = 0, raw: bool = False,
                 start_offset: int = 0,
                 on_reject: Optional[Callable[[int, int, bytes, str], None]] = None):
        """
        raw=True: a feed of bare JSON lines with no seq discipline 
        (e.g. hook processes appending concurrently); 
        torn-line handling only.
        start_offset: resume after the lines an earlier reader consumed.
        on_reject(start, end, line, error): raw mode only — a line that is
        not a JSON object is handed here and skipped; the lines after it
        are still read (without it, the first bad line raises).
        """
        self.path = path
        self.on_event = on_event
        self.on_reject = on_reject
        self.raw = raw
        self._offset = max(0, int(start_offset or 0))
        self._buffer = b""
        self._expected = start_seq  # last seq already consumed
        # byte offset just past the line being delivered to on_event
        self.position = self._offset

    @property
    def consumed(self) -> int:
        """Byte offset just past the last complete line consumed."""
        return self._offset - len(self._buffer)

    def poll(self) -> int:
        """Consume every complete new line; returns how many events fired."""
        if not os.path.exists(self.path):
            return 0
        size = os.path.getsize(self.path)
        if size < self._offset:
            # the feed was replaced by a shorter one: nothing resumes there
            self._offset, self._buffer = 0, b""
        chunk = b""
        if size > self._offset:
            with open(self.path, "rb") as f:
                f.seek(self._offset)
                chunk = f.read(size - self._offset)
        elif b"\n" not in self._buffer:
            return 0   # nothing new, and no complete line left unhandled

        base = self._offset - len(self._buffer)   # file offset of data[0]
        data = self._buffer + chunk
        self._offset += len(chunk)
        last_newline = data.rfind(b"\n")
        if last_newline < 0:
            self._buffer = data          # still torn: wait for more
            return 0

        # every line is handled on its own, and the consumed offset moves
        # past a line only once it is handled: a line that raises stays
        # unconsumed, the lines before it are not read again, and none
        # after it is lost
        fired, pos = 0, 0
        try:
            while pos <= last_newline:
                end = data.index(b"\n", pos) + 1
                raw = data[pos:end].strip()
                self.position = base + end
                if raw:
                    event = self._parse(raw, base + pos, base + end)
                    if event is not None:
                        if not self.raw:
                            self._expected += 1
                            if event["seq"] != self._expected:
                                raise TailError(
                                    f"stream corrupted: expected seq "
                                    f"{self._expected}, got {event['seq']}"
                                )
                        self.on_event(event)
                        fired += 1
                pos = end
        finally:
            self._buffer = data[pos:]
        return fired

    def _parse(self, raw: bytes, start: int, end: int):
        """The line's JSON object; in raw mode with a reject handler, None
        for a line that is not one (reported, then skipped)."""
        if not (self.raw and self.on_reject is not None):
            return json.loads(raw.decode("utf-8"))
        try:
            event = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, ValueError, RecursionError) as err:
            self.on_reject(start, end, raw, f"not JSON: {err}")
            return None
        if not isinstance(event, dict):
            self.on_reject(start, end, raw, "not a JSON object")
            return None
        return event

    def follow(self, interval: float = 0.2, should_stop: Callable[[], bool] = lambda: False) -> None:
        while not should_stop():
            self.poll()
            time.sleep(interval)
        self.poll()  # drain once after stop
