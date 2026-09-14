"""Bounded JSON-line metadata parsing without retaining oversized string bodies.

Raw CLI output is retained by the caller. This projection validates every UTF-8
byte and every string escape, then replaces oversized string values with null.
The remaining JSON grammar is validated by the standard decoder. Oversized keys,
structural frames, malformed strings and malformed JSON fail closed.
"""
from __future__ import annotations

import codecs
import json


class JsonEventStream:
    def __init__(self, *, frame_limit=2 * 1024 * 1024, string_limit=4096):
        self.frame_limit = frame_limit
        self.string_limit = string_limit
        self._reset()

    def _reset(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")("strict")
        self.parts = []
        self.frame_size = 0
        self.token = []
        self.token_size = 0
        self.in_string = False
        self.escape = False
        self.hex_remaining = 0
        self.invalid = False
        self.redacted = False
        self.started = False

    def _append(self, text):
        self.frame_size += len(text.encode("utf-8"))
        if self.frame_size > self.frame_limit:
            self.invalid = True
            self.parts.clear()
        elif not self.invalid:
            self.parts.append(text)

    def _text(self, text):
        for char in text:
            if self.invalid:
                return
            if not self.in_string:
                if char == '"':
                    self.in_string = True
                    self.token = ['"']
                    self.token_size = 1
                else:
                    self._append(char)
                continue
            self.token_size += 1
            if self.token_size <= self.string_limit:
                self.token.append(char)
            else:
                self.token.clear()
            if self.hex_remaining:
                if char not in "0123456789abcdefABCDEF":
                    self.invalid = True
                self.hex_remaining -= 1
            elif self.escape:
                self.escape = False
                if char == "u":
                    self.hex_remaining = 4
                elif char not in '"\\/bfnrt':
                    self.invalid = True
            elif char == "\\":
                self.escape = True
            elif char == '"':
                self.in_string = False
                if self.token_size > self.string_limit:
                    self._append("null")
                    self.redacted = True
                else:
                    self._append("".join(self.token))
                self.token.clear()
            elif ord(char) < 32:
                self.invalid = True

    def _finish(self):
        try:
            value = None if self.invalid or self.in_string else json.loads("".join(self.parts))
        except (ValueError, RecursionError):
            value = None
        result = (value if isinstance(value, dict) else None, self.redacted)
        self._reset()
        return result

    def feed(self, chunk):
        """Yield one (event-or-None, strings-redacted) pair per complete line."""
        pieces = chunk.split(b"\n")
        for index, piece in enumerate(pieces):
            terminal = index < len(pieces) - 1
            self.started = self.started or bool(piece) or terminal
            try:
                self._text(self.decoder.decode(piece, final=terminal))
            except UnicodeError:
                self.invalid = True
            if terminal:
                yield self._finish()

    def finish(self):
        """Inspect a terminal unterminated line, including incomplete UTF-8."""
        if not self.started:
            return None
        try:
            self._text(self.decoder.decode(b"", final=True))
        except UnicodeError:
            self.invalid = True
        return self._finish()
