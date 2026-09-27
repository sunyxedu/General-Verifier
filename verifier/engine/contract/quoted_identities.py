"""Identities an obligation quotes from the task-start record, kept exact.

The compiler sees the task-start record (file digests, commit ids) and
may quote an identity into an obligation's text so the check can compare
against it. A model copying a long hexadecimal string drops, adds, or
alters a character now and then, and a check that compares the delivered
world against the corrupted copy then "refutes" a file that never
changed. The record is on disk, so the copy need not be trusted: a hex
token in an obligation that is not a recorded identity but is a near
copy of one (a length within a couple of characters, and a shared
prefix long enough to name it and longer than any other recorded
identity's) is restored to the recorded value before the obligation is
materialized. Tokens with no near recorded identity are left alone, and
nothing is ever invented.
"""
from __future__ import annotations

import json
import os
import re
from typing import Iterable, List, Set, Tuple

# a whole run of lowercase hex, long enough to be an identity rather than
# a word or a short number
HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])[0-9a-f]{16,128}(?![0-9A-Fa-f])")
FULL_IDENTITY = re.compile(r"[0-9a-f]{40}|[0-9a-f]{64}")
PREFIX_MIN = 16
LENGTH_SLACK = 2


def _walk(value) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for key, item in value.items():
            yield from _walk(key)
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)


def recorded_identities(state_dir: str) -> Set[str]:
    """Every full digest or commit id the task-start record carries."""
    found: Set[str] = set()
    base = os.path.join(state_dir or "", "baseline")
    for name in ("manifest.json", "git.json", "repositories.json"):
        try:
            with open(os.path.join(base, name), encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, ValueError):
            continue
        for text in _walk(data):
            for token in HEX_TOKEN.findall(text):
                if FULL_IDENTITY.fullmatch(token):
                    found.add(token)
    return found


def _shared(token: str, identity: str) -> int:
    """How far a copy agrees with a recorded identity from the start (0
    when their lengths differ by more than a dropped or doubled
    character or two)."""
    if abs(len(token) - len(identity)) > LENGTH_SLACK:
        return 0
    shared = 0
    for a, b in zip(token, identity):
        if a != b:
            break
        shared += 1
    return shared


def restore(text: str, identities: Set[str]) -> Tuple[str, List[Tuple[str, str]]]:
    """The text with every near copy of a recorded identity restored, and
    the (copied, recorded) pairs that were restored."""
    if not identities or not text:
        return text, []
    restored: List[Tuple[str, str]] = []

    def fix(match):
        token = match.group(0)
        if token in identities:
            return token
        scored = sorted(((_shared(token, i), i) for i in identities),
                        reverse=True)
        best = scored[0] if scored else (0, "")
        runner_up = scored[1][0] if len(scored) > 1 else 0
        # the one recorded identity the copy agrees with longest, and only
        # when it is long enough to be that identity and no other ties it
        if best[0] < PREFIX_MIN or best[0] == runner_up:
            return token
        restored.append((token, best[1]))
        return best[1]

    return HEX_TOKEN.sub(fix, text), restored
