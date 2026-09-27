"""Which failed ``codex exec`` attempts are the provider's transient
fault, and how long to wait them out."""
from __future__ import annotations

import json
import re

from ..schema import RoleError
from .stream import _native_jsonl_lines, _stream_text


class TransientBackendError(RoleError):
    """Provider-side transient failure (capacity, rate limit, transport):
    the request never got a completed turn, so retrying with backoff is
    sound — nothing semantic was produced to second-guess."""


# Bounded backoff for provider capacity flaps ("Selected model is at
# capacity"). Codex exec fails in seconds in that state, so a handful of
# spaced retries rides out the common minutes-long window without
# babysitting; a genuinely down provider still fails closed afterwards.
TRANSIENT_RETRIES = 5
TRANSIENT_BACKOFF_SECONDS = (15, 45, 90, 180, 300)

_TRANSIENT_EVENT_PATTERNS = (
    "at capacity", "rate limit", "rate-limit", "overload",
    "try again", "temporarily unavailable", "connection reset",
    "timed out", "timeout", "bad gateway", "service unavailable",
)
_TRANSIENT_STDERR_PATTERNS = (
    "at capacity", "rate limit", "rate-limit", "overloaded",
    "temporarily unavailable", "connection reset", "bad gateway",
    "service unavailable", "gateway timeout", "upstream timed out",
)
_PERMANENT_CLI_PATTERNS = (
    "usage limit", "purchase more credits", "authentication",
    "unauthorized", "not logged in", "invalid api key",
    "config.toml", "configuration field", "unknown configuration",
    "invalid configuration", "failed to parse", "toml parse",
    "unrecognized option", "unexpected argument", "invalid value",
    "permission denied", "no such file", "command not found",
)
# Without --json, codex writes the seat's whole transcript to stderr — its
# commands, their output, the diffs it applied — after a session header.
# Only codex's own lines speak for the attempt: what it printed before the
# header (a launch or configuration failure never gets that far) and its
# error lines after it ("ERROR: ..." and its log's "<time>Z ERROR ...").
# Matching the transcript let a seat's own "no such file" make a provider
# capacity failure permanent, and a seat's curl printing 503 rerun a call.
_CODEX_SESSION_RULE = re.compile(r"^-{8,}\s*$")
_CODEX_ERROR_LINE = re.compile(
    r"^(?:ERROR\b|error:|Error:|\d{4}-\d\d-\d\dT\S*Z\s+ERROR\b)")


def _codex_own_stderr(stderr: str) -> str:
    """The lines of a non-JSON codex stderr that codex itself wrote."""
    lines = stderr.splitlines()
    header = next((index for index, line in enumerate(lines)
                   if _CODEX_SESSION_RULE.match(line)), None)
    if header is None:
        return stderr
    return "\n".join(lines[:header] + [
        line for line in lines[header:] if _CODEX_ERROR_LINE.match(line)])


_TRANSIENT_HTTP_STATUS = re.compile(
    r"(?:\bhttp(?:\s+status)?|\bstatus(?:\s+code)?)\D{0,8}"
    r"(?:429|502|503|504)\b|(?:^|\s)(?:429|502|503|504)(?:\s|$)")


def _transient_failure_reason(
    stdout: str, returncode: int, stderr: str = "",
) -> str:
    """A transient-failure description, or "" when the stream looks like a
    real (retryable-by-nobody) protocol/semantic failure. Detection is
    deliberately narrow: an explicit error/turn.failed event whose message
    matches a known transient pattern, or an explicit provider/HTTP status in
    stderr. An unknown nonzero exit is not evidence of a provider outage."""
    messages = []
    for raw in _native_jsonl_lines(stdout or ""):
        raw = raw.strip()
        if not raw:
            continue
        try:
            event = json.loads(raw)
        except (ValueError, RecursionError):
            continue
        if not isinstance(event, dict):
            continue
        if event.get("type") == "error":
            messages.append(str(event.get("message") or ""))
        elif event.get("type") == "turn.failed":
            error = event.get("error")
            if isinstance(error, dict):
                messages.append(str(error.get("message") or ""))
    stderr_text = _codex_own_stderr(_stream_text(stderr)).lower()
    all_text = "\n".join(
        [message.lower() for message in messages] + [stderr_text])
    # Permanent account, configuration, and local-launch failures often also
    # say "try again". They must fail fast rather than masquerade as provider
    # capacity and sleep through the retry schedule.
    if any(pattern in all_text for pattern in _PERMANENT_CLI_PATTERNS):
        return ""
    if re.search(r"\b(?:401|403)\b", all_text):
        return ""
    for message in messages:
        lowered = message.lower()
        if any(pattern in lowered for pattern in _TRANSIENT_EVENT_PATTERNS):
            return "provider event reported a transient failure"
    if returncode != 0 and not messages:
        if any(pattern in stderr_text
               for pattern in _TRANSIENT_STDERR_PATTERNS):
            return "provider stderr reported a transient failure"
        if _TRANSIENT_HTTP_STATUS.search(stderr_text):
            return "provider stderr reported a transient HTTP status"
    return ""
