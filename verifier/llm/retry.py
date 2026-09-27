"""Waiting out a runtime's transient failures.

A transient failure is the transport's or provider's, never the model's
answer: a rate limit, an outage, exhausted credit. Retrying it immediately
is useless and giving up at once loses a whole seat, so a runtime waits
through a bounded schedule and only then fails. Each runtime declares its
own rules (which failures, which waits, what it records along the way);
the driver that applies them exists once. A failure no rule names
propagates untouched.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any, Awaitable, Callable, Iterator, Optional, Sequence, Tuple, TypeVar,
)

T = TypeVar("T")


@dataclass(frozen=True)
class Backoff:
    """The waits between attempts: one per retry, ``retries`` of them
    (default: one per delay), the last delay repeating when there are more
    retries than delays."""
    delays: Tuple[float, ...]
    retries: Optional[int] = None

    def waits(self) -> Iterator[float]:
        count = len(self.delays) if self.retries is None else self.retries
        for index in range(count):
            yield self.delays[min(index, len(self.delays) - 1)]


@dataclass(frozen=True)
class RetryRule:
    """Retry failures of ``errors`` through ``backoff``. ``on_failure`` sees
    each such failure with the wait before the next attempt (None: the
    waits are spent); then ``exhausted`` builds the error that ends the
    call."""
    errors: Tuple[type, ...]
    backoff: Backoff
    exhausted: Callable[[BaseException], BaseException]
    on_failure: Optional[Callable[[BaseException, Optional[float]], None]] = None


class _Schedules:
    """Each rule's remaining waits within one call."""

    def __init__(self, rules: Sequence[RetryRule]):
        self.rules = tuple(rules)
        self.errors = tuple(error for rule in self.rules for error in rule.errors)
        self._waits = [rule.backoff.waits() for rule in self.rules]

    def next_wait(self, error: BaseException) -> float:
        """The wait before retrying ``error``; raises it or its rule's
        exhaustion error when there is none."""
        for rule, waits in zip(self.rules, self._waits):
            if isinstance(error, rule.errors):
                wait = next(waits, None)
                if rule.on_failure is not None:
                    rule.on_failure(error, wait)
                if wait is None:
                    raise rule.exhausted(error)
                return wait
        raise error


def retry_transient(attempt: Callable[[], T], rules: Sequence[RetryRule],
                    sleep: Callable[[float], Any]) -> T:
    """``attempt()`` until it returns, waiting out the failures ``rules``
    name."""
    schedules = _Schedules(rules)
    while True:
        try:
            return attempt()
        except schedules.errors as error:
            sleep(schedules.next_wait(error))


async def retry_transient_async(attempt: Callable[[], Awaitable[T]],
                                rules: Sequence[RetryRule],
                                sleep: Callable[[float], Awaitable[Any]]) -> T:
    """``retry_transient`` for a coroutine attempt."""
    schedules = _Schedules(rules)
    while True:
        try:
            return await attempt()
        except schedules.errors as error:
            await sleep(schedules.next_wait(error))
