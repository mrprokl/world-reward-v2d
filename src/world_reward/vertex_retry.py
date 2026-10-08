"""Bounded retries of an identical Vertex request, never of its predictions.

The transport must perform ONE request with the supplied socket timeout (and no
SDK-level retries). Validate model content only after this function returns.
Official guidance: https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/retry-strategy
"""
from __future__ import annotations

import errno
import math
import random
import socket
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from email.utils import parsedate_to_datetime
from typing import Any
from urllib.error import HTTPError, URLError


RETRYABLE_HTTP = frozenset({408, 429, 500, 502, 503, 504})
_NETWORK_ERRNOS = frozenset({
    errno.ECONNRESET, errno.ECONNABORTED, errno.ECONNREFUSED,
    errno.EPIPE, errno.ETIMEDOUT, errno.ENETDOWN, errno.ENETUNREACH,
    errno.EHOSTUNREACH,
})


@dataclass(frozen=True)
class RetryResult:
    response: Any
    attempts: tuple[dict, ...]


class VertexRetryFailure(RuntimeError):
    """Only nonsensitive categories/ledger survive the transport exception."""

    def __init__(self, reason: str, attempts: list[dict]):
        self.reason = reason
        self.attempts = tuple(dict(row) for row in attempts)
        super().__init__(reason)


class RetryExhausted(VertexRetryFailure):
    """An attempt limit or whole-operation deadline was reached."""


class PermanentFailure(VertexRetryFailure):
    """Nonretryable HTTP, connection configuration or transport failure."""


def _finite_positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f'{name} must be finite and positive')
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    return value


def _failure(error: Exception) -> tuple[bool, str, int | None]:
    # HTTPError inherits URLError, so its explicit status takes precedence.
    if isinstance(error, HTTPError):
        return error.code in RETRYABLE_HTTP, 'http_error', error.code
    reason = error.reason if isinstance(error, URLError) else error
    if isinstance(reason, TimeoutError):
        return True, 'timeout', None
    if isinstance(reason, ConnectionError):
        return True, 'connection_error', None
    if isinstance(reason, socket.gaierror):
        return reason.errno == socket.EAI_AGAIN, 'dns_error', None
    if isinstance(reason, OSError) and reason.errno in _NETWORK_ERRNOS:
        return True, 'connection_error', None
    # Invalid JSON, schema errors, certificates, authorization and unknown
    # exceptions are not evidence that generating another answer is justified.
    return False, 'non_retryable_transport', None


def _retry_after(error: Exception, wall_clock: Callable[[], float]) -> float | None:
    if not isinstance(error, HTTPError) or error.headers is None:
        return None
    value = error.headers.get('Retry-After')
    if not isinstance(value, str):
        return None
    value = value.strip()
    # HTTP delay-seconds are decimal nonnegative integers, not arbitrary floats.
    if value.isascii() and value.isdigit():
        delay = float(value)
        # Any astronomical valid server delay exceeds a finite practical
        # deadline. Retain a finite ledger value and fail rather than ignore it.
        return delay if math.isfinite(delay) else sys.float_info.max
    try:
        date = parsedate_to_datetime(value)
        if date.tzinfo is None:
            return None
        delay = date.timestamp() - wall_clock()
        return max(0.0, delay) if math.isfinite(delay) else None
    except (TypeError, ValueError, OverflowError):
        return None


def call_with_retry(
    payload: bytes,
    transport: Callable[[bytes, float], Any],
    *,
    deadline: float,
    attempt_timeout: float,
    max_attempts: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], None] = time.sleep,
    wall_clock: Callable[[], float] = time.time,
    random01: Callable[[], float] = random.random,
) -> RetryResult:
    """Return the FIRST successful technical response and an attempt ledger.

    ``deadline`` is an absolute monotonic deadline, not a per-attempt duration.
    ``max_attempts`` counts the original request and is bounded at three. Full
    jitter is drawn in [0, exponential cap], with Google's one-second minimum.
    Retry-After is a server lower bound: do not retry early if it cannot fit the
    deadline. The caller owns concurrency control and response interpretation.
    """
    if type(payload) is not bytes or not payload:
        raise ValueError('An immutable nonempty request body is required')
    if type(max_attempts) is not int or not 1 <= max_attempts <= 3:
        raise ValueError('One original attempt and at most two retries required')
    if isinstance(deadline, bool) or not isinstance(deadline, (int, float)) or not math.isfinite(deadline):
        raise ValueError('A finite absolute monotonic deadline is required')
    attempt_timeout = _finite_positive(attempt_timeout, 'attempt_timeout')
    initial_delay = _finite_positive(initial_delay, 'initial_delay')
    max_delay = _finite_positive(max_delay, 'max_delay')
    if initial_delay < 1 or max_delay < initial_delay:
        raise ValueError('Backoff requires a one-second floor and a valid cap')
    attempts = []
    for attempt in range(1, max_attempts + 1):
        started = clock()
        remaining = deadline - started
        if remaining <= 0:
            raise RetryExhausted('deadline_exceeded', attempts) from None
        timeout = min(attempt_timeout, remaining)
        row = {'attempt': attempt, 'timeout_seconds': timeout,
               'duration_seconds': 0.0, 'outcome': 'success',
               'error_type': None, 'http_status': None, 'delay_seconds': 0.0}
        try:
            response = transport(payload, timeout)
        except Exception as error:
            ended = clock()
            retryable, error_type, http_status = _failure(error)
            row.update(duration_seconds=max(0.0, ended - started),
                       outcome='transient_error' if retryable else 'permanent_error',
                       error_type=error_type, http_status=http_status)
            attempts.append(row)
            if not retryable:
                raise PermanentFailure('non_retryable_failure', attempts) from None
            if ended >= deadline:
                raise RetryExhausted('deadline_exceeded', attempts) from None
            if attempt == max_attempts:
                raise RetryExhausted('attempts_exhausted', attempts) from None
            draw = random01()
            if isinstance(draw, bool) or not isinstance(draw, (int, float)) or not 0 <= draw <= 1:
                raise ValueError('Jitter must be a finite value in [0, 1]') from None
            cap = min(max_delay, initial_delay * 2 ** (attempt - 1))
            delay = max(1.0, draw * cap)
            server_delay = _retry_after(error, wall_clock)
            if server_delay is not None:
                row['retry_after_seconds'] = server_delay
                delay = max(delay, server_delay)
            # A required server wait is never silently shortened to max_delay.
            # Stop rather than sleep away the deadline or retry early.
            if delay >= deadline - clock():
                row['outcome'] = 'retry_deadline_exceeded'
                row['required_delay_seconds'] = delay
                raise RetryExhausted('deadline_exceeded', attempts) from None
            sleep(delay)
            row['delay_seconds'] = delay
        else:
            ended = clock()
            row['duration_seconds'] = max(0.0, ended - started)
            attempts.append(row)
            if ended >= deadline:
                row['outcome'] = 'late_response'
                raise RetryExhausted('deadline_exceeded', attempts) from None
            return RetryResult(response, tuple(dict(item) for item in attempts))
    raise AssertionError('Unreachable bounded retry state')
