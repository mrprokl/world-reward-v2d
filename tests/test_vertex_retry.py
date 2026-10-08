import errno
import json
import socket
from email.message import Message
from urllib.error import HTTPError, URLError

import pytest

from world_reward.vertex_retry import (
    PermanentFailure, RetryExhausted, RETRYABLE_HTTP, call_with_retry,
)


class Clock:
    def __init__(self):
        self.now = 10.0
        self.waits = []

    def __call__(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def http_error(status, retry_after=None):
    headers = Message()
    if retry_after is not None:
        headers['Retry-After'] = retry_after
    return HTTPError('https://secret.invalid/private?token=do-not-log', status,
                     'secret error body', headers, None)


def invoke(transport, clock, **overrides):
    options = dict(deadline=100.0, attempt_timeout=20.0, clock=clock,
                   sleep=clock.sleep, wall_clock=lambda: 1_000_000_000.0,
                   random01=lambda: 0.5)
    options.update(overrides)
    return call_with_retry(b'one exact immutable request', transport, **options)


def test_first_success_after_timeout_identical_payload_and_exact_ledger():
    clock = Clock()
    calls = []
    response = {'candidates': [{'content': 'first successful prediction'}]}

    def transport(payload, timeout):
        calls.append((payload, timeout))
        clock.now += 2
        if len(calls) == 1:
            raise URLError(socket.timeout('secret transport message'))
        return response

    result = invoke(transport, clock)
    assert result.response is response
    assert calls[0][0] is calls[1][0]
    assert calls == [(b'one exact immutable request', 20.0)] * 2
    assert clock.waits == [1.0]
    assert result.attempts == (
        dict(attempt=1, timeout_seconds=20.0, duration_seconds=2.0,
             outcome='transient_error', error_type='timeout', http_status=None,
             delay_seconds=1.0),
        dict(attempt=2, timeout_seconds=20.0, duration_seconds=2.0,
             outcome='success', error_type=None, http_status=None, delay_seconds=0.0),
    )
    assert 'secret' not in json.dumps(result.attempts)


@pytest.mark.parametrize('status', sorted(RETRYABLE_HTTP))
def test_retry_only_declared_http_statuses(status):
    clock = Clock()
    calls = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            raise http_error(status)
        return b'first response'

    result = invoke(transport, clock)
    assert len(calls) == 2
    assert result.attempts[0]['http_status'] == status


@pytest.mark.parametrize('status', [400, 401, 403, 404, 409, 499, 501, 505])
def test_permanent_http_errors_not_retried_or_leaked(status):
    clock = Clock()

    def transport(*_):
        raise http_error(status)

    with pytest.raises(PermanentFailure) as caught:
        invoke(transport, clock)
    assert len(caught.value.attempts) == 1
    assert caught.value.attempts[0]['outcome'] == 'permanent_error'
    assert clock.waits == []
    assert 'secret' not in str(caught.value)
    assert 'secret' not in json.dumps(caught.value.attempts)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize('retry_after,delay', [
    ('7', 7.0), ('Sun, 09 Sep 2001 01:46:47 GMT', 7.0),
    ('Sun, 09 Sep 2001 01:46:00 GMT', 1.0), ('broken', 1.0),
    ('-4', 1.0), ('NaN', 1.0), ('inf', 1.0), ('1.5', 1.0),
])
def test_retry_after_delay_seconds_and_http_date(retry_after, delay):
    clock = Clock()
    calls = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            raise http_error(429, retry_after)
        return None

    result = invoke(transport, clock)
    assert result.response is None
    assert clock.waits == [delay]


def test_server_retry_after_not_cut_to_cap_or_deadline():
    clock = Clock()

    def transport(*_):
        raise http_error(429, '70')

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock, deadline=50.0, max_delay=2.0)
    assert caught.value.reason == 'deadline_exceeded'
    assert caught.value.attempts[0]['required_delay_seconds'] == 70.0
    assert clock.waits == []


@pytest.mark.parametrize('retry_after', ['9999999999999', '9' * 400])
def test_arbitrarily_large_valid_server_delay_fails_not_retried_early(retry_after):
    clock = Clock()

    def transport(*_):
        raise http_error(429, retry_after)

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock)
    assert caught.value.reason == 'deadline_exceeded'
    assert clock.waits == []
    json.dumps(caught.value.attempts, allow_nan=False)


def test_attempts_exhausted_bounded_and_full_jitter_is_injectable():
    clock = Clock()

    def transport(*_):
        clock.now += 0.25
        raise ConnectionResetError('secret connection reset details')

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock, initial_delay=4, max_delay=6, random01=lambda: 0.75)
    assert caught.value.reason == 'attempts_exhausted'
    assert clock.waits == [3.0, 4.5]
    assert len(caught.value.attempts) == 3
    assert caught.value.attempts[-1]['delay_seconds'] == 0.0


@pytest.mark.parametrize('response', [None, {}, {'person': None, 'object': None},
                                      {'promptFeedback': {'blockReason': 'SAFETY'}},
                                      b'{malformed JSON', b''])
def test_never_retry_successful_technical_response_even_invalid_or_null(response):
    clock = Clock()
    calls = []

    def transport(*args):
        calls.append(args)
        return response

    result = invoke(transport, clock)
    assert result.response is response
    assert len(calls) == 1 and clock.waits == []


@pytest.mark.parametrize('error', [ValueError('invalid schema'),
                                  URLError('not a proven transient failure'),
                                  URLError(socket.gaierror(socket.EAI_NONAME, 'bad DNS'))])
def test_unknown_and_parsing_failures_not_retried(error):
    clock = Clock()

    def transport(*_):
        raise error

    with pytest.raises(PermanentFailure) as caught:
        invoke(transport, clock)
    assert len(caught.value.attempts) == 1 and clock.waits == []


@pytest.mark.parametrize('error', [URLError(ConnectionResetError()),
                                  URLError(OSError(errno.ENETUNREACH, 'network')),
                                  URLError(socket.gaierror(socket.EAI_AGAIN, 'temporary'))])
def test_proven_transient_connections_retried(error):
    clock = Clock()
    calls = []

    def transport(*args):
        calls.append(args)
        if len(calls) == 1:
            raise error
        return b'ok'

    assert invoke(transport, clock).response == b'ok'
    assert len(calls) == 2


def test_timeout_capped_to_remaining_whole_deadline_on_each_attempt():
    clock = Clock()
    timeouts = []

    def transport(_, timeout):
        timeouts.append(timeout)
        if len(timeouts) == 1:
            clock.now += 2
            raise TimeoutError()
        clock.now += 1
        return b'ok'

    assert invoke(transport, clock, deadline=15).response == b'ok'
    assert timeouts == [5.0, 2.0]


def test_expired_deadline_no_first_request():
    clock = Clock()

    def transport(*_):
        pytest.fail('No request should be issued after deadline')

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock, deadline=10)
    assert caught.value.reason == 'deadline_exceeded'
    assert caught.value.attempts == ()


def test_late_success_rejected_not_used_as_prediction_or_retried():
    clock = Clock()
    calls = []

    def transport(*args):
        calls.append(args)
        clock.now = 100
        return b'late response'

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock)
    assert caught.value.attempts[0]['outcome'] == 'late_response'
    assert len(calls) == 1 and clock.waits == []


def test_deadline_consumed_by_timeout_no_retry_sleep():
    clock = Clock()

    def transport(*_):
        clock.now = 100
        raise TimeoutError()

    with pytest.raises(RetryExhausted) as caught:
        invoke(transport, clock)
    assert caught.value.reason == 'deadline_exceeded'
    assert len(caught.value.attempts) == 1 and clock.waits == []


@pytest.mark.parametrize('options', [
    {'max_attempts': 0}, {'max_attempts': 4}, {'max_attempts': True},
    {'deadline': float('inf')}, {'attempt_timeout': 0},
    {'attempt_timeout': float('nan')}, {'initial_delay': .5},
    {'max_delay': .5},
])
def test_bounded_configuration_rejects_invalid_values(options):
    clock = Clock()
    with pytest.raises(ValueError):
        invoke(lambda *_: b'ok', clock, **options)


def test_mutable_payload_rejected():
    with pytest.raises(ValueError):
        call_with_retry(bytearray(b'request'), lambda *_: b'ok', deadline=100,
                        attempt_timeout=1)
