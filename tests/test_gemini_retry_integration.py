"""Run actual native request/recovery functions with in-memory fake transport.

AST extraction avoids importing cloud/runtime modules or decoding any dataset.
No network, credentials, Azure, image files or output/log writes are involved.
"""
import ast
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import urllib.request

import pytest

from world_reward.gemini_localization import build_request, parse_response
from world_reward.vertex_retry import call_with_retry, RetryExhausted, PermanentFailure


SOURCE = Path(__file__).resolve().parents[1] / 'infra' / 'gemini_initial.py'


class Clock:
    def __init__(self):
        self.now = 10.0
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def task(ep=14, index=14):
    # Builder validates the signature; no real image is needed by fake transport.
    request = build_request(b'\x89PNG\r\n\x1a\nfixture', index, 'a pan', 'hold')
    return dict(ep=ep, index=index, width=100, height=100,
                rgb_sha256='original-rgb-fixture', png_sha256='original-png-fixture',
                request=request)


def request_bytes(t):
    return json.dumps(t['request'], separators=(',', ':')).encode()


def original(t, status='invalid_response'):
    row = {k: v for k, v in t.items() if k != 'request'}
    row.update(request_sha256=hashlib.sha256(request_bytes(t)).hexdigest(),
               status=status, seconds=60.0, person_bbox=None, object_bbox=None)
    if status == 'pair_returned':
        row.update(person_bbox=[0, 0, 100, 100], object_bbox=[10, 10, 20, 20],
                   response_text='immutable original successful response')
    else:
        row['failure_type'] = 'TimeoutError'
    return row


def vertex_response(index=14, *, object_box=None, malformed=False):
    box = [100, 100, 200, 200] if object_box is None else object_box
    label = dict(frame_index=index,
                 person=dict(label='person', box_2d=[0, 0, 1000, 1000]),
                 object=dict(label='pan', box_2d=box))
    text = '{invalid model JSON' if malformed else json.dumps(label)
    return json.dumps(dict(
        candidates=[dict(finishReason='STOP', content=dict(parts=[dict(text=text)]))],
        modelVersion='fake-fixture-model', responseId='fake-response',
    )).encode()


def harness(outcomes, tasks=None):
    tree = ast.parse(SOURCE.read_text())
    native = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'native')
    extracted = [n for n in native.body if (
        isinstance(n, ast.ClassDef) and n.name == 'NoRedirect') or (
        isinstance(n, ast.FunctionDef) and n.name == 'ask')]
    assert len(extracted) == 2
    c = Clock()
    calls, saves, parses = [], [], []
    remaining = iter(outcomes)
    tasks = [task()] if tasks is None else tasks
    preserved = {(t['ep'], t['index']): original(t) for t in tasks}

    class Reply:
        def __init__(self, data):
            self.data = data

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def read(self, maximum):
            assert maximum == 200001
            return self.data

    class Opener:
        def open(self, request, timeout):
            calls.append(dict(payload=request.data, timeout=timeout))
            c.now += .25
            value = next(remaining)
            if isinstance(value, Exception):
                raise value
            return Reply(value)

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def parse(*args, **kwargs):
        parses.append((args, kwargs))
        return parse_response(*args, **kwargs)

    def retry(*args, **kwargs):
        kwargs.update(clock=c.monotonic, sleep=c.sleep, wall_clock=lambda: 0,
                      random01=lambda: .5)
        return call_with_retry(*args, **kwargs)

    fake_urllib = SimpleNamespace(request=SimpleNamespace(
        HTTPRedirectHandler=urllib.request.HTTPRedirectHandler,
        ProxyHandler=urllib.request.ProxyHandler, Request=urllib.request.Request,
        build_opener=lambda *_: Opener(),
    ))
    ns = dict(
        time=SimpleNamespace(monotonic=c.monotonic), json=json, hashlib=hashlib,
        urllib=fake_urllib, token='FAKE_FIXTURE_NOT_A_CREDENTIAL',
        url='https://not-a-real-service.invalid', start=c.now,
        out=Path('/unused-never-written'), require=require, strict=json.loads,
        parse_response=parse, call_with_retry=retry, RetryExhausted=RetryExhausted,
        PermanentFailure=PermanentFailure, save=lambda p, r: saves.append((p, dict(r))),
        c=dict(call_timeout_seconds=90, budget_seconds=420),
        recovery=dict(max_total_attempts=3, initial_delay_seconds=1, max_delay_seconds=30),
        preserved=preserved, tasks=tasks,
    )
    exec(compile(ast.Module(body=extracted, type_ignores=[]), '<native-ask-fixture>', 'exec'), ns)
    return SimpleNamespace(ns=ns, ask=ns['ask'], clock=c, calls=calls, saves=saves,
                           parses=parses, tasks=tasks, preserved=preserved, native=native)


def test_all_twenty_successes_preserved_exact_same_objects_without_api_or_parsing():
    tasks = [task(ep=1, index=i) for i in range(20)]
    h = harness([], tasks)
    for t in tasks:
        key = (t['ep'], t['index'])
        h.preserved[key] = original(t, 'pair_returned')
        assert h.ask(t) is h.preserved[key]
    assert h.calls == h.saves == h.parses == h.clock.waits == []


def test_recovered_timeout_identical_request_bytes_sha_and_original_ledger():
    t = task()
    h = harness([vertex_response()])
    row = h.ask(t)
    assert row['status'] == 'pair_returned'
    assert h.calls == [dict(payload=request_bytes(t), timeout=90.0)]
    assert row['request_sha256'] == h.preserved[(14, 14)]['request_sha256']
    assert row['original_attempt'] == dict(status='transport_timeout', seconds=60.0,
                                          failure_type='TimeoutError')
    assert len(row['attempts']) == len(h.parses) == len(h.saves) == 1
    assert 'FAKE_FIXTURE' not in json.dumps(row)


@pytest.mark.parametrize('response', [b'{bad outer JSON', vertex_response(malformed=True)])
def test_first_successful_technical_malformed_json_is_invalid_schema_not_retried(response):
    h = harness([response])
    row = h.ask(task())
    assert row['status'] == 'invalid_schema'
    assert row['failure_type'] == 'JSONDecodeError'
    assert len(h.calls) == len(row['attempts']) == 1
    assert h.clock.waits == []


def test_safety_block_without_candidates_classified_and_not_retried():
    raw = json.dumps(dict(promptFeedback=dict(blockReason='SAFETY'))).encode()
    h = harness([raw])
    row = h.ask(task())
    assert row['status'] == 'blocked_or_incomplete'
    assert len(h.calls) == len(row['attempts']) == 1
    assert h.parses == h.clock.waits == []


def test_transient_timeout_then_success_two_new_attempts_plus_original():
    h = harness([TimeoutError('fake confidential error'), vertex_response()])
    row = h.ask(task())
    assert row['status'] == 'pair_returned'
    assert len(h.calls) == len(row['attempts']) == 2
    assert row['original_attempt']['status'] == 'transport_timeout'
    assert len(row['attempts']) + 1 == h.ns['recovery']['max_total_attempts']
    assert h.calls[0]['payload'] is h.calls[1]['payload']
    assert h.clock.waits == [1.0]
    assert len(h.parses) == 1
    assert 'confidential' not in json.dumps(row)


def test_two_new_timeouts_retry_exhausted_not_generic_invalid_response():
    h = harness([TimeoutError(), TimeoutError()])
    row = h.ask(task())
    assert row['status'] == 'transport_retry_exhausted'
    assert row['retry_reason'] == 'attempts_exhausted'
    assert len(h.calls) == len(row['attempts']) == 2
    assert h.clock.waits == [1.0] and h.parses == []


def test_semantic_null_abstention_not_retried():
    value = dict(frame_index=14,
                 person=dict(label='person', box_2d=[0, 0, 1000, 1000]),
                 object=dict(label='', box_2d=None))
    raw = json.dumps(dict(candidates=[dict(finishReason='STOP', content=dict(
        parts=[dict(text=json.dumps(value))]))])).encode()
    h = harness([raw])
    row = h.ask(task())
    assert row['status'] == 'abstained' and row['object_bbox'] is None
    assert len(h.calls) == len(h.parses) == 1 and h.clock.waits == []


def test_all_request_hashes_preflight_before_any_billable_call():
    tasks = [task(ep=14, index=14), task(ep=18, index=0)]
    h = harness([], tasks)
    h.preserved[(18, 0)]['request_sha256'] = 'wrong original request hash'
    # The generic cohort check is the only recovery branch preceding NoRedirect
    # that iterates over all fully decoded request tasks.
    preflight = [n for n in h.native.body if isinstance(n, ast.If)
                 and isinstance(n.test, ast.Name) and n.test.id == 'recovery'
                 and any(isinstance(c, ast.For) and isinstance(c.iter, ast.Name)
                         and c.iter.id == 'tasks' for c in n.body)]
    assert len(preflight) == 1
    with pytest.raises(ValueError, match='cohort preflight'):
        exec(compile(ast.Module(body=preflight, type_ignores=[]), '<preflight-fixture>', 'exec'), h.ns)
    assert h.calls == h.saves == []
