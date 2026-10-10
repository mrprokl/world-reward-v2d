"""Bounded read-only wait for an entire explicitly named owned service.

Success is not a GPU lease, a quality gate, or permission to read references.
By default failed, missing and malformed units fail immediately. An explicit
terminal-only observation can release independent work after a coherent failure;
it never declares predecessor success. Nothing is restarted.
"""
import argparse
import json
import re
import subprocess
import time


MAX_WAIT = 43200
PROPERTIES = ('LoadState', 'ActiveState', 'SubState', 'Result', 'ExecMainStatus', 'MainPID')


def require(value, message):
    if not value:
        raise ValueError(message)


def owned_unit(value):
    require(type(value) is str and re.fullmatch(
        r'world-reward-[a-z0-9][a-z0-9-]{0,80}(?:\.service)?', value), 'Explicit owned service required')
    return value if value.endswith('.service') else value + '.service'


def unit_ready(state):
    require(type(state) is dict and set(state) == set(PROPERTIES) and
        state['LoadState'] == 'loaded' and
        re.fullmatch(r'(?:0|[1-9][0-9]{0,2})', state['ExecMainStatus']) and
        int(state['ExecMainStatus']) <= 255 and
        re.fullmatch(r'(?:0|[1-9][0-9]{0,9})', state['MainPID']), 'Loaded well-formed service required')
    active = state['ActiveState']
    if active == 'inactive':
        require(state['SubState'] == 'dead' and state['Result'] == 'success' and
            state['ExecMainStatus'] == '0' and state['MainPID'] == '0',
            'Entire service must have successfully exited, with no live main process')
        return True
    require(active in ('active', 'activating', 'deactivating', 'reloading'),
        'Failed or unknown predecessor must not release downstream work')
    return False


def snapshot(unit, expected_command=()):
    properties = (*PROPERTIES, 'ExecStart') if expected_command else PROPERTIES
    result = subprocess.run(['/usr/bin/systemctl', 'show', unit,
        *(f'--property={key}' for key in properties)], capture_output=True, timeout=15, check=False)
    require(result.returncode == 0 and 0 < len(result.stdout) <= 65536, 'Bounded service snapshot required')
    fields = {}
    for line in result.stdout.decode('utf-8', errors='strict').splitlines():
        key, separator, value = line.partition('=')
        require(separator and key in properties and key not in fields and value, 'Unique complete service properties required')
        fields[key] = value
    require(set(fields) == set(properties), 'Complete service snapshot required')
    if expected_command:
        command = fields.pop('ExecStart')
        require(all(type(token) is str and token and re.search(
            r'(?<![^\s;])' + re.escape(token) + r'(?=$|[\s;])', command)
            for token in expected_command), 'Owned service command must bind the exact original producer')
    return fields


def wait_success_unit(unit, *, timeout=MAX_WAIT, expected_command=()):
    unit = owned_unit(unit)
    require(type(timeout) is int and 0 < timeout <= MAX_WAIT, 'Bounded scheduling budget required')
    deadline = time.monotonic() + timeout
    while True:
        state = snapshot(unit, expected_command)
        if unit_ready(state):
            return dict(unit=unit, **state)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Entire predecessor scheduling budget exhausted')
        time.sleep(min(30, remaining))


def unit_terminal(state):
    if type(state) is not dict or state.get('ActiveState') != 'failed':
        return unit_ready(state)
    require(set(state) == set(PROPERTIES) and state['LoadState'] == 'loaded' and
        state['SubState'] == 'failed' and state['MainPID'] == '0' and
        type(state['ExecMainStatus']) is str and
        re.fullmatch(r'(?:0|[1-9][0-9]{0,2})', state['ExecMainStatus']) and
        int(state['ExecMainStatus']) <= 255, 'Coherent failed service with no live main process required')
    result = state['Result']
    require(result in ('exit-code', 'signal', 'core-dump') and int(state['ExecMainStatus']) > 0 or
        result in ('timeout', 'watchdog', 'oom-kill', 'resources', 'protocol', 'start-limit-hit'),
        'Explicit coherent terminal failure required; never claim success')
    return True


def wait_terminal_unit(unit, *, timeout=MAX_WAIT, expected_command=()):
    """Observe termination only, for scientifically independent successor work."""
    unit = owned_unit(unit)
    require(type(timeout) is int and 0 < timeout <= MAX_WAIT, 'Bounded scheduling budget required')
    deadline = time.monotonic() + timeout
    while True:
        state = snapshot(unit, expected_command)
        if unit_terminal(state):
            return dict(unit=unit, **state)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('Entire predecessor scheduling budget exhausted')
        time.sleep(min(30, remaining))


def main():
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('unit', type=owned_unit)
    parser.add_argument('--timeout-seconds', type=int, default=MAX_WAIT)
    parser.add_argument('--allow-failed-terminal', action='store_true',
        help='Observe coherent termination only; not predecessor success or quality')
    args = parser.parse_args()
    observe = wait_terminal_unit if args.allow_failed_terminal else wait_success_unit
    state = observe(args.unit, timeout=args.timeout_seconds)
    print(json.dumps(dict(status='terminal_observed' if args.allow_failed_terminal else 'terminal_success',
        **state)), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(json.dumps(dict(status='fail', error_type=type(error).__name__)), flush=True)
        raise SystemExit(1)
