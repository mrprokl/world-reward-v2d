"""Opt-in dispatch/mount contracts only; no local geometry, native QEM or models."""
import ast
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
IMAGE = 'sha256:c8fb1632a6908a82aeeeb73c36a00f17a26b81f95f4d2d498c53f75894e21137'


@pytest.fixture
def driver(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/'infra'))
    spec = importlib.util.spec_from_file_location('conditioned_dispatch_test', REPO/'infra/object_budget_volume.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('episode', range(30))
def test_episode_backend_output_are_explicit_and_separate(driver, episode):
    argv = ['--episode', str(episode)]
    assert driver._argument_parser().parse_args(argv).backend is None
    assert driver.output_relative(episode) == f'outputs/episode_{episode:06d}/object_budget_volume'
    for backend in ('volume', 'conditioned'):
        args = driver._argument_parser().parse_args(argv+['--backend', backend])
        assert args.episode == episode and args.backend == backend
        assert driver.output_relative(episode, backend) == f'outputs/episode_{episode:06d}/object_budget_{backend}'


@pytest.mark.parametrize('argv', [
    ['--episode', '9', '--backend', 'conditioned', '--backend', 'conditioned'],
    ['--episode', '9', '--backend', 'volume', '--backend', 'conditioned'],
    ['--episode', '9', '--backend', 'other'], ['--episode', '9', '--backend'],
    ['--episode', '9', '--back', 'conditioned'], ['--episode', '9', '--backend', ''],
    ['--episode', '9', '--backend', 'conditioned', '--adopt'],
    ['--episode', '9', '--backend', 'conditioned', '--budget', '1801'],
])
def test_backend_malformed_or_repeated_rejected_before_runtime(driver, argv, monkeypatch):
    monkeypatch.setattr(driver.platform, 'system', lambda: pytest.fail('Runtime reached'))
    with pytest.raises(SystemExit) as error:
        driver.main(argv)
    assert error.value.code == 2


@pytest.mark.parametrize('backend', [None, True, 1, 'guarded', ' conditioned', 'volume/../conditioned'])
def test_output_backend_no_path_or_implicit_fallback(driver, backend):
    with pytest.raises(ValueError):
        driver.output_relative(9, backend)


def test_conditioned_dispatches_once_integer_and_propagates_failure_without_volume(driver, monkeypatch):
    calls = []
    def main(episode):
        calls.append(episode)
        raise ValueError('Qualification pins absent')
    monkeypatch.setitem(sys.modules, 'object_budget_conditioned', SimpleNamespace(main=main))
    monkeypatch.setattr(driver.proposal, 'produce', lambda *a, **k: pytest.fail('Old backend ran'))
    with pytest.raises(ValueError, match='Qualification pins absent'):
        driver.main(['--episode', '9', '--backend', 'conditioned'])
    assert calls == [9] and type(calls[0]) is int


def test_legacy_numerical_functions_and_main_after_dispatch_are_ast_unchanged():
    historical = subprocess.check_output(['rtk', 'git', 'show', '23c5731e792482cc9a229422d32aacf370b084fc:infra/object_budget_volume.py'],
                                         cwd=REPO, text=True)
    old, new = ast.parse(historical), ast.parse((REPO/'infra/object_budget_volume.py').read_text())
    functions = lambda module: {n.name: n for n in module.body if isinstance(n, ast.FunctionDef)}
    a, b = functions(old), functions(new)
    assert ast.dump(a['prerequisites'], include_attributes=False) == ast.dump(b['prerequisites'], include_attributes=False)
    old_body, new_body = a['main'].body, b['main'].body
    # The sole new statement is the explicit conditioned dispatch after parsing.
    assert ast.dump(new_body[1], include_attributes=False).startswith('If(')
    assert [ast.dump(n, include_attributes=False) for n in old_body] == [
        ast.dump(n, include_attributes=False) for n in [new_body[0], *new_body[2:]]]


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path/'root'; (root/'outputs').mkdir(parents=True); (root/'results').mkdir()
    (root/'results/image-volume-qem.json').write_text(json.dumps({'status': 'pass', 'image_id': IMAGE}))
    helper = root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py'
    helper.parent.mkdir(parents=True); helper.write_text('public official helper fixture')
    code = tmp_path/'code'; code.mkdir(); binary = tmp_path/'bin'; binary.mkdir(); log = tmp_path/'docker.json'
    for name, body in {'id': 'printf "1000\\n"', 'chown': 'exit 0',
                       'timeout': 'shift 3; exec "$@"'}.items():
        p = binary/name; p.write_text('#!/bin/sh\n'+body+'\n'); p.chmod(0o755)
    docker = binary/'docker'
    docker.write_text(f'#!{sys.executable}\nimport json,os,sys\n'
                      'with open(os.environ["CALLS"],"x") as f:json.dump(sys.argv[1:],f)\n'
                      'sys.exit(int(os.environ.get("EXIT","0")))\n'); docker.chmod(0o755)
    env = dict(os.environ, WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION='a'*40,
               CALLS=str(log), PATH=str(binary)+':'+os.environ['PATH'])
    def run(argv):
        return subprocess.run(['rtk', 'proxy', 'bash', str(REPO/'infra/run_object_budget_volume.sh'), *argv],
                              env=env, capture_output=True, text=True, timeout=5)
    return root, code, helper, log, env, run


@pytest.mark.parametrize('episode', [0, 9, 29])
def test_conditioned_single_container_exact_narrow_readonly_cpu_mounts(runtime, episode):
    root, code, helper, log, env, run = runtime
    base = root/f'outputs/episode_{episode:06d}'; base.mkdir()
    for name in ('object_pose_full', 'object_budget_volume'):
        (base/name).mkdir(); (base/name/'report.json').write_text('preserved old failure')
    argv = ['--episode', str(episode), '--backend', 'conditioned']
    result = run(argv); assert result.returncode == 0, result.stderr
    args = json.loads(log.read_text()); out = base/'object_budget_conditioned'
    assert out.is_dir() and not any(out.iterdir())
    assert args[-6:] == [IMAGE, str(code/'infra/object_budget_volume.py'), *argv]
    assert '--gpus' not in args and '--read-only' in args
    for flag, expected in (('--network', 'none'), ('--user', '1000:1000'), ('--memory', '16g'),
                           ('--cpus', '4'), ('--cap-drop', 'ALL'), ('--security-opt', 'no-new-privileges')):
        assert args[args.index(flag)+1] == expected
    assert args[args.index('--tmpfs')+1] == '/tmp:rw,nosuid,nodev,noexec,size=8g'
    mounts = [args[i+1] for i, item in enumerate(args) if item == '--mount']
    assert f'type=bind,src={helper},dst={helper},readonly' in mounts
    assert f'type=bind,src={root}/vendor/v2d_submission_kit,dst={root}/vendor/v2d_submission_kit,readonly' not in mounts
    assert not any('validation/volume_qem_v1' in item for item in mounts)
    assert f'type=bind,src={base},dst={base},readonly' in mounts
    assert f'type=bind,src={root}/results,dst={root}/results,readonly' in mounts
    assert [m for m in mounts if not m.endswith(',readonly')] == [f'type=bind,src={out},dst={out}']
    for name in ('object_pose_full', 'object_budget_volume'):
        assert (base/name/'report.json').read_text() == 'preserved old failure'


@pytest.mark.parametrize('argv', [[], ['--episode', '00'], ['--episode', '30'], ['--episode', '9', '--backend'],
                                ['--backend', 'conditioned', '--episode', '9'],
                                ['--episode', '9', '--backend', 'conditioned', '--backend', 'volume'],
                                ['--episode', '9', '--backend', 'other'],
                                ['--episode', '9', '--backend', 'conditioned', '--resume']])
def test_shell_duplicate_or_unknown_options_create_no_output(runtime, argv):
    root, code, helper, log, env, run = runtime
    result = run(argv)
    assert result.returncode == 2 and not log.exists() and not any((root/'outputs').iterdir())


@pytest.mark.parametrize('kind', ['directory', 'file', 'symlink'])
def test_conditioned_preserves_existing_namespace_no_overwrite(runtime, kind):
    root, code, helper, log, env, run = runtime
    base = root/'outputs/episode_000009'; base.mkdir(); out = base/'object_budget_conditioned'
    if kind == 'directory': out.mkdir(); (out/'report.json').write_text('retained')
    elif kind == 'file': out.write_text('retained')
    else: out.symlink_to('absent')
    assert run(['--episode', '9', '--backend', 'conditioned']).returncode != 0
    assert not log.exists()
    if kind == 'directory': assert (out/'report.json').read_text() == 'retained'
    elif kind == 'file': assert out.read_text() == 'retained'
    else: assert out.is_symlink()


def test_conditioned_image_and_helper_fail_closed_before_docker(runtime):
    root, code, helper, log, env, run = runtime
    (root/'outputs/episode_000009').mkdir()
    (root/'results/image-volume-qem.json').write_text(json.dumps({'status': 'pass', 'image_id': 'sha256:'+'a'*64}))
    assert run(['--episode', '9', '--backend', 'conditioned']).returncode != 0 and not log.exists()


def test_shell_syntax_shared_route_and_no_automatic_adoption():
    path = REPO/'infra/run_object_budget_volume.sh'
    subprocess.run(['rtk', 'proxy', 'bash', '-n', str(path)], check=True)
    source = path.read_text()
    assert source.count('docker run') == 1
    assert 'RUN=(timeout --signal=TERM --kill-after=5s 903s docker run)' in source
    assert 'RUN[3]=1803s' in source
    assert 'object_pose' not in source and 'mv ' not in source and 'rm ' not in source.replace('--rm ', '')
