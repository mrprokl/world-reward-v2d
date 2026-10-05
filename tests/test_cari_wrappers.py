"""Run wrappers against tiny fake shell commands, never Docker/Azure or media."""

import importlib.util
import hashlib
import json
import os
from pathlib import Path
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[1]
WRAPPERS = {
    "prepare": ("run_cari_prepare.sh", "cari_prepare.py", "object_pose_full", "world-reward-object-pose-full.service"),
    "forward": ("run_cari_forward.sh", "cari_forward.py", "cari_inputs", "world-reward-cari-prepare-v2.service"),
    "converter": ("run_cari_converter.sh", "cari_converter.py", "cari_forward", "world-reward-cari-forward.service"),
    "adapter": ("run_cari_body_adapter_smoke.sh", "cari_body_adapter_smoke.py", "body_full", None),
}


@pytest.fixture
def fake_shell(tmp_path):
    binary = tmp_path / "bin"
    binary.mkdir()
    log, state = tmp_path / "commands.log", tmp_path / "state"
    state.write_text("inactive")
    commands = {
        "docker": 'printf "docker\\0" >> "$FAKE_LOG"; printf "%s\\0" "$@" >> "$FAKE_LOG"',
        "id": 'printf "123\\n"',
        "systemctl": '''printf "systemctl %s\\n" "$*" >> "$FAKE_CTL_LOG"
case "$3" in
  --property=LoadState) printf '%s\\n' "$FAKE_LOAD" ;;
  --property=ActiveState) cat "$FAKE_STATE" ;;
  *) exit 9 ;;
esac''',
        "sleep": '''printf "sleep %s\\n" "$*" >> "$FAKE_CTL_LOG"
printf '%s' "$FAKE_STOP_STATE" > "$FAKE_STATE"
if [[ -n "${FAKE_CREATE_REPORT:-}" ]]; then mkdir -p "$(dirname "$FAKE_CREATE_REPORT")"; printf '{}' > "$FAKE_CREATE_REPORT"; fi''',
    }
    for name, source in commands.items():
        path = binary / name
        path.write_text("#!/usr/bin/env bash\nset -eu\n" + source + "\n")
        path.chmod(0o755)
    environment = dict(os.environ, PATH=str(binary) + os.pathsep + os.environ["PATH"],
                       WR_ROOT=str(tmp_path / "runtime"), WR_CODE=str(ROOT), WR_CODE_REVISION="a" * 40,
                       FAKE_LOG=str(log), FAKE_CTL_LOG=str(tmp_path / "ctl.log"), FAKE_STATE=str(state),
                       FAKE_LOAD="loaded", FAKE_STOP_STATE="inactive")
    def run(mode, *arguments):
        return subprocess.run(["bash", str(ROOT / "infra" / WRAPPERS[mode][0]), *arguments],
                              env=environment, text=True, capture_output=True, timeout=5)
    def report(mode, episode=15, stage=None):
        path = Path(environment["WR_ROOT"]) / f"outputs/episode_{episode:06d}" / (stage or WRAPPERS[mode][2]) / "report.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}")
        return path
    return environment, run, report, log, state


def docker_arguments(log):
    assert log.exists()
    return log.read_bytes().decode().strip("\0").split("\0")


@pytest.mark.parametrize("mode", WRAPPERS)
def test_default15_preserves_original_stage_dependencies_and_passes_explicit_episode(fake_shell, mode):
    env, run, report, log, _ = fake_shell
    report(mode)
    result = run(mode)
    assert result.returncode == 0, result.stderr
    arguments = docker_arguments(log)
    assert arguments[-3:] == [str(ROOT / "infra" / WRAPPERS[mode][1]), "--episode", "15"]
    assert "--network" in arguments and arguments[arguments.index("--network") + 1] == "none"
    expected_unit = WRAPPERS[mode][3]
    ctl = Path(env["FAKE_CTL_LOG"])
    if expected_unit:
        assert expected_unit in ctl.read_text()
    else:
        assert not ctl.exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("episode", [0, 1, 29])
def test_nondefault_episode_requires_own_report_without_implicit_episode15_wait(fake_shell, mode, episode):
    env, run, report, log, _ = fake_shell
    report(mode, episode)
    result = run(mode, "--episode", str(episode))
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-2:] == ["--episode", str(episode)]
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("episode", [0, 15, 29])
def test_explicit_report_only_mode_skips_units_but_never_forwards_flag(fake_shell, mode, episode):
    env, run, report, log, state = fake_shell
    report(mode, episode)
    # Even an active historical unit is irrelevant to an explicitly serial route.
    state.write_text("active")
    result = run(mode, "--episode", str(episode), "--no-wait")
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-2:] == ["--episode", str(episode)]
    assert "--no-wait" not in docker_arguments(log)
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
def test_report_only_mode_still_requires_selected_report(fake_shell, mode):
    env, run, report, log, _ = fake_shell
    report(mode, 0)
    result = run(mode, "--episode", "15", "--no-wait")
    assert result.returncode != 0 and "episode_000015" in result.stderr
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("arguments", [
    ("--no-wait", "--no-wait"),
    ("--no-wait", "--wait-for", "world-reward-x"),
    ("--wait-for", "world-reward-x", "--no-wait"),
])
def test_report_only_duplicate_or_conflicting_wait_fails_before_io(fake_shell, mode, arguments):
    env, run, _, log, _ = fake_shell
    result = run(mode, *arguments)
    assert result.returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("arguments", [("--kernel-only", "--no-wait"), ("--no-wait", "--kernel-only")])
def test_checkpoint_only_rejects_redundant_report_only_flag(fake_shell, arguments):
    env, run, _, log, _ = fake_shell
    assert run("forward", *arguments).returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
def test_other_episode_report_cannot_satisfy_selected_clip(fake_shell, mode):
    env, run, report, log, _ = fake_shell
    report(mode, 15)
    result = run(mode, "--episode", "0")
    assert result.returncode != 0 and "episode_000000" in result.stderr
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("arguments", [
    ("--episode",), ("--episode", "-1"), ("--episode", "30"), ("--episode", "true"),
    ("--episode", "False"), ("--episode", "15.0"), ("--episode", "01"), ("--episode", "+1"),
    ("--episode", "1", "--episode", "2"), ("--unknown",), ("--episode=1",),
    ("--wait-for",), ("--wait-for", "other-service"), ("--wait-for", "world-reward-x;evil"),
    ("--wait-for", "world-reward-../x"), ("--wait-for", "world-reward-x", "--wait-for", "world-reward-y"),
])
def test_invalid_unknown_duplicate_alias_or_injection_arguments_fail_before_commands(fake_shell, mode, arguments):
    env, run, _, log, _ = fake_shell
    result = run(mode, *arguments)
    assert result.returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", ["prepare", "converter", "adapter"])
def test_kernel_only_not_accepted_by_nonforward_wrappers(fake_shell, mode):
    env, run, _, log, _ = fake_shell
    result = run(mode, "--kernel-only")
    assert result.returncode == 2 and not log.exists()
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_checkpoint_only_has_body_dependency_but_never_waits_for_prepare(fake_shell, episode):
    env, run, report, log, _ = fake_shell
    report("forward", episode, "body_full")
    result = run("forward", "--episode", str(episode), "--kernel-only")
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-3:] == ["--episode", str(episode), "--kernel-only"]
    assert not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("arguments", [("--kernel-only", "--kernel-only"),
                                        ("--kernel-only", "--wait-for", "world-reward-prepare")])
def test_checkpoint_only_rejects_duplicate_or_preparation_wait(fake_shell, arguments):
    env, run, _, log, _ = fake_shell
    assert run("forward", *arguments).returncode == 2
    assert not log.exists() and not Path(env["FAKE_CTL_LOG"]).exists()


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("suffix", ["", ".service"])
def test_explicit_wait_unit_normalized_not_forwarded_to_python(fake_shell, mode, suffix):
    env, run, report, log, _ = fake_shell
    report(mode, 0)
    result = run(mode, "--wait-for", "world-reward-own-episode0" + suffix, "--episode", "0")
    assert result.returncode == 0, result.stderr
    assert "world-reward-own-episode0.service" in Path(env["FAKE_CTL_LOG"]).read_text()
    assert "--wait-for" not in docker_arguments(log)


@pytest.mark.parametrize("mode", WRAPPERS)
def test_active_explicit_producer_waits_once_then_requires_selected_report(fake_shell, mode):
    env, run, _, log, state = fake_shell
    state.write_text("active")
    target = Path(env["WR_ROOT"]) / "outputs/episode_000029" / WRAPPERS[mode][2] / "report.json"
    env["FAKE_CREATE_REPORT"] = str(target)
    result = run(mode, "--episode", "29", "--wait-for", "world-reward-own29")
    assert result.returncode == 0, result.stderr
    assert target.exists() and log.exists()
    assert Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


@pytest.mark.parametrize("mode", WRAPPERS)
@pytest.mark.parametrize("state_value", ["inactive", "failed", "mystery"])
def test_stopped_failed_unknown_producer_without_report_fails_without_waiting(fake_shell, mode, state_value):
    env, run, _, log, state = fake_shell
    state.write_text(state_value)
    result = run(mode, "--episode", "0", "--wait-for", "world-reward-no-report")
    assert result.returncode != 0 and not log.exists()
    assert "sleep" not in Path(env["FAKE_CTL_LOG"]).read_text()


@pytest.mark.parametrize("mode", WRAPPERS)
def test_missing_unit_cannot_wait_forever_or_claim_success_without_report(fake_shell, mode):
    env, run, _, log, _ = fake_shell
    env["FAKE_LOAD"] = "not-found"
    result = run(mode, "--episode", "0", "--wait-for", "world-reward-missing")
    assert result.returncode != 0 and "absent" in result.stderr and not log.exists()
    assert "sleep" not in Path(env["FAKE_CTL_LOG"]).read_text()


def test_collected_old_default_unit_can_use_frozen_report(fake_shell):
    env, run, report, log, _ = fake_shell
    report("prepare")
    env["FAKE_LOAD"] = "not-found"
    assert run("prepare").returncode == 0
    assert log.exists()


def test_failed_unit_is_not_rehabilitated_by_existing_report(fake_shell):
    _, run, report, log, state = fake_shell
    report("forward")
    state.write_text("failed")
    result = run("forward")
    assert result.returncode != 0 and "failed" in result.stderr and not log.exists()


def test_successful_process_completion_without_report_still_fails(fake_shell):
    env, run, _, log, state = fake_shell
    state.write_text("active")
    result = run("prepare", "--episode", "29", "--wait-for", "world-reward-own29")
    assert result.returncode != 0 and "stopped without" in result.stderr and not log.exists()
    assert Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


def test_existing_report_does_not_bypass_active_producer(fake_shell):
    env, run, report, log, state = fake_shell
    report("converter", 0)
    state.write_text("active")
    result = run("converter", "--episode", "0", "--wait-for", "world-reward-own0")
    assert result.returncode == 0, result.stderr
    assert log.exists() and Path(env["FAKE_CTL_LOG"]).read_text().count("sleep 30") == 1


def test_wait_is_bounded_without_real_sleep_or_subprocess_systemctl(tmp_path):
    helper = ROOT / "infra/cari_wrapper_common.sh"
    script = f'''set -eu
source "{helper}"
systemctl() {{ if [[ "$3" == --property=LoadState ]]; then echo loaded; else echo active; fi; }}
sleep() {{ :; }}
wr_require_dependency_report "{tmp_path}/missing.json" world-reward-running.service
'''
    result = subprocess.run(["bash", "-c", script], text=True, capture_output=True, timeout=5)
    assert result.returncode != 0 and "12 hours" in result.stderr


def test_immutable_launcher_import_closure_includes_common_shell_helper_and_python_entrypoint():
    spec = importlib.util.spec_from_file_location("world_reward_test_wrapper_launcher", ROOT / "infra/azure_job.py")
    launcher = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(launcher)
    files = {str(path.relative_to(ROOT)): path.read_bytes()
             for base in ("infra", "src", "configs") for path in (ROOT / base).rglob("*")
             if path.is_file() and "__pycache__" not in path.parts}
    files["pyproject.toml"] = (ROOT / "pyproject.toml").read_bytes()
    for shell, python, _, _ in WRAPPERS.values():
        closure = launcher.runtime_bundle_paths(files, "infra/" + shell)
        assert "infra/cari_wrapper_common.sh" in closure
        assert "infra/" + python in closure


@pytest.mark.parametrize("episode", [0, 15, 29])
def test_refined_converter_uses_only_refined_dependency_and_distinct_explicit_argv(fake_shell, episode):
    env, run, report, log, state = fake_shell
    report("converter", episode, "cari_refined")
    state.write_text("active")
    result = run("converter", "--episode", str(episode), "--bundle-source", "refined", "--no-wait")
    assert result.returncode == 0, result.stderr
    assert docker_arguments(log)[-4:] == ["--episode", str(episode), "--bundle-source", "refined"]
    assert not Path(env["FAKE_CTL_LOG"]).exists()


def test_refined_conversion_never_falls_back_to_forward_report(fake_shell):
    _, run, report, log, _ = fake_shell
    report("converter", 15)
    result = run("converter", "--bundle-source", "refined")
    assert result.returncode != 0 and "cari_refined" in result.stderr and not log.exists()


@pytest.mark.parametrize("args", [("--bundle-source",), ("--bundle-source", "other"),
                                  ("--bundle-source", "forward", "--bundle-source", "refined")])
def test_invalid_bundle_routing_rejected_before_docker(fake_shell, args):
    _, run, _, log, _ = fake_shell
    assert run("converter", *args).returncode == 2 and not log.exists()


@pytest.mark.parametrize("mode", ["prepare", "forward", "adapter"])
def test_bundle_routing_converter_only(fake_shell, mode):
    _, run, _, log, _ = fake_shell
    assert run(mode, "--bundle-source", "refined").returncode == 2 and not log.exists()


@pytest.mark.parametrize('profile',['solid','surface'])
@pytest.mark.parametrize('episode',[0,15,29])
def test_explicit_geometry_profile_routes_its_own_pose_report(fake_shell,profile,episode):
    env,run,report,log,_=fake_shell
    if profile=='surface':
        # Numerical/source authentication is tested separately below; no old snapshot is executed.
        spy=Path(env['PATH'].split(os.pathsep)[0])/'python3'
        spy.write_text('#!/usr/bin/env bash\ncat >/dev/null\n'+''.join('printf "%s\\n" '+repr(str(Path(env['WR_ROOT'])/'jobs'/('b'*40)/entry))+'\n' for entry in ('run_object_budget_solid','run_surface_qslim_qualify','run_surface_identity_qualify')))
        spy.chmod(0o755)
    report('prepare',episode,stage='object_pose_full_'+profile)
    result=run('prepare','--episode',str(episode),'--mesh-source',profile)
    assert result.returncode==0,result.stderr
    assert docker_arguments(log)[-4:]==['--episode',str(episode),'--mesh-source',profile]
    if profile=='surface':
        args=docker_arguments(log)
        for entry in ('run_object_budget_solid','run_surface_qslim_qualify','run_surface_identity_qualify'):
            parent=Path(env['WR_ROOT'])/'jobs'/('b'*40)/entry
            assert f'type=bind,src={parent},dst={parent},readonly' in args
        assert f'type=bind,src={env["WR_ROOT"]}/jobs,dst={env["WR_ROOT"]}/jobs,readonly' not in args
    else:
        assert not any('/jobs/' in arg for arg in docker_arguments(log))
    assert not Path(env['FAKE_CTL_LOG']).exists()


@pytest.mark.parametrize('profile',['solid','surface'])
def test_geometry_profile_never_falls_back_to_default_report(fake_shell,profile):
    _,run,report,log,_=fake_shell;report('prepare',29)
    result=run('prepare','--episode','29','--mesh-source',profile)
    assert result.returncode!=0 and 'object_pose_full_'+profile in result.stderr and not log.exists()


@pytest.mark.parametrize('mode',['forward','converter','adapter'])
def test_geometry_profile_prepare_only(fake_shell,mode):
    _,run,_,log,_=fake_shell
    assert run(mode,'--mesh-source','surface').returncode==2 and not log.exists()


def test_surface_has_no_solid_query_requalification_alias(fake_shell):
    _,run,_,log,_=fake_shell
    assert run('prepare','--mesh-source','surface','--query-requalification').returncode==2 and not log.exists()


@pytest.fixture
def surface_prepare_history(tmp_path):
    """Only authored tiny source/receipt metadata; never an actual model/native invocation."""
    import sys
    spec=importlib.util.spec_from_file_location('surface_prepare_test_rt',ROOT/'infra/mediapipe_cpu_runtime_verify.py')
    rt=importlib.util.module_from_spec(spec);spec.loader.exec_module(rt)
    root=tmp_path/'runtime';revision='a'*40;code=root/'jobs'/revision/'run_cari_prepare/code';code.mkdir(parents=True)
    def write(p,b):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b);p.chmod(0o444)
        return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
    def record(p,v):return write(p,json.dumps(v,sort_keys=True).encode())
    def seal(p):
        for child in sorted((p,*p.rglob('*')),key=lambda x:len(x.parts),reverse=True):
            if child.is_dir():child.chmod(0o555)
    def markers(c,r):write(c.parent/'revision',(r+'\n').encode());write(c.parent/'source-sha256',('f'*64+'\n').encode())
    for n in ('mediapipe_cpu_runtime_verify.py','run_cari_prepare.sh','cari_wrapper_common.sh'):
        write(code/'infra'/n,(ROOT/'infra'/n).read_bytes())
    markers(code,revision)
    old=[];identities=[]
    for r,entry in zip(('b'*40,'c'*40,'d'*40),('run_object_budget_solid','run_surface_qslim_qualify','run_surface_identity_qualify')):
        c=root/'jobs'/r/entry/'code';write(c/'infra/original.cpp',b'// Immutable source text, never executed.\n');markers(c,r);seal(c)
        binding=rt.source(root,c,r,entry,())
        p=(root/'outputs/episode_000009'/('object_budget_surface_'+r)/'report.json' if not old else root/'results'/('surface-qslim-qualify-'if len(old)==1 else'surface-identity-qualify-')/r/'native.json')
        # The published result directories concatenate the revision into their name.
        if old:p=root/'results'/(('surface-qslim-qualify-'if len(old)==1 else'surface-identity-qualify-')+r)/'native.json'
        identities.append(record(p,dict(source_binding=binding)if not old else dict(source_proof=dict(source_binding=binding))))
        old.append(c.parent)
    record(code/'configs/surface_mesh_000009_pins.json',dict(schema='world_reward.surface_mesh_pins.v1',episode_index=9,input_sha256='e'*64,metric_scale_baked_once=1,report=dict(**identities[0],producer_revision='b'*40),files={},source_helpers={}))
    record(code/'configs/surface_qslim_qualification_pins.json',dict(schema='world_reward.surface_qslim_qualification_pins.v1',producer_revision='c'*40,native=identities[1]))
    record(code/'configs/surface_identity_qualification_pins.json',dict(schema='world_reward.surface_identity_qualification_pins.v1',producer_revision='d'*40,native=identities[2]))
    seal(code)
    source=(ROOT/'infra/run_cari_prepare.sh').read_text().split("<<'PYSURFACE'\n",1)[1].split('\nPYSURFACE',1)[0]
    def call():return subprocess.run([sys.executable,'-I','-B','-',str(root),str(code),revision,'9'],input=source,text=True,capture_output=True,timeout=10)
    return call,old,code


def test_surface_prepare_authenticates_exact_three_source_only_parents(surface_prepare_history):
    call,parents,_=surface_prepare_history;result=call()
    assert result.returncode==0,result.stderr
    assert result.stdout.splitlines()==list(map(str,parents))


@pytest.mark.parametrize('fault',['source_mutation','extra_parent_entry','wrong_revision','receipt_mutation','source_binary'])
def test_surface_prepare_rejects_changed_or_foreign_historical_proof(surface_prepare_history,fault):
    call,parents,code=surface_prepare_history
    if fault=='extra_parent_entry':p=parents[0]/'foreign';p.write_bytes(b'not source')
    elif fault=='wrong_revision':
        p=parents[0]/'revision';p.chmod(0o644);p.write_text('e'*40+'\n');p.chmod(0o444)
    elif fault=='receipt_mutation':
        p=code/'configs/surface_qslim_qualification_pins.json';p.chmod(0o644);v=json.loads(p.read_text());v['native']['sha256']='0'*64;p.write_text(json.dumps(v));p.chmod(0o444)
    else:
        p=parents[0]/'code/infra/original.cpp';p.chmod(0o644);p.write_bytes(b'ELF\0binary'if fault=='source_binary'else b'changed oldsource');p.chmod(0o444)
    result=call();assert result.returncode!=0 and result.stdout==''
