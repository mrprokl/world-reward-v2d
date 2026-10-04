"""Public initializer scheduling only, with tiny real files and fake commands."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[1]
WRAPPER = REPO / "infra/run_track1_initializers_only.sh"
TARGETS = ("automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded",
           "body_full", "depth_full", "body_full/cari_adapter")
CHILDREN = ("run_automatic_masks.sh", "run_body_smoke.sh", "run_depth_smoke.sh", "run_scale_smoke.sh",
            "run_object_smoke.sh", "run_cari_body_adapter_smoke.sh")
PRODUCERS = ("automatic_masks.py", "body_smoke.py", "depth_smoke.py", "scale_smoke.py", "object_smoke.py",
             "cari_body_adapter_smoke.py")


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / "runtime"; rev = "a" * 40
    code = root / "jobs" / rev / "run_track1_initializers_only/code"
    (code / "infra").mkdir(parents=True); (root / "results").mkdir()
    (root / "results/camera-render.json").write_text('{"status":"pass"}')
    (root / "jobs/.world-reward-h100.lock").touch()
    (code.parent / "revision").write_text(rev + "\n")
    (code.parent / "source-sha256").write_text("b" * 64 + "\n")
    wrapper = code / "infra" / WRAPPER.name
    wrapper.write_text(WRAPPER.read_text().replace("/srv/scenesmith/world-reward", str(root)))
    for name in ("run_episode_initializers.sh", "cari_wrapper_common.sh"):
        (code / "infra" / name).write_bytes((REPO / "infra" / name).read_bytes())
    for name in PRODUCERS: (code / "infra" / name).write_text("tiny source\n")
    child = '''#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")";episode="$2";printf -v padded '%06d' "$episode"
case "$name" in
 run_automatic_masks.sh) stage=automatic_masks ;;
 run_body_smoke.sh) if [[ " $* " == *" --full-video "* ]];then stage=body_full;else stage=body_smoke;fi ;;
 run_depth_smoke.sh) if [[ " $* " == *" --full-video "* ]];then stage=depth_full;else stage=depth_smoke;fi ;;
 run_scale_smoke.sh) stage=scale_smoke ;;
 run_object_smoke.sh) stage=object_grounded ;;
 run_cari_body_adapter_smoke.sh) stage=cari_adapter ;;
 *) exit 9 ;;
esac
python3 -I -B - "$WR_ROOT/jobs/.world-reward-h100.lock" <<'PYFD'
import fcntl,os,sys
p=sys.argv[1];assert os.fstat(9).st_ino==os.stat(p).st_ino
with open(p,'rb') as probe:
 try:fcntl.flock(probe,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:pass
 else:raise RuntimeError('Parent FD9 lock was not held')
PYFD
printf 'child|%s|%s|fd9-held\\n' "$stage" "$*" >> "$FAKE_LOG"
if [[ "$stage" == "${FAKE_FAIL_STAGE:-}" ]];then exit 7;fi
target="$stage";[[ "$stage" != cari_adapter ]] || target=body_full/cari_adapter
mkdir -p "$WR_ROOT/outputs/episode_$padded/$target"
printf '{}' > "$WR_ROOT/outputs/episode_$padded/$target/report.json"
if [[ "$stage" == cari_adapter && -n "${FAKE_MUTATION:-}" ]];then
 python3 -I -B - "$WR_CODE" "$WR_ROOT" "$FAKE_MUTATION" <<'PYCHANGE'
from pathlib import Path
import sys
code,root=map(Path,sys.argv[1:3]);kind=sys.argv[3]
if kind=='source':
 p=code/'infra/body_smoke.py';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
elif kind=='marker':(code.parent/'revision').write_text('c'*40+'\\n')
elif kind=='inventory':
 code.chmod(0o755);p=code/'extra.py';p.write_text('new');p.chmod(0o444);code.chmod(0o555)
elif kind=='lock':
 p=root/'jobs/.world-reward-h100.lock';p.unlink();p.touch()
PYCHANGE
fi
'''
    for name in CHILDREN: (code / "infra" / name).write_text(child)
    for path in code.rglob("*"): path.chmod(0o555 if path.is_dir() else 0o444)
    code.chmod(0o555)
    bin_dir = tmp_path / "bin"; bin_dir.mkdir(); log = tmp_path / "calls.log"
    stubs = {
        "timeout": '''#!/bin/sh
printf 'timeout|%s|%s\\n' "$3" "$4" >> "$FAKE_LOG"
shift 3
exec "$@"
''',
        "nvidia-smi": '''#!/bin/sh
printf 'smi\\n' >> "$FAKE_LOG"
[ "${FAKE_SMI_FAIL:-0}" != 1 ] || exit 8
[ "${FAKE_GPU_BUSY:-0}" != 1 ] || printf '1234\\n'
''',
        "flock": '''#!/usr/bin/env python3
import fcntl,os,sys
with open(os.environ['FAKE_LOG'],'a') as f:f.write('flock|'+ ' '.join(sys.argv[1:])+'\\n')
if os.environ.get('FAKE_LOCK_BUSY')=='1':sys.exit(1)
fcntl.flock(int(sys.argv[2]),fcntl.LOCK_EX|fcntl.LOCK_NB)
'''}
    for name, source in stubs.items():
        path = bin_dir / name; path.write_text(source); path.chmod(0o755)
    env = dict(WR_ROOT=str(root), WR_CODE=str(code), WR_CODE_REVISION=rev,
               FAKE_LOG=str(log), PATH=str(bin_dir) + ":" + os.environ["PATH"], HOME=str(tmp_path), LANG="C.UTF-8")
    def run(*args):
        return subprocess.run(["bash", str(wrapper), *args], env=env, capture_output=True, text=True, timeout=10)
    return env, run, log, wrapper


@pytest.mark.parametrize("ep", [0, 15, 29])
def test_real_initializer_children_fixed_policy_full_order_lock_held(runtime, ep):
    env, run, log, _ = runtime; result = run("--episode", str(ep))
    assert result.returncode == 0, result.stderr
    rows = log.read_text().splitlines()
    children = [r for r in rows if r.startswith("child|")]
    expected = [("automatic_masks", "--seed-frames 16 --actor-seed-observations 16"),
                ("body_smoke", "--inference-type body"), ("depth_smoke", ""), ("scale_smoke", ""),
                ("object_grounded", "--aligned-pointmap"), ("body_full", "--full-video --inference-type body"),
                ("depth_full", "--full-video"), ("cari_adapter", "")]
    assert children == [f"child|{s}|--episode {ep}{' '+a if a else ''}|fd9-held" for s, a in expected]
    assert [r.split("|")[1] for r in rows if r.startswith("timeout|") and "nvidia-smi" not in r] == ["600s", "5400s"]
    assert rows.count("flock|--nonblock 9") == 1 and rows.count("smi") == 3
    assert rows.index("flock|--nonblock 9") < rows.index(children[0])
    base = Path(env["WR_ROOT"]) / f"outputs/episode_{ep:06d}"
    assert all((base / target / "report.json").is_file() for target in TARGETS)
    assert not (base / "object_pose_full").exists() and not (base / "cari_inputs").exists()
    events = [json.loads(line) for line in result.stdout.splitlines()]
    own = [e for e in events if e.get("mode") == "initializers_only_fixed_all16"]
    assert [(e["stage"],e["phase"]) for e in own] == [(s,p) for s in ("preflight","gpu_preflight","automatic_masks","episode_initializers") for p in ("start","pass")]


@pytest.mark.parametrize("args", [[], ["--episode"], ["--episode", "-1"], ["--episode", "30"],
    ["--episode", "00"], ["--episode", "true"], ["--episode=15"], ["--episode", "0", "--episode", "1"],
    ["--episode", "0", "--actor-policy", "fixed_all16"], ["--episode", "0", "--seed-frames", "3"],
    ["--episode", "0", "--wait-for", "world-reward-old"], ["--episode", "0", "--resume"]])
def test_invalid_arguments_no_gpu_or_outputs(runtime, args):
    env, run, log, _ = runtime
    assert run(*args).returncode == 2 and not log.exists()
    assert not (Path(env["WR_ROOT"]) / "outputs").exists()


@pytest.mark.parametrize("target", TARGETS)
@pytest.mark.parametrize("kind", ["directory", "file", "broken_symlink"])
def test_eight_fresh_targets_before_any_lock_or_gpu(runtime, target, kind):
    env, run, log, _ = runtime
    path = Path(env["WR_ROOT"]) / "outputs/episode_000015" / target; path.parent.mkdir(parents=True, exist_ok=True)
    if kind == "directory": path.mkdir()
    elif kind == "file": path.write_text("frozen")
    else: path.symlink_to("missing")
    result = run("--episode", "15")
    assert result.returncode != 0 and "Frozen initializer target" in result.stderr and not log.exists()


@pytest.mark.parametrize("stage", ["automatic_masks", "body_smoke", "depth_smoke", "scale_smoke", "object_grounded", "body_full", "depth_full", "cari_adapter"])
def test_first_child_failure_preserves_status_and_stops(runtime, stage):
    env, run, log, _ = runtime; env["FAKE_FAIL_STAGE"] = stage
    result = run("--episode", "0")
    assert result.returncode == 7
    children = [r.split("|")[1] for r in log.read_text().splitlines() if r.startswith("child|")]
    assert children[-1] == stage and len(children) == len(set(children))
    assert json.loads(result.stdout.splitlines()[-1])["phase"] == "fail"


@pytest.mark.parametrize("fault", ["GPU_BUSY", "SMI_FAIL", "LOCK_BUSY"])
def test_external_gpu_or_lock_conflict_failfast(runtime, fault):
    env, run, log, _ = runtime; env["FAKE_"+fault] = "1"
    assert run("--episode", "0").returncode != 0
    assert not any(r.startswith("child|") for r in log.read_text().splitlines())


@pytest.mark.parametrize("fault", ["missing", "symlink", "hardlink"])
def test_existing_lock_only_no_creation_or_alias(runtime, fault, tmp_path):
    env, run, log, _ = runtime; path = Path(env["WR_ROOT"]) / "jobs/.world-reward-h100.lock"; path.unlink()
    if fault == "symlink": path.symlink_to(tmp_path / "outside")
    elif fault == "hardlink":
        source = tmp_path / "another"; source.touch(); os.link(source, path)
    result = run("--episode", "0")
    assert result.returncode != 0 and not log.exists()
    if fault == "missing": assert not path.exists()


@pytest.mark.parametrize("fault", ["source", "marker", "inventory", "lock"])
def test_postflight_mutation_rejected_after_all_children(runtime, fault):
    env, run, log, _ = runtime; env["FAKE_MUTATION"] = fault
    result = run("--episode", "0")
    assert result.returncode != 0
    assert len([r for r in log.read_text().splitlines() if r.startswith("child|")]) == 8
    assert json.loads(result.stdout.splitlines()[-1])["phase"] == "fail"


@pytest.mark.parametrize("fault", ["writable", "symlink", "missing", "revision", "archive", "gate"])
def test_source_and_markers_preflight(runtime, fault):
    env, run, log, _ = runtime; code = Path(env["WR_CODE"]); path = code / "infra/run_body_smoke.sh"
    if fault == "writable": path.chmod(0o644)
    elif fault in ("symlink", "missing"):
        path.parent.chmod(0o755);path.unlink()
        if fault == "symlink":path.symlink_to("run_automatic_masks.sh")
    elif fault == "revision":(code.parent / "revision").write_text("main\n")
    elif fault == "archive":(code.parent / "source-sha256").write_text("bad\n")
    else:(Path(env["WR_ROOT"]) / "results/camera-render.json").unlink()
    assert run("--episode", "0").returncode != 0 and not log.exists()


@pytest.mark.parametrize("variable", ["WR_ROOT", "WR_CODE", "WR_CODE_REVISION"])
def test_missing_runtime_not_defaulted(runtime, variable):
    env,run,log,_=runtime;env.pop(variable)
    assert run("--episode","0").returncode!=0 and not log.exists()


def test_entrypoint_cannot_be_relocated(runtime,tmp_path):
    env,_,log,wrapper=runtime;outside=tmp_path/"outside.sh";outside.write_text(wrapper.read_text())
    result=subprocess.run(["bash",str(outside),"--episode","0"],env=env,capture_output=True,text=True,timeout=8)
    assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize("target", ["outputs", "outputs/episode_000000"])
def test_output_symlink_before_gpu(runtime,tmp_path,target):
    env,run,log,_=runtime;path=Path(env['WR_ROOT'])/target;path.parent.mkdir(exist_ok=True)
    external=tmp_path/'external';external.mkdir();path.symlink_to(external,target_is_directory=True)
    assert run('--episode','0').returncode!=0 and not log.exists()


def test_original_scope_bash_and_actual_static_closure():
    subprocess.run(["bash", "-n", str(WRAPPER)], check=True)
    source = WRAPPER.read_text()
    assert len(source.splitlines()) <= 120
    assert "run_object_pose_smoke" not in source and "run_cari_prepare" not in source
    assert "--actor-policy" not in source and "--seed-frames 16 --actor-seed-observations 16" in source
    spec = importlib.util.spec_from_file_location("initializer_archive", REPO / "infra/azure_job.py")
    module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files = {str(p.relative_to(REPO)):p.read_bytes() for name in ("infra","src","configs") for p in (REPO/name).rglob('*')
             if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"] = (REPO / "pyproject.toml").read_bytes()
    selected = set(module.runtime_bundle_paths(files, "infra/"+WRAPPER.name))
    assert selected >= {"infra/"+n for n in (*CHILDREN,*PRODUCERS,"run_episode_initializers.sh","cari_wrapper_common.sh")}
    assert not {"infra/object_pose_smoke.py","infra/run_object_pose_smoke.sh","infra/cari_prepare.py","infra/cari_forward.py"}&selected


@pytest.fixture
def queued_runtime(runtime):
    env,_,log,wrapper=runtime;original=Path(env['WR_CODE'])
    queued_parent=original.parent.with_name('run_track1_frontends_queued');original.parent.rename(queued_parent)
    code=queued_parent/'code';env['WR_CODE']=str(code)
    wrapper=code/'infra'/WRAPPER.name
    wrapper.parent.chmod(0o755)
    scheduling=wrapper.with_name('run_track1_frontends_queued.sh')
    scheduling.write_text('# Procedural source marker, never executed.\n');scheduling.chmod(0o444)
    wrapper.parent.chmod(0o555)
    def run(*args):return subprocess.run(['bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=10)
    return env,run,log,scheduling


def test_exact_readonly_queued_namespace_runs_same_initializers_and_full_lock(queued_runtime):
    env,run,log,_=queued_runtime;result=run('--episode','23')
    assert result.returncode==0,result.stderr
    children=[row for row in log.read_text().splitlines()if row.startswith('child|')]
    assert len(children)==8 and all('--episode 23'in row and row.endswith('|fd9-held')for row in children)
    assert children[0]=='child|automatic_masks|--episode 23 --seed-frames 16 --actor-seed-observations 16|fd9-held'
    base=Path(env['WR_ROOT'])/'outputs/episode_000023'
    assert all((base/name/'report.json').is_file()for name in TARGETS)
    assert not(base/'object_pose_full').exists()and not(base/'cari_inputs').exists()


@pytest.mark.parametrize('fault',['missing','writable','symlink','hardlink'])
def test_queued_namespace_needs_actual_regular_readonly_scheduling_source(queued_runtime,fault,tmp_path):
    _,run,log,path=queued_runtime
    if fault=='writable':path.chmod(0o644)
    elif fault=='hardlink':os.link(path,tmp_path/'alias')
    else:
        path.parent.chmod(0o755);path.unlink()
        if fault=='symlink':path.symlink_to(WRAPPER.name)
        path.parent.chmod(0o555)
    assert run('--episode','23').returncode!=0 and not log.exists()
