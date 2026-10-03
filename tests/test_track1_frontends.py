"""Bounded clean frontend wrapper with tiny fake commands, no GPU/cloud."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/"infra/run_track1_frontends.sh"
TARGETS=("automatic_masks","body_smoke","depth_smoke","scale_smoke","object_grounded","body_full","depth_full",
         "body_full/cari_adapter","object_pose_full","cari_inputs")
WRAPPERS=("run_automatic_masks.sh","run_body_smoke.sh","run_depth_smoke.sh","run_scale_smoke.sh","run_object_smoke.sh",
          "run_cari_body_adapter_smoke.sh","run_object_pose_smoke.sh","run_cari_prepare.sh")
PRODUCERS=("automatic_masks.py","body_smoke.py","depth_smoke.py","scale_smoke.py","object_smoke.py",
           "cari_body_adapter_smoke.py","object_pose_smoke.py","cari_prepare.py")


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/"runtime";revision="a"*40
    code=root/"jobs"/revision/"run_track1_frontends/code";(code/"infra").mkdir(parents=True)
    (root/"results").mkdir();(root/"results/camera-render.json").write_text('{"status":"pass"}')
    (code.parent/"revision").write_text(revision+"\n");(code.parent/"source-sha256").write_text("b"*64+"\n")
    wrapper=code/"infra/run_track1_frontends.sh"
    # Fixture-only literal replacement preserves strict launcher layout, not a
    # production root override/CLI escape hatch.
    wrapper.write_text(WRAPPER.read_text().replace("/srv/scenesmith/world-reward",str(root)))
    for name in ("run_episode_initializers.sh","cari_wrapper_common.sh"):
        (code/"infra"/name).write_bytes((ROOT/"infra"/name).read_bytes())
    for name in PRODUCERS:(code/"infra"/name).write_bytes(b"tiny immutable source")
    child='''#!/usr/bin/env bash
set -euo pipefail
name="$(basename "$0")"
episode="$2";printf -v padded '%06d' "$episode"
case "$name" in
 run_automatic_masks.sh) stage=automatic_masks ;;
 run_body_smoke.sh) if [[ " $* " == *" --full-video "* ]];then stage=body_full;else stage=body_smoke;fi ;;
 run_depth_smoke.sh) if [[ " $* " == *" --full-video "* ]];then stage=depth_full;else stage=depth_smoke;fi ;;
 run_scale_smoke.sh) stage=scale_smoke ;;
 run_object_smoke.sh) stage=object_grounded ;;
 run_cari_body_adapter_smoke.sh) stage=cari_adapter ;;
 run_object_pose_smoke.sh) stage=object_pose_full ;;
 run_cari_prepare.sh) stage=cari_inputs ;;
 *) exit 9 ;;
esac
printf 'child|%s|%s\\n' "$stage" "$*" >> "$FAKE_LOG"
if [[ "$stage" == "${FAKE_FAIL_STAGE:-}" ]];then exit 7;fi
if [[ "$stage" == cari_inputs ]];then
 [[ "$*" == "--episode $episode --no-wait" ]] || exit 9
 source "$WR_CODE/infra/cari_wrapper_common.sh";ROOT="$WR_ROOT"
 wr_parse_cari_arguments prepare "$@";[[ -z "$WR_WAIT_FOR" ]] || exit 9;wr_cari_dependency prepare
fi
target="$stage";[[ "$stage" != cari_adapter ]] || target=body_full/cari_adapter
mkdir -p "$WR_ROOT/outputs/episode_$padded/$target"
printf '{}' > "$WR_ROOT/outputs/episode_$padded/$target/report.json"
'''
    for name in WRAPPERS:(code/"infra"/name).write_text(child)
    for path in code.rglob("*"):
        if path.is_file():path.chmod(0o444)
        elif path.is_dir():path.chmod(0o555)
    code.chmod(0o555)
    bin=tmp_path/"bin";bin.mkdir();log=tmp_path/"calls.log"
    scripts={
        "nvidia-smi":'''#!/bin/sh
printf 'smi|%s\\n' "$*" >> "$FAKE_LOG"
if [ "${FAKE_SMI_FAIL:-0}" = 1 ];then exit 8;fi
[ "${FAKE_GPU_BUSY:-0}" != 1 ] || printf '1234\\n'
''',
        "flock":'''#!/bin/sh
printf 'lock|%s\\n' "$*" >> "$FAKE_LOG"
if [ "$1" = --nonblock ] && [ "${FAKE_LOCK_BUSY:-0}" = 1 ];then exit 1;fi
''',
        "timeout":'''#!/bin/sh
printf 'timeout|%s|%s|%s|%s\\n' "$1" "$2" "$3" "$5" >> "$FAKE_LOG"
shift 3
exec "$@"
'''}
    for name,source in scripts.items():p=bin/name;p.write_text(source);p.chmod(0o755)
    env=dict(WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,FAKE_LOG=str(log),PATH=str(bin)+":"+os.environ["PATH"])
    def run(*args):return subprocess.run(["bash",str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return env,run,log,wrapper


def events(result):return [json.loads(line) for line in result.stdout.splitlines()]


@pytest.mark.parametrize("ep",[0,15,29])
def test_exact_frontend_only_order_seven_real_initializer_children(runtime,ep):
    env,run,log,_=runtime;result=run("--episode",str(ep));assert result.returncode==0,result.stderr
    rows=log.read_text().splitlines();children=[row for row in rows if row.startswith("child|")]
    expected=[f"child|automatic_masks|--episode {ep}",f"child|body_smoke|--episode {ep} --inference-type body",
        f"child|depth_smoke|--episode {ep}",f"child|scale_smoke|--episode {ep}",f"child|object_grounded|--episode {ep} --aligned-pointmap",
        f"child|body_full|--episode {ep} --full-video --inference-type body",f"child|depth_full|--episode {ep} --full-video",
        f"child|cari_adapter|--episode {ep}",f"child|object_pose_full|--episode {ep} --full-video",f"child|cari_inputs|--episode {ep} --no-wait"]
    assert children==expected
    assert [row.split("|")[3] for row in rows if row.startswith("timeout|")]==["600s","5400s","7200s","7200s"]
    assert sum(row.startswith("smi|") for row in rows)==3
    assert rows.index("lock|--nonblock 9")<next(i for i,v in enumerate(rows) if v.startswith("child|"))
    assert rows.index("lock|--unlock 9")<next(i for i,v in enumerate(rows) if v.startswith("child|cari_inputs"))
    own=[e for e in events(result) if "mode" in e]
    assert [(e["stage"],e["phase"]) for e in own]==[(s,p) for s in ("preflight","gpu_preflight","automatic_masks","episode_initializers","object_pose_full","cari_inputs") for p in ("start","pass")]
    assert all(set(e)=={"mode","stage","phase","timestamp_utc"} and e["mode"]=="public_frontends_only" for e in own)


@pytest.mark.parametrize("args",[[],["--episode"],["--episode","30"],["--episode","-1"],["--episode","00"],["--episode","15","--episode","16"],
    ["--episode=15"],["--episode","15","--wait-for","x"],["--episode","15","--root","/tmp"],["--episode","15","--resume"],
    ["--episode","15","--no-wait"],["--episode","15","--overwrite"],["--episode","true"]])
def test_explicitonly_controls_before_any_gpu_or_mutation(runtime,args):
    env,run,log,_=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists() and not (Path(env["WR_ROOT"])/"outputs").exists()


@pytest.mark.parametrize("target",TARGETS)
@pytest.mark.parametrize("kind",["directory","file","broken_symlink"])
def test_all_frontend_absence_checks_before_lock_gpu_children(runtime,target,kind):
    env,run,log,_=runtime;path=Path(env["WR_ROOT"])/"outputs/episode_000015"/target;path.parent.mkdir(parents=True,exist_ok=True)
    if kind=="directory":path.mkdir()
    elif kind=="file":path.write_text("frozen")
    else:path.symlink_to("missing")
    result=run("--episode","15")
    assert result.returncode!=0 and not log.exists() and "Frozen frontend target" in result.stderr


@pytest.mark.parametrize("stage",["automatic_masks","body_smoke","depth_smoke","scale_smoke","object_grounded","body_full","depth_full","cari_adapter","object_pose_full","cari_inputs"])
def test_failure_abort_preserves_child_status_no_resume(runtime,stage):
    env,run,log,_=runtime;env["FAKE_FAIL_STAGE"]=stage;result=run("--episode","0")
    assert result.returncode==7
    calls=[row.split("|")[1] for row in log.read_text().splitlines() if row.startswith("child|")]
    assert calls[-1]==stage and len(calls)==len(set(calls))
    assert events(result)[-1]["phase"]=="fail"


@pytest.mark.parametrize("mode",["GPU_BUSY","SMI_FAIL","LOCK_BUSY"])
def test_gpu_external_jobs_or_competing_frontend_failfast(runtime,mode):
    env,run,log,_=runtime;env["FAKE_"+mode]="1";result=run("--episode","15")
    assert result.returncode!=0 and not any(row.startswith("child|") for row in log.read_text().splitlines())
    assert events(result)[-1]["stage"]=="gpu_preflight"


@pytest.mark.parametrize("fault",["childmissing","childwritable","childsymlink","gate","revisionmarker","archivemarker","WR_ROOT","WR_CODE","WR_CODE_REVISION","entrypoint"])
def test_immutable_launcher_preflight(runtime,fault,tmp_path):
    env,run,log,wrapper=runtime;code=Path(env["WR_CODE"])
    path=code/"infra/run_object_pose_smoke.sh"
    if fault=="childmissing":path.parent.chmod(0o755);path.unlink()
    elif fault=="childwritable":path.chmod(0o644)
    elif fault=="childsymlink":path.parent.chmod(0o755);path.unlink();path.symlink_to("run_automatic_masks.sh")
    elif fault=="gate":(Path(env["WR_ROOT"])/"results/camera-render.json").unlink()
    elif fault=="revisionmarker":(code.parent/"revision").write_text("main")
    elif fault=="archivemarker":(code.parent/"source-sha256").write_text("bad")
    elif fault=="WR_ROOT":env["WR_ROOT"]="/tmp/notcanonical"
    elif fault=="WR_CODE":env["WR_CODE"]=str(tmp_path/"elsewhere")
    elif fault=="WR_CODE_REVISION":env["WR_CODE_REVISION"]="A"*40
    else:
        outside=tmp_path/"outside.sh";outside.write_text(wrapper.read_text())
        result=subprocess.run(["bash",str(outside),"--episode","15"],env=env,capture_output=True,text=True)
        assert result.returncode!=0 and not log.exists();return
    result=run("--episode","15");assert result.returncode!=0 and not log.exists()


def test_actual_bash_syntax_scope_no_historical_forward_converter_claims():
    subprocess.run(["bash","-n",str(WRAPPER)],check=True)
    text=WRAPPER.read_text()
    assert "run_cari_forward.sh" not in text and "run_cari_converter.sh" not in text and "systemctl" not in text
    assert "--no-wait" in text and "flock --nonblock 9" in text and "--query-compute-apps=pid" in text
    assert "legacy/full-refine" in text and "Resource/mount hardening" in text and "broad existing mounts" in text
    assert "--kill-after=10s" in text and "eval " not in text and "&\n" not in text


def test_actual_archive_closure_all_children_no_oldprediction_route():
    spec=importlib.util.spec_from_file_location("frontend_archive",ROOT/"infra/azure_job.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes() for base in ("infra","src","configs") for p in (ROOT/base).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    selected=set(module.runtime_bundle_paths(files,"infra/run_track1_frontends.sh"))
    assert selected>={"infra/"+name for name in (*WRAPPERS,*PRODUCERS,"run_episode_initializers.sh","cari_wrapper_common.sh")}
    assert "infra/run_cari_forward.sh" not in selected and "infra/run_cari_converter.sh" not in selected
    assert "infra/cari_forward.py" not in selected and "infra/cari_converter.py" not in selected


@pytest.mark.parametrize("ep", [0, 4, 15, 29])
def test_explicit_fixed_all16_policy_changes_only_real_observation_count(runtime, ep):
    env, run, log, _ = runtime
    result = run("--episode", str(ep), "--actor-policy", "fixed_all16")
    assert result.returncode == 0, result.stderr
    children = [row for row in log.read_text().splitlines() if row.startswith("child|")]
    assert children[0] == f"child|automatic_masks|--episode {ep} --seed-frames 16 --actor-seed-observations 16"
    assert len(children) == 10
    assert all("--actor-policy" not in row and "--actor-seed-observations" not in row for row in children[1:])


@pytest.mark.parametrize("args", [
    ["--episode", "4", "--actor-policy", "adaptive"],
    ["--episode", "4", "--actor-seed-observations", "2"],
    ["--episode", "4", "--actor-policy", "16"],
    ["--episode", "4", "--actor-policy"],
    ["--episode", "4", "--actor-policy", "fixed_all16", "--confidence", "0.1"],
])
def test_no_arbitrary_or_episode_specific_actor_policy_controls(runtime, args):
    env, run, log, _ = runtime
    result = run(*args)
    assert result.returncode == 2 and not log.exists()
