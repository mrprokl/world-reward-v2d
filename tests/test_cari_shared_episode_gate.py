"""CPU consumer receipt lifecycle + actual peer/bootstrap/archive interfaces."""
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"src"));monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("episode_gate_test",ROOT/"infra/cari_shared_episode_gate.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(gate,tmp_path,monkeypatch,n=501,episode=15):
    # Callback lifecycle tests emulate the fresh production process, rather
    # than inheriting earlier tests' optional imports in the pytest worker.
    # The real production guard stays strict; actual imports are separately
    # verified in a fresh subprocess below. Monkeypatch restores worker state.
    for name in ("torch","joblib"):
        monkeypatch.delitem(sys.modules,name,raising=False)
    code=tmp_path/"code";root=tmp_path/"root";code.mkdir();root.mkdir()
    for name in ("infra/cari_shared_episode_gate.py","infra/run_cari_shared_episode_gate.sh","infra/cari_shared_episode_loader.py"):
        path=code/name;path.parent.mkdir(exist_ok=True);path.write_bytes((ROOT/name).read_bytes());path.chmod(0o444)
    spec=dict(episode_index=episode,total_frames=n,camera_name="front_stereo_camera_left",height=1152,width=1536)
    files={name:dict(sha256="a"*64,bytes=1) for name in ("report.json","trajectory.npz","native_parameters.npz","target.npy","object_aligned.glb")}
    pins=dict(schema="world-reward-cari-shared-export-pins-v1",clip_spec=spec,export_files=files,
        export=files["report.json"]|dict(producer_revision="b"*40,script_sha256="c"*64))
    path=code/f"configs/cari_clip_{episode:06d}_shared_export_pins.json";path.parent.mkdir();path.write_text(json.dumps(pins));path.chmod(0o444)
    out=root/gate.output_relative(episode);out.mkdir(parents=True)
    calls=[]
    def consumer(r,c,s,p):
        calls.append((r,c,s,p))
        manifest=dict(stage="world_reward_shared_native_episode_consumer",episode_index=episode,frames=n,clip_spec=asdict(s),
            input_track="track_1",ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],original_frame_coverage_verified=True,
            integrity_and_schema_verified=True,native_direct_export_consumed=True,old_LM_conversion_used=False,
            numerical_geometry_independently_reverified=False,quality_verified=False,challenge_performance_verified=False,
            submission_eligibility_verified=False,submission_eligible=False,final_Parquet_produced=False,
            export_report_sha256=pins["export"]["sha256"],export_files=files)
        return SimpleNamespace(episode=SimpleNamespace(total_video_frames=n,validate=lambda:None),manifest=manifest)
    return root,code,out,path,pins,consumer,calls


@pytest.mark.parametrize("n,ep",[(96,0),(97,29),(501,15)])
def test_exact_cpu_gate_one_frozen_report_no_prediction_copy(gate,tmp_path,monkeypatch,n,ep):
    root,code,out,path,pins,consumer,calls=fixture(gate,tmp_path,monkeypatch,n,ep)
    report=gate.execute(root,out,code,ep,"d"*40,consumer=consumer)
    assert report["status"]=="pass" and report["stage"]==gate.STAGE and len(calls)==1
    assert set(p.name for p in out.iterdir())=={"report.json"} and not (out/"report.json").stat().st_mode&0o222
    assert json.loads((out/"report.json").read_text())==report
    assert report["consumer_manifest"]["frames"]==n and report["export_pins"]==gate.identity(path)
    assert report["GPU_used"] is False and report["model_calls"]==0 and report["Parquet_calls"]==0
    assert report["numerical_geometry_independently_reverified"] is False and report["quality_verified"] is False


@pytest.mark.parametrize("fault",["callback","quality","frame","schema","pin","helper","extraoutput","coverage"])
def test_fail_receipt_frozen_no_false_pass(gate,tmp_path,monkeypatch,fault):
    root,code,out,path,pins,consumer,calls=fixture(gate,tmp_path,monkeypatch)
    def bad(*args):
        if fault=="callback":raise RuntimeError("synthetic consumer failure")
        value=consumer(*args)
        if fault=="quality":value.manifest["quality_verified"]=True
        elif fault=="frame":value.episode.total_video_frames=500
        elif fault=="schema":value.manifest["stage"]="old_LM"
        elif fault=="pin":path.chmod(0o644);path.write_bytes(b"changed");path.chmod(0o444)
        elif fault=="helper":
            source=code/"infra/cari_shared_episode_loader.py";source.chmod(0o644);source.write_bytes(b"changed");source.chmod(0o444)
        elif fault=="extraoutput":(out/"trajectory.npz").write_bytes(b"unexpected copied prediction")
        else:value.manifest["export_files"]={}
        return value
    with pytest.raises((ValueError,RuntimeError)):gate.execute(root,out,code,15,"d"*40,consumer=bad)
    report=json.loads((out/"report.json").read_text())
    assert report["status"]=="fail" and not (out/"report.json").stat().st_mode&0o222
    assert report["quality_verified"] is False and report["final_Parquet_produced"] is False


def test_no_overwrite_existing_engineering_receipt(gate,tmp_path,monkeypatch):
    root,code,out,*_=fixture(gate,tmp_path,monkeypatch);(out/"report.json").write_bytes(b"frozen previous")
    with pytest.raises(ValueError):gate.execute(root,out,code,15,"d"*40)
    assert (out/"report.json").read_bytes()==b"frozen previous"


def test_callback_fixture_isolates_prior_worker_optional_imports(gate,tmp_path,monkeypatch):
    for name in ("torch","joblib"):
        monkeypatch.setitem(sys.modules,name,SimpleNamespace(prior_test_import=True))
    root,code,out,_,_,consumer,_=fixture(gate,tmp_path,monkeypatch)
    assert "torch" not in sys.modules and "joblib" not in sys.modules
    assert gate.execute(root,out,code,15,"d"*40,consumer=consumer)["status"]=="pass"


@pytest.mark.parametrize("args",[[],["--episode","30"],["--episode","-1"],["--ep","15"],["--episode","15","--episode","16"],["--episode","15","--root","/tmp"]])
def test_explicit_generic_episode_only(gate,args):
    with pytest.raises(SystemExit):gate.parser().parse_args(args)


@pytest.mark.parametrize("ep",[0,15,29])
def test_episode_parser_no_implicit_selection(gate,ep):
    assert gate.parser().parse_args(["--episode",str(ep)]).episode==ep
    assert gate.output_relative(ep)==f"outputs/episode_{ep:06d}/cari_shared_episode_v1"


def test_fresh_actual_loader_api_and_gate_import_do_not_load_torch_joblib():
    source=f"""
import inspect,sys
sys.path[:0]=[{str(ROOT/'src')!r},{str(ROOT/'infra')!r}]
import cari_shared_episode_gate as gate
assert 'numpy' not in sys.modules
import cari_shared_episode_loader as consumer
assert str(inspect.signature(consumer.load_shared_track1_episode))=='(root, code, spec, export_pins)'
assert consumer.export.__file__.endswith('/infra/cari_full_export.py')
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
print('actual consumer CPU API PASS')
"""
    result=subprocess.run([sys.executable,"-c",source],check=True,capture_output=True,text=True)
    assert result.stdout.strip()=="actual consumer CPU API PASS"


def test_actual_wrapper_no_gpu_model_exec_or_broad_data_mount():
    wrapper=ROOT/"infra/run_cari_shared_episode_gate.sh";subprocess.run(["bash","-n",str(wrapper)],check=True)
    text=wrapper.read_text()
    assert "--gpus" not in text and "--network none --memory 4g --cpus 2" in text and "303s docker run" in text
    assert "source_paths(spec,object_source=source_profile(pins))" in text and "prepare forward refined export" in text
    assert "src=$ROOT/data" not in text and "src=$ROOT/outputs,dst=$ROOT/outputs" not in text
    assert "--mount \"type=bind,src=$OUT,dst=$OUT\"" in text
    assert "sam3d_body/checkpoints/sam-3d-body-dinov3" in text and "weights/mhr/mhr_model.pt" in text
    assert "cari-refinement-assets.json" in text and "weights-acquisition.json" in text


def test_host_bootstrap_stdlib_only_no_bare_host_numpy(tmp_path):
    text=(ROOT/"infra/run_cari_shared_episode_gate.sh").read_text()
    source=text.split("<<'PYPATHS'\n",1)[1].split("\nPYPATHS",1)[0]
    program=f"""
import builtins,sys
sys.path[:0]=[{str(ROOT/'src')!r},{str(ROOT/'infra')!r}]
original=builtins.__import__
def guarded(name,*args,**kwargs):
 if name.split('.')[0] in {{'numpy','torch','joblib','h5py','pyarrow','trimesh'}}:raise AssertionError(name)
 return original(name,*args,**kwargs)
builtins.__import__=guarded
sys.argv=['bootstrap',{str(ROOT/'configs/cari_clip_000015_input_pins.json')!r},'15']
exec({source!r},{{}})
"""
    result=subprocess.run([sys.executable,"-c",program],check=True,capture_output=True,text=True)
    assert len(result.stdout.splitlines())==15


def test_complete_actual_archive_closure_under_control_cap():
    s=importlib.util.spec_from_file_location("consumer_bundle",ROOT/"infra/azure_job.py")
    launcher=importlib.util.module_from_spec(s);s.loader.exec_module(launcher)
    files={str(p.relative_to(ROOT)):p.read_bytes() for base in ("infra","src","configs") for p in (ROOT/base).rglob("*") if p.is_file() and "__pycache__" not in p.parts}
    files["pyproject.toml"]=(ROOT/"pyproject.toml").read_bytes()
    selected=launcher.runtime_bundle_paths(files,"infra/run_cari_shared_episode_gate.sh")
    assert set(selected)>={"infra/cari_shared_episode_gate.py","infra/run_cari_shared_episode_gate.sh","infra/cari_shared_episode_loader.py",
        "infra/cari_full_forward.py","infra/cari_full_refine.py","infra/cari_full_export.py","infra/run_cari96_prepare.sh"}
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode="w") as archive:
        for name in selected:
            entry=tarfile.TarInfo(name);entry.size=len(files[name]);entry.mode=0o444;archive.addfile(entry,io.BytesIO(files[name]))
    encoded,_=launcher.encoded_runtime_archive(stream.getvalue())
    assert len(encoded)<=launcher.MAX_CODE_CONTROL_BYTES
