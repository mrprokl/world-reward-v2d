"""Tiny full-native consumer fixtures; real peer API, no model/data/runtime."""
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/"src"));monkeypatch.syspath_prepend(str(ROOT/"infra"))
    spec=importlib.util.spec_from_file_location("shared_episode_consumer_test",ROOT/"infra/cari_shared_episode_loader.py")
    module=importlib.util.module_from_spec(spec);monkeypatch.setitem(sys.modules,spec.name,module)
    spec.loader.exec_module(module);return module


def fixture(gate,tmp_path,monkeypatch,n=97,episode=15):
    root=tmp_path/"root";code=tmp_path/"code";root.mkdir();code.mkdir()
    spec=gate.public.PublicClipSpec(episode,n,"front_stereo_camera_left",1152,1536)
    directory=root/gate.export.output_relative(episode);directory.mkdir(parents=True)
    data=dict(pose=np.zeros((n,136),np.float32),scales=np.full(68,.03,np.float32),shape=np.full(45,.04,np.float32),
        expression=np.zeros(72,np.float32),object_rotation=np.tile(np.eye(3,dtype=np.float32),(n,1,1)),
        object_translation=np.tile(np.array([0,0,3],np.float32),(n,1)),object_scale=np.array(1,np.float32),
        object_vertices=np.array([[0,0,0],[.1,0,0],[0,.1,0]],np.float32),object_faces=np.array([[0,1,2]],np.int64),
        camera_K=gate.public.inferred_camera(spec),frame_index=np.arange(n,dtype=np.int64))
    data["pose"][:,2]=np.linspace(1,2,n,dtype=np.float32)
    with (directory/"trajectory.npz").open("xb") as f:np.savez(f,**data)
    (directory/"native_parameters.npz").write_bytes(b"opaque native archive: MUST NOT DESERIALIZE")
    (directory/"target.npy").write_bytes(b"opaque heavy target: MUST NOT DESERIALIZE")
    (directory/"object_aligned.glb").write_bytes(b"opaque original metric GLB")
    paths=gate.public.relative_paths(spec)
    for name in gate.public.source_paths(spec):
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(b"tiny public dependency")
    (root/paths["mesh"]).write_bytes((directory/"object_aligned.glb").read_bytes())
    wild=dict(schema="cari4d.mhr_wild_export.v2",sequence=spec.sequence,frame_count=n,camera_id=0,height=1152,width=1536,
        object_mesh_file=str(root/paths["mesh"]),depth_backend="moge2",object_pose_frame="centered_axis_aligned",
        object_pose_frame_revision="cari4d.object_pose_frame.centered_axis_aligned.v1",object_pose_storage_frame="output_aligned_mesh_frame",
        object_pose_storage_to_training_transform=np.eye(4).tolist(),object_mesh_to_training_transform=np.eye(4).tolist(),
        intrinsics=gate.public.inferred_camera(spec).tolist())
    K=gate.public.inferred_camera(spec)
    edex=[dict(cameras=[dict(name=spec.camera_name,intrinsics=dict(focal=[K[0,0],K[1,1]],principal=[K[0,2],K[1,2]]),
        transform=np.eye(4)[:3].tolist())])]
    (root/paths["wild_export"]).write_text(json.dumps(wild));(root/paths["export_seq"]/"edex").write_text(json.dumps(edex))
    source_files={name:gate.public.identity(root/name) for name in gate.public.source_paths(spec)}
    input_pins=dict(schema="world-reward-cari-clip-input-pins-v1",clip_spec=asdict(spec),source_files=source_files,
        input_report=source_files[paths["input_report"]]|dict(producer_revision="b"*40,script_sha256="c"*64))
    input_path=code/f"configs/cari_clip_{episode:06d}_input_pins.json";input_path.parent.mkdir();input_path.write_text(json.dumps(input_pins));input_path.chmod(0o444)
    refined_dir=root/f"outputs/episode_{episode:06d}/cari_shared_refined_v1";refined_dir.mkdir()
    (refined_dir/"refined.pth").write_bytes(b"opaque native refined bundle");(refined_dir/"report.json").write_bytes(b"tiny passing gate fixture")
    for path in refined_dir.iterdir():path.chmod(0o444)
    refined_files={path.name:gate.public.identity(path) for path in refined_dir.iterdir()}
    refined_pins=dict(schema="world-reward-cari-shared-refined-pins-v1",clip_spec=asdict(spec),refined_files=refined_files,
        refined=refined_files["report.json"]|dict(producer_revision="d"*40,script_sha256="e"*64))
    refined_path=code/f"configs/cari_clip_{episode:06d}_shared_refined_pins.json";refined_path.write_text(json.dumps(refined_pins));refined_path.chmod(0o444)
    helper_names=tuple(gate.export.source_helpers.__code__.co_consts)
    names=next(v for v in helper_names if type(v) is tuple and "infra/cari_full_export.py" in v)
    for name in names:
        p=code/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/name).read_bytes());p.chmod(0o444)
    prepared=dict(input_sha256="9"*64,body_assets={"body":"synthetic gate"},inference_source_identity={"source":"synthetic gate"},decoder_identity={"decoder":"synthetic gate"})
    refined=dict(input_sha256=prepared["input_sha256"],object_mesh_sha256=source_files[paths["mesh"]]["sha256"])
    chain=dict(report=refined,files=refined_files,prepare=dict(report=prepared,source_files=source_files),
        bindings={root/name:row for name,row in source_files.items()}|{refined_dir/name:row for name,row in refined_files.items()})
    # Mock only an already audited hash/JSON gate, never a peer module or Torch.
    monkeypatch.setattr(gate.export.lineage,"verify_refined_artifacts",lambda *_:chain)
    chunks=gate.export.finite_chunks(n,16)
    report=dict(stage=gate.export.STAGE,status="pass",phase="complete",episode_index=episode,clip_spec=asdict(spec),image_id=gate.export.IMAGE,
        frames=n,original_frame_indices=list(range(n)),chunk_counts=[s.stop-s.start for s in chunks],input_track="track_1",
        ground_truth_used=False,ground_truth_read=False,private_truth_read=False,hand_labeled_test=False,oracle_modes=[],network="none",
        budget_seconds=gate.export.BUDGET,learned_inference_calls=0,optimizer_calls=0,converter_LM_calls=0,identity_reselection_performed=False,
        quality_verified=False,adoption_authorized=False,submission_produced=False,submission_eligible=False,final_Parquet_produced=False,
        predictions_frozen_before_replays=True,unchanged_refined_predictions_verified=True,stored_native_replay_verified=True,
        native_Track1Episode_schema_verified=True,original_frame_coverage_verified=True,frozen_outputs_rehashed_after_reference=True,
        source_inputs_assets_rehashed=True,source_helpers_rehashed=True,raw_masks_contacts_object_pose_unchanged=True,
        full_original_native_export_verified=True,reference_per_frame_mean_mm=[.001]*n,native_direct_max_point_mm=.001,native_replay_max_point_mm=.001,
        refined_report_sha256=refined_files["report.json"]["sha256"],refined_bundle_sha256=refined_files["refined.pth"]["sha256"],
        aligned_object_mesh_sha256=source_files[paths["mesh"]]["sha256"],input_pins=gate.export.lineage.identity(input_path),
        refined_pins=gate.export.lineage.identity(refined_path),source_files=source_files,source_helpers=gate.export.source_helpers(code),
        **{k:prepared[k] for k in ("body_assets","inference_source_identity","decoder_identity")},producer_revision="f"*40,
        script_sha256=gate.public.identity(code/"infra/cari_full_export.py")["sha256"])
    poses=np.tile(np.eye(4,dtype=np.float32),(n,1,1));poses[:,:3,3]=data["object_translation"]
    report["object_roundtrip"]=gate.export.object_roundtrip(data["object_vertices"],data["object_faces"],poses,n)
    report.update({name+suffix:len(chunks) for name in gate.export.CALLS for suffix in ("_attempts","_returns","_validated")})
    def save():
        for p in directory.iterdir():p.chmod(0o444)
        report["output_files"]={name:gate.public.identity(directory/name) for name in gate.export.OUTPUTS}
        path=directory/"report.json"
        if path.exists():path.chmod(0o644)
        path.write_text(json.dumps(report));path.chmod(0o444)
        files={p.name:gate.public.identity(p) for p in directory.iterdir()}
        return dict(schema="world-reward-cari-shared-export-pins-v1",clip_spec=asdict(spec),export_files=files,
            export=files["report.json"]|{k:report[k] for k in ("producer_revision","script_sha256")})
    pins=save()
    return root,code,spec,pins,report,data,chain,save


@pytest.mark.parametrize("n,ep",[(96,0),(97,29),(501,15)])
def test_frozen_full_trajectory_consumer_no_alias_or_heavy_reads(gate,tmp_path,monkeypatch,n,ep):
    root,code,spec,pins,report,data,_,_=fixture(gate,tmp_path,monkeypatch,n,ep)
    real_load=gate.np.load;opened=[]
    def load(path,*a,**kw):
        opened.append(Path(path).name)
        assert Path(path).name=="trajectory.npz" and kw==dict(allow_pickle=False)
        return real_load(path,*a,**kw)
    monkeypatch.setattr(gate.np,"load",load)
    loaded=gate.load_shared_track1_episode(root,code,spec,pins)
    assert opened==["trajectory.npz"] and loaded.episode.total_video_frames==n
    assert loaded.episode.reconstruction.pose.tobytes()==data["pose"].tobytes()
    assert not np.shares_memory(loaded.episode.reconstruction.pose,data["pose"])
    assert loaded.manifest["integrity_and_schema_verified"] is True and loaded.manifest["quality_verified"] is False
    assert loaded.manifest["numerical_geometry_independently_reverified"] is False
    assert loaded.episode.provenance["input_sha256"]=="9"*64
    json.dumps(loaded.manifest,allow_nan=False)


@pytest.mark.parametrize("fault",["schema","missing","extra","clipbool","hash","boolbytes","revision","reportlink"])
def test_explicit_exact_pins_failclosed(gate,tmp_path,monkeypatch,fault):
    _,_,spec,pins,*_=fixture(gate,tmp_path,monkeypatch)
    if fault=="schema":pins["schema"]="old_LM"
    elif fault=="missing":pins["export_files"].pop("target.npy")
    elif fault=="extra":pins["export_files"]["GT.npy"]=dict(sha256="a"*64,bytes=1)
    elif fault=="clipbool":pins["clip_spec"]["episode_index"]=True
    elif fault=="hash":pins["export_files"]["target.npy"]["sha256"]="bad"
    elif fault=="boolbytes":pins["export_files"]["target.npy"]["bytes"]=True
    elif fault=="revision":pins["export"]["producer_revision"]="a"*39
    else:pins["export"]["sha256"]="a"*64
    with pytest.raises(ValueError):gate.validate_export_pins(spec,pins)


@pytest.mark.parametrize("fault",["missing","extra","writable","symlink","tamper"])
def test_allfive_hashbefore_json_or_numpy(gate,tmp_path,monkeypatch,fault):
    root,code,spec,pins,*_=fixture(gate,tmp_path,monkeypatch)
    directory=root/gate.export.output_relative(15);path=directory/"target.npy"
    if fault=="missing":path.unlink()
    elif fault=="extra":(directory/"old_predictions.npy").write_bytes(b"noise")
    elif fault=="writable":path.chmod(0o644)
    elif fault=="symlink":path.unlink();path.symlink_to(directory/"native_parameters.npz")
    else:path.chmod(0o644);path.write_bytes(b"tamper");path.chmod(0o444)
    monkeypatch.setattr(gate,"_json",lambda *_:pytest.fail("No JSON before five SHA gates"))
    with pytest.raises((ValueError,FileNotFoundError)):gate.load_shared_track1_episode(root,code,spec,pins)


@pytest.mark.parametrize("fault",["GT","quality","producer","helpers","inputpin","refinedpin","refinedbundle","body","decoder","source","mesh"])
def test_actual_producer_predecessor_chain_no_silent_relabel(gate,tmp_path,monkeypatch,fault):
    root,code,spec,_,report,_,_,save=fixture(gate,tmp_path,monkeypatch)
    if fault=="GT":report["ground_truth_used"]=True
    elif fault=="quality":report["quality_verified"]=True
    elif fault=="producer":report["script_sha256"]="a"*64
    elif fault=="helpers":report["source_helpers"]={}
    elif fault=="inputpin":report["input_pins"]["sha256"]="a"*64
    elif fault=="refinedpin":report["refined_pins"]["sha256"]="a"*64
    elif fault=="refinedbundle":report["refined_bundle_sha256"]="a"*64
    elif fault=="body":report["body_assets"]={}
    elif fault=="decoder":report["decoder_identity"]={}
    elif fault=="source":report["source_files"]={}
    else:report["aligned_object_mesh_sha256"]="a"*64
    pins=save()
    with pytest.raises(ValueError):gate.load_shared_track1_episode(root,code,spec,pins)


@pytest.mark.parametrize("fault",["dtype","missingframe","repeat","reorder","perframescales","expression","scale","K","verts","faces","negativeZ","objectdtype","extrakey","duplicate"])
def test_trajectory_exact_native_no_fit_no_repair(gate,tmp_path,monkeypatch,fault):
    root,code,spec,_,_,data,_,save=fixture(gate,tmp_path,monkeypatch)
    if fault=="dtype":data["pose"]=data["pose"].astype(np.float64)
    elif fault=="missingframe":data["pose"]=data["pose"][:-1]
    elif fault=="repeat":data["frame_index"][-1]=data["frame_index"][-2]
    elif fault=="reorder":data["frame_index"]=data["frame_index"][::-1]
    elif fault=="perframescales":data["scales"]=np.tile(data["scales"],(97,1))
    elif fault=="expression":data["expression"][0]=1e-10
    elif fault=="scale":data["object_scale"]=np.array(.9,np.float32)
    elif fault=="K":data["camera_K"][0,0]+=1
    elif fault=="verts":data["object_vertices"]=np.zeros((3,3),np.float32)
    elif fault=="faces":data["object_faces"]=data["object_faces"].astype(np.int32)
    elif fault=="negativeZ":data["object_translation"][:,2]=-1
    elif fault=="objectdtype":data["pose"]=data["pose"].astype(object)
    elif fault=="extrakey":data["cached_geometry"]=np.zeros(1,np.float32)
    path=root/gate.export.output_relative(15)/"trajectory.npz";path.chmod(0o644)
    with path.open("wb") as stream:np.savez(stream,**data)
    if fault=="duplicate":
        with zipfile.ZipFile(path,"a") as stream:
            with pytest.warns(UserWarning):stream.writestr("pose.npy",stream.read("pose.npy"))
    pins=save()
    with pytest.raises(ValueError):gate.load_shared_track1_episode(root,code,spec,pins)


def test_duplicate_jsonkeys_and_returned_copy_manifest(gate,tmp_path,monkeypatch):
    root,code,spec,pins,*_=fixture(gate,tmp_path,monkeypatch)
    path=code/"configs/cari_clip_000015_input_pins.json";path.chmod(0o644)
    path.write_text('{"schema":1,"schema":2}');path.chmod(0o444)
    with pytest.raises(ValueError,match="Duplicate"):gate._json(path)


@pytest.mark.parametrize("fault",["sourcebyte","helperbyte","roundtrip","negativezero","rootlink"])
def test_source_and_geometry_receipt_not_just_selfconsistent_export(gate,tmp_path,monkeypatch,fault):
    root,code,spec,_,report,data,chain,save=fixture(gate,tmp_path,monkeypatch)
    if fault=="sourcebyte":
        path=root/gate.public.relative_paths(spec)["mesh"];path.write_bytes(b"different metric mesh")
    elif fault=="helperbyte":
        path=code/"infra/cari_full_export.py";path.chmod(0o644);path.write_bytes(b"different generator");path.chmod(0o444)
    elif fault=="roundtrip":report["object_roundtrip"]["aligned_local_frame_retained"]=1
    elif fault=="negativezero":report["object_roundtrip"]["vertices_per_frame"]=True
    else:
        alias=tmp_path/"linked";alias.symlink_to(root,target_is_directory=True);root=alias
    pins=save()
    with pytest.raises(ValueError):gate.load_shared_track1_episode(root,code,spec,pins)


def test_fresh_actual_peer_imports_signatures_no_torch_joblib():
    source=f"""
import inspect,sys
sys.path[:0]=[{str(ROOT/'src')!r},{str(ROOT/'infra')!r}]
import cari_shared_episode_loader as loader
assert str(inspect.signature(loader.export.validate_export_report))=='(report, spec)'
assert str(inspect.signature(loader.export.lineage.verify_refined_artifacts))=='(root, code, spec, pins, source_code=None)'
assert loader.export.__file__.endswith('/infra/cari_full_export.py')
assert loader.export.lineage.__file__.endswith('/infra/cari_full_refine.py')
assert 'torch' not in sys.modules and 'joblib' not in sys.modules
print('actual peer API PASS')
"""
    result=subprocess.run([sys.executable,"-c",source],check=True,capture_output=True,text=True)
    assert result.stdout.strip()=="actual peer API PASS"

@pytest.mark.parametrize('fault',[None,'binding','old_source'])
def test_historical_source_explicit_binding_before_trajectory(gate,tmp_path,monkeypatch,fault):
    from types import SimpleNamespace
    root,code,spec,pins,report,data,chain,save=fixture(gate,tmp_path,monkeypatch)
    path=code/f"configs/cari_clip_{spec.episode_index:06d}_historical_source_pins.json"
    path.write_text('{"independently_pinned":"fixture"}');path.chmod(0o444)
    old=tmp_path/'old-code';old.mkdir();proof={'files_verified':77,'queue_status_reclassified':False}
    report['historical_source_binding']=proof.copy()
    if fault=='binding':report['historical_source_binding']['files_verified']=76
    pins=save();calls=[]
    def verify(root_arg,path_arg):
        calls.append((root_arg,path_arg))
        if fault=='old_source':raise ValueError('Original historical file changed')
        return old,proof
    monkeypatch.setitem(sys.modules,'cari_historical_source',SimpleNamespace(verify_historical_source=verify))
    def lineage(*args,**kwargs):
        assert args[:2]==(root,code) and kwargs=={'source_code':old}
        return chain
    monkeypatch.setattr(gate.export.lineage,'verify_refined_artifacts',lineage)
    if fault:
        with pytest.raises(ValueError):gate.load_shared_track1_episode(root,code,spec,pins)
    else:
        loaded=gate.load_shared_track1_episode(root,code,spec,pins)
        assert loaded.episode.total_video_frames==spec.total_frames and len(calls)==2
