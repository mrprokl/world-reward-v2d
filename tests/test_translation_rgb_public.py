"""H97 pure translation/lineage firewall; no model or private-data execution."""
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA));monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec=importlib.util.spec_from_file_location("translation_public_test",INFRA/"translation_rgb_public.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def fixture(gate,monkeypatch):
    n=gate.native;monkeypatch.setattr(n,"VERTICES",100)
    raw={k:np.zeros(s,np.float32)for k,s in n.BLOCKS.items()};raw["pred_cam_t"][:]=[.1,-.1,4]
    raw["global_rot"][:]=[.1,.2,.3];controls=np.zeros(204,np.float32);controls[3:6]=raw["global_rot"]
    pair=dict(shared_model_controls=controls,shared_shape_params=np.zeros(45,np.float32),shared_scale_params=np.zeros(28,np.float32),
        shared_vertices_camera_m=np.tile(np.array([.5,.3,4],np.float32),(100,1)),
        shared_keypoints_camera_m=np.c_[np.linspace(-.3,.3,308),np.linspace(.5,-.5,308),np.full(308,4)].astype(np.float32),
        shared_joints_camera_m=np.tile(np.array([.5,.3,4],np.float32),(127,1)),
        shared_joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)),human_faces=np.zeros((36874,3),np.int64),
        camera_K=gate.previous.baseline.CAMERA_K.copy(),hand_mask_left=np.arange(100)<50,hand_mask_right=np.arange(100)>=50)
    record=dict(file="clip_00_frame_000.png",clip_index=0,frame_index=0,sha256="a"*64)
    raw.update(clip_index=np.array(0,np.int64),frame_index=np.array(0,np.int64))
    u=np.array([.15,-.2,.1]);d=gate.policy.bounded_delta(u);T=gate.policy.camera_translation(raw["pred_cam_t"],d)
    v,k,j=gate.translated_geometry(pair,raw["pred_cam_t"],T)
    data={key:value.copy()for key,value in raw.items()}
    data.update(mhr_model_params=controls.copy(),pred_cam_t=T.astype(np.float32),vertices_camera_m=v,keypoints_camera_m=k,joints_camera_m=j,
        joint_global_rotations=pair["shared_joint_global_rotations"].copy(),human_faces=pair["human_faces"].copy(),
        camera_K=pair["camera_K"].copy(),hand_mask_left=pair["hand_mask_left"].copy(),hand_mask_right=pair["hand_mask_right"].copy(),
        clip_index=np.array(0,np.int64),frame_index=np.array(0,np.int64),latent=u.copy(),physical_delta=d)
    return raw,pair,record,data


def test_external_translation_rigid_not_depth_or_identity_scaling(gate,monkeypatch):
    raw,pair,record,data=fixture(gate,monkeypatch)
    assert gate.validate_candidate(data,record,raw,pair)is data
    displacement=data["vertices_camera_m"].astype(np.float64)-pair["shared_vertices_camera_m"]
    assert np.ptp(displacement,axis=0).max()<1e-6
    assert data["mhr_model_params"].tobytes()==pair["shared_model_controls"].tobytes()


def test_full_geometry_camera_plane_not_only_selected_training_points(gate,monkeypatch):
    raw,pair,record,data=fixture(gate,monkeypatch)
    pair["shared_joints_camera_m"][100,2]=-1
    with pytest.raises(ValueError):gate.translated_geometry(pair,raw["pred_cam_t"],raw["pred_cam_t"])
    with pytest.raises(ValueError):gate.translated_geometry(pair,np.ma.array(raw["pred_cam_t"],mask=False),raw["pred_cam_t"])


@pytest.mark.parametrize("fault",["native","rotation","shape","PCA","body","hands","K","verticesScale","negativeZ","latent","delta","translation","masked","frame","extra"])
def test_candidate_no_hidden_other_degrees_of_freedom(gate,monkeypatch,fault):
    raw,pair,record,data=fixture(gate,monkeypatch)
    if fault=="native":data["mhr_model_params"][136]=.1
    elif fault=="rotation":data["joint_global_rotations"][0,0,0]=-1
    elif fault=="shape":data["shape_params"][0]=.1
    elif fault=="PCA":data["scale_params"][0]=.1
    elif fault=="body":data["body_pose_params"][0]=.1
    elif fault=="hands":data["hand_pose_params"][0]=.1
    elif fault=="K":data["camera_K"][0,0]+=1
    elif fault=="verticesScale":data["vertices_camera_m"]*=1.01
    elif fault=="negativeZ":data["keypoints_camera_m"][0,2]=0
    elif fault=="latent":data["latent"][0]+=.01
    elif fault=="delta":data["physical_delta"][0]+=.01
    elif fault=="translation":data["pred_cam_t"][0]+=.01
    elif fault=="masked":data["latent"]=np.ma.array(data["latent"],mask=False)
    elif fault=="frame":data["frame_index"]=np.array(1,np.int64)
    else:data["oracle_transform"]=np.eye(3)
    with pytest.raises(ValueError):gate.validate_candidate(data,record,raw,pair)


def test_proxies_are_reused_never_generated_or_selected_by_candidate(gate):
    source=Path(gate.__file__).read_text()
    assert"previous.frozen_proxies"in source and"FAILED_SHA"in source
    assert"raster_camera_mesh("not in source and"fit_shared_depth_scale("not in source


def test_source_only_identical_hashes_no_import_fit(gate,tmp_path):
    original=INFRA/"translation_rgb_fit.py";copy=tmp_path/"translation_rgb_fit.py";copy.write_bytes(original.read_bytes());copy.chmod(0o444)
    assert gate.helper_identities(copy)==gate.helper_identities()


def test_full_prior_public_arrays_and_frozen_proxies_no_recomputation(gate,tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location("prior_fixture",Path(__file__).with_name("test_keypoint_rgb_fit.py"))
    fixture_module=importlib.util.module_from_spec(spec);spec.loader.exec_module(fixture_module)
    prior=SimpleNamespace(baseline=gate.previous.baseline,native=gate.native,public=gate.previous,sha256=gate.sha256)
    base,_,_,_=fixture_module.public_fixture(prior,tmp_path,monkeypatch)
    records,raw,pairs,dw,lineage=gate.previous.public_predictions(tmp_path)
    out=base/"root_fit_v1";out.mkdir();(out/"candidates").mkdir();(out/"proxies").mkdir()
    rows=[]
    for record,original in zip(records,raw):
        points,pixels=gate.previous.object_sample(original,1.)
        data=dict(object_points_camera_m=points,pixel_indices=pixels,shared_depth_scale=np.array(1.),
            clip_index=original["clip_index"],frame_index=original["frame_index"])
        rows.append(gate.native.save(out,"proxies",record,data))
    report=dict(stage=gate.previous.STAGE,status="fail",phase="native_root_optimization",producer_revision=gate.FAILED_REV,
        script_sha256=gate.FAILED_SCRIPT,image_id=gate.native.IMAGE_ID,network="none",device="cuda",private_truth_read=False,
        ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],object_proxies_frozen_before_fit=True,
        all_candidates_frozen=False,quality_verified=False,accuracy_verified=False,adoption_authorized=False,
        error_type="ValueError",error="Complete249 native bounds/rootEuler mapping violated; no clipping",candidate_outputs=[],proxy_outputs=rows,
        counters=dict(proxy_rasters=15,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0))
    path=out/"report.json";path.write_text(json.dumps(report));path.chmod(0o444);monkeypatch.setattr(gate,"FAILED_SHA",gate.sha256(path))
    result=gate.public_predictions(tmp_path)
    assert len(result[4])==15 and result[-1]["failed_fit_report_sha256"]==gate.FAILED_SHA
    assert len(result[-1]["frozen_files"])==len(lineage["frozen_files"])+16
    proxy=out/rows[0]["artifact"];proxy.chmod(0o644);proxy.write_bytes(b"changed");proxy.chmod(0o444)
    with pytest.raises(ValueError):gate.public_predictions(tmp_path)
