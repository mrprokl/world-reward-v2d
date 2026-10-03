"""Tiny H98 numerical/provenance contracts; no local native CUDA/assets."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

REPO=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(REPO/"infra"));monkeypatch.syspath_prepend(str(REPO/"src"))
    import root5_rgb_protocol, keypoint_rgb_baseline
    import root5_rgb_public
    path=REPO/"infra/root5_rgb_fit.py";spec=importlib.util.spec_from_file_location("h98_fit_test",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def fixture():
    r=np.array([.1,-.2,.3],np.float32);t=np.array([.2,-.3,4.],np.float32);c=np.zeros(204,np.float32);c[3:6]=r
    return c,r,t,dict(pred_cam_t=t.copy()),dict(shared_model_controls=c.copy()),np.tile([-1.,1.],(3,1))


def test_exact60evaluated59updates_best_first_tie(gate):
    evaluated=[];updated=[]
    losses,best=gate.optimization_schedule(lambda i:(evaluated.append(i)or float(1 if i in(2,3)else 2)),lambda i:updated.append(i))
    assert evaluated==list(range(60))and updated==list(range(59))and best==2 and len(losses)==60
    with pytest.raises(ValueError):gate.optimization_schedule(lambda i:np.nan,lambda i:None)


def test_capability_root_contract_parity_and_unregularized_rank_are_unchanged(gate):
    original=ast.parse((REPO/"infra/root5_native_capability.py").read_text());own=ast.parse(Path(gate.__file__).read_text())
    for name in("root_contract","parity","jacobian_evidence"):
        a=next(n for n in original.body if isinstance(n,ast.FunctionDef)and n.name==name)
        b=next(n for n in own.body if isinstance(n,ast.FunctionDef)and n.name==name)
        if name=="parity":
            # Only reference module alias differs, not the numerical contract.
            text=ast.unparse(a).replace("public.baseline.rotations","baseline.rotations");a=ast.parse(text).body[0]
        assert ast.dump(a,include_attributes=False)==ast.dump(b,include_attributes=False)
    c,r,t,raw,pair,bounds=fixture();c[17]=.5;pair["shared_model_controls"][17]=.5
    gate.root_contract(c,r,t,raw,pair,bounds)
    t[2]+=.1
    with pytest.raises(ValueError):gate.root_contract(c,r,t,raw,pair,bounds)


def test_rank_and_analyticXY_use_only_same_initial_native_points(gate):
    points=np.column_stack((np.linspace(-.4,.4,10),np.linspace(-.3,.3,10),np.linspace(2.,5.,10))).astype(np.float32)
    K=np.array([[1280.,0.,512.],[0.,1280.,384.],[0.,0.,1.]],np.float32);J=np.zeros((20,5),np.float32)
    J[::2,0]=np.float32(.3)*K[0,0]/points[:,2];J[1::2,1]=np.float32(.3)*K[1,1]/points[:,2]
    assert gate.analytic_xy_evidence(J,points,K)["additional_native_calls"]==0
    assert gate.jacobian_evidence(J,10)["minimum_maximum_ratio"]==0
    J[0,1]=1e-3
    with pytest.raises(ValueError):gate.analytic_xy_evidence(J,points,K)


def test_safeguard_full15_never_selects_or_blocks_frozen_candidates(gate):
    before=np.full(15,.75);after=before+.01
    result=gate.silhouette_safeguard(before,after)
    assert result["safeguard_pass"]and not result["used_for_state_selection"]and not result["candidate_export_blocked_by_safeguard"]
    after[4]=before[4]-.011
    assert not gate.silhouette_safeguard(before,after)["safeguard_pass"]
    with pytest.raises(ValueError):gate.silhouette_safeguard(before[:14],after[:14])
    with pytest.raises(ValueError):gate.silhouette_safeguard(np.ma.array(before,mask=False),after)


def test_fixed_automatic_object_support_no_candidate_dependent_mask(gate,monkeypatch):
    monkeypatch.setattr(gate.native,"HEIGHT",8);monkeypatch.setattr(gate.native,"WIDTH",16)
    human=np.ones((8,16),bool);obj=np.zeros_like(human);obj[:2]=True;prediction=human.copy();prediction[:2]=False
    assert gate.mask_iou(prediction,human,obj)==1
    prediction[2]=False;assert gate.mask_iou(prediction,human,obj)==80/96
    human[2:]=False
    with pytest.raises(ValueError):gate.mask_iou(prediction,human,obj)


def test_native_attempt_counter_precedes_any_return_and_no_kernel_suppression(gate,monkeypatch):
    class Tensor:
        def __init__(self,v):self.value=np.asarray(v)
        def __getitem__(self,k):return Tensor(self.value[k])
        def __mul__(self,x):return Tensor(self.value*(x.value if isinstance(x,Tensor)else x))
        def __add__(self,x):return Tensor(self.value+(x.value if isinstance(x,Tensor)else x))
    torch=SimpleNamespace(float32=np.float32,tensor=lambda v,**k:Tensor(v),tanh=lambda v:Tensor(np.tanh(v.value)),
        stack=lambda v:Tensor(np.stack([x.value for x in v])),zeros_like=lambda v:Tensor(np.zeros_like(v.value)))
    row=dict(native_heads_attempted=0,native_heads_returned=0,native_heads_validated=0)
    report=dict(counters=dict(objective_native_heads_attempted=0,objective_native_heads_returned=0,objective_native_heads_validated=0));seen=[]
    def failed(**kwargs):seen.append((report["counters"].copy(),row.copy()));raise RuntimeError("backend")
    fixed={k:Tensor(np.zeros((1,*s)))for k,s in gate.native.BLOCKS.items()if k!="mhr_model_params"};fixed["model_controls"]=Tensor(np.zeros(204))
    monkeypatch.setattr(gate,"strict_forward",lambda t:None)
    with pytest.raises(RuntimeError,match="backend"):
        gate.differentiable_native(torch,SimpleNamespace(enable_hand_model=False,mhr_forward=failed),fixed,{},Tensor(np.zeros(5)),None,report,row,lambda:None,"objective_native_heads")
    assert seen[0][0]["objective_native_heads_attempted"]==1 and seen[0][0]["objective_native_heads_returned"]==0
    text=Path(gate.__file__).read_text();assert"use_deterministic_algorithms(False"not in text and"inference_mode()"not in text


def test_source_freezes_before_safeguard_and_unchanged_GPU_recipe(gate):
    source=Path(gate.__file__).read_text()
    assert source.index('report.update(all_candidates_frozen=True')<source.index('after_iou.append(')
    assert 'torch.optim.Adam([u],lr=.01,betas=(.9,.999),eps=1e-8)'in source
    assert 'BOUNDS=np.array((.30,.30,.30/np.sqrt(3.)'in source and 'origin[2]))'in source
    assert 'initial_observation_jacobian_values'in source and 'evaluated_projected_points'in source and 'update_gradients'in source
    shell=REPO/"infra/run_root5_rgb_fit.sh";subprocess.run(["bash","-n",str(shell)],check=True)
    text=shell.read_text();assert "eval_private"not in text and "TIMEOUT=303"in text and "TIMEOUT=123"in text and "--network none"in text and "--gpus all"in text
    assert '--public-pins'in text and '--fit-pins'in text and 'root_fit_v1,readonly'in text


def trace_fixture(gate):
    xy=np.zeros((1,133,2),np.float64);scores=np.ones((1,133),np.float32)
    xy[0,:,0]=np.arange(133)*4.;xy[0,:,1]=np.arange(133)*3.
    observation=dict(keypoints=xy,scores=scores);target,indices,valid=gate.policy.training_observations(xy[0],scores[0])
    n=len(target);J=np.tile(np.eye(5),(4,1))[:n*2];evidence=gate.jacobian_evidence(J,n)
    row=dict(training_validity=valid.tolist(),training_MHR_indices=indices.tolist(),training_points=n,adam_updates=59,observation_compute_dtype="float32",
        heldout_used_for_fit=False,heldout_COCO_indices=list(gate.policy.HELDOUT_COCO),evaluated_latents=np.zeros((60,5)).tolist(),
        evaluated_physical_deltas=np.zeros((60,5)).tolist(),evaluated_projected_points=np.tile(target,(60,1,1)).tolist(),evaluated_losses=[0.]*60,
        update_gradients=np.zeros((59,5)).tolist(),initial_observation_jacobian_values=J.tolist(),initial_observation_jacobian=evidence,
        best_evaluated_index=0,selected_total_objective=0.,selected_physical_delta=[0.]*5)
    candidate=dict(latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32))
    return row,observation,candidate


def test_frozen_schedule_replay_has_no_native_or_optimizer_calls(gate):
    row,observation,candidate=trace_fixture(gate)
    assert gate.validate_trace(row,observation,candidate)["manual_Adam_and_public_objectives_verified"]
    row["update_gradients"][0][0]=1
    with pytest.raises(ValueError,match="Adam"):gate.validate_trace(row,observation,candidate)


def test_full_nonzero_recorded_F32_Adam_schedule_and_native_objective(gate):
    row,observation,candidate=trace_fixture(gate);target,_,_=gate.policy.training_observations(observation["keypoints"][0],observation["scores"][0])
    rng=np.random.default_rng(981);grads=rng.normal(0,.04,(59,5)).astype(np.float32);u=np.zeros((60,5),np.float32);m=np.zeros(5,np.float32);v=m.copy()
    for i,g in enumerate(grads):
        m=np.float32(.9)*m+np.float32(.1)*g;v=np.float32(.999)*v+np.float32(.001)*g*g
        u[i+1]=(u[i]-.01*(m/(1-.9**(i+1)))/(np.sqrt(v/(1-.999**(i+1)))+1e-8)).astype(np.float32)
    d=(np.tanh(u)*gate.BOUNDS.astype(np.float32)).astype(np.float32);xy=np.tile(target.astype(np.float32),(60,1,1))
    losses=[gate.policy.objective(p.astype(np.float64),target.astype(np.float32).astype(np.float64),np.r_[q[:2],0.,q[2:]].astype(np.float64))["total"]for p,q in zip(xy,d)]
    row.update(evaluated_latents=u.tolist(),evaluated_physical_deltas=d.tolist(),evaluated_projected_points=xy.tolist(),evaluated_losses=losses,
        update_gradients=grads.tolist());best=int(np.argmin(losses));row.update(best_evaluated_index=best,selected_total_objective=losses[best],selected_physical_delta=d[best].tolist())
    candidate.update(latent=u[best],physical_delta=d[best]);assert gate.validate_trace(row,observation,candidate)["states"]==60


def test_actual_new_public_candidate_contract_keeps_Z_and_all_nonroot_controls(gate,monkeypatch):
    monkeypatch.setattr(gate.native,"VERTICES",100)
    raw={k:np.zeros(s,np.float32)for k,s in gate.native.BLOCKS.items()};raw["pred_cam_t"]=np.array([.2,-.3,4.],np.float32)
    raw["global_rot"]=np.array([.1,-.2,.3],np.float32);raw["mhr_model_params"][3:6]=raw["global_rot"]
    faces=np.tile(np.array([0,1,2],np.int64),(36874,1));K=gate.native.CAMERA_K.copy();left=np.arange(100)<50;right=~left
    pair=dict(shared_shape_params=np.zeros(45,np.float32),shared_scale_params=np.zeros(28,np.float32),shared_model_controls=raw["mhr_model_params"].copy(),
        human_faces=faces,camera_K=K,hand_mask_left=left,hand_mask_right=right)
    candidate={k:v.copy()for k,v in raw.items()};candidate.update(vertices_camera_m=np.ones((100,3),np.float32),keypoints_camera_m=np.ones((308,3),np.float32),
        joints_camera_m=np.ones((127,3),np.float32),joint_global_rotations=np.tile(np.eye(3,dtype=np.float32),(127,1,1)),human_faces=faces,camera_K=K,
        hand_mask_left=left,hand_mask_right=right,clip_index=np.array(0,np.int64),frame_index=np.array(0,np.int64),latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32))
    record=dict(clip_index=0,frame_index=0);assert gate.validate_candidate(candidate,record,raw,pair)is candidate
    candidate["pred_cam_t"][2]+=.001
    with pytest.raises(ValueError):gate.validate_candidate(candidate,record,raw,pair)


@pytest.mark.parametrize("fault",["loss","best","delta","rank","support","states","masked"])
def test_schedule_trace_tampering_fails_before_native_replay(gate,fault):
    row,observation,candidate=trace_fixture(gate)
    if fault=="loss":row["evaluated_losses"][0]=1.
    elif fault=="best":row["best_evaluated_index"]=1
    elif fault=="delta":candidate["physical_delta"][0]=1e-5
    elif fault=="rank":row["initial_observation_jacobian_values"]=np.zeros((20,5)).tolist()
    elif fault=="support":row["training_validity"][0]=False
    elif fault=="states":row["evaluated_latents"].pop()
    else:row["evaluated_latents"]=np.ma.array(row["evaluated_latents"],mask=False)
    with pytest.raises(ValueError):gate.validate_trace(row,observation,candidate)


def frozen_cohort(gate,monkeypatch,tmp_path):
    """Actual 15-frame producer schemas and immutable IO, with only image/mesh sizes reduced."""
    native=gate.native
    monkeypatch.setattr(native,"WIDTH",16);monkeypatch.setattr(native,"HEIGHT",8);monkeypatch.setattr(native,"VERTICES",100)
    root=tmp_path;out=root/gate.OUT
    for folder in("candidates","proxies"):(out/folder).mkdir(parents=True)
    K=native.CAMERA_K.copy();yy,xx=np.indices((8,16));depth=np.full((8,16),2.,np.float32)
    points=np.stack(((xx+.5-K[0,2])/K[0,0]*depth,(yy+.5-K[1,2])/K[1,1]*depth,depth),-1).astype(np.float32)
    faces=np.tile(np.array([0,1,2],np.int64),(36874,1));R=np.tile(np.eye(3,dtype=np.float32),(127,1,1))
    human=np.ones((8,16),bool);obj=np.zeros_like(human);obj[:2]=True
    raw_frames=[];pairs=[];records=[];candidate_rows=[];proxy_rows=[];traces=[];observations=[]
    for index in range(15):
        clip,frame=divmod(index,5)
        record=dict(file=f"clip_{clip:02d}_frame_{frame:03d}.png",clip_index=clip,frame_index=frame,sha256=f"{index:064x}")
        raw={k:np.zeros(shape,np.float32)for k,shape in native.BLOCKS.items()}
        raw["pred_cam_t"][:]=[.02*frame,-.03,3.+clip];raw["global_rot"][:]=[.01*frame,-.02,.03]
        raw["mhr_model_params"][3:6]=raw["global_rot"]
        raw["mhr_model_params"][6:136]=np.float32(.001*frame)
        raw["mhr_model_params"][136:]=np.float32(.002*clip)
        raw["shape_params"][:]=np.float32(.02*frame+.03*clip);raw["scale_params"][:]=np.float32(.01*frame+.02*clip)
        raw.update(raw_depth=depth.copy(),raw_points=points.copy(),validity=np.ones((8,16),bool),rendered_depth=depth.copy(),silhouette=human.copy(),
            human_mask=human.copy(),object_mask=obj.copy(),raw_vertices_camera_m=np.ones((100,3),np.float32),
            raw_keypoints_camera_m=np.ones((308,3),np.float32),raw_joints_camera_m=np.ones((127,3),np.float32),raw_joint_global_rotations=R.copy(),
            human_faces=faces.copy(),camera_K=K.copy(),clip_index=np.array(clip,np.int64),frame_index=np.array(frame,np.int64))
        pair={}
        for mode in("raw","shared"):
            for suffix,source in(("vertices_camera_m","raw_vertices_camera_m"),("keypoints_camera_m","raw_keypoints_camera_m"),
                ("joints_camera_m","raw_joints_camera_m"),("joint_global_rotations","raw_joint_global_rotations"),
                ("model_controls","mhr_model_params"),("shape_params","shape_params"),("scale_params","scale_params")):
                pair[mode+"_"+suffix]=raw[source].copy()
        first=raw if frame==0 else raw_frames[clip*5]
        pair.update(shared_shape_params=first["shape_params"].copy(),shared_scale_params=first["scale_params"].copy(),
            human_faces=faces.copy(),camera_K=K.copy(),pred_cam_t=raw["pred_cam_t"].copy(),expression=raw["expr_params"].copy(),
            hand_mask_left=np.arange(100)<50,hand_mask_right=np.arange(100)>=50,clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy())
        gate.baseline.validate_raw(raw,record);gate.baseline.validate_pair(pair,record,raw)
        candidate={k:raw[k].copy()for k in native.BLOCKS}
        candidate.update(shape_params=pair["shared_shape_params"].copy(),scale_params=pair["shared_scale_params"].copy(),mhr_model_params=pair["shared_model_controls"].copy(),
            vertices_camera_m=pair["shared_vertices_camera_m"].copy(),keypoints_camera_m=pair["shared_keypoints_camera_m"].copy(),joints_camera_m=pair["shared_joints_camera_m"].copy(),
            joint_global_rotations=R.copy(),human_faces=faces.copy(),camera_K=K.copy(),hand_mask_left=pair["hand_mask_left"].copy(),hand_mask_right=pair["hand_mask_right"].copy(),
            clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy(),latent=np.zeros(5,np.float32),physical_delta=np.zeros(5,np.float32))
        gate.validate_candidate(candidate,record,raw,pair)
        selected,pixels=gate.object_sample(raw,.83+.02*clip)
        proxy=dict(object_points_camera_m=selected,pixel_indices=pixels,shared_depth_scale=np.array(.83+.02*clip,np.float64),
            clip_index=raw["clip_index"].copy(),frame_index=raw["frame_index"].copy())
        candidate_rows.append(native.save(out,"candidates",record,candidate));proxy_rows.append(native.save(out,"proxies",record,proxy))
        row,observation,_=trace_fixture(gate);row["file"]=record["file"];traces.append(row);observations.append(observation)
        records.append(record);raw_frames.append(raw);pairs.append(pair)
    gate.baseline.validate_clip_constants(pairs,raw_frames)
    return root,records,raw_frames,pairs,candidate_rows,proxy_rows,traces,observations


def test_full15_frozen_candidate_proxy_trace_roundtrip_without_native_backend(gate,monkeypatch,tmp_path):
    root,records,raw,pairs,crows,prows,traces,dw=frozen_cohort(gate,monkeypatch,tmp_path)
    candidates=gate.frozen_candidates(root,records,crows,raw,pairs)
    proxies=gate.frozen_proxies(root,records,prows,raw,pairs)
    checks=[gate.validate_schedule(row,obs,candidate)for row,obs,candidate in zip(traces,dw,candidates)]
    assert len(candidates)==len(proxies)==len(checks)==15
    assert sum(check["states"]for check in checks)==900 and sum(check["updates"]for check in checks)==885
    for index,(candidate,proxy)in enumerate(zip(candidates,proxies)):
        assert candidate["pred_cam_t"][2:].tobytes()==raw[index]["pred_cam_t"][2:].tobytes()
        assert candidate["mhr_model_params"][6:].tobytes()==pairs[index]["shared_model_controls"][6:].tobytes()
        assert len(proxy["pixel_indices"])==32
    for row in crows+prows:
        assert not Path(row["path"]).stat().st_mode&0o222
        assert gate.native.regular(row["path"],row["sha256"],immutable=True)["bytes"]==row["bytes"]
    # This validates numerical records and IO only; no fake native model/gradient is invoked.
    assert gate.frozen_candidates(root,records,crows,raw,pairs)[7]["latent"].tobytes()==candidates[7]["latent"].tobytes()


@pytest.mark.parametrize("fault",["candidate_content","RGB_binding","missing_frame","unexpected_artifact","trace_missing"])
def test_frozen15_contract_rejects_tampering_before_any_native_or_private_read(gate,monkeypatch,tmp_path,fault):
    root,records,raw,pairs,crows,prows,traces,dw=frozen_cohort(gate,monkeypatch,tmp_path)
    if fault=="candidate_content":
        path=Path(crows[7]["path"])
        with np.load(path,allow_pickle=False)as archive:data={k:archive[k]for k in archive.files}
        data["pred_cam_t"][2]+=np.float32(.01);path.chmod(0o644)
        with path.open("wb")as stream:np.savez_compressed(stream,**data)
        path.chmod(0o444);crows[7].update(gate.native.regular(path,immutable=True))
    elif fault=="RGB_binding":crows[7]["rgb_sha256"]="f"*64
    elif fault=="missing_frame":crows.pop(7)
    elif fault=="unexpected_artifact":(root/gate.OUT/"candidates/extra.npz").write_bytes(b"unexpected")
    else:
        traces[7]["evaluated_latents"].pop()
        with pytest.raises(ValueError,match="training trace"):gate.validate_schedule(traces[7],dw[7],gate.frozen_candidates(root,records,crows,raw,pairs)[7])
        return
    with pytest.raises(ValueError):gate.frozen_candidates(root,records,crows,raw,pairs)


def test_main_requires_canonical_immutable_pins_before_public_data(gate,monkeypatch,tmp_path):
    called=[];monkeypatch.setattr(gate.public,"public_predictions",lambda *a:called.append(a))
    monkeypatch.setenv("WR_CODE",str(tmp_path));config=tmp_path/"configs/root5_rgb_public_pins_v1.json";config.parent.mkdir()
    pins=dict(schema=gate.public.PIN_SCHEMA,producer_revision=dict(masks=gate.public.PRODUCER_REVISION,baseline=gate.public.NATIVE_PRODUCER_REVISION,dwpose=gate.public.NATIVE_PRODUCER_REVISION),
        manifest_sha256=gate.public.MANIFEST_SHA,automatic_masks_sha256="a"*64,baseline_sha256="b"*64,dwpose_sha256="c"*64)
    config.write_text(json.dumps(pins));config.chmod(0o644)
    with pytest.raises(ValueError,match="frozen public artifact"):gate.main(["fit","--public-pins",str(config)])
    config.chmod(0o444)
    with pytest.raises(ValueError,match="source-bundle configuration"):gate.main(["fit","--public-pins",str(tmp_path/"foreign.json")])
    with pytest.raises(SystemExit):gate.main(["replay","--public-pins",str(config)])
    assert called==[]


def test_fit_top_level_updates_are_real_not_replay_zero_claim(gate):
    tree=ast.parse(Path(gate.__file__).read_text())
    fit=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="fit_frame")
    update=next(n for n in ast.walk(fit)if isinstance(n,ast.FunctionDef)and n.name=="update")
    increments={ast.unparse(n.target):ast.unparse(n.value)for n in ast.walk(update)if isinstance(n,ast.AugAssign)}
    assert increments['report[\'optimizer_updates\']']=="1"and increments['report[\'native_backwards\']']=="1"
    replay=next(n for n in tree.body if isinstance(n,ast.FunctionDef)and n.name=="replay_perform")
    assert "optimizer_updates=0"in ast.unparse(replay)and"native_backwards=0"in ast.unparse(replay)
