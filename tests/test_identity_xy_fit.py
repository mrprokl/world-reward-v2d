"""Real tiny pixel, objective, NPZ and lineage tests; no CUDA/model execution."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest


@pytest.fixture
def fit(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/"infra"
    monkeypatch.syspath_prepend(str(infra));monkeypatch.syspath_prepend(str(infra.parent/"src"))
    spec=importlib.util.spec_from_file_location("own_identity_xy",infra/"identity_xy_fit.py")
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_predeclared_protocol_and_exact_budget(fit):
    assert fit.ANCHORS==tuple(np.rint(np.linspace(0,500,12)).astype(int))
    assert fit.MIDPOINTS==tuple((np.asarray(fit.ANCHORS[:-1])+fit.ANCHORS[1:])//2)
    assert len(fit.SELECTED)==23 and not set(fit.ANCHORS)&set(fit.MIDPOINTS)
    assert (fit.STEPS,fit.BUDGET,fit.LR,fit.DELTA_BOUND_M)==(30,600,.005,.05)
    assert (fit.PRIOR_STD_M,fit.PRIOR_WEIGHT,fit.SIGMA,fit.GAMMA,fit.FPP)==(.02,.001,1e-4,1e-4,8)


@pytest.fixture
def small(fit,monkeypatch):
    monkeypatch.setattr(fit,"WIDTH",96);monkeypatch.setattr(fit,"HEIGHT",96)
    monkeypatch.setattr(fit,"LOW_WIDTH",16);monkeypatch.setattr(fit,"LOW_HEIGHT",16)
    return np.ones((96,96),bool),np.zeros((96,96),bool)


def test_majority19of36_and_object_any_conservative_support(fit,small):
    human,obj=small;human[:6,:6]=False;human.flat[:18]=True
    # Explicit18 pixels in one6x6 block do NOT produce low human foreground.
    human[:6,:6]=False;human[:3,:6]=True
    low,train,support=fit.mask_protocol(human,obj)
    assert not low[0,0]
    human[3,0]=True;assert fit.mask_protocol(human,obj)[0][0,0]
    obj[0,0]=True
    assert not fit.mask_protocol(human,obj)[1][0,0]
    assert len(support)==2 and min(support)>32


def test_checkerboard_reserved_full_cells_exactly_correspond_and_no_input_mutation(fit,small):
    human,obj=small;before=human.copy();lowtrain=fit.split_mask(16,16,8);fulltrain=fit.split_mask(96,96,48)
    assert np.array_equal(fulltrain,lowtrain.repeat(6,axis=0).repeat(6,axis=1))
    _,train,_=fit.mask_protocol(human,obj)
    assert np.array_equal(train,lowtrain) and np.array_equal(human,before)
    assert (fulltrain&~fulltrain).sum()==0 and fulltrain.sum()==fulltrain.size//2


@pytest.mark.parametrize("fault",["shape","dtype","masked","onlytrain","onlyholdout","allobject","lowempty"])
def test_support_fail_closed_every_frame_both_partitions(fit,small,fault):
    human,obj=small
    if fault=="shape":human=human[:90]
    elif fault=="dtype":human=human.astype(np.uint8)
    elif fault=="masked":human=np.ma.array(human,mask=False)
    elif fault=="onlytrain":human=fit.split_mask(96,96,48)
    elif fault=="onlyholdout":human=~fit.split_mask(96,96,48)
    elif fault=="allobject":obj[:]=True
    else:
        human[:]=False;human[::6,::6]=True # Fullsupport>32 bothparts, no majority block.
    with pytest.raises(ValueError):fit.mask_protocol(human,obj)


def test_only_XY_bounded_physical_variables_not_geometry_scale(fit):
    w=np.zeros((23,2),np.float32);w[:,0]=.1;w[:,1]=-.1;before=w.copy()
    delta=fit.bounded_xy(w)
    np.testing.assert_allclose(delta,.05*np.tanh(w/.05),rtol=0,atol=0)
    assert np.abs(delta).max()<.05 and np.array_equal(w,before)


@pytest.mark.parametrize("fault",["shape","nan","dtype","masked"])
def test_XY_variable_validation(fit,fault):
    w=np.zeros((23,2),np.float32)
    if fault=="shape":w=w[:22]
    elif fault=="nan":w[0,0]=np.nan
    elif fault=="dtype":w=w.astype(np.int64)
    else:w=np.ma.array(w,mask=False)
    with pytest.raises(ValueError):fit.bounded_xy(w)


def test_real_loss_equal_frame_weight_train_only_prior_and_no_heldout_selection(fit,small):
    shape=(23,16,16);observed=np.zeros(shape,bool);train=np.repeat(fit.split_mask(16,16,8)[None],23,axis=0)
    alpha=np.zeros(shape,np.float32);alpha[0,train[0]]=1
    delta=np.zeros((23,2),np.float32)
    assert fit.training_loss(alpha,observed,train,delta)==pytest.approx(1/23)
    changed=alpha.copy();changed[~train]=1
    assert fit.training_loss(changed,observed,train,delta)==fit.training_loss(alpha,observed,train,delta)
    delta[:]=.01
    assert fit.training_loss(alpha,observed,train,delta)==pytest.approx(1/23+.001*(.01/.02)**2)
    train[0]=False
    with pytest.raises(ValueError):fit.training_loss(alpha,observed,train,delta)


def test_hard_holdout_outside_object_same_region_both_falsepositives_kept(fit,small):
    human,obj=small;pred=human.copy();train=fit.split_mask(96,96,48)
    pred[train]=False
    assert fit.hard_evidence(pred,human,obj)==1.
    human[:48,48:]=False
    assert fit.hard_evidence(pred,human,obj)==.5
    obj[:48,48:]=True
    assert fit.hard_evidence(pred,human,obj)==1.


def rows_and_branches(fit):
    rows=[dict(frame_index=i,raw=.2,M0_before=.3,M1_before=.3,M0_after=.4,M1_after=.42) for i in fit.SELECTED]
    branches={name:dict(actual_optimizer_steps=30,actual_backward_calls=30,all_backward_flags_restored=True,
        initial_train_loss=.1,final_train_loss=.09) for name in ("M0","M1")}
    return rows,branches


def test_paired_gates_mean_median_worst_train_nonincrease_no_raw_selection(fit):
    rows,branches=rows_and_branches(fit);result=fit.decision(rows,branches)
    assert result["hypothesis_gate_pass"] and not result["adoption_authorized"] and not result["accuracy_verified"]
    for row in rows:row["raw"]=1. # Raw per-frameidentity cannot veto validpairedtrial.
    assert fit.decision(rows,branches)["hypothesis_gate_pass"]
    rows[0]["M1_after"]=.38
    assert not fit.decision(rows,branches)["hypothesis_gate_pass"]
    rows,branches=rows_and_branches(fit);branches["M0"]["final_train_loss"]=.11
    assert not fit.decision(rows,branches)["hypothesis_gate_pass"]
    for row in rows:row["M1_after"]=.405 # mean5e-3 but median<1e-2.
    assert not fit.decision(rows,rows_and_branches(fit)[1])["hypothesis_gate_pass"]


@pytest.mark.parametrize("fault",["drop","order","nan","range","partialsteps","partialbackward","restore","missingbranch"])
def test_pair_contract_fail_not_abstention_or_sample_drop(fit,fault):
    rows,branches=rows_and_branches(fit)
    if fault=="drop":rows.pop()
    elif fault=="order":rows.reverse()
    elif fault=="nan":rows[0]["M0_after"]=np.nan
    elif fault=="range":rows[0]["raw"]=2.
    elif fault=="partialsteps":branches["M0"]["actual_optimizer_steps"]=29
    elif fault=="partialbackward":branches["M1"]["actual_backward_calls"]=29
    elif fault=="restore":branches["M0"]["all_backward_flags_restored"]=False
    else:del branches["M0"]
    with pytest.raises(ValueError):fit.decision(rows,branches)


def receipts(fit):
    common=dict(status="pass",episode_index=15,input_track="track_1",input_sha256=fit.VIDEO_SHA,
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
    raw=dict(common,stage="sam3d_body_full_video_initializer",total_video_frames=fit.FRAMES,frame_indices=list(range(fit.FRAMES)),
        script_sha256=fit.BODY_SCRIPT_SHA,predictions_sha256=fit.PRED_SHA,mask_report_sha256=fit.MASK_SHA,prompts_sha256=fit.PROMPTS_SHA,
        network="none",inference_type="body",body_revision=fit.body.BODY_REVISION,upstream_revision=fit.body.UPSTREAM_REVISION,
        camera_intrinsics="RGB_size_default_FOV",vertices=fit.VERTICES,faces=fit.FACES,geometry_units="metres",
        geometry_frame="SAM3D_camera_x_right_y_down_z_forward",translation="vertices_camera_m = vertices_root_camera_m + pred_cam_t (exactly once)",
        mhr_geometry_forward_verified=True,human_mask_id=0,prompt_mode="automatic_mask_and_derived_bbox_no_fallback",
        model_mask_range="uint8_0_1_matches_CARI_prepare_batch",
        frames=[dict(frame_index=i,mask_sha256="a"*64,decoded_rgb_sha256="b"*64) for i in range(fit.FRAMES)],
        checkpoint_loading=dict(mode="strict_network_and_head_state_with_explicit_asset_buffer_retention",unexpected_keys=[],
            parameter_tensors_loaded=1101,retained_mhr_asset_buffer_names=[str(i) for i in range(113)]),
        inference_source_identity=dict(python_files=46,sha256="14f583ce78e8e786cb77739addc2358a1245b23f79ab676c4094231694fe8f3c"))
    masks=dict(common,stage="automatic_masks",frames=fit.FRAMES,script_sha256=fit.MASK_SCRIPT_SHA)
    return raw,masks


def test_legacy_missing_dataset_basis_not_invented_currenthelper_not_oldscript(fit):
    raw,masks=receipts(fit);fit.validate_receipts(raw,masks)
    assert "input_dataset_revision" not in raw and "mhr_geometry_forward_basis" not in raw
    assert fit.BODY_SCRIPT_SHA!=fit.sha256(Path(fit.body.__file__))
    assert fit.MASK_SCRIPT_SHA=="2f991e4f144231db57f4caab92a13d8af5486d82c9f4e5b8bcc36b5b6f4b6c52"


@pytest.mark.parametrize("fault",["gt","oracle","frames","indexbool","mode","source","datasetpresentbad","masksha","buffers","fullfusion"])
def test_exact_legacy_receipts_fail_closed(fit,fault):
    raw,masks=receipts(fit)
    if fault=="gt":raw["ground_truth_used"]=True
    elif fault=="oracle":masks["oracle_modes"]=["calibration"]
    elif fault=="frames":masks["frames"]-=1
    elif fault=="indexbool":raw["episode_index"]=True
    elif fault=="mode":raw["checkpoint_loading"]["mode"]="lenient"
    elif fault=="source":raw["script_sha256"]="f"*64
    elif fault=="datasetpresentbad":raw["input_dataset_revision"]="unknown"
    elif fault=="masksha":raw["mask_report_sha256"]="f"*64
    elif fault=="buffers":raw["checkpoint_loading"]["retained_mhr_asset_buffer_names"].pop()
    else:raw["inference_type"]="full"
    with pytest.raises(ValueError):fit.validate_receipts(raw,masks)


def tiny_arrays(fit,monkeypatch):
    monkeypatch.setattr(fit,"VERTICES",4);monkeypatch.setattr(fit,"FACES",2)
    arrays={k:np.zeros((501,*v),np.float32) for k,v in fit.body.PARAMETER_SHAPES.items() if k!="pred_vertices"}
    verts=np.array([[0,0,.1],[1,0,.1],[0,1,.1],[.2,.3,1]],np.float32)
    arrays.update(vertices_root_camera_m=np.repeat(verts[None],501,axis=0),faces=np.array([[0,1,2],[0,2,3]],np.int64),
        frame_index=np.arange(501,dtype=np.int64))
    arrays["pred_cam_t"][:,2]=2;arrays["focal_length"][:]=1920.
    arrays["vertices_camera_m"]=arrays["vertices_root_camera_m"]+arrays["pred_cam_t"][:,None]
    return arrays


def test_actual_full501_numpy_loader_selects_protocol_not_targeterrors(fit,monkeypatch,tmp_path):
    arrays=tiny_arrays(fit,monkeypatch);path=tmp_path/"pred.npz";np.savez_compressed(path,**arrays)
    loaded=fit.load_predictions(path)
    assert loaded["vertices_camera_m"].shape==(23,4,3)
    assert np.array_equal(loaded["pred_cam_t"],arrays["pred_cam_t"][list(fit.SELECTED)])
    assert "pred_pose_raw" not in loaded


@pytest.mark.parametrize("fault",["extra","missing","dtype","unusednan","expression","focal","translation","index"])
def test_numpy_loader_checks_all501_and_exact_ABI(fit,monkeypatch,tmp_path,fault):
    arrays=tiny_arrays(fit,monkeypatch)
    if fault=="extra":arrays["gt_pose"]=np.zeros(1)
    elif fault=="missing":del arrays["pred_pose_raw"]
    elif fault=="dtype":arrays["shape_params"]=arrays["shape_params"].astype(np.float64)
    elif fault=="unusednan":arrays["pred_global_rots"][1,0,0,0]=np.nan
    elif fault=="expression":arrays["expr_params"][1,0]=.1
    elif fault=="focal":arrays["focal_length"][1]=900
    elif fault=="translation":arrays["vertices_camera_m"][fit.SELECTED[1],0,2]+=1
    else:arrays["frame_index"][-1]=0
    path=tmp_path/"pred.npz";np.savez_compressed(path,**arrays)
    with pytest.raises(ValueError):fit.load_predictions(path)


def test_frozen_npz_signed_zero_identity_and_overwrite_forbidden(fit,tmp_path):
    a=np.array([0.,-0.,1.],np.float32);path=tmp_path/"identity.npz"
    p,digest=fit.freeze_npz(path,identity=a)
    assert p==path and fit.sha256(path)==digest and not path.stat().st_mode&0o222
    with np.load(path) as data:assert data["identity"].tobytes()==a.tobytes()
    with pytest.raises(FileExistsError):fit.freeze_npz(path,identity=a)


def test_source_wiring_true_train_only_bothfinal_freeze_before_all_hard_scoring(fit):
    source=Path(fit.__file__).read_text();tree=ast.parse(source)
    run=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="run")
    text=ast.unparse(run)
    assert text.index("geometry_and_masks_frozen_before_optimization=True")<text.index("torch.optim.Adam")
    assert text.index("final_parameters_frozen_before_reserved_scoring=True")<text.index("camera.raster_camera_mesh(")
    assert "range(STEPS)" in text and "torch.optim.Adam([w], lr=LR)" in text
    assert "DELTA_BOUND_M * torch.tanh(w / DELTA_BOUND_M)" in text
    assert "with scoped_backward(torch, scope):" in text and "loss.backward()" in text
    objective=next(n for n in ast.walk(run) if isinstance(n,ast.FunctionDef) and n.name=="objective")
    objtext=ast.unparse(objective)
    assert "train" in objtext and "holdout" not in objtext and "hard_evidence" not in objtext
    assert "torch.zeros((len(SELECTED), 1)" in objtext # Z offset exactlyzero, not differentiable.
    assert "!= 115" in text and "!= 64" in text and "!= 4" in text
    assert "best" not in text and "argmin" not in text # No later result-dependent iteration/identity selection.


def test_wrapper_scoped_readonly_no_video_private_or_nativeoutputs(fit):
    path=Path(fit.__file__).with_name("run_identity_xy_fit.sh");subprocess.run(["bash","-n",str(path)],check=True)
    source=path.read_text();mounts=[s for s in source.splitlines() if '--mount "' in s]
    assert len(mounts)==13 and all("readonly" in s for s in mounts[:-1]) and "src=$OUT,dst=$OUT" in mounts[-1]
    assert "--network none" in source and "603s" in source and "--memory 32g --cpus 4" in source
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in source and "! -e \"$OUT\"" in source
    assert not any(s in source for s in ("eval_private","videos/","cari_forward","weights/mhr","src=$ROOT/outputs,dst"))


def test_cli_rejects_tuning_or_other_episode_before_runtime(fit):
    for argv in (["--episode","0"],["--steps","60"],["--lr",".1"]):
        with pytest.raises(SystemExit) as exc:fit.main(argv)
        assert exc.value.code==2


def test_GPU_envelope_reports_peak_not_just_live_allocation(fit):
    cuda=SimpleNamespace(memory_allocated=lambda:100,memory_reserved=lambda:200,
        max_memory_allocated=lambda:300,max_memory_reserved=lambda:400)
    report={};fit.memory_check(SimpleNamespace(cuda=cuda),report)
    assert report==dict(max_gpu_allocated_bytes=300,max_gpu_reserved_bytes=400)
    cuda.max_memory_reserved=lambda:fit.MEMORY_LIMIT+1
    with pytest.raises(MemoryError):fit.memory_check(SimpleNamespace(cuda=cuda),report)


@pytest.mark.parametrize("fault",[None,"disabled","warn","matmultf32","cudnntf32","cudnndeterministic","benchmark"])
def test_each_forward_explicitly_requires_strict_flags(fit,fault):
    state=dict(enabled=True,warn=False)
    matmul=SimpleNamespace(allow_tf32=False);cudnn=SimpleNamespace(allow_tf32=False,deterministic=True,benchmark=False)
    torch=SimpleNamespace(are_deterministic_algorithms_enabled=lambda:state["enabled"],
        is_deterministic_algorithms_warn_only_enabled=lambda:state["warn"],backends=SimpleNamespace(cuda=SimpleNamespace(matmul=matmul),cudnn=cudnn))
    if fault=="disabled":state["enabled"]=False
    elif fault=="warn":state["warn"]=True
    elif fault=="matmultf32":matmul.allow_tf32=True
    elif fault=="cudnntf32":cudnn.allow_tf32=True
    elif fault=="cudnndeterministic":cudnn.deterministic=False
    elif fault=="benchmark":cudnn.benchmark=True
    if fault is None:fit.strict_forward(torch)
    else:
        with pytest.raises(ValueError):fit.strict_forward(torch)
