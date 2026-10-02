"""Small complete-parameter fixtures, never native GPU or large target execution."""
import ast
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_full_direct", infra / "cari_direct_shared_full.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def params(gate):
    return {key: np.zeros((790, dim), np.float32) if key == "mhr_face" else
            np.repeat(np.arange(790, dtype=np.float32)[:, None], dim, axis=1)
            for key, dim in gate.probe.convert.PARAMETER_DIMS.items()}


def test_full790_source_identity_preserves_every_pose_and_zero_face(gate):
    original = params(gate); before = copy.deepcopy(original); result = gate.full_inputs(original)
    for key in original:
        assert np.array_equal(original[key], before[key]) and not np.shares_memory(original[key], result[key])
        expected = np.repeat(original[key][:1], 790, axis=0) if key in ("mhr_shape", "mhr_scale") else original[key]
        assert np.array_equal(result[key], expected)
    assert not result["mhr_face"].any()


def test_complete_chunk16_tail6_no_repeats_or_padding(gate):
    chunks = gate.chunks(); assert len(chunks) == 50
    assert (chunks[-1].start, chunks[-1].stop) == (784, 790)
    assert [i for chunk in chunks for i in range(chunk.start, chunk.stop)] == list(range(790))


@pytest.mark.parametrize("fault", ["shape", "dtype", "extra", "face", "nan"])
def test_original_inputs_fail_closed(gate, fault):
    values = params(gate)
    if fault == "shape": values["mhr_hand"] = values["mhr_hand"][:789]
    elif fault == "dtype": values["mhr_shape"] = values["mhr_shape"].astype(np.float64)
    elif fault == "extra": values["gt"] = np.zeros(1)
    elif fault == "face": values["mhr_face"][9, 2] = 1
    else: values["mhr_trans"][0, 0] = np.nan
    with pytest.raises(ValueError): gate.full_inputs(values)


@pytest.fixture
def prerequisite(gate, monkeypatch, tmp_path):
    source = dict(native_parameter_identities={}, inference_source_identity={}, body_assets={}, decoder_identity={})
    prior = dict(stage=gate.probe.STAGE,status="pass",phase="complete",producer_revision=gate.PROBE_REVISION,
        script_sha256=gate.PROBE_SCRIPT_SHA,image_id=gate.probe.IMAGE,input_track="track_1",network="none",
        ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],new_identity_source=True,accuracy_verified=False,
        adoption_authorized=False,submission_produced=False,original_conversion_rerun=False,old_target_read=False,
        original_native_geometry_recovered=False,historical_target_recovered=False,strict_reproducibility_verified=False,
        full_frame_fit_verified=False,solver_calls=0,alignment_calls=0,frames=12,original_frames=790,
        frame_indices=list(gate.probe.INDICES),identity_selection_rule="native_frame_zero_protocol_order_not_error",
        shared_identity_frame_index=0,settings=copy.deepcopy(gate.SETTINGS),actual_direct_head_calls=12,
        actual_native_vertices_only_calls=1,actual_official_reference_calls=1,actual_total_native_head_calls=13,
        source_bindings_runtime_verified=True,direct_representation_fidelity_verified=True,
        original_native_parameters_unchanged=True,predictions_frozen_before_replay=True,
        native_parameter_identities={},inference_source_identity={},body_assets={},original_decoder_identity={},
        native_max_point_mm_bound=.01,reference_mean_mm_bound=2.,reference_max_point_is_diagnostic_only=True,
        reference_precision="float32",reference_residual_dtype="float64",reference_model_chunk=16,
        official_converter_sha256=gate.probe.convert.CONVERTER_SHA256,
        reference_model_sha256=gate.probe.convert.REFERENCE_MODEL_SHA256,sealed_report_sha256=gate.probe.SEALED_SHA,
        failed_probe_report_sha256=gate.probe.FAILED_SHA,failed_reseal_report_sha256=gate.probe.RESEAL_SHA,
        runtime_helper_sha256={name:gate.sha256(Path(gate.probe.__file__).with_name(name)) for name in
            ("cari_target_reseal.py","cari_identity_probe_runtime.py","cari_converter.py")},
        native_vs_direct={"per_frame_mean_mm":[.001]*12,"max_point_mm":.002},
        reference_vs_direct={"per_frame_mean_mm":[.0002]*12,"max_point_mm":.0015})
    path = tmp_path/"outputs/episode_000000/cari_direct_shared_probe_refined_v1/report.json";path.parent.mkdir(parents=True)
    def write():
        path.write_text(json.dumps(prior));monkeypatch.setattr(gate,"PROBE_SHA",gate.sha256(path))
    write()
    return tmp_path, source, prior, path, write


def test_actual_prerequisite_source_and_replay_contract(gate, prerequisite):
    root,source,prior,path,_=prerequisite
    result,frozen=gate.prerequisite(root,source)
    assert result==prior and frozen[0][0]==path and len(frozen)==2
    assert gate.sha256(Path(gate.probe.__file__))==gate.PROBE_SCRIPT_SHA


@pytest.mark.parametrize("fault", ["status","revision","script","gt","oracle","calls","coverage","params",
    "settings","identityrule","fidelity","helper","nativegate","referencegate","precision","model","oldclaim","nan"])
def test_prerequisite_rejects_unknown_or_failing_probe(gate, prerequisite, fault):
    root,source,prior,path,write=prerequisite
    key,value={"status":("status","fail"),"revision":("producer_revision","0"*40),"script":("script_sha256","0"*64),
        "gt":("ground_truth_used",True),"oracle":("oracle_modes",["truth"]),"calls":("actual_direct_head_calls",11),
        "coverage":("frame_indices",list(range(12))),"params":("native_parameter_identities",{"bad":"hash"}),
        "settings":("settings",{}),"identityrule":("identity_selection_rule","best_error"),
        "fidelity":("direct_representation_fidelity_verified",False),"helper":("runtime_helper_sha256",{}),
        "precision":("reference_precision","float64"),"model":("reference_model_sha256","0"*64),
        "oldclaim":("historical_target_recovered",True)}.get(fault,(None,None))
    if key:prior[key]=value
    elif fault=="nativegate":prior["native_vs_direct"]["max_point_mm"] = .0101
    elif fault=="referencegate":prior["reference_vs_direct"]={"per_frame_mean_mm":[2.01]*12,"max_point_mm":3.}
    else:prior["reference_vs_direct"]["per_frame_mean_mm"][0]=np.nan
    write()
    with pytest.raises(ValueError):gate.prerequisite(root,source)


def test_full_predictions_frozen_before_replay_nooverwrite(gate, monkeypatch, tmp_path):
    monkeypatch.setattr(gate.probe,"VERTICES",2)
    shared=gate.full_inputs(params(gate));target=np.zeros((790,2,3),np.float32);controls=np.zeros((790,204),np.float32)
    loaded,saved,files=gate.freeze(tmp_path,target,controls,shared)
    assert loaded.shape==(790,2,3) and not loaded.flags.writeable and len(files)==2
    assert saved["pose"].shape==(790,136) and saved["scales"].shape==(68,) and saved["shape"].shape==(45,)
    assert np.array_equal(saved["frame_index"],np.arange(790)) and saved["expression"].shape==(790,72)
    assert all(not path.stat().st_mode&0o222 for path,_ in files)
    with pytest.raises(FileExistsError):gate.freeze(tmp_path,target,controls,shared)


@pytest.mark.parametrize("fault",["scale","signedzero","coverage","dtype","nan","expression","shared"])
def test_freeze_all790_strict_constant_identity(gate,monkeypatch,tmp_path,fault):
    monkeypatch.setattr(gate.probe,"VERTICES",2)
    shared=gate.full_inputs(params(gate));target=np.zeros((790,2,3),np.float32);controls=np.zeros((790,204),np.float32)
    if fault=="scale":controls[789,136]=1
    elif fault=="signedzero":controls[789,136]=-0.
    elif fault=="coverage":target=target[:-1]
    elif fault=="dtype":controls=controls.astype(np.float64)
    elif fault=="nan":target[0,0,0]=np.nan
    elif fault=="expression":shared["mhr_face"][9,0]=1
    else:shared["mhr_shape"][7,0]=1
    with pytest.raises(ValueError):gate.freeze(tmp_path,target,controls,shared)
    assert not list(tmp_path.iterdir())


def test_chunk_residuals_keep_all_frames_and_units(gate):
    report={"native_vs_direct":{"per_frame_mean_mm":[],"max_point_mm":0.}}
    for count in (16,6):
        target=np.zeros((count,2,3),np.float32);recovered=target.astype(np.float64);recovered[...,0]=.000001
        gate.append_residual(report,"native_vs_direct",target,recovered,"m")
    assert len(report["native_vs_direct"]["per_frame_mean_mm"])==22
    assert report["native_vs_direct"]["max_point_mm"]==pytest.approx(.001)


def test_no_globalpatch_solver_conversion_or_target_reuse(gate):
    source=Path(gate.__file__).read_text();tree=ast.parse(source)
    attrs={node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)}
    assert not {"convert","lm_joint","lm_pose","decode_target","frozen_target"}&attrs
    assert "setattr(" not in source and "probe.INDICES =" not in source
    assert "global_trans=trans*context.flip" in source and "return_model_params=True" in source
    assert source.index("freeze(out, target, controls, shared)")<source.index("layer.mhr_forward_vertices(tensors(selection))")
    assert '"old_target_read": False' in source and '"accuracy_verified": False' in source
    assert gate.BUDGET==300
    wrapper=Path(gate.__file__).with_name("run_cari_direct_shared_full.sh");subprocess.run(["bash","-n",str(wrapper)],check=True)
    assert "--network none" in wrapper.read_text() and "--memory 32g" in wrapper.read_text() and "303s" in wrapper.read_text()
