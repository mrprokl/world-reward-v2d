"""Small D81 chain/error-contract fixtures, not official GPU solve proof."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def probe(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_frozen_identity", infra / "cari_frozen_identity_probe.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def arrays():
    historical = np.full(790, 2.083, np.float32); historical[1] = 6.28
    precise = historical.astype(np.float64)+1e-5; replayed = precise.astype(np.float32)
    return dict(frame_index=np.arange(790, dtype=np.int64), historical_errors_f32=historical,
        replay_errors_f32=replayed, replay_errors_float64=precise,
        drift_mm=replayed.astype(np.float64)-historical.astype(np.float64))


def test_full790_stored_d81_error_arithmetic_and_original_bytes(probe):
    values = arrays(); result = probe.error_arrays(values, values["historical_errors_f32"].copy())
    assert result["historical_error_agreement"] is True and result["drift_failure_frames"] == []
    assert result["historical_mean_mm"] > 2.  # Agreement is not conversion-fidelity approval.


@pytest.mark.parametrize("fault", ["extra", "coverage", "indexdtype", "histdtype", "precdtype", "driftdtype", "nan",
    "negative", "originalbytes", "cast", "drift", "disagreement"])
def test_error_arrays_fail_closed(probe, fault):
    values = arrays(); original = values["historical_errors_f32"].copy()
    if fault == "extra": values["truth"] = np.zeros(1)
    elif fault == "coverage": values["frame_index"][-1] = 788
    elif fault == "indexdtype": values["frame_index"] = values["frame_index"].astype(np.int32)
    elif fault == "histdtype": values["historical_errors_f32"] = original.astype(np.float64)
    elif fault == "precdtype": values["replay_errors_float64"] = values["replay_errors_float64"].astype(np.float32)
    elif fault == "driftdtype": values["drift_mm"] = values["drift_mm"].astype(np.float32)
    elif fault == "nan": values["replay_errors_float64"][0] = np.nan
    elif fault == "negative": values["replay_errors_float64"][0] = -1
    elif fault == "originalbytes": original[0] += .01
    elif fault == "cast": values["replay_errors_f32"][0] += 1
    elif fault == "drift": values["drift_mm"][0] += .001
    else:
        values["replay_errors_float64"][0] += .001
        values["replay_errors_f32"] = values["replay_errors_float64"].astype(np.float32)
        values["drift_mm"] = values["replay_errors_f32"].astype(np.float64)-original.astype(np.float64)
    with pytest.raises(ValueError): probe.error_arrays(values, original)


@pytest.fixture
def d81(probe, monkeypatch, tmp_path):
    directory = tmp_path / "outputs/episode_000000/cari_frozen_target_replay_refined_v1"; directory.mkdir(parents=True)
    values = arrays(); errorspath = directory / "replay_errors.npz"; np.savez_compressed(errorspath, **values); errorspath.chmod(0o444)
    monkeypatch.setattr(probe, "REPLAY_ERRORS_SHA", probe.sha256(errorspath))
    original = dict(pose=np.zeros((790,136),np.float32), scales=np.zeros(68,np.float32), shape=np.zeros(45,np.float32),
                    per_frame_vertex_error_mm=values["historical_errors_f32"].copy())
    source = dict(provenance={"native_bundle_sha256":"a"*64}, native_vertices_identity={"sha256":"b"*64})
    image = "sha256:"+"c"*64
    target = dict(image_id=image, workers=[dict(target_vertices_identity={"sha256":probe.FIRST_IDENTITY_SHA},
        target_file_sha256=probe.TARGET_FILE_SHA, torch="fixture", CUDA="fixture")])
    report = dict(stage=probe.REPLAY_STAGE, status="pass", phase="complete", producer_revision=probe.REPLAY_REVISION,
        script_sha256=probe.REPLAY_SCRIPT_SHA, image_id=image, network="none", input_track="track_1", ground_truth_used=False,
        hand_labeled_test=False, oracle_modes=[], research_only=True, submission_produced=False, adoption_authorized=False,
        strict_reproducibility_verified=False, historical_target_recovered=False, new_inverse_target=True,
        selection_rule="first_worker_protocol_order_not_error", full_frame_fit_verified=False, actual_native_decode_calls=0,
        solver_calls=0, original_conversion_rerun=False, source_bindings_runtime_verified=True,
        warmstart_numerical_agreement_only=True, original_parameters_unchanged=True,
        sealed_report_sha256=probe.SEALED_SHA, failed_probe_report_sha256=probe.FAILED_SHA, failed_reseal_report_sha256=probe.RESEAL_SHA,
        original_parameter_identities={k:probe.convert.canonical_array_identity(v) for k,v in original.items()},
        historical_native_vertices_identity=source["native_vertices_identity"], target_vertices_identity=target["workers"][0]["target_vertices_identity"],
        target_file_sha256=probe.TARGET_FILE_SHA, provenance={**source["provenance"], "native_vertices_sha256":probe.FIRST_IDENTITY_SHA,
            "historical_native_vertices_sha256":source["native_vertices_identity"]["sha256"]},
        actual_official_reference_replay_calls=1, reference_precision="float32", reference_model_chunk=256,
        reference_residual_dtype="float64", reference_error_storage_dtype="float32", reference_settings=copy.deepcopy(probe.SETTINGS),
        reference_torch="fixture", reference_CUDA="fixture", official_converter_sha256=probe.convert.CONVERTER_SHA256,
        reference_model_sha256=probe.convert.REFERENCE_MODEL_SHA256, replay_errors_sha256=probe.REPLAY_ERRORS_SHA,
        file_inventory=[dict(file=errorspath.name,sha256=probe.REPLAY_ERRORS_SHA,bytes=errorspath.stat().st_size)],
        **probe.error_arrays(values,original["per_frame_vertex_error_mm"]))
    def write():
        path = directory / "report.json"; path.write_text(json.dumps(report)); monkeypatch.setattr(probe,"REPLAY_SHA",probe.sha256(path))
    write()
    return tmp_path, original, source, target, image, report, errorspath, write


def test_actual_d81_pass_is_numerical_agreement_not_adoption(probe, d81):
    root, original, source, target, image, report, _, _ = d81
    actual, paths = probe.replay_receipt(root,original,source,target,image)
    assert actual == report and len(paths)==2
    assert actual["strict_reproducibility_verified"] is False and actual["full_frame_fit_verified"] is False


@pytest.mark.parametrize("fault", ["hash", "status", "revision", "script", "image", "gt", "oracle", "strictclaim",
    "fitclaim", "targetfile", "targetcanonical", "oldsha", "params", "precision", "calls", "settings", "inventory",
    "errorhash", "writable", "errorschanged", "originalchanged", "agreement"])
def test_d81_chain_fails_before_model_import(probe,d81,fault):
    root,original,source,target,image,report,errorspath,write=d81
    if fault=="hash": report["unrecorded"]=True
    elif fault=="status": report["status"]="fail"
    elif fault=="revision": report["producer_revision"]="0"*40
    elif fault=="script": report["script_sha256"]="0"*64
    elif fault=="image": image="sha256:"+"d"*64
    elif fault=="gt": report["ground_truth_used"]=True
    elif fault=="oracle": report["oracle_modes"]=["identity"]
    elif fault=="strictclaim": report["strict_reproducibility_verified"]=True
    elif fault=="fitclaim": report["full_frame_fit_verified"]=True
    elif fault=="targetfile": target["workers"][0]["target_file_sha256"]="0"*64
    elif fault=="targetcanonical": target["workers"][0]["target_vertices_identity"]={"sha256":"0"*64}
    elif fault=="oldsha": report["failed_reseal_report_sha256"]="0"*64
    elif fault=="params": report["original_parameter_identities"]["pose"]={}
    elif fault=="precision": report["reference_precision"]="float64"
    elif fault=="calls": report["actual_official_reference_replay_calls"]=2
    elif fault=="settings": report["reference_settings"]["TF32"]=True
    elif fault=="inventory": report["file_inventory"][0]["bytes"]+=1
    elif fault=="errorhash": report["replay_errors_sha256"]="0"*64
    elif fault=="writable": errorspath.chmod(0o644)
    elif fault=="errorschanged": errorspath.chmod(0o644);errorspath.write_bytes(b"changed");errorspath.chmod(0o444)
    elif fault=="originalchanged": original["pose"][0,0]=1
    else:report["historical_error_agreement"]=False
    write()
    if fault=="hash": probe.REPLAY_SHA="0"*64
    with pytest.raises((ValueError,RuntimeError)):probe.replay_receipt(root,original,source,target,image)


def test_unknownargs_fail_before_runtime(probe):
    with pytest.raises(SystemExit):probe.main(["--skip-replay"])


def test_wrapper_and_runtime_preserve_original_protocol_and_failures(probe):
    source=Path(probe.__file__).read_text();wrapper=Path(probe.__file__).with_name("run_cari_frozen_identity_probe.sh")
    subprocess.run(["bash","-n",str(wrapper)],check=True);text=wrapper.read_text()
    assert "903s" in text and "--network none" in text and "--memory 32g" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and "src=$ROOT/outputs,dst=$ROOT/outputs,readonly" in text
    assert "torch.inference_mode(), torch.jit.optimized_execution(False)" in source
    assert 'calls != {"joint": 1, "pose": 1, "replay": 2}' in source
    assert "decode_target(" not in source and "converter.convert(" not in source and "decode_mhr_vertices_numpy(" not in source
    assert '"strict_reproducibility_verified": False' in source and '"historical_target_recovered": False' in source
    assert "identity_probe(original, params, target" in source
    assert "cari_identity_probe_frozen_refined_v1" in text
