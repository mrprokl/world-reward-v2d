"""Tiny first-target lineage fixtures; no full mesh arrays, models, or GPU."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def replay(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("own_frozen_target_replay", infra / "cari_frozen_target_replay.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


@pytest.fixture
def frozen(replay, monkeypatch, tmp_path):
    directory = tmp_path / "outputs/episode_000000/cari_target_reseal_refined_v1"; directory.mkdir(parents=True)
    target = np.arange(18, dtype=np.float32).reshape(2, 3, 3)
    path = directory / "firsttarget_f32.npy"; np.save(path, target); path.chmod(0o444)
    identity = replay.convert.canonical_array_identity(target)
    monkeypatch.setattr(replay, "TARGET_SHAPE", target.shape)
    monkeypatch.setattr(replay, "FIRST_IDENTITY_SHA", identity["sha256"])
    # Small dimensions retain every production validation field, no large local allocation.
    original = dict(native_parameter_identities={"x": "fixture"}, inference_source_identity={"x": "fixture"}, body_assets={"x": "fixture"})
    workers = []
    common = dict(stage=replay.RESEAL_STAGE, status="pass", phase="complete", settings=copy.deepcopy(replay.SETTINGS),
        native_full_frame_decode_verified=True, producer_revision=replay.RESEAL_REVISION, script_sha256="a"*64,
        image_id="sha256:"+"b"*64, sealed_report_sha256=replay.SEALED_SHA, failed_probe_report_sha256=replay.FAILED_SHA,
        torch="fixture", CUDA="fixture", cudnn=1, GPU="fixture", **original)
    for i in range(2):
        worker = dict(copy.deepcopy(common), worker_index=i, worker_pid=100+i,
            target_file="firsttarget_f32.npy" if i == 0 else "secondtarget_f32.npy",
            target_file_sha256=replay.sha256(path), target_file_bytes=path.stat().st_size,
            target_vertices_identity=copy.deepcopy(identity))
        if i: worker["target_vertices_identity"]["sha256"] = "c"*64
        workers.append(worker)
    receipt = dict(stage=replay.RESEAL_STAGE, status="fail", phase="paired_target_comparison",
        error="Fresh worker source/version/settings/target identity differs", producer_revision=replay.RESEAL_REVISION,
        network="none", input_track="track_1", ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        research_only=True, submission_produced=False, adoption_authorized=False, old_target_bitexact_recovered=False,
        old_target_array_available=False, original_conversion_rerun=False, solver_calls=0, second_target_removed=False,
        sealed_report_sha256=replay.SEALED_SHA, failed_probe_report_sha256=replay.FAILED_SHA,
        script_sha256=common["script_sha256"], image_id=common["image_id"], workers=workers, comparison={"bitexact": False})
    def write():
        hashes = []
        for i, worker in enumerate(workers):
            p = directory / f"worker_{i}.json"; p.write_text(json.dumps(worker)); hashes.append(replay.sha256(p))
        receipt["worker_report_sha256"] = hashes
        p = directory / "report.json"; p.write_text(json.dumps(receipt)); monkeypatch.setattr(replay, "RESEAL_SHA", replay.sha256(p))
    # Production validates full canonical dimensions; substitute only that tiny fixture check.
    original_fields = replay.require_fields
    def fixture_fields(value, expected, description):
        if description == "Complete canonical FP32 target identity required":
            expected = dict(expected, bytes=target.nbytes)
        original_fields(value, expected, description)
    monkeypatch.setattr(replay, "require_fields", fixture_fields)
    write()
    return tmp_path, directory, original, receipt, workers, write


def test_exact_first_saved_target_not_error_selection_and_no_second_load(replay, frozen):
    root, directory, original, receipt, workers, _ = frozen
    target, actual, paths = replay.frozen_target(root, original)
    assert isinstance(target, np.memmap) and not target.flags.writeable
    assert actual == receipt and len(paths) == 4
    assert not (directory / "secondtarget_f32.npy").exists()  # Never needed for this diagnostic.
    np.testing.assert_array_equal(target, np.arange(18, dtype=np.float32).reshape(2, 3, 3))


@pytest.mark.parametrize("fault", ["receipt_hash", "status", "phase", "error", "revision", "gt", "oracle", "strictclaim",
    "samepid", "workerstatus", "settings", "version", "params", "bodyassets", "firstsha", "same_targetsha",
    "workerfile", "targetfile", "writable", "targetchanged", "symlink", "filename"])
def test_failed_reseal_is_never_relabeled_or_unknown_target_accepted(replay, frozen, fault):
    root, directory, original, receipt, workers, write = frozen
    if fault == "receipt_hash": receipt["unrecorded"] = True
    elif fault == "status": receipt["status"] = "pass"
    elif fault == "phase": receipt["phase"] = "complete"
    elif fault == "error": receipt["error"] = "different"
    elif fault == "revision": receipt["producer_revision"] = "0"*40
    elif fault == "gt": receipt["ground_truth_used"] = True
    elif fault == "oracle": receipt["oracle_modes"] = ["human"]
    elif fault == "strictclaim": receipt["old_target_bitexact_recovered"] = True
    elif fault == "samepid": workers[1]["worker_pid"] = workers[0]["worker_pid"]
    elif fault == "workerstatus": workers[1]["status"] = "fail"
    elif fault == "settings": workers[1]["settings"]["TF32"] = True
    elif fault == "version": workers[1]["torch"] = "different"
    elif fault == "params": workers[0]["native_parameter_identities"] = {}
    elif fault == "bodyassets": workers[1]["body_assets"] = {}
    elif fault == "firstsha": workers[0]["target_vertices_identity"]["sha256"] = "d"*64
    elif fault == "same_targetsha": workers[1]["target_vertices_identity"] = workers[0]["target_vertices_identity"].copy()
    elif fault == "filename": workers[0]["target_file"] = "../../unknown.npy"
    elif fault == "writable": (directory / "firsttarget_f32.npy").chmod(0o644)
    elif fault == "targetchanged":
        p = directory / "firsttarget_f32.npy"; p.chmod(0o644); np.save(p, np.zeros((2,3,3),np.float32)); p.chmod(0o444)
    elif fault == "symlink":
        p = directory / "firsttarget_f32.npy"; p.rename(directory / "elsewhere.npy"); p.symlink_to(directory / "elsewhere.npy")
    write()
    if fault == "receipt_hash":
        monkey_sha = "0"*64
        replay.RESEAL_SHA = monkey_sha
    elif fault == "workerfile": (directory / "worker_0.json").write_text("{}")
    elif fault == "targetfile": workers[0]["target_file_sha256"] = "0"*64; write()
    with pytest.raises((ValueError, RuntimeError)): replay.frozen_target(root, original)


def test_unknown_args_fail_before_remote_runtime(replay):
    with pytest.raises(SystemExit): replay.main(["--allow-new-target"])


def test_offline_wrapper_and_zero_decode_solve_source(replay):
    source = Path(replay.__file__).read_text(); wrapper = Path(replay.__file__).with_name("run_cari_frozen_target_replay.sh")
    subprocess.run(["bash", "-n", str(wrapper)], check=True)
    text = wrapper.read_text()
    assert "--network none" in text and "--memory 32g" in text and "183s" in text
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text and "src=$ROOT/outputs,dst=$ROOT/outputs,readonly" in text
    assert '"strict_reproducibility_verified": False' in source and '"historical_target_recovered": False' in source
    assert '"selection_rule": "first_worker_protocol_order_not_error"' in source
    assert '"actual_native_decode_calls": 0' in source and '"solver_calls": 0' in source
    assert "decode_mhr_vertices_numpy(" not in source and "lm_joint(" not in source and "converter.convert(" not in source
    assert "validate_workers(" not in source
