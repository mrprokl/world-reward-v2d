"""Pure J3 public firewall and arrays, not model/media/GPU execution proof."""
import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def infer(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra)); monkeypatch.syspath_prepend(str(infra.parent / "src"))
    spec = importlib.util.spec_from_file_location("test_joint_affine_infer_module", infra / "joint_affine_infer.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def public(infer, tmp_path):
    base = tmp_path / "validation/joint_affine_rgb_v1"
    inputs, masks = base / "inputs", base / "automatic_masks"
    inputs.mkdir(parents=True); masks.mkdir()
    images, records = [], []
    for clip in range(3):
        for frame in range(6):
            file = f"clip_{clip:02d}_frame_{frame:03d}.png"; stem = Path(file).stem
            (inputs/file).write_bytes(f"own tiny RGB {clip} {frame}".encode())
            image = {"file": file, "sha256": infer.sha256(inputs/file), "width": 1024, "height": 768}
            record = {"file": file, "clip_index": clip, "frame_index": frame, "rgb_sha256": image["sha256"]}
            for kind in ("human", "object"):
                name = f"{stem}_{kind}.png"; (masks/name).write_bytes(f"own binary {kind} {clip} {frame}".encode())
                record[kind+"_mask_file"] = name; record[kind+"_mask_sha256"] = infer.sha256(masks/name)
            images.append(image); records.append(record)
    (inputs/"manifest.json").write_text(json.dumps({"schema": infer.SCHEMA, "images": images}))
    report = {"stage": "public_joint_affine_rgb_automatic_masks", "status": "pass", "frames": 18,
              "private_truth_read": False, "challenge_inputs_used": False, "ground_truth_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "human_query": "person.", "object_query": "bottle.",
              "input_manifest_sha256": infer.sha256(inputs/"manifest.json"), "records": records}
    (masks/"report.json").write_text(json.dumps(report))
    return tmp_path, inputs, masks


def test_full_public_timeline_exact_identities_no_private_read(infer, public):
    root, inputs, masks = public
    (root/"validation/joint_affine_rgb_v1/eval_private").mkdir()
    records, hashes = infer.public_inputs(root)
    assert len(records) == 18
    assert [(r["clip_index"], r["frame_index"]) for r in records] == [(c,f) for c in range(3) for f in range(6)]
    assert hashes["public_inputs_sha"] == infer.sha256(inputs/"manifest.json")
    assert all(r["image_path"].parent == inputs for r in records)
    assert all(r["human_mask_path"].parent == masks for r in records)


@pytest.mark.parametrize("failure", ["schema", "private_field", "frames", "traversal", "grid_bool", "mask_order", "gt", "false_integer",
                                     "query", "manifest_hash", "rgb_bytes", "mask_bytes", "extra_input", "extra_mask", "symlink"])
def test_public_firewall_rejects_invalid_binding(infer, public, failure):
    root, inputs, masks = public
    manifest = json.loads((inputs/"manifest.json").read_text()); report = json.loads((masks/"report.json").read_text())
    if failure == "schema": manifest["schema"] = "world-reward-joint-rgb-v1"
    elif failure == "private_field": manifest["images"][0]["camera_K"] = [1]
    elif failure == "frames": manifest["images"].pop()
    elif failure == "traversal": manifest["images"][0]["file"] = "../eval_private/truth.npz"
    elif failure == "grid_bool": manifest["images"][0]["width"] = True
    elif failure == "mask_order": report["records"][0]["frame_index"] = 1
    elif failure == "gt": report["ground_truth_used"] = True
    elif failure == "false_integer": report["private_truth_read"] = 0
    elif failure == "query": report["object_query"] = "unknown."
    elif failure == "manifest_hash": report["input_manifest_sha256"] = "0"*64
    elif failure == "rgb_bytes": (inputs/manifest["images"][0]["file"]).write_bytes(b"changed")
    elif failure == "mask_bytes": (masks/report["records"][0]["human_mask_file"]).write_bytes(b"changed")
    elif failure == "extra_input": (inputs/"truth.npz").write_bytes(b"forbidden")
    elif failure == "extra_mask": (masks/"camera.json").write_bytes(b"forbidden")
    elif failure == "symlink":
        p=inputs/manifest["images"][0]["file"]; p.rename(inputs/"held.bin"); p.symlink_to(inputs/"held.bin")
    (inputs/"manifest.json").write_text(json.dumps(manifest))
    if failure != "manifest_hash": report["input_manifest_sha256"] = infer.sha256(inputs/"manifest.json")
    (masks/"report.json").write_text(json.dumps(report))
    with pytest.raises(ValueError): infer.public_inputs(root)


@pytest.fixture
def arrays(infer, monkeypatch):
    monkeypatch.setattr(infer, "WIDTH", 4); monkeypatch.setattr(infer, "HEIGHT", 3)
    monkeypatch.setattr(infer, "VERTICES", 4); monkeypatch.setattr(infer, "FACES", 4)
    K = np.array([[8.,0.,2.],[0.,8.,1.5],[0.,0.,1.]], np.float64); monkeypatch.setattr(infer,"CAMERA_K",K)
    yy,xx=np.indices((3,4)); depth=np.full((3,4),3.,np.float32)
    points=np.stack(((xx+.5-K[0,2])*depth/8,(yy+.5-K[1,2])*depth/8,depth),-1).astype(np.float32)
    validity=np.ones((3,4),bool); validity[0,0]=False; depth[0,0]=np.nan; points[0,0]=np.nan
    silhouette=np.zeros((3,4),bool); silhouette[1:,1:]=True
    rendered=np.full((3,4),np.nan,np.float32); rendered[silhouette]=3
    result={"raw_depth":depth,"raw_points":points,"validity":validity,"silhouette":silhouette,
            "rendered_depth":rendered,"human_mask":silhouette.copy(),"object_mask":silhouette.copy(),
            "human_vertices_camera_m":np.array([[0,0,3],[.1,0,3],[0,.1,3],[0,0,3.1]],np.float32),
            "human_faces":np.array([[0,2,1],[0,1,3],[1,2,3],[2,0,3]],np.int64),
            "K":K.copy(),"clip_index":np.asarray(0,np.int64),"frame_index":np.asarray(5,np.int64)}
    return result, np.diag([1/4,1/3,1])@K


def test_original_invalid_values_preserved_no_fit_and_exact_array_abi(infer, arrays):
    values,normalized=arrays; before=copy.deepcopy(values)
    report=infer.validate_arrays(values,normalized)
    assert report["excluded_pixels"]==1 and report["valid_pixels"]==11
    assert set(values)==infer.ARRAY_KEYS
    for key in values: np.testing.assert_array_equal(values[key],before[key])
    assert infer.validate_observation(values,0,5)["valid_pixels"]==11
    with pytest.raises(ValueError): infer.validate_observation(values,0,4)
    with pytest.raises(ValueError): infer.validate_observation(values,False,5)


@pytest.mark.parametrize("failure", ["extra", "missing", "depth64", "points_shape", "validity_int", "invalid_valid", "wrong_ray",
                                     "K", "render_outside", "render_inside", "vertices", "faces", "repeated_face", "frame_bool", "clip_range", "masked"])
def test_observation_arrays_fail_without_repair(infer, arrays, failure):
    values,normalized=arrays
    if failure=="extra": values["shared_scale"]=np.array(1.)
    elif failure=="missing": del values["raw_depth"]
    elif failure=="depth64": values["raw_depth"]=values["raw_depth"].astype(np.float64)
    elif failure=="points_shape": values["raw_points"]=values["raw_points"][:2]
    elif failure=="validity_int": values["validity"]=values["validity"].astype(np.uint8)
    elif failure=="invalid_valid": values["validity"][0,0]=True
    elif failure=="wrong_ray": values["raw_points"][1,1,0]+=.01
    elif failure=="K": values["K"][0,0]=9
    elif failure=="render_outside": values["rendered_depth"][0,0]=0
    elif failure=="render_inside": values["rendered_depth"][1,1]=np.nan
    elif failure=="vertices": values["human_vertices_camera_m"][0,2]=0
    elif failure=="faces": values["human_faces"][0,0]=4
    elif failure=="repeated_face": values["human_faces"][0,0]=values["human_faces"][0,1]
    elif failure=="frame_bool": values["frame_index"]=np.asarray(True)
    elif failure=="clip_range": values["clip_index"]=np.asarray(3,np.int64)
    elif failure=="masked": values["raw_points"]=np.ma.array(values["raw_points"],mask=False)
    with pytest.raises(ValueError): infer.validate_arrays(values,normalized)


def test_wrapper_syntax_private_firewall_and_no_runtime_locally(infer):
    wrapper=Path(infer.__file__).with_name("run_joint_affine_infer.sh")
    result=subprocess.run(["bash","-n",str(wrapper)],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    source=wrapper.read_text()
    assert "eval_private" not in source and "joint_rgb_v1" not in source
    assert "--network none" in source and "--memory 32g --cpus 4" in source
    assert "603s" in source and "predictions_v1" in source
    assert 'src=$BASE/inputs,dst=$BASE/inputs,readonly' in source
    assert 'src=$BASE/automatic_masks,dst=$BASE/automatic_masks,readonly' in source
    assert source.count('src=$OUT,dst=$OUT')==1


def test_main_unknown_arguments_rejected_before_linux_or_models(infer):
    with pytest.raises(SystemExit) as result: infer.main(["--fit-scale"])
    assert result.value.code==2
