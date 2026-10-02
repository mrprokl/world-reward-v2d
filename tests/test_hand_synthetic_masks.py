"""RGB-only manifest, detector abstention, source and mask contracts; no models."""

import copy
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from world_reward.prompt_selection import BoxDetection


INFRA = Path(__file__).parents[1] / "infra"
spec = importlib.util.spec_from_file_location("hand_masks_tests", INFRA / "hand_synthetic_masks.py")
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def inputs(tmp_path):
    records = []
    for index in range(6):
        path = tmp_path / f"case_{index:03d}.png"
        path.write_bytes(f"tiny-fixture-{index}".encode())
        records.append({"file": path.name, "sha256": gate.identity(path)["sha256"], "width": 8, "height": 6})
    manifest = {"schema": gate.SCHEMA, "images": records}
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, manifest


def test_public_only_original_six_ordered_hashes(tmp_path):
    path, manifest = inputs(tmp_path)
    saved = copy.deepcopy(manifest)
    records, receipt = gate.validate_inputs(tmp_path)
    assert [r["frame_index"] for r in records] == list(range(6))
    assert [r["file"] for r in records] == [f"case_{i:03d}.png" for i in range(6)]
    assert receipt == gate.identity(path) and manifest == saved
    assert all(r["path"].parent == tmp_path for r in records)


@pytest.mark.parametrize("kind", ["schema", "private", "private_record", "order", "duplicate", "count", "sha", "grid_bool", "grid_zero", "grid_float", "traversal"])
def test_invalid_or_private_input_manifest_rejected(tmp_path, kind):
    path, data = inputs(tmp_path)
    if kind == "schema": data["schema"] = "unknown"
    elif kind == "private": data["poses"] = []
    elif kind == "private_record": data["images"][0]["mask"] = "oracle.png"
    elif kind == "order": data["images"].reverse()
    elif kind == "duplicate": data["images"][1] = data["images"][0]
    elif kind == "count": data["images"].pop()
    elif kind == "sha": data["images"][0]["sha256"] = "f" * 64
    elif kind == "grid_bool": data["images"][0]["height"] = True
    elif kind == "grid_zero": data["images"][0]["width"] = 0
    elif kind == "grid_float": data["images"][0]["width"] = 8.0
    else: data["images"][0]["file"] = "../case_000.png"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        gate.validate_inputs(tmp_path)


def test_rgb_tamper_and_symlink_rejected(tmp_path):
    inputs(tmp_path)
    path = tmp_path / "case_000.png"
    path.write_bytes(b"tampered")
    with pytest.raises(ValueError): gate.validate_inputs(tmp_path)
    other = tmp_path / "elsewhere"
    other.write_bytes(b"tampered")
    path.unlink(); path.symlink_to(other)
    with pytest.raises(ValueError, match="nonsymlink"): gate.validate_inputs(tmp_path)


def test_nms_duplicates_do_not_invent_box_and_highest_unique_selected():
    best = BoxDetection((1., 1., 7., 5.), .8)
    chosen = gate.select_person([BoxDetection((1.1, 1., 7., 5.), .79), best], 8, 6)
    assert chosen == best


@pytest.mark.parametrize("boxes", [[], [BoxDetection((0., 0., 2., 3.), .299)],
    [BoxDetection((0., 0., 2., 3.), .8), BoxDetection((5., 1., 8., 5.), .75)],
    [BoxDetection((-1., 0., 2., 3.), .99)], [BoxDetection((0., 0., 2., 3.), float("nan"))]])
def test_missing_invalid_or_distinct_near_tied_person_abstains(boxes):
    with pytest.raises(ValueError, match="abstained"):
        gate.select_person(boxes, 8, 6)


def test_binary_original_grid_no_mask_warp_or_input_mutation():
    masks = np.array([[[True, False], [False, True]]])
    saved = masks.copy()
    result = gate.binary_mask(masks, np.array([.7]), 2, 2)
    np.testing.assert_array_equal(result, [[255, 0], [0, 255]])
    np.testing.assert_array_equal(masks, saved)
    assert result.dtype == np.uint8


@pytest.mark.parametrize("kind", ["grid", "count", "score", "nonfinite", "empty", "masked"])
def test_invalid_sam2_output_abstains(kind):
    masks = np.ones((1, 2, 2)); scores = np.array([.8])
    if kind == "grid": masks = masks[:, :, :1]
    elif kind == "count": masks = np.repeat(masks, 3, axis=0)
    elif kind == "score": scores[0] = np.nan
    elif kind == "nonfinite": masks[0, 0, 0] = np.inf
    elif kind == "empty": masks[:] = 0
    else: masks = np.ma.array(masks)
    with pytest.raises(RuntimeError): gate.binary_mask(masks, scores, 2, 2)


def source(tmp_path):
    for name in ("build_sam.py", "sam2_image_predictor.py", "modeling/sam2_base.py", "utils/transforms.py", "extra.py",
                 "configs/sam2.1/sam2.1_hiera_l.yaml"):
        path = tmp_path / name
        path.parent.mkdir(exist_ok=True, parents=True)
        path.write_text(name)
    return {"url": "https://github.com/facebookresearch/sam2.git", "vcs_info": {"vcs": "git", "commit_id": "a" * 40}}


def test_actual_installed_source_vcs_full_python_aggregate_not_claimed_upstream_pin(tmp_path):
    direct = source(tmp_path)
    report = gate.sam2_source_identity(tmp_path, direct)
    assert report["vcs"] == direct and report["python_files"] == 5
    assert report["prior_upstream_source_pin_verified"] is False
    assert report["full_import_license_closure_verified"] is False
    (tmp_path / "extra.py").write_text("changed")
    assert gate.sam2_source_identity(tmp_path, direct)["python_source_sha256"] != report["python_source_sha256"]


@pytest.mark.parametrize("kind", ["missing", "wrong_repo", "wrong_vcs", "missing_commit", "short_commit", "inventory"])
def test_unverified_sam2_source_fails_before_model(kind, tmp_path):
    direct = source(tmp_path)
    if kind == "missing": direct = None
    elif kind == "wrong_repo": direct["url"] = "https://github.com/unknown/sam2.git"
    elif kind == "wrong_vcs": direct["vcs_info"]["vcs"] = "svn"
    elif kind == "missing_commit": direct["vcs_info"].pop("commit_id")
    elif kind == "short_commit": direct["vcs_info"]["commit_id"] = "main"
    else: (tmp_path / "build_sam.py").unlink()
    with pytest.raises(RuntimeError): gate.sam2_source_identity(tmp_path, direct)


def test_frozen_asset_metadata_independent_revision_weight_sha():
    assert gate.DETECTOR_REVISION == "12bdfa3120f3e7ec7b434d90674b3396eccf88eb"
    assert gate.SAM2_REVISION == "665f8e2ad61cf5f53d65644ff27c8ee525124610"
    assert len(gate.ASSETS) == 9
    for digest, size in gate.ASSETS.values():
        assert len(digest) == 64 and type(size) is int and size > 0


def test_wrapper_only_public_inputs_output_and_integrity_receipts_no_private_mount():
    wrapper = (INFRA / "run_hand_synthetic_masks.sh").read_text()
    assert 'world-reward/grounding:0.1' in wrapper and "--network none" in wrapper
    assert 'src=$INPUT,dst=$INPUT,readonly' in wrapper and 'src=$OUTPUT,dst=$OUTPUT' in wrapper
    assert 'src=$ROOT/validation,dst=' not in wrapper and 'eval_private' not in wrapper
    assert 'src=$ROOT/results,dst=$ROOT/results' not in wrapper and 'src=$ROOT/data' not in wrapper
    assert 'mkdir "$OUTPUT"' in wrapper and 'chown scenesmith:scenesmith "$OUTPUT"' in wrapper and '123s docker' in wrapper
    result = subprocess.run(["bash", str(INFRA / "run_hand_synthetic_masks.sh"), "--unknown"], capture_output=True, text=True)
    assert result.returncode == 2
