"""Full501 parameter-only tiny fixtures; no models, geometry, media or GPU."""

import ast
import copy
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from world_reward.shared_identity import (
    HISTORICAL_GEOMETRY_CHECKS, IDENTITY_BLOCKS, NATIVE_PARAMETER_DIMS,
    share_first_frame_identity, shared_initializer_metadata, validate_native_parameters,
)


def native_parameters(count=96):
    values = {name: np.zeros((count, dimension), np.float32) for name, dimension in NATIVE_PARAMETER_DIMS.items()}
    values["mhr_global_rot6d"][:] = [1, 0, 0, 1, 0, 0]
    values["mhr_trans"][:, 2] = np.linspace(1, 2, count, dtype=np.float32)
    for name in ("mhr_shape", "mhr_scale", "mhr_hand", "mhr_body_pose_cont"):
        values[name][:] = np.arange(count, dtype=np.float32)[:, None] / 100
    return values


def metadata():
    return dict(
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        mhr_geometry_forward_verified=True, human_identity_clip_constant=False,
        native_roundtrip_max_error_m={"vertices": 0.}, translation_once_max_error_m=0.,
        projection_max_error_px=.01, source={"sha256": "a" * 64, "nested": [1, 2]},
        camera_intrinsics=[[1920, 0, 768], [0, 1920, 576], [0, 0, 1]],
        identity_policy="original_per_frame_initializer_unchanged_not_final_submission",
        quality_verified=True, adoption_authorized=True, submission_eligible=True,
        challenge_performance_verified=True,
    )


def ast_parameter_dims(path):
    tree = ast.parse(path.read_text())
    return next(ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "PARAMETER_DIMS" for target in node.targets))


def test_single_public_canonical_dimensions_match_existing_audited_native_abi():
    root = Path(__file__).resolve().parents[1]
    for name in ("cari_converter.py", "cari_body_adapter.py", "cari96_inputs.py"):
        assert dict(NATIVE_PARAMETER_DIMS) == ast_parameter_dims(root / "infra" / name)
    with pytest.raises(TypeError):
        NATIVE_PARAMETER_DIMS["mhr_shape"] = 46


@pytest.mark.parametrize("count", [96, 97, 501])
def test_full_original_identity_is_copied_before_geometry_and_other5blocks_are_exact(count):
    original = native_parameters(count)
    before = {name: value.tobytes() for name, value in original.items()}
    shared = share_first_frame_identity(MappingProxyType(original), np.int64(count))
    assert set(shared) == set(NATIVE_PARAMETER_DIMS)
    validate_native_parameters(shared, count, require_shared_identity=True)
    assert before == {name: value.tobytes() for name, value in original.items()}
    for name in NATIVE_PARAMETER_DIMS:
        assert shared[name].dtype == np.float32 and shared[name].shape == original[name].shape
        assert shared[name].flags.owndata and shared[name].flags.c_contiguous
        assert not any(np.shares_memory(shared[name], other) for other in original.values())
        if name in IDENTITY_BLOCKS:
            assert all(row.tobytes() == original[name][0].tobytes() for row in shared[name])
        else:
            assert shared[name].tobytes() == before[name]
    for index, name in enumerate(NATIVE_PARAMETER_DIMS):
        assert not any(np.shares_memory(shared[name], shared[other]) for other in tuple(NATIVE_PARAMETER_DIMS)[index + 1:])
    assert not {"mhr_joints", "mhr_keypoints", "vertices"}.intersection(shared)


def test_all_source_blocks_may_alias_without_shared_outputs_aliasing_anything():
    storage = np.zeros((96, 260), np.float32)
    values = {name: storage[:, :dimension] for name, dimension in NATIVE_PARAMETER_DIMS.items()}
    # All blocks share storage; the first6 entries satisfy the root and trans.
    storage[:, :6] = [1, 0, 1, 1, 0, 0]
    # Face must be zero, so use its own valid original storage.
    values["mhr_face"] = np.zeros((96, 72), np.float32)
    shared = share_first_frame_identity(values, 96)
    assert all(not np.shares_memory(a, b) for a in shared.values() for b in values.values())


def test_noncontiguous_and_readonly_sources_keep_logical_bytes_and_are_not_chmoded():
    original = {name: np.asfortranarray(value) for name, value in native_parameters().items()}
    for value in original.values(): value.flags.writeable = False
    shared = share_first_frame_identity(original, 96)
    for name in NATIVE_PARAMETER_DIMS:
        assert not original[name].flags.writeable
        assert shared[name].flags.writeable and shared[name].flags.c_contiguous
        if name not in IDENTITY_BLOCKS: assert shared[name].tobytes() == original[name].tobytes()


def test_signed_zero_identity_bytes_are_preserved_and_numerical_equality_is_not_constancy():
    original = native_parameters()
    original["mhr_shape"][0, 0] = -0.
    original["mhr_scale"][0, 1] = -0.
    original["mhr_face"][:] = -0.
    shared = share_first_frame_identity(original, 96)
    assert np.signbit(shared["mhr_shape"][:, 0]).all()
    assert np.signbit(shared["mhr_scale"][:, 1]).all()
    assert shared["mhr_face"].tobytes() == original["mhr_face"].tobytes()
    shared["mhr_shape"][1, 0] = 0.
    assert np.array_equal(shared["mhr_shape"][0], shared["mhr_shape"][1])
    with pytest.raises(ValueError, match="byte-constant"):
        validate_native_parameters(shared, 96, require_shared_identity=True)


@pytest.mark.parametrize("count", [95, 0, -1, True, False, np.bool_(True), 96., "96", None])
@pytest.mark.parametrize("function", [validate_native_parameters, share_first_frame_identity])
def test_full_original_frame_count_cannot_be_short_coerced_or_padded(count, function):
    with pytest.raises(ValueError):
        function(native_parameters(), count)


@pytest.mark.parametrize("name", tuple(NATIVE_PARAMETER_DIMS))
@pytest.mark.parametrize("fault", ["missing", "float64", "int", "nan", "inf", "wrongdim", "short", "masked", "list"])
def test_every_native_block_is_strict_float32_fullN_unmasked_and_finite(name, fault):
    original = native_parameters(97)
    if fault == "missing": del original[name]
    elif fault == "float64": original[name] = original[name].astype(np.float64)
    elif fault == "int": original[name] = original[name].astype(np.int32)
    elif fault == "nan": original[name][0, 0] = np.nan
    elif fault == "inf": original[name][0, 0] = np.inf
    elif fault == "wrongdim": original[name] = original[name][:, :-1]
    elif fault == "short": original[name] = original[name][:-1]
    elif fault == "masked": original[name] = np.ma.array(original[name], mask=False)
    elif fault == "list": original[name] = original[name].tolist()
    with pytest.raises(ValueError):
        share_first_frame_identity(original, 97)


@pytest.mark.parametrize("extra", ["mhr_joints", "mhr_keypoints", "vertices", "metadata", "frame_indices"])
def test_no_stale_geometry_or_implicit_frame_payload_can_enter_exact7block_api(extra):
    original = native_parameters()
    original[extra] = np.zeros((96, 3), np.float32)
    with pytest.raises(ValueError):
        share_first_frame_identity(original, 96)


@pytest.mark.parametrize("fault", ["firstzero", "secondzero", "parallel", "tinyfirst", "tinysecond", "negativez", "zeroz", "expression"])
def test_degenerate_root_negative_translation_or_nonzero_expression_never_repaired(fault):
    original = native_parameters()
    if fault == "firstzero": original["mhr_global_rot6d"][20, [0, 2, 4]] = 0
    elif fault == "secondzero": original["mhr_global_rot6d"][20, [1, 3, 5]] = 0
    elif fault == "parallel": original["mhr_global_rot6d"][20] = [1, 1, 0, 0, 0, 0]
    elif fault == "tinyfirst": original["mhr_global_rot6d"][20, 0] = 1e-9
    elif fault == "tinysecond": original["mhr_global_rot6d"][20, 3] = 1e-9
    elif fault == "negativez": original["mhr_trans"][20, 2] = -1
    elif fault == "zeroz": original["mhr_trans"][20, 2] = 0
    elif fault == "expression": original["mhr_face"][20, 5] = 1e-9
    before = {name: value.tobytes() for name, value in original.items()}
    with pytest.raises(ValueError):
        share_first_frame_identity(original, 96)
    assert before == {name: value.tobytes() for name, value in original.items()}


def test_input_and_output_mutations_are_independent():
    original = native_parameters()
    shared = share_first_frame_identity(original, 96)
    shared["mhr_hand"][0, 0] = 12
    assert original["mhr_hand"][0, 0] == 0
    original["mhr_shape"][0, 0] = 13
    assert shared["mhr_shape"][0, 0] == 0


@pytest.mark.parametrize("bad", [None, [], np.zeros(7), True])
def test_mapping_must_be_explicit(bad):
    with pytest.raises(ValueError):
        validate_native_parameters(bad, 96)


@pytest.mark.parametrize("bad", [1, None, np.bool_(True), "yes"])
def test_constancy_policy_must_be_explicit_python_bool(bad):
    with pytest.raises(ValueError):
        validate_native_parameters(native_parameters(), 96, require_shared_identity=bad)


@pytest.mark.parametrize("count", [96, 97, 501])
def test_metadata_relabels_old_checks_and_forbids_stale_joints_without_quality_claim(count):
    old = metadata()
    before = copy.deepcopy(old)
    result = shared_initializer_metadata(old, count)
    assert old == before
    assert result["historical_original_initializer_checks"] == {name: old[name] for name in HISTORICAL_GEOMETRY_CHECKS}
    assert all(name not in result for name in HISTORICAL_GEOMETRY_CHECKS if name != "mhr_geometry_forward_verified")
    assert result["mhr_geometry_forward_verified"] is False
    assert result["identity_selection_frame_index"] == 0
    assert result["original_frame_indices"] == list(range(count)) and result["source_frames"] == count
    assert result["joint_keypoint_redecode_required"] is True
    assert result["source_joint_keypoint_reuse_permitted"] is False
    assert result["identity_choice_phase"] == "before_native_geometry_crop_render_cache"
    for name in ("quality_verified", "adoption_authorized", "submission_eligible", "challenge_performance_verified",
                 "original_geometry_recovered", "projection_reverified_after_identity_change", "identity_quality_selection_performed"):
        assert result[name] is False
    assert result["camera_intrinsics"] == old["camera_intrinsics"]
    result["source"]["nested"].append(3)
    result["historical_original_initializer_checks"]["native_roundtrip_max_error_m"]["vertices"] = 4
    assert old == before


@pytest.mark.parametrize("fault", ["gtmissing", "gttrue", "gtzero", "handmissing", "handtrue", "oraclemissing", "oracle", "gtread", "privateread", "history", "notdict"])
def test_metadata_never_fabricates_provenance_or_overwrites_original_history(fault):
    old = metadata()
    if fault == "gtmissing": old.pop("ground_truth_used")
    elif fault == "gttrue": old["ground_truth_used"] = True
    elif fault == "gtzero": old["ground_truth_used"] = 0
    elif fault == "handmissing": old.pop("hand_labeled_test")
    elif fault == "handtrue": old["hand_labeled_test"] = True
    elif fault == "oraclemissing": old.pop("oracle_modes")
    elif fault == "oracle": old["oracle_modes"] = ["hidden"]
    elif fault == "gtread": old["ground_truth_read"] = True
    elif fault == "privateread": old["private_truth_read"] = True
    elif fault == "history": old["historical_original_initializer_checks"] = {"original": "keep"}
    elif fault == "notdict": old = MappingProxyType(old)
    with pytest.raises(ValueError):
        shared_initializer_metadata(old, 96)


@pytest.mark.parametrize("bad", [95, True, np.bool_(True), 96., None])
def test_metadata_frame_policy_is_full_original_and_not_inferred(bad):
    with pytest.raises(ValueError):
        shared_initializer_metadata(metadata(), bad)
