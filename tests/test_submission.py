"""Generated-only rows/poses/meshes; no real sample values or reference data."""

from dataclasses import replace
import json
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from world_reward.contracts import Reconstruction
from world_reward.submission import (
    ARRAY_ORDER, EPISODE_FRAME, PARAMETER_COUNTS,
    Track1Episode, Track1Layout, assemble_submission, decode_submission,
    parameters_from_official_converter, read_submission, read_template,
    verify_roundtrip, write_submission,
)


COMMIT = "https://github.com/world-reward/v2d/commit/" + "a" * 40
KIT = Path("/tmp/v2d-audit/v2d_submission_kit")


def row_ids(episodes=(3,), frames=(2, 3, 4, 8, 9, 10), vertex_rows=8, face_rows=8):
    rows = []
    for episode in episodes:
        for array, scope in ((0, frames), (1, [EPISODE_FRAME]), (2, [EPISODE_FRAME]),
                             (3, frames), (4, frames), (5, [EPISODE_FRAME]),
                             (6, [EPISODE_FRAME]), (7, [EPISODE_FRAME])):
            count = {0: 46, 1: 23, 2: 15, 3: 3, 4: 1, 5: 1, 6: vertex_rows, 7: face_rows}[array]
            rows.extend(f"t1_{episode:06d}_{frame:06d}_{array}_{point:06d}"
                        for frame in scope for point in range(count))
    return rows


def episode(frames=12):
    pose = np.arange(frames * 136, dtype=np.float32).reshape(frames, 136) / 17
    angles = np.linspace(0, 0.3, frames)[:, None]
    rotation = Rotation.from_euler("z", angles).as_matrix()
    rec = Reconstruction(
        pose=pose, scales=np.arange(68, dtype=np.float32) / 1000,
        shape=np.arange(45, dtype=np.float32) / 100,
        object_rotation=rotation,
        object_translation=np.column_stack((np.arange(frames) / 11, np.zeros((frames, 2)))),
        object_scale=0.25,
    )
    # One small tetrahedron; padding is not interpreted as extra surface.
    vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    return Track1Episode(
        rec, vertices, faces, np.zeros(72),
        {"input_track": "track_1", "ground_truth_used": False, "oracle_modes": [], "hand_labeled_test": False},
        frames,
    )


def assembled():
    ids = row_ids(episodes=(3, 9))
    rng = np.random.default_rng(20)
    layout = Track1Layout.from_row_ids([ids[index] for index in rng.permutation(len(ids))])
    predictions = {3: episode(), 9: episode()}
    return layout, predictions, assemble_submission(layout, predictions, code_commit_url=COMMIT)


def test_exact_nonuniform_original_frame_selection_and_shared_identity_roundtrip():
    layout, predictions, rows = assembled()
    decoded = decode_submission(rows, layout)
    assert rows.row_ids == layout.row_ids
    for index, output in decoded.items():
        np.testing.assert_array_equal(output.frame_indices, [2, 3, 4, 8, 9, 10])
        np.testing.assert_array_equal(output.reconstruction.pose, predictions[index].reconstruction.pose[output.frame_indices])
        np.testing.assert_array_equal(output.reconstruction.scales, predictions[index].reconstruction.scales)
        np.testing.assert_array_equal(output.reconstruction.shape, predictions[index].reconstruction.shape)
        assert output.reconstruction.object_scale == 0.25
    report = verify_roundtrip(rows, layout, predictions)
    assert report == {"schema_roundtrip_verified": True, "mhr_geometry_forward_verified": False,
                      "episodes": 2, "rows": len(rows.row_ids), "scored_frames": 12}


def test_padding_matches_official_rule_and_has_no_surface():
    layout, predictions, rows = assembled()
    mesh = decode_submission(rows, layout)[3]
    np.testing.assert_array_equal(mesh.object_vertices[:4], predictions[3].object_vertices)
    np.testing.assert_array_equal(mesh.object_vertices[4:], np.tile(predictions[3].object_vertices[0], (4, 1)))
    np.testing.assert_array_equal(mesh.object_faces[4:], np.zeros((4, 3)))
    a, b, c = (mesh.object_vertices[mesh.object_faces[:, index]] for index in range(3))
    assert np.linalg.norm(np.cross(b - a, c - a), axis=1)[4:].sum() == 0


def test_generated_schema_matches_audited_official_helpers():
    if not KIT.is_dir():
        pytest.skip("audited official kit unavailable; core synthetic tests remain independent")
    sys.path.insert(0, str(KIT))
    try:
        from v2dlb import mhr_submission as official
        assert tuple(official.array_order()) == ARRAY_ORDER
        assert official.EPISODE_FRAME == EPISODE_FRAME
        assert official.PARAMETER_COUNTS == PARAMETER_COUNTS
        layout, predictions, rows = assembled()
        for array, name, count in ((0, "mhr_pose", 136), (1, "mhr_scales", 68), (2, "mhr_shape", 45)):
            keys = layout.keys
            selected = np.flatnonzero((keys[:, 0] == 3) & (keys[:, 2] == array))
            order = np.lexsort((keys[selected, 3], keys[selected, 1]))
            packed = rows.values[selected[order]]
            if array == 0:
                packed = packed.reshape(6, -1, 3)
            unpacked = official._unpack(packed, count, name)
            expected = {0: predictions[3].reconstruction.pose[[2, 3, 4, 8, 9, 10]],
                        1: predictions[3].reconstruction.scales, 2: predictions[3].reconstruction.shape}[array]
            np.testing.assert_array_equal(unpacked, expected)
        np.testing.assert_allclose(official._rotations(predictions[3].reconstruction.object_rotation),
                                   predictions[3].reconstruction.object_rotation, atol=1e-14)
    finally:
        sys.path.remove(str(KIT))


def test_parquet_template_reads_only_row_ids_and_written_artifact_roundtrips(tmp_path):
    pa = pytest.importorskip("pyarrow")
    pq = pytest.importorskip("pyarrow.parquet")
    ids = row_ids()
    template = tmp_path / "generated_template.parquet"
    # Deliberately unusable sample values prove they cannot be predictions.
    pq.write_table(pa.table({"row_id": ids, "x": [float("nan")] * len(ids),
                            "ref_x": ["MUST_NOT_BE_READ"] * len(ids)}), template)
    layout = read_template(template)
    predictions = {3: episode()}
    rows = assemble_submission(layout, predictions, code_commit_url=COMMIT)
    artifact = tmp_path / "frozen.parquet"
    report = write_submission(artifact, rows, layout, predictions)
    recovered = read_submission(artifact, layout)
    np.testing.assert_array_equal(recovered.values, rows.values)
    assert report["schema_roundtrip_verified"]
    assert not list(tmp_path.glob(".frozen.parquet.*"))
    with pytest.raises(FileExistsError):
        write_submission(artifact, rows, layout, predictions)


def test_parquet_missing_optional_dependency_reports_actionable_error(monkeypatch, tmp_path):
    import builtins
    original_import = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name.startswith("pyarrow"):
            raise ImportError("deliberately unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="optional pyarrow"):
        read_template(tmp_path / "unused.parquet")


@pytest.mark.parametrize("changes", [
    lambda ids: [], lambda ids: ids + ids[:1], lambda ids: ids[1:],
    lambda ids: [ids[0].replace("t1_", "t2_")] + ids[1:],
    lambda ids: [ids[0].replace("000003", "3", 1)] + ids[1:],
    lambda ids: [item for item in ids if item.split("_")[3] != "2"],
    lambda ids: [item.replace("999999_1", "000001_1") for item in ids],
    lambda ids: [item.replace("000002_3", "000001_3") for item in ids],
])
def test_invalid_or_partial_layout_fails(changes):
    with pytest.raises(ValueError):
        Track1Layout.from_row_ids(changes(row_ids()))


def test_short_scored_stretch_and_overbudget_layout_fail():
    with pytest.raises(ValueError, match="three frames"):
        Track1Layout.from_row_ids(row_ids(frames=(2, 3)))
    with pytest.raises(ValueError, match="4096"):
        Track1Layout.from_row_ids(row_ids(vertex_rows=4097))
    with pytest.raises(ValueError, match="consistent"):
        Track1Layout.from_row_ids(row_ids() + row_ids(episodes=(4,), vertex_rows=9))


@pytest.mark.parametrize("changes, message", [
    ({"scales": np.zeros((12, 68))}, "expected"),
    ({"shape": np.zeros((12, 45))}, "expected"),
    ({"pose": np.zeros((12, 260))}, "expected"),
    ({"pose": np.full((12, 136), np.nan)}, "finite"),
    ({"object_translation": np.zeros((11, 3))}, "expected"),
    ({"object_rotation": np.tile(np.diag([-1, 1, 1]), (12, 1, 1))}, "reflections"),
    ({"object_scale": np.ones(12)}, "one scalar"),
    ({"object_scale": 0}, "positive"),
    ({"object_scale": np.inf}, "finite"),
])
def test_full_trajectory_and_constant_identity_contract_fail_closed(changes, message):
    prediction = episode()
    prediction = replace(prediction, reconstruction=replace(prediction.reconstruction, **changes))
    with pytest.raises(ValueError, match=message):
        assemble_submission(Track1Layout.from_row_ids(row_ids()), {3: prediction}, code_commit_url=COMMIT)


def test_bad_unscored_frame_and_compact_scored_trajectory_are_not_accepted():
    prediction = episode()
    pose = prediction.reconstruction.pose.copy()
    pose[0] = np.nan  # Frame 0 is not scored, but the full-video source still matters.
    with pytest.raises(ValueError, match="finite"):
        replace(prediction, reconstruction=replace(prediction.reconstruction, pose=pose)).validate()
    with pytest.raises(ValueError, match="last scored"):
        assemble_submission(Track1Layout.from_row_ids(row_ids()), {3: episode(frames=6)}, code_commit_url=COMMIT)
    with pytest.raises(ValueError, match="expected"):
        replace(episode(), total_video_frames=13).validate()
    with pytest.raises(ValueError, match="positive integer"):
        replace(episode(), total_video_frames=True).validate()


@pytest.mark.parametrize("changes, message", [
    ({"human_expression": np.ones(72)}, "zero facial"),
    ({"human_expression": np.zeros((12, 72))}, "expected"),
    ({"mhr_parameter_format": "cari4d_body_cont260"}, "native MHR"),
    ({"provenance": {}}, "ground_truth_used"),
    ({"provenance": {"input_track": "track_2", "ground_truth_used": False, "oracle_modes": []}}, "Track 1"),
    ({"provenance": {"input_track": "track_1", "ground_truth_used": False, "oracle_modes": ["first_rotation"]}}, "oracle"),
    ({"provenance": {"input_track": "track_1", "ground_truth_used": False, "oracle_modes": [], "hand_labeled_test": True}}, "hand-labeled"),
    ({"provenance": {"input_track": "track_1", "ground_truth_used": False, "oracle_modes": []}}, "hand_labeled_test=False"),
    ({"object_faces": np.array([[0, 1, 4]])}, "outside"),
    ({"object_faces": np.array([[0, 1, 2.5]])}, "integer"),
    ({"object_faces": np.array([[-1, 1, 2]])}, "outside"),
    ({"object_faces": np.zeros((1, 3))}, "surface area"),
    ({"object_vertices": np.zeros((4097, 3))}, "4096"),
])
def test_invalid_expression_provenance_and_mesh_fail(changes, message):
    with pytest.raises(ValueError, match=message):
        replace(episode(), **changes).validate()


def test_mesh_budget_and_episode_set_cannot_be_silently_changed():
    layout = Track1Layout.from_row_ids(row_ids(vertex_rows=3, face_rows=1))
    with pytest.raises(ValueError, match="exceeds template"):
        assemble_submission(layout, {3: episode()}, code_commit_url=COMMIT)
    layout = Track1Layout.from_row_ids(row_ids())
    with pytest.raises(ValueError, match="exactly match"):
        assemble_submission(layout, {4: episode()}, code_commit_url=COMMIT)


@pytest.mark.parametrize("commit", ["main", COMMIT[:-1], COMMIT + "/", COMMIT.replace("github.com", "evilgithub.com")])
def test_unpinned_or_spoofed_commit_url_fails(commit):
    with pytest.raises(ValueError, match="pinned"):
        assemble_submission(Track1Layout.from_row_ids(row_ids()), {3: episode()}, code_commit_url=commit)


@pytest.mark.parametrize("array, point, coordinate, value, message", [
    (0, 45, 1, 1, "nonzero unused"), (1, 22, 2, 1, "nonzero unused"),
    (5, 0, 1, 1, "scale"), (5, 0, 0, 0, "positive"),
    (7, 0, 0, 999, "outside"), (7, 0, 0, 0.2, "integer"),
])
def test_decode_rejects_invalid_parameter_tails_scale_and_mesh(array, point, coordinate, value, message):
    layout, _, rows = assembled()
    values = rows.values.copy()
    selected = np.flatnonzero((layout.keys[:, 0] == 3) & (layout.keys[:, 2] == array) & (layout.keys[:, 3] == point))[0]
    values[selected, coordinate] = value
    with pytest.raises(ValueError, match=message):
        decode_submission(replace(rows, values=values), layout)


def test_roundtrip_detects_parameter_or_geometry_corruption_without_gt():
    layout, predictions, rows = assembled()
    values = rows.values.copy()
    values[np.flatnonzero(layout.keys[:, 2] == 0)[0], 0] += 1
    with pytest.raises(ValueError, match="pose changed"):
        verify_roundtrip(replace(rows, values=values), layout, predictions)
    with pytest.raises(ValueError, match="row IDs/order"):
        decode_submission(replace(rows, row_ids=tuple(reversed(rows.row_ids))), layout)
    with pytest.raises(ValueError, match="finite"):
        decode_submission(replace(rows, values=np.full_like(rows.values, np.nan)), layout)


def converter_result():
    # Fabricated *converter-format* fixture only; it is not a claimed MHR fit.
    rec = episode(frames=3).reconstruction
    errors = np.array([0.001, 0.002, 0.001], dtype=np.float32)
    return dict(pose=rec.pose, scales=rec.scales, shape=rec.shape,
                valid_input=np.ones(3, dtype=bool), per_frame_vertex_error_mm=errors,
                report=dict(frames=3, fitted_frames=3, invalid_input_frames=[],
                            vertex_error_mm=dict(mean=float(errors.mean()), worst_frame_mean=float(errors.max()),
                                                 max=0.003, p99=0.0025)))


def test_real_converter_contract_is_validated_not_pca_reshaped():
    result = converter_result()
    result["report"] = np.array(json.dumps(result["report"]))  # Official NPZ representation.
    converted = parameters_from_official_converter(result, max_mean_vertex_error_mm=0.01)
    np.testing.assert_array_equal(converted.pose, result["pose"])
    np.testing.assert_array_equal(converted.scales, result["scales"])
    np.testing.assert_array_equal(converted.expression, np.zeros(72))
    assert not np.shares_memory(converted.pose, result["pose"])


@pytest.mark.parametrize("change, message", [
    (lambda result: result.pop("valid_input"), "lacks"),
    (lambda result: result.update(pose=np.zeros((3, 260))), "136"),
    (lambda result: result.update(scales=np.zeros(28)), "expected"),
    (lambda result: result.update(valid_input=np.array([True, False, True])), "no nearest-frame"),
    (lambda result: result.update(per_frame_vertex_error_mm=np.array([0, 1, 0])), "residual exceeds"),
    (lambda result: result["report"].update(invalid_input_frames=[1]), "every original frame"),
    (lambda result: result["report"]["vertex_error_mm"].update(mean=0.009), "disagrees"),
    (lambda result: result["report"]["vertex_error_mm"].update(worst_frame_mean=0.009), "disagrees"),
    (lambda result: result.update(report="not json"), "valid JSON"),
])
def test_invalid_converter_outputs_fail_before_assembly(change, message):
    result = converter_result()
    change(result)
    with pytest.raises(ValueError, match=message):
        parameters_from_official_converter(result, max_mean_vertex_error_mm=0.01)
