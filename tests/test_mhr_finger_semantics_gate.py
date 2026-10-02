"""Tiny procedural metadata/kinematic fixtures; never torch, assets or CUDA."""
import copy
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def gate():
    path = Path(__file__).resolve().parents[1] / "infra/mhr_finger_semantics_gate.py"
    spec = importlib.util.spec_from_file_location("wr_test_finger_semantics", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


@pytest.fixture
def metadata():
    joints, parents = ["root"], [-1]
    for side in "rl":
        wrist = len(joints); joints.append(f"{side}_wrist"); parents.append(0)
        for finger in ("thumb", "index", "middle", "ring", "pinky"):
            parent = wrist
            for segment in ("0", "1", "2", "3", "null"):
                joints.append(f"{side}_{finger}{segment}"); parents.append(parent); parent = len(joints)-1
    suffixes = ["thumb0_ry", "thumb0_rz", "thumb1_rx", "thumb1_ry", "thumb1_rz", "thumb2_rz", "thumb3_rz",
                "index1_ry", "ring1_ry", "pinky1_ry", "middle1_ry", "index1_rz", "index2_rz", "index3_rz",
                "middle1_rz", "middle2_rz", "middle3_rz", "ring1_rz", "ring2_rz", "ring3_rz",
                "pinky1_rz", "pinky2_rz", "pinky3_rz", "index1_rx", "ring1_rx", "pinky1_rx", "middle1_rx"]
    parameters = [f"unused_{i}" for i in range(249)]
    parameters[68:122] = [f"{side}_{suffix}" for side in "rl" for suffix in suffixes]
    matrix = np.zeros((len(joints)*7, 249), dtype=np.float64)
    for i in range(68, 122):
        stem, axis = parameters[i].rsplit("_", 1)
        matrix[joints.index(stem)*7+3+"xyz".index(axis[-1]), i] = 1
    return [joints, parameters, np.asarray(parents), matrix, np.arange(95, 122), np.arange(68, 95)]


def test_exact_named_body_index_partition_and_real_descendants(gate, metadata):
    records = gate.finger_partition(*metadata)
    assert len(records) == 54 and records[0]["side"] == "r" and records[27]["side"] == "l"
    assert metadata[0].index("r_thumb0") in records[0]["global_closure"]
    assert metadata[0].index("r_thumbnull") in records[0]["global_closure"]
    assert metadata[0].index("r_wrist") not in records[0]["global_closure"]


@pytest.mark.parametrize("fault", ["other_side", "other_finger", "wrist", "translation", "scale", "zero", "nan"])
def test_local_map_must_be_same_finger_rotation_only(gate, metadata, fault):
    joints, _, _, matrix, *_ = metadata
    if fault in ("other_side", "other_finger", "wrist"):
        name = {"other_side": "l_thumb0", "other_finger": "r_index1", "wrist": "r_wrist"}[fault]
        matrix[joints.index(name)*7+3, 68] = 1
    elif fault in ("translation", "scale"):
        matrix[joints.index("r_thumb0")*7+(0 if fault == "translation" else 6), 68] = 1
    elif fault == "zero": matrix[:, 68] = 0
    else: matrix[0, 68] = np.nan
    with pytest.raises(ValueError): gate.finger_partition(*metadata)


@pytest.mark.parametrize("fault", ["cycle", "two_roots", "range", "wrong_wrist_ancestry", "foreign_descendant", "float_parents"])
def test_real_parent_hierarchy_is_validated(gate, metadata, fault):
    joints, _, parents, *_ = metadata
    index = joints.index("r_thumb0")
    if fault == "cycle": parents[index] = index
    elif fault == "two_roots": parents[index] = -1
    elif fault == "range": parents[index] = len(joints)
    elif fault == "wrong_wrist_ancestry": parents[index] = joints.index("l_wrist")
    elif fault == "foreign_descendant": parents[joints.index("r_index0")] = index
    else: metadata[2] = parents.astype(float)
    with pytest.raises(ValueError): gate.finger_partition(*metadata)


@pytest.mark.parametrize("fault", ["duplicate", "wrong_name", "hidden_finger", "different_indices", "float_indices", "wrong_columns"])
def test_names_and_checkpoint_indices_fail_closed(gate, metadata, fault):
    if fault == "duplicate": metadata[0][1] = metadata[0][0]
    elif fault == "wrong_name": metadata[1][68] = "r_wrist_rx"
    elif fault == "hidden_finger": metadata[1][0] = "r_thumb0_rx"
    elif fault == "different_indices": metadata[4], metadata[5] = metadata[5], metadata[4]
    elif fault == "float_indices": metadata[4] = metadata[4].astype(float)
    else: metadata[3] = metadata[3][:, :204]
    with pytest.raises(ValueError): gate.finger_partition(*metadata)


def test_schemas_exact_not_guessed_upstream(gate):
    schemas = {name: ([], result) for name, result in gate.METHODS.items()}
    schemas["forward"] = ([('identity_coeffs', 'Tensor'), ('model_parameters', 'Tensor'),
                           ('face_expr_coeffs', 'Tensor'), ('apply_correctives', 'bool')], "Tuple[Tensor, Tensor]")
    gate.validate_schemas(schemas)
    for key in schemas:
        bad = copy.deepcopy(schemas); bad[key] = ([], "Tensor") if key != "get_parameter_transform" else ([], "Tuple[Tensor, Tensor]")
        with pytest.raises(ValueError): gate.validate_schemas(bad)


def support_fixture():
    neutral_s = np.zeros((3, 8)); neutral_s[:, 6:8] = 1
    skeleton = np.tile(neutral_s, (4, 1, 1))
    for i, angle in enumerate((-.001, .001, -.002, .002)):
        skeleton[i, 1, 3] = np.sin(angle/2); skeleton[i, 1, 6] = np.cos(angle/2)
    neutral_v, vertices = np.zeros((2, 3)), np.zeros((4, 2, 3))
    vertices[:, 0, 0] = [-.001, .001, -.002, .002]
    record = [{"column": 68, "parameter": "r_thumb0_rx", "global_closure": [1]}]
    return neutral_v, neutral_s, vertices, skeleton, record


def test_orientation_not_joint_center_required_and_sign_equivalent(gate):
    args = support_fixture(); result = gate.support_evidence(*args)
    assert result[0]["steps"][0]["orientation_derivative_max_per_rad"] > .49
    args[3][1, :, 3:7] *= -1
    gate.support_evidence(*args)


def test_completed_control_evidence_survives_later_failure(gate):
    args = support_fixture(); skeleton = np.concatenate((args[3], args[3]), axis=0)
    skeleton[4, 0, 0] = 1e-3
    records = args[4] + [dict(args[4][0], column=69, parameter="r_thumb0_ry")]
    evidence = []
    with pytest.raises(ValueError):
        gate.support_evidence(args[0], args[1], np.concatenate((args[2], args[2])), skeleton, records, evidence=evidence)
    assert len(evidence) == 2 and evidence[0]["status"] == "pass" and evidence[1]["status"] == "fail"
    assert evidence[1]["steps"][0]["excluded_joint_position_max_m"] > 0


def test_zero_vertex_effect_fails_despite_correct_skeleton_orientation(gate):
    args = support_fixture(); args[2][:] = 0; evidence = []
    with pytest.raises(ValueError, match="vertex derivative"):
        gate.support_evidence(*args, evidence=evidence)
    assert evidence[0]["steps"][0]["orientation_derivative_max_per_rad"] > 0
    assert evidence[0]["steps"][0]["vertex_derivative_max_m_per_rad"] == 0


def test_progress_callback_sees_compact_evidence_before_failure(gate):
    args = support_fixture(); args[3][0, 0, 0] = 1e-3
    evidence, snapshots = [], []
    with pytest.raises(ValueError):
        gate.support_evidence(*args, evidence=evidence, on_progress=lambda: snapshots.append(copy.deepcopy(evidence)))
    assert snapshots[0][0]["status"] == "fail"
    assert snapshots[0][0]["steps"][0]["excluded_joint_position_max_m"] > 0


@pytest.mark.parametrize("fault", ["outside_position", "outside_orientation", "scale", "inactive", "step_inconsistent"])
def test_excluded_joint_or_unobservable_or_unstable_effect_rejected(gate, fault):
    args = support_fixture(); skeleton = args[3]
    if fault == "outside_position": skeleton[0, 0, 0] = 1e-3
    elif fault == "outside_orientation": skeleton[0, 0, 3] = 1e-3
    elif fault == "scale": skeleton[0, 1, 7] = 1.01
    elif fault == "inactive": skeleton[:] = args[1]
    else: skeleton[2:4, 1, 3] *= 2
    with pytest.raises(ValueError): gate.support_evidence(*args)


@pytest.mark.parametrize("fault", ["nan_vertices", "zero_scale", "nonunit_quaternion", "wrong_joint_width"])
def test_geometry_requires_actual8state_unit_quaternions(gate, fault):
    v = np.zeros((1, 18439, 3), dtype=np.float32); s = np.zeros((1, 3, 8), dtype=np.float32); s[..., 6:8] = 1
    gate.check_geometry(v, s, 3)
    if fault == "nan_vertices": v[0, 0, 0] = np.nan
    elif fault == "zero_scale": s[0, 0, 7] = 0
    elif fault == "nonunit_quaternion": s[0, 0, 6] = 2
    else: s = s[..., :3]
    with pytest.raises(ValueError): gate.check_geometry(v, s, 3)


def test_failed_report_exclusive_before_torch_or_forward(gate, tmp_path, monkeypatch):
    (tmp_path / "results").mkdir(); (tmp_path / "results/weights-acquisition.json").write_text('{"assets": []}')
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setattr(gate.platform, "system", lambda: "Linux")
    original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path, "iterdir", lambda self: [Path("lo")] if str(self) == "/sys/class/net" else original(self))
    with pytest.raises(ValueError, match="acquisition"): gate.main([])
    path = tmp_path / "results/mhr-finger-semantics-v2.json"; report = json.loads(path.read_text())
    assert report["status"] == "fail" and report["forward_calls"] == 0 and report["adoption_performed"] is False
    frozen = path.read_bytes()
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_unknown_arguments_and_offline_remote_readonly_scope(gate):
    with pytest.raises(SystemExit): gate.main(["--episode", "15"])
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_mhr_finger_semantics_gate.sh").read_text()
    assert "import pymomentum" not in source and "import torch" in source
    assert "handle.seek(0)" in source and "handle.truncate()" in source and "os.fsync" in source
    assert "--network none" in wrapper and "303s docker run" in wrapper and '"$IMAGE" python' in wrapper
    assert not any(f"src=$ROOT/{p}" in wrapper for p in ("data", "outputs", "vendor"))
    assert "weights/mhr,readonly" in wrapper and "dinov3,readonly" in wrapper
    assert gate.BODY_SHA != "78ff5cb874dd012a272382e3f2d8bc11226d5b7d0ecc739a60fbb4a97a5a5ba3"


@pytest.mark.parametrize("count", [1, 6, 216])
def test_exact_mhrdemo_identity_rows_not_generic_broadcast(gate, count):
    shared = np.arange(45, dtype=np.float32)/100
    rows = gate.identity_rows(shared, count)
    expression = np.zeros((count, 72), np.float32)
    assert rows.shape == (count, 45) and len(rows) == len(expression)
    assert all(row.tobytes() == shared.tobytes() for row in rows)
    rows[0, 0] = 1
    assert shared[0] == 0


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_identity_row_count_explicit(gate, count):
    with pytest.raises(ValueError): gate.identity_rows(np.zeros(45, np.float32), count)
