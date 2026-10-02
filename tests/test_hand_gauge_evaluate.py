"""Tiny synthetic Sim3 diagnostics; no torch, models, challenge or GPU."""
import importlib.util
import json
from pathlib import Path
import numpy as np
import pytest


@pytest.fixture
def gate(monkeypatch):
    root = Path(__file__).resolve().parents[1]; monkeypatch.syspath_prepend(str(root/"infra"))
    spec = importlib.util.spec_from_file_location("wr_test_hand_gauge", root/"infra/hand_gauge_evaluate.py")
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module


def fixture():
    v = np.array([[0.,0.,0.], [1.,0.,0.], [0.,1.,0.], [0.,0.,1.], [.3,.1,.1], [-.3,.1,.1]])
    truth = np.tile(v, (6,1,1)); j = np.tile(v[:4], (6,1,1))
    rotation = np.array([[0.,-1.,0.], [1.,0.,0.], [0.,0.,1.]])
    before, joints = (truth-[.2,-.1,2.])@rotation.T/1.3, (j-[.2,-.1,2.])@rotation.T/1.3
    predictions = {mode+"_"+field: array.copy() for mode in ("baseline","candidate")
                   for field,array in (("vertices_camera_m",before),("joints_camera_m",joints))}
    regions = {"left": {"mask": np.array([False]*4+[True,False]), "wrist":0,"fingers":np.array([1])},
               "right": {"mask": np.array([False]*5+[True]), "wrist":1,"fingers":np.array([2])}}
    return predictions, (truth,j), regions


def test_constant_shared_sim3_removed_all_cases(gate):
    predictions, truth, regions = fixture(); result = gate.diagnose(predictions, truth, regions)
    assert result["positive_scale"] == pytest.approx(1.3) and result["rotation_determinant"] == pytest.approx(1)
    assert result["summary"]["baseline"]["raw"] > 1000
    assert result["summary"]["baseline"]["shared_Sim3"] < 1e-10
    assert result["fit_frame_index"] == 0 and result["summary_cases"] == [1,2,3,4,5]
    assert "NOT_official" in result["alignment_role"]


def test_same_baseline_fit_does_not_erase_candidate_articulation(gate):
    predictions, truth, regions = fixture()
    predictions["candidate_vertices_camera_m"][:,4,0] += .1
    result = gate.diagnose(predictions, truth, regions)
    assert result["summary"]["baseline"]["shared_Sim3"] < 1e-10
    assert result["cases"][0]["modes"]["candidate"]["hands"]["left"]["shared_Sim3"]["hand_pve_mm"] == pytest.approx(130)
    assert result["cases"][0]["modes"]["candidate"]["hands"]["right"]["shared_Sim3"]["hand_pve_mm"] < 1e-10


def test_time_varying_global_error_not_removed_by_frame_zero_fit(gate):
    predictions, truth, regions = fixture()
    for mode in ("baseline","candidate"):
        predictions[mode+"_vertices_camera_m"][3] += [.1,0.,0.]
        predictions[mode+"_joints_camera_m"][3] += [.1,0.,0.]
    result = gate.diagnose(predictions, truth, regions)
    assert result["cases"][0]["modes"]["baseline"]["shared_Sim3_nonhand_pve_mm"] < 1e-10
    assert result["cases"][3]["modes"]["baseline"]["shared_Sim3_nonhand_pve_mm"] == pytest.approx(130)
    assert result["cases"][3]["modes"]["baseline"]["hands"]["left"]["shared_Sim3"]["wrist_relative_hand_pve_mm_diagnostic"] < 1e-10


def test_fitting_never_reflects_even_reflected_target(gate):
    a = np.array([[0.,0.,0.],[1.,0.,0.],[0.,2.,0.],[0.,0.,3.],[1.,2.,3.]])
    transform = gate.fit_sim3(a, a*[-1.,1.,1.])
    assert transform[0] > 0 and np.linalg.det(transform[1]) == pytest.approx(1)
    assert not np.allclose(gate.apply_sim3(a, transform), a*[-1.,1.,1.])


@pytest.mark.parametrize("fault", ["nan", "dtype", "shape", "collinear", "zero"])
def test_invalid_similarity_no_fallback(gate, fault):
    a = np.eye(3); b = a.copy()
    if fault == "nan": a[0,0] = np.nan
    elif fault == "dtype": a = a.astype(int)
    elif fault == "shape": b = b[:2]
    elif fault == "collinear": a = np.tile(np.arange(3)[:,None], (1,3))
    else: a[:] = 0
    with pytest.raises(ValueError): gate.fit_sim3(a,b)


def test_inputs_unchanged_no_aligned_prediction_export(gate):
    predictions, truth, regions = fixture(); frozen = {k:v.tobytes() for k,v in predictions.items()}
    result = gate.diagnose(predictions, truth, regions)
    assert frozen == {k:v.tobytes() for k,v in predictions.items()}
    assert not any(isinstance(v, np.ndarray) for v in result.values())


def test_failed_integrity_exclusive_report_preserved(gate, tmp_path, monkeypatch):
    output = tmp_path/"validation/hands_rgb_v1/quality-gauge"; output.mkdir(parents=True)
    monkeypatch.setenv("WR_ROOT", str(tmp_path)); monkeypatch.setenv("WR_CODE_REVISION", "a"*40)
    monkeypatch.setenv("WR_IMAGE_ID", "sha256:"+"b"*64); monkeypatch.setattr(gate.platform,"system",lambda:"Linux")
    original = gate.Path.iterdir
    monkeypatch.setattr(gate.Path,"iterdir",lambda self:[Path("lo")] if str(self)=="/sys/class/net" else original(self))
    with pytest.raises(FileNotFoundError): gate.main([])
    path = output/"report.json"; frozen = path.read_bytes(); report = json.loads(frozen)
    assert report["status"] == "fail" and report["predictions_modified"] is False and report["new_quality_selection_performed"] is False
    with pytest.raises(FileExistsError): gate.main([])
    assert path.read_bytes() == frozen


def test_proxy_truth_eval_only_no_models_private_writable_or_adoption(gate):
    with pytest.raises(SystemExit): gate.main(["--frame", "3"])
    source = Path(gate.__file__).read_text(); wrapper = Path(gate.__file__).with_name("run_hand_gauge_evaluate.sh").read_text()
    assert "evaluate.run(root, validated)" in source and 'original_H1_hypothesis_rejected=True' in source
    assert "official_alignment_roles_used" in source and "before[0, nonhand]" in source
    assert '--gpus' not in wrapper and 'weights' not in wrapper and 'vendor' not in wrapper
    assert 'src=$BASE/eval_private,dst=$BASE/eval_private,readonly' in wrapper
    assert 'src=$BASE,dst=$BASE' not in wrapper and 'chown -R' not in wrapper
    assert '63s docker run' in wrapper and '--network none' in wrapper
