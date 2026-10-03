"""H97 real tiny NumPy fitting, no CUDA/model or private quality claim."""
import importlib.util
from pathlib import Path
import subprocess

import numpy as np
import pytest

INFRA=Path(__file__).parents[1]/"infra"


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA));monkeypatch.syspath_prepend(str(INFRA.parent/"src"))
    spec=importlib.util.spec_from_file_location("translation_fit_test",INFRA/"translation_rgb_fit.py")
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def tiny_frame(gate,monkeypatch):
    spec=importlib.util.spec_from_file_location("translation_fixture",Path(__file__).with_name("test_translation_rgb_public.py"))
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    raw,pair,record,_=m.fixture(gate.public,monkeypatch)
    points=pair["shared_keypoints_camera_m"]
    indices=list(gate.policy.TRAIN_MHR)
    points[indices]=np.c_[np.linspace(-.7,.7,10),np.sin(np.arange(10))*.6,np.linspace(3.5,4.5,10)]
    target,_=gate.policy.project_and_jacobian(points[indices].astype(np.float64),raw["pred_cam_t"],np.array([.1,-.08,.07]),pair["camera_K"])
    xy=np.zeros((1,133,2),np.float64);xy[0,list(gate.policy.TRAIN_COCO)]=target
    observation=dict(keypoints=xy,scores=np.ones((1,133),np.float32))
    report=dict(fit_records=[],counters=dict(evaluated_states=0,adam_updates=0,initial_jacobian_rows=0))
    return raw,pair,record,observation,report


def test_sixty_actual_states_59_manual_updates_best_evaluated(gate,monkeypatch):
    raw,pair,record,observed,report=tiny_frame(gate,monkeypatch)
    candidate=gate.fit_frame(raw,pair,observed,record,report,lambda:None)
    row=report["fit_records"][0]
    assert row["cpu_evaluations"]==60 and row["adam_updates"]==59 and len(row["evaluated_losses"])==60
    assert row["best_evaluated_index"]==int(np.argmin(row["evaluated_losses"]))
    assert row["selected_total_objective"]<=row["evaluated_losses"][0]
    assert report["counters"]==dict(evaluated_states=60,adam_updates=59,initial_jacobian_rows=20)
    assert candidate["mhr_model_params"].tobytes()==pair["shared_model_controls"].tobytes()
    assert gate.public.validate_schedule(row,raw,pair,observed,candidate)==20


@pytest.mark.parametrize("fault",["initial","latent","loss","best","jacobian","gradient"])
def test_full_analytic_schedule_replay_fails_before_private(gate,monkeypatch,fault):
    raw,pair,record,observed,report=tiny_frame(gate,monkeypatch)
    candidate=gate.fit_frame(raw,pair,observed,record,report,lambda:None);row=report["fit_records"][0]
    if fault=="initial":row["evaluated_latents"][0][0]=.01
    elif fault=="latent":row["evaluated_latents"][30][0]+=.01
    elif fault=="loss":row["evaluated_losses"][5]+=.01
    elif fault=="best":row["best_evaluated_index"]=59 if row["best_evaluated_index"]!=59 else 58
    elif fault=="jacobian":row["initial_observation_jacobian"]["singular_values"][0]+=.01
    else:candidate["latent"][0]+=.01
    with pytest.raises(ValueError):gate.public.validate_schedule(row,raw,pair,observed,candidate)


def test_underidentified_training_rank_fails_before_update(gate,monkeypatch):
    raw,pair,record,observed,report=tiny_frame(gate,monkeypatch)
    pair["shared_keypoints_camera_m"][list(gate.policy.TRAIN_MHR)]=[0.,0.,4.]
    with pytest.raises(ValueError):gate.fit_frame(raw,pair,observed,record,report,lambda:None)
    assert report["counters"]["adam_updates"]==0


def test_native_backend_not_called_cpu_only_wrapper(gate):
    source=Path(gate.__file__).read_text();shell=INFRA/"run_translation_rgb_fit.sh"
    subprocess.run(["bash","-n",str(shell)],check=True)
    assert"import torch"not in source and"mhr_forward"not in source and"--gpus"not in shell.read_text()
    assert"--network none"in shell.read_text()and"--memory 8g"in shell.read_text()and"63s"in shell.read_text()
    assert"eval_private"not in shell.read_text() and'OUT="$BASE/translation_fit_v1"'in shell.read_text()
