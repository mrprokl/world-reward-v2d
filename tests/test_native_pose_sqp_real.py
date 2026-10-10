"""Data-free integration contracts; actual MHR/E2E validation runs on Azure."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).parents[1]/'infra'))
import native_pose_sqp_real as run
from test_contact_feasible_real import tiny
from test_native_pose_sqp_adapter import parameters


def test_runner_source_closure_full_native_schedule_and_original_B_binding():
    assert run.ENTRY=='run_native_pose_sqp_real' and run.BUDGET==900
    assert all((Path(__file__).parents[1]/p).is_file() for p in run.HELPERS)
    assert run.native.HELPERS==run.saved.native.HELPERS
    source=(Path(__file__).parents[1]/'infra/native_pose_sqp_real.py').read_text()
    adapter=(Path(__file__).parents[1]/'src/world_reward/native_pose_sqp_adapter.py').read_text()
    assert "asdict(cfg)!=b['report']['native_config']" in source
    assert 'scheduled_PEN_and_silhouette_retained=True' in source
    assert 'i.params_fixed=t' not in adapter and 'OBJECTIVE_STAGE = 181' in adapter
    assert "i.loss(indices,OBJECTIVE_STAGE,include_diagnostics=False)" in adapter
    assert 'VJPs_per_batch=6,chunk=16' in adapter
    assert not any(s in source+adapter for s in ('form_hoi_external','eval_private','track_2','track_3'))


def test_wrapper_exclusive_GPU_lease_original_geometry_offline_hard_timeout():
    wrapper=(Path(__file__).parents[1]/'infra/run_native_pose_sqp_real.sh').read_text()
    assert 'run_native_pose_sqp_real/code' in wrapper and 'native-pose-sqp-real-$REV' in wrapper
    assert '--gpus all' in wrapper and 'flock -n 8' in wrapper and '--network none' in wrapper
    assert '903s docker' in wrapper and '--cidfile' in wrapper and 'container_absence_verified' in wrapper
    assert 'vendor/video_to_data' in wrapper and 'jobs/$B_SOURCE/run_native_joint_real' in wrapper
    assert '--label "world_reward.native_pose_sqp_real.owner=$REV"' in wrapper


def test_failed_native_constructor_keeps_A_B_and_actual_native_A_reference(tmp_path,monkeypatch):
    bank,src,b,_=tiny();src['forward']=dict(decoder_identity={});src['native'].update(parameters(96));b['cfg']={"gates":{v:1.00000001 for v in run.replay.METRIC_GATE_KEYS.values()}}
    b['evidence']=SimpleNamespace();b['report']=dict(baseline_binding={},metrics={run.saved.A_NAME:{},run.saved.B_NAME:{}})
    monkeypatch.setenv('WR_CODE_REVISION','f'*40)
    monkeypatch.setattr(run.native,'original_sources',lambda *_:(bank,src))
    monkeypatch.setattr(run.real.saved,'load_npz',lambda *_:{})
    monkeypatch.setattr(run.real.contact,'hand_indices',lambda *_:src['QA_hand_ids'])
    monkeypatch.setattr(run.saved,'saved_B',lambda *_:b)
    monkeypatch.setattr(run.native,'quality',lambda *_:{})
    monkeypatch.setattr(run.replay,'layer_factory',lambda *_:SimpleNamespace(decoder_identity=lambda:src['forward']['decoder_identity']))
    monkeypatch.setattr(run.replay,'decode_geometry',lambda *_:dict(human_vertices=src['human'].copy()))
    monkeypatch.setattr(run,'prime_native_autograd',lambda *_:dict(persistent_inference_tensors=0))
    def fail(*_):raise ValueError('Original native constructor activation mismatch')
    monkeypatch.setattr(run,'native_instance',fail)
    row=run.run_episode(9,tmp_path/'episode_000009',SimpleNamespace(),{},SimpleNamespace(cuda=SimpleNamespace(empty_cache=lambda:None)),dict(deadline=1e100),Path('.'))
    out=tmp_path/'episode_000009'
    assert row['status']=='fail' and (out/'A_B_QA.json').is_file() and (out/'baseline_witness_reference.npz').is_file()
    assert (out/'report.json').is_file() and not (out/'target.npy').exists()


def test_measured_callback_budget_abstains_without_dropping_work(monkeypatch):
    monkeypatch.setattr(run.time,'monotonic',lambda:100.)
    calls=[dict(kind='full_native_gradient_and_same_ID_VJP',seconds=5.),dict(kind='full_native_objective',seconds=3.)]
    with pytest.raises(run._BudgetExhausted):run.budget_for_next_step(112.,calls,[dict(seconds=3.)],2.)
    run.budget_for_next_step(115.,calls,[dict(seconds=3.)],2.)
    run.budget_for_next_step(100.,[],[],0.) # no estimate before authentic first callbacks


def test_native_encoding_gauge_receipt_reports_only_numeric_not_quality():
    p=parameters(96);p['mhr_body_pose_cont'][:,:138]*=2.;p['mhr_body_pose_cont'][:,138:254]*=3.
    r=run.control_gauge_receipt(p)
    assert r['SO3_first_norm_min']==pytest.approx(2.,abs=2e-7)
    assert r['SO2_radius_max']==pytest.approx(3.,abs=3e-7)
    assert r['raw_encoding_gauge_preserved_under_retraction']
    assert not any('gain' in k or 'validated' in k for k in r)


def test_runtime_primes_lazy_native_before_any_inference_decode_and_retains_tracebacks():
    text=(Path(__file__).parents[1]/'infra/native_pose_sqp_real.py').read_text()
    assert text.index("report['native_autograd_priming']=prime_native_autograd")<text.index('native_a=decode(a_parameters)')
    assert "native_autograd_priming.json" in text
    assert "failure_traceback" in text and "traceback.extract_tb" in text
    adapter=(Path(__file__).parents[1]/'src/world_reward/native_pose_sqp_adapter.py').read_text()
    assert 'with torch.inference_mode(False),torch.enable_grad()' in adapter
