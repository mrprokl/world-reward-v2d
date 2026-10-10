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


def test_explicit_queued_GPU_wait_bounded_same_inode_and_immutable_source():
    wrapper=(Path(__file__).parents[1]/'infra/run_native_pose_sqp_real.sh').read_text()
    assert '--after-gpu-lock' not in wrapper and '"$1" == --after-terminal' in wrapper and 'flock -w "$REMAINING" 8' in wrapper
    assert 'else\n exec 8<"$LOCK"\n flock -n 8' in wrapper # legacy mode unchanged
    assert 'os.fstat(8)' in wrapper and 'value.st_nlink!=1' in wrapper
    assert '$(lock_identity fd)' in wrapper and '$(source_identity)' in wrapper
    assert wrapper.index('flock -w "$REMAINING" 8')<wrapper.index('903s docker run')
    assert '! -e "$OUT" && ! -L "$OUT"' in wrapper
    assert not any(x in wrapper for x in ('truncate -','systemctl restart','Result=success'))


def test_queued_lock_exact_existing_regular_singlelink_canonical_guard(tmp_path):
    import subprocess
    wrapper=(Path(__file__).parents[1]/'infra/run_native_pose_sqp_real.sh').read_text()
    code=wrapper.split("<<'PYLOCK'\n",1)[1].split('\nPYLOCK',1)[0]
    lock=tmp_path/'lock';lock.touch()
    def guard(path):return subprocess.run([sys.executable,'-I','-B','-',str(path),'path'],input=code,text=True,capture_output=True)
    expected=lock.stat();valid=guard(lock)
    assert valid.returncode==0 and valid.stdout.strip()==f'{expected.st_dev}:{expected.st_ino}'
    alias=tmp_path/'alias';alias.symlink_to(lock)
    assert guard(alias).returncode!=0
    alias.unlink();alias.hardlink_to(lock)
    assert guard(lock).returncode!=0
    assert guard(tmp_path).returncode!=0


def test_queue_fd_inode_must_equal_path_not_replaced_lock(tmp_path):
    import subprocess
    wrapper=(Path(__file__).parents[1]/'infra/run_native_pose_sqp_real.sh').read_text()
    code=wrapper.split("<<'PYLOCK'\n",1)[1].split('\nPYLOCK',1)[0]
    lock=tmp_path/'lock';other=tmp_path/'other';lock.touch();other.touch()
    shell='exec 8<"$1"; exec "$2" -I -B - "$3" fd'
    def guard(opened):return subprocess.run(['bash','-c',shell,'guard',str(opened),sys.executable,str(lock)],input=code,text=True,capture_output=True)
    assert guard(lock).returncode==0
    assert guard(other).returncode!=0


def test_explicit_whole_unit_terminal_wait_precedes_GPU_lease_without_claiming_success():
    import subprocess
    wrapper=(Path(__file__).parents[1]/'infra/run_native_pose_sqp_real.sh').read_text()
    assert '"$1" == --after-terminal' in wrapper
    assert wrapper.index('unit_state; (( UNIT_READY )) && break')<wrapper.index('exec 8<"$LOCK"')
    assert '43200-(SECONDS-WAIT_STARTED)' in wrapper
    code=wrapper.split('unit_state() {',1)[1].split('\nif (( WAIT_GPU )); then',1)[0]
    fn='unit_state() {'+code
    def check(active,result,main,load='loaded',pid=0):
        row=f'LoadState={load}\nActiveState={active}\nResult={result}\nExecMainStatus={main}\nMainPID={pid}'
        shell='set -e; WAIT_FOR=world-reward-whole.service; systemctl(){ printf "%s\\n" "$STATE"; }; '+fn+'\nunit_state; printf "%s" "$UNIT_READY"'
        return subprocess.run(['bash','-c',shell],env={'STATE':row},capture_output=True,text=True)
    for args,ready in [(('inactive','success',0),'1'),(('failed','exit-code',1),'1'),(('active','success',0),'0'),(('failed','timeout',0),'1')]:
        r=check(*args);assert r.returncode==0 and r.stdout==ready
    for args in [('inactive','exit-code',1),('failed','success',0),('failed','exit-code',0),('inactive','success',0,'not-found'),('inactive','success',0,'loaded',123),('failed','exit-code',1,'loaded',123)]:
        assert check(*args).returncode!=0
