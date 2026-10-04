"""Tiny orchestration/source contracts; never claim real native qualification."""
import ast
import copy
import importlib.util
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'infra'))
spec=importlib.util.spec_from_file_location('joint_point_native_qualify',ROOT/'infra/joint_point_native_qualify.py')
q=importlib.util.module_from_spec(spec);spec.loader.exec_module(q)


def counters():
    return {k:0 for k in ('constructor_attempts','constructor_returns','probe_attempts','probe_returns','run_attempts','run_returns')}


def exercise(*,probe_change=False,result_change=False,constructor_fail=False):
    events=[];report=counters()
    def construct(arm):
        events.append(('construct',arm))
        if constructor_fail:raise RuntimeError('manufactured failure')
        return arm
    def probe(arm,step):
        events.append(('probe',arm,step))
        return {'step':step,'gradient':'different' if probe_change and arm.startswith('B') else 'exact'}
    def run(arm):
        events.append(('run',arm));return {'history':(0,100,200,300),'updates':301,
            'parameter_bytes':b'changed' if result_change and arm.startswith('B') else b'original'}
    def execute():
        q.paired_execution(construct,probe,run,lambda:events.append(('reset',)),
            lambda:events.append(('release',)),lambda _:dict(initial='same'),lambda value,_:copy.deepcopy(value),report,lambda:None)
    return execute,events,report


def test_pair_order_and_complete_counts():
    execute,events,report=exercise();execute()
    assert events==[(kind,*args) for arm in ('A_original','B_point_weight_zero') for kind,args in
        [('reset',()),('construct',(arm,)),('probe',(arm,0)),('probe',(arm,181)),('run',(arm,)),('release',())]]
    assert report['constructor_attempts']==report['constructor_returns']==2
    assert report['probe_attempts']==report['probe_returns']==4
    assert report['run_attempts']==report['run_returns']==2
    assert report['actual_native_updates_total']==602


def test_probe_difference_stops_before_second_run():
    execute,events,report=exercise(probe_change=True)
    with pytest.raises(ValueError,match='initial'):execute()
    assert report['run_attempts']==1 and ('run','B_point_weight_zero')not in events


def test_result_difference_stops_without_parity_claim():
    execute,_,report=exercise(result_change=True)
    with pytest.raises(ValueError,match='result'):execute()
    assert report['run_returns']==2 and 'exact_full_result_history_parity'not in report


def test_attempt_distinct_from_constructor_return():
    execute,_,report=exercise(constructor_fail=True)
    with pytest.raises(RuntimeError):execute()
    assert report['constructor_attempts']==1 and report['constructor_returns']==0 and report['run_attempts']==0


def test_fixed_arguments():
    assert q.selected_episode(['--episode','21'])==21
    for argv in ([],['--episode','0'],['--episode','21','--episode','21'],['--ep','21']):
        with pytest.raises((SystemExit,ValueError)):q.selected_episode(argv)


def test_no_torch_or_models_at_import():
    top=ast.parse((ROOT/'infra/joint_point_native_qualify.py').read_text())
    imports=[n for n in top.body if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('torch'not in ast.unparse(n) and 'numpy'not in ast.unparse(n) for n in imports)


def test_actual_original_calls_and_control_scope():
    source=(ROOT/'infra/joint_point_native_qualify.py').read_text()
    assert 'optimizer.MHRParityPostOptimizer' in source and 'instance.run()' in source
    assert 'instance.loss(indices,step,include_diagnostics=True)' in source and 'total.backward()' in source
    assert 'checkpoint_path=None' in source and 'PointObjectiveConfig(1.,0.,REFERENCE)' in source
    assert 'np.zeros((spec.total_frames,q),bool)' in source
    assert "['depth_h5']" in source and 'inferred_camera(spec)' in source
    assert 'instance._object_state(indices,include_surface=False)' in source
    assert source.index('evidence=numerical_control') < source.index('return extension(source')
    assert 'torch.use_deterministic_algorithms'not in source and 'setattr('not in source
    assert q.PROBES==(0,181) and q.BUDGET==7200 and q.FRAMES==563


def test_wrapper_narrow_offline_scope_and_actual_markers():
    source=(ROOT/'infra/run_joint_point_native_qualify.sh').read_text()
    assert '--network none --read-only --cap-drop ALL' in source and '--gpus all' in source
    assert '7203s docker run' in source and '--memory 64g --cpus 4' in source
    assert '/proc/$$/fd/9' in source and 'nvidia-smi --query-compute-apps=pid' in source
    assert 'q.host_mounts' in source and 'q.source_binding' in source
    assert 'world-reward.revision' in source and 'docker rm -f "$cid"' in source
    assert 'validation/'not in source and 'refined.pth'not in source


def test_complete_runtime_closure():
    import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes() for folder in ('infra','src/world_reward','configs')
        for p in (ROOT/folder).rglob('*') if p.is_file()}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    paths=azure_job.runtime_bundle_paths(files,'infra/run_joint_point_native_qualify.sh')
    assert set(q.HELPERS)<=set(paths)
    assert 'infra/mediapipe_cpu_runtime_verify.py'in paths


def test_shell_syntax():
    import subprocess
    subprocess.run(['bash','-n',str(ROOT/'infra/run_joint_point_native_qualify.sh')],check=True)
