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
    numeric=ast.unparse(next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name=='numerical_control'))
    assert 'observations'not in numeric


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


def first_mask_fixture(tmp_path):
    import json
    import numpy as np
    from PIL import Image
    code=tmp_path/'code';(code/'configs').mkdir(parents=True)
    path=tmp_path/'outputs/episode_000021/automatic_masks/masks/1/000000.png';path.parent.mkdir(parents=True)
    Image.fromarray(np.array([[0,255,0],[255,0,0]],np.uint8)).save(path)
    identity=q.runtime().identity(path,readonly=False)
    report=tmp_path/'outputs/episode_000021/object_pose_full/report.json';report.parent.mkdir()
    frames=[{'frame_index':i,'object_mask_sha256':identity['sha256']}for i in range(q.FRAMES)]
    report.write_text(json.dumps({'frames':frames}))
    relative=str(report.relative_to(tmp_path))
    (code/'configs/cari_clip_000021_input_pins.json').write_text(json.dumps({
        'source_files':{relative:q.runtime().identity(report,readonly=False)}}))
    return code,path,identity,report


def test_original_first_png_pin_and_binary_decoder(tmp_path):
    import numpy as np
    code,path,identity,_=first_mask_fixture(tmp_path)
    assert q.first_mask_binding(tmp_path,code)==(path,identity)
    mask=q.load_first_mask(path,identity,(2,3))
    assert mask.dtype==np.bool_ and mask.shape==(2,3) and mask.sum()==2 and not mask.flags.writeable
    assert np.array_equal(mask,[[False,True,False],[True,False,False]])
    assert not (path.parent/'000001.png').exists()


def test_first_mask_report_or_png_mutation_rejected(tmp_path):
    code,path,identity,report=first_mask_fixture(tmp_path)
    with pytest.raises(ValueError,match='full-grid'):q.load_first_mask(path,identity,(224,224))
    path.write_bytes(path.read_bytes()+b'changed')
    with pytest.raises(ValueError,match='identity'):q.first_mask_binding(tmp_path,code)
    with pytest.raises(ValueError,match='before decode'):q.load_first_mask(path,identity,(2,3))
    report.write_text('{}')
    with pytest.raises(ValueError,match='pose report'):q.first_mask_binding(tmp_path,code)


@pytest.mark.parametrize('value,mode',[(1,'L'),(0,'L'),(255,'RGB')])
def test_nonbinary_empty_or_rgb_first_mask_rejected(tmp_path,value,mode):
    import numpy as np
    from PIL import Image
    _,path,_,_=first_mask_fixture(tmp_path)
    pixels=np.full((2,3,3)if mode=='RGB'else(2,3),value,np.uint8)
    Image.fromarray(pixels).save(path)
    identity=q.runtime().identity(path,readonly=False)
    with pytest.raises(ValueError):q.load_first_mask(path,identity,(2,3))


def test_host_mounts_add_only_original_first_mask(tmp_path,monkeypatch):
    import json
    code,path,identity,report=first_mask_fixture(tmp_path)
    pin_path=code/'configs/cari_clip_000021_input_pins.json'
    pin=json.loads(pin_path.read_bytes());pin.update(schema='world-reward-cari-clip-input-pins-v1',
        clip_spec=dict(episode_index=21,total_frames=563,camera_name='front_stereo_camera_left',height=1152,width=1536))
    pin['source_files'].update({f'outputs/episode_000021/tiny_{i}.json':{}for i in range(14)})
    pin_path.write_text(json.dumps(pin));monkeypatch.setattr(q,'historical',lambda *_:{})
    mounts=q.host_mounts(tmp_path,code)
    assert [p for p in mounts if 'automatic_masks'in p.parts]==[path]
    assert path.parent not in mounts and all(p.name!='000001.png'for p in mounts)
