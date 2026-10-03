"""Tiny public contracts and sensor scoring only; no remote dataset or GPU."""
import copy
import importlib.util
from pathlib import Path
import numpy as np
import pytest

@pytest.fixture
def gate(monkeypatch):
    root=Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'infra'));monkeypatch.syspath_prepend(str(root/'src'))
    spec=importlib.util.spec_from_file_location('wr_whole_quality',root/'infra/tudl_whole_support_evaluate.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def fake_inputs(gate):
    ids=[[s,f] for s in (1,2,3) for f in (10,20,30,40)]
    identity=dict(bytes=1,sha256='a'*64)
    return dict(schema=gate.inputs.PINS_SCHEMA,ordered_frames=ids,
        acquisition_report={**identity,'producer_revision':'b'*40,'script_sha256':'c'*64},
        public_files={'manifest.json':identity.copy(),**{f'scene_{s:06d}_frame_{f:06d}.png':identity.copy() for s,f in ids}})

@pytest.mark.parametrize('fault',['schema','bytes','producer','outputs','coefficient','extra'])
def test_independent_complete_pins_failclosed(gate,fault):
    public=fake_inputs(gate);identity=dict(bytes=1,sha256='a'*64)
    pins=dict(schema='world_reward.tudl_whole_support_prediction_pins.v1',report={**identity,'producer_revision':'b'*40,'script_sha256':'c'*64},
        outputs={Path(n).stem+'.npz':identity.copy() for n in gate.inputs.filenames(public)},coefficients=[1.,1.,1.])
    gate.validate_pins(pins,public)
    if fault=='schema':pins['schema']='oldborder'
    elif fault=='bytes':pins['report']['bytes']=True
    elif fault=='producer':pins['report']['producer_revision']='c'*39
    elif fault=='outputs':pins['outputs'].pop(next(iter(pins['outputs'])))
    elif fault=='coefficient':pins['coefficients'][0]=np.nan
    else:pins['private_gt']='oracle'
    with pytest.raises(ValueError):gate.validate_pins(pins,public)

def test_same_sensor_scoring_views_no_alignment(gate):
    depth=np.array([[1.,2.]],np.float32);points=np.array([[[1.,2.,3.],[4.,5.,6.]]],np.float32);valid=np.ones((1,2),bool);k=np.eye(3)
    data={'scene_id':np.array(1),'frame_id':np.array(10),'K':k}
    for mode in ('moge','anchored'):data.update({mode+'_depth':depth,mode+'_points':points,mode+'_validity':valid})
    result=gate.score_arrays(data)
    assert result['fixed_points'] is points and result['learned_K'] is k and result['fixed_validity'] is valid
    assert gate.original_audit.scoring.SAMPLES==8192
    np.testing.assert_array_equal(gate.original_audit.scoring.sample_indices(10000),np.sort(np.random.default_rng(0).choice(10000,8192,replace=False)))
    # Truth sensor rays are integer BOP, predictions are not adjusted to them.
    xyz=gate.original_audit.scoring.sensor_points(depth,k,valid)
    np.testing.assert_array_equal(xyz,np.array([[0.,0.,1.],[2.,0.,2.]]))

def test_scene_mean4_median_gain_no_drop_or_scene_regression(gate):
    pairs=[tuple(x) for x in fake_inputs(gate)['ordered_frames']]
    rows=[dict(scene_id=s,frame_id=f,fixed={'sensor_visible_camera_chamfer_half_cm':100.,'sensor_valid_object_coverage':1.},
        learned={'sensor_visible_camera_chamfer_half_cm':90.,'sensor_valid_object_coverage':1.},exact_candidate_validity_matches_baseline=True) for s,f in pairs]
    result=gate.decision(rows,pairs)
    assert result['real_object_camera_hypothesis_supported'] and result['median_paired_scene_relative_gain']==.1
    assert not result['verified_victory_over_CARI4D'] and not result['independent_scenes_or_objects']
    changed=copy.deepcopy(rows)
    for row in changed[:4]:row['learned']['sensor_visible_camera_chamfer_half_cm']=106.
    assert not gate.decision(changed,pairs)['real_object_camera_hypothesis_supported']
    rows[0]['fixed']['sensor_valid_object_coverage']=rows[0]['learned']['sensor_valid_object_coverage']=.94
    assert not gate.decision(rows,pairs)['coverage_gate_pass']
    rows[0]['exact_candidate_validity_matches_baseline']=False
    with pytest.raises(ValueError):gate.decision(rows,pairs)
    with pytest.raises(ValueError):gate.decision(rows[:-1],pairs)

def test_private_never_precedes_full_public_validation(gate,monkeypatch,tmp_path):
    # Call-boundary simulation, not actual private dataset/eval evidence.
    import inspect
    source=inspect.getsource(gate.main)
    assert source.index('public_predictions(root, code)')<source.index('private_inputs(root, code, public, records)')
    source=inspect.getsource(gate.private_inputs)
    assert source.index('recheck(frozen)')<source.index('annotations[kind]=values')
    wrapper=(Path(__file__).resolve().parents[1]/'infra/run_tudl_whole_support_evaluate.sh').read_text()
    assert '--gpus' not in wrapper and 'weights/' not in wrapper and '--network none' in wrapper
    assert '183s docker run' in wrapper and '--name "$CONTAINER"' in wrapper
    assert 'docker ps -aq' in wrapper and '--kill-after=2s' in wrapper
