"""Tiny lineage/firewall checks only, not proof of native CUDA joint fitting."""
import importlib.util
import json
from pathlib import Path
import subprocess

import numpy as np
import pytest


@pytest.fixture
def runtime(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('own_identity_probe_runtime',infra/'cari_identity_probe_runtime.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def sealed(runtime,tmp_path,monkeypatch):
    out=tmp_path/'outputs/episode_000000/cari_converter_diagnostic_refined_v1';(out/'sealed_original').mkdir(parents=True)
    errors=np.full(790,2.1,np.float32)
    converted=dict(pose=np.zeros((790,136),np.float32),scales=np.zeros(68,np.float32),shape=np.zeros(45,np.float32),
        valid_input=np.ones(790,bool),per_frame_vertex_error_mm=errors,
        report=dict(frames=790,fitted_frames=790,invalid_input_frames=[],precision='float32',
                    vertex_error_mm=dict(mean=float(errors.mean()),worst_frame_mean=float(errors.max()))))
    p=out/'sealed_original/original_converter.npz';np.savez_compressed(p,**{k:v for k,v in converted.items() if k!='report'},report=np.array(json.dumps(converted['report'])))
    params={k:np.zeros((790,dim),np.float32) for k,dim in runtime.convert.PARAMETER_DIMS.items()}
    p=out/'native_parameters.npz';np.savez_compressed(p,**params,frame_index=np.arange(790,dtype=np.int64))
    (out/'sealed_original/original_report.json').write_text('{}')
    report=dict(stage='world_reward_cari_converter_runtime_diagnostic',status='pass',phase='complete',episode_index=0,bundle_source='refined',frames=790,
        research_only=True,network='none',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],submission_produced=False,adoption_authorized=False,
        original_gate_pass=False,source_bindings_runtime_verified=True,native_full_frame_decode_verified=True,provenance=runtime.PINS.copy(),
        original_report_sha256=runtime.sha256(out/'sealed_original/original_report.json'),
        original_archive=dict(sha256=runtime.sha256(out/'sealed_original/original_converter.npz')),
        native_parameters_sha256=runtime.sha256(out/'native_parameters.npz'),
        native_parameter_identities={k:runtime.convert.canonical_array_identity(v) for k,v in params.items()})
    p=out/'report.json';p.write_text(json.dumps(report));monkeypatch.setattr(runtime,'SEALED_SHA',runtime.sha256(p))
    return tmp_path,out,report


def test_source_read_is_frozen_complete_and_no_original_reconvert(runtime,sealed):
    root,out,report=sealed;original,params,receipt,frozen=runtime.sealed_inputs(root)
    assert len(original['pose'])==790 and set(params)==set(runtime.convert.PARAMETER_DIMS) and len(frozen)==4
    assert receipt==report


@pytest.mark.parametrize('fault',['source','pin','gt','phase','frames','truth_added','changed_param','symlink','index_order'])
def test_sealed_source_mismatch_fatal_before_model_import(runtime,sealed,monkeypatch,fault):
    root,out,report=sealed
    if fault=='source':report['unrecorded_change']=True
    elif fault=='pin':report['provenance']['converter_sha256']='0'*64
    elif fault=='gt':report['ground_truth_used']=True
    elif fault=='phase':report['phase']='probe_polish'
    elif fault=='frames':report['frames']=789
    elif fault in ['truth_added','changed_param','index_order']:
        p=out/'native_parameters.npz'
        with np.load(p,allow_pickle=False) as f:data={k:f[k].copy() for k in f.files}
        if fault=='truth_added':data['ground_truth']=np.zeros(1)
        elif fault=='changed_param':data['mhr_shape'][0,0]=1
        else:data['frame_index'][1]=0
        np.savez_compressed(p,**data);report['native_parameters_sha256']=runtime.sha256(p)
    elif fault=='symlink':
        p=out/'sealed_original/original_converter.npz';p.rename(out/'archive.npz');p.symlink_to(out/'archive.npz')
    p=out/'report.json';p.write_text(json.dumps(report))
    if fault!='source':monkeypatch.setattr(runtime,'SEALED_SHA',runtime.sha256(p))
    with pytest.raises((ValueError,RuntimeError)):runtime.sealed_inputs(root)


def test_unknown_argument_fails_before_runtime(runtime):
    with pytest.raises(SystemExit):runtime.main(['--loosen-mm','7'])


def test_offline_wrapper_research_only_provenance_and_no_reconvert(runtime):
    p=Path(runtime.__file__).with_name('run_cari_identity_probe.sh');subprocess.run(['bash','-n',str(p)],check=True)
    text=p.read_text();source=Path(runtime.__file__).read_text()
    assert '--network none' in text and '--memory 32g' in text and '903s' in text
    assert 'src=$ROOT/outputs,dst=$ROOT/outputs,readonly' in text and 'src=$OUT,dst=$OUT' in text
    assert 'converter.convert(' not in source and 'submission_produced=False' in source
    assert 'target vertices changed' in source and 'Original native source/assets changed during probe' in source
