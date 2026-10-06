"""Tiny original metadata controls; no Torch/NumPy/session/image execution."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import dwpose_metadata_context as m

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def delegates(monkeypatch):
    identity=m.rt.identity
    # Working source is writable locally; production requires immutable files.
    def authored(path,maximum,**kwargs):
        p=Path(path);raw=p.read_bytes()
        return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(m.rt,'identity',authored)
    result=m.metadata_delegate(ROOT)
    monkeypatch.setattr(m.rt,'identity',identity)
    return result


def test_exact_original_definitions_and_constants_only(delegates):
    smoke,capability=delegates
    assert not hasattr(smoke,'np') and not hasattr(smoke,'perform') and not hasattr(smoke,'SessionProxy')
    assert not hasattr(capability,'np') and not hasattr(capability,'public_inputs')
    assert {k for k in vars(smoke) if k.isupper()}==set(m.SMOKE_CONSTANTS)
    assert {k for k in vars(capability) if k.isupper()}==set(m.CAPABILITY_CONSTANTS)
    actual=smoke.source_identity()
    assert actual['dwpose_smoke.py']==m.PINS['infra/dwpose_smoke.py'] and actual['dwpose_acquire.py']==m.PINS['infra/dwpose_acquire.py']
    for name in m.SMOKE_NAMES:
        assert getattr(smoke,name).__code__.co_filename==str(ROOT/'infra/dwpose_smoke.py')
    assert capability.SMOKE_SOURCE_SHA==actual['dwpose_smoke.py']['sha256']


@pytest.mark.parametrize('fault',['changed','foreign_acquisition','foreign_audit'])
def test_delegate_rejects_changed_pin_or_cached_foreign_import(monkeypatch,fault):
    def identity(path,maximum):
        raw=Path(path).read_bytes();return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    if fault=='changed':
        base=identity
        def identity(path,maximum):
            result=base(path,maximum)
            return dict(result,sha256='0'*64) if Path(path).name=='dwpose_smoke.py' else result
    else:
        actual=m.importlib.import_module
        def imported(name):
            module=actual(name)
            return SimpleNamespace(__file__='/foreign/'+name+'.py') if name==('dwpose_acquire' if fault=='foreign_acquisition' else 'dwpose_wheel_audit') else module
        monkeypatch.setattr(m.importlib,'import_module',imported)
    monkeypatch.setattr(m.rt,'identity',identity)
    with pytest.raises(ValueError):m.metadata_delegate(ROOT)


def metadata_receipts(smoke,capability):
    previous=dict(stage='native_dwpose_rgb133_cpu_abi_and_replay',status='fail',phase='native_source_load',
        producer_revision=smoke.PREVIOUS_SMOKE_REVISION,script_sha256=smoke.PREVIOUS_SMOKE_SOURCE_SHA,image_id=smoke.audit.IMAGE,
        error_type='ValueError',error='Require native two SimCC133 graph outputs',sessions=[],private_prefix_packages_installed=True,
        private_prefix_removed=True,device='cpu',network='none',native_cpu_abi_verified=False,two_session_byte_replay_verified=False,
        gpu_used=False,own_feed_cast=False,native_source_modified=False,private_truth_read=False,ground_truth_used=False,
        challenge_inputs_used=False,oracle_modes=[],adoption_authorized=False)
    passed=dict(stage=smoke.STAGE,status='pass',phase='complete',producer_revision=capability.SMOKE_REVISION,
        script_sha256=capability.SMOKE_SOURCE_SHA,image_id=smoke.audit.IMAGE,previous_failed_smoke_sha256=smoke.PREVIOUS_SMOKE_SHA,
        previous_failure_rewritten=False,device='cpu',network='none',native_cpu_abi_verified=True,two_session_byte_replay_verified=True,
        private_prefix_packages_installed=True,private_prefix_removed=True,final_inputs_source_assets_rehashed=True,
        native_source_modified=False,own_feed_cast=False,channel_swap=False,full_image_fallback=False,raw_scores_clamped=False,
        confidence_threshold_applied=False,wrapper_neck134_used=False,global_image_modified=False,gpu_used=False,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        independent_quality_cohort=False,accuracy_verified=False,license_clearance_verified=False,adoption_authorized=False)
    predictions=[dict(file=str(i),keypoints=dict(hash=str(i)),scores=dict(hash=str(i)),validity=dict(hash=str(i)),raw_simcc=[dict(hash=str(i))]) for i in range(2)]
    passed['sessions']=[dict(predictions=copy.deepcopy(predictions),calls=[dict(run_completed=True),dict(run_completed=True)]) for _ in range(2)]
    return previous,passed


@pytest.mark.parametrize('fault',['none','old_rewritten','replay','calls','old_pin','pass_pin','oracle'])
def test_original_smoke_and_previous_failure_rules_unchanged(delegates,monkeypatch,tmp_path,fault):
    smoke,capability=delegates;previous,passed=metadata_receipts(smoke,capability)
    if fault=='old_rewritten':previous['status']='pass'
    elif fault=='replay':passed['sessions'][1]['predictions'][0]['keypoints']['hash']='different'
    elif fault=='calls':passed['sessions'][0]['calls'][0]['run_completed']=False
    elif fault=='oracle':passed['oracle_modes']=['truth']
    paths={tmp_path/smoke.PREVIOUS_SMOKE:previous,tmp_path/smoke.OUT/'report.json':passed}
    for path,row in paths.items():path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(row))
    def identity(path,sha=None,size=None):
        wanted=smoke.PREVIOUS_SMOKE_SHA if Path(path)==tmp_path/smoke.PREVIOUS_SMOKE else capability.SMOKE_SHA
        if fault==('old_pin' if Path(path)==tmp_path/smoke.PREVIOUS_SMOKE else 'pass_pin'):wanted='0'*64
        if sha!=wanted:raise ValueError('authored exact pin mismatch')
        return dict(bytes=Path(path).stat().st_size,sha256=wanted)
    monkeypatch.setitem(smoke.identity.__globals__,'identity',identity)
    monkeypatch.setattr(smoke,'identity',identity)
    if fault!='none':
        with pytest.raises(ValueError):capability.validate_smoke(tmp_path)
    else:
        before=copy.deepcopy((previous,passed));result=capability.validate_smoke(tmp_path)
        assert result['actual_pass_receipt']['sha256']==capability.SMOKE_SHA and result['previous_failed_receipt']['sha256']==smoke.PREVIOUS_SMOKE_SHA
        assert (previous,passed)==before


@pytest.mark.parametrize('fault',['missing','duplicate','order','nonliteral'])
def test_strict_definition_and_literal_allowlist(fault):
    raw=b'VALUE=3\nEXCLUDED=999\ndef a(): return VALUE\ndef b(): return a()\n'
    if fault=='missing':raw=raw.replace(b'VALUE=3\n',b'')
    elif fault=='duplicate':raw=b'VALUE=2\n'+raw
    elif fault=='order':raw=b'VALUE=3\ndef b():return a()\ndef a():return VALUE\n'
    else:raw=raw.replace(b'VALUE=3',b'VALUE=arbitrary_callback()')
    with pytest.raises(ValueError):m._definitions(Path('/authored'),raw,('a','b'),('VALUE',),{})


def test_host_import_and_delegate_under_heavy_import_denial():
    script='''import sys,hashlib,importlib.abc
from pathlib import Path
class Deny(importlib.abc.MetaPathFinder):
 def find_spec(self,name,path=None,target=None):
  if name.split('.')[0] in ('numpy','torch','PIL','onnxruntime','transformers','dwpose_smoke','keypoint_rgb_dwpose'):raise ImportError('denied')
sys.meta_path.insert(0,Deny());sys.path[:0]=['infra','src'];import dwpose_metadata_context as m
m.rt.identity=lambda p,maximum:dict(bytes=len(Path(p).read_bytes()),sha256=hashlib.sha256(Path(p).read_bytes()).hexdigest())
smoke,capability=m.metadata_delegate(Path.cwd());assert not hasattr(smoke,'np');smoke.source_identity()
'''
    result=subprocess.run([sys.executable,'-I','-B','-c',script],capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    tree=ast.parse(Path(m.__file__).read_text())
    assert not any(isinstance(n,ast.Assign) and any(isinstance(t,ast.Attribute) for t in n.targets) for n in ast.walk(tree))
