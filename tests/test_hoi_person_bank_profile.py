"""Six-frame diagnostic profile/mount controls; no native/network/GT inference."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]


def driver():
    spec=importlib.util.spec_from_file_location('bank_gate',ROOT/'infra/hoi_detr_model_qualify.py')
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def config(m):return json.loads((ROOT/m.PERSON_PROTOCOL).read_text())


def test_explicit_honest_profile_not_external_or_heldout():
    m=driver();p=config(m)
    raw=(ROOT/m.PERSON_PROTOCOL).read_bytes()
    assert m.PERSON_PROTOCOL_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    rt=SimpleNamespace(pinned=lambda *_:p)
    assert m.protocol(rt,ROOT,person_bank=True)==p
    assert m.selected_protocol_pin(p)==m.PERSON_PROTOCOL_PIN
    assert m.evidence_flags(p)['challenge_inputs_used'] is True
    assert all(v is False for k,v in m.evidence_flags(p).items() if k!='challenge_inputs_used')
    assert 'external_RGB' not in p and 'external_cohort' not in p and 'procedural_RGB' not in p
    assert p['person_bank_profile']['zero_pair_outputs_are_valid'] is True
    with pytest.raises(ValueError):m.protocol(rt,ROOT,person_bank=True,external_rgb=True)


@pytest.mark.parametrize('fault',['challenge_false','ownership','manual','census','empty','schema'])
def test_profile_fail_closed(fault):
    m=driver();p=copy.deepcopy(config(m))
    if fault=='challenge_false':p['challenge_inputs_used']=False
    elif fault=='ownership':p['person_bank_profile']['ownership_verified']=True
    elif fault=='manual':p['person_bank_profile']['hand_labeled_test']=True
    elif fault=='census':p['person_bank_profile']['banks'].pop()
    elif fault=='empty':p['person_bank_profile']['zero_pair_outputs_are_valid']=False
    else:p['schema']='world_reward.hoi_detr_external_cohort.v1'
    with pytest.raises(ValueError):m.protocol(SimpleNamespace(pinned=lambda *_:p),ROOT,person_bank=True)


def test_mounts_exact_two_videos_six_banks_receipt_no_masks_or_sourcecalibration(tmp_path):
    m=driver();p=config(m);proof=dict(runtime=dict(image=dict(Id='sha256:'+'a'*64)),pin=dict(bytes=1,sha256='b'*64))
    _,_,mounts,args=m.container_plan(tmp_path/'code',tmp_path/'out','c'*40,p,proof,'model',1800)
    leaves=m.person_bank_mounts(p)
    assert len(leaves)==9 and sum(x.suffix=='.npz' for x in leaves)==6 and sum(x.suffix=='.mp4' for x in leaves)==2
    assert all((x,x,True) in mounts for x in leaves) and args[-1]=='--person-bank'
    assert not any('automatic_masks' in str(x) or 'calib' in str(x) or 'body_full' in str(x) for x,_,_ in mounts)
    _,_,cpu,_=m.container_plan(tmp_path/'code',tmp_path/'out','c'*40,p,proof,'overlay',1800)
    assert not any(x in leaves for x,_,_ in cpu)


def test_full_native_bootstrap_and_old_protocols_unchanged():
    m=driver();new=(ROOT/'infra/hoi_detr_model_qualify.py').read_text()
    old=subprocess.check_output(['rtk','proxy','git','show','ea3de92a337648e4d1b6c8f50a157a7868cbeba4:infra/hoi_detr_model_qualify.py'],text=True)
    names=('strict_checkpoint','register_original_checkpoint_buffers','configuration_policy','cpu_overlay',
           'authenticate_runtime','authenticate_acquisition','derive_source','cleanup','validate_container')
    def functions(s):return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
    a,b=functions(old),functions(new)
    assert all(a[n]==b[n] for n in names)
    # Keep bootstrap/loading/native numeric policy literally identical before the new scope.
    assert old.split('def gpu_model(',1)[1].split("    external = 'external_RGB'",1)[0]==new.split('def gpu_model(',1)[1].split("    external = 'external_RGB'",1)[0]
    for name,pin in [(m.PROTOCOL,m.PROTOCOL_PIN),(m.IMAGE_PROTOCOL,m.IMAGE_PROTOCOL_PIN),(m.COHORT_PROTOCOL,m.COHORT_PROTOCOL_PIN),(m.COHORT128_PROTOCOL,m.COHORT128_PROTOCOL_PIN)]:
        raw=(ROOT/name).read_bytes();assert pin==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
