"""Tiny saved-array/AST controls; no Docker, original metadata, model or GPU."""
from dataclasses import asdict
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
SCRIPT=ROOT/'infra/run_mhr_scale_span.sh'
TEXT=SCRIPT.read_text()
DIAG=TEXT.split("<<'PYDIAG'\n",1)[1].split('\nPYDIAG',1)[0]


def saved_fixture(path):
    path.mkdir();(path/'native').mkdir()
    arrays={n:np.zeros((1,),np.float32)for n in ('bounds','parents','transform','lbs_indices','lbs_weights','faces',
        'hand_pose_mean','hand_pose_comps','hand_left_indices','hand_right_indices')}
    arrays['scale_mean']=np.zeros(68,np.float32);arrays['scale_mean'][-1]=1
    arrays['scale_comps']=np.eye(28,68,dtype=np.float32)
    np.savez(path/'native/mhr_metadata.npz',**arrays)
    h=hashlib.sha256();h.update(b'dict')
    for n in sorted(arrays):
        a=arrays[n];h.update(repr(n).encode());h.update(str((a.shape,a.dtype.str)).encode());h.update(a.tobytes())
    (path/'native/mhr_metadata.json').write_text(json.dumps(dict(parameter_names=['manufactured']*249,
        joint_names=['manufactured']*127,network_executed=False,metadata_sha256=h.hexdigest())))
    return arrays


def run_manufactured(path):
    program='''import os,sys
from pathlib import Path
os.geteuid=lambda:1000
old=Path.iterdir
Path.iterdir=lambda p:iter([Path('lo')])if str(p)=='/sys/class/net'else old(p)
exec(%r)
'''%DIAG
    return subprocess.run([sys.executable,'-B','-c',program,str(ROOT),str(path)],
        env=dict(os.environ,PYTHONPATH=str(ROOT/'src')),capture_output=True,text=True)


def test_manufactured_saved_array_span_certificate_no_fits(tmp_path):
    path=tmp_path/'fixture';arrays=saved_fixture(path);before=(path/'native/mhr_metadata.npz').read_bytes()
    result=run_manufactured(path);assert result.returncode==0,result.stderr
    r=json.loads(result.stdout)
    assert r['status']=='diagnostic_complete'and r['nonmembership_certified']is True
    assert r['result']['rank_matrix']==28 and r['result']['rank_augmented']==29
    assert r['result']['status']=='EXACT_NOT_IN_AFFINE_SPAN'and r['membership_certified']is False
    assert r['models_loaded']is False and r['fit_performed']is False and r['adoption']is False
    assert (path/'native/mhr_metadata.npz').read_bytes()==before
    assert r['result']['input_identities'][0][3]==hashlib.sha256(arrays['scale_comps'].T.tobytes()).hexdigest()


def test_changed_saved_array_fingerprint_abstains(tmp_path):
    path=tmp_path/'fixture';arrays=saved_fixture(path);arrays['scale_mean'][0]=2
    np.savez(path/'native/mhr_metadata.npz',**arrays)
    result=run_manufactured(path)
    assert result.returncode!=0 and 'fingerprint differs'in result.stderr and not result.stdout


def test_source_abi_and_read_only_budget_mount_contract():
    subprocess.run(['bash','-n',str(SCRIPT)],check=True)
    for marker in('PYAUTH','PYDIAG','PYFINAL'):
        ast.parse(TEXT.split("<<'"+marker+"'\n",1)[1].split('\n'+marker,1)[0])
    assert 'remaining=$((75-(SECONDS-START)))'in TEXT and 'bounded 31 docker run'in TEXT
    assert 'signal.alarm(30)'in DIAG and 'allow_pickle=False'in DIAG
    assert 'gate.affine_span_gate(comps.T,mean,np.zeros(68,np.float32))'in DIAG
    assert 'rt.source(root,root/\'jobs\'/old/'in TEXT and "historical['entries']==245"in TEXT
    assert "('producer_revision','markers','entries','closure_sha256')"in TEXT
    assert '[[ "$(auth)" == "$BEFORE" ]]'in TEXT
    assert '--gpus'not in TEXT and 'flock'not in TEXT and 'torch'not in DIAG.replace("'torch'in sys.modules",'')
    assert '$ROOT/weights'not in TEXT and '$ROOT/data'not in TEXT and '--mount "type=bind,src=$BASE/'in TEXT
    assert 'docker run --rm -i --network none --read-only --cap-drop ALL'in TEXT


def test_final_stdout_only_after_posthash_and_certificate_or_inconclusive():
    final=TEXT.split("<<'PYFINAL'\n",1)[1].split('\nPYFINAL',1)[0]
    for status in ('EXACT_NOT_IN_AFFINE_SPAN','INCONCLUSIVE','MEMBER'):
        proof={'source':{'manufactured':True},'old_source':{},'input_identities':{}}
        value={'status':'diagnostic_complete','result':{'status':status}}
        r=subprocess.run([sys.executable,'-c',final,json.dumps(proof),json.dumps(value)],capture_output=True,text=True)
        assert (r.returncode==0)is(status!='MEMBER')
        if not r.returncode:assert json.loads(r.stdout)['source_inputs_rehashed_after']is True
