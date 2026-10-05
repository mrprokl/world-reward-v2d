"""Frozen qualification lifecycle spies; native execution remains Azure only."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture
def gate(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'infra'))
    spec=importlib.util.spec_from_file_location('surface_replay_control_test',ROOT/'infra/object_budget_solid.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


def test_replay_protocol_exact_and_source_closure(gate):
    raw=(ROOT/gate.SURFACE_REPLAY_PROTOCOL).read_bytes()
    assert gate.SURFACE_REPLAY_PIN==dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    protocol=json.loads(raw);assert protocol['native_QEM_calls']==2 and len(protocol['mutation_negatives'])==16
    gate.surface_profile('surface','surface_replay_v1')
    assert gate.SURFACE_SECONDS==600 and gate.SURFACE_HOST_SECONDS==700
    text=(ROOT/'infra/object_budget_solid.py').read_text()
    assert text.index("Every fixed preflight must precede QEM")<text.index('for name,(v,f)in rows[:2]')
    assert 'Original complete authored native preflight required'in text
    assert 'maximum_qem_calls=2 if control'in text


def test_replay_exact_cli_no_episode_or_mixed_controls(gate,monkeypatch,tmp_path):
    monkeypatch.setattr(gate,'ROOT',tmp_path);rev='a'*40
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE',str(tmp_path/'code'));monkeypatch.setenv('WR_CODE_REVISION',rev)
    monkeypatch.setattr(sys,'argv',['driver','--domain','surface','--control','surface_replay_v1'])
    monkeypatch.setattr(gate,'surface_modules',lambda _:())
    calls=[];monkeypatch.setattr(gate,'surface_host',lambda *a,**kw:calls.append((a,kw))or 0)
    assert gate.main()==0 and calls[0][0][0]==-1 and calls[0][1]==dict(control='surface_replay_v1')
    for tail in (['--episode','26'],['--native','--native'],['--control','surface_consumer_v1']):
        monkeypatch.setattr(sys,'argv',['driver','--domain','surface','--control','surface_replay_v1',*tail])
        with pytest.raises(ValueError):gate.main()


def test_all_preflights_before_qem_and_positive_stdout_enforced(gate,monkeypatch,tmp_path):
    protocol=json.loads((ROOT/gate.SURFACE_REPLAY_PROTOCOL).read_bytes())
    rt=SimpleNamespace(pinned=lambda *_:protocol,strict=json.loads,identity=lambda *_:dict(bytes=1,sha256='0'*64))
    import surface_replay_adapter as adapter
    monkeypatch.setattr(adapter,'verifier',lambda _:lambda *_:None)
    import surface_qslim_qualify as q
    calls=[]
    def run(binary,args,left,counts,kind,expected=0):
        counts[kind+'_attempts']+=1;counts[kind+'_returns']+=1;calls.append(kind)
        if kind=='qem':pytest.fail('Unexpected QEM before positive preflight authentication')
        return SimpleNamespace(stdout=b'{}',stderr=b'')
    stub=SimpleNamespace(fixtures=q.fixtures,topology=q.topology,write_obj=lambda *_:None,run_native=run,equal=q.equal)
    with pytest.raises(ValueError,match='preflight'):
        gate.surface_replay_control(ROOT,tmp_path/'no_binary',tmp_path,lambda:600,{},rt,stub,None)
    assert calls==['preflight']


def test_progress_scalar_flush_not_geometry_and_bounded(gate,capsys):
    report={};gate.surface_checkpoint(report,'mapping_replay',0.,committed_collapses=17)
    line=capsys.readouterr().out
    assert line.startswith('WORLD_REWARD_SURFACE_PROGRESS_V1:')and report['phase']=='mapping_replay'
    with pytest.raises(ValueError):gate.surface_checkpoint(report,'x'*3000,0.)


def test_replay_owned_cleanup_allowlist_and_foreign_refusal(gate,tmp_path):
    import mediapipe_cpu_runtime_verify as rt
    work=tmp_path/'scratch';work.mkdir();owner=work.lstat()
    (work/'fresh_curved_holed_patch.mapping.json').write_bytes(b'owned')
    (work/'foreign.npz').write_bytes(b'foreign')
    with pytest.raises(ValueError):gate.surface_remove_work(work,owner,rt,control='surface_replay_v1')
    (work/'foreign.npz').unlink();gate.surface_remove_work(work,owner,rt,control='surface_replay_v1');assert not work.exists()
