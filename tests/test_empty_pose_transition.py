"""Tiny inactive-unit/source/topology failure archival tests, never model/data work."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import types

import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
SPEC=importlib.util.spec_from_file_location('empty_pose_transition_test',REPO/'infra/empty_pose_transition.py')
gate=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(gate)


def save(path,raw,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(raw);path.chmod(mode);return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())

def code_at(root,rev,entry,names):
    code=root/'jobs'/rev/entry/'code'
    for name in names:save(code/name,('procedural immutable '+name).encode())
    save(code.parent/'revision',(rev+'\n').encode());save(code.parent/'source-sha256',('c'*64+'\n').encode())
    for p in(code,*code.rglob('*')):p.chmod(0o555 if p.is_dir()else 0o444)
    return code


def failed(pid=54321):
    return dict(LoadState='loaded',ActiveState='failed',SubState='failed',Result='exit-code',ExecMainCode='1',ExecMainStatus='1',MainPID='0',ExecMainPID=str(pid),ControlGroup='')


@pytest.fixture
def case(tmp_path,monkeypatch):
    root=tmp_path;rev='b'*40;oldrev='a'*40;episode=6
    code=code_at(root,rev,gate.JOB,gate.SOURCES)
    oldcode=code_at(root,oldrev,'run_track1_frontends',('infra/run_track1_frontends.sh','infra/run_object_pose_smoke.sh','infra/object_pose_smoke.py'))
    raw=b'Traceback (most recent call last):\nValueError: No topology-preserving approximation fits both budgets; procedural\n'+json.dumps(dict(mode='public_frontends_only',stage='object_pose_full',phase='fail'),separators=(',',':')).encode()+b'\n'
    logpin=save(root/'results/track1-episode6-frontends.log',raw,0o600)
    original=root/'outputs/episode_000006/object_pose_full';original.mkdir(parents=True,mode=0o755)
    args=types.SimpleNamespace(episode=episode,failed_unit='world-reward-track1-episode6-frontends.service',failed_revision=oldrev,failed_entrypoint='run_track1_frontends',failed_pid=54321,
        failed_script_sha256=gate.archive.identity(oldcode/'infra/object_pose_smoke.py')['sha256'],failed_log_sha256=logpin['sha256'],failed_log_bytes=logpin['bytes'])
    proc=root/'proc';proc.mkdir();cgroups=root/'cgroups';cgroups.mkdir()
    original_lstat=Path.lstat
    def lstat(path):
        value=original_lstat(path)
        if path in(original,original.parent/'object_pose_topology_failed_v1'):
            values={k:getattr(value,k)for k in dir(value)if k.startswith('st_')};values['st_uid']=1000;return types.SimpleNamespace(**values)
        return value
    monkeypatch.setattr(Path,'lstat',lstat);monkeypatch.setattr(gate.os,'geteuid',lambda:0)
    volume=dict(pin=dict(bytes=3,sha256='d'*64),files={'all7actual':dict(bytes=4,sha256='e'*64)},report_status='pass')
    monkeypatch.setattr(gate,'volume_binding',lambda *args:copy.deepcopy(volume))
    return dict(root=root,code=code,rev=rev,oldcode=oldcode,args=args,original=original,log=root/'results/track1-episode6-frontends.log',proc=proc,cgroups=cgroups,volume=volume)


def validate(case,record=None):return gate.validate(case['root'],case['code'],case['rev'],case['args'],lambda _:record or failed(),case['proc'],case['cgroups'])

def perform(plan,case,**kwargs):
    def rename(src,dst):assert not dst.exists();src.rename(dst)
    return gate.transition(plan,probe=lambda _:failed(),rename=rename,proc=case['proc'],cgroups=case['cgroups'],gpu=lambda:None,lock=lambda _:('procedural','held'),sync=lambda _:None,**kwargs)


def test_preflight_pure_then_exact_empty_inode_archive(case):
    plan=validate(case);before=gate.state(case['original']);logbefore=gate.state(case['log'])
    assert not plan['destination'].exists()and not plan['receipt_dir'].exists()
    receipt=perform(plan,case);target=plan['receipt_dir']/'report.json'
    assert not case['original'].exists()and not list(plan['destination'].iterdir())
    assert gate.state(plan['destination'])[:8]==before[:8]and gate.state(case['log'])==logbefore
    assert receipt['original_status']=='fail'and receipt['original_failure_reinterpreted']is False and receipt['files_deleted']is False and receipt['geometry_changed']is False
    assert json.loads(target.read_bytes())==receipt and stat.S_IMODE(target.stat().st_mode)==0o400
    with pytest.raises(FileNotFoundError):validate(case)


@pytest.mark.parametrize('fault',('nonempty','occupied','log','script','markers','current','receipt'))
def test_preflight_failures_do_not_move_original(case,fault):
    if fault=='nonempty':save(case['original']/'unexpected.npz',b'not predictions')
    if fault=='occupied':(case['original'].parent/'object_pose_topology_failed_v1').mkdir()
    if fault=='log':case['log'].write_bytes(b'changed')
    if fault=='script':case['args'].failed_script_sha256='f'*64
    if fault=='markers':p=case['oldcode'].parent/'revision';p.chmod(0o600);p.write_bytes(b'changed\n')
    if fault=='current':case['code'].chmod(0o755)
    if fault=='receipt':(case['root']/'results'/f"empty-pose-transition-000006-{case['rev']}").mkdir()
    with pytest.raises((ValueError,FileNotFoundError)):validate(case)
    assert case['original'].is_dir()


@pytest.mark.parametrize('field,value',(('ActiveState','active'),('MainPID','54321'),('Result','success'),('ExecMainStatus','0'),('ExecMainPID','9999'),('ControlGroup','/unowned'),('LoadState','error')))
def test_inactive_unit_strict(case,field,value):
    value_record=failed();value_record[field]=value
    with pytest.raises(ValueError):validate(case,value_record)
    assert case['original'].is_dir()


def test_gc_notfound_requires_original_pid_absent(case):
    record=failed();record.update(LoadState='not-found',ActiveState='inactive',SubState='dead',Result='success',ExecMainCode='0',ExecMainStatus='0',ExecMainPID='0')
    assert validate(case,record)['unit']==record
    (case['proc']/str(case['args'].failed_pid)).mkdir()
    with pytest.raises(ValueError):validate(case,record)


def test_original_cgroup_descendant_live_process_rejected(case):
    record=failed();record['ControlGroup']='/system.slice/'+case['args'].failed_unit
    p=case['cgroups']/record['ControlGroup'].lstrip('/');save(p/'cgroup.procs',b'\n');save(p/'child/cgroup.procs',b'12345\n')
    with pytest.raises(ValueError):validate(case,record)
    (p/'child/cgroup.procs').chmod(0o644);(p/'child/cgroup.procs').write_text('');assert validate(case,record)['unit']==record


@pytest.mark.parametrize('fault',('nonempty','log','source','volume'))
def test_toctou_revalidation_before_move(case,fault,monkeypatch):
    plan=validate(case)
    if fault=='nonempty':save(case['original']/'unexpected',b'x')
    if fault=='log':case['log'].write_bytes(b'changed')
    if fault=='source':p=case['oldcode']/'infra/object_pose_smoke.py';p.chmod(0o644);p.write_bytes(b'changed');p.chmod(0o444)
    if fault=='volume':monkeypatch.setattr(gate,'volume_binding',lambda *a:dict(changed=True))
    with pytest.raises(ValueError):perform(plan,case)
    assert case['original'].is_dir()


def test_gpu_and_lock_failure_before_rename(case):
    plan=validate(case)
    def busy():raise ValueError('gpu busy')
    with pytest.raises(ValueError):gate.transition(plan,gpu=busy,lock=lambda _:('test','inode'))
    assert case['original'].exists()


def test_actual_volume_inventory_gate_called_without_decode(case,monkeypatch):
    code=case['code'];code.chmod(0o755);(code/'configs').mkdir();path=code/'configs/volume_mesh_000006_pins.json'
    paths=gate.volume.paths(6);pins=dict(report=dict(producer_revision='d'*40,script_sha256='e'*64),input_sha256='f'*64,metric_scale_baked_once=2.5,files={paths['object']:dict(sha256='a'*64),paths['alignment']:dict(sha256='b'*64)})
    save(path,json.dumps(pins).encode());(code/'configs').chmod(0o555);code.chmod(0o555)
    calls=[]
    def verify(*args):calls.append(args);return dict(status='pass'),{'all7':'verified'}
    monkeypatch.setattr(gate.volume,'verify_pinned_artifacts',verify)
    monkeypatch.setattr(gate,'binding',lambda *args:dict(files={gate.volume.PRODUCER_SCRIPT:dict(sha256='e'*64)}))
    # Restore actual stdlib binding function rather than fixture's placeholder.
    spec=importlib.util.spec_from_file_location('empty_pose_volume_fixture',REPO/'infra/empty_pose_transition.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    monkeypatch.setattr(module,'binding',gate.binding);value=module.volume_binding(case['root'],code,6)
    assert value['report_status']=='pass'and calls[0][2:]==(6,'f'*64,'a'*64,'b'*64,2.5)
    monkeypatch.setattr(gate.volume,'verify_pinned_artifacts',lambda *args:(_ for _ in ()).throw(ValueError('actual gate failure')))
    with pytest.raises(ValueError):module.volume_binding(case['root'],code,6)


def test_cli_exact_controls_and_duplicate_rejection():
    args=['--episode','6','--failed-entrypoint','run_track1_frontends','--failed-unit','world-reward-track1-episode6-frontends.service','--failed-revision','a'*40,
          '--failed-script-sha256','b'*64,'--failed-log-sha256','c'*64,'--failed-log-bytes','95538','--failed-pid','1062403']
    assert gate.parser().parse_args(args).failed_pid==1062403
    for extra in (['--episode','6'],['--resume'],['--failed-entrypoint','other']):
        with pytest.raises((ValueError,SystemExit)):gate.parser().parse_args(args+extra)


def test_native_noreplace_has_no_nonatomic_fallback(tmp_path,monkeypatch):
    if os.uname().sysname!='Linux':
        with pytest.raises(ValueError):gate.archive.rename_noreplace(tmp_path/'none',tmp_path/'other')
    else:
        original=tmp_path/'old';original.mkdir();destination=tmp_path/'new';destination.mkdir()
        with pytest.raises(OSError):gate.archive.rename_noreplace(original,destination)
        assert original.is_dir()and destination.is_dir()


def test_wrapper_has_existing_fd9_and_actual_source_closure():
    path=REPO/'infra/run_empty_pose_transition.sh';text=path.read_text()
    assert subprocess.run(['bash','-n',str(path)],capture_output=True).returncode==0
    assert 'exec 9<"$LOCK";flock --nonblock 9'in text and 'exec env -i' in text
    assert 'docker'not in text and 'data/'not in text and 'rm 'not in text
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for d in ('infra','configs','src')for p in(REPO/d).rglob('*')if p.is_file()and p.suffix!='.pyc'}
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_empty_pose_transition.sh'))
    assert set(gate.SOURCES)<=selected


def test_garbage_collected_unit_still_checks_derived_cgroup(case):
    record=failed();record.update(LoadState='not-found',ActiveState='inactive',SubState='dead',ExecMainPID='0')
    p=case['cgroups']/'system.slice'/case['args'].failed_unit;save(p/'cgroup.procs',b'777\n')
    with pytest.raises(ValueError):validate(case,record)
