"""Tiny forced-export/listener/pull controls only; no SSH, GPU, models or Azure."""
import base64
import copy
import io
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import types

import pytest

REPO=Path(__file__).resolve().parents[1];sys.path.insert(0,str(REPO/'infra'))
import pose_return_inputs as payload
import pose_return_inventory as inventory
import pose_return_export as exporter
import pose_return_server as server
import pose_return_pull as puller
from test_pose_return_inputs import result_fixture,write,dispatch,seal,archive_fixture
from test_pose_peer_inputs import cohort


def key():return 'ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+b'x'*32).decode()


@pytest.fixture
def sealed_archive(result_fixture):
    v=result_fixture;report=inventory.inventory(v['root'],v['code'],v['revision'],8)
    out=v['root']/'results'/f'pose-return-inventory-000008-{v["revision"]}'
    pins=dict(schema='world_reward.pose_return.pins.v1',episode_index=8,source_pins=v['sourceid'],manifest=report['manifest'],archive=report['archive'],
        inventory_report=dict(path=str((out/'report.json').relative_to(v['root'])),**payload.identity(out/'report.json',True),
            producer_revision=v['revision'],script_sha256=report['script_sha256']))
    v.update(pins=pins,archive=out/'archive.tar')
    return v


def own_code(value,entry,helpers):
    revision='f'*40;code=dispatch(value['root'],revision,entry,helpers)
    write(code/'configs/pose_return_000008_source_pins.json',payload.raw_json(value['source']))
    write(code/'configs/pose_return_000008_pins.json',payload.raw_json(value['pins']));seal(code)
    return code,revision


@pytest.mark.parametrize('peer,command',[('10.0.0.5 123 10.0.0.9 2222','world-reward-pose-results-000008'),
    ('10.0.0.4 0 10.0.0.9 2222','world-reward-pose-results-000008'),('10.0.0.4 123 10.0.0.9 22','world-reward-pose-results-000008'),
    ('10.0.0.4 123 10.0.0.9 2222','world-reward-pose-results-000008;other')])
def test_export_only_exact_authorized_private_peer(peer,command):
    with pytest.raises(ValueError):exporter.authorize(peer,command,8)
    exporter.authorize('10.0.0.4 123 10.0.0.9 2222','world-reward-pose-results-000008',8)


def test_forced_export_exact_stream_and_original_source_post(sealed_archive):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_server',exporter.HELPERS);stream=io.BytesIO()
    exporter.export(v['root'],code,rev,8,stream)
    assert stream.getvalue()==v['archive'].read_bytes()
    original=v['source']['outputs']
    assert all(payload.identity(v['folder']/n,True)==row for n,row in original.items())


def test_inventory_mutation_blocks_export_before_any_bytes(sealed_archive):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_server',exporter.HELPERS)
    path=v['root']/v['pins']['inventory_report']['path'];write(path,b'{"status":"pass"}')
    stream=io.BytesIO()
    with pytest.raises(ValueError):exporter.export(v['root'],code,rev,8,stream)
    assert stream.getvalue()==b''


def test_ssh_exact_private_host_key_options_and_no_agent_forward():
    argv=puller.ssh_arguments(Path('/owned/key'),Path('/owned/known'),8)
    assert argv[:6]==['/usr/bin/ssh','-F','/dev/null','-T','-p','2222']and argv[-2:]==['root@10.0.0.9','world-reward-pose-results-000008']
    assert 'StrictHostKeyChecking=yes'in argv and 'IdentityAgent=none'in argv and 'GlobalKnownHostsFile=/dev/null'in argv
    assert '-A'not in argv and '-L'not in argv and '-R'not in argv


def test_forced_exporter_same_actual_server_snapshot_no_receiver_clone(monkeypatch,tmp_path):
    monkeypatch.setattr(payload,'ROOT',tmp_path);code=tmp_path/'jobs'/('f'*40)/'run_pose_return_server/code'
    forced=server.forced(code,'f'*40,8)
    assert forced.endswith(f'/bin/bash {code}/infra/run_pose_return_export.sh --episode 8')and 'SSH_CONNECTION="$SSH_CONNECTION"'in forced
    config=server.control.configuration(tmp_path/'control',forced)
    assert 'ListenAddress 10.0.0.9\nPort 2222'in config and 'PermitRootLogin forced-commands-only'in config
    assert 'PermitTTY no'in config and 'AllowAgentForwarding no'in config and 'AllowTcpForwarding no'in config
    with pytest.raises(ValueError):server.forced(tmp_path/'jobs'/('f'*40)/'run_pose_return_pull/code','f'*40,8)


def test_accept_exact_three_bytes_only_original_target_uid_modes(sealed_archive,tmp_path,monkeypatch):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_pull',puller.HELPERS)
    receiverroot=tmp_path/'vm01';(receiverroot/'outputs/episode_000008').mkdir(parents=True)
    out=receiverroot/'results/new';(out/'stream').mkdir(parents=True,mode=0o700)
    write(out/'stream/archive.tar',v['archive'].read_bytes(),0o400)
    before=payload.binding(code,rev,puller.HELPERS)[0];_,_,pinid=payload.load_transport(code,8)
    monkeypatch.setattr(puller.os,'chown',lambda *args:None)
    def promote(source,target):assert not target.exists();source.rename(target)
    monkeypatch.setattr(payload,'promote',promote)
    target=puller.accept(receiverroot,code,rev,8,out,v['pins'],v['source'],before,pinid)
    assert target==receiverroot/'outputs/episode_000008/object_pose_full'and target.stat().st_uid==1000
    assert set(p.name for p in target.iterdir())==set(payload.FILES)
    assert all(payload.identity(target/n,True)==wanted for n,wanted in v['source']['outputs'].items())
    assert (out/'extracted/producer-report.json').exists()and not(target/'producer-report.json').exists()
    with pytest.raises(ValueError):puller.accept(receiverroot,code,rev,8,out,v['pins'],v['source'],before,pinid)


@pytest.mark.parametrize('fault',['source','archive','target'])
def test_accept_fails_before_native_promotion_on_tamper_or_existing(sealed_archive,tmp_path,monkeypatch,fault):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_pull',puller.HELPERS)
    root=tmp_path/'receiver';(root/'outputs/episode_000008').mkdir(parents=True);out=root/'results/new';(out/'stream').mkdir(parents=True)
    write(out/'stream/archive.tar',v['archive'].read_bytes(),0o400);before=payload.binding(code,rev,puller.HELPERS)[0];_,_,pinid=payload.load_transport(code,8)
    monkeypatch.setattr(puller.os,'chown',lambda *args:None)
    monkeypatch.setattr(payload,'promote',lambda *args:pytest.fail('Invalidstate reachedpromotion'))
    if fault=='source':write(code/'infra/pose_return_pull.py',b'changed source')
    elif fault=='archive':write(out/'stream/archive.tar',b'changed stream')
    else:(root/'outputs/episode_000008/object_pose_full').mkdir()
    with pytest.raises(ValueError):puller.accept(root,code,rev,8,out,v['pins'],v['source'],before,pinid)


@pytest.mark.parametrize('raw',[b'abc',b'abcde'])
def test_existing_exact_receiver_rejects_truncated_or_extra(tmp_path,raw):
    import hashlib
    out=tmp_path/'stream';out.mkdir(mode=0o700)
    report=puller.receiver.receive(io.BytesIO(raw),out,4,hashlib.sha256(b'abcd').hexdigest())
    assert report['status']=='fail'


def test_new_wrappers_and_runtime_closures_no_numeric_dependency_import():
    import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    for entry,module in(('inventory',inventory),('server',server),('pull',puller)):
        name=f'infra/run_pose_return_{entry}.sh';subprocess.run(['bash','-n',str(REPO/name)],check=True)
        selected=azure_job.runtime_bundle_paths(files,name)
        assert set(module.HELPERS)<=set(selected)
        # The existing code_binding whitelist keeps provenance-only historical
        # literals in the host closure; no tracker is imported or executed.
        for own in(module.__file__,payload.__file__):
            text=Path(own).read_text();assert 'import object_pose_smoke'not in text and 'import pose_peer_run'not in text
        assert not any(p.endswith(('.mp4','.npz','.glb'))for p in selected)
    subprocess.run(['bash','-n',str(REPO/'infra/run_pose_return_export.sh')],check=True)


@pytest.fixture
def server_fixture(sealed_archive,tmp_path,monkeypatch):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_server',server.HELPERS)
    runtime=tmp_path/'run';runtime.mkdir();monkeypatch.setattr(server.control,'RUNTIME',runtime)
    calls=[];fault={};state={'started':False};control_dir=runtime/f'world-reward-pose-return-000008-{rev}'
    original_directory=server.control.directory
    def directory(path,owner=0,private=False):
        if path==Path('/run/sshd'):return(1,2,0,0o40755)
        result=original_directory(path,os.geteuid(),private);return result[:2]+(owner,result[3])
    monkeypatch.setattr(server.control,'directory',directory)
    monkeypatch.setattr(server,'stat_key',lambda s:s.st_mode&0o777==0o600 and s.st_nlink==1)
    original_open=Path.open
    def no_private_reads(path,*args,**kwargs):
        if path.name=='host_ed25519':assert (args[0]if args else kwargs.get('mode'))in('wb','xb')
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_private_reads)
    def command(argv,*args,**kwargs):
        calls.append(argv)
        if argv[0]=='/usr/bin/ss':
            if fault.get('port'):return'LISTEN foreign'
            return'LISTEN 0 128 10.0.0.9:2222 0.0.0.0:* users:(("sshd",pid=4321,fd=3))\n'if state['started']else''
        if argv[0]=='/usr/bin/ssh-keygen':
            path=Path(argv[-1]);path.write_bytes(b'fake key not read');path.chmod(0o600);write(path.with_suffix('.pub'),(key()+' fixture\n').encode());return''
        if argv[0]=='/usr/bin/systemd-run':state['started']=True;return''
        if argv[0]=='/usr/bin/systemctl':
            if '--property=LoadState'in argv:return'loaded\n'if fault.get('unit')else'not-found\n'
            if '--property=ExecStart'in argv:return f'ExecStart={{ path=/usr/sbin/sshd ; argv[]=/usr/sbin/sshd -D -e -f {control_dir}/sshd_config ; }}\nFragmentPath=/run/systemd/transient/{argv[2]}\n'
            if argv[1]=='stop':return''
            return'ActiveState=failed\nMainPID=4321\n'if fault.get('start')else'ActiveState=active\nMainPID=4321\n'
        return''
    monkeypatch.setattr(server.control,'command',command)
    return dict(value=v,code=code,revision=rev,calls=calls,fault=fault,runtime=runtime,control=control_dir)


def test_server_mocked_real_control_binding_only_exporter_and_public_key(server_fixture):
    f=server_fixture;report,runtime=server.start(f['value']['root'],f['code'],f['revision'],8,key())
    assert report['status']=='pass'and report['transport_verified']is False and report['private_key_read']is False
    assert report['host_public_key']==key()and report['main_pid']==4321 and report['source_rehashed_after']
    authorized=(runtime/'authorized_keys').read_text()
    assert authorized.startswith('from="10.0.0.4",restrict,command="')and 'run_pose_return_export.sh --episode 8'in authorized
    assert '\\"$SSH_CONNECTION\\"'in authorized and 'fake key not read'not in str(report)
    argv=next(x for x in f['calls']if x[0]=='/usr/bin/systemd-run')
    assert '--property=RuntimeMaxSec=1200'in argv and '--property=Restart=no'in argv


@pytest.mark.parametrize('fault',['unit','port','control'])
def test_server_preflight_no_key_or_listener_on_occupied_namespace(server_fixture,fault):
    f=server_fixture
    if fault=='control':f['control'].mkdir(mode=0o700)
    else:f['fault'][fault]=True
    with pytest.raises(ValueError):server.start(f['value']['root'],f['code'],f['revision'],8,key())
    assert not any(x[0]in('/usr/bin/ssh-keygen','/usr/bin/systemd-run')for x in f['calls'])


def test_server_failed_start_stops_only_exact_bound_owned_unit(server_fixture):
    f=server_fixture;f['fault']['start']=True
    report,runtime=server.start(f['value']['root'],f['code'],f['revision'],8,key())
    assert report['status']=='fail'and report['owned_stop_succeeded']
    assert any(x[:2]==['/usr/bin/systemctl','stop']for x in f['calls'])
    assert (runtime/'server-receipt.json').exists()


@pytest.fixture
def pull_fixture(sealed_archive,tmp_path,monkeypatch):
    v=sealed_archive;code,rev=own_code(v,'run_pose_return_pull',puller.HELPERS)
    keypath=v['root']/'transfer/pose-return-client-000008-v1/client_ed25519';write(keypath,b'fake private key',0o600)
    keypath.parent.chmod(0o700);write(keypath.with_suffix('.pub'),(key()+'\n').encode(),0o400)
    v['folder'].rename(v['folder'].with_name('original_vm02_source'))
    monkeypatch.setattr(puller,'key_state',lambda path:('independentlycheckedmetadata',))
    monkeypatch.setattr(puller.os,'chown',lambda *args:None)
    monkeypatch.setattr(payload,'promote',lambda source,target:source.rename(target))
    calls=[];fault={};real_read=Path.open
    def safe_open(path,*args,**kwargs):
        assert path!=keypath,'Private key must never be opened by Python';return real_read(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',safe_open)
    class Process:
        def __init__(self,argv,**kwargs):
            calls.append((argv,kwargs));assert set(kwargs['env'])=={'PATH','HOME'}and kwargs['stdin']==subprocess.DEVNULL
            self.stdout=io.BytesIO(v['archive'].read_bytes());self.done=False
        def wait(self,timeout=None):self.done=True;return 1 if fault.get('ssh')else 0
        def poll(self):return 0 if self.done else None
        def terminate(self):calls.append('terminate');self.done=True
        def kill(self):calls.append('kill');self.done=True
    monkeypatch.setattr(puller.subprocess,'Popen',Process)
    return dict(value=v,code=code,revision=rev,calls=calls,fault=fault,key=keypath)


def test_pull_mocked_source_exit_and_byte_exact_native_publication(pull_fixture):
    f=pull_fixture;v=f['value'];report=puller.pull(v['root'],f['code'],f['revision'],8,key())
    assert report['status']=='pass'and report['output_promoted']and report['source_rehashed_after']and report['ssh_exit_status']==0
    assert report['frames']==3 and report['private_key_read']is False and report['quality_verified']is False
    folder=v['root']/'outputs/episode_000008/object_pose_full'
    assert set(p.name for p in folder.iterdir())==set(payload.FILES)
    assert all(payload.identity(folder/n,True)==wanted for n,wanted in v['source']['outputs'].items())
    assert folder.stat().st_mode&0o777==0o755
    with pytest.raises(ValueError):puller.pull(v['root'],f['code'],f['revision'],8,key())


def test_pull_source_exit_nonzero_never_promotes_even_complete_valid_stream(pull_fixture):
    f=pull_fixture;f['fault']['ssh']=True;v=f['value'];report=puller.pull(v['root'],f['code'],f['revision'],8,key())
    assert report['status']=='fail'and report['output_promoted']is False
    assert not(v['root']/'outputs/episode_000008/object_pose_full').exists()


def test_metadata_private_client_key_only_no_read(tmp_path,monkeypatch):
    path=tmp_path/'private/client_ed25519';write(path,b'fake private key',0o600);path.parent.chmod(0o700)
    old_lstat=Path.lstat
    def uid(path,*a,**k):
        s=old_lstat(path,*a,**k);values=list(s);values[4]=0;return os.stat_result(values)
    monkeypatch.setattr(Path,'lstat',uid)
    monkeypatch.setattr(Path,'open',lambda *a,**k:pytest.fail('Privatekeymetadata verifier openedbytes'))
    assert puller.key_state(path)
    path.chmod(0o644)
    with pytest.raises(ValueError):puller.key_state(path)
