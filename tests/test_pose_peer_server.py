"""Tiny server control/source contracts; no real SSH, keys, Azure or GPU."""
import base64
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'infra'))
import pose_peer_server as server
import pose_peer_inputs as inputs
from test_pose_peer_run import transport,dispatch,seal,write
from test_pose_peer_inputs import cohort


def key():return 'ssh-ed25519 '+base64.b64encode(struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)+b'x'*32).decode()


@pytest.mark.parametrize('value',['','ssh-rsa AAAA','ssh-ed25519 AAAA',key()+' comment',key()+'\n',key().replace(' ','  '),key()[:-1]])
def test_only_canonical_public_client_key(value):
    with pytest.raises((ValueError,Exception)):server.public_key(value)
    assert server.public_key(key())==key()
    assert server.public_key(key()+' harmless',True)==key()


def test_forced_command_actual_receiver_namespace_and_environment(monkeypatch,tmp_path):
    root=tmp_path/'root';monkeypatch.setattr(inputs,'ROOT',root);rev='a'*40;receiver=root/'jobs'/rev/'run_pose_peer_receive/code'
    forced=server.forced_command(receiver,rev,8)
    assert forced.startswith('/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent')
    assert f'WR_CODE={receiver}'in forced and forced.endswith(f'/bin/bash {receiver}/infra/run_pose_peer_receive.sh --episode 8')
    assert 'SSH_CONNECTION="$SSH_CONNECTION"'in forced and 'SSH_ORIGINAL_COMMAND="$SSH_ORIGINAL_COMMAND"'in forced
    for foreign in(root/'jobs'/rev/'run_pose_peer_server/code',Path('/foreign')):
        with pytest.raises(ValueError):server.forced_command(foreign,rev,8)
    config=server.configuration(tmp_path/'control',forced);values=dict(line.split(' ',1)for line in config.splitlines())
    assert values['ForceCommand']==forced and values['ListenAddress']=='10.0.0.9'and values['Port']=='2222'
    assert all(values[k]=='no'for k in('PasswordAuthentication','KbdInteractiveAuthentication','UsePAM','PermitUserEnvironment','PermitUserRC','PermitTTY','AllowTcpForwarding','AllowAgentForwarding','X11Forwarding','PermitTunnel'))
    assert values['PermitRootLogin']=='forced-commands-only'and 'AcceptEnv'not in values and 'Include'not in values


@pytest.fixture
def fixture(transport,tmp_path,monkeypatch):
    root=transport['root'];rev='f'*40;code=dispatch(root,rev,'run_pose_peer_server',server.HELPERS)
    write(code/'configs/pose_peer_000008_pins.json',inputs.digest_json(transport['pins']));seal(code)
    runtime=tmp_path/'run';runtime.mkdir();monkeypatch.setattr(server,'RUNTIME',runtime)
    monkeypatch.setattr(inputs,'ROOT',root);calls=[];fault={};state={'started':False}
    original_directory=server.directory
    def directory(path,owner=0,private=False):
        if path==Path('/run/sshd'):return(1,2,0,0o40755)
        value=original_directory(path,os.geteuid(),private)
        return value[:2]+(owner,value[3])
    monkeypatch.setattr(server,'directory',directory)
    monkeypatch.setattr(server.os,'statvfs',lambda path:type('S',(),dict(f_bavail=100*1024**3,f_frsize=1))())
    original_lstat=Path.lstat
    def root_private(path,*args,**kwargs):
        result=original_lstat(path,*args,**kwargs)
        if path.name=='host_ed25519':values=list(result);values[4]=0;return os.stat_result(values)
        return result
    monkeypatch.setattr(Path,'lstat',root_private)
    original_open=Path.open
    def no_private_reads(path,*args,**kwargs):
        if path.name=='host_ed25519':assert (args[0]if args else kwargs.get('mode'))in('wb','xb'),'Python must not read private hostkey'
        return original_open(path,*args,**kwargs)
    monkeypatch.setattr(Path,'open',no_private_reads)
    def command(args,*a,**kw):
        calls.append(args)
        if args[0]=='/usr/bin/findmnt':return json.dumps({'filesystems':[dict(target=str(inputs.DEST),uuid='foreign'if fault.get('disk')else server.UUID,options='rw,noatime')]})
        if args[0]=='/usr/bin/ss':
            if fault.get('port'):return 'LISTEN 0 128 0.0.0.0:2222 0.0.0.0:* users:(("foreign",pid=123,fd=3))\n'
            return 'LISTEN 0 128 10.0.0.9:2222 0.0.0.0:* users:(("sshd",pid=4321,fd=3))\n'if state['started']else ''
        if args[0]=='/usr/bin/ssh-keygen':
            private=Path(args[-1]);private.write_bytes(b'not a real key; test only');private.chmod(0o600);private.with_suffix('.pub').write_text(key()+' fixture\n');return ''
        if args[0]=='/usr/bin/systemd-run':state['started']=True;return ''
        if args[0]=='/usr/bin/systemctl':
            if '--property=LoadState'in args:return 'loaded\n'if fault.get('unit')else 'not-found\n'
            if '--property=ExecStart'in args:
                unit=args[2];control=runtime/f'world-reward-pose-peer-000008-{rev}'
                return f'ExecStart={{ path=/usr/sbin/sshd ; argv[]=/usr/sbin/sshd -D -e -f {control}/sshd_config ; }}\nFragmentPath=/run/systemd/transient/{unit}\n'
            if args[1]=='stop':return ''
            return 'ActiveState=failed\nMainPID=4321\n'if fault.get('start')else 'ActiveState=active\nMainPID=4321\n'
        return ''
    monkeypatch.setattr(server,'command',command)
    return dict(root=root,code=code,revision=rev,runtime=runtime,calls=calls,fault=fault,transport=transport)


def test_byte_exact_receiver_copy_no_alias_existing_namespace_refused(fixture):
    receiver,before=server.copy_receiver(fixture['code'],fixture['revision'])
    assert receiver==fixture['root']/'jobs'/fixture['revision']/'run_pose_peer_receive/code'
    assert inputs.code_binding(receiver,fixture['revision'],{})==before
    assert {str(p.relative_to(receiver))for p in receiver.rglob('*')}=={str(p.relative_to(fixture['code']))for p in fixture['code'].rglob('*')}
    for source in fixture['code'].rglob('*'):
        if source.is_file():
            copied=receiver/source.relative_to(fixture['code']);assert copied.read_bytes()==source.read_bytes()and copied.stat().st_ino!=source.stat().st_ino
    with pytest.raises(ValueError):server.copy_receiver(fixture['code'],fixture['revision'])


def test_actual_mocked_start_public_receipt_only_and_no_incoming_creation(fixture):
    value=fixture;report,control=server.start(value['root'],value['code'],value['revision'],8,key())
    assert report['status']=='pass'and report['transport_verified']is False and report['private_key_read']is False
    assert report['host_public_key']==key()and report['main_pid']==4321 and report['source_rehashed_after']is True
    assert not(inputs.DEST/f'pose-peer-000008-{value["revision"]}').exists()
    assert (control/'server-receipt.json').stat().st_mode&0o777==0o400
    authorized=(control/'authorized_keys').read_text();assert authorized.startswith('from="10.0.0.4",restrict,command="/usr/bin/env -i ')
    assert '\\"$SSH_CONNECTION\\"'in authorized and authorized.endswith(' '+key()+'\n')
    argv=next(a for a in value['calls']if a[0]=='/usr/bin/systemd-run');assert '--property=RuntimeMaxSec=2400'in argv
    assert argv[-5:]==['/usr/sbin/sshd','-D','-e','-f',str(control/'sshd_config')]
    assert 'not a real key'not in str(report)


@pytest.mark.parametrize('fault',['port','unit','disk','incoming','control','receiver','source'])
def test_preflight_refusal_no_key_generation(fixture,fault):
    value=fixture
    if fault=='incoming':(inputs.DEST/f'pose-peer-000008-{value["revision"]}').mkdir(mode=0o700)
    elif fault=='control':(value['runtime']/f'world-reward-pose-peer-000008-{value["revision"]}').mkdir(mode=0o700)
    elif fault=='receiver':(value['code'].parents[1]/'run_pose_peer_receive').mkdir()
    elif fault=='source':(value['code']/'infra/pose_peer_receive.py').chmod(0o644)
    else:value['fault'][fault]=True
    with pytest.raises(ValueError):server.start(value['root'],value['code'],value['revision'],8,key())
    assert not any(a[0]in('/usr/bin/ssh-keygen','/usr/bin/systemd-run')for a in value['calls'])


def test_failure_stops_only_bound_owned_unit_and_retains_receipt(fixture):
    fixture['fault']['start']=True;report,control=server.start(fixture['root'],fixture['code'],fixture['revision'],8,key())
    assert report['status']=='fail'and report['owned_stop_succeeded']is True
    assert any(a[:2]==['/usr/bin/systemctl','stop']for a in fixture['calls'])
    assert (control/'server-receipt.json').exists()


@pytest.mark.parametrize('projection',['ExecStart=foreign\nFragmentPath=/run/systemd/transient/unit.service\n','ExecStart=path=/usr/sbin/sshd\nFragmentPath=/etc/systemd/system/unit.service\n'])
def test_cleanup_never_stops_foreign_unit(monkeypatch,tmp_path,projection):
    calls=[];monkeypatch.setattr(server,'command',lambda args:(calls.append(args),projection)[1])
    with pytest.raises(ValueError):server.owned_stop('unit.service',tmp_path)
    assert len(calls)==1 and 'stop'not in calls[0]


def test_bounded_command_sanitized(monkeypatch):
    calls=[]
    def fake(args,**kwargs):calls.append((args,kwargs));return subprocess.CompletedProcess(args,1,'NEVER_PRINT','NEVER_PRINT')
    monkeypatch.setattr(server.subprocess,'run',fake)
    with pytest.raises(ValueError)as caught:server.command(['/usr/bin/systemctl','show','owned'])
    assert 'NEVER_PRINT'not in str(caught.value)
    assert calls[0][0][:4]==['/usr/bin/timeout','--signal=TERM','--kill-after=2s','8s']and calls[0][1]['timeout']==12
    assert set(calls[0][1]['env'])=={'PATH','HOME'}


def test_wrapper_syntax_and_static_complete_receiver_closure():
    subprocess.run(['rtk','proxy','bash','-n',str(ROOT/'infra/run_pose_peer_server.sh')],check=True)
    import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in('infra','src','configs')for p in(ROOT/folder).rglob('*')if p.is_file()and '__pycache__'not in p.parts};files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_pose_peer_server.sh')
    assert set(server.HELPERS)<=set(selected)
    source=(ROOT/'infra/pose_peer_server.py').read_text()
    assert 'docker'not in source and 'nvidia-smi'not in source and 'azure'not in source.lower()


def test_listener_startup_wait_is_bounded_without_false_empty_race(monkeypatch):
    calls=[];sleeps=[]
    def projection(args):
        calls.append(args)
        return ''if len(calls)<3 else 'LISTEN 0 128 10.0.0.9:2222 0.0.0.0:* users:(("sshd",pid=4321,fd=3))\n'
    monkeypatch.setattr(server,'command',projection);monkeypatch.setattr(server.time,'sleep',sleeps.append)
    server.listener_pid('4321');assert len(calls)==3 and sleeps==[.2,.2]
    calls.clear();sleeps.clear();monkeypatch.setattr(server,'command',lambda args:(calls.append(args),'')[1])
    with pytest.raises(ValueError,match='startup budget'):server.listener_pid('4321')
    assert len(calls)==10 and sleeps==[.2]*9


@pytest.mark.parametrize('projection',['LISTEN 0 128 0.0.0.0:2222 0.0.0.0:* users:(("sshd",pid=4321,fd=3))\n',
    'LISTEN 0 128 10.0.0.9:2222 0.0.0.0:* users:(("foreign",pid=999,fd=3))\n'])
def test_foreign_listener_immediately_rejected(monkeypatch,projection):
    calls=[];monkeypatch.setattr(server,'command',lambda args:(calls.append(args),projection)[1])
    with pytest.raises(ValueError):server.listener_pid('4321')
    assert len(calls)==1
