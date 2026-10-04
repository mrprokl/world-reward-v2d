"""Owned expiring VM02 SSH receiver; public pose inputs only, no cloud calls."""
from __future__ import annotations
import base64
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import signal
import stat
import struct
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pose_peer_inputs as inputs

UUID='24df126a-5f5f-41d8-801c-9ddaa7a582d8'
RUNTIME=Path('/run')
HELPERS=('infra/pose_peer_server.py','infra/run_pose_peer_server.sh','infra/run_pose_peer_receive.sh',
    'infra/pose_peer_receive.py','infra/frontend_peer_receive.py','infra/pose_peer_inputs.py')


def public_key(value,comment=False):
    inputs.require(type(value)is str and len(value)<512 and re.fullmatch(r'ssh-ed25519 [A-Za-z0-9+/]+={0,2}'+(r'(?: [^\x00-\x1f\x7f]+)?'if comment else ''),value),'Canonical single Ed25519 public key required')
    encoded=value.split(' ')[1];raw=base64.b64decode(encoded,validate=True)
    inputs.require(len(raw)==51 and raw[:19]==struct.pack('>I',11)+b'ssh-ed25519'+struct.pack('>I',32)and base64.b64encode(raw).decode()==encoded,'Canonical Ed25519 public blob required')
    return 'ssh-ed25519 '+encoded


def directory(path,owner=0,private=False):
    inputs.canonical(path);s=path.lstat()
    inputs.require(stat.S_ISDIR(s.st_mode)and s.st_uid==owner and not s.st_mode&0o022 and(not private or s.st_mode&0o777==0o700),'Expected existing owned directory required')
    return s.st_dev,s.st_ino,s.st_uid,s.st_mode


def command(args,seconds=8,missing=False):
    try:r=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=2s',str(seconds)+'s',*args],capture_output=True,text=True,timeout=seconds+4,env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'})
    except(OSError,subprocess.TimeoutExpired):raise ValueError('Bounded server control command unavailable')from None
    inputs.require(r.returncode==0 or(missing and r.returncode in(1,4)and r.stdout.strip()=='not-found'),'Server control command failed: '+Path(args[0]).name)
    inputs.require(len(r.stdout)<16384,'Bounded server control projection required');return r.stdout


def source(code,revision,index):
    required={name:inputs.identity(code/name,True,2_000_000)for name in HELPERS}
    binding=inputs.code_binding(code,revision,required)
    pinpath=code/f'configs/pose_peer_{index:06d}_pins.json';pin=inputs.identity(pinpath,True,inputs.MAX_PINS)
    pins=inputs.frozen_pins(inputs.strict_json(pinpath.read_bytes()),index)
    return binding,pin,pins,required


def copy_receiver(code,revision):
    target=code.parents[1]/'run_pose_peer_receive';inputs.canonical(target)
    inputs.require(not target.exists(),'Fresh real receiver snapshot required, no alias or overwrite')
    before=inputs.code_binding(code,revision,{});target.mkdir(mode=0o700);destination=target/'code';destination.mkdir(mode=0o700)
    for path in sorted(code.rglob('*')):
        relative=path.relative_to(code);output=destination/relative
        if path.is_dir():output.mkdir(mode=0o700)
        else:
            row=inputs.identity(path,True,2_000_000);inputs.write(output,path.read_bytes())
            inputs.require(inputs.identity(output,True)==row,'Receiver copied source identity differs')
    for name in('revision','source-sha256'):inputs.write(target/name,(code.parent/name).read_bytes())
    for path in sorted(destination.rglob('*'),reverse=True):
        if path.is_dir():path.chmod(0o555)
    destination.chmod(0o555)
    inputs.require(inputs.code_binding(destination,revision,{})==before and inputs.code_binding(code,revision,{})==before,'Copied original receiver closure/markers must be identical')
    return destination,before


def forced_command(code,revision,index):
    inputs.require(code==inputs.ROOT/'jobs'/revision/'run_pose_peer_receive/code'and re.fullmatch('[0-9a-f]{40}',revision),'Exact real receiver code namespace required')
    return(f'/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT={inputs.ROOT} WR_CODE={code} WR_CODE_REVISION={revision} '
        'SSH_CONNECTION="$SSH_CONNECTION" SSH_ORIGINAL_COMMAND="$SSH_ORIGINAL_COMMAND" '
        f'/bin/bash {code}/infra/run_pose_peer_receive.sh --episode {inputs.episode(index)}')


def configuration(control,forced):
    return '\n'.join(('ListenAddress 10.0.0.9','Port 2222','Protocol 2',f'HostKey {control}/host_ed25519',
        f'AuthorizedKeysFile {control}/authorized_keys',f'ForceCommand {forced}','PermitRootLogin forced-commands-only',
        'AllowUsers root','StrictModes yes','AuthenticationMethods publickey','PubkeyAuthentication yes','PasswordAuthentication no',
        'KbdInteractiveAuthentication no','UsePAM no','PermitUserEnvironment no','PermitUserRC no','PermitTTY no',
        'AllowTcpForwarding no','AllowAgentForwarding no','X11Forwarding no','PermitTunnel no','GatewayPorts no',
        'HostbasedAuthentication no','GSSAPIAuthentication no','AuthorizedKeysCommand none','MaxSessions 1','MaxAuthTries 2',
        'LogLevel ERROR',f'PidFile {control}/sshd.pid',''))


def owned_stop(unit,control):
    fields=dict(line.split('=',1)for line in command(['/usr/bin/systemctl','show',unit,'--property=ExecStart','--property=FragmentPath']).splitlines()if '='in line)
    inputs.require(fields.get('FragmentPath')=='/run/systemd/transient/'+unit and 'path=/usr/sbin/sshd'in fields.get('ExecStart','')and
        str(control/'sshd_config')in fields.get('ExecStart',''),'Only uniquely bound owned SSH unit may be stopped')
    command(['/usr/bin/systemctl','stop',unit])


def listener_pid(pid):
    # Type=exec confirms exec, not that sshd has bound its socket yet.
    for attempt in range(10):
        rows=command(['/usr/bin/ss','-H','-ltnp','sport = :2222']).splitlines()
        if rows:
            inputs.require(len(rows)==1 and len(rows[0].split())>=6 and rows[0].split()[3]=='10.0.0.9:2222'and
                re.findall(r'pid=(\d+)',rows[0])==[pid],'Exact private listener address and owned PID required')
            return
        if attempt<9:time.sleep(.2)
    raise ValueError('Owned private listener did not bind within fixed startup budget')


def start(root,code,revision,index,key):
    inputs.require(root==inputs.ROOT and code==root/'jobs'/revision/'run_pose_peer_server/code','Exact immutable server dispatch required')
    key=public_key(key);before,pin,pins,helpers=source(code,revision,index)
    control=inputs.canonical(RUNTIME/f'world-reward-pose-peer-{index:06d}-{revision}')
    unit=f'world-reward-pose-peer-{index:06d}-sshd-{revision[:12]}.service'
    incoming=inputs.canonical(inputs.DEST/f'pose-peer-{index:06d}-{revision}')
    directory(root,1000);directory(inputs.DEST,1000);directory(control.parent);directory(Path('/run/sshd'))
    inputs.require(not control.exists()and not incoming.exists()and not(code.parents[1]/'run_pose_peer_receive').exists(),'Fresh receiver/control/incoming namespaces required')
    inputs.require(command(['/usr/bin/systemctl','show',unit,'--property=LoadState','--value'],missing=True).strip()=='not-found','Absent server unit required')
    inputs.require(not command(['/usr/bin/ss','-H','-ltnp','sport = :2222']).strip(),'Private receiver port2222 already occupied')
    mount=inputs.strict_json(command(['/usr/bin/findmnt','--json','--mountpoint',str(inputs.DEST),'--output','TARGET,UUID,OPTIONS']))
    fs=mount.get('filesystems',[]);inputs.require(len(fs)==1 and fs[0].get('target')==str(inputs.DEST)and fs[0].get('uuid')==UUID and 'rw'in fs[0].get('options','').split(','),'Exact writable VM02 data disk required')
    space=os.statvfs(inputs.DEST);inputs.require(space.f_bavail*space.f_frsize>2*pins['archive']['bytes']+3*1024**3,'Room for archive plus full extracted copy required')
    control.mkdir(mode=0o700);owned=directory(control,private=True);attempted=False;started=time.monotonic()
    report=dict(schema='world_reward.pose_peer.server.v1',stage='pose_peer_server',status='fail',episode_index=index,producer_revision=revision,
        archive=pins['archive'],manifest=pins['manifest'],pins=pin,private_key_read=False,transport_verified=False,quality_verified=False,phase='receiver_copy')
    try:
        receiver,receiver_binding=copy_receiver(code,revision)
        report['phase']='host_key'
        command(['/usr/bin/ssh-keygen','-q','-t','ed25519','-N','','-f',str(control/'host_ed25519')],15)
        secret=(control/'host_ed25519').lstat();inputs.require(stat.S_ISREG(secret.st_mode)and secret.st_nlink==1 and secret.st_uid==0 and secret.st_mode&0o777==0o600,'New private hostkey metadata required')
        pub=control/'host_ed25519.pub';pub.chmod(0o400);hostkey=public_key(pub.read_text().rstrip('\n'),True)
        forced=forced_command(receiver,revision,index);escaped=forced.replace('\\','\\\\').replace('"','\\"')
        inputs.write(control/'authorized_keys',(f'from="10.0.0.4",restrict,command="{escaped}" {key}\n').encode())
        inputs.write(control/'sshd_config',configuration(control,forced).encode())
        public={name:inputs.identity(control/name,True,10000)for name in('authorized_keys','sshd_config','host_ed25519.pub')}
        report['phase']='sshd_config_validation'
        command(['/usr/sbin/sshd','-t','-f',str(control/'sshd_config')])
        inputs.require(source(code,revision,index)[:2]==(before,pin)and not incoming.exists(),'Source/incoming changed before listener start')
        inputs.require(command(['/usr/bin/systemctl','show',unit,'--property=LoadState','--value'],missing=True).strip()=='not-found'and
            not command(['/usr/bin/ss','-H','-ltnp','sport = :2222']).strip(),'Server unit/port appeared before listener start')
        report['phase']='server_start';attempted=True;command(['/usr/bin/systemd-run','--quiet','--unit='+unit,'--property=Type=exec','--property=Restart=no','--property=RuntimeMaxSec=2400','/usr/sbin/sshd','-D','-e','-f',str(control/'sshd_config')])
        values=dict(line.split('=',1)for line in command(['/usr/bin/systemctl','show',unit,'--property=ActiveState','--property=MainPID']).splitlines()if '='in line)
        inputs.require(values.get('ActiveState')=='active'and re.fullmatch('[1-9][0-9]*',values.get('MainPID','')),'Owned SSH listener must be active')
        listener_pid(values['MainPID'])
        inputs.require(source(code,revision,index)[:2]==(before,pin)and inputs.code_binding(receiver,revision,{})==receiver_binding and
            directory(control,private=True)==owned and not incoming.exists()and
            {name:inputs.identity(control/name,True,10000)for name in public}==public,'Post-start source/control/receiver changed')
        report.update(status='pass',phase='complete',host_public_key=hostkey,host_public_file_identity=public['host_ed25519.pub'],public_control_identities=public,
            unit=unit,main_pid=int(values['MainPID']),listen_address='10.0.0.9',port=2222,runtime_max_seconds=2400,
            incoming_path=str(incoming),receiver_source_binding=receiver_binding,source_helpers=helpers,source_rehashed_after=True,data_disk_uuid=UUID)
    except Exception:
        report['error']='Owned private SSH server gate failed; partial evidence retained'
        if attempted:
            try:owned_stop(unit,control);report['owned_stop_succeeded']=True
            except Exception:report['owned_stop_succeeded']=False
    report['elapsed_seconds']=time.monotonic()-started
    try:inputs.write(control/'server-receipt.json',inputs.digest_json(report))
    except Exception:
        if attempted:owned_stop(unit,control)
        raise
    return report,control


def main(argv=None):
    parser=inputs.parser();parser.add_argument('--client-public-key',required=True,action=inputs.Once);a=parser.parse_args(argv)
    code=Path(os.environ['WR_CODE']);inputs.require(platform.system()=='Linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'and
        os.environ['WR_ROOT']==str(inputs.ROOT)and Path(__file__)==code/HELPERS[0],'Actual immutable VM02 root server entry required')
    def stopped(*_):raise TimeoutError('Owned server setup interrupted')
    previous={s:signal.signal(s,stopped)for s in(signal.SIGTERM,signal.SIGINT)}
    try:report,control=start(inputs.ROOT,code,os.environ['WR_CODE_REVISION'],a.episode,a.client_public_key)
    finally:
        for sig,handler in previous.items():signal.signal(sig,handler)
    summary={k:report.get(k)for k in('stage','status','episode_index','host_public_key','transport_verified','source_rehashed_after')}
    summary['receipt']=inputs.identity(control/'server-receipt.json',True);print(json.dumps(summary));return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Private pose peer server preflight failed',file=sys.stderr);sys.exit(1)
