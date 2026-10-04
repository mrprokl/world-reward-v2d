"""Owned expiring private result exporter, never an inference/model service."""
import os
from pathlib import Path
import re
import signal
import sys
import time
import pose_peer_server as control
import pose_return_inputs as payload
import pose_return_export as exporter

HELPERS=('infra/pose_return_server.py','infra/run_pose_return_server.sh',*exporter.HELPERS,'infra/pose_peer_server.py')


def forced(code,revision,index):
    payload.require(code==payload.ROOT/'jobs'/revision/'run_pose_return_server/code'and re.fullmatch('[0-9a-f]{40}',revision),'Actual server/export immutable shared namespace required')
    return(f'/usr/bin/env -i PATH=/usr/bin:/bin HOME=/nonexistent WR_ROOT={payload.ROOT} WR_CODE={code} WR_CODE_REVISION={revision} '
        'SSH_CONNECTION="$SSH_CONNECTION" SSH_ORIGINAL_COMMAND="$SSH_ORIGINAL_COMMAND" '
        f'/bin/bash {code}/infra/run_pose_return_export.sh --episode {payload.original.episode(index)}')


def start(root,code,revision,index,key):
    key=control.public_key(key);before,helpers=payload.binding(code,revision,HELPERS)
    archive,pins,_,pinid=exporter.archive_proof(root,code,index)
    runtime=payload.canonical(control.RUNTIME/f'world-reward-pose-return-{index:06d}-{revision}')
    unit=f'world-reward-pose-return-{index:06d}-sshd-{revision[:12]}.service'
    control.directory(root,1000);control.directory(runtime.parent);control.directory(Path('/run/sshd'))
    payload.require(not runtime.exists(),'Fresh owned private SSH control namespace required')
    payload.require(control.command(['/usr/bin/systemctl','show',unit,'--property=LoadState','--value'],missing=True).strip()=='not-found'and
        not control.command(['/usr/bin/ss','-H','-ltnp','sport = :2222']).strip(),'Absent owned unit and unoccupied private port required')
    runtime.mkdir(mode=0o700);owned=control.directory(runtime,private=True);attempted=False;started=time.monotonic()
    report=dict(stage='pose_return_server',status='fail',episode_index=index,producer_revision=revision,archive=pins['archive'],
        private_key_read=False,transport_verified=False,quality_verified=False,model_or_GPU_used=False)
    try:
        control.command(['/usr/bin/ssh-keygen','-q','-t','ed25519','-N','','-f',str(runtime/'host_ed25519')],15)
        private=(runtime/'host_ed25519').lstat();payload.require(stat_key(private),'New private hostkey metadata required; bytes never read')
        pub=runtime/'host_ed25519.pub';pub.chmod(0o400);hostkey=control.public_key(pub.read_text().rstrip('\n'),True)
        command=forced(code,revision,index);escaped=command.replace('\\','\\\\').replace('"','\\"')
        payload.original.write(runtime/'authorized_keys',(f'from="10.0.0.4",restrict,command="{escaped}" {key}\n').encode())
        payload.original.write(runtime/'sshd_config',control.configuration(runtime,command).encode())
        public={name:payload.identity(runtime/name,True,10000)for name in('host_ed25519.pub','authorized_keys','sshd_config')}
        control.command(['/usr/sbin/sshd','-t','-f',str(runtime/'sshd_config')])
        payload.require(payload.binding(code,revision,HELPERS)[0]==before and payload.identity(archive,True,payload.MAX_ARCHIVE)==pins['archive']and
            control.command(['/usr/bin/systemctl','show',unit,'--property=LoadState','--value'],missing=True).strip()=='not-found'and
            not control.command(['/usr/bin/ss','-H','-ltnp','sport = :2222']).strip(),'Immutable source/archive/unit/port changed before start')
        attempted=True;control.command(['/usr/bin/systemd-run','--quiet','--unit='+unit,'--property=Type=exec','--property=Restart=no','--property=RuntimeMaxSec=1200',
            '/usr/sbin/sshd','-D','-e','-f',str(runtime/'sshd_config')])
        values=dict(line.split('=',1)for line in control.command(['/usr/bin/systemctl','show',unit,'--property=ActiveState','--property=MainPID']).splitlines()if '='in line)
        payload.require(values.get('ActiveState')=='active'and re.fullmatch('[1-9][0-9]*',values.get('MainPID','')),'Exact active owned listener required')
        control.listener_pid(values['MainPID'])
        payload.require(payload.binding(code,revision,HELPERS)[0]==before and payload.load_transport(code,index)[2]==pinid and
            control.directory(runtime,private=True)==owned and {n:payload.identity(runtime/n,True,10000)for n in public}==public,'Owned source/public controls changed after listener start')
        report.update(status='pass',phase='complete',host_public_key=hostkey,unit=unit,main_pid=int(values['MainPID']),listen_address='10.0.0.9',port=2222,
            runtime_max_seconds=1200,source_helpers=helpers,source_rehashed_after=True,public_control_identities=public)
    except Exception:
        report['error']='Owned private pose export server failed closed'
        if attempted:
            try:control.owned_stop(unit,runtime);report['owned_stop_succeeded']=True
            except Exception:report['owned_stop_succeeded']=False
    report['elapsed_seconds']=time.monotonic()-started
    try:payload.original.write(runtime/'server-receipt.json',payload.raw_json(report))
    except Exception:
        if attempted:control.owned_stop(unit,runtime)
        raise
    return report,runtime


def stat_key(s):
    import stat
    return stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==0 and s.st_mode&0o777==0o600


def main(argv=None):
    parser=payload.original.parser();parser.add_argument('--client-public-key',required=True,action=payload.original.Once);a=parser.parse_args(argv)
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    payload.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02'and os.environ['WR_ROOT']==str(payload.ROOT)and
        code==payload.ROOT/'jobs'/revision/'run_pose_return_server/code'and Path(__file__)==code/HELPERS[0],'Actual immutable root VM02 server required')
    def stopped(*unused):raise TimeoutError('Owned result server setup interrupted')
    old={s:signal.signal(s,stopped)for s in(signal.SIGTERM,signal.SIGINT)}
    try:report,runtime=start(payload.ROOT,code,revision,a.episode,a.client_public_key)
    finally:
        for s,handler in old.items():signal.signal(s,handler)
    print(payload.raw_json(dict(stage=report['stage'],status=report['status'],host_public_key=report.get('host_public_key'),
        receipt=payload.identity(runtime/'server-receipt.json',True),transport_verified=False)).decode(),end='')
    return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose return server preflight failed closed',file=sys.stderr);sys.exit(1)
