"""VM01 pulls exact Azure-private bytes then promotes a fresh native pose folder."""
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time
import frontend_peer_receive as receiver
import pose_peer_server as control
import pose_return_inputs as payload

HELPERS=('infra/pose_return_pull.py','infra/run_pose_return_pull.sh','infra/pose_return_inputs.py','infra/pose_peer_inputs.py',
    'infra/frontend_peer_receive.py','infra/pose_peer_server.py')


def key_state(path):
    payload.canonical(path);payload.canonical(path.parent);s=path.lstat();d=path.parent.lstat()
    import stat
    payload.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==0 and s.st_mode&0o777==0o600 and
        1<=s.st_size<=10000 and stat.S_ISDIR(d.st_mode)and d.st_uid==0 and d.st_mode&0o777==0o700,'Existing root600 private client key in root700 directory required')
    # Metadata only: never read/hash/copy private key bytes.
    return (tuple(getattr(s,k)for k in('st_dev','st_ino','st_mode','st_uid','st_size','st_mtime_ns','st_ctime_ns')),
        tuple(getattr(d,k)for k in('st_dev','st_ino','st_mode','st_uid','st_mtime_ns','st_ctime_ns')))


def ssh_arguments(key,known,index):
    return ['/usr/bin/ssh','-F','/dev/null','-T','-p','2222','-i',str(key),'-o','IdentitiesOnly=yes','-o','IdentityAgent=none','-o','BatchMode=yes',
        '-o','StrictHostKeyChecking=yes','-o','UserKnownHostsFile='+str(known),'-o','GlobalKnownHostsFile=/dev/null',
        '-o','ConnectTimeout=15','-o','ServerAliveInterval=30','-o','ServerAliveCountMax=3','root@10.0.0.9',f'world-reward-pose-results-{index:06d}']


def accept(root,code,revision,index,out,pins,source,before,pinid):
    manifest=payload.inspect_archive(out/'stream/archive.tar',pins,source);extracted=out/'extracted'
    payload.extract(out/'stream/archive.tar',manifest,extracted)
    parent=payload.checked(extracted/payload.PARENT,manifest['files'][payload.PARENT])
    native=payload.checked(extracted/'report.json',source['outputs']['report.json'],payload.MAX_ARCHIVE)
    payload.records(source,parent,native)
    pending=out/'native';pending.mkdir(mode=0o700)
    for name,wanted in source['outputs'].items():
        os.rename(extracted/name,pending/name);payload.require(payload.identity(pending/name,True,payload.MAX_ARCHIVE)==wanted,'Original returned bytes changed before promotion')
        os.chown(pending/name,1000,1000);(pending/name).chmod(0o444)
    os.chown(pending,1000,1000);pending.chmod(0o755)
    payload.require(payload.binding(code,revision,HELPERS)[0]==before and payload.load_transport(code,index)==(pins,source,pinid)and
        payload.identity(out/'stream/archive.tar',True,payload.MAX_ARCHIVE)==pins['archive'],'Return source/pins/archive changed before native publication')
    target=payload.canonical(root/f'outputs/episode_{index:06d}/object_pose_full')
    payload.require(target.parent.is_dir()and not target.exists(),'Canonical original target remains absent; no merging/reuse')
    payload.promote(pending,target)
    payload.require({p.name for p in target.iterdir()}==set(payload.FILES)and target.stat().st_uid==1000 and target.stat().st_gid==1000,'Exact returned native folder required')
    for name,wanted in source['outputs'].items():
        path=target/name;s=path.stat();payload.require(s.st_uid==1000 and s.st_gid==1000 and s.st_mode&0o777==0o444 and
            payload.identity(path,True,payload.MAX_ARCHIVE)==wanted,'Published original byte/mode/ownership differs')
    return target


def pull(root,code,revision,index,serverkey):
    serverkey=control.public_key(serverkey,True);before,helpers=payload.binding(code,revision,HELPERS)
    pins,source,pinid=payload.load_transport(code,index)
    target=payload.canonical(root/f'outputs/episode_{index:06d}/object_pose_full');payload.require(target.parent.is_dir()and not target.exists(),'Fresh absent native target required before network')
    key=payload.canonical(root/'transfer'/f'pose-return-client-{index:06d}-v1/client_ed25519');key_before=key_state(key)
    pub=key.with_suffix('.pub');publicpin=payload.identity(pub,True,1000);control.public_key(pub.read_text().strip(),True)
    out=payload.canonical(root/'results'/f'pose-return-pull-{index:06d}-{revision}');payload.require(out.parent.is_dir()and not out.exists(),'Fresh private pull result namespace required')
    out.mkdir(mode=0o700);streamdir=out/'stream';streamdir.mkdir(mode=0o700)
    known=out/'known_hosts';payload.original.write(known,('[10.0.0.9]:2222 '+serverkey+'\n').encode());knownpin=payload.identity(known,True,1000)
    report=dict(stage='pose_return_pull',status='fail',episode_index=index,producer_revision=revision,archive=pins['archive'],
        manifest=pins['manifest'],private_key_read=False,model_or_GPU_used=False,prediction_algorithm_changed=False,quality_verified=False,
        output_promoted=False,original_frame_coverage_verified=False)
    started=time.monotonic();process=None
    def stopped(*unused):raise TimeoutError('Fixed600s private result return deadline exceeded')
    old={s:signal.signal(s,stopped)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(600)
    try:
        env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent'}
        with (out/'ssh.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400)
            process=subprocess.Popen(ssh_arguments(key,known,index),env=env,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=log)
            received=receiver.receive(process.stdout,streamdir,pins['archive']['bytes'],pins['archive']['sha256'])
            status=process.wait(timeout=max(.01,600-(time.monotonic()-started)))
        payload.require(status==0 and received['status']=='pass'and received['receipt_written']is True,'Exact source export exit0 and frozen incoming archive required')
        payload.require(key_state(key)==key_before and payload.identity(pub,True,1000)==publicpin and payload.identity(known,True,1000)==knownpin,'Client credential metadata/public host pin changed')
        returned=accept(root,code,revision,index,out,pins,source,before,pinid)
        payload.require(payload.binding(code,revision,HELPERS)[0]==before and payload.load_transport(code,index)==(pins,source,pinid)and
            key_state(key)==key_before,'Immutable pull source/pins/key changed after publication')
        report.update(status='pass',phase='complete',frames=source['frames'],output_files=source['outputs'],output_promoted=True,
            original_frame_coverage_verified=True,source_helpers=helpers,source_rehashed_after=True,source_pins=pins['source_pins'],ssh_exit_status=status)
    except Exception:report['error']='Exact pose result return failed closed; owned partial evidence retained'
    finally:
        signal.alarm(0)
        if process is not None and process.poll()is None:
            process.terminate()
            try:process.wait(timeout=5)
            except subprocess.TimeoutExpired:process.kill();process.wait(timeout=5)
        for s,handler in old.items():signal.signal(s,handler)
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>600:report.update(status='fail',error='Frozen600s private return budget exceeded')
        payload.original.write(out/'report.json',payload.raw_json(report))
    return report


def main(argv=None):
    parser=payload.original.parser();parser.add_argument('--server-public-key',required=True,action=payload.original.Once);a=parser.parse_args(argv)
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    payload.require(sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01'and os.environ['WR_ROOT']==str(payload.ROOT)and
        code==payload.ROOT/'jobs'/revision/'run_pose_return_pull/code'and Path(__file__)==code/HELPERS[0],'Actual immutable root VM01 result pull required')
    report=pull(payload.ROOT,code,revision,a.episode,a.server_public_key)
    print(payload.raw_json({k:report.get(k)for k in('stage','status','episode_index','frames','output_promoted','elapsed_seconds')}).decode(),end='')
    return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose result pull preflight failed closed',file=sys.stderr);sys.exit(1)
