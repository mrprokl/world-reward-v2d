"""Unmodified full-video object tracker on VM02 with only pinned public inputs."""
from __future__ import annotations
import ctypes
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import time

sys.path.insert(0,str(Path(__file__).resolve().parent))
import pose_peer_inputs as inputs

HELPERS=('infra/pose_peer_run.py','infra/run_pose_peer_run.sh','infra/pose_peer_inputs.py')


def lock_state(root,lock_fd):
    # Require the inherited descriptor really owns the original exclusive lock.
    lock=inputs.canonical(root/'jobs/.world-reward-h100.lock');state=lock.lstat();actual=os.fstat(lock_fd)
    inputs.require((state.st_dev,state.st_ino)==(actual.st_dev,actual.st_ino)and state.st_nlink==1 and stat.S_ISREG(state.st_mode),'Original cooperative lock descriptor required')
    with lock.open('rb')as probe:
        try:fcntl.flock(probe.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:pass
        else:fcntl.flock(probe.fileno(),fcntl.LOCK_UN);raise ValueError('Inherited exclusive cooperative lock must already be held')
    try:fcntl.flock(lock_fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise ValueError('Inherited descriptor, not another process, must own original exclusive lock')
    return state.st_dev,state.st_ino


def promote(source,target):
    """Linux atomic RENAME_NOREPLACE: never merge or overwrite another output."""
    inputs.canonical(source);inputs.canonical(target)
    libc=ctypes.CDLL(None,use_errno=True);rename=libc.renameat2
    rename.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];rename.restype=ctypes.c_int
    if rename(-100,os.fsencode(source),-100,os.fsencode(target),1)!=0:
        error=ctypes.get_errno();raise OSError(error,'Owned native pose directory atomic promotion failed')


def prepare(root,code,revision,index,lock_fd=9):
    lock=lock_state(root,lock_fd)
    pinpath=code/f'configs/pose_peer_{index:06d}_pins.json';pin_identity=inputs.identity(pinpath,True,inputs.MAX_PINS)
    pins=inputs.frozen_pins(inputs.strict_json(pinpath.read_bytes()),index)
    helpers={**pins['tracker_helpers'],**{name:inputs.identity(code/name,True)for name in HELPERS}}
    bind=inputs.code_binding(code,revision,helpers)
    receivepin=code/f'configs/pose_peer_{index:06d}_received_pins.json';receive_identity=inputs.identity(receivepin,True,inputs.MAX_PINS)
    received=inputs.strict_json(receivepin.read_bytes())
    inputs.require(set(received)=={'schema','episode_index','producer_revision','receipt'}and received['schema']=='world_reward.pose_peer.received_pins.v1'and
        received['episode_index']==index and re.fullmatch('[0-9a-f]{40}',received['producer_revision']),'Independent actual receiver receipt pins required')
    inputs.pin(received['receipt']);destination=inputs.DEST/f'pose-peer-{index:06d}-{received["producer_revision"]}'
    receiptpath=destination/'pose-receipt.json';inputs.require(inputs.identity(receiptpath,True)==received['receipt'],'Actual receive receipt SHA required')
    receipt=inputs.strict_json(receiptpath.read_bytes());inputs.require(receipt.get('stage')=='pose_peer_receive'and receipt.get('status')=='pass'and
        receipt.get('episode_index')==index and receipt.get('producer_revision')==received['producer_revision']and receipt.get('archive')==pins['archive']and receipt.get('manifest')==pins['manifest']and
        receipt.get('original_frame_coverage_verified')is True and receipt.get('extraction_performed')is True,'Actual complete full input receiver required')
    payload=inputs.canonical(destination/'inputs')
    manifestpath=payload/inputs.MANIFEST_NAME
    inputs.require(inputs.identity(manifestpath,True,inputs.MAX_MANIFEST)==pins['manifest'],'Sealed received manifest SHA required before parsing')
    manifest=inputs.load_manifest(manifestpath.read_bytes(),pins,index)
    for name,row in manifest['files'].items():inputs.require(inputs.identity(payload/name,True)==row,'All received full pose input SHA required before GPU')
    if manifest['mesh_source']=='volume':
        volume=code/f'configs/volume_mesh_{index:06d}_pins.json';inputs.require(inputs.strict_json(volume.read_bytes())==manifest['volume_pins'],'Exact unchanged committed volume pins required')
    out=inputs.canonical(root/f'outputs/episode_{index:06d}/object_pose_full')
    inputs.require(out.parent.is_dir()and not out.exists(),'Fresh original pose output required, no resume/merge')
    return dict(pins=pins,manifest=manifest,manifestpath=manifestpath,payload=payload,output=out,binding=bind,helpers=helpers,
        pinpath=pinpath,pin_identity=pin_identity,receivepin=receivepin,receive_identity=receive_identity,
        receiptpath=receiptpath,receipt_identity=received['receipt'],lock=lock)


def image(root):
    result=subprocess.run(['docker','image','inspect',inputs.IMAGE,'--format','{{json .Id}} {{json .Architecture}} {{json .Os}} {{json .RootFS}}'],
        env={'PATH':'/usr/bin:/bin','DOCKER_HOST':'unix://'+str(root/'docker.sock')},capture_output=True,timeout=15)
    inputs.require(result.returncode==0 and len(result.stdout)<16384,'Bounded existing image identity required');raw=result.stdout.decode().strip();decoder=json.JSONDecoder();values=[]
    while raw:value,end=decoder.raw_decode(raw);values.append(value);raw=raw[end:].lstrip()
    inputs.require(len(values)==4 and values[:3]==[inputs.IMAGE,'amd64','linux']and values[3]['Type']=='layers'and len(values[3]['Layers'])==44,'Exact selected original VM02 image/44layers required')
    return {'image_id':values[0],'ordered_rootfs_sha256':hashlib.sha256(inputs.digest_json(values[3]['Layers'])).hexdigest()}


def run(root,code,revision,index,lock_fd=9):
    inputs.require(code==root/'jobs'/revision/'run_pose_peer_run/code','Actual pose-only source snapshot required')
    current=prepare(root,code,revision,index,lock_fd);image_before=image(root)
    resultdir=inputs.canonical(root/'results'/f'pose-peer-run-{index:06d}-{revision}');inputs.require(not resultdir.exists(),'Fresh pose-only result namespace required')
    name=f'world-reward-pose-peer-{index:06d}-{revision[:12]}'
    safe_env={'PATH':'/usr/bin:/bin','DOCKER_HOST':'unix://'+str(root/'docker.sock'),'HOME':'/nonexistent'}
    existing=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],env=safe_env,capture_output=True,timeout=15)
    inputs.require(existing.returncode==0 and not existing.stdout.strip(),'Pose-only container namespace occupied')
    resultdir.mkdir(mode=0o700)
    started=time.monotonic();owned=False
    report=dict(stage='pose_peer_run',status='fail',episode_index=index,producer_revision=revision,image=image_before,
        tracker_helpers=current['manifest']['tracker_helpers'],source_helpers={name:inputs.identity(code/name,True)for name in HELPERS},
        pins=current['pin_identity'],received_receipt=current['receipt_identity'],script_sha256=current['helpers'][HELPERS[0]]['sha256'],
        full_original_indices=True,prediction_algorithm_changed=False,
        model_weights_read=False,ground_truth_used=False,quality_verified=False)
    try:
        apps=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],env=safe_env,capture_output=True,timeout=5)
        inputs.require(apps.returncode==0 and not apps.stdout.strip(),'GPU must remain idle under original lock')
        mounts=['--mount',f'type=bind,src={code},dst={code},readonly']
        # Unchanged tracker creates its exclusive object_pose_full itself.
        work=resultdir/'prediction-parent';work.mkdir(mode=0o755);os.chown(work,1000,1000);work_inode=work.stat()
        # Output parent overlays only this episode, not any data/model/other clip.
        mounts+=['--mount',f'type=bind,src={work},dst={root}/outputs/episode_{index:06d}']
        for namepath in sorted(current['manifest']['files']):mounts+=['--mount',f'type=bind,src={current["payload"]/namepath},dst={root/namepath},readonly']
        args=['docker','run','--rm','--name',name,'--cidfile',str(resultdir/'container.id'),'--label','world_reward_pose_peer='+revision,
            '--gpus','all','--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','16g','--cpus','4',
            '--tmpfs','/tmp:rw,noexec,nosuid,size=128m','--env','WR_ROOT='+str(root),'--env','PYTHONPATH='+str(code/'src'),
            '--env','PYTHONDONTWRITEBYTECODE=1','--env','HOME=/tmp',*mounts,'--entrypoint','python',inputs.IMAGE,'-B',str(code/'infra/object_pose_smoke.py'),
            '--episode',str(index),'--full-video','--mesh-source',current['manifest']['mesh_source']]
        owned=True;log=resultdir/'native.log'
        with log.open('xb')as stream:os.fchmod(stream.fileno(),0o400);result=subprocess.run(['/usr/bin/timeout','--signal=TERM','--kill-after=10s','10800s',*args],env=safe_env,stdout=stream,stderr=subprocess.STDOUT,timeout=10812)
        inputs.require(result.returncode==0,'Unchanged full pose child failed; inspect owned Azure log');owned=False
        output=work/'object_pose_full';native=inputs.strict_json((output/'report.json').read_bytes())
        inputs.require(native.get('stage')=='fixed_scale_full_object_pose_initializer'and native.get('status')=='pass'and native.get('episode_index')==index and
            native.get('original_frame_coverage_verified')is True and native.get('input_sha256')==current['manifest']['clip_spec']['video_sha256']and
            [row['frame_index']for row in native['frames']]==list(range(current['manifest']['clip_spec']['total_frames'])),'Every unchanged original pose frame required')
        expected=dict(input_track='track_1',ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],fixed_shape=True,
            mesh_source=current['manifest']['mesh_source'],script_sha256=current['manifest']['tracker_helpers']['infra/object_pose_smoke.py']['sha256'],
            budget_source_sha256=current['manifest']['files'][inputs.KIT]['sha256'])
        for field,relative in(('object_report_sha256','object_grounded'),('alignment_report_sha256','scale_smoke'),('full_depth_report_sha256','depth_full')):
            expected[field]=current['manifest']['files'][f'outputs/episode_{index:06d}/{relative}/report.json']['sha256']
        inputs.require(all(type(native.get(k))is type(v)and native[k]==v for k,v in expected.items())and
            inputs.identity(output/'geometry_and_poses.npz')['sha256']==native['geometry_and_poses_sha256']and
            inputs.identity(output/'object_fixed_canonical.glb')['sha256']==native['fixed_canonical_mesh_sha256'],'Actual unchanged native source/input/output proof required')
        inputs.require(sorted(p.name for p in output.iterdir())==['geometry_and_poses.npz','object_fixed_canonical.glb','report.json'],'Exact full pose payload required')
        inputs.require((work.stat().st_dev,work.stat().st_ino)==(work_inode.st_dev,work_inode.st_ino)and
            output.stat().st_uid==1000 and output.stat().st_gid==1000,'Original newly owned native output identity required')
        for file in output.iterdir():
            inputs.require(file.lstat().st_uid==1000 and file.lstat().st_gid==1000,'New native output files must retain original UID1000 ownership')
            inputs.identity(file);file.chmod(0o444)
        inputs.require(lock_state(root,lock_fd)==current['lock'],'Original exclusive lock changed before output promotion')
        promote(output,current['output'])
        # Docker creates RO-bind placeholders in owned work; never recursive-rm.
        report.update(status='pass',frames=len(native['frames']),output_files={p.name:inputs.identity(p,True)for p in current['output'].iterdir()})
    except Exception:report['error']='Pose-only native execution failed; owned evidence retained'
    finally:
        try:
            cid=resultdir/'container.id'
            if cid.exists():
                cid.chmod(0o400);value=cid.read_text().strip();inputs.require(re.fullmatch('[0-9a-f]{64}',value),'Exact owned container ID required')
                query=['docker','ps','-aq','--no-trunc','--filter','id='+value,'--filter','label=world_reward_pose_peer='+revision]
                found=subprocess.run(query,env=safe_env,capture_output=True,timeout=10);ids=found.stdout.decode().split();inputs.require(found.returncode==0 and ids in([],[value]),'Exact owned pose-container query required')
                if owned and ids:inputs.require(subprocess.run(['docker','rm','--force',value],env=safe_env,capture_output=True,timeout=15).returncode==0,'Owned pose cleanup failed')
                survivors=subprocess.run(query,env=safe_env,capture_output=True,timeout=10);inputs.require(survivors.returncode==0 and not survivors.stdout.strip(),'Owned pose-container survives')
            else:
                remaining=subprocess.run(['docker','ps','-aq','--filter','name=^/'+name+'$'],env=safe_env,capture_output=True,timeout=10)
                inputs.require(remaining.returncode==0 and not remaining.stdout.strip(),'Unidentified pose-container survives; never remove unknown ownership')
            inputs.require(lock_state(root,lock_fd)==current['lock']and image(root)==image_before and inputs.code_binding(code,revision,current['helpers'])==current['binding']and
                all(inputs.identity(current['payload']/name,True)==row for name,row in current['manifest']['files'].items())and
                inputs.identity(current['manifestpath'],True)==current['pins']['manifest']and
                inputs.identity(current['pinpath'],True)==current['pin_identity']and inputs.identity(current['receivepin'],True)==current['receive_identity']and
                inputs.identity(current['receiptpath'],True)==current['receipt_identity'],'Pose source/original received bytes changed')
            report['source_rehashed_after']=True
        except Exception:report.update(status='fail',source_rehashed_after=False,error='Pose post-run immutable/runtime gate failed')
        report['elapsed_seconds']=time.monotonic()-started
        if(resultdir/'native.log').exists():report['native_log']=inputs.identity(resultdir/'native.log',True)
        inputs.write(resultdir/'report.json',inputs.digest_json(report))
    return report


def main(argv=None):
    a=inputs.parser().parse_args(argv)
    code=Path(os.environ['WR_CODE']);inputs.require(Path(__file__)==code/HELPERS[0],'Actual immutable pose entrypoint required')
    report=run(inputs.ROOT,code,os.environ['WR_CODE_REVISION'],a.episode)
    print(json.dumps({k:report.get(k)for k in('stage','status','episode_index','frames','elapsed_seconds')}));return 0 if report['status']=='pass'else 1


if __name__=='__main__':
    try:sys.exit(main())
    except Exception:print('Pose-only peer preflight failed',file=sys.stderr);sys.exit(1)
