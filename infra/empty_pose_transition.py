"""Archive only an authenticated empty pre-pose topology failure, never resume it."""
import argparse
import json
import fcntl
import os
from pathlib import Path
import re
import stat
import subprocess
import time

import failed_masks_transition as archive
import volume_mesh_pin_inventory as volume

ROOT=Path('/srv/scenesmith/world-reward')
JOB='run_empty_pose_transition'
SOURCES=('infra/empty_pose_transition.py','infra/run_empty_pose_transition.sh','infra/failed_masks_transition.py','infra/volume_mesh_pin_inventory.py')
UNIT_FIELDS=('LoadState','ActiveState','SubState','Result','ExecMainCode','ExecMainStatus','MainPID','ExecMainPID','ControlGroup')


def require(value,message):
    if not value:raise ValueError(message)


def state(path):
    s=path.lstat();return (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)


def binding(code,revision):
    archive.canonical(code);require(code.is_dir(),'Existing immutable source required')
    require((code.parent/'revision').read_bytes()==(revision+'\n').encode() and re.fullmatch(b'[0-9a-f]{64}\n',(code.parent/'source-sha256').read_bytes()),'Original dispatch markers required')
    files={}
    for path in (code,*sorted(code.rglob('*'))):
        archive.canonical(path);mode=path.lstat().st_mode
        require(not mode&0o222 and (stat.S_ISREG(mode)or stat.S_ISDIR(mode)),'Entire original source snapshot must be readonly and regular')
        if path.is_file():files[str(path.relative_to(code))]=archive.identity(path,2_000_000)
    return dict(files=files,markers={name:archive.identity(code.parent/name,128)for name in ('revision','source-sha256')})


def command(args):
    result=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,timeout=5,check=False,text=True)
    require(result.returncode==0 and len(result.stdout)<=4096,'Bounded local runtime-state query failed');return result.stdout


def unit_state(unit):
    result={}
    for line in command(['systemctl','show','--no-pager',*('--property='+key for key in UNIT_FIELDS),unit]).splitlines():
        key,sep,value=line.partition('=');require(sep and key in UNIT_FIELDS and key not in result,'Exact original unit projection required');result[key]=value
    require(set(result)==set(UNIT_FIELDS),'Complete original unit projection required');return result


def inactivity(unit,pid,probe=unit_state,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup')):
    record=probe(unit)
    require(type(record)is dict and set(record)==set(UNIT_FIELDS) and all(type(v)is str for v in record.values()),'Exact systemd strings required')
    failed=(record['LoadState']=='loaded' and record['ActiveState']==record['SubState']=='failed' and record['Result']=='exit-code' and record['ExecMainCode']==record['ExecMainStatus']=='1')
    collected=(record['LoadState']=='not-found' and record['ActiveState']=='inactive' and record['SubState']=='dead')
    require((failed or collected)and record['MainPID']=='0' and record['ExecMainPID']in ('0',str(pid)),'Only inactive original failed or garbage-collected unit allowed')
    require(not (proc/str(pid)).exists() and not (proc/str(pid)).is_symlink(),'Historical PID alive/reused; never kill it')
    group=record['ControlGroup'];expected='/system.slice/'+unit
    require(not group or group==expected,'Exact owned original cgroup required')
    path=archive.canonical(cgroups/expected.lstrip('/'))
    if path.exists():
        require(path.is_dir(),'Original cgroup must be directory')
        files=list(path.rglob('cgroup.procs'));require(files,'Cannot prove original cgroup empty')
        for file in files:
            archive.canonical(file);require(file.is_file()and not file.read_text().strip(),'Original cgroup still contains processes')
    return record


def gpu_idle():
    require(not command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits']).strip(),'GPU compute process present; no archive')


def empty_state(path):
    archive.canonical(path);s=path.lstat()
    require(stat.S_ISDIR(s.st_mode)and s.st_uid==1000 and s.st_nlink==2 and not list(path.iterdir()),'Only original empty UID1000 pose directory allowed')
    return state(path)


def lock_state(root):
    path=archive.canonical(root/'jobs/.world-reward-h100.lock');s=path.lstat();fd=os.fstat(9)
    require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and (s.st_dev,s.st_ino)==(fd.st_dev,fd.st_ino),'Held existing FD9 lock inode differs')
    fcntl.flock(9,fcntl.LOCK_EX|fcntl.LOCK_NB)
    return (s.st_dev,s.st_ino)


def volume_binding(root,code,episode):
    pinpath=code/f'configs/volume_mesh_{episode:06d}_pins.json';pin=archive.identity(pinpath,32_768)
    require(not pinpath.stat().st_mode&0o222,'Committed volume pins must be readonly')
    pins=volume.strict_json(pinpath.read_text());paths=volume.paths(episode)
    report,observed=volume.verify_pinned_artifacts(root,pins,episode,pins.get('input_sha256'),pins.get('files',{}).get(paths['object'],{}).get('sha256'),pins.get('files',{}).get(paths['alignment'],{}).get('sha256'),pins.get('metric_scale_baked_once'))
    producer=root/'jobs'/pins['report']['producer_revision']/'run_object_budget_volume/code'
    old=binding(producer,pins['report']['producer_revision']);require(old['files'].get(volume.PRODUCER_SCRIPT,{}).get('sha256')==pins['report']['script_sha256'],'Actual volume producer source differs')
    require(archive.identity(pinpath,32_768)==pin,'Volume pin changed during validation')
    return dict(pin=pin,files=observed,producer_source=old,report_status=report['status'])


def validate(root,code,revision,args,probe=unit_state,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup')):
    require(code==root/'jobs'/revision/JOB/'code','Actual current transition source required')
    current=binding(code,revision);require(all(name in current['files']for name in SOURCES),'Complete transition source closure required')
    oldcode=root/'jobs'/args.failed_revision/args.failed_entrypoint/'code';old=binding(oldcode,args.failed_revision)
    require(old['files'].get('infra/object_pose_smoke.py',{}).get('sha256')==args.failed_script_sha256
            and all(name in old['files']for name in ('infra/run_object_pose_smoke.sh','infra/'+args.failed_entrypoint+'.sh')),'Pinned actual historical pose producer required')
    unit=args.failed_unit;require(re.fullmatch(rf'world-reward-track1-episode{args.episode}-[a-z][a-z0-9-]{{0,60}}\.service',unit),'Episode-bound owned failed unit required')
    log=root/'results'/(unit.removeprefix('world-reward-').removesuffix('.service')+'.log')
    logpin=dict(bytes=args.failed_log_bytes,sha256=args.failed_log_sha256)
    require(archive.identity(log,2_000_000)==logpin,'Original failure log differs')
    raw=log.read_bytes();require(b'Traceback (most recent call last):'in raw and b'No topology-preserving approximation fits both budgets'in raw,'Only original before-pose topology-budget traceback allowed')
    phases=[]
    for line in raw.splitlines():
        if line.startswith(b'{'):
            try:record=volume.strict_json(line)
            except ValueError:continue
            if type(record)is dict and record.get('mode')in ('public_frontends_only','pinned_volume_frontends_only'):phases.append(record)
    require(phases and phases[-1].get('stage')=='object_pose_full' and phases[-1].get('phase')=='fail','Original launcher must end at pose topology FAIL')
    base=archive.canonical(root/f'outputs/episode_{args.episode:06d}');original=base/'object_pose_full';destination=base/'object_pose_topology_failed_v1'
    empty=empty_state(original);archive.canonical(destination);require(not destination.exists(),'Archive destination occupied')
    receipt_dir=archive.canonical(root/'results'/f'empty-pose-transition-{args.episode:06d}-{revision}')
    require(not receipt_dir.exists(),'Exclusive transition receipt already occupied')
    inactive=inactivity(unit,args.failed_pid,probe,proc,cgroups)
    vetted=volume_binding(root,code,args.episode)
    require(archive.identity(log,2_000_000)==logpin and empty_state(original)==empty,'Original failure changed during preflight')
    return dict(root=root,code=code,revision=revision,args=args,current_source=current,old_source=old,log=log,log_identity=logpin,unit=inactive,original=original,destination=destination,empty=empty,receipt_dir=receipt_dir,volume=vetted)


def transition(plan,probe=unit_state,rename=archive.rename_noreplace,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup'),gpu=gpu_idle,lock=lock_state,sync=archive._sync_directory):
    require(os.geteuid()==0,'Only host root may archive original namespace')
    inode=lock(plan['root']);gpu()
    fresh=validate(plan['root'],plan['code'],plan['revision'],plan['args'],probe,proc,cgroups)
    require(all(fresh[key]==plan[key]for key in ('current_source','old_source','log_identity','unit','empty','volume')),'Failure/source/volume state changed before archive')
    fresh['receipt_dir'].mkdir(mode=0o700)
    require(lock(plan['root'])==inode,'GPU lock changed');gpu()
    require(empty_state(fresh['original'])==fresh['empty'] and not fresh['destination'].exists(),'Original empty directory changed before atomic rename')
    rename(fresh['original'],fresh['destination']);sync(fresh['destination'].parent)
    after=empty_state(fresh['destination']);require(after[:7]==fresh['empty'][:7]and after[7]==fresh['empty'][7],'Archived original inode/owner/mode/content changed')
    require(not fresh['original'].exists()and not fresh['original'].is_symlink(),'Original namespace still exists')
    require(binding(fresh['code'],fresh['revision'])==fresh['current_source'] and binding(fresh['root']/'jobs'/fresh['args'].failed_revision/fresh['args'].failed_entrypoint/'code',fresh['args'].failed_revision)==fresh['old_source']
            and archive.identity(fresh['log'],2_000_000)==fresh['log_identity'] and volume_binding(fresh['root'],fresh['code'],fresh['args'].episode)==fresh['volume'],'Original log/source/volume changed after archive')
    require(inactivity(fresh['args'].failed_unit,fresh['args'].failed_pid,probe,proc,cgroups)==fresh['unit'] and lock(plan['root'])==inode,'Original inactivity/lock changed');gpu()
    receipt=dict(stage='empty_pose_transition',status='pass',original_status='fail',episode_index=fresh['args'].episode,producer_revision=fresh['revision'],
        original_producer_revision=fresh['args'].failed_revision,original_failed_unit=fresh['args'].failed_unit,historical_pid=fresh['args'].failed_pid,unit_state=fresh['unit'],
        current_source=fresh['current_source'],original_source=fresh['old_source'],original_log=fresh['log_identity'],volume=fresh['volume'],
        before=dict(path=str(fresh['original'].relative_to(fresh['root'])),state=list(fresh['empty'])),after=dict(path=str(fresh['destination'].relative_to(fresh['root'])),state=list(after)),
        atomic_rename_noreplace=True,files_deleted=False,original_failure_reinterpreted=False,geometry_changed=False,inference_performed=False,ground_truth_used=False,
        completed_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))
    target=fresh['receipt_dir']/'report.json';fd=os.open(target,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400)
    with os.fdopen(fd,'wb')as stream:stream.write((json.dumps(receipt,indent=2,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
    sync(target.parent);return receipt


def parser():
    class Once(argparse.Action):
        def __call__(self,parser,namespace,value,option_string=None):
            require(getattr(namespace,self.dest,None)is None,'Each explicit argument must appear once');setattr(namespace,self.dest,value)
    result=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    def integer(value):
        require(bool(re.fullmatch('0|[1-9][0-9]*',value)),'Canonical integer required');return int(value)
    result.add_argument('--episode',type=integer,choices=range(30),required=True,action=Once)
    result.add_argument('--failed-entrypoint',choices=('run_track1_frontends','run_track1_volume_frontends'),required=True,action=Once)
    for name in ('failed-unit','failed-revision','failed-script-sha256','failed-log-sha256'):result.add_argument('--'+name,required=True,action=Once)
    for name in ('failed-log-bytes','failed-pid'):result.add_argument('--'+name,type=integer,required=True,action=Once)
    return result


def main(argv=None):
    args=parser().parse_args(argv);root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    require(os.uname().sysname=='Linux'and root==ROOT and code==root/'jobs'/revision/JOB/'code'and Path(__file__)==code/SOURCES[0],'Actual canonical Linux host transition required')
    require(Path(archive.__file__)==code/SOURCES[2]and Path(volume.__file__)==code/SOURCES[3],'Actual audited archive/inventory helper paths required')
    require(re.fullmatch('[0-9a-f]{40}',revision)and re.fullmatch('[0-9a-f]{40}',args.failed_revision)and all(re.fullmatch('[0-9a-f]{64}',getattr(args,key))for key in ('failed_script_sha256','failed_log_sha256'))
            and args.failed_pid>1 and args.failed_log_bytes>0,'Independent historical SHA/PID/size controls required')
    plan=validate(root,code,revision,args);receipt=transition(plan)
    print(json.dumps(dict(stage=receipt['stage'],status=receipt['status'],original_status='fail',episode_index=args.episode,receipt=archive.identity(plan['receipt_dir']/'report.json',2_000_000))))


if __name__=='__main__':main()
