"""One atomic archive of the original download-only YCBV timeout; no data reads."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess

import atomic_metadata as atomic

ROOT=Path('/srv/scenesmith/world-reward')
BASE='validation/ycbv_point_pose_v1'
ARCHIVE=BASE+'_acquisition_failed_v1'
JOB='run_ycbv_acquire_transition'
FAILED_PINS='configs/ycbv_point_failed_acquisition_pins.json'
CONTINUATION_PINS='configs/ycbv_point_continuation_pins.json'
FILES=('infra/ycbv_point_acquire.py','infra/run_ycbv_point_acquire.sh','infra/tudl_acquire.py','configs/ycbv_point_protocol.json')
FIELDS=('LoadState','ActiveState','SubState','Result','MainPID','ExecMainPID','ControlGroup')


def require(value,message):
    if not value:raise ValueError(message)


def read(path,pin=None,*,readonly=True):
    path=atomic.canonical(path);s=path.lstat()
    require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and (not readonly or not s.st_mode&0o222) and 0<s.st_size<=200000,'Bounded immutable original metadata required')
    raw=path.read_bytes();actual=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    require(atomic._state(path)==(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink),'Metadata changed')
    if pin is not None:require(actual==atomic._pin(pin),'Independent metadata bytes differ')
    return raw,actual


def load(path,pin=None):
    raw,identity=read(path,pin);return atomic._json(raw),identity


def source(code,revision):
    atomic.canonical(code)
    require(re.fullmatch('[0-9a-f]{40}',revision),'Full immutable producer required')
    files={n:read(code/n)[1] for n in FILES}
    markers={n:atomic.identity(code.parent/n,128) for n in ('revision','source-sha256')}
    require((code.parent/'revision').read_bytes()==(revision+'\n').encode()and re.fullmatch(b'[0-9a-f]{64}\n',(code.parent/'source-sha256').read_bytes()),'Original dispatch markers differ')
    return dict(files=files,markers=markers)


def validate_pins(pins):
    require(type(pins)is dict and set(pins)=={'schema','failure','unit','pid','log','cid'}
        and pins['schema']=='world-reward-ycbv-failed-acquisition-pins-v1','Exact original failure pins required')
    fail=pins['failure'];require(type(fail)is dict and set(fail)=={'bytes','sha256','producer_revision','script_sha256'},'Exact failed producer required')
    atomic._pin({k:fail[k]for k in ('bytes','sha256')})
    require(re.fullmatch('[0-9a-f]{40}',str(fail['producer_revision']))and re.fullmatch('[0-9a-f]{64}',str(fail['script_sha256'])),'Original full source pins required')
    require(type(pins['pid'])is int and pins['pid']>1 and type(pins['unit'])is str
        and re.fullmatch('world-reward-ycbv-[a-z0-9-]{1,80}\\.service',pins['unit']),'Owned original unit and PID required')
    atomic._pin(pins['log']);atomic._pin(pins['cid'])


def command(args):
    value=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=5,check=False)
    require(value.returncode==0 and len(value.stdout)<=4096,'Bounded runtime-state query failed');return value.stdout


def inactive(pins,*,query=command,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup')):
    values={}
    for line in query(['systemctl','show','--no-pager',*('--property='+k for k in FIELDS),pins['unit']]).splitlines():
        key,sep,value=line.partition('=');require(sep and key in FIELDS and key not in values,'Exact original unit fields required');values[key]=value
    require(set(values)==set(FIELDS)and values['MainPID']=='0'and values['ExecMainPID']in ('0',str(pins['pid'])),'Original unit must have no running main process')
    failed=values['LoadState']=='loaded'and values['ActiveState']==values['SubState']=='failed'and values['Result']in ('exit-code','timeout','signal')
    collected=values['LoadState']=='not-found'and values['ActiveState']=='inactive'and values['SubState']=='dead'
    require(failed or collected,'Only failed or collected original unit allowed')
    require(not (proc/str(pins['pid'])).exists()and not (proc/str(pins['pid'])).is_symlink(),'Original PID alive/reused; never kill it')
    expected='/system.slice/'+pins['unit'];require(values['ControlGroup']in ('',expected),'Original cgroup differs')
    group=atomic.canonical(cgroups/expected.lstrip('/'))
    if group.exists():
        paths=list(group.rglob('cgroup.procs'));require(paths,'Cannot prove original cgroup empty')
        require(all(not p.read_text().strip()for p in paths),'Original cgroup still running')
    return values


def failure(root,pins,base):
    validate_pins(pins);base=atomic.canonical(base);s=base.lstat()
    require(stat.S_ISDIR(s.st_mode)and s.st_uid==1000 and s.st_mode&0o777==0o700
        and {p.name for p in base.iterdir()}=={'report.json','.container.cid'},'Only original report/CID after complete cleanup allowed')
    report,pin=load(base/'report.json',{k:pins['failure'][k]for k in ('bytes','sha256')})
    expected=dict(stage='external_ycbv_contiguous_rgb_only_acquisition',status='fail',phase='download',error_type='TimeoutError',
        producer_revision=pins['failure']['producer_revision'],script_sha256=pins['failure']['script_sha256'],budget_seconds=900,
        device='cpu',gpu_used=False,inference_performed=False,challenge_inputs_used=False,
        models_downloaded=False,train_downloaded=False,sparse_test_downloaded=False,source_rehashed_after=True,disposable_archives_removed=True)
    require(all(type(report.get(k))is type(v)and report[k]==v for k,v in expected.items()),'Only original sealed download-only timeout qualifies')
    require(not any(k in report for k in ('selected_frames','selection_before_private_annotation_values','archive_members','public_manifest','retention_receipt','all_instances_retained')),'ZIP/private interpretation already occurred')
    require(report.get('active_archive')=='ycbv_test_all.zip'and 'cleanup_error_type'not in report,'Full original download cleanup required')
    old=root/'jobs'/pins['failure']['producer_revision']/'run_ycbv_point_acquire/code'
    sources=source(old,pins['failure']['producer_revision'])
    require(sources==report.get('source_helpers')and sources['files'][FILES[0]]['sha256']==pins['failure']['script_sha256'],'Original report/source ancestry differs')
    protocol=load(old/FILES[-1])[0];require(protocol.get('limits',{}).get('seconds')==900 and protocol.get('output',{}).get('base')==BASE,'Only first acquisition attempt qualifies')
    raw,cidpin=read(base/'.container.cid',pins['cid']);cid=raw.decode().strip()
    require(re.fullmatch('[0-9a-f]{64}',cid),'Exact original Docker CID required')
    return dict(report=pin,cid=cid,cid_identity=cidpin,source=sources,directory_inode=[s.st_dev,s.st_ino])


def own_source(code,revision):
    require(code==ROOT/'jobs'/revision/JOB/'code'and Path(__file__).resolve()==code/'infra/ycbv_acquire_transition.py','Actual transition dispatch required')
    atomic._markers(code,revision)
    return {n:read(code/n)[1]for n in ('infra/ycbv_acquire_transition.py','infra/run_ycbv_acquire_transition.sh','infra/atomic_metadata.py',FAILED_PINS)}


def archive(root,code,revision,*,query=command,proc=Path('/proc'),cgroups=Path('/sys/fs/cgroup'),rename=atomic.rename_noreplace):
    before=own_source(code,revision);pins,pin=load(code/FAILED_PINS);validate_pins(pins)
    src=root/BASE;dst=atomic.canonical(root/ARCHIVE);out=atomic.canonical(root/'results'/('ycbv-acquire-continuation-'+revision))
    require(not dst.exists()and not out.exists(),'Archive/receipt occupied: no second transition')
    proof=failure(root,pins,src);unit=inactive(pins,query=query,proc=proc,cgroups=cgroups)
    require(not query(['docker','ps','-aq','--no-trunc','--filter','id='+proof['cid']]).strip(),'Original labelled container still exists')
    log=root/'results'/(pins['unit'].removeprefix('world-reward-').removesuffix('.service')+'.log')
    _,logpin=read(log,pins['log'],readonly=False)
    require(failure(root,pins,src)==proof and inactive(pins,query=query,proc=proc,cgroups=cgroups)==unit
        and own_source(code,revision)==before,'Original failure/runtime changed before rename')
    out.mkdir(mode=0o700)
    result=dict(schema='world-reward-ycbv-acquire-continuation-v1',stage='archive_first_ycbv_download_timeout',status='fail',
        producer_revision=revision,script_sha256=before['infra/ycbv_acquire_transition.py']['sha256'],source_helpers=before,
        failed_pins_identity=pin,original_failure=proof,failed_unit=unit,original_log=logpin,archive_relative=ARCHIVE,
        original_failure_preserved=False,atomic_noreplace=False,private_annotation_values_read=False,acquisition_attempts_authorized=0)
    with os.fdopen(os.open(out/'report.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400),'w')as stream:
        try:
            require(failure(root,pins,src)==proof and own_source(code,revision)==before
                and inactive(pins,query=query,proc=proc,cgroups=cgroups)==unit
                and not query(['docker','ps','-aq','--no-trunc','--filter','id='+proof['cid']]).strip(),'Failure changed at rename boundary')
            rename(src,dst);result['atomic_noreplace']=True;atomic._sync_directory(src.parent)
            require(failure(root,pins,dst)==proof and own_source(code,revision)==before and read(log,pins['log'],readonly=False)[1]==logpin,'Archived original changed')
            result.update(status='pass',original_failure_preserved=True,acquisition_attempts_authorized=2)
        except BaseException as error:
            result['error_type']=type(error).__name__;raise
        finally:
            json.dump(result,stream,allow_nan=False);stream.write('\n');stream.flush();os.fsync(stream.fileno());atomic._sync_directory(out)
    return result


def continuation(root,code):
    """Host pre/post authorization: occupied v1 prevents any third acquisition."""
    pins,pin=load(code/CONTINUATION_PINS)
    require(type(pins)is dict and set(pins)=={'schema','failed_pins','transition'}
        and pins['schema']=='world-reward-ycbv-continuation-pins-v1','Independent actual continuation receipt required')
    oldpins,_=load(code/FAILED_PINS,pins['failed_pins']);validate_pins(oldpins)
    transition=pins['transition'];require(type(transition)is dict and set(transition)=={'bytes','sha256','producer_revision','script_sha256'},'Exact actual transition producer required')
    rev=transition['producer_revision'];require(re.fullmatch('[0-9a-f]{40}',str(rev)),'Full transition revision required')
    path=root/'results'/('ycbv-acquire-continuation-'+rev)/'report.json'
    report,_=load(path,{k:transition[k]for k in ('bytes','sha256')})
    require(report.get('schema')=='world-reward-ycbv-acquire-continuation-v1'and report.get('status')=='pass'
        and report.get('stage')=='archive_first_ycbv_download_timeout'and report.get('producer_revision')==rev
        and report.get('script_sha256')==transition['script_sha256']and report.get('original_failure_preserved')is True
        and report.get('atomic_noreplace')is True and report.get('private_annotation_values_read')is False
        and report.get('acquisition_attempts_authorized')==2 and report.get('archive_relative')==ARCHIVE,'Genuine one-time technical transition required')
    original=failure(root,oldpins,root/ARCHIVE)
    original_log=root/'results'/(oldpins['unit'].removeprefix('world-reward-').removesuffix('.service')+'.log')
    require(read(original_log,oldpins['log'],readonly=False)[1]==report.get('original_log'),'Original failed log changed')
    first_code=root/'jobs'/oldpins['failure']['producer_revision']/'run_ycbv_point_acquire/code'
    old_protocol=load(first_code/FILES[-1])[0];new_protocol=load(code/FILES[-1])[0]
    expected=json.loads(json.dumps(old_protocol));expected['limits']['seconds']=3600;expected['limits']['cleanup_seconds']=180
    require(new_protocol==expected,'Only the predeclared acquisition/cleanup time budgets may change')
    require(original==report.get('original_failure'),'Original archived FAIL differs')
    historical=root/'jobs'/rev/JOB/'code'
    require(type(report.get('source_helpers'))is dict and set(report['source_helpers'])==
        {'infra/ycbv_acquire_transition.py','infra/run_ycbv_acquire_transition.sh','infra/atomic_metadata.py',FAILED_PINS},'Exact transition source inventory required')
    helpers={n:read(historical/n)[1]for n in report['source_helpers']}
    require(helpers==report['source_helpers']and helpers.get('infra/ycbv_acquire_transition.py',{}).get('sha256')==transition['script_sha256']
        and helpers.get(FAILED_PINS)==pins['failed_pins'],'Actual transition sources/pins differ')
    atomic._markers(historical,rev)
    return hashlib.sha256(json.dumps(dict(pins=pins,pin=pin,original=original),sort_keys=True).encode()).hexdigest()


def main():
    p=argparse.ArgumentParser(allow_abbrev=False);p.add_argument('--verify-continuation',action='store_true');args=p.parse_args()
    root=Path(os.environ['WR_ROOT']);code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION']
    require(root==ROOT and os.geteuid()==0 and os.uname().nodename=='world-reward-ncc-h100-02','Owned Azure VM02 root host only')
    if args.verify_continuation:print(continuation(root,code))
    else:print(json.dumps({'status':archive(root,code,rev)['status'],'archive_relative':ARCHIVE}))


if __name__=='__main__':main()
