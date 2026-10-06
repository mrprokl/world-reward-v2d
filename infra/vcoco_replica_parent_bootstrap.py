"""Create only the missing private VM01 parent; no transfer/import/model retry."""
import os
from pathlib import Path
import signal
import stat
import sys
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import vcoco_observation_replica as original
rt,ROOT=original.rt,original.ROOT
ENTRY='run_vcoco_replica_parent_bootstrap'
PARENT=original.DEST.parent
OUTPUT=ROOT/'results/vcoco-replica-parent-bootstrap-v1'
BUDGET=30
REV='7c11662f98cf6f3db28909dce147473f1ab3bd0c'
FAILED=ROOT/f'results/vcoco-observation-replica-import-{REV}'
PINS={'report.json':dict(bytes=4969,sha256='877d595c1e91ec937270553595c40e6d8cb782269dd5f62b2eeddbf9de775ee1'),
 'manifest.json':dict(bytes=5166,sha256='7c0786e64de612080c01cf4699a63fafaf3b70f9ac6a9b70fc77e4d8d9b40174'),
 'export-receipt.json':dict(bytes=5638,sha256='cc7874d9c9fd21a6137f48066d7edfe8290af962eed5f8db4e9faff6a35591e3')}
DIAGNOSTIC='configs/vcoco_replica_parent_diagnostic.json'
DIAGNOSTIC_PIN=dict(bytes=2159,sha256='5db587939b68a8e3955039c3cf68e6dd391e29d1cdfa100ab3a9ba7c93941fa0')
HELPERS=tuple(dict.fromkeys(('infra/vcoco_replica_parent_bootstrap.py','infra/run_vcoco_replica_parent_bootstrap.sh',DIAGNOSTIC,*original.HELPERS)))


def source(code,revision):
    value=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    rt.require(Path(__file__).resolve()==code/HELPERS[0]and Path(original.__file__).resolve()==code/'infra/vcoco_observation_replica.py','Actual unchanged helper origins required')
    old=ROOT/'jobs'/REV/original.ENTRY/'code';oldsource=rt.source(ROOT,old,REV,original.ENTRY,original.HELPERS)
    rt.require(oldsource['entries']==327 and sum(p.is_file()for p in old.rglob('*'))==322
        and all(value['helpers'][n]==p for n,p in oldsource['helpers'].items()),'Complete unchanged original import source required')
    return dict(source=value,states=original.bank.source_state(code),original_source=oldsource,original_states=original.bank.source_state(old))


def prior(code):
    values={n:rt.pinned(FAILED/n,p,256 << 10)for n,p in PINS.items()};r=values['report.json']
    rt.require(r['schema']==original.SCHEMA and r['phase']=='import'and r['status']=='fail'and r['producer_revision']==REV
        and r['error_type']=='ValueError'and r['elapsed_seconds']==.5756366139976308 and r['files']==36
        and r['source_inputs_rehashed_after']is r['outputs_sealed']is r['archive_removed']is True and r['blob_cleanup_verified']is False,
        'Preserved original immutable import failure required')
    original.validate_manifest(values['manifest.json'],REV)
    original.receipt_from_base64(__import__('base64').b64encode((FAILED/'export-receipt.json').read_bytes()).decode(),PINS['export-receipt.json'],REV)
    d=rt.pinned(code/DIAGNOSTIC,DIAGNOSTIC_PIN,16 << 10)
    rt.require(d['status']=='pass'and d['metadata_only']is d['source_and_receipt_unchanged']is d['parent_metadata_unchanged']is True
        and d['no_directory_created']is d['no_rename_called']is True and d['producer_revision']==REV and d['original_report']==PINS['report.json']
        and d['parent_canonical']is True and d['parent_is_dir']is False,'Independently observed missing-parent cause required')
    rt.require({p.name for p in FAILED.iterdir()}==set(PINS)and stat.S_IMODE(FAILED.lstat().st_mode)==0o500
        and FAILED.lstat().st_uid==FAILED.lstat().st_gid==0,'Original closed failure namespace required')
    return dict(identities=PINS,states={str(p):original.snapshot(p)for p in (FAILED,*FAILED.iterdir(),code/DIAGNOSTIC)},
        diagnostic_identity=DIAGNOSTIC_PIN,diagnostic_is_saved_source_declaration=True,original_source=r['source_binding'])


def prepare_parent(parent):
    """Only one exact root-private directory; never chmod an existing namespace."""
    rt.canonical(parent)
    for p in (parent.parent,*parent.parent.parents):
        s=p.lstat();rt.require(stat.S_ISDIR(s.st_mode)and s.st_uid==s.st_gid==0 and stat.S_IMODE(s.st_mode)==0o755,'Canonical original root/srv required')
    created=False
    if not parent.exists():parent.mkdir(mode=0o700);created=True
    s=parent.lstat();rt.require(stat.S_ISDIR(s.st_mode)and s.st_uid==s.st_gid==0 and stat.S_IMODE(s.st_mode)==0o700,
        'Existing/new parent must already be private root700')
    rt.require(not any(parent.iterdir()),'Fresh empty parent required; no existing datasets touched')
    original.sync(parent);original.sync(parent.parent)
    return dict(created=created,path=str(parent),state=[s.st_dev,s.st_ino,s.st_uid,s.st_gid,s.st_mode],private_mode='0o700')


def parent_after(parent,proof):
    rt.canonical(parent);s=parent.lstat()
    rt.require([s.st_dev,s.st_ino,s.st_uid,s.st_gid,s.st_mode]==proof['state']and not any(parent.iterdir()),'Created private parent changed')


def run(code,revision):
    started=time.monotonic();deadline=started+BUDGET
    def expired(*_):raise TimeoutError('Inclusive bootstrap budget')
    handlers={s:signal.signal(s,expired)for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    rt.require(sys.platform=='linux'and os.geteuid()==0,'Azure root-only bootstrap');original.transport.verify_azure_peer('import')
    rt.require(not OUTPUT.exists()and not OUTPUT.is_symlink(),'Fresh bootstrap receipt');OUTPUT.mkdir(mode=0o700);s=OUTPUT.lstat();owner=(s.st_dev,s.st_ino,s.st_uid)
    before=old=parent=None;report=dict(schema='world_reward.vcoco_replica_parent_bootstrap.v1',stage='private_replica_parent_bootstrap',status='fail',producer_revision=revision,
        original_failed_producer=REV,failed_receipt_pins=PINS,diagnostic_identity=DIAGNOSTIC_PIN,source_inputs_rehashed_after=False,
        data_imported=False,blob_read=False,media_decoded=False,models_loaded=False,GPU_used=False,reference_metadata_read=False,quality_verified=False,ownership_verified=False,adoption=False)
    try:
        before=source(code,revision);old=prior(code)
        rt.require(old['original_source']==before['original_source'],'Failed receipt must bind exact original source')
        for p in (original.DEST,PARENT/(original.DEST.name+'.stage-'+REV)):
            rt.canonical(p);rt.require(not p.exists()and not p.is_symlink(),'No original destination/staging permitted')
        parent=prepare_parent(PARENT);report.update(parent=parent,source_binding=before['source'],original_source=before['original_source'],status='pass')
    except BaseException as exc:report['error_type']=original.bank.error(exc)
    finally:
        try:
            rt.require(before is not None and old is not None and source(code,revision)==before and prior(code)==old,'Whole current/original source/failure/diagnostic changed')
            if parent is not None:parent_after(PARENT,parent)
            rt.require(not original.DEST.exists(),'Bootstrap cannot install a replica');report['source_inputs_rehashed_after']=True
        except BaseException as exc:report.update(status='fail',post_error_type=original.bank.error(exc))
        try:original.publish(OUTPUT,report,deadline,started,owner,set())
        finally:
            signal.alarm(0)
            for s,h in handlers.items():signal.signal(s,h)
    return report


if __name__=='__main__':
    value=run(Path(os.environ.get('WR_CODE','/invalid')),os.environ.get('WR_CODE_REVISION',''))
    print(original.encode({k:value[k]for k in ('stage','status')}).decode(),end='');raise SystemExit(0 if value['status']=='pass'else 1)
