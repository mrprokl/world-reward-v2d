"""Azure-only immutable public RGB/full-bank replica; no model or reference read."""
import argparse
import base64
import hashlib
import io
import math
import os
from pathlib import Path
import re
import signal
import stat
import sys
import tarfile
import time

sys.path[:0]=[str(Path(__file__).resolve().parent),str(Path(__file__).resolve().parents[1]/'src')]
import articulated_runtime_transfer as transport
import atomic_metadata as atomic
import vcoco_pilot_endpoint_bank as bank
rt=bank.rt
ROOT=rt.ROOT
ENTRY='run_vcoco_observation_replica'
DEST=Path('/srv/world-reward-data/vcoco_observation_replica_v1')
REV='e1a01f303f533344a10ef0e4eb216c2eb13c9182'
CLOSURE='ddefd7f4663b15fee9dd8f6bb14693ea1f0d343e4407100b1152aafbcdc939ec'
SCHEMA='world_reward.vcoco_observation_replica.v1'
BUDGET=300
MAXIMUM=32 << 20
MAX_MANIFEST=256 << 10
PINS={'host.json':dict(bytes=86447,sha256='2cc2244058da86265a44240a04ea943b374741b1abfd219173487fde8b4d7123'),
 'native.json':dict(bytes=59764,sha256='2de5f967f0f378b4b739489dc754896de9ab787ac034eccb53fa67abf9cfb4bc'),
 'proof.json':dict(bytes=9264,sha256='c1dc7df9f78f15b32589f56dfbe02f53bef6805a877d9eaed86bc4118a250a78')}
PUBLIC=dict(bytes=3293,sha256='b938a7d22737f4399fc14cc0683863e920efaecef5ec544f115180119e4f1680')
CID=dict(bytes=64,sha256='60fc6792a5ee39b4446493e8399901cee63eaf8aa9215aceeeca34e89c6ba671')
AUDIT=dict(file='results/audits/vcoco_pilot_endpoint_banks_v1_actual.json',bytes=4285,
 sha256='12d71f316d7e1859f55146fd836dce54d86c610d90c99e7acfb24e17e20e4274',
 declaration_only=True,live_original_source_on_receiver=False,quality_verified=False)
HELPERS=tuple(dict.fromkeys(('infra/vcoco_observation_replica.py','infra/run_vcoco_observation_replica.sh',
 'infra/articulated_runtime_transfer.py','infra/runtime_image_archive.py','infra/atomic_metadata.py',*bank.HELPERS)))
encode=bank.encode


def pin(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def check(deadline):
    if not math.isfinite(deadline)or time.monotonic()>=deadline:raise TimeoutError('Inclusive300s replica deadline')
def snapshot(path):
    p=rt.canonical(path);s=p.lstat()
    return tuple(getattr(s,k)for k in('st_dev','st_ino','st_mode','st_size','st_nlink','st_uid','st_gid','st_mtime_ns','st_ctime_ns'))
def fixed_pin(value,maximum=MAXIMUM):
    rt.require(type(value)is dict and set(value)=={'bytes','sha256'}and type(value['bytes'])is int
        and 0<value['bytes']<=maximum and type(value['sha256'])is str and re.fullmatch('[0-9a-f]{64}',value['sha256']),
        'Exact independent byte/SHA pin required');return value

def authenticate(code,revision):
    source=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    rt.require(Path(__file__).resolve()==code/HELPERS[0]and Path(bank.__file__).resolve()==code/'infra/vcoco_pilot_endpoint_bank.py'
        and Path(transport.__file__).resolve()==code/'infra/articulated_runtime_transfer.py'
        and Path(atomic.__file__).resolve()==code/'infra/atomic_metadata.py','Actual helper origins required')
    return dict(source=source,states=bank.source_state(code),markers={n:snapshot(code.parent/n)for n in ('revision','source-sha256')})


def export_inputs(code,source):
    old=ROOT/'jobs'/REV/bank.ENTRY/'code';original=rt.source(ROOT,old,REV,bank.ENTRY,bank.HELPERS)
    rt.require(original['entries']==321 and original['closure_sha256']==CLOSURE
        and sum(p.is_file()for p in old.rglob('*'))==316,'Original complete316-file bank source required')
    rt.require(all(source['helpers'][n]==original['helpers'][n]for n in bank.HELPERS),'Reused original producer helpers changed')
    recipe,_=bank.configuration(code,source)
    public=bank.public_inputs(PUBLIC,16)
    rt.require([bank.rgb_inputs.original_slot(r,16)for r in public['images']]==list(range(16)),'All original16 slots required')
    values={n:rt.pinned(bank.OUTPUT/n,p,2 << 20)for n,p in PINS.items()}
    host,native,proof=(values[n]for n in ('host.json','native.json','proof.json'))
    rt.require(host['status']=='pass'and host['stage']=='vcoco_pilot_endpoint_bank_host'and host['schema']==recipe['schema']
        and host['producer_revision']==REV and host['source_binding']==original and host['public_inputs_identity']==PUBLIC
        and host['native_report_identity']==PINS['native.json']and host['native_exit_status']==0 and host['acquired_images']==16
        and host['native_images']==native['images']and host['owned_cleanup_verified']is host['outputs_sealed']is
            host['source_inputs_runtime_assets_rehashed_after']is True
        and all(host[k]is False for k in ('ground_truth_used','reference_metadata_read','split_metadata_read','FIT_performed',
            'ownership_verified','quality_verified','adoption')),'Original sealed label-blind bank host required')
    bank.validate_report(native,recipe,REV,PINS['proof.json'],public)
    rt.require(proof['source']==original and proof['native_files']=={n:original['helpers'][n]for n in bank.NATIVE_FILES}
        and proof['inputs_identity']==PUBLIC and proof['images']==16 and proof['image_id']==bank.IMAGE
        and native['runtime_identity']==proof['owl_runtime']==host['original_qualification']['owl']['native_runtime'],
        'Original saved proof/runtime declaration differs')
    rt.require(host['original_recipe_identity']==recipe['original_recipe']and proof['assets'],
        'Original policy and model declarations required')
    paths={'inputs/manifest.json':bank.DATA/'manifest.json',**{'inputs/'+r['file']:bank.DATA/r['file']for r in public['images']},
        **{'banks/'+r['file']:bank.OUTPUT/r['file']for r in native['images']},**{'banks/'+n:bank.OUTPUT/n for n in PINS}}
    files={n:rt.identity(p,MAXIMUM)for n,p in paths.items()}
    rt.require(all(stat.S_IMODE(p.lstat().st_mode)==0o400 and p.lstat().st_uid==p.lstat().st_gid==0 for p in paths.values()),
        'Original private root-owned400 leaves required')
    rt.require({p.name for p in bank.OUTPUT.iterdir()}=={n.split('/')[1]for n in files if n.startswith('banks/')}|{'.container.cid'}
        and rt.identity(bank.OUTPUT/'.container.cid',100)==CID,'Exact original output inventory required')
    for folder in (bank.DATA,bank.OUTPUT):
        rt.require(stat.S_IMODE(folder.lstat().st_mode)==0o500 and folder.lstat().st_uid==0,'Original private readonly directory required')
    manifest=dict(schema=SCHEMA,export_revision=None,producer_revision=REV,
        original_source_declaration=dict(entries=321,closure_sha256=CLOSURE),original_audit_declaration=AUDIT,
        public_identity=PUBLIC,files=files)
    validate_manifest(manifest,allow_unbound=True)
    states={str(p):snapshot(p)for folder in (bank.DATA,bank.OUTPUT)for p in (folder,*sorted(folder.iterdir()))}
    states.update({str(old.parent/n):snapshot(old.parent/n)for n in ('revision','source-sha256')})
    return manifest,paths,dict(source=original,source_states=bank.source_state(old),states=states)


def validate_manifest(m,revision=None,*,allow_unbound=False):
    keys={'schema','export_revision','producer_revision','original_source_declaration','original_audit_declaration','public_identity','files'}
    names={'inputs/manifest.json',*[f'inputs/image_{i:06d}.jpg'for i in range(16)],
        *[f'banks/image_{i:06d}.npz'for i in range(16)],*['banks/'+n for n in PINS]}
    rt.require(type(m)is dict and set(m)==keys and m['schema']==SCHEMA and m['producer_revision']==REV
        and m['original_source_declaration']==dict(entries=321,closure_sha256=CLOSURE)and m['original_audit_declaration']==AUDIT
        and m['public_identity']==PUBLIC and type(m['files'])is dict and set(m['files'])==names,'Exact36-file replica manifest required')
    rt.require((allow_unbound and m['export_revision']is None)or type(m['export_revision'])is str
        and re.fullmatch('[0-9a-f]{40}',m['export_revision'])and(revision is None or m['export_revision']==revision),
        'Original exclusive export revision required')
    for p in m['files'].values():fixed_pin(p,16 << 20)
    rt.require(m['files']['inputs/manifest.json']==PUBLIC and all(m['files']['banks/'+n]==p for n,p in PINS.items())
        and sum(p['bytes']for p in m['files'].values())<=MAXIMUM-(1 << 20),'Bounded original archive payload required')
    return m


def export_archive(blob,manifest,paths,deadline):
    raw=encode(manifest);rt.require(len(raw)<=MAX_MANIFEST,'Bounded first manifest required');writer=transport.BlockWriter(blob)
    with tarfile.open(fileobj=writer,mode='w|',format=tarfile.USTAR_FORMAT)as archive:
        for name in ('manifest.json',*sorted(paths)):
            check(deadline);size=len(raw)if name=='manifest.json'else manifest['files'][name]['bytes']
            row=tarfile.TarInfo(name);row.size=size;row.mode=0o400;row.uid=row.gid=row.mtime=0
            if name=='manifest.json':archive.addfile(row,io.BytesIO(raw))
            else:
                rt.require(rt.identity(paths[name],16 << 20)==manifest['files'][name],'Source bytes changed before upload')
                with paths[name].open('rb')as stream:archive.addfile(row,stream)
            rt.require(writer.size<=MAXIMUM,'Bounded USTAR output required')
    rt.require(writer.size<=MAXIMUM,'Bounded complete USTAR output required');archive_pin=writer.finish();check(deadline)
    with blob.request('HEAD')as response:
        etag=response.headers.get('ETag','');rt.require(response.status==200
            and response.headers.get('Content-Length')==str(archive_pin['bytes'])and valid_etag(etag),'Exclusive committed blob HEAD differs')
    return dict(archive_identity=archive_pin,manifest_identity=pin(raw),blob_etag=etag)


def valid_etag(value):return type(value)is str and re.fullmatch(r'"[0-9A-Za-z-]{1,128}"',value)is not None

def verify_archive(path,archive_pin,manifest_pin,revision,deadline):
    rt.require(rt.identity(path,MAXIMUM)==fixed_pin(archive_pin),'Independent original archive identity differs')
    fixed_pin(manifest_pin,MAX_MANIFEST);seen=set();manifest=None
    with path.open('rb')as stream:
        while True:
            check(deadline);header=stream.read(512);rt.require(len(header)==512,'Truncated USTAR header')
            if header==b'\0'*512:
                tail=stream.read();rt.require(len(tail)>=512 and len(tail)%512==0 and not any(tail),'Only complete zero TAR tail required');break
            rt.require(header[257:265]==b'ustar\00000'and not any(header[345:500]),'Only original USTAR, no prefix/PAX/GNU extension')
            info=tarfile.TarInfo.frombuf(header,'utf-8','strict')
            rt.require(info.type==tarfile.REGTYPE and not info.linkname and info.mode==0o400 and info.uid==info.gid==0
                and not info.uname and not info.gname and info.mtime==0 and info.name not in seen,'Only unique regular unaliased TAR entries')
            if not seen:rt.require(info.name=='manifest.json'and info.size==manifest_pin['bytes'],'First pinned manifest required')
            else:rt.require(len(seen)<=36 and info.name==sorted(manifest['files'])[len(seen)-1]
                and info.size==manifest['files'][info.name]['bytes'],'Unknown/out-of-order archive member/size')
            expected=manifest_pin if not seen else manifest['files'][info.name]
            digest=hashlib.sha256();remaining=info.size;first=bytearray()
            while remaining:
                data=stream.read(min(1 << 20,remaining));rt.require(data,'Truncated TAR payload')
                digest.update(data);remaining-=len(data)
                if not seen:first.extend(data)
            rt.require(dict(bytes=info.size,sha256=digest.hexdigest())==expected,'Archive member original SHA differs')
            if not seen:manifest=validate_manifest(rt.strict(first),revision)
            pad=stream.read((-info.size)%512);rt.require(not any(pad)and len(pad)==(-info.size)%512,'Nonzero/truncated TAR padding')
            seen.add(info.name)
    rt.require(seen=={'manifest.json',*manifest['files']}and rt.identity(path,MAXIMUM)==archive_pin,'Complete immutable37-member archive required')
    return manifest


def sync(path):
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW)
    try:os.fsync(fd)
    finally:os.close(fd)
def rename_new(stage,target):
    rt.canonical(stage);rt.canonical(target);atomic.rename_noreplace(stage,target)
def clean_stage(stage,owner,owned):
    s=stage.lstat();rt.require((s.st_dev,s.st_ino,s.st_uid)==owner,'Only owned staging directory cleanup')
    rt.require({str(p.relative_to(stage))for p in stage.rglob('*')}==set(owned),'Foreign staging entry')
    for folder in stage.iterdir():
        rt.require(folder.name in ('inputs','banks')and folder.is_dir()and not folder.is_symlink()
            and snapshot(folder)[:2]==owned[folder.name][:2],'Owned staging directory replaced')
        rt.canonical(folder);folder.chmod(0o700)
        for leaf in folder.iterdir():
            rt.canonical(leaf);rt.require(re.fullmatch(r'(manifest\.json|image_[0-9]{6}\.(jpg|npz)|host\.json|native\.json|proof\.json)',leaf.name)
                and stat.S_ISREG(leaf.lstat().st_mode)and leaf.lstat().st_nlink==1 and leaf.lstat().st_uid==os.getuid()
                and snapshot(leaf)[:2]==owned[str(leaf.relative_to(stage))][:2],'Only owned stage leaves')
            leaf.unlink()
        folder.rmdir()
    stage.chmod(0o700);stage.rmdir()


def install(path,stage,manifest,deadline):
    rt.canonical(stage);rt.canonical(DEST)
    rt.require(stage.parent.is_dir()and not DEST.exists()and not DEST.is_symlink(),'Fresh canonical destination required')
    stage.mkdir(mode=0o700);s=stage.lstat();owner=(s.st_dev,s.st_ino,s.st_uid);owned={}
    try:
        for name in ('inputs','banks'):
            (stage/name).mkdir(mode=0o700);owned[name]=snapshot(stage/name)
        seen={'manifest.json'}
        with tarfile.open(path,'r:')as archive:
            for member in archive:
                check(deadline)
                if member.name=='manifest.json':continue
                rt.require(member.name in manifest['files']and member.name not in seen and member.isreg()
                    and member.size==manifest['files'][member.name]['bytes']and not member.pax_headers,'Authenticated original member required');seen.add(member.name)
                target=stage/member.name
                with target.open('xb')as output,archive.extractfile(member)as source:
                    os.fchmod(output.fileno(),0o400);owned[member.name]=snapshot(target)
                    for block in iter(lambda:source.read(1 << 20),b''):output.write(block)
                    output.flush();os.fsync(output.fileno())
                rt.require(rt.identity(target,16 << 20)==manifest['files'][member.name],'Installed original bytes differ')
        rt.require(seen=={'manifest.json',*manifest['files']},'Complete extracted36 files required')
        for name in ('inputs','banks'):(stage/name).chmod(0o500);sync(stage/name)
        rt.require({str(p.relative_to(stage))for p in stage.rglob('*')}==set(owned)
            and all(snapshot(stage/n)[:2]==v[:2]for n,v in owned.items()),'Owned stage changed before commit')
        sync(stage);check(deadline);rename_new(stage,DEST);sync(DEST.parent)
    except BaseException:
        if stage.exists():clean_stage(stage,owner,owned)
        raise


def replica_state():return pin(encode({str(p.relative_to(DEST)):snapshot(p)for p in (DEST,*sorted(DEST.rglob('*')))}))

def replica_identity(manifest):
    rt.canonical(DEST);rt.require({p.name for p in DEST.iterdir()}=={'inputs','banks'}and stat.S_IMODE(DEST.lstat().st_mode)==0o700 and DEST.lstat().st_uid==DEST.lstat().st_gid==0,
        'Exclusive readonly replica namespace required')
    files={}
    for folder in DEST.iterdir():
        rt.require(folder.is_dir()and not folder.is_symlink()and stat.S_IMODE(folder.lstat().st_mode)==0o500 and folder.lstat().st_uid==folder.lstat().st_gid==0
            and {str(p.relative_to(DEST))for p in folder.iterdir()}=={n for n in manifest['files']if n.startswith(folder.name+'/')},
            'Complete readonly36-file replica required')
        for leaf in folder.iterdir():
            rt.require(stat.S_IMODE(leaf.lstat().st_mode)==0o400 and leaf.lstat().st_uid==leaf.lstat().st_gid==0,'Private root-owned replica leaf required')
            files[str(leaf.relative_to(DEST))]=rt.identity(leaf,16 << 20)
    rt.require(files==manifest['files'],'Complete replica original hashes differ');return files


def publish(out,report,deadline,started,owner,allowed,after_seal=None):
    with(out/'report.json').open('x+b')as stream:
        os.fchmod(stream.fileno(),0o400);opened=os.fstat(stream.fileno())
        def update():
            stream.seek(0);stream.write(encode(report));stream.truncate();stream.flush();os.fsync(stream.fileno())
        try:
            s=out.lstat();rt.require((s.st_dev,s.st_ino,s.st_uid)==owner and s.st_nlink>=2 and stat.S_IMODE(s.st_mode)==0o700
                and {p.name for p in out.iterdir()}==allowed|{'report.json'},'Foreign receipt namespace')
            leaves={p.name:rt.identity(p,MAXIMUM)for p in out.iterdir()if p.name!='report.json'}
            rt.require(all(stat.S_IMODE(p.lstat().st_mode)==0o400 and p.lstat().st_uid==opened.st_uid for p in out.iterdir()),'Owned immutable receipt leaves')
            out.chmod(0o500);sync(out);sync(out.parent);report.update(outputs_sealed=True,elapsed_seconds=time.monotonic()-started);update()
            rt.require(rt.identity(out/'report.json',256 << 10)==pin(encode(report))
                and snapshot(out/'report.json')[:2]==(opened.st_dev,opened.st_ino)
                and {p.name for p in out.iterdir()}==allowed|{'report.json'}
                and all(rt.identity(out/n,MAXIMUM)==p for n,p in leaves.items()),'Sealed publication changed');check(deadline)
            if after_seal is not None and report['status']=='pass':
                after_seal();report['blob_cleanup_verified']=True;report['elapsed_seconds']=time.monotonic()-started;update()
                rt.require(rt.identity(out/'report.json',256 << 10)==pin(encode(report))
                    and snapshot(out/'report.json')[:2]==(opened.st_dev,opened.st_ino)
                    and {p.name for p in out.iterdir()}==allowed|{'report.json'}
                    and all(rt.identity(out/n,MAXIMUM)==p for n,p in leaves.items()),'Final cleanup publication changed');check(deadline)
        except BaseException:report.update(status='fail',publication_failed=True);update()
    return report


def receipt_from_base64(value,expected,revision):
    rt.require(type(value)is str and len(value)<=24 << 10,'Bounded canonical export receipt encoding')
    raw=base64.b64decode(value,validate=True)
    rt.require(base64.b64encode(raw).decode()==value and pin(raw)==fixed_pin(expected,16 << 10),'Independent exact export receipt differs')
    r=rt.strict(raw)
    rt.require(r['schema']==SCHEMA and r['phase']=='export'and r['status']=='pass'and r['producer_revision']==revision
        and r['outputs_sealed']is r['source_inputs_rehashed_after']is True and r['files']==36
        and r['original_producer_revision']==REV and valid_etag(r['blob_etag'])
        and r['original_source_declaration']==dict(entries=321,closure_sha256=CLOSURE)and r['original_audit_declaration']==AUDIT
        and all(r[k]is False for k in ('models_loaded','GPU_used','reference_metadata_read','quality_verified','ownership_verified','adoption')),'Original sealed exclusive export required')
    return raw,r


class OwnedDownload:
    def __init__(self,path,owner):self.path,self.owner=path,owner
    def __fspath__(self):return str(self.path)
    def open(self,mode):
        rt.require(mode=='xb'and not self.owner,'Exclusive fresh download file required')
        stream=self.path.open(mode);s=os.fstat(stream.fileno());self.owner.append((s.st_dev,s.st_ino,s.st_uid));return stream


def run(args,code,revision):
    started=time.monotonic();deadline=started+BUDGET;before=authenticate(code,revision)
    transport.verify_azure_peer(args.phase);out=ROOT/f'results/vcoco-observation-replica-{args.phase}-{revision}'
    rt.require(not out.exists()and not out.is_symlink(),'Fresh replica receipt required');out.mkdir(mode=0o700)
    s=out.lstat();owner=(s.st_dev,s.st_ino,s.st_uid);allowed=set();stage=None;archive_path=out/'archive.tar';archive_owner=[];original=None;manifest=None;replica_before=None
    report=dict(schema=SCHEMA,source_binding=before['source'],source_stat_identity=before['states'],phase=args.phase,status='fail',producer_revision=revision,original_producer_revision=REV,
        files=36,budget_seconds=BUDGET,models_loaded=False,GPU_used=False,reference_metadata_read=False,blob_cleanup_verified=False,
        quality_verified=False,ownership_verified=False,adoption=False,source_inputs_rehashed_after=False,outputs_sealed=False)
    def expired(*_):raise TimeoutError('Inclusive replica deadline')
    old={s:signal.signal(s,expired)for s in (signal.SIGALRM,signal.SIGTERM)};signal.alarm(max(1,math.ceil(deadline-time.monotonic())))
    try:
        check(deadline);export_revision=revision if args.phase=='export'else args.export_revision
        url='https://stworldrewardresearch26.blob.core.windows.net/runtime-transfers/articulated-runtime-'+export_revision+'.tar'
        blob=transport.Blob(url,export_revision,managed_identity=True)
        if args.phase=='export':
            manifest,paths,original=export_inputs(code,before['source']);manifest['export_revision']=revision
            report.update(export_archive(blob,manifest,paths,deadline))
            report['original_source_declaration']=manifest['original_source_declaration'];report['original_audit_declaration']=AUDIT
        else:
            raw,export=receipt_from_base64(args.export_receipt_base64,args.export_receipt_pin,args.export_revision)
            rt.require(export['archive_identity']==args.archive_pin and export['manifest_identity']==args.manifest_pin,'Independent exported archive/manifest pins differ')
            for name,data in (('export-receipt.json',raw),):rt.write(out/name,data);allowed.add(name)
            rt.require(not DEST.exists()and not DEST.is_symlink(),'Fresh no-overwrite replica required')
            transport.download(blob,OwnedDownload(archive_path,archive_owner),args.archive_pin);archive_path.chmod(0o400)
            manifest=verify_archive(archive_path,args.archive_pin,args.manifest_pin,args.export_revision,deadline)
            rt.write(out/'manifest.json',encode(manifest));allowed.add('manifest.json')
            stage=DEST.parent/(DEST.name+'.stage-'+revision);install(archive_path,stage,manifest,deadline)
            rt.require(rt.identity(archive_path,MAXIMUM)==args.archive_pin,'Archive changed during installation');replica_identity(manifest);replica_before=replica_state();report.update(replica_directory=str(DEST),archive_identity=args.archive_pin,
                manifest_identity=args.manifest_pin,export_receipt_identity=args.export_receipt_pin,blob_etag=export['blob_etag'],
                original_source_on_receiver_live_verified=False,original_audit_declaration=AUDIT)
        check(deadline);report['status']='pass'
    except BaseException as exc:report['error_type']=bank.error(exc)
    finally:
        try:
            rt.require(authenticate(code,revision)==before,'Current full source changed')
            if args.phase=='export':
                again,_,proof=export_inputs(code,before['source']);again['export_revision']=revision
                rt.require(original is not None and again==manifest and proof==original,'Original source/public/bank inputs changed or preflight absent')
            elif args.phase=='import'and replica_before is not None:
                replica_identity(manifest);rt.require(replica_state()==replica_before,'Replica inode/modes changed')
            report['source_inputs_rehashed_after']=True
        except BaseException as exc:report.update(status='fail',post_error_type=bank.error(exc))
        if archive_path.exists():
            try:
                rt.require(archive_owner and snapshot(archive_path)[:2]==archive_owner[0][:2],'Refuse foreign archive cleanup')
                rt.identity(archive_path,MAXIMUM,readonly=False,empty=True);archive_path.unlink()
            except BaseException as exc:report.update(status='fail',cleanup_error_type=bank.error(exc));allowed.add('archive.tar')
        report['archive_removed']=not archive_path.exists()
        def cleanup_blob():
            check(deadline);replica_identity(manifest);rt.require(replica_state()==replica_before,'Replica changed before blob cleanup')
            with blob.request('DELETE',headers={'If-Match':report['blob_etag']})as response:rt.require(response.status==202,'Owned exclusive blob cleanup failed')
        try:publish(out,report,deadline,started,owner,allowed,cleanup_blob if args.phase=='import'else None)
        finally:
            signal.alarm(0)
            for s,handler in old.items():signal.signal(s,handler)
    return report


def arguments(argv):
    p=argparse.ArgumentParser(description=__doc__,allow_abbrev=False);p.add_argument('phase',choices=('export','import'))
    p.add_argument('--export-revision');p.add_argument('--export-receipt-base64')
    for name in ('archive','manifest','export-receipt'):
        p.add_argument('--'+name+'-bytes',type=int);p.add_argument('--'+name+'-sha256')
    args=p.parse_args(argv)
    if args.phase=='export':rt.require(all(v is None for k,v in vars(args).items()if k!='phase'),'Export has no imported pins')
    else:
        rt.require(type(args.export_revision)is str and re.fullmatch('[0-9a-f]{40}',args.export_revision),'Pinned export revision required')
        for name in ('archive','manifest','export_receipt'):
            setattr(args,name+'_pin',fixed_pin(dict(bytes=getattr(args,name+'_bytes'),sha256=getattr(args,name+'_sha256')),
                MAXIMUM if name=='archive'else MAX_MANIFEST if name=='manifest'else 16 << 10))
        rt.require(type(args.export_receipt_base64)is str,'Original encoded export receipt required')
    return args


if __name__=='__main__':
    args=arguments(sys.argv[1:]);code=Path(os.environ.get('WR_CODE','/invalid'));revision=os.environ.get('WR_CODE_REVISION','')
    rt.require(os.geteuid()==0 and sys.platform=='linux'and os.environ.get('WR_ROOT')==str(ROOT),'Azure root-only replica')
    result=run(args,code,revision);print(encode({k:result[k]for k in ('phase','status','files','outputs_sealed')}).decode(),end='')
    raise SystemExit(0 if result['status']=='pass'else 1)
