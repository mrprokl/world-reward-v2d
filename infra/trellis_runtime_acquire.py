"""Azure source/checkpoint acquisition only; never import upstream code or models.

MIT publisher declarations are not training-overlap or complete-stack clearance.
Archives are checked against every frozen Git blob; only code/notices retained.
"""
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import pwd
import re
import shutil
import signal
import stat
import sys
import tarfile
import time
import urllib.parse
import urllib.request

ROOT=Path('/srv/scenesmith/world-reward')
DATA=Path('/srv/world-reward-data/trellis_v1')
JOB='run_trellis_runtime_acquire'
PROTOCOL='configs/trellis_runtime_protocol_v1.json'
PROTOCOL_PIN={'bytes':53755,'sha256':'9da06336d02220211d10bf6b39039f3522b6c62f3f4e973dc0366100d9c7e462'}
HELPERS=('infra/trellis_runtime_acquire.py','infra/run_trellis_runtime_acquire.sh',PROTOCOL,
         'infra/mediapipe_cpu_runtime_verify.py','infra/mediapipe_hands_acquire.py')
BLOCK=1<<20


def helpers(code):
    loaded=[]
    for name in HELPERS[-2:]:
        path=Path(code)/name;s=path.lstat()
        if path.resolve()!=path or any(p.is_symlink()for p in(path,*path.parents)) or not stat.S_ISREG(s.st_mode) or s.st_nlink!=1 or s.st_mode&0o222 or not 0<s.st_size<=2_000_000:
            raise ValueError('Readonly original helper required')
        spec=importlib.util.spec_from_file_location('wr_trellis_'+path.stem,path)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if Path(module.__file__)!=path:raise ValueError('Imported helper origin differs')
        loaded.append(module)
    return tuple(loaded)


def binding(rt,root,code,revision):
    rt.require(Path(__file__).resolve()==code/HELPERS[0],'Actual immutable acquisition source required')
    rt.require({p.name for p in code.parent.iterdir()}=={'code','revision','source-sha256'},'Exact source snapshot parent required')
    return rt.source(root,code,revision,JOB,HELPERS)


def relative(name):
    p=PurePosixPath(name)
    if not name or p.is_absolute() or '..'in p.parts or p.as_posix()!=name or '\\'in name or any(ord(c)<32 for c in name):
        raise ValueError('Strict relative artifact path required')
    return p


def manifest(rt,code):
    value=rt.pinned(code/PROTOCOL,PROTOCOL_PIN,100000)
    rt.require(value['schema']=='world_reward.trellis_runtime_protocol.v1' and value['stage']=='source_model_acquisition_only'
               and value['data_root']==str(DATA) and value['report']=='results/trellis-runtime-acquire-v1.json'
               and value['budget_seconds']==3600 and value['outer_seconds']==3630 and value['minimum_free_bytes']==30_000_000_000
               and value['maximum_total_bytes']==4_000_000_000 and value['model_count']==6 and value['model_bytes']==3_006_922_800,
               'Frozen acquisition scope differs')
    files=[]
    for row in value['assets']:
        p=relative(row['file']);files.append(row['file'])
        rt.require(p.parts[0]in('source','weights') and type(row['bytes'])is int and 0<row['bytes']<=2_000_000_000
                   and re.fullmatch('[0-9a-f]{64}',row['sha256']),'Exact bounded asset pin required')
        endpoint(row['url'])
    rt.require(len(files)==len(set(files)) and sum(r['bytes']for r in value['assets'])+sum(r['maximum_archive_bytes']for r in value['sources'])<=value['maximum_total_bytes'], 'Exact total asset bound required')
    for row in value['sources']:
        relative(row['name']);relative(row['archive_prefix']);endpoint(row['url'])
        rows=row['members'];names=[r['path']for r in rows]
        rt.require(len(names)==len(set(names)) and set(row['retained_files'])<=set(names)
                   and 0<sum(r['bytes']for r in rows)<=row['maximum_expanded_bytes']<=50_000_000
                   and 0<row['maximum_archive_bytes']<=80_000_000,'Frozen source inventory differs')
        for member in rows:
            relative(member['path']);rt.require(type(member['bytes'])is int and member['bytes']>=0 and re.fullmatch('[0-9a-f]{40}',member['git_blob_sha1']),'Exact Git blob pin required')
    return value


def endpoint(url):
    p=urllib.parse.urlsplit(url);h=p.hostname or ''
    if not(p.scheme=='https' and not p.username and not p.password and p.port in(None,443) and not p.fragment
           and (h in('huggingface.co','raw.githubusercontent.com','codeload.github.com')or h.endswith(('.hf.co','.huggingface.co')))):
        raise ValueError('Public closed HTTPS allowlist required')
    return url


class Redirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        endpoint(newurl)
        if any(k.lower()in('authorization','cookie')for k in req.headers):raise ValueError('Authenticated acquisition forbidden')
        return super().redirect_request(req,fp,code,msg,headers,newurl)


def check(rt,deadline):rt.require(time.monotonic()<deadline,'Inclusive acquisition deadline')


def download(rt,mp,data,row,opener,deadline,owned,records,*,archive=False):
    check(rt,deadline);target=rt.canonical(data/row['file']);target.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
    part=target.with_name(target.name+'.part');maximum=row['maximum_archive_bytes']if archive else row['bytes']
    request=urllib.request.Request(endpoint(row['url']),headers={'Accept-Encoding':'identity'})
    with opener.open(request,timeout=min(30,max(.001,deadline-time.monotonic())))as response:
        endpoint(response.geturl());rt.require(response.status==200 and response.headers.get('Content-Encoding','identity').lower()=='identity'
            and response.headers.get_content_type().lower()!='text/html','Publisher transport differs')
        if urllib.parse.urlsplit(row['url']).hostname in('raw.githubusercontent.com','codeload.github.com'):
            rt.require(response.geturl()==row['url'],'Immutable source redirect rejected')
        length=response.headers.get('Content-Length')
        rt.require(length is None or re.fullmatch('[0-9]+',length)and(0<int(length)<=maximum if archive else int(length)==maximum),'Publisher length differs')
        sha=hashlib.sha256();count=0
        with part.open('xb')as stream:
            os.fchmod(stream.fileno(),0o600);s=part.lstat();owned.append((part,(s.st_dev,s.st_ino,s.st_uid)))
            while True:
                check(rt,deadline);block=response.read(min(BLOCK,maximum-count+1))
                rt.require(len(block)<=BLOCK and count+len(block)<=maximum,'Publisher byte overflow')
                if not block:break
                if count==0:rt.require(not block.lstrip().lower().startswith((b'<html',b'<!doctype html'))and not block.startswith(b'version https://git-lfs.github.com/spec/'),'HTML/LFS pointer rejected')
                stream.write(block);sha.update(block);count+=len(block)
            rt.require(count>0 and(archive or count==row['bytes']and sha.hexdigest()==row['sha256']),'Publisher exact byte/SHA differs')
            stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
        check(rt,deadline);mp.publish(part,target)
        record=dict(file=row['file'],bytes=count,sha256=sha.hexdigest(),hash_basis='measured_archive_sha256'if archive else row['hash_basis'])
        records.append(record)
    return target,record


def publish_bytes(rt,mp,target,raw,owned):
    part=target.with_name(target.name+'.part')
    with part.open('xb')as stream:
        os.fchmod(stream.fileno(),0o600);s=part.lstat();owned.append((part,(s.st_dev,s.st_ino,s.st_uid)))
        stream.write(raw);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),0o444)
    mp.publish(part,target)


def archive_inventory(rt,mp,data,path,row,deadline,records,owned):
    expected={r['path']:r for r in row['members']};retained=set(row['retained_files']);seen=set();total=0
    prefix=row['archive_prefix'];directories={prefix}
    for submodule in row.get('submodules',[]):
        relative(submodule['path']);directories.add(prefix+'/'+submodule['path'])
    for name in expected:
        directories.update(prefix+'/'+str(p)for p in PurePosixPath(name).parents if str(p)!='.')
    with tarfile.open(path,'r:gz')as archive:
        for member in archive:
            check(rt,deadline);relative(member.name.rstrip('/')if member.isdir()else member.name)
            rt.require(not(member.issym()or member.islnk())and(member.isdir()or member.isfile()),'Source links/special entries rejected')
            if member.isdir():rt.require(member.name.rstrip('/')in directories,'Foreign archive directory');continue
            rt.require(member.name.startswith(prefix+'/'),'Original archive prefix required')
            name=member.name[len(prefix)+1:];rt.require(name in expected and name not in seen,'Foreign/duplicate archive file');seen.add(name)
            wanted=expected[name];total+=member.size
            rt.require(member.size==wanted['bytes']and total<=row['maximum_expanded_bytes'],'Exact expanded size differs')
            digest=hashlib.sha1(('blob '+str(member.size)+'\0').encode());sha=hashlib.sha256();raw=bytearray();count=0
            with archive.extractfile(member)as stream:
                while True:
                    check(rt,deadline);block=stream.read(BLOCK)
                    if not block:break
                    count+=len(block);digest.update(block);sha.update(block)
                    if name in retained:raw.extend(block)
            rt.require(count==wanted['bytes']and digest.hexdigest()==wanted['git_blob_sha1'],'Pinned source Git blob differs')
            if name in retained:
                target=rt.canonical(data/'source'/row['name']/name);target.parent.mkdir(parents=True,mode=0o700,exist_ok=True)
                publish_bytes(rt,mp,target,raw,owned);records.append(dict(file=str(target.relative_to(data)),bytes=count,sha256=sha.hexdigest(),hash_basis='pinned_git_blob_sha1_and_measured_sha256'))
    rt.require(seen==set(expected),'Complete original archive inventory required')
    return dict(repository=row['repository'],revision=row['revision'],members=len(seen),expanded_bytes=total,
                retained_files=len(retained),all_git_blobs_verified=True,submodules_fetched=False,upstream_code_executed=False)


def declarations(rt,mp,data,c,records,owned):
    rt.require(b'MIT License'in(data/'source/trellis/LICENSE').read_bytes()and
        b'TRELLIS models and the majority of the code are licensed under the [MIT License](LICENSE)'in(data/'source/trellis/README.md').read_bytes()
        and b'license: mit'in(data/'weights/README.md').read_bytes()and
        b'Apache License'in(data/'source/flexicubes/LICENSE.txt').read_bytes()and
        b'Apache License'in(data/'source/kaolin_apache/LICENSE').read_bytes(),'Exact first-party license grants required before weights')
    pipeline=rt.strict((data/'weights/pipeline.json').read_bytes());models=pipeline['args']['models']
    names={p['file'].removeprefix('weights/').removesuffix('.safetensors')for p in c['assets']if p['file'].endswith('.safetensors')}
    rt.require(pipeline['name']=='TrellisImageTo3DPipeline'and len(models)==6 and set(models.values())==names
               and pipeline['args']['image_cond_model']=='dinov2_vitl14_reg','Original six-model factory/DINO contract differs')
    for name in names:
        value=rt.strict((data/'weights'/ (name+'.json')).read_bytes());rt.require(type(value.get('name'))is str and type(value.get('args'))is dict,'Native checkpoint configuration required')
    spec=c['kaolin_checker'];source=(data/spec['source']).read_text();node=next((n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)and n.name==spec['function']),None)
    rt.require(node is not None,'Original checker definition missing');segment=(ast.get_source_segment(source,node)+'\n').encode()
    rt.require(len(segment)==spec['exact_function_bytes']and hashlib.sha256(segment).hexdigest()==spec['exact_function_sha256'],'Exact Apache checker AST source differs')
    raw=('\n'.join(source.splitlines()[:14])+'\n# Exact isolated check_tensor; no Kaolin package import.\nimport torch\n\n').encode()+segment
    rt.require(len(raw)==spec['generated_bytes']and hashlib.sha256(raw).hexdigest()==spec['generated_sha256'],'Exact generated checker differs')
    target=rt.canonical(data/spec['output']);publish_bytes(rt,mp,target,raw,owned)
    records.append(dict(file=spec['output'],bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),hash_basis='exact_unchanged_Apache_function_with_original_notice'))


def acquire(root,code,revision,*,opener=None,namespace_lease=None):
    started=time.monotonic();rt,mp=helpers(code);root=rt.canonical(root);code=rt.canonical(code);before=binding(rt,root,code,revision);c=manifest(rt,code)
    data=rt.canonical(Path(c['data_root']));out=rt.canonical(root/c['report']);deadline=started+c['budget_seconds']
    rt.require(data.parent.is_dir()and out.parent.is_dir()and not out.exists()and(namespace_lease is not None or not data.exists()),'Fresh fixed acquisition only; no resume')
    rt.require(shutil.disk_usage(data.parent).free>=c['minimum_free_bytes'],'At least 30GB free required before transfer')
    owned=[];records=[];created=[];failure=None;archives=[]
    report=dict(schema='world_reward.trellis_runtime_acquisition.v1',stage='trellis_source_model_acquisition',status='fail',phase='preflight',
        producer_revision=revision,source_binding=before,protocol_identity=PROTOCOL_PIN,artifacts=records,source_archives=archives,
        budget_seconds=c['budget_seconds'],budget_scope='downloads_archive_blob_checks_license_checks_public_sealing_source_artifact_posthash',
        receipt_publication_outer_seconds=c['outer_seconds'],scope=c['scope'],future_runtime=c['future_runtime'],
        first_party_license_grants_verified=False,source_rehashed_after=False,artifacts_rehashed_after=False,owned_partials_removed=False,owned_archives_removed=False)
    try:
        if namespace_lease is not None:created=mp.validate_namespace_lease(namespace_lease,[data],before['closure_sha256'])
        else:data.mkdir(mode=0o700);data.chmod(0o700);s=data.lstat();created=[(data,(s.st_dev,s.st_ino))]
        opener=opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirect())
        for source in c['sources']:
            report['phase']='source_'+source['name'];row=dict(source,file='.archives/'+source['name']+'.tar.gz')
            path,pin=download(rt,mp,data,row,opener,deadline,owned,records,archive=True)
            archives.append(dict(archive_identity={k:pin[k]for k in('bytes','sha256')},**archive_inventory(rt,mp,data,path,source,deadline,records,owned)))
            rt.require(rt.identity(path,source['maximum_archive_bytes'])=={k:pin[k]for k in('bytes','sha256')},'Owned archive changed');path.unlink();records.remove(pin)
        for row in c['assets']:
            if row['file'].endswith('.safetensors')and not report['first_party_license_grants_verified']:
                report['phase']='license_and_source_preflight';declarations(rt,mp,data,c,records,owned);report['first_party_license_grants_verified']=True
            report['phase']='model_download'if row['file'].endswith('.safetensors')else'text_download'
            download(rt,mp,data,row,opener,deadline,owned,records)
        report['phase']='complete';report['status']='pass'
    except Exception as exc:failure=exc;report['error_type']=type(exc).__name__
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            for path,inode in owned:
                if path.exists()or path.is_symlink():
                    s=path.lstat();rt.require(not path.is_symlink()and(s.st_dev,s.st_ino,s.st_uid)==inode and s.st_nlink==1,'Owned partial replaced');path.unlink()
            report['owned_partials_removed']=all(not p.exists()and not p.is_symlink()for p,_ in owned)
            for pin in list(records):
                if pin['file'].startswith('.archives/'):
                    p=data/pin['file'];rt.require(rt.identity(p,80_000_000)=={k:pin[k]for k in('bytes','sha256')},'Owned archive replacement');p.unlink();records.remove(pin)
            report['owned_archives_removed']=not tuple((data/'.archives').glob('*'))if(data/'.archives').exists()else True
            for p,inode in created:
                s=p.lstat();rt.require(rt.canonical(p)==p and(s.st_dev,s.st_ino)==inode and s.st_uid==os.getuid(),'Owned namespace replaced')
                expected={data/r['file']for r in records};prospective={data/r['file']for r in c['assets']}|{data/'source'/r['name']/n for r in c['sources']for n in r['retained_files']}
                expected_dirs={q for p in prospective for q in p.parents if q==data or q.is_relative_to(data)}|{data/'.archives'}
                entries=sorted(data.rglob('*'),reverse=True)
                for q in entries:
                    rt.canonical(q);t=q.lstat();rt.require(t.st_uid==os.getuid()and(q in expected_dirs if stat.S_ISDIR(t.st_mode)else stat.S_ISREG(t.st_mode)and t.st_nlink==1 and q in expected),'Foreign namespace entry')
                for q in entries:q.chmod(0o555 if q.is_dir()else 0o444)
                data.chmod(0o555)
            report['artifacts_rehashed_after']=all(rt.identity(data/r['file'],2_000_000_000,empty=r['bytes']==0)=={k:r[k]for k in('bytes','sha256')}for r in records)
        except Exception as exc:failure=failure or exc;report['post_error_type']=type(exc).__name__
        try:report['source_rehashed_after']=binding(rt,root,code,revision)==before
        except Exception as exc:failure=failure or exc;report['source_post_error_type']=type(exc).__name__
        report['elapsed_seconds']=time.monotonic()-started
        if failure or report['elapsed_seconds']>c['budget_seconds']or not all(report[k]for k in('first_party_license_grants_verified','source_rehashed_after','artifacts_rehashed_after','owned_partials_removed','owned_archives_removed')):report['status']='fail'
        rt.write(out,(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
    if report['status']!='pass':raise ValueError('Acquisition failed; immutable receipt retained')from None
    return report


def cancelled(*_):raise TimeoutError('Acquisition cancellation')


def main():
    if len(sys.argv)!=1 or sys.platform!='linux' or os.geteuid()!=1000 or os.uname().nodename!='world-reward-ncc-h100-02' or os.environ.get('WR_ROOT')!=str(ROOT):raise ValueError('Exact Azure VM02 unprivileged acquisition required')
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    rt,_=helpers(code);c=manifest(rt,code)
    signal.signal(signal.SIGALRM,cancelled);signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGINT,cancelled);signal.setitimer(signal.ITIMER_REAL,c['budget_seconds'])
    try:
        lease=rt.strict(os.environ['WR_NAMESPACE_LEASE'])if'WR_NAMESPACE_LEASE'in os.environ else None
        value=acquire(ROOT,code,revision,namespace_lease=lease);print(json.dumps({k:value[k]for k in('stage','status','elapsed_seconds')}))
    finally:signal.setitimer(signal.ITIMER_REAL,0)


if __name__=='__main__':
    try:main()
    except Exception as exc:print(json.dumps(dict(stage='trellis_source_model_acquisition',status='fail',error_type=type(exc).__name__)));raise SystemExit(1)from None
