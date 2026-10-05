"""One isolated author-version CPU import build; not a MASA/operator qualifier.

The ef12 image and its packages are never changed. Public wheels are hashed
before offline installation into a fresh venv without inherited site packages.
"""
import email
import hashlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import signal
import stat
import subprocess
import sys
import time
import urllib.parse
import urllib.request
import zipfile

ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_masa_runtime_build'
CONFIG = 'configs/masa_runtime_v1.json'
CONFIG_PIN = dict(bytes=55973,sha256='c9c33343295a1baf9e3d9369d1ce9ca94a72cf5e94b7e8f47fa59966207f01b5')
BASE = 'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
VENV = '/opt/world-reward-masa'
BUDGET, GRACE, BLOCK = 1800, 10, 1 << 20
HELPERS = ('infra/masa_runtime_build.py','infra/run_masa_runtime_build.sh',CONFIG,
           'infra/mediapipe_cpu_runtime_verify.py','infra/mediapipe_hands_acquire.py')
HELPER_PINS = {
 'infra/mediapipe_cpu_runtime_verify.py':dict(bytes=23559,sha256='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'),
 'infra/mediapipe_hands_acquire.py':dict(bytes=21381,sha256='8293510c02777cd5b844865285736ff1a653631a8803849a9047ee9030196570')}
SAFE_ENV = dict(PATH='/usr/bin:/bin:/usr/sbin:/sbin',HOME='/nonexistent',LANG='C.UTF-8',
                DOCKER_HOST='unix://'+str(ROOT/'docker.sock'),DOCKER_BUILDKIT='0')


def helpers(code):
    result=[]
    for name,pin in HELPER_PINS.items():
        p=code/name;s=p.lstat()
        if (p.resolve()!=p or any(q.is_symlink()for q in(p,*p.parents)) or not stat.S_ISREG(s.st_mode)
            or s.st_nlink!=1 or s.st_mode&0o222 or s.st_size!=pin['bytes']
            or hashlib.sha256(p.read_bytes()).hexdigest()!=pin['sha256']):
            raise ValueError('Original stdlib helper differs')
        a=p.lstat()
        if any(getattr(s,k)!=getattr(a,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')):
            raise ValueError('Original helper changed while hashing')
        spec=importlib.util.spec_from_file_location('wr_masa_build_'+p.stem,p)
        module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        if Path(module.__file__)!=p:raise ValueError('Imported helper origin differs')
        result.append(module)
    return tuple(result)


def binding(rt,code,revision):
    rt.require(Path(__file__).resolve()==code/HELPERS[0] and rt.canonical(ROOT)==ROOT
               and set(p.name for p in code.parent.iterdir())=={'code','revision','source-sha256'},'Exact current source snapshot required')
    proof=rt.source(ROOT,code,revision,ENTRY,HELPERS)
    rt.require(all(proof['helpers'][n]==pin for n,pin in HELPER_PINS.items()),'Pinned reused helpers differ')
    return proof


def config(rt,code):
    rt.require(rt.identity(code/CONFIG,100000)==CONFIG_PIN,'Frozen runtime config differs')
    c=rt.strict((code/CONFIG).read_bytes());rows=c['wheels']
    rt.require(c['schema']=='world_reward.masa_runtime.v1' and c['base_image']==BASE
      and c['venv']==VENV and c['system_site_packages'] is False and c['budget_seconds']==BUDGET
      and c['publication_grace_seconds']==GRACE and c['cpus']==4 and c['memory_bytes']==8<<30
      and c['minimum_free_bytes']==20<<30 and c['maximum_download_bytes']==3000000000
      and c['azure_vm_name']=='world-reward-ncc-h100-02'
      and [r['name']for r in rows]==sorted({r['name']for r in rows}), 'Fixed isolated author runtime policy required')
    for r in rows:
        rt.require(re.fullmatch(r'[A-Za-z0-9_.+-]+\.whl',r['filename']) and type(r['bytes']) is int
                   and 0<r['bytes']<=2500000000 and r['publication_date']<=c['publication_cutoff']
                   and (r['sha256'] is not None and re.fullmatch('[0-9a-f]{64}',r['sha256'])
                        or r['name']=='mmcv' and r['sha256'] is None and r['publisher_md5']=='65199300b098827d1ff8e6fabdb5082c'),
                   'Exact bounded publisher wheel pin required')
        endpoint(r['url']);rt.require(urllib.parse.unquote(r['url']).endswith('/'+r['filename']),'Original wheel filename required')
    rt.require(sum(r['bytes']for r in rows)+sum(r['bytes']for r in c['publisher_notices'])<=c['maximum_download_bytes'], 'Frozen download cap exceeded')
    return c


def endpoint(url):
    p=urllib.parse.urlsplit(url)
    if (p.scheme!='https' or p.username or p.password or p.query or p.fragment or p.port not in(None,443)
        or p.hostname not in('files.pythonhosted.org','download.pytorch.org','download.openmmlab.com','raw.githubusercontent.com')):
        raise ValueError('Unlisted unsigned public publisher')
    return url


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,*_):raise ValueError('Publisher redirect forbidden')


def fetch(rt,mp,row,path,opener,deadline,partials):
    rt.require(time.monotonic()<deadline,'Inclusive runtime deadline')
    req=urllib.request.Request(endpoint(row['url']),headers={'Accept-Encoding':'identity'})
    part=path.with_name(path.name+'.part');h=hashlib.sha256();md5=hashlib.md5();count=0
    with opener.open(req,timeout=min(30,max(.001,deadline-time.monotonic())))as response:
        rt.require(response.status==200 and response.geturl()==row['url']
                   and response.headers.get('Content-Encoding','identity').lower()=='identity'
                   and response.headers.get_content_type().lower()not in('text/html','application/xhtml+xml'),'Exact publisher transport required')
        length=response.headers.get('Content-Length')
        rt.require(length is None or re.fullmatch('[0-9]+',length)and int(length)==row['bytes'],'Publisher byte length differs')
        with part.open('xb')as f:
            os.fchmod(f.fileno(),0o600);s=part.lstat();partials.append((part,(s.st_dev,s.st_ino,s.st_uid)))
            while True:
                rt.require(time.monotonic()<deadline,'Inclusive runtime deadline')
                block=response.read(min(BLOCK,row['bytes']-count+1))
                rt.require(len(block)<=BLOCK and count+len(block)<=row['bytes'],'Publisher body overflow')
                if not block:break
                if not count:rt.require(not block.lstrip().lower().startswith((b'<html',b'<!doctype html')),'HTML body rejected')
                f.write(block);h.update(block);md5.update(block);count+=len(block)
            rt.require(count==row['bytes'] and (h.hexdigest()==row['sha256']if row.get('sha256') else md5.hexdigest()==row['publisher_md5']), 'Publisher content identity differs')
            f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
        mp.publish(part,path)
    return dict(bytes=count,sha256=h.hexdigest(),publisher_md5=md5.hexdigest()if row.get('publisher_md5')else None)


def wheel_record(rt,path,row):
    """Inventory and preserve notices; never extract or execute upstream ZIPs."""
    with zipfile.ZipFile(path)as z:
        members=z.infolist();names=[m.filename for m in members]
        rt.require(len(names)==len(set(names)) and len(names)<100000 and sum(m.file_size for m in members)<10_000_000_000,
                   'Bounded unique wheel inventory required')
        for m in members:
            p=PurePosixPath(m.filename);mode=m.external_attr>>16
            rt.require(not p.is_absolute() and '..'not in p.parts and '\\'not in m.filename and p.as_posix().rstrip('/')==m.filename.rstrip('/')
                       and (not stat.S_IFMT(mode) or stat.S_ISREG(mode)or stat.S_ISDIR(mode)) and not m.flag_bits&1,
                       'Canonical nonlink unencrypted wheel members required')
        metadata=[m for m in members if m.filename.endswith('.dist-info/METADATA')]
        rt.require(len(metadata)==1 and metadata[0].file_size<300000,'Single bounded wheel metadata required')
        raw=z.read(metadata[0]);m=email.message_from_bytes(raw);pin=row['metadata']
        rt.require(len(raw)==pin['bytes'] and hashlib.sha256(raw).hexdigest()==pin['sha256']
                   and m['Name'].lower().replace('_','-')==row['name'] and m['Version']==row['version'],'Original package metadata differs')
        notices={}
        for info in members:
            n=PurePosixPath(info.filename).name.upper()
            if not info.is_dir()and any(word in n for word in('LICENSE','LICENCE','COPYING','NOTICE','COPYRIGHT')):
                rt.require(info.file_size<=2_000_000,'Bounded embedded license/notice required')
                text=z.read(info);notices[info.filename]=dict(bytes=len(text),sha256=hashlib.sha256(text).hexdigest())
        inventory=[dict(file=m.filename,bytes=m.file_size,crc32=m.CRC)for m in members]
        return dict(metadata=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),embedded_notices=notices,
                    embedded_notices_present=bool(notices),members=len(members),expanded_bytes=sum(m.file_size for m in members),
                    inventory_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),license_eligibility_verified=False)


def command(args,deadline,*,log=None):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Inclusive runtime deadline')
    if log is None:
        result=subprocess.run(args,env=SAFE_ENV,capture_output=True,timeout=min(30,remaining))
        if len(result.stdout)+len(result.stderr)>100000:raise ValueError('Bounded Docker response required')
    else:
        with log.open('xb')as f:
            os.fchmod(f.fileno(),0o400)
            result=subprocess.run(args,env=SAFE_ENV,stdout=f,stderr=subprocess.STDOUT,timeout=remaining)
    return result


def image(rt,name,deadline,*,absent=False,owner=None):
    r=command(['docker','image','inspect',name,'--format','{{json .}}'],deadline)
    if absent:
        rt.require(r.returncode==1 and r.stdout.strip()in(b'',b'[]')and r.stderr.strip()in(tuple(
            (prefix+name).encode()for prefix in('Error: No such image: ',
                'error: no such image: ','Error response from daemon: No such image: '))),
            'Fresh owned image tag must be absent')
        return None
    rt.require(r.returncode==0,'Actual immutable image required');c=rt.strict(r.stdout)
    rt.require(c['Architecture']=='amd64' and c['Os']=='linux' and re.fullmatch('sha256:[0-9a-f]{64}',c['Id'])
               and c['RootFS']['Type']=='layers' and all(re.fullmatch('sha256:[0-9a-f]{64}',x)for x in c['RootFS']['Layers']), 'Linux layered image required')
    if owner is not None:rt.require((c['Config'].get('Labels')or{}).get('world_reward_masa_runtime_owner')==owner,'Exact owned image label required')
    return dict(image_id=c['Id'],layers=c['RootFS']['Layers'])


def namespace_absent(rt,name,deadline):
    r=command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline)
    rt.require(r.returncode==0 and not r.stdout.strip()and not r.stderr.strip(),'Fresh exact container namespace required')


BASE_PROBE = """import sys,venv,ensurepip,tempfile,os,json
assert sys.version_info[:2]==(3,11)
with tempfile.TemporaryDirectory() as d:
 venv.EnvBuilder(with_pip=True,system_site_packages=False).create(d+'/probe')
 assert 'include-system-site-packages = false' in open(d+'/probe/pyvenv.cfg').read().lower()
 assert os.path.isfile(d+'/probe/bin/python')
print(json.dumps({'python':'3.11','venv_with_pip':True,'system_site_packages':False}))
"""


def recipe(c):
    bootstrap=' '.join('/opt/masa-build/wheels/'+r['filename']for r in c['wheels']if r['name']in('pip','setuptools','wheel'))
    all_wheels=' '.join('/opt/masa-build/wheels/'+r['filename']for r in c['wheels'])
    pip=VENV+'/bin/python -I -m pip --isolated install --no-index --no-deps --no-cache-dir --disable-pip-version-check '
    return ('FROM '+BASE+'\nCOPY wheels /opt/masa-build/wheels\nCOPY notices /opt/masa-build/notices\n'
      'RUN /usr/bin/env -i PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin HOME=/tmp python -I -m venv '+VENV+'\n'
      'RUN '+pip+bootstrap+' && '+pip+all_wheels+' && '+VENV+'/bin/python -I -m pip check\n'
      'ENV VIRTUAL_ENV='+VENV+'\nENV PATH='+VENV+'/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin\n'
      'ENV PYTHONNOUSERSITE=1\nENV LD_LIBRARY_PATH=""\n'
      'RUN rm -rf /opt/masa-build/wheels\n').encode()


def probe(c):
    versions={r['name']:r['version']for r in c['wheels']}
    return ("import importlib.metadata as m,json,sys,torch,numpy,torchvision,mmcv,mmengine,mmdet,einops\n"
      "assert sys.prefix=="+repr(VENV)+" and sys.version_info[:2]==(3,11)\n"
      "assert 'include-system-site-packages = false' in open(sys.prefix+'/pyvenv.cfg').read().lower()\n"
      "assert not any('site-packages' in p and not p.startswith(sys.prefix+'/') for p in sys.path)\n"
      "expected="+repr(versions)+"\nassert {n:m.version(n)for n in expected}==expected\n"
      "from mmcv.ops import RoIAlign,ModulatedDeformConv2d\n"
      "from mmdet.models.trackers.base_tracker import BaseTracker\n"
      "from mmdet.models.tracking_heads import QuasiDenseTrackHead\n"
      "assert torch.version.cuda=='11.8' and not torch.cuda.is_initialized()\n"
      "print(json.dumps({'versions':expected,'python':'3.11','isolated_venv':True,'cuda_initialized':False,'cuda_build':'11.8','extension_imported':True,'operator_executed':False,'model_constructed':False}))\n")


def cpu(rt,image_id,out,label,script,deadline,*,isolated=False):
    name='wr-masa-runtime-'+label;cid=out/(label+'.cid');log=out/(label+'.log')
    namespace_absent(rt,name,deadline)
    r=command(['docker','run','--rm','--cidfile',str(cid),'--name',name,'--label','world_reward_masa_runtime_owner='+label,
       '--network','none','--read-only','--cap-drop','ALL','--security-opt','no-new-privileges','--memory','8g','--cpus','4',
       '--tmpfs','/tmp:rw,exec,nosuid,size=512m','--entrypoint','/usr/bin/env',image_id,'-i',
       'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','LD_LIBRARY_PATH=','CUDA_VISIBLE_DEVICES=',
       'PYTHONDONTWRITEBYTECODE=1','MPLBACKEND=Agg',VENV+'/bin/python'if isolated else 'python','-I','-B','-c',script],deadline,log=log)
    rt.require(r.returncode==0 and log.stat().st_size<100000,'Offline CPU import/probe failed')
    lines=log.read_bytes().splitlines();rt.require(bool(lines),'Native CPU receipt missing')
    result=rt.strict(lines[-1]);return result


def cleanup_container(rt,cid,name,label,deadline):
    if not cid.exists():return
    pin=rt.identity(cid,100,readonly=False);value=cid.read_text().strip()
    rt.require(re.fullmatch('[0-9a-f]{64}',value),'Exact owned container CID required')
    r=command(['docker','inspect','--type','container','--format','{{json .}}',value],deadline)
    if r.returncode==1:
        rt.require(r.stdout.strip()in(b'',b'[]')and r.stderr.strip()in(tuple(
          (prefix+value).encode()for prefix in('Error: No such object: ','error: no such object: ',
                                             'Error: No such container: ','error: no such container: ',
                                             'Error response from daemon: No such container: '))),'Exact absent owned container required')
    else:
        rt.require(r.returncode==0,'Owned container inspect failed');c=rt.strict(r.stdout)
        rt.require(c['Id']==value and c['Name']=='/'+name and(c['Config'].get('Labels')or{}).get('world_reward_masa_runtime_owner')==label,
                   'Never remove a foreign container')
        rt.require(command(['docker','rm','-f',value],deadline).returncode==0,'Owned container cleanup failed')
        r=command(['docker','inspect','--type','container','--format','{{json .}}',value],deadline)
        rt.require(r.returncode==1 and r.stdout.strip()in(b'',b'[]')and r.stderr.strip()in(tuple(
          (prefix+value).encode()for prefix in('Error: No such object: ','error: no such object: ',
                                             'Error: No such container: ','error: no such container: ',
                                             'Error response from daemon: No such container: '))),'Owned container survives cleanup')
    rt.require(rt.identity(cid,100,readonly=False)==pin,'Owned CID changed');cid.chmod(0o444)


def run(code,revision,*,opener=None,started=None):
    start=time.monotonic()if started is None else started;deadline=start+BUDGET
    if not isinstance(start,(int,float))or not 0<start<=time.monotonic()<deadline:
        raise ValueError('Original finite runtime start required')
    rt,mp=helpers(code);before=binding(rt,code,revision);c=config(rt,code)
    rt.require(time.monotonic()<deadline,'Original runtime deadline required')
    out=rt.canonical(ROOT/'results'/('masa-runtime-build-'+revision));target=c['target_prefix']+revision
    rt.require(out.parent.is_dir()and not out.exists(),'Fresh revision-scoped runtime result required');out.mkdir(mode=0o700)
    owner=(out.stat().st_dev,out.stat().st_ino,out.stat().st_uid);partials=[];assets={};asset_states={};base=None;child=None;built=False
    report=dict(schema='world_reward.masa_runtime_build.v1',stage='masa_author_isolated_runtime_build',status='fail',phase='preflight',
      producer_revision=revision,source_binding=before,config_identity=CONFIG_PIN,budget_seconds=BUDGET,publication_grace_seconds=GRACE,
      gpu_used=False,operator_qualified=False,model_constructed=False,model_or_dataset_read=False,quality_verified=False,adopted=False,
      license_eligibility_verified=False,source_rehashed_after=False,artifacts_rehashed_after=False,base_rechecked_after=False,
      owned_containers_removed=False,owned_partial_cleanup=False,system_site_packages=False)
    try:
        rt.require(shutil.disk_usage(out).free>=c['minimum_free_bytes'],'Minimum free disk unavailable')
        base=image(rt,BASE,deadline);rt.require(base['image_id']==BASE,'Original selected base required');image(rt,target,deadline,absent=True)
        report['base_probe']=cpu(rt,BASE,out,revision+'-base',BASE_PROBE,deadline)
        rt.require(report['base_probe']==dict(python='3.11',venv_with_pip=True,system_site_packages=False),'Base Python/venv ABI unverified')
        wheels=out/'wheels';notices=out/'notices';wheels.mkdir(mode=0o700);notices.mkdir(mode=0o700)
        opener=opener or urllib.request.build_opener(urllib.request.ProxyHandler({}),NoRedirect())
        report['phase']='public_wheels';report['publisher_notices']={};report['wheels']={}
        for row in c['publisher_notices']:
            path=notices/row['file'];pin=fetch(rt,mp,row,path,opener,deadline,partials);assets[path]=pin
            s=path.stat();asset_states[path]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
            report['publisher_notices'][row['file']]=pin
        for row in c['wheels']:
            path=wheels/row['filename'];pin=fetch(rt,mp,row,path,opener,deadline,partials);assets[path]=pin
            s=path.stat();asset_states[path]=(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)
            report['wheels'][row['name']]=dict(identity=pin,hash_basis=row['hash_basis'],inventory=wheel_record(rt,path,row))
        report['phase']='offline_build';rt.write(out/'Dockerfile',recipe(c));rt.write(out/'.dockerignore',b'*\n!Dockerfile\n!wheels/\n!wheels/**\n!notices/\n!notices/**\n')
        rt.require(binding(rt,code,revision)==before,'Source changed before isolated build');image(rt,target,deadline,absent=True)
        built=True
        r=command(['docker','build','--network','none','--memory','8g','--cpu-period','100000','--cpu-quota','400000',
          '--label','world_reward_masa_runtime_owner='+revision,'--tag',target,'--file',str(out/'Dockerfile'),str(out)],deadline,log=out/'build.log')
        rt.require(r.returncode==0,'Offline isolated build failed')
        child=image(rt,target,deadline,owner=revision)
        rt.require(child['image_id']!=BASE and child['layers'][:len(base['layers'])]==base['layers'],'Child must preserve original base layers')
        report['phase']='cpu_import';report['cpu_import']=cpu(rt,child['image_id'],out,revision+'-child',probe(c),deadline,isolated=True)
        expected={r['name']:r['version']for r in c['wheels']}
        rt.require(report['cpu_import']==dict(versions=expected,python='3.11',isolated_venv=True,cuda_initialized=False,cuda_build='11.8',
           extension_imported=True,operator_executed=False,model_constructed=False),'Exact isolated import evidence required')
        report.update(status='pass',phase='complete',child_image=child,target_tag=target)
    except BaseException:
        report['error']='bounded_runtime_gate_failed'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        cleanup_deadline=deadline+GRACE
        try:
            rt.require(out.resolve()==out and(out.stat().st_dev,out.stat().st_ino,out.stat().st_uid)==owner,'Owned result namespace changed')
            for label in (revision+'-base',revision+'-child'):
                cleanup_container(rt,out/(label+'.cid'),'wr-masa-runtime-'+label,label,cleanup_deadline)
                namespace_absent(rt,'wr-masa-runtime-'+label,cleanup_deadline)
            report['owned_containers_removed']=True
        except BaseException:report.update(status='fail',error='owned_cleanup_failed')
        try:
            rt.require(binding(rt,code,revision)==before,'Current source changed');report['source_rehashed_after']=True
            if base is not None:rt.require(image(rt,BASE,cleanup_deadline)==base,'Base image changed');report['base_rechecked_after']=True
            for p,pin in assets.items():
                s=p.stat();rt.require((s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_mode)==asset_states[p]
                  and rt.identity(p,2500000000)=={k:pin[k]for k in('bytes','sha256')},'Public wheel/notice changed')
            report['artifacts_rehashed_after']=True
            if report['status']=='pass':rt.require(image(rt,target,cleanup_deadline,owner=revision)==child,'Owned child changed')
        except BaseException:report.update(status='fail',error='source_artifact_or_image_postcheck_failed')
        try:
            for p,key in partials:
                if not p.exists()and not p.is_symlink():continue
                s=p.lstat();rt.require(rt.canonical(p)==p and stat.S_ISREG(s.st_mode)and s.st_nlink==1
                    and(s.st_dev,s.st_ino,s.st_uid)==key,'Owned partial replaced');p.unlink()
            report['owned_partial_cleanup']=True
            if report['status']!='pass'and built:
                try:
                    owned=image(rt,target,cleanup_deadline,owner=revision)
                    rt.require(owned['image_id']!=BASE,'Never remove original base')
                    rt.require(command(['docker','image','rm',target],cleanup_deadline).returncode==0,'Owned child cleanup failed')
                    report['owned_failed_image_removed']=True
                except BaseException:report['owned_failed_image_removed']=False
        except BaseException:report.update(status='fail',error='owned_partial_cleanup_failed')
        try:
            expected={out/'Dockerfile',out/'.dockerignore',out/'build.log',out/'wheels',out/'notices'}|set(assets)
            expected|={out/(revision+s+ext)for s in('-base','-child')for ext in('.cid','.log')}
            for p in out.rglob('*'):
                s=p.lstat();rt.require(rt.canonical(p)==p and s.st_uid==owner[2] and p in expected
                    and(stat.S_ISDIR(s.st_mode)or stat.S_ISREG(s.st_mode)and s.st_nlink==1),'Owned output inventory changed')
            for p in out.rglob('*'):
                if p.is_file():p.chmod(0o444)
            for p in sorted((x for x in out.rglob('*')if x.is_dir()),reverse=True):p.chmod(0o555)
        except BaseException:report.update(status='fail',error='owned_output_sealing_failed')
        report['elapsed_seconds']=time.monotonic()-start
        if report['elapsed_seconds']>=BUDGET:report.update(status='fail',error='inclusive_runtime_deadline_exceeded')
        # Keep the original owned FD through publication; a late completion is
        # demoted on that SAME inode, never replaced or advertised as PASS.
        report_path=out/'report.json';fd=os.open(report_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(fd,'wb',closefd=True)as stream:
            raw=(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode();stream.write(raw);stream.flush();os.fsync(stream.fileno())
            os.fchmod(stream.fileno(),0o444);out.chmod(0o555)
            elapsed=time.monotonic()-start
            if elapsed>=BUDGET and report['status']=='pass':
                report.update(status='fail',elapsed_seconds=elapsed,error='inclusive_runtime_deadline_exceeded')
                stream.seek(0);stream.truncate();stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
                stream.flush();os.fsync(stream.fileno())
    if report['status']!='pass':raise ValueError('Isolated runtime failed; immutable bounded receipt retained')from None
    return report


def main():
    if (len(sys.argv)!=1 or sys.platform!='linux' or os.geteuid()!=0 or os.environ.get('WR_ROOT')!=str(ROOT)):
        raise ValueError('Actual owned Azure builder required')
    start=float(os.environ['WR_BUILD_STARTED']);remaining=start+BUDGET-time.monotonic()
    if not 0<remaining<=BUDGET:raise ValueError('Original runtime deadline required')
    cancelled=lambda *_:(_ for _ in()).throw(TimeoutError('Builder cancellation'))
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGALRM,cancelled);signal.setitimer(signal.ITIMER_REAL,remaining)
    result=run(Path(os.environ['WR_CODE']),os.environ['WR_CODE_REVISION'],started=start)
    print(json.dumps(dict(stage=result['stage'],status=result['status'],elapsed_seconds=result['elapsed_seconds'],gpu_used=False)))


if __name__=='__main__':
    try:main()
    except BaseException:print('MASA isolated runtime failed',file=sys.stderr);raise SystemExit(1)from None
