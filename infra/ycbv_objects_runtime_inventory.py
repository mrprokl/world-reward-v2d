"""Azure CPU-only Objects prerequisites; no RGB, checkpoint load or CUDA.

The full runtime manifest stays Azure. SHA identities obtained here are measured
facts, not independent upstream fingerprints or model-training overlap proofs.
Raw historical image settings are never output; image.json is a safe projection.
"""
from __future__ import annotations
import hashlib
import json
import os
from pathlib import Path
import posixpath
import re
import signal
import stat
import subprocess
import sys
import time

import ycbv_point_objects as objects

ROOT=objects.ROOT;IMAGE=objects.IMAGE
ENTRY='run_ycbv_objects_runtime_inventory';STAGE='CPU_existing_Objects_runtime_inventory'
BUDGET=300;CLEANUP=30
HELPERS=('infra/ycbv_objects_runtime_inventory.py','infra/run_ycbv_objects_runtime_inventory.sh',*objects.HELPERS)
require=objects.require;canonical=objects.canonical;identity=objects.identity;safe=objects.safe;strict=objects.strict
STABLE_FIELDS=('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid')
def stable(value):return tuple(getattr(value,k)for k in STABLE_FIELDS)


def control(args,deadline):
    env={'PATH':'/usr/bin:/bin:/usr/sbin:/sbin','HOME':'/nonexistent','LANG':'C.UTF-8','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock'),
        'GIT_OPTIONAL_LOCKS':'0','GIT_NO_LAZY_FETCH':'1','GIT_TERMINAL_PROMPT':'0','GIT_CONFIG_NOSYSTEM':'1','GIT_CONFIG_GLOBAL':'/dev/null'}
    result=subprocess.run(args,env=env,capture_output=True,timeout=min(15,max(.01,deadline-time.monotonic())))
    require(result.returncode==0 and len(result.stdout)<=2_000_000,'Bounded readonly metadata command failed; output omitted')
    return result.stdout


def source(code,revision):
    canonical(code);require(code==ROOT/'jobs'/revision/ENTRY/'code'and re.fullmatch('[0-9a-f]{40}',revision),'Actual immutable inventory dispatch namespace required')
    digest=hashlib.sha256();rows={};markers={}
    for name in ('revision','source-sha256'):
        path=code.parent/name;value=identity(path);raw=path.read_bytes()
        require(raw==(revision+'\n').encode()if name=='revision'else bool(re.fullmatch(b'[0-9a-f]{64}\n',raw)),'Original inventory dispatch marker required');markers[name]=value;digest.update(raw)
    for path in (code,*sorted(code.rglob('*'))):
        canonical(path);mode=path.lstat().st_mode
        require(not mode&0o222 and(stat.S_ISREG(mode)or stat.S_ISDIR(mode)),'Complete readonly code-only dispatch closure required')
        if path.is_file():row=identity(path,True,True);rows[str(path.relative_to(code))]=row;digest.update(str(path.relative_to(code)).encode()+b'\0'+bytes.fromhex(row['sha256']))
    require(set(HELPERS)<=set(rows),'Complete genuine stdlib helper closure required')
    return dict(closure_sha256=digest.hexdigest(),helpers={n:rows[n]for n in HELPERS},markers=markers)


def image(deadline):
    row=strict(control(['docker','image','inspect',IMAGE,'--format','{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"RootFS":{{json .RootFS}}}'],deadline))
    require(type(row)is dict and set(row)=={'Id','Architecture','Os','RootFS'}and row['Id']==IMAGE and row['Architecture']=='amd64'and row['Os']=='linux','Original exact Objects image/platform required')
    rootfs=row['RootFS'];require(type(rootfs)is dict and set(rootfs)=={'Type','Layers'}and rootfs['Type']=='layers'and type(rootfs['Layers'])is list and rootfs['Layers']and
        all(type(n)is str and re.fullmatch('sha256:[0-9a-f]{64}',n)for n in rootfs['Layers']),'Actual full ordered layer identities required; no assumed layer count')
    return row


def receipt(root,name):
    path=root/name;pin=identity(path);require(pin['bytes']<=2_000_000,'Bounded original model/image receipt required')
    return objects.bound_json(path,pin),pin


def source_selection(name):
    safe(name);return name in ('hubconf.py','LICENSE','MODEL_CARD.md')or name.startswith('dinov2/')and name.endswith('.py')


def dino_source(root,deadline):
    folder=canonical(root/objects.DINO);git=['git','-c','safe.directory='+str(folder),'-C',str(folder)]
    require(control([*git,'rev-parse','HEAD'],deadline).decode().strip()==objects.DINO_REV,'Original DINO revision required')
    prefixes=['dinov2','hubconf.py','LICENSE','MODEL_CARD.md'];rows={}
    for raw in control([*git,'ls-tree','-r','-z',objects.DINO_REV,'--',*prefixes],deadline).split(b'\0'):
        if not raw:continue
        meta,name=raw.decode().split('\t',1);mode,kind,blob=meta.split()
        require(mode in ('100644','100755')and kind=='blob'and source_selection(name),'Only selected DINO Python/card source files allowed')
        path=folder/name;row=identity(path,empty=True);require(row['bytes']<=1_000_000,'Bounded public DINO source required')
        raw=path.read_bytes();require(hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==blob,'Original selected DINO Git blob differs')
        rows[objects.DINO+'/'+name]=row
    require({objects.DINO+'/'+n for n in ('hubconf.py','LICENSE','MODEL_CARD.md')}<=set(rows)and any(n.startswith(objects.DINO+'/dinov2/')for n in rows),'Complete DINO loader/source/cards required')
    require(not control([*git,'status','--porcelain','--untracked-files=all','--',*prefixes],deadline).strip(),'Original selected DINO source modified')
    return rows


def moge_graph(root):
    require({p.name for p in canonical(root/objects.MOGE/'snapshots').iterdir()}=={objects.MOGE_REV},'Native firstsnapshot must see only original MoGe1 revision')
    files={};links={}
    for leaf in ('model.pt','README.md'):
        name=objects.SNAPSHOT+'/'+leaf;seen=set()
        for _ in range(5):
            require(name not in seen,'MoGe1 graph cycle forbidden');seen.add(name);path=root/safe(name);canonical(path.parent)
            before=path.lstat()
            if stat.S_ISLNK(before.st_mode):
                target=os.readlink(path);require(target and not target.startswith('/')and '\\'not in target and '\0'not in target,'Only original relative MoGe1 links allowed')
                destination=posixpath.normpath(posixpath.join(posixpath.dirname(name),target));safe(destination)
                require(re.fullmatch(re.escape(objects.MOGE)+r'/blobs/[0-9a-f]{40,64}',destination)or
                    re.fullmatch(re.escape(objects.HF)+r'/blobs/[0-9a-f]{2}/[0-9a-f]{40,64}',destination),'MoGe1 graph escapes exact original blob roots')
                require(stable(before)==stable(path.lstat()),'MoGe1 link changed');links[name]=target;name=destination
            else:
                row=identity(path);require(row['bytes']<=1_256_823_446 if leaf=='model.pt'else row['bytes']<=4096,'Bounded original MoGe1 model/card required')
                if leaf=='model.pt':require(row==dict(bytes=1256823446,sha256='da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f'),'Independent primary MoGe1 model identity differs')
                files[name]=row;break
        else:raise ValueError('MoGe1 graph exceeds four original links')
    require({objects.SNAPSHOT+'/model.pt',objects.SNAPSHOT+'/README.md'}<=set(links),'Original two snapshot model/card links required by unchanged consumer')
    return files,links


def model_license(path):
    row=identity(path);require(row['bytes']<=100_000,'Bounded original model license notice required')
    raw=path.read_bytes();text=raw.decode('utf-8');require(text.strip()and 'license'in text.lower()and
        any(term in text.lower()for term in ('permission','grant','use','rights')),'Original readable licensing/permission terms required')
    return dict(identity=row,content_notice_verified=True,OSI_or_competition_eligibility_verified=False,
        source_model_license_equality_assumed=False)


def collect(root,deadline):
    rawimage,imagepin=receipt(root,'results/image-sam3d-runtime.json');actual=image(deadline)
    require(all(rawimage.get(k)==v for k,v in actual.items()),'Actual image disagrees with historical image receipt projection; raw settings omitted')
    acquisition,acqpin=receipt(root,'results/weights-acquisition.json');aux,auxpin=receipt(root,'results/auxiliary-assets.json')
    for repo,rev,key,path in (('facebook/sam-3d-objects',objects.OBJECT_REV,'path',objects.OBJECT),('Ruicheng/moge-vitl',objects.MOGE_REV,'cache_dir',objects.HF)):
        rows=[r for r in acquisition.get('assets',[])if r.get('repo_id')==repo]
        require(len(rows)==1 and rows[0].get('revision')==rev and rows[0].get(key)==str(root/path),'Original model acquisition path/revision required')
    require(aux.get('source_revisions',{}).get('dinov2')==objects.DINO_REV,'Original DINO acquisition revision required')
    models={}
    for names,suffix,pins in ((objects.CKPTS,'.ckpt',objects.CHECKPOINT_PINS),(objects.YAMLS,'.yaml',objects.YAML_PINS)):
        for name,(size,sha)in zip(names,pins):
            relative=objects.OBJECT+'/checkpoints/'+name+suffix;row=identity(root/relative)
            require(row==dict(bytes=size,sha256=sha),'Independent primary Objects checkpoint/config bytes differ');models[relative]=row
    for name in objects.REG4:
        rows=[r for r in aux.get('checkpoints',[])if r.get('filename')==name];relative=objects.WEIGHTS+'/torch_home/hub/checkpoints/'+name
        require(len(rows)==1 and rows[0].get('hash_source')=='first_observed_https_download'and
            rows[0].get('url')=='https://dl.fbaipublicfiles.com/dinov2/dinov2_'+('vitl14'if 'vitl14'in name else'vitb14')+'/'+name,'Genuine first-observed official DINO receipt required; no independent release claim')
        models[relative]=identity(root/relative);require(models[relative]=={k:rows[0].get(k)for k in ('bytes','sha256')},'Original DINO reg4 receipt/checkpoint differs')
    license=model_license(root/objects.OBJECT/'LICENSE');models[objects.OBJECT+'/LICENSE']=license['identity']
    moge,links=moge_graph(root);models.update(moge);sources=dino_source(root,deadline)
    runtime=dict(model_files=models,source_files=sources,moge_links=links,installed_sources={},
        acquisition_receipts={'results/weights-acquisition.json':acqpin,'results/auxiliary-assets.json':auxpin})
    return runtime,actual,dict(original_image_receipt=imagepin,model_license=license,original_raw_image_settings_exported=False,
        Objects_revision=objects.OBJECT_REV,DINO_revision=objects.DINO_REV,MoGe1_revision=objects.MOGE_REV,DINO_checkpoint_hash_source='first_observed_https_download')


# Runs with NO model/data/source mounts and NO GPU device request. Only native
# installed Python files are read. find_spec may import package parents; any
# dependency requiring an absent model fails, never receives model/GT mounts.
PROBE=r'''import hashlib,importlib.util,json,os,stat
from pathlib import Path
assert os.getuid()==0 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
result={};total=0
def stable(s):return tuple(getattr(s,k)for k in ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid'))
for name in ('sam3d_objects','v2d.common','v2d.sam3d.lib','moge'):
 spec=importlib.util.find_spec(name)
 assert spec and spec.submodule_search_locations and len(spec.submodule_search_locations)==1
 folder=Path(next(iter(spec.submodule_search_locations)))
 assert folder.is_absolute() and folder.resolve()==folder and not any(p.is_symlink()for p in(folder,*folder.parents))
 rows={}
 for p in sorted(folder.rglob('*.py')):
  s=p.lstat();assert stat.S_ISREG(s.st_mode) and s.st_nlink==1 and s.st_size<=2000000 and p.resolve()==p and not any(x.is_symlink()for x in(p,*p.parents))
  raw=p.read_bytes();assert stable(s)==stable(p.lstat());total+=len(raw);assert total<=32000000
  rows[str(p.relative_to(folder))]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 assert rows;result[name]=rows
print(json.dumps(result,separators=(',',':'),allow_nan=False))
'''


def probe_arguments(out,revision):
    return ['docker','run','--rm','--name','world-reward-ycbv-objects-runtime-'+revision[:12],'--cidfile',str(out/'.probe.cid'),
        '--label','world-reward.job='+ENTRY,'--label','world-reward.revision='+revision,'--network','none','--read-only','--user','0:0',
        '--cap-drop','ALL','--security-opt','no-new-privileges','--memory','4g','--cpus','2','--tmpfs','/tmp:rw,noexec,nosuid,size=64m',
        '--entrypoint','/usr/bin/env',IMAGE,'-i','PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','CUDA_VISIBLE_DEVICES=',
        'HF_HUB_OFFLINE=1','TRANSFORMERS_OFFLINE=1','PYTHONDONTWRITEBYTECODE=1','/opt/conda/bin/python','-I','-B','-c',PROBE]


def validate_installed(value):
    require(type(value)is dict and set(value)==set(objects.MODULES),'Exactly four original installed native packages required')
    for rows in value.values():
        require(type(rows)is dict and rows,'Complete nonempty installed source inventory required')
        for name,row in rows.items():safe(name);require(name.endswith('.py'),'Only Python installed source metadata allowed');objects.pin(row,True)
    return value


def exclusive(path,value):
    raw=(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode();require(len(raw)<=2_000_000,'Azure-only metadata bound exceeded')
    with path.open('xb')as stream:os.fchmod(stream.fileno(),0o400);stream.write(raw);stream.flush();os.fsync(stream.fileno())
    return identity(path,True)


def cleanup(out,revision):
    cidfile=out/'.probe.cid'
    if not cidfile.exists():return
    cid=canonical(cidfile).read_text().strip();require(re.fullmatch('[0-9a-f]{64}',cid),'Exact owned probe CID required')
    deadline=time.monotonic()+CLEANUP
    if control(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline).strip():
        actual=control(['docker','inspect',cid,'--format','{{.Image}}|{{.Name}}|{{index .Config.Labels "world-reward.job"}}|{{index .Config.Labels "world-reward.revision"}}'],deadline).decode().strip()
        require(actual==f'{IMAGE}|/world-reward-ycbv-objects-runtime-{revision[:12]}|{ENTRY}|{revision}','Only exact owned CPU probe cleanup permitted')
        control(['docker','rm','-f',cid],deadline)
        require(not control(['docker','ps','-aq','--no-trunc','--filter','id='+cid],deadline).strip(),'Owned probe cleanup incomplete')
    cidfile.chmod(0o400)


def main(argv=None):
    require(not argv,'No selectors, paths or override arguments allowed')
    code=Path(os.environ['WR_CODE']);rev=os.environ['WR_CODE_REVISION'];root=Path(os.environ['WR_ROOT'])
    require(root==ROOT and sys.platform=='linux'and os.geteuid()==0 and os.uname().nodename=='scenesmith-ncc-h100-01'and Path(__file__)==code/HELPERS[0],'Actual owned VM01 immutable CPU inventory required')
    require(Path(objects.__file__)==code/'infra/ycbv_point_objects.py','Actual genuine Objects helper must belong to immutable inventory closure')
    before=source(code,rev);out=canonical(root/'results'/('ycbv-objects-runtime-'+rev));require(out.parent.is_dir()and not out.exists(),'Fresh owned runtime inventory namespace required')
    out.mkdir(mode=0o700);started=time.monotonic();deadline=started+BUDGET;complete=None
    report=dict(stage=STAGE,status='fail',phase='model_provenance',producer_revision=rev,script_sha256=before['helpers'][HELPERS[0]]['sha256'],source_helpers=before,
        budget_seconds=BUDGET,cleanup_grace_seconds=CLEANUP,GPU_used=False,models_loaded=False,RGB_or_labels_read=False,source_commit_parity_verified=False,installed_source_commit_verified=False,
        training_overlap_verified=False,license_eligibility_verified=False,replica_ready=False,runtime_facts_independently_published=False)
    def expired(*unused):raise TimeoutError('Frozen300s CPU inventory deadline exceeded')
    old={s:signal.signal(s,expired)for s in(signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
    try:
        runtime,img,evidence=collect(root,deadline);initial=(runtime.copy(),img,evidence);require(not control(['docker','ps','-aq','--filter','name=^/world-reward-ycbv-objects-runtime-'+rev[:12]+'$'],deadline).strip(),'Owned CPU probe namespace occupied')
        report['phase']='installed_source_probe'
        env={'PATH':'/usr/bin:/bin','HOME':'/nonexistent','DOCKER_HOST':'unix://'+str(root/'docker.sock')}
        with (out/'probe.log').open('xb')as log:
            os.fchmod(log.fileno(),0o400)
            completed=subprocess.run(probe_arguments(out,rev),env=env,stdout=subprocess.PIPE,stderr=log,timeout=max(.01,deadline-time.monotonic()))
        require(completed.returncode==0 and len(completed.stdout)<=2_000_000,'Original source-only CPU probe failed; bounded original log preserved')
        runtime['installed_sources']=validate_installed(strict(completed.stdout));installedpin=exclusive(out/'installed-source.json',runtime['installed_sources'])
        imagepin=exclusive(out/'image.json',img);runtime['image_receipt']=dict(path=str((out/'image.json').relative_to(root)),**imagepin)
        require(collect(root,deadline)==initial,'Original selected model/source/image provenance changed after CPU probe')
        require(source(code,rev)==before,'Original complete source changed after inventory')
        report['all_selected_models_receipts_source_and_image_after_reverified']=True
        report.update(image_id=IMAGE,actual_rootfs_layers=len(img['RootFS']['Layers']),ordered_rootfs_sha256=hashlib.sha256(json.dumps(img['RootFS']['Layers'],separators=(',',':')).encode()).hexdigest(),
            selected_model_bytes=sum(r['bytes']for r in runtime['model_files'].values()),selected_source_files=len(runtime['source_files']),
            installed_source_files=sum(len(v)for v in runtime['installed_sources'].values()),installed_source_identity=installedpin,evidence=evidence)
        complete=runtime
    except BaseException as error:report.update(error_type=type(error).__name__,error='Frozen CPU inventory failed at recorded phase; paths/settings/private values omitted')
    finally:
        report['CPU_inventory_elapsed_seconds']=time.monotonic()-started
        if report['CPU_inventory_elapsed_seconds']>BUDGET:complete=None;report.update(error_type='TimeoutError',error='Frozen CPU inventory budget exceeded')
        signal.alarm(0)
        for s,h in old.items():signal.signal(s,h)
        try:cleanup(out,rev);report['source_rehashed_after']=source(code,rev)==before
        except BaseException:report.update(source_rehashed_after=False,owned_cleanup_verified=False)
        report['elapsed_seconds']=time.monotonic()-started
        if complete is not None and report.get('source_rehashed_after')and report['elapsed_seconds']<=BUDGET+CLEANUP:
            report['runtime_manifest']=exclusive(out/'runtime.json',complete);report.update(status='pass',phase='complete',owned_cleanup_verified=True)
        exclusive(out/'report.json',report)
    print(json.dumps({k:report.get(k)for k in ('stage','status','elapsed_seconds','runtime_manifest','selected_model_bytes','installed_source_files')}))
    if report['status']!='pass':raise SystemExit(1)


if __name__=='__main__':main(sys.argv[1:])
