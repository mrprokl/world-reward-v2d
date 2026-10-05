"""Fresh data-free MASA/CUDA contract only; no tracking/identity/quality claim."""
import argparse
import fcntl
import hashlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import signal
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
import zipfile
import time

ROOT=Path('/srv/scenesmith/world-reward')
DATA=Path('/srv/world-reward-data/masa_native_v1')
ENTRY='run_masa_native_qualify'
CONFIG='configs/masa_native_qualification_v1.json'
CONFIG_PIN=dict(bytes=3645,sha256='ad5e4a3cb78926512ab655a65ca3285c105f5542f68895ec014ddeaca5f31d2c')
SM90_CONFIG='configs/masa_sm90_build_v1.json'
SM90_CONFIG_PIN=dict(bytes=20926,sha256='fdc0134a5037bf446f8f7c755c4252749b4ffba492dda93a43932d14674f252b')
CUOBJDUMP_PIN=dict(bytes=162092,sha256='28218273db8ffeb3ae4b31bfb4e4d90f0ae3373454c7970703c063dfd0377ba7')
VENV='/opt/world-reward-masa'
BUDGET,GRACE=600,30
HELPERS=('infra/masa_native_qualify.py','infra/run_masa_native_qualify.sh',CONFIG,
 'infra/mediapipe_cpu_runtime_verify.py','infra/masa_acquire.py','configs/masa_acquisition_v1.json',
 'infra/masa_runtime_build.py','configs/masa_runtime_v1.json','infra/mediapipe_hands_acquire.py',SM90_CONFIG)
ENV=dict(PATH='/usr/bin:/bin',HOME='/nonexistent',LANG='C.UTF-8',DOCKER_HOST='unix://'+str(ROOT/'docker.sock'))
FAILURE_GATES=frozenset(('bootstrap','runtime_import','operators','registry','config','model_construct','weights_decode',
 'strict_state','model_device','preprocess','encoder','embedding','native_posthash','native_deadline','gpu_lock',
 'embedded_architecture','gpu_idle','container_absence','container_dispatch','native_receipt','host_cleanup','host_posthash','output_seal','host_deadline'))
OPERATOR_SUBGATES=frozenset(('imports','dcn_construct','dcn_zero_offset','dcn_nonzero','roi_construct',
                           'roi_cuda','roi_cpu','roi_compare','complete'))
EXCEPTION_CLASSES={cls:cls.__name__ for cls in (ImportError,ModuleNotFoundError,ValueError,TypeError,KeyError,AttributeError,
 RuntimeError,OSError,FileNotFoundError,PermissionError,TimeoutError,MemoryError,AssertionError,subprocess.TimeoutExpired)}


def failure(report,gate,exc):
    """First failure only: fixed gate/type labels, never exception text or args."""
    report.setdefault('failure_gate',gate if gate in FAILURE_GATES else 'other')
    report.setdefault('failure_class',EXCEPTION_CLASSES.get(type(exc),'other'))


def operator_error_category(exc):
    """Project bounded builtin CUDA diagnostics to labels; never publish text."""
    if type(exc)is not RuntimeError:return 'unclassified'
    args=exc.args
    if type(args)is not tuple or len(args)!=1 or type(args[0])is not str or len(args[0])>4096:
        return 'unclassified'
    text=args[0].lower()
    patterns=(('cuda_no_kernel_image',('no kernel image is available',)),
      ('cuda_invalid_device_function',('invalid device function',)),
      ('cuda_illegal_memory_access',('illegal memory access',)),
      ('cuda_driver',('driver version is insufficient','no nvidia driver','cuda driver initialization failed')),
      ('extension_device_unavailable',('implementation for device cuda:0 not found','not compiled with gpu support',
                                       'not compiled with cuda','implementation for device cuda not found')),
      ('tensor_device_mismatch',('expected all tensors to be on the same device','inconsistent device')),
      ('tensor_shape',("input shape and kernel shape won't match","input shape and kernel channels won't match")),
      ('cudnn_engine_unavailable',('unable to find an engine','unable to find a valid cudnn algorithm')),
      ('cudnn_status',('cudnn_status_',)),('cublas_status',('cublas_status_',)),
      ('cuda_out_of_memory',('cuda out of memory',)))
    matches=[label for label,fragments in patterns if any(fragment in text for fragment in fragments)]
    return matches[0]if len(matches)==1 else 'unclassified'


def rt_helper(code):
    p=code/'infra/mediapipe_cpu_runtime_verify.py';s=p.lstat();expected='936ad97c5ffca3b7f86e3462a600a247b6fa540d9b669e748892ff45e12aedf2'
    if (p.resolve()!=p or any(q.is_symlink()for q in(p,*p.parents))or not stat.S_ISREG(s.st_mode)
        or s.st_nlink!=1 or s.st_mode&0o222 or s.st_size!=23559 or hashlib.sha256(p.read_bytes()).hexdigest()!=expected):
        raise ValueError('Original stdlib identity/source helper differs')
    after=p.lstat()
    if any(getattr(s,k)!=getattr(after,k)for k in('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns')):
        raise ValueError('Original helper changed while hashing')
    spec=importlib.util.spec_from_file_location('wr_masa_qual_rt',p);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def current_source(rt,code,revision):
    rt.require(Path(__file__).resolve()==code/HELPERS[0] and set(p.name for p in code.parent.iterdir())=={'code','revision','source-sha256'},
               'Exact current executable snapshot required')
    return rt.source(ROOT,code,revision,ENTRY,HELPERS)


def snapshot(rt,revision,entry,proof):
    old=ROOT/'jobs'/revision/entry/'code';rt.canonical(old)
    rt.require(set(p.name for p in old.parent.iterdir())=={'code','revision','source-sha256'},'Exact original source parent required')
    rt.require(rt.source(ROOT,old,revision,entry,tuple(proof['helpers']))==proof,'Original whole source/marker proof differs')
    return old


def policy(rt,code):
    c=rt.pinned(code/CONFIG,CONFIG_PIN,16384)
    rt.require(c['schema']=='world_reward.masa_native_qualification.v1'and c['budget_seconds']==BUDGET
               and c['cpus']==4 and c['memory_bytes']==8<<30 and c['precision']=='float32_no_autocast_tf32_disabled',
               'Frozen native contract policy required')
    return c


def same(rt,row,facts):
    rt.require(all(type(row.get(k))is type(v)and row[k]==v for k,v in facts.items()),'Exact producer facts required')


def authenticate(rt,code,args):
    c=policy(rt,code);a=c['acquisition'];files={};oldparents=[]
    path=ROOT/'results/masa-acquisition-v1.json';r=rt.pinned(path,a['report'],100000);files[str(path)]=a['report']
    same(rt,r,dict(schema='world_reward.masa_acquisition_receipt.v1',stage='masa_native_r50_source_weight_acquisition',status='pass',phase='complete',
      producer_revision=a['producer_revision'],models_loaded=False,weights_decoded=False,packages_installed=False,rgb_or_datasets_used=False,
      ground_truth_used=False,gpu_used=False,challenge_inputs_used=False,oracle_modes=[],quality_evaluated=False,adopted=False,
      source_rehashed_after=True,artifacts_rehashed_after=True,public_sealed=True,owned_partials_removed=True))
    old=snapshot(rt,a['producer_revision'],'run_masa_acquire',r['source_binding']);oldparents.append(old.parent)
    rt.require(rt.identity(old/'infra/masa_acquire.py')==a['source_helper']and rt.identity(old/'configs/masa_acquisition_v1.json')==a['config'],
               'Original acquisition generator/config differs')
    ac=rt.pinned(old/'configs/masa_acquisition_v1.json',a['config'],16384)
    rt.require(r['config_identity']==a['config']and r['source_scope']==ac['source_scope']and len(r['assets'])==len(ac['assets']),
               'Exact authenticated selected source/weight subset required')
    wanted=set()
    for original,row in zip(ac['assets'],r['assets']):
        rt.require(all(row[k]==original[k]for k in('file','bytes','sha256','kind')),'Original publisher asset identity differs')
        p=DATA/row['file'];pin={k:row[k]for k in('bytes','sha256')};rt.require(rt.identity(p,600000000)==pin,'Acquisition asset differs')
        files[str(p)]=pin;wanted.add(p)
    rt.require({p for p in DATA.rglob('*')if p.is_file()}==wanted and all(not p.is_symlink()and not p.stat().st_mode&0o222 for p in DATA.rglob('*')),
               'Exact sealed acquisition inventory required')
    rr=ROOT/'results'/('masa-runtime-build-'+args.runtime_revision)/'report.json'
    runtime_pin=dict(bytes=args.runtime_report_bytes,sha256=args.runtime_report_sha256);runtime=rt.pinned(rr,runtime_pin,2<<20);files[str(rr)]=runtime_pin
    same(rt,runtime,dict(schema='world_reward.masa_runtime_build.v1',stage='masa_author_isolated_runtime_build',status='pass',phase='complete',
      producer_revision=args.runtime_revision,gpu_used=False,operator_qualified=False,model_constructed=False,model_or_dataset_read=False,
      quality_verified=False,adopted=False,license_eligibility_verified=False,system_site_packages=False,
      source_rehashed_after=True,artifacts_rehashed_after=True,base_rechecked_after=True,owned_containers_removed=True,owned_partial_cleanup=True))
    old=snapshot(rt,args.runtime_revision,'run_masa_runtime_build',runtime['source_binding']);oldparents.append(old.parent)
    rt.require(runtime['child_image']['image_id']==args.image_id and runtime['target_tag']=='world-reward/masa-author-runtime:'+args.runtime_revision,
               'Independent actual child image identity differs')
    rc=rt.pinned(old/'configs/masa_runtime_v1.json',runtime['config_identity'],100000)
    versions={w['name']:w['version']for w in rc['wheels']}
    rt.require(runtime['cpu_import']==dict(versions=versions,python='3.11',isolated_venv=True,cuda_initialized=False,cuda_build='11.8',
      extension_imported=True,operator_executed=False,model_constructed=False),'Original CPU import does not qualify GPU kernels')
    for row in rc['wheels']:
        p=rr.parent/'wheels'/row['filename'];identity=runtime['wheels'][row['name']]['identity'];ip={k:identity[k]for k in('bytes','sha256')}
        rt.require(rt.identity(p,2500000000)==ip and ip['bytes']==row['bytes']and(row['sha256']is None or ip['sha256']==row['sha256']),
                   'Original independently inventoried runtime wheel differs');files[str(p)]=ip
    for row in rc['publisher_notices']:
        p=rr.parent/'notices'/row['file'];notice_pin={k:row[k]for k in('bytes','sha256')};notice=runtime['publisher_notices'][row['file']]
        rt.require(set(notice)=={'bytes','sha256','publisher_md5'} and notice['publisher_md5']is None
                   and {k:notice[k]for k in('bytes','sha256')}==notice_pin and rt.identity(p,500000)==notice_pin,
                   'Original runtime publisher notice differs');files[str(p)]=notice_pin
    return dict(policy=c,source_binding=current_source(rt,code,args.revision),acquisition_report=a['report'],runtime_report=runtime_pin,
      runtime_revision=args.runtime_revision,image_id=args.image_id,image=runtime['child_image'],files=files,
      acquisition_source=r['source_binding'],runtime_source=runtime['source_binding'],old_source_parents=list(map(str,oldparents)))



def profile(args):
    return bool(getattr(args,'sm90_build_revision',None)or getattr(args,'sm90_child',False))


def result_name(revision,sm90=False):
    return ('masa-native-sm90-qualification-'if sm90 else'masa-native-qualification-')+revision


def native_stage(sm90=False):
    return 'masa_native_sm90_data_free_contract'if sm90 else'masa_native_data_free_contract'


def authenticate_profile(rt,code,args):
    """Original runtime stays original; new build is separately byte-authenticated."""
    if not profile(args):return authenticate(rt,code,args)
    c=rt.pinned(code/SM90_CONFIG,SM90_CONFIG_PIN,30000);parent=c['prior_runtime']
    original=argparse.Namespace(**vars(args));original.image_id=parent['image_id']
    rt.require(args.runtime_revision==parent['producer_revision']and
      dict(bytes=args.runtime_report_bytes,sha256=args.runtime_report_sha256)==parent['report'],'Pinned original parent required')
    proof=authenticate(rt,code,original);folder=ROOT/'results'/('masa-sm90-build-'+args.sm90_build_revision)
    pin=dict(bytes=args.sm90_build_report_bytes,sha256=args.sm90_build_report_sha256)
    r=rt.pinned(folder/'report.json',pin,2<<20)
    same(rt,r,dict(schema='world_reward.masa_sm90_build_receipt.v1',stage='masa_full_original_mmcv_sm90_cpu_build',status='pass',
      producer_revision=args.sm90_build_revision,config_identity=SM90_CONFIG_PIN,prior_runtime_revision=args.runtime_revision,
      prior_runtime_report=proof['runtime_report'],prior_image=proof['image'],gpu_used=False,model_constructed=False,
      checkpoint_read=False,dataset_read=False,operator_qualified=False,quality_verified=False,adopted=False,
      license_eligibility_verified=False,source_rehashed_after=True,prior_runtime_rehashed_after=True,assets_rehashed_after=True,
      prior_image_unchanged=True,owned_containers_removed=True,owned_disposable_cleanup=True,public_sealed=True))
    required={'infra/masa_sm90_build.py','infra/run_masa_sm90_build.sh',SM90_CONFIG,'infra/masa_runtime_build.py',
      'configs/masa_runtime_v1.json','infra/mediapipe_cpu_runtime_verify.py','infra/mediapipe_hands_acquire.py'}
    rt.require(required<=set(r['source_binding']['helpers']),'Complete original SM90 builder helper closure required')
    old=snapshot(rt,args.sm90_build_revision,'run_masa_sm90_build',r['source_binding'])
    rt.require(rt.identity(old/SM90_CONFIG)==SM90_CONFIG_PIN and r['full_source']['root_tree']==c['source']['root_tree']
      and r['full_source']['blobs']==c['source']['blob_count_including_symlink'],'Original whole MMCV source/config differs')
    same(rt,r['compile'],dict(compiler='11.8.89',gcc='11.4.0',gxx='11.4.0',ninja='1.11.1',arch='sm_90',
      tiny_object_compiled=True,ATen_CUDAContext_and_Python_headers_compiled=True,full_original_source_posthash=True,gpu_used=False))
    rt.require(r['target_tag']=='world-reward/masa-sm90-runtime:'+args.sm90_build_revision and
      r['child_image']['image_id']==args.image_id and args.image_id!=parent['image_id'] and
      len(r['child_image']['layers'])>len(proof['image']['layers'])and
      r['child_image']['layers'][:len(proof['image']['layers'])]==proof['image']['layers'],'New child lineage differs')
    before,after=r['base_cpu_import'],r['child_cpu_import'];versions=before['versions']
    runtime=rt.pinned(ROOT/'results'/('masa-runtime-build-'+args.runtime_revision)/'report.json',proof['runtime_report'],2<<20)
    rt.require(versions==runtime['cpu_import']['versions'] and len(versions)==52 and
      set(before['non_mmcv_distribution_file_digests'])==set(versions)-{'mmcv'} and
      all(re.fullmatch('[0-9a-f]{64}',v)for v in before['non_mmcv_distribution_file_digests'].values())and
      {k:v for k,v in before.items()if k!='mmcv_extension_sha256'}=={k:v for k,v in after.items()if k!='mmcv_extension_sha256'},
      'All original non-MMCV distributions must remain byte-identical')
    same(rt,after,dict(python='3.11',isolated_venv=True,cuda_build='11.8',cuda_initialized=False,operator_executed=False,model_constructed=False))
    w=r['compiled_wheel'];rt.require(re.fullmatch(r'mmcv-2\.1\.0-cp311-cp311-linux_x86_64\.whl',w['file'])and
      after['mmcv_extension_sha256']==w['inventory']['extension_sha256'],'Actual installed extension differs from compiled wheel')
    rt.require(before['mmcv_extension_sha256']!=after['mmcv_extension_sha256'],'New source-built extension required')
    rt.require(w['inventory']['extension_file']=='mmcv/_ext.cpython-311-x86_64-linux-gnu.so'and
      0<w['inventory']['extension_bytes']<150000000 and re.fullmatch('[0-9a-f]{64}',w['inventory']['extension_sha256']),
      'Bounded sole native extension required')
    expected={('notices/'if x.get('file','').endswith('.html')else'downloads/')+x['file']:{k:x[k]for k in('bytes','sha256')}
      for x in c['publisher_metadata']+c['assets']+c['header_assets']}
    expected['downloads/mmcv-tree.json']={k:c['source']['recursive_tree_metadata'][k]for k in('bytes','sha256')}
    expected[w['file']]=w['identity']
    for name,lic in [('CUDA-11.8-EULA.html',c['license']['cuda_eula']),
      ('NVIDIA-NC-StyleGAN2.html',c['license']['mmcv']['research_only_stylegan2_notice'])]:
        expected['notices/'+name]={k:lic[k]for k in('bytes','sha256')}
    rt.require(set(r['public_assets'])==set(expected)|{'downloads/mmcv-full-source.tar.gz'},'Exact original public build asset set required')
    rt.require(all(r['public_assets'].get(k)==v for k,v in expected.items()),'Complete pinned build assets required')
    proof['files'][str(folder/'report.json')]=pin
    for name,asset in r['public_assets'].items():
        p=PurePosixPath(name);rt.require(not p.is_absolute()and '..'not in p.parts and '\\'not in name and p.as_posix()==name,
          'Bounded original build asset path required')
        path=folder/name;rt.require(rt.identity(path,1000000000)==asset,'Original compiled build asset differs');proof['files'][str(path)]=asset
    rows={}
    for path in sorted((folder/'notices').rglob('*')):
        if path.is_file():rows[path.relative_to(folder/'notices').as_posix()]=rt.identity(path,1000000);proof['files'][str(path)]=rows[path.relative_to(folder/'notices').as_posix()]
    rt.require(dict(members=len(rows),bytes=sum(x['bytes']for x in rows.values()),
      sha256=hashlib.sha256(json.dumps(rows,sort_keys=True).encode()).hexdigest())==r['preserved_notices'],'Preserved original notices differ')
    allowed={folder/'report.json',folder/'downloads',folder/'notices',folder/'Dockerfile',folder/'.dockerignore',folder/'build.log'}
    allowed|={folder/name for name in r['public_assets']}|set((folder/'notices').rglob('*'))
    allowed|={folder/(label+suffix)for label in('base','compile','child')for suffix in('.cid','.log')}
    rt.require(set(folder.rglob('*'))==allowed,'Exact original sealed build inventory required')
    for p in (folder,*folder.rglob('*')):
        st=p.lstat();rt.require(not p.is_symlink()and (stat.S_ISDIR(st.st_mode)and stat.S_IMODE(st.st_mode)==0o555 or
          stat.S_ISREG(st.st_mode)and st.st_nlink==1 and stat.S_IMODE(st.st_mode)==0o444),'Sealed original build inventory required')
    proof.update(original_image=proof['image'],image=r['child_image'],image_id=args.image_id,
      sm90_child=dict(producer_revision=args.sm90_build_revision,report=pin,source_binding=r['source_binding'],
        compiled_wheel=w,config_identity=SM90_CONFIG_PIN,prior_image=r['prior_image'],child_cpu_import=after))
    proof['old_source_parents'].append(str(old.parent));proof['build_folder']=str(folder)
    return proof


def recheck_sources(rt,proof):
    snapshot(rt,proof['policy']['acquisition']['producer_revision'],'run_masa_acquire',proof['acquisition_source'])
    snapshot(rt,proof['runtime_revision'],'run_masa_runtime_build',proof['runtime_source'])
    if 'sm90_child'in proof:
        r=proof['sm90_child'];snapshot(rt,r['producer_revision'],'run_masa_sm90_build',r['source_binding'])



def architecture_write(rt,path,raw,mode,owned):
    fd=os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
    st=os.fstat(fd);owned[path]=((st.st_dev,st.st_ino,st.st_uid),None)
    with os.fdopen(fd,'wb')as stream:
        stream.write(raw);stream.flush();os.fsync(stream.fileno());os.fchmod(stream.fileno(),mode)
    owned[path]=(owned[path][0],rt.identity(path,150000000))

def architecture(rt,proof,deadline):
    """One fixed CPU-only binary inspection; not native operator qualification."""
    c=rt.pinned(Path(proof['old_source_parents'][-1])/'code'/SM90_CONFIG,SM90_CONFIG_PIN,30000)
    row=next(x for x in c['optional_assets']if x['component']=='cuda_cuobjdump')
    rt.require(row['version']=='11.8.86'and {k:row[k]for k in('bytes','sha256')}==CUOBJDUMP_PIN and
      row['url']=='https://developer.download.nvidia.com/compute/cuda/redist/cuda_cuobjdump/linux-x86_64/'+row['file'],'Fixed official architecture tool required')
    scratch=Path(tempfile.mkdtemp(prefix='wr-masa-sm90-arch-',dir=ROOT/'results'));owner=namespace_key(scratch);owned={};result=None
    try:
        path=scratch/'tool.tar.xz';request=urllib.request.Request(row['url'],headers={'Accept-Encoding':'identity'})
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),type('NoRedirect',(urllib.request.HTTPRedirectHandler,),
          {'redirect_request':lambda *_:(_ for _ in()).throw(ValueError('Tool redirect forbidden'))})())
        with opener.open(request,timeout=min(30,max(.001,deadline-time.monotonic())))as response:
            rt.require(response.status==200 and response.geturl()==row['url']and response.headers.get('Content-Encoding','identity')=='identity',
              'Pinned direct tool response required');raw=response.read(row['bytes']+1)
        rt.require(len(raw)==row['bytes']and hashlib.sha256(raw).hexdigest()==row['sha256'],'Official tool bytes differ')
        architecture_write(rt,path,raw,0o444,owned)
        prefix=row['file'].removesuffix('.tar.xz');tool=scratch/'cuobjdump'
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:xz')as archive:
            members=archive.getmembers();rt.require(len({m.name for m in members})==len(members)and len(members)<=64 and sum(m.size for m in members)<4000000,'Bounded tool archive required')
            for m in members:
                p=PurePosixPath(m.name);rt.require(not p.is_absolute()and '..'not in p.parts and p.parts and p.parts[0]==prefix and '\\'not in m.name
                  and p.as_posix().rstrip('/')==m.name.rstrip('/')and (m.isdir()or m.isfile())and not m.mode&0o6000,'Safe original tool archive required')
            candidates=[m for m in members if m.name==prefix+'/bin/cuobjdump'];rt.require(len(candidates)==1 and candidates[0].isfile(),'Sole official tool required')
            architecture_write(rt,tool,archive.extractfile(candidates[0]).read(),0o555,owned)
            notices={m.name.removeprefix(prefix+'/'):{'bytes':m.size,'sha256':hashlib.sha256(archive.extractfile(m).read()).hexdigest()}
              for m in members if m.isfile()and re.search(r'(^|/)(LICENSE|EULA|COPYING|NOTICE)',m.name,re.I)}
            rt.require(notices,'Original tool license/notice material required')
        wheel=Path(proof['build_folder'])/proof['sm90_child']['compiled_wheel']['file'];w=proof['sm90_child']['compiled_wheel']
        rt.require(rt.identity(wheel,1000000000)==w['identity'],'Frozen wheel changed before architecture check')
        ext=scratch/'extension.so'
        with zipfile.ZipFile(wheel)as z:
            matches=[x for x in z.infolist()if x.filename==w['inventory']['extension_file']]
            rt.require(len(matches)==1 and matches[0].file_size==w['inventory']['extension_bytes']and not matches[0].flag_bits&1
              and not stat.S_ISLNK(matches[0].external_attr>>16),'Sole bounded wheel extension required')
            architecture_write(rt,ext,z.read(matches[0]),0o444,owned)
        rt.require(owned[ext][1]['sha256']==w['inventory']['extension_sha256'],'Actual ELF identity differs')
        r=command([str(tool),'--list-elf',str(ext)],deadline)
        rt.require(r.returncode==0 and len(r.stdout)+len(r.stderr)<200000 and not r.stderr.strip()and
          re.search(rb'(?<![A-Za-z0-9_])sm_90(?![A-Za-z0-9_])',r.stdout),'Actual embedded sm_90 ELF required before GPU')
        result=dict(tool_archive={k:row[k]for k in('bytes','sha256')},tool=owned[tool][1],extension=owned[ext][1],
          listing=dict(bytes=len(r.stdout),sha256=hashlib.sha256(r.stdout).hexdigest()),embedded_sm90=True,
          gpu_used=False,operator_qualified=False,checkpoint_read=False,notice_members=notices,_archive_bytes=raw)
        for p,(key,pin)in owned.items():rt.require(namespace_key(p)==key and rt.identity(p,150000000)==pin,'Architecture material changed')
    finally:
        rt.require(namespace_key(scratch)==owner and set(scratch.iterdir())==set(owned),'Owned architecture scratch changed')
        for p,(key,pin)in owned.items():
            st=p.lstat();rt.require(namespace_key(p)==key and stat.S_ISREG(st.st_mode)and st.st_nlink==1 and
              (pin is None or rt.identity(p,150000000)==pin),'Never remove foreign architecture material');p.unlink()
        scratch.rmdir()
    rt.require(time.monotonic()<deadline,'Architecture check exceeded inclusive budget');result['owned_scratch_removed']=True
    return result

def command(args,deadline,*,log=None,pass_fds=()):
    remaining=deadline-time.monotonic()
    if remaining<=0:raise TimeoutError('Inclusive native qualification deadline')
    if log is None:r=subprocess.run(args,env=ENV,capture_output=True,timeout=min(30,remaining),pass_fds=pass_fds)
    else:
        with log.open('xb')as f:
            os.fchmod(f.fileno(),0o400);r=subprocess.run(args,env=ENV,stdout=f,stderr=subprocess.STDOUT,timeout=remaining,pass_fds=pass_fds)
    return r


def image(rt,identity,deadline):
    r=command(['docker','image','inspect',identity,'--format','{{json .}}'],deadline)
    rt.require(r.returncode==0 and len(r.stdout)<100000,'Bounded actual runtime image inspect required');value=rt.strict(r.stdout)
    rt.require(value['Id']==identity and value['Architecture']=='amd64'and value['Os']=='linux'and value['RootFS']['Type']=='layers',
               'Actual immutable Linux CUDA runtime required')
    return dict(image_id=value['Id'],layers=value['RootFS']['Layers'])



def active_image(rt,args,proof,deadline):
    value=image(rt,args.image_id,deadline)
    if profile(args):
        target='world-reward/masa-sm90-runtime:'+args.sm90_build_revision
        r=command(['docker','image','inspect',target,'--format','{{json .}}'],deadline)
        rt.require(r.returncode==0 and len(r.stdout)<100000,'Actual owned child tag required')
        j=rt.strict(r.stdout)
        rt.require(j['Id']==value['image_id']and j['RootFS']['Layers']==value['layers']and
          (j['Config'].get('Labels')or{}).get('world_reward_masa_sm90_owner')==args.sm90_build_revision,
          'Actual child tag/owner differs')
    return value

def strict_state(torch,payload,expected):
    """Only explicit common native layouts; no prefix/EMA/partial-load guessing."""
    if isinstance(payload,dict)and payload and all(isinstance(k,str)and torch.is_tensor(v)for k,v in payload.items()):
        state=payload;layout='tensor_mapping'
    elif isinstance(payload,dict)and 'state_dict'in payload and set(payload)<= {'state_dict','meta','optimizer','param_schedulers','message_hub'}:
        state=payload['state_dict'];layout='state_dict'
    else:raise ValueError('Unsupported actual native checkpoint layout')
    if not isinstance(state,dict)or set(state)!=set(expected):raise ValueError('Strict original state key set differs')
    inventory={}
    for k,v in state.items():
        e=expected[k]
        if not torch.is_tensor(v)or v.shape!=e.shape or v.dtype!=e.dtype or not torch.isfinite(v).all():
            raise ValueError('Strict original state shape/dtype/finite values differ')
        inventory[k]=dict(shape=list(v.shape),dtype=str(v.dtype))
    return state,dict(layout=layout,keys=len(state),state_layout_sha256=hashlib.sha256(json.dumps(inventory,sort_keys=True).encode()).hexdigest(),
                      prefix_rewrite=False,ema_selection=False,strict=True)


def operators(torch,c,progress):
    progress['operator_subgate']='imports'
    from mmcv.ops import ModulatedDeformConv2d,RoIAlign
    progress['operator_subgate']='dcn_construct'
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    x=(torch.arange(2*9*11,dtype=torch.float32,device='cuda').reshape(1,2,9,11)%17-8)/16
    conv=ModulatedDeformConv2d(2,3,3,padding=1,bias=True).cuda().eval()
    with torch.no_grad():
        progress['operator_subgate']='dcn_zero_offset'
        progress['operator_step']='initialize'
        conv.weight.copy_((torch.arange(conv.weight.numel(),device='cuda').reshape_as(conv.weight)%7-3)/32)
        conv.bias.copy_(torch.arange(3,device='cuda')/64)
        offset=torch.zeros((1,18,9,11),device='cuda');mask=torch.ones((1,9,9,11),device='cuda')
        progress['operator_step']='native_forward'
        y=conv(x,offset,mask)
        progress['operator_step']='conv2d_reference'
        ref=torch.nn.functional.conv2d(x,conv.weight,conv.bias,padding=1)
        progress['operator_step']='synchronize'
        torch.cuda.synchronize()
        g=c['operator_gates_before_checkpoint'];tol=g['dcn_zero_offset_mask_one_vs_conv2d']
        progress['operator_step']='compare'
        if not torch.allclose(y,ref,**tol):raise ValueError('Native DCNv2 zero-offset identity differs')
        progress['operator_subgate']='dcn_nonzero'
        offset.fill_(.125);mask.copy_(.25+torch.arange(mask.numel(),device='cuda').reshape_as(mask)%5/8)
        non=conv(x,offset,mask);repeat=conv(x,offset,mask);torch.cuda.synchronize()
        if not torch.isfinite(non).all()or not torch.equal(non,repeat)or torch.equal(non,y):raise ValueError('Native nontrivial DCNv2 unsupported')
        progress['operator_subgate']='roi_construct'
        roi=RoIAlign(output_size=7,sampling_ratio=0)
        boxes=torch.tensor([[0,1.125,2.25,8.5,7.75],[0,0,0,4,5]],dtype=torch.float32)
        progress['operator_subgate']='roi_cuda'
        gpu=roi(x,boxes.cuda())
        progress['operator_subgate']='roi_cpu'
        cpu=roi(x.cpu(),boxes)
        progress['operator_subgate']='roi_compare'
        torch.cuda.synchronize()
        if not torch.isfinite(gpu).all()or not torch.allclose(gpu.cpu(),cpu,**g['roi_align_cuda_vs_cpu']):raise ValueError('Native RoIAlign CPU/CUDA differs')
    progress['operator_subgate']='complete'
    return dict(dcn_zero_offset_max_abs=float((y-ref).abs().max().cpu()),dcn_nontrivial_repeat_exact=True,
      roi_cuda_cpu_max_abs=float((gpu.cpu()-cpu).abs().max()),roi_defaults=dict(output_size=list(roi.output_size),sampling_ratio=roi.sampling_ratio,
      spatial_scale=roi.spatial_scale,aligned=roi.aligned,pool_mode=roi.pool_mode,use_torchvision=roi.use_torchvision),gpu_synchronized=True,checkpoint_read=False)


def native_model(torch,c,deadline,progress):
    progress['subgate']='registry'
    import numpy as np
    import mmdet.models
    import mmdet.datasets.transforms
    from mmcv.transforms import Compose
    from mmengine.config import Config
    from mmengine.dataset import default_collate
    from mmengine.model.utils import revert_sync_batchnorm
    from mmengine.registry import init_default_scope
    from mmdet.registry import MODELS
    for i,name in enumerate(c['source_execution']['registry_leaves']):
        p=DATA/'source'/name;spec=importlib.util.spec_from_file_location('wr_masa_registry_'+str(i),p)
        m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
        if Path(m.__file__)!=p:raise ValueError('Native registry source origin differs')
    if any(n=='masa'or n.startswith('masa.')for n in sys.modules):raise ValueError('Root MASA package import forbidden')
    progress['subgate']='config'
    cfg=Config.fromfile(str(DATA/'source'/c['source_execution']['config']));init_default_scope(cfg.default_scope)
    if not(cfg.model.given_dets and not cfg.model.load_public_dets and cfg.model.use_masa_backbone):raise ValueError('Original model branch differs')
    progress['subgate']='model_construct'
    model=revert_sync_batchnorm(MODELS.build(cfg.model))
    progress['model_constructed']=True
    progress['subgate']='weights_decode'
    payload=torch.load(DATA/c['checkpoint_policy']['file'],map_location='cpu',weights_only=True)
    progress['weights_decoded']=True
    progress['subgate']='strict_state'
    state,layout=strict_state(torch,payload,model.state_dict());model.load_state_dict(state,strict=True)
    progress['model_loaded']=True
    progress['subgate']='model_device'
    model.to('cuda').eval()
    progress['subgate']='preprocess'
    p=c['procedural_model_control'];y,x,ch=np.indices((p['height'],p['width'],3))
    rgb=((x*3+y*5+ch*17)%256).astype(np.uint8)
    pipeline=Compose(cfg.inference_pipeline)
    data=pipeline(dict(img=[rgb.astype(np.float32)],frame_id=[0],ori_shape=[rgb.shape[:2]],img_id=[1],ori_video_length=[1]))
    processed=model.data_preprocessor(default_collate([data]),training=False)
    inputs=processed['inputs'];samples=processed['data_samples']
    if inputs.ndim!=5 or inputs.shape[:3]!=(1,1,3)or len(samples)!=1:raise ValueError('Original full-image native preprocessor shape differs')
    sample=samples[0].video_data_samples[0];scale=torch.tensor(sample.metainfo['scale_factor'],dtype=torch.float32,device='cuda')
    if scale.shape!=(2,)or not torch.isfinite(scale).all()or not (scale>0).all():raise ValueError('Actual native resize scale differs')
    # Independent bytewise reference for the original float-BGR resize,
    # channel reversal, normalization and divisor32 bottom/right padding.
    import mmcv
    original_resize=mmcv.imrescale(rgb.astype(np.float32),(1024,1024),return_scale=False,interpolation='bilinear',backend='cv2')
    tensor=torch.from_numpy(np.ascontiguousarray(original_resize.transpose(2,0,1))).cuda()
    mean=torch.tensor([123.675,116.28,103.53],dtype=torch.float32,device='cuda').reshape(3,1,1)
    std=torch.tensor([58.395,57.12,57.375],dtype=torch.float32,device='cuda').reshape(3,1,1)
    if (model.data_preprocessor.mean.numel()!=3 or model.data_preprocessor.std.numel()!=3
        or not torch.equal(model.data_preprocessor.mean.reshape(3,1,1),mean)
        or not torch.equal(model.data_preprocessor.std.reshape(3,1,1),std)
        or model.data_preprocessor.pad_size_divisor!=32 or model.data_preprocessor.pad_value!=0):
        raise ValueError('Original pinned preprocessing constants differ')
    expected=(tensor[[2,1,0]]-mean)/std
    padded_h=(original_resize.shape[0]+31)//32*32;padded_w=(original_resize.shape[1]+31)//32*32
    if inputs.shape[-2:]!=(padded_h,padded_w):raise ValueError('Original independent divisor32 padded dimensions differ')
    expected=torch.nn.functional.pad(expected,(0,padded_w-expected.shape[-1],0,padded_h-expected.shape[-2]),value=0)
    if not torch.equal(inputs[0,0],expected):raise ValueError('Original BGR normalization/resize/pad bytes differ')
    actual_scale=[original_resize.shape[1]/p['width'],original_resize.shape[0]/p['height']]
    if scale.cpu().tolist()!=torch.tensor(actual_scale,dtype=torch.float32).tolist():raise ValueError('Original full-image resize scale differs')
    boxes=torch.tensor(p['boxes_xyxy'],dtype=torch.float32,device='cuda');scaled=boxes*scale.repeat(2)
    with torch.no_grad():
        progress['subgate']='encoder'
        features=model.masa_adapter(model.backbone.forward(inputs[:,0].contiguous()))
        progress['subgate']='embedding'
        embeddings=model.track_head.predict(features,[scaled]);permutation=torch.tensor(p['permutation'],device='cuda')
        permuted=model.track_head.predict(features,[scaled[permutation]])
        empty=model.track_head.predict(features,[scaled[:0]])
        torch.cuda.synchronize()
    if (embeddings.shape!=(len(boxes),256)or empty.shape!=(0,256)or permuted.shape!=embeddings.shape
        or not torch.isfinite(embeddings).all()or not torch.isfinite(permuted).all()or not torch.isfinite(empty).all()):
        raise ValueError('Full native embedding bank shape/finite values differ')
    if not torch.equal(embeddings[0],embeddings[2]):raise ValueError('Duplicate native slots altered')
    if not torch.allclose(permuted,embeddings[permutation],atol=p['permutation_atol'],rtol=p['permutation_rtol']):raise ValueError('Native embedding slot permutation differs')
    if time.monotonic()>=deadline:raise TimeoutError('Inclusive native model deadline')
    return dict(checkpoint=layout,native_encoder_calls=1,native_embedding_calls=3,native_slots=4,embedding_channels=256,
       empty_shape=list(empty.shape),zero_area_slot_preserved=True,duplicate_embeddings_equal=True,permutation_preserved=True,
       scale_factor=scale.cpu().tolist(),original_image_size=[p['height'],p['width']],preprocessed_shape=list(inputs.shape),
       features_shapes=[list(f.shape)for f in features],embedding_sha256=hashlib.sha256(embeddings.cpu().numpy().tobytes()).hexdigest(),
       given_dets=True,load_public_dets=False,tracker_called=False,dataset_builder_called=False,root_masa_imported=False,
       preprocessing_byte_reference_exact=True)


def run_native(code,revision,out,proof_pin,deadline,*,sm90_child=False):
    rt=rt_helper(code);proof=rt.pinned(out/'proof.json',proof_pin,4<<20);c=proof['policy']
    rt.require(('sm90_child'in proof)==sm90_child,'Explicit runtime profile differs from pinned proof')
    rt.require(current_source(rt,code,revision)==proof['source_binding'],'Current native source differs before execution')
    recheck_sources(rt,proof)
    if 'sm90_child'in proof:
        rt.require(proof['embedded_architecture']['embedded_sm90']is True and proof['embedded_architecture']['owned_scratch_removed']is True
          and proof['embedded_architecture']['extension']['sha256']==proof['sm90_child']['compiled_wheel']['inventory']['extension_sha256'],
          'Actual completed CPU architecture evidence required before Torch')
    for path,pin in proof['files'].items():rt.require(rt.identity(Path(path),2500000000)==pin,'Frozen source/checkpoint/runtime bytes differ')
    report=dict(stage=native_stage('sm90_child'in proof),status='fail',phase='operators',producer_revision=revision,source_binding=proof['source_binding'],
       image_id=proof['image_id'],protocol_identity=CONFIG_PIN,proof_identity=proof_pin,gpu_used=False,model_loaded=False,
       model_constructed=False,weights_decoded=False,
       challenge_inputs_used=False,ground_truth_used=False,quality_verified=False,adoption=False,tracking_correctness_verified=False,
       identity_or_physical_ownership_verified=False,source_artifacts_rehashed_after=False,subgate='runtime_import')
    if 'sm90_child'in proof:report.update(schema='world_reward.masa_sm90_native_qualification.v1',
      sm90_child=proof['sm90_child'],embedded_architecture=proof['embedded_architecture'])
    try:
        import torch
        rt.require(torch.__version__=='2.1.2+cu118'and torch.cuda.is_available()and torch.cuda.get_device_capability()==(9,0)
                   and 'H100'in torch.cuda.get_device_name()and sys.prefix==VENV,'Actual author H100 runtime required')
        report['gpu_used']=True
        report['subgate']='operators'
        report['operators']=operators(torch,c,report);report['phase']='strict_checkpoint_model'
        report['model']=native_model(torch,c,deadline,report)
        report.update(status='pass',phase='complete',subgate='complete')
    except BaseException as exc:
        failure(report,report['subgate'],exc);report['error']='bounded_native_contract_failed'
        if report['subgate']=='operators':report['operator_error_category']=operator_error_category(exc)
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:
            rt.require(current_source(rt,code,revision)==proof['source_binding'],'Native source changed')
            recheck_sources(rt,proof)
            rt.require(rt.identity(out/'proof.json',4<<20)==proof_pin,'Native invocation proof changed')
            for path,pin in proof['files'].items():rt.require(rt.identity(Path(path),2500000000)==pin,'Frozen artifact changed after execution')
            report['source_artifacts_rehashed_after']=True
        except BaseException as exc:
            failure(report,'native_posthash',exc);report.update(status='fail',error='native_source_artifact_posthash_failed')
        if time.monotonic()>=deadline:
            failure(report,'native_deadline',TimeoutError());report.update(status='fail',error='inclusive_native_deadline_exceeded')
        rt.write(out/'native.json',(json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode(),0o444)
    if report['status']!='pass':raise ValueError('Native contract failed')from None
    return report


def cleanup(rt,cid,name,image_id,revision,deadline):
    if cid.exists():
        pin=rt.identity(cid,100,readonly=False);value=cid.read_text().strip();rt.require(re.fullmatch('[0-9a-f]{64}',value),'Exact owned CID required')
        r=command(['docker','inspect','--type','container','--format','{{json .}}',value],deadline)
        if r.returncode==0:
            actual=rt.strict(r.stdout);rt.require(actual['Id']==value and actual['Name']=='/'+name and actual['Image']==image_id
                and(actual['Config'].get('Labels')or{}).get('world_reward.masa_native.owner')==revision,'Never remove foreign container')
            rt.require(command(['docker','rm','-f',value],deadline).returncode==0,'Owned container cleanup failed')
        elif not(r.returncode==1 and r.stdout.strip()in(b'',b'[]')and r.stderr.strip()in(tuple(
          (prefix+value).encode()for prefix in('Error: No such object: ','error: no such object: ','Error: No such container: ',
            'error: no such container: ','Error response from daemon: No such container: ')))):
            raise ValueError('Exact absent owned CID required')
        rt.require(rt.identity(cid,100,readonly=False)==pin,'Owned CID changed');cid.chmod(0o444)
    r=command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline)
    rt.require(r.returncode==0 and not r.stdout.strip(),'Owned native container survives')


def acquire_lock_fd(lock):
    fd=os.open(lock,os.O_RDONLY|os.O_NOFOLLOW);os.dup2(fd,9)
    if fd!=9:os.close(fd)
    return 9


def namespace_key(path):
    s=path.lstat()
    return s.st_dev,s.st_ino,s.st_uid


def seal_outputs(rt,out,native,out_owner,native_owner,proof_pin):
    """Validate all owned entries before changing modes; never seal a replacement."""
    rt.require(rt.canonical(out)==out and rt.canonical(native)==native and namespace_key(out)==out_owner
      and namespace_key(native)==native_owner and out.is_dir()and native.is_dir(),'Owned result namespace changed')
    allowed={native:None,out/'container.cid':(out_owner[2],100),out/'native.log':(out_owner[2],1<<20),
      native/'proof.json':(out_owner[2],4<<20),native/'native.json':(native_owner[2],100000)}
    if (out/'cuobjdump.tar.xz').exists():allowed[out/'cuobjdump.tar.xz']=(out_owner[2],1000000)
    entries=list(out.rglob('*'))
    for p in entries:
        s=p.lstat();rt.require(p in allowed and rt.canonical(p)==p,'Unowned result entry or alias')
        if p==native:continue
        uid,cap=allowed[p]
        rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and s.st_uid==uid and s.st_size<=cap,
                   'Owned regular bounded result entry required')
    rt.require(rt.identity(native/'proof.json',4<<20)==proof_pin,'Owned invocation proof changed')
    pins={p:rt.identity(p,allowed[p][1],readonly=False)for p in entries if p!=native}
    for p in entries:
        if p!=native:p.chmod(0o444)
    native.chmod(0o555)
    for p,pin in pins.items():rt.require(rt.identity(p,allowed[p][1])==pin,'Sealed result bytes changed')
    return {p.relative_to(out).as_posix():pin for p,pin in pins.items()}


def dispatch(args,code):
    started=time.monotonic();deadline=started+BUDGET;rt=rt_helper(code)
    cancelled=lambda *_:(_ for _ in()).throw(TimeoutError('Inclusive host qualification deadline'))
    signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGALRM,cancelled);signal.setitimer(signal.ITIMER_REAL,BUDGET)
    proof=authenticate_profile(rt,code,args);rt.require(active_image(rt,args,proof,deadline)==proof['image'],'Actual runtime layers differ')
    out=rt.canonical(ROOT/'results'/result_name(args.revision,profile(args)));rt.require(not out.exists(),'Fresh native qualification only')
    if profile(args):
        rt.require(image(rt,proof['original_image']['image_id'],deadline)==proof['original_image'],'Original parent image changed')
        proof['embedded_architecture']=architecture(rt,proof,deadline)
    out.mkdir(mode=0o755);native=out/'native';native.mkdir(mode=0o700);os.chown(native,1000,1000)
    out_owner=namespace_key(out);native_owner=namespace_key(native)
    if profile(args):
        archive=proof['embedded_architecture'].pop('_archive_bytes');rt.write(out/'cuobjdump.tar.xz',archive,0o444)
        rt.require(rt.identity(out/'cuobjdump.tar.xz',1000000)==proof['embedded_architecture']['tool_archive'],'Retained original tool archive differs')
    proof['host_start_monotonic']=started;rt.write(native/'proof.json',(json.dumps(proof,sort_keys=True)+'\n').encode(),0o444)
    proof_pin=rt.identity(native/'proof.json',4<<20);cid=out/'container.cid';name='world-reward-masa-native-'+args.revision
    lock=rt.canonical(ROOT/'jobs/.world-reward-h100.lock');s=lock.lstat();rt.require(stat.S_ISREG(s.st_mode)and s.st_nlink==1,'Existing cooperative GPU lock required')
    fd=acquire_lock_fd(lock)
    report=dict(stage='masa_native_sm90_qualification_host'if profile(args)else'masa_native_qualification_host',status='fail',phase='preflight',source_binding=proof['source_binding'],producer_revision=args.revision,
       image_id=args.image_id,protocol_identity=CONFIG_PIN,quality_verified=False,adoption=False,owned_cleanup_verified=False,
       source_artifacts_rehashed_after=False,subgate='gpu_lock')
    if profile(args):report.update(schema='world_reward.masa_sm90_native_qualification_host.v1',
      sm90_child=proof['sm90_child'],embedded_architecture=proof['embedded_architecture'])
    try:
        fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
        rt.require((os.fstat(fd).st_dev,os.fstat(fd).st_ino)==(s.st_dev,s.st_ino),'Existing cooperative lock changed')
        report['subgate']='gpu_idle'
        idle=command(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],deadline)
        rt.require(idle.returncode==0 and not idle.stdout.strip(),'GPU busy after lock')
        report['subgate']='container_absence'
        absent=command(['docker','ps','-aq','--no-trunc','--filter','name=^/'+name+'$'],deadline)
        rt.require(absent.returncode==0 and not absent.stdout.strip(),'Fresh exact native container name required')
        mounts=[code.parent,*map(Path,proof['old_source_parents']),DATA,ROOT/'results'/('masa-runtime-build-'+args.runtime_revision)]
        if profile(args):mounts.append(Path(proof['build_folder']))
        argv=['docker','run','--rm','--name',name,'--cidfile',str(cid),'--label','world_reward.masa_native.owner='+args.revision,
          '--gpus','all','--network','none','--read-only','--user','1000:1000','--cap-drop','ALL','--security-opt','no-new-privileges',
          '--cpus','4','--memory','8g','--tmpfs','/tmp:rw,exec,nosuid,size=512m']
        for p in mounts:argv+=['--mount',f'type=bind,src={p},dst={p},readonly']
        acquisition_path=ROOT/'results/masa-acquisition-v1.json';argv+=['--mount',f'type=bind,src={acquisition_path},dst={acquisition_path},readonly']
        argv+=['--mount',f'type=bind,src={native},dst={native}','--entrypoint','/usr/bin/env',args.image_id,
          '-i','PATH='+VENV+'/bin:/opt/conda/bin:/usr/local/bin:/usr/bin:/bin','HOME=/tmp','LD_LIBRARY_PATH=',
          'MPLBACKEND=Agg','PYTHONDONTWRITEBYTECODE=1',VENV+'/bin/python',
          '-I','-B',str(code/HELPERS[0]),'--native','--revision',args.revision,'--out',str(native),'--proof-bytes',str(proof_pin['bytes']),
          '--proof-sha256',proof_pin['sha256'],'--deadline',repr(deadline)]
        if profile(args):argv+=['--sm90-child']
        report['subgate']='container_dispatch'
        report['phase']='native';r=command(argv,deadline,log=out/'native.log',pass_fds=(fd,));report['native_exit_status']=r.returncode
        report['subgate']='native_receipt'
        report['native_report_identity']=rt.identity(native/'native.json',100000)
        n=rt.pinned(native/'native.json',report['native_report_identity'],100000)
        same(rt,n,dict(stage=native_stage(profile(args)),status='pass',phase='complete',producer_revision=args.revision,
          image_id=args.image_id,protocol_identity=CONFIG_PIN,proof_identity=proof_pin,source_binding=proof['source_binding'],
          source_artifacts_rehashed_after=True,quality_verified=False,adoption=False,challenge_inputs_used=False,ground_truth_used=False,
          model_loaded=True,model_constructed=True,weights_decoded=True,gpu_used=True,tracking_correctness_verified=False,
          identity_or_physical_ownership_verified=False))
        if profile(args):
            same(rt,n,dict(schema='world_reward.masa_sm90_native_qualification.v1',sm90_child=proof['sm90_child'],
              embedded_architecture=proof['embedded_architecture']))
        rt.require(r.returncode==0 and n['operators']['checkpoint_read']is False and n['model']['native_encoder_calls']==1
          and n['model']['native_embedding_calls']==3,'Real native operator-before-checkpoint/full bank census required')
        report.update(status='pass',phase='complete',subgate='complete',native_report_identity=rt.identity(native/'native.json',100000))
    except BaseException as exc:
        failure(report,report['subgate'],exc);report['error']='bounded_native_lifecycle_failed'
    finally:
        signal.setitimer(signal.ITIMER_REAL,0)
        try:cleanup(rt,cid,name,args.image_id,args.revision,deadline+GRACE);report['owned_cleanup_verified']=True
        except BaseException as exc:
            failure(report,'host_cleanup',exc);report.update(status='fail',error='owned_native_cleanup_failed')
        try:
            rt.require(authenticate_profile(rt,code,args)=={k:v for k,v in proof.items()if k not in('host_start_monotonic','embedded_architecture')},'Native inputs/source changed after cleanup')
            if profile(args):rt.require(rt.identity(out/'cuobjdump.tar.xz',1000000)==proof['embedded_architecture']['tool_archive'],'Retained official tool changed')
            rt.require(active_image(rt,args,proof,deadline+GRACE)==proof['image'],'Actual runtime changed')
            if profile(args):rt.require(image(rt,proof['original_image']['image_id'],deadline+GRACE)==proof['original_image'],'Original runtime changed')
            report['source_artifacts_rehashed_after']=True
        except BaseException as exc:
            failure(report,'host_posthash',exc);report.update(status='fail',error='source_artifact_runtime_postcheck_failed')
        report['elapsed_seconds']=time.monotonic()-started
        if report['elapsed_seconds']>=BUDGET:
            failure(report,'host_deadline',TimeoutError());report.update(status='fail',error='inclusive_host_deadline_exceeded')
        try:
            if 'native_report_identity'in report:
                rt.require(rt.identity(native/'native.json',100000)==report['native_report_identity'],'Actual native receipt changed after cleanup')
            if report['status']=='pass':
                expected={native,cid,out/'native.log',native/'proof.json',native/'native.json'}
                if profile(args):expected.add(out/'cuobjdump.tar.xz')
                rt.require(set(out.rglob('*'))==expected,
                           'Complete native success output inventory required')
            report['outputs']=seal_outputs(rt,out,native,out_owner,native_owner,proof_pin);report['owned_output_sealed']=True
        except BaseException as exc:
            failure(report,'output_seal',exc);report.update(status='fail',error='owned_output_sealing_failed',owned_output_sealed=False)
        rt.require(rt.canonical(out)==out and namespace_key(out)==out_owner,'Never publish into a foreign result namespace')
        report_path=out/'report.json';report_fd=os.open(report_path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400)
        with os.fdopen(report_fd,'wb')as stream:
            stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode());stream.flush();os.fsync(stream.fileno())
            os.fchmod(stream.fileno(),0o444);out.chmod(0o555)
            elapsed=time.monotonic()-started
            if elapsed>=BUDGET and report['status']=='pass':
                failure(report,'host_deadline',TimeoutError())
                report.update(status='fail',elapsed_seconds=elapsed,error='inclusive_host_deadline_exceeded')
                stream.seek(0);stream.truncate();stream.write((json.dumps(report,sort_keys=True,allow_nan=False)+'\n').encode())
                stream.flush();os.fsync(stream.fileno())
        os.close(fd)
    if report['status']!='pass':raise ValueError('Native qualification failed; immutable receipts retained')from None
    return report


def parse(argv):
    options=[a.split('=',1)[0]for a in argv if a.startswith('--')]
    if len(options)!=len(set(options)):raise ValueError('Duplicate CLI options forbidden')
    p=argparse.ArgumentParser();p.add_argument('--native',action='store_true');p.add_argument('--sm90-child',action='store_true');p.add_argument('--revision',required=True)
    for n in('runtime-revision','runtime-report-sha256','image-id','out','proof-sha256','deadline','sm90-build-revision','sm90-build-report-sha256'):p.add_argument('--'+n)
    for n in('runtime-report-bytes','proof-bytes','sm90-build-report-bytes'):p.add_argument('--'+n,type=int)
    a=p.parse_args(argv)
    if not re.fullmatch('[0-9a-f]{40}',a.revision):raise ValueError('Exact source revision required')
    if not a.native:
        if a.sm90_child:raise ValueError('Native-only explicit child profile forbidden on host')
        optional=(a.sm90_build_revision,a.sm90_build_report_bytes,a.sm90_build_report_sha256)
        if any(x is not None for x in optional)and not(all(x is not None for x in optional)and
          re.fullmatch('[0-9a-f]{40}',a.sm90_build_revision)and 0<a.sm90_build_report_bytes<=2<<20 and
          re.fullmatch('[0-9a-f]{64}',a.sm90_build_report_sha256)):
            raise ValueError('All independent SM90 build pins required together')
        if not(a.runtime_revision and re.fullmatch('[0-9a-f]{40}',a.runtime_revision)and a.runtime_report_bytes and 0<a.runtime_report_bytes<=2<<20
               and a.runtime_report_sha256 and re.fullmatch('[0-9a-f]{64}',a.runtime_report_sha256)and a.image_id and re.fullmatch('sha256:[0-9a-f]{64}',a.image_id)):
            raise ValueError('Independent exact runtime pins required before invocation')
        if any(getattr(a,n)is not None for n in('out','proof_bytes','proof_sha256','deadline')):raise ValueError('Native-only flags forbidden')
    else:
        if any(getattr(a,n)is not None for n in('runtime_revision','runtime_report_bytes','runtime_report_sha256','image_id','sm90_build_revision','sm90_build_report_bytes','sm90_build_report_sha256')):
            raise ValueError('Host-only runtime pins forbidden in native mode')
        if not(a.out and a.proof_bytes and 0<a.proof_bytes<=4<<20 and a.proof_sha256 and re.fullmatch('[0-9a-f]{64}',a.proof_sha256)
               and a.deadline and math.isfinite(float(a.deadline)) and time.monotonic()<float(a.deadline)<=time.monotonic()+BUDGET):
            raise ValueError('Exact native proof/deadline required')
    return a


def main():
    a=parse(sys.argv[1:]);code=ROOT/'jobs'/a.revision/ENTRY/'code'
    if a.native:
        out=Path(a.out);deadline=float(a.deadline)
        if (sys.platform!='linux'or os.geteuid()!=1000
            or out!=ROOT/'results'/result_name(a.revision,a.sm90_child)/'native'):
            raise ValueError('Exact unprivileged native output required')
        cancelled=lambda *_:(_ for _ in()).throw(TimeoutError('Inclusive native deadline'))
        signal.signal(signal.SIGTERM,cancelled);signal.signal(signal.SIGALRM,cancelled)
        signal.setitimer(signal.ITIMER_REAL,deadline-time.monotonic())
        run_native(code,a.revision,out,dict(bytes=a.proof_bytes,sha256=a.proof_sha256),deadline,sm90_child=a.sm90_child)
    else:
        if sys.platform!='linux'or os.geteuid()!=0:raise ValueError('Actual Azure host dispatch required')
        result=dispatch(a,code);print(json.dumps(dict(stage=result['stage'],status=result['status'])))


if __name__=='__main__':
    try:main()
    except BaseException as exc:
        diagnostic={};failure(diagnostic,'bootstrap',exc)
        print('MASA native contract failed '+json.dumps(diagnostic,sort_keys=True),file=sys.stderr);raise SystemExit(1)from None
