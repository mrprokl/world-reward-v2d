"""Tiny process/provenance controls; no Azure, compilation, CGAL or mesh input."""
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(REPO/'infra'))
import cgal_sum_control as gate
import certified_solid_build as build


def write(path,data,mode=0o444):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists(): path.chmod(0o644)
    path.write_bytes(data); path.chmod(mode); return build.identity(path,empty=not data)


def process_result(arm,phases=gate.PHASES,returncode=0,trace=(),timed_out=False):
    rows=[dict(arm=arm,terms=gate.TERMS,phase=p,exact_zero_verified=i>=2)for i,p in enumerate(phases)]
    stdout='\n'.join(json.dumps(row)for row in rows)+'\n'
    return dict(stdout=stdout,stdout_bytes=len(stdout),stderr_bytes=123,stderr_sha256='b'*64,
                sanitizer_excerpt=list(trace),stderr_prefix_truncated=False,elapsed_seconds=.01,returncode=returncode,timed_out=timed_out)


TRACE=['ERROR: AddressSanitizer: stack-overflow on address 0x123',
       '#0 0x1 in CGAL::Lazy_exact_Add<int>::update_exact()',
       '#1 0x2 in CGAL::Lazy_exact_Add<int>::update_exact()',
       '#2 0x3 in CGAL::Handle_for<CGAL::Lazy_rep>::~Handle_for()']


def paired():
    return {arm:gate.arm_result(process_result(arm,gate.PHASES if arm=='balanced' else gate.PHASES[:2],
                0 if arm=='balanced' else -6,() if arm=='balanced' else TRACE),arm,build)for arm in ('sequential','balanced')}


def test_fixed_exact_terms_phases_and_no_geometry_cpp():
    source=(REPO/'infra/cgal_sum_control.cpp').read_text()
    assert 'constexpr std::size_t TERMS = 131072;' in source
    assert 'i % 2 ? -1 : 1' in source and 'sum += term;' in source
    assert 'next.emplace_back(terms[i] + terms[i + 1]);' in source and 'CGAL::exact(sum) == 0' in source
    assert 'CGAL_VERSION_NR == 1060011000' in source and '__FAST_MATH__' in source
    assert all(name in source for name in gate.PHASES)
    assert all(word not in source for word in ('read_mesh','load_mesh','repair','argv[2]'))


def test_flags_preserve_exact_arithmetic_and_asan_stack_attribution():
    assert gate.FLAGS == ['-O1','-g','-std=c++17','-fsanitize=address','-fno-omit-frame-pointer',
        '-fno-optimize-sibling-calls','-ffp-contract=off','-frounding-math','-fno-fast-math']
    assert (gate.TOTAL,gate.COMPILE,gate.ARM,gate.TERMS)==(600,180,60,131072)


def test_balanced_zero_alone_never_attributes_production_failure():
    rows={a:gate.arm_result(process_result(a),a,build)for a in ('sequential','balanced')}
    d=gate.decision(rows)
    assert d['status']=='inconclusive' and not d['lazy_exact_stack_overflow_supported']
    assert d['adoption'] is d['production_failure_explained'] is False


def test_only_actual_sequential_stack_overflow_and_repeated_lazy_frames_supported():
    result=gate.decision(paired())
    assert result['status']=='pass' and result['lazy_exact_stack_overflow_supported'] is True
    assert result['observed_sequential_stack_frames']==3
    assert result['production_failure_explained'] is result['adoption'] is False


@pytest.mark.parametrize('fault',['other_asan','generic_segv','few_frames','no_balanced_zero','timeout','before_build','exit0'])
def test_false_positive_attribution_rejected(fault):
    rows=paired();s=rows['sequential']
    if fault=='other_asan':s['sanitizer_excerpt'][0]='ERROR: AddressSanitizer: heap-use-after-free'
    elif fault=='generic_segv':s['sanitizer_excerpt']=[]
    elif fault=='few_frames':s['sanitizer_excerpt']=s['sanitizer_excerpt'][:2]
    elif fault=='no_balanced_zero':rows['balanced']['exact_zero_verified']=False
    elif fault=='timeout':s['timed_out']=True
    elif fault=='before_build':s['phases']=s['phases'][:1]
    else:s['returncode']=0
    assert gate.decision(rows)['status']=='inconclusive'


def test_teardown_stack_overflow_is_distinguished_from_exact_force():
    rows=paired();rows['sequential']=gate.arm_result(process_result('sequential',gate.PHASES[:4],-6,TRACE),'sequential',build)
    assert gate.decision(rows)['status']=='pass'
    assert rows['sequential']['phases'][-1]['phase']=='before_teardown'


@pytest.mark.parametrize('fault',['count','boolean_count','wrong_arm','extra_field','zero_claim','reordered','extra_phase','no_phase','nonjson','overflow'])
def test_strict_native_phase_schema_and_original_order(fault):
    r=process_result('balanced');rows=[json.loads(l)for l in r['stdout'].splitlines()]
    if fault=='count':rows[0]['terms']=8
    elif fault=='boolean_count':rows[0]['terms']=True
    elif fault=='wrong_arm':rows[0]['arm']='sequential'
    elif fault=='extra_field':rows[0]['confidence']=1.
    elif fault=='zero_claim':rows[1]['exact_zero_verified']=True
    elif fault=='reordered':rows[0],rows[1]=rows[1],rows[0]
    elif fault=='extra_phase':rows.append(rows[-1])
    elif fault=='no_phase':rows=[]
    elif fault=='nonjson':r['stdout']='not JSON'
    else:r['stdout_bytes']=8193
    if fault not in ('nonjson','overflow'):r['stdout']='\n'.join(json.dumps(x)for x in rows)
    with pytest.raises(ValueError):gate.arm_result(r,'balanced',build)


def test_command_drains_without_full_stderr_retention():
    program="import sys;print('small stdout');sys.stderr.write('noise'*40000+'\\nERROR: AddressSanitizer: stack-overflow\\n')"
    r=gate.command([sys.executable,'-I','-B','-c',program],3)
    assert r['returncode']==0 and r['stderr_bytes']>131072 and r['stderr_prefix_truncated']
    assert r['stdout']=='small stdout\n' and 'stderr' not in r
    assert len(json.dumps(r))<2000


def test_command_timeout_is_failure_not_zero_or_retried():
    r=gate.command([sys.executable,'-I','-B','-c','import time;time.sleep(1)'],.05)
    assert r['timed_out'] is True and r['returncode']<0 and r['elapsed_seconds']<1


def source_fixture(tmp_path,monkeypatch):
    root=tmp_path/'root';revision='a'*40;code=root/'jobs'/revision/gate.ENTRY/'code'
    for name in gate.HELPERS:write(code/name,b'new immutable source')
    write(code/build.CONFIG,json.dumps(build.EXPECTED).encode())
    write(code/'src/world_reward/__init__.py',b'')
    write(code.parent/'revision',(revision+'\n').encode());write(code.parent/'source-sha256',('b'*64+'\n').encode())
    for p in (code,*code.rglob('*')):
        if p.is_dir():p.chmod(0o555)
    monkeypatch.setattr(gate,'ROOT',root);monkeypatch.setattr(gate,'__file__',str(code/'infra/cgal_sum_control.py'))
    return root,code,revision


def test_complete_source_and_markers_not_legacy_producer(tmp_path,monkeypatch):
    _,code,revision=source_fixture(tmp_path,monkeypatch);p=gate.binding(code,revision,build)
    assert p['files']==len(gate.HELPERS)+1 and set(p['helpers'])==set(gate.HELPERS)
    assert p['markers']['revision']['bytes']==41 and gate.binding(code,revision,build)==p
    with pytest.raises(ValueError):gate.binding(code,'c'*40,build)
    with pytest.raises(ValueError):gate.binding(code.parent/'run_certified_solid_build'/'code',revision,build)


@pytest.mark.parametrize('fault',['writable','marker','symlink','hardlink','protocol','extra_snapshot'])
def test_source_mutation_refused_without_repair(tmp_path,monkeypatch,fault):
    _,code,revision=source_fixture(tmp_path,monkeypatch);p=code/gate.HELPERS[0]
    if fault=='writable':p.chmod(0o644)
    elif fault=='marker':write(code.parent/'revision',('c'*40+'\n').encode())
    elif fault=='symlink':p.parent.chmod(0o755);p.unlink();p.symlink_to(code/gate.HELPERS[1])
    elif fault=='hardlink':os.link(p,tmp_path/'foreign')
    elif fault=='protocol':write(code/build.CONFIG,b'{}')
    else:write(code.parent/'unexpected',b'foreign')
    with pytest.raises(ValueError):gate.binding(code,revision,build)


def test_owned_scratch_cleanup_and_foreign_symlink_preserved(tmp_path):
    scratch=tmp_path/'scratch';scratch.mkdir();p=scratch/'owned';write(p,b'owned');scratch.chmod(0o555)
    owner=(scratch.stat().st_dev,scratch.stat().st_ino);gate.cleanup_scratch(scratch,owner);assert not scratch.exists()
    scratch.mkdir();foreign=tmp_path/'foreign';write(foreign,b'foreign');(scratch/'alias').symlink_to(foreign)
    owner=(scratch.stat().st_dev,scratch.stat().st_ino)
    with pytest.raises(ValueError):gate.cleanup_scratch(scratch,owner)
    assert foreign.read_bytes()==b'foreign' and scratch.exists()


def test_owned_scratch_replaced_inode_never_deleted(tmp_path):
    p=tmp_path/'scratch';p.mkdir();owner=(p.stat().st_dev,p.stat().st_ino);p.rename(tmp_path/'original');p.mkdir()
    with pytest.raises(ValueError):gate.cleanup_scratch(p,owner)
    assert p.exists() and (tmp_path/'original').exists()


def test_wrapper_source_closure_no_inputs_gpu_or_legacy_namespace():
    script=REPO/'infra/run_cgal_sum_control.sh';r=subprocess.run(['bash','-n',str(script)],capture_output=True)
    assert r.returncode==0
    text=script.read_text();assert '/infra/cgal_sum_control.py' in text and '/infra/cgal_sum_control.cpp' in text
    assert 'run_cgal_sum_control/code' in text and '640s' in text and '(($#' not in text
    assert all(word not in text for word in ('--gpus','track_1','outputs/episode','apt','pip install'))


def test_actual_static_runtime_closure_retains_source_and_headers_contract():
    import azure_job
    files={p.relative_to(REPO).as_posix():p.read_bytes()for folder in ('infra','src','configs')
           for p in (REPO/folder).rglob('*')if p.is_file()and p.suffix in ('.py','.sh','.cpp','.hpp','.h','.json')}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_cgal_sum_control.sh')
    assert set(gate.HELPERS)<=set(selected)
    assert 'infra/certified_solid_query.cpp'in selected


def test_host_bootstrap_is_stdlib_only():
    program="""import importlib.abc,importlib.util,sys
class Guard(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,*args):
  if fullname.split('.')[0] in {'numpy','torch','trimesh','PIL','scipy'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Guard())
s=importlib.util.spec_from_file_location('new_control',sys.argv[1]);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
assert m.TERMS==131072
"""
    r=subprocess.run([sys.executable,'-I','-B','-S','-c',program,str(REPO/'infra/cgal_sum_control.py')],capture_output=True)
    assert r.returncode==0,r.stderr.decode()


def test_primary_header_correspondence_and_linked_rights(tmp_path,monkeypatch):
    assert gate.HEADER_PINS['CGAL/Lazy_exact_nt.h']==dict(bytes=46566,sha256='a02d707ddf05cdd14126dca87f74181ebad8398c7c7f6a25a18ede49c9739576')
    assert gate.HEADER_PINS['CGAL/Exact_predicates_exact_constructions_kernel.h']==dict(bytes=2647,sha256='98a9963c9460978f735edcb055fa1a511023d3125663721bbc1542bca72638ff')
    assert gate.RIGHTS['binary_is_apache_only'] is gate.RIGHTS['competition_eligibility_verified'] is False
    assert 'GPL-3.0-or-later' in gate.RIGHTS['linked_cgal_scope']
    headers=tmp_path/'headers';pins={n:write(headers/n,n.encode())for n in gate.HEADER_PINS}
    monkeypatch.setattr(gate,'HEADER_PINS',pins)
    first=gate.header_ledger(headers,build);write(headers/next(iter(pins)),b'changed')
    with pytest.raises(ValueError):gate.header_ledger(headers,build)
    assert len(first)==64


def test_readonly_mode_ledger_and_private_directory_rejection(tmp_path,monkeypatch):
    _,code,revision=source_fixture(tmp_path,monkeypatch);first=gate.binding(code,revision,build)
    (code/gate.HELPERS[0]).chmod(0o555);second=gate.binding(code,revision,build)
    assert first['files_sha256']==second['files_sha256']
    assert first['readonly_ledger_sha256']!=second['readonly_ledger_sha256']
    (code/'infra').chmod(0o500)
    with pytest.raises(ValueError):gate.binding(code,revision,build)


def test_compiler_failure_tail_bounded_and_not_saved_for_native_arm():
    r=gate.command([sys.executable,'-I','-B','-c',"import sys;sys.stderr.write('x'*8000+' compiler failure');sys.exit(2)"],3)
    assert r['returncode']==2 and len(r['failure_tail'])<=1000 and r['failure_tail'].endswith('compiler failure')
    native=process_result('balanced');native['failure_tail']='not a native diagnostic'
    assert 'failure_tail' not in gate.arm_result(native,'balanced',build)


def native_context(tmp_path,monkeypatch):
    root,code,revision=source_fixture(tmp_path,monkeypatch);work=root/'work';work.mkdir(mode=0o700)
    image='sha256:'+'d'*64;write(code/gate.PINS,json.dumps({'child_image_id':image}).encode())
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'getuid',lambda:1000)
    for k,v in {'WR_NATIVE_NETWORK':'none','CUDA_VISIBLE_DEVICES':'-1','WR_CPU_IMAGE_ID':image}.items():monkeypatch.setenv(k,v)
    actual_stat=Path.stat;actual_iterdir=Path.iterdir
    def stat_work(p,*args,**kwargs):
        s=actual_stat(p,*args,**kwargs)
        return SimpleNamespace(st_uid=1000,st_mode=s.st_mode)if p==work else s
    monkeypatch.setattr(Path,'stat',stat_work)
    monkeypatch.setattr(Path,'iterdir',lambda p:iter([Path('lo')])if str(p)=='/sys/class/net'else actual_iterdir(p))
    monkeypatch.setattr(gate,'header_ledger',lambda *_:'e'*64)
    return root,code,revision,work


@pytest.mark.parametrize('compile_fails',[False,True])
def test_native_compile_and_each_arm_separate_once(tmp_path,monkeypatch,compile_fails):
    _,code,revision,work=native_context(tmp_path,monkeypatch);calls=[]
    def command(args,seconds):
        calls.append((args,seconds))
        if args[0]=='c++':
            result=process_result('balanced');result['stdout']='';result['stdout_bytes']=0
            result.update(returncode=2 if compile_fails else 0,failure_tail='compiler failure'if compile_fails else '')
            if not compile_fails:write(work/'sum-control',b'fake compiled binary',0o555)
            return result
        return process_result(args[-1],gate.PHASES if args[-1]=='balanced'else gate.PHASES[:2],0 if args[-1]=='balanced'else -6,()if args[-1]=='balanced'else TRACE)
    monkeypatch.setattr(gate,'command',command)
    result=gate.native(code,revision,work,Path('/unused'),build);receipt=build.strict_json((work/'native.json').read_bytes())
    assert result==int(compile_fails) and receipt['source_rehashed_after'] is receipt['headers_rehashed_after'] is True
    assert calls[0][0][1:1+len(gate.FLAGS)]==gate.FLAGS and calls[0][1]<=180
    assert len(calls)==(1 if compile_fails else 3)
    if compile_fails:assert receipt['compile']['failure_tail']=='compiler failure' and 'arms'not in receipt
    else:
        assert [a[-1]for a,_ in calls[1:]]==['sequential','balanced'] and all(t<=60 for _,t in calls[1:])
        assert receipt['status']=='pass' and receipt['adoption'] is receipt['production_failure_explained'] is False


@pytest.mark.parametrize('bad_native',[False,True])
def test_host_narrow_mounts_and_cleanup_before_receipt_parse(tmp_path,monkeypatch,bad_native):
    root,code,revision=source_fixture(tmp_path,monkeypatch);(root/'results').mkdir()
    monkeypatch.setattr(gate.sys,'platform','linux');monkeypatch.setattr(gate.os,'getuid',lambda:0)
    monkeypatch.setattr(gate.os,'chown',lambda *_:None);monkeypatch.setenv('DOCKER_HOST','unix://'+str(root/'docker.sock'))
    monkeypatch.setattr(gate,'header_ledger',lambda *_:'e'*64)
    image='sha256:'+'d'*64;archive=b'fake archive';checksum=hashlib.sha256(archive).hexdigest()
    sums=(checksum+' CGAL-6.0.1-library.tar.xz\n').encode();calls=[];cleanups=[]
    def pin(raw):return dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    def download(p,path,deadline):write(path,archive if p['sha256']==checksum else sums)
    def extract(_,headers,__):write(headers/'CGAL/owned.h',b'owned header');return {'files':1}
    def run(args,seconds,log=None):
        calls.append(args)
        if args[1:3]==['image','inspect']:return image.encode()+b'\n'
        if args[1]=='ps':return b''
        work=root/'results'/('cgal-sum-control-'+revision)/'disposable/work'
        before=gate.binding(code,revision,build)
        receipt=dict(stage='cgal_sum_control_native_v1',status='inconclusive',phase='complete',terms=gate.TERMS,image_id=image,
            source_binding=before,source_binding_after=before,source_rehashed_after=True,headers_rehashed_after=True,
            elapsed_seconds=.01,compiler_flags=gate.FLAGS,arms={a:gate.arm_result(process_result(a),a,build)for a in ('sequential','balanced')},
            gpu_used=False,gt_used=False,challenge_inputs_used=False,adoption=False,production_failure_explained=False)
        write(work/'native.json',b'bad JSON'if bad_native else json.dumps(receipt).encode());return b''
    fake=SimpleNamespace(CONFIG=build.CONFIG,EXPECTED=build.EXPECTED,strict_json=build.strict_json,identity=build.identity,
        seal=build.seal,write_json=build.write_json,CGAL=pin(archive),SUM=pin(sums),download=download,extract_headers=extract,
        run=run,cleanup_container=lambda *a:cleanups.append(a))
    certificate=SimpleNamespace(qualification=lambda *_:({'child_image_id':image},(),{'status':'pass','independently_pinned':True}))
    assert gate.host(code,revision,fake,certificate)==int(bad_native)
    out=root/'results'/('cgal-sum-control-'+revision);receipt=build.strict_json((out/'report.json').read_bytes())
    assert cleanups and receipt['owned_container_removed'] is receipt['owned_scratch_removed'] is True
    assert not(out/'disposable').exists() and stat.S_IMODE(out.stat().st_mode)==0o555
    docker=next(a for a in calls if a[:2]==['docker','run'])
    assert '--gpus'not in docker and docker[docker.index('--user')+1]=='1000:1000' and '--network'in docker
    mounts=[docker[i+1]for i,a in enumerate(docker)if a=='--mount']
    assert len(mounts)==5 and sum(m.endswith(',readonly')for m in mounts)==4
    assert all('track_1'not in m and 'outputs/episode'not in m for m in mounts)
    if bad_native:assert receipt['status']=='fail' and 'posthash_failure_type'in receipt
    else:assert receipt['status']=='inconclusive' and receipt['adoption'] is False
