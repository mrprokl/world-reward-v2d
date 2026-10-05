"""Manufactured CPU source/receipt and shell spies; never media or GPU execution."""
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
WRAPPER=REPO/'infra/run_object_pose_smoke.sh'
HOST=WRAPPER.read_text().split("<<'PYSOLID'\n",1)[1].split('\nPYSOLID',1)[0]
IMAGE='sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'


def pin(p):return dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())


@pytest.fixture
def runtime_factory(tmp_path):
 def create(profile='solid'):
  return _runtime(tmp_path/profile,profile)
 return create


@pytest.fixture
def runtime(runtime_factory):return runtime_factory()


def _runtime(tmp_path,profile):
 root=tmp_path/'root';rev='a'*40;code=root/'jobs'/rev/'run_object_pose_smoke/code';code.mkdir(parents=True)
 base=root/'outputs/episode_000009';base.mkdir(parents=True)
 def write(p,value):
  p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(value if isinstance(value,bytes)else json.dumps(value).encode());return p
 for name in('revision','source-sha256'):write(code.parent/name,((rev if name=='revision'else'b'*64)+'\n').encode()).chmod(0o444)
 write(root/'jobs/.world-reward-h100.lock',b'original nontruncated cooperative lock')
 # Entire actual source snapshot is host-only; GPU selection follows imports.
 for folder in('infra','src/world_reward'):
  for p in(REPO/folder).glob('*'):
   if p.is_file()and p.suffix in('.py','.hpp'):write(code/p.relative_to(REPO),p.read_bytes())
 write(code/'infra/run_object_pose_smoke.sh',WRAPPER.read_bytes())
 proposal=base/('object_budget_'+profile+'_'+'c'*40);qualification=root/'results'/(('surface-qslim-qualify-'if profile=='surface'else'solid-chart-v2-qualify-')+'d'*40)
 originals=[base/'object_grounded'/n for n in('report.json','object.glb','transform.json','intrinsics.json')]+[base/'scale_smoke/report.json']
 artifacts=[proposal/n for n in('report.json','native.json','geometry.npz','object_fixed_canonical.glb')]+originals+[qualification/n for n in('report.json','native.json')]
 if profile=='surface':artifacts+=[proposal/'candidate_geometry.npz',proposal/'mapping.json',root/'results'/('surface-identity-qualify-'+'e'*40)/'native.json',root/'results/surface-qslim-independent-v2/report.json']
 for p in artifacts:write(p,b'explicit synthetic artifact, not qualified geometry')
 if profile=='surface':
  import importlib.util
  rspec=importlib.util.spec_from_file_location('test_hist_rt',REPO/'infra/mediapipe_cpu_runtime_verify.py');rt=importlib.util.module_from_spec(rspec);rspec.loader.exec_module(rt)
  for revision,entrypoint,receipt in(('c'*40,'run_object_budget_solid',proposal/'report.json'),('d'*40,'run_surface_qslim_qualify',qualification/'native.json'),('e'*40,'run_surface_identity_qualify',root/'results'/('surface-identity-qualify-'+'e'*40)/'native.json')):
   old=root/'jobs'/revision/entrypoint/'code';write(old/'infra/original.cpp',b'// historical source, never executed\n')
   for n,val in(('revision',revision),('source-sha256','f'*64)):write(old.parent/n,(val+'\n').encode()).chmod(0o444)
   for p in(old,*old.rglob('*')):p.chmod(0o555 if p.is_dir()else 0o444)
   source=rt.source(root,old,revision,entrypoint,())
   write(receipt,{'source_binding':source}if entrypoint=='run_object_budget_solid'else{'source_proof':{'source_binding':source}})
 for directory in(proposal,qualification):
  for p in directory.iterdir():p.chmod(0o444)
  directory.chmod(0o555)
 video=write(root/'data/track_1/videos/chunk-000/observation.images.exo_camera/episode_000009.mp4',b'not a video, source hash only')
 meta=write(root/'data/track_1/meta/episodes.jsonl',b'{"episode_index":9,"length":3}\n')
 manifest=dict(track='track_1',repo_id='nvidia/video_to_data_challenge',revision='5f68335f3acc802033d1e80728c1633197521de8',files=[dict(path=str(p.relative_to(root/'data')),**pin(p))for p in(meta,video)])
 write(root/'results/input-manifest.json',manifest)
 write(base/'automatic_masks/report.json',dict(synthetic=True));write(base/'automatic_masks/prompts.json',dict(synthetic=True))
 for k in(0,1):
  for i in range(3):write(base/f'automatic_masks/masks/{k}/{i:06d}.png',b'not a PNG, bytes only')
 shared=dict(status='pass',episode_index=9,total_video_frames=3,input_track='track_1',input_sha256=pin(video)['sha256'],ground_truth_used=False,hand_labeled_test=False,oracle_modes=[])
 records=[]
 for i in range(3):
  p=write(base/f'depth_full/{i:06d}.npz',b'not an NPZ, bytes only');records.append(dict(frame_index=i,output_sha256=pin(p)['sha256']))
 write(base/'depth_full/report.json',dict(shared,stage='monocular_moge2_full_video',frames=records))
 write(base/'body_full/report.json',dict(shared,stage='sam3d_body_full_video_initializer',frames=[dict(frame_index=i)for i in range(3)]))
 for name in('body_smoke','depth_smoke'):write(base/name/'report.json',dict(synthetic=True))
 import importlib.util
 spec=importlib.util.spec_from_file_location('synthetic_selected_loader_schema',REPO/f'infra/{profile}_geometry_loader.py');loader=importlib.util.module_from_spec(spec)
 sys.path.insert(0,str(REPO/'infra'));spec.loader.exec_module(loader)
 helpers={name:pin(code/name)for name in loader.SOURCE_HELPERS}
 pins=dict(schema=loader.SCHEMA,episode_index=9,input_sha256=pin(video)['sha256'],metric_scale_baked_once=.375,report=dict(pin(proposal/'report.json'),producer_revision='c'*40,script_sha256='e'*64),files={str(p.relative_to(root)):pin(p)for p in artifacts},source_helpers=helpers)
 pinpath=write(code/f'configs/{profile}_mesh_000009_pins.json',pins)
 if profile=='surface':
  write(code/'configs/surface_qslim_qualification_pins.json',dict(producer_revision='d'*40,independent_audit_path='results/surface-qslim-independent-v2/report.json'))
  write(code/'configs/surface_identity_qualification_pins.json',dict(producer_revision='e'*40))
  write(code/'configs/surface_qslim_build_pins.json',dict(synthetic=True))
 else:
  write(code/'configs/solid_chart_v2_qualification_pins.json',dict(producer_revision='d'*40))
  for name in('solid_chart_v2_build_pins','certified_solid_qualification_pins'):write(code/f'configs/{name}.json',dict(synthetic=True))
 official=write(root/'vendor/v2d_submission_kit/v2dlb/mesh_budget.py',b'original placeholder fixture official helper')
 write(official.with_name('__init__.py'),b'')
 for p in(code,*code.rglob('*')):p.chmod(0o555 if p.is_dir()else 0o444)
 control=root/'results'/f'object-pose-{profile}-000009-{rev}';out=base/f'object_pose_full_{profile}';lock=root/'jobs/.world-reward-h100.lock'
 # Only the manufactured root and known official fixture hash are substituted.
 host=HOST.replace("dict(bytes=2031,sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0')",repr(pin(official)))
 host=host.replace('if os.getuid()!=0 or entry!=', 'if entry!=')
 args=[str(root),str(code),rev,'9',str(control),str(out),str(lock)]
 def call(mode):
  env=dict(os.environ)
  if(control/'proof.json').exists():env['WR_POSE_PROOF_SHA256']=pin(control/'proof.json')['sha256']
  return subprocess.run(['rtk','proxy',sys.executable,'-I','-B','-',*args,mode,str(code/'infra/run_object_pose_smoke.sh'),'0',profile],input=host,text=True,capture_output=True,timeout=10,env=env)
 def complete():
  out.mkdir();write(out/'geometry_and_poses.npz',b'actual payload checked in tracker, fixture bytes only')
  write(out/'object_fixed_canonical.glb',(proposal/'object_fixed_canonical.glb').read_bytes())
  proof=json.loads((control/'proof.json').read_text())
  record=dict(shared,stage='fixed_scale_full_object_pose_initializer',mesh_source=profile,execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,challenge_performance_verified=False,frames=[dict(frame_index=i)for i in range(3)],temporal_selection=dict(candidate_indices=[0,1,2]),geometry_and_poses_sha256=pin(out/'geometry_and_poses.npz')['sha256'],fixed_canonical_mesh_sha256=pin(out/'object_fixed_canonical.glb')['sha256'],topology_budget=dict(committed_pins_sha256=pin(pinpath)['sha256']),script_sha256=proof['selected_source'][str(code/'infra/object_pose_smoke.py')]['sha256'])
  write(out/'report.json',record)
 return dict(root=root,code=code,rev=rev,base=base,out=out,control=control,lock=lock,call=call,complete=complete,pins=pinpath,video=video,proposal=proposal,qualification=qualification,official=official)


def test_real_host_source_import_closure_and_fullt_freeze(runtime):
 before=runtime['call']('before');assert before.returncode==0,before.stderr
 mounts=runtime['call']('mounts');assert mounts.returncode==0,mounts.stderr
 rows=mounts.stdout.splitlines();code=runtime['code'];root=runtime['root']
 assert str(runtime['proposal'])in rows and str(runtime['qualification'])in rows
 assert str(code)not in rows and str(root/'outputs')not in rows
 assert all('/weights/'not in p and '/validation/'not in p and not p.endswith(('mesh_conditioned_chart_v2','certified_solid_query'))for p in rows)
 assert str(code/'infra/solid_geometry_loader.py')in rows and str(code/'infra/object_budget_endpoint.py')in rows
 assert str(code/'infra/object_budget_solid.py')not in rows and str(code/'infra/oriented_solid_compiler.py')not in rows
 assert {str(p.relative_to(code))for p in map(Path,rows)if p.is_relative_to(code/'configs')}=={'configs/solid_mesh_000009_pins.json','configs/solid_chart_v2_qualification_pins.json','configs/solid_chart_v2_build_pins.json','configs/certified_solid_qualification_pins.json'}
 runtime['complete']();after=runtime['call']('complete');assert after.returncode==0,after.stderr
 assert after.stdout==before.stdout and runtime['out'].stat().st_mode&0o777==0o555
 assert all(p.stat().st_mode&0o777==0o444 for p in runtime['out'].iterdir())


def test_surface_host_exact15_sources_narrow_gpu_and_complete_proof(runtime_factory):
 r=runtime_factory('surface');before=r['call']('before');assert before.returncode==0,before.stderr
 mounts=r['call']('mounts');assert mounts.returncode==0,mounts.stderr
 rows=mounts.stdout.splitlines();proof=json.loads((r['control']/'proof.json').read_text())
 assert 'surface_pins_identity' in proof and 'solid_pins_identity' not in proof
 assert str(r['code']/'infra/surface_geometry_loader.py') in rows
 assert str(r['code']/'infra/object_budget_solid.py') not in rows and not any(p.endswith('surface_qslim')for p in rows)
 assert str(r['proposal']) in rows and str(r['qualification']) in rows
 assert str(r['root']/'results/surface-qslim-independent-v2/report.json') in rows
 historical=proof['hash_only_historical_sources'];assert len(historical)==3
 assert set(historical)<=set(rows) and all(Path(p).parent.parent==r['root']/'jobs'for p in historical)
 assert not any(str(r['root']/'jobs')==p for p in rows)
 assert all('/validation/'not in p and '/weights/'not in p for p in rows)
 r['complete']();after=r['call']('complete');assert after.returncode==0,after.stderr
 assert after.stdout==before.stdout and r['out'].name=='object_pose_full_surface' and r['out'].stat().st_mode&0o777==0o555


@pytest.mark.parametrize('fault',['missingpin','missingcandidate','extraartifact','wrongprofile','qualificationpath','historical_binary','historical_mutation'])
def test_surface_host_fail_closed_before_native_output(runtime_factory,fault):
 r=runtime_factory('surface');p=r['pins'];p.chmod(0o644);v=json.loads(p.read_text())
 if fault=='missingpin':p.parent.chmod(0o755);p.unlink();p.parent.chmod(0o555)
 elif fault=='missingcandidate':v['files'].pop(str((r['proposal']/'candidate_geometry.npz').relative_to(r['root'])))
 elif fault=='extraartifact':v['files']['results/foreign.json']=dict(bytes=1,sha256='f'*64)
 elif fault=='wrongprofile':v['schema']='world_reward.solid_mesh_pins.v1'
 elif fault=='qualificationpath':
  q=r['code']/'configs/surface_qslim_qualification_pins.json';q.chmod(0o644);x=json.loads(q.read_text());x['independent_audit_path']='results/foreign/report.json';q.write_text(json.dumps(x));q.chmod(0o444)
 elif fault=='historical_binary':
  old=r['root']/'jobs'/('c'*40)/'run_object_budget_solid/code';old.chmod(0o755);(old/'binary').write_bytes(b'ELF not source');(old/'binary').chmod(0o444);old.chmod(0o555)
 else:
  p0=r['root']/'jobs'/('c'*40)/'run_object_budget_solid/code/infra/original.cpp';p0.chmod(0o644);p0.write_bytes(b'changed original');p0.chmod(0o444)
 if fault!='missingpin':p.write_text(json.dumps(v));p.chmod(0o444)
 result=r['call']('before');assert result.returncode!=0 and not r['out'].exists() and not r['control'].exists()


@pytest.mark.parametrize('fault',['missingpin','extraartifact','videohash','fullframes','directoryextra','sourcewritable','binaryinqualification','preexistingout'])
def test_host_fail_closed_before_reserved_output(runtime,fault):
 if fault=='missingpin':runtime['pins'].parent.chmod(0o755);runtime['pins'].unlink();runtime['pins'].parent.chmod(0o555)
 elif fault=='extraartifact':
  p=runtime['pins'];p.chmod(0o644);v=json.loads(p.read_text());v['files']['outputs/episode_000009/foreign.json']=dict(bytes=2,sha256='0'*64);p.write_text(json.dumps(v));p.chmod(0o444)
 elif fault=='videohash':runtime['video'].write_bytes(b'changed')
 elif fault=='fullframes':
  p=runtime['base']/'body_full/report.json';v=json.loads(p.read_text());v['frames']=v['frames'][:-1];p.write_text(json.dumps(v))
 elif fault=='directoryextra':(runtime['base']/'depth_full/foreign').write_bytes(b'foreign')
 elif fault=='sourcewritable':(runtime['code']/'infra/body_smoke.py').chmod(0o644)
 elif fault=='binaryinqualification':
  runtime['qualification'].chmod(0o755);p=runtime['qualification']/'binary';p.write_bytes(b'never mounted');p.chmod(0o555)
 elif fault=='preexistingout':runtime['out'].mkdir();(runtime['out']/'old').write_bytes(b'historical')
 result=runtime['call']('before');assert result.returncode!=0
 assert not runtime['control'].exists()
 if fault!='preexistingout':assert not runtime['out'].exists()
 else:assert(runtime['out']/'old').read_bytes()==b'historical'


@pytest.mark.parametrize('fault',['source','marker','pin','depth','mask','video'])
def test_posthash_catches_actual_original_input_mutation(runtime,fault):
 assert runtime['call']('before').returncode==0
 choices=dict(source=runtime['code']/'infra/body_smoke.py',marker=runtime['code'].parent/'revision',pin=runtime['pins'],depth=runtime['base']/'depth_full/000000.npz',mask=runtime['base']/'automatic_masks/masks/1/000000.png',video=runtime['video'])
 p=choices[fault];p.chmod(0o644);p.write_bytes(b'changed original')
 assert runtime['call']('after').returncode!=0


@pytest.mark.parametrize('fault',['wrongmode','coverage','GT','artifact','extra','script'])
def test_native_fakepass_scope_and_fullt_artifacts_rejected(runtime,fault):
 assert runtime['call']('before').returncode==0;runtime['complete']()
 p=runtime['out']/'report.json';v=json.loads(p.read_text())
 if fault=='wrongmode':v['mesh_source']='default'
 elif fault=='coverage':v['frames']=v['frames'][:-1]
 elif fault=='GT':v['ground_truth_used']=True
 elif fault=='artifact':(runtime['out']/'geometry_and_poses.npz').write_bytes(b'changed')
 elif fault=='extra':(runtime['out']/'foreign').write_bytes(b'foreign')
 elif fault=='script':v['script_sha256']='0'*64
 p.write_text(json.dumps(v));assert runtime['call']('complete').returncode!=0


def test_original_legacy_docker_branch_exact_and_solid_resource_source_contract():
 old=subprocess.check_output(['rtk','git','show','1a51c715a46c376d82650c595d008e83bf250554:infra/run_object_pose_smoke.sh'],cwd=REPO,text=True)
 text=WRAPPER.read_text();assert text[text.rfind('docker run --rm --gpus all'):]==old[old.index('docker run --rm --gpus all'):]
 for value in('--timeout 43200 9','exec 9<"$LOCK"','7200s docker run','--memory 16g','--cpus 4','--user 1000:1000','--read-only','--cap-drop ALL','no-new-privileges','size=128m','WR_POSE_OUTPUT_RESERVED=1'):
  assert value in text
 assert text.index('flock --timeout 43200 9')<text.index('APPS="$(timeout 5s nvidia-smi')<text.index('7200s docker run')
 assert '9>"'not in text and 'src=$ROOT/outputs,dst=$ROOT/outputs'not in text[:text.rfind('fi\ndocker run')]
 subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)


@pytest.fixture
def shell_runtime(tmp_path):
 root=tmp_path/'root';rev='a'*40;code=root/'jobs'/rev/'run_object_pose_smoke/code';code.mkdir(parents=True)
 base=root/'outputs/episode_000009';base.mkdir(parents=True);lock=root/'jobs/.world-reward-h100.lock';lock.write_text('unchanged original lock')
 path=code/'infra/run_object_pose_smoke.sh';path.parent.mkdir()
 # Source validator is tested above; the shell spy never executes a native method.
 text=WRAPPER.read_text();start=text.index('  timeout --signal=TERM --kill-after=5s 300s python3');stop=text.index('\nPYSOLID',start)+len('\nPYSOLID')
 text=text[:start]+'  "$FAKE_HOST" "$1"\n'+text[stop:];path.write_text(text.replace('/srv/scenesmith/world-reward',str(root)))
 bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'calls.jsonl';control=root/'results'/f'object-pose-solid-000009-{rev}'
 def spy(name,body):
  p=bindir/name;p.write_text('#!'+sys.executable+'\nimport json,os,pathlib,sys\nwith open(os.environ["FAKE_LOG"],"a")as h:h.write(json.dumps([pathlib.Path(sys.argv[0]).name,*sys.argv[1:]])+"\\n")\n'+body);p.chmod(0o755);return p
 host=spy('host',r'''mode=sys.argv[1];root=pathlib.Path(os.environ['WR_ROOT']);rev=os.environ['WR_CODE_REVISION'];profile=os.environ.get('FAKE_GEOMETRY_PROFILE','solid');control=root/'results'/('object-pose-'+profile+'-000009-'+rev)
if mode=='before':
 control.mkdir(parents=True);(control/'proof.json').write_text('{}')
if mode=='lock':
 s=(root/'jobs/.world-reward-h100.lock').stat();f=os.fstat(9);assert(f.st_dev,f.st_ino)==(s.st_dev,s.st_ino);print(str(s.st_dev)+':'+str(s.st_ino))
elif mode=='mounts':print(str(pathlib.Path(os.environ['WR_CODE'])/'infra/object_pose_smoke.py'))
else:print('verified-proof')
''')
 spy('uname',"print('Linux')")
 spy('timeout',r'''args=sys.argv[1:]
while args and(args[0].startswith('--')or args[0].endswith('s')):args=args[1:]
os.execvp(args[0],args)
''')
 spy('flock',r'''assert sys.argv[1:]==['--timeout','43200','9'];os.fstat(9);sys.exit(int(os.environ.get('FAKE_FLOCK','0')))
''')
 spy('nvidia-smi',r'''os.fstat(9);print(os.environ.get('FAKE_APPS',''));sys.exit(int(os.environ.get('FAKE_GPU_ERROR','0')))
''')
 spy('chown',r'''assert sys.argv[1]=='1000:1000';assert pathlib.Path(sys.argv[2]).is_dir()
''')
 spy('docker',r'''args=sys.argv[1:]
if args[:2]==['image','inspect']:print(os.environ.get('FAKE_IMAGE','''+repr(IMAGE)+r'''))
elif args[0]=='ps':print(os.environ.get('FAKE_EXISTING',''))
elif args[0]=='run':
 os.fstat(9);assert '--read-only'in args and '1000:1000'in args
 cid=pathlib.Path(args[args.index('--cidfile')+1]);cid.write_text('f'*64)
 sys.exit(int(os.environ.get('FAKE_NATIVE_STATUS','0')))
elif args[0]=='inspect':raise AssertionError('Foreign container must not be inspected/removed in these cases')
elif args[0]=='rm':raise AssertionError('Alreadyremoved native or foreign container must not be removed')
else:raise AssertionError(args)
''')
 env=dict(PATH=str(bindir)+':'+os.environ['PATH'],WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=rev,FAKE_HOST=str(host),FAKE_LOG=str(log),HOME=str(tmp_path))
 def run(args=None):return subprocess.run(['rtk','proxy','bash',str(path),*(args if args is not None else ['--episode','9','--full-video','--mesh-source','solid'])],capture_output=True,text=True,env=env,timeout=10)
 def calls():return[json.loads(row)for row in log.read_text().splitlines()]if log.exists()else[]
 return dict(run=run,calls=calls,env=env,out=base/'object_pose_full_solid',base=base,lock=lock)


def test_shell_exact_native_arguments_resources_fd9_and_unchanged_lock(shell_runtime):
 result=shell_runtime['run']();assert result.returncode==0,result.stderr
 calls=shell_runtime['calls']();native=next(r for r in calls if r[:2]==['docker','run'])
 assert native[-5:]==['--episode','9','--full-video','--mesh-source','solid']
 assert native[native.index('--memory')+1]=='16g'and native[native.index('--cpus')+1]=='4'
 assert 'WR_POSE_OUTPUT_RESERVED=1'in native and '--cap-drop'in native and 'no-new-privileges'in native
 assert [r[1]for r in calls if r[0]=='host'].count('after')==1
 assert shell_runtime['lock'].read_text()=='unchanged original lock'
 assert next(i for i,r in enumerate(calls)if r[0]=='flock')<next(i for i,r in enumerate(calls)if r[0]=='nvidia-smi')<calls.index(native)


def test_surface_shell_same_gpu_policy_unique_output_and_no_query_alias(shell_runtime):
 shell_runtime['env']['FAKE_GEOMETRY_PROFILE']='surface'
 result=shell_runtime['run'](['--episode','9','--full-video','--mesh-source','surface'])
 assert result.returncode==0,result.stderr
 native=next(r for r in shell_runtime['calls']()if r[:2]==['docker','run'])
 assert native[-5:]==['--episode','9','--full-video','--mesh-source','surface']
 assert (shell_runtime['base']/'object_pose_full_surface').exists() and not shell_runtime['out'].exists()
 assert 'wr-object-pose-surface-000009-'+'a'*40 in native
 assert shell_runtime['lock'].read_text()=='unchanged original lock'


@pytest.mark.parametrize('fault,value',[('FAKE_FLOCK','1'),('FAKE_APPS','234'),('FAKE_GPU_ERROR','1'),('FAKE_IMAGE','sha256:wrong'),('FAKE_EXISTING','foreign-cid')])
def test_shell_gate_failures_never_reserve_or_infer(shell_runtime,fault,value):
 shell_runtime['env'][fault]=value;result=shell_runtime['run']();assert result.returncode!=0
 assert not shell_runtime['out'].exists()and not any(r[:2]==['docker','run']for r in shell_runtime['calls']())
 assert not any(r[:2]==['docker','rm']for r in shell_runtime['calls']())


@pytest.mark.parametrize('args',[['--mesh-source','solid'],['--episode','09','--full-video','--mesh-source','solid'],['--episode','30','--full-video','--mesh-source','solid'],['--episode','9','--full-video','--mesh-source','solid','--full-video'],['--episode','9','--mesh-source','solid'],['--episode','9','--full-video','--mesh-source=solid'],['--mesh-source','surface'],['--episode','9','--full-video','--mesh-source','surface','--query-requalification'],['--episode','9','--full-video','--mesh-source=surface']])
def test_solid_malformed_arguments_fail_before_external_tools(shell_runtime,args):
 assert shell_runtime['run'](args).returncode==2 and not shell_runtime['calls']()
