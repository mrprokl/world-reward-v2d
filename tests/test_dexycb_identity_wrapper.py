"""Isolated stdlib wrapper/tool spies only; no Azure, Docker, GPU or real data."""
import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = ROOT / 'infra/run_dexycb_identity_infer.sh'


@pytest.fixture
def runtime(tmp_path):
    root = tmp_path / 'root'; revision = 'a' * 40
    code = root / 'jobs' / revision / 'run_dexycb_identity_infer/code'; (code / 'infra').mkdir(parents=True)
    inputs = root / 'validation/dexycb_identity_v1/inputs'; inputs.mkdir(parents=True); (root / 'results').mkdir()
    (code / 'infra/dexycb_identity_infer.py').write_text('actual frozen driver placeholder')
    lock = root / 'jobs/.world-reward-h100.lock'; lock.write_bytes(b'original lock')
    source = WRAPPER.read_text().split("<<'PYWRAPPER'\n", 1)[1].rsplit('\nPYWRAPPER', 1)[0]
    tree = ast.parse(source); tree.body = tree.body[:-1]  # define production functions, omit automatic main/catch
    source = ast.unparse(tree).replace('/srv/scenesmith/world-reward', str(root))
    spy = r'''
fault=os.environ.get('FAULT',''); calls=[]; state={'hosts':0,'running':False}; orig_run=subprocess.run
def fake(args,**kwargs):
 calls.append(args)
 if args[0]=='/usr/bin/python3':
  if args[-1]=='--mounts':
   text=str(code)+'\t'+str(code)+'\n'+str(ROOT/'validation/dexycb_identity_v1/inputs')+'\t'+str(ROOT/'validation/dexycb_identity_v1/inputs')
   if fault=='private_mount':text+='\n'+str(ROOT/'validation/dexycb_identity_v1/report.json')+'\t'+str(ROOT/'validation/dexycb_identity_v1/report.json')
   return subprocess.CompletedProcess(args,0,text.encode())
  state['hosts']+=1
  proof='c'*64 if fault=='source_post'and state['hosts']>=3 else'b'*64
  return subprocess.CompletedProcess(args,9 if fault=='host'else 0,proof.encode())
 if args[0]=='nvidia-smi':return subprocess.CompletedProcess(args,0,b'123\n'if fault=='gpu'else b'')
 if args[0]=='flock':
  assert args==['flock','--nonblock','9'] and os.fstat(9).st_ino==(ROOT/'jobs/.world-reward-h100.lock').stat().st_ino
  return subprocess.CompletedProcess(args,1 if fault=='lock_busy'else 0,b'')
 if args[:2]==['docker','run']:
  out=Path(args[args.index('--cidfile')+1]).parent;cid=out/'.container.cid'
  cid.write_text('d'*64+'\n');state['running']=True
  stage=args[args.index('--stage')+1]
  receipt={'schema':'world-reward-dexycb-identity-infer-v1','stage':stage,'status':'fail'if fault=='native_receipt'else'pass','phase':'complete',
   'producer_revision':rev,'image_id':images[stage],'host_proof_sha256':'b'*64,'original_rehashed_after':True,
   'private_annotations_read':False,'quality_verified':False}
  if fault=='bool_as_integer':receipt['original_rehashed_after']=1
  (out/'report.json').write_text(json.dumps(receipt));(out/'report.json').chmod(0o644 if fault=='receipt_writable'else 0o444)
  if fault=='lock_replace':
   path=ROOT/'jobs/.world-reward-h100.lock';path.rename(path.with_name('retained'));path.write_bytes(b'foreign')
  if fault=='auto_removed':state['running']=False
  if fault=='missing_cid':cid.unlink();state['running']=False
  if fault=='timeout':raise subprocess.TimeoutExpired(args,kwargs['timeout'])
  return subprocess.CompletedProcess(args,7 if fault=='native'else 0,b'')
 if args[:2]==['docker','ps']:
  present=state['running']or fault=='occupied'
  return subprocess.CompletedProcess(args,1 if fault=='cleanup_query'and state['running']else 0,
                                     ('d'*64+'\n').encode()if present else b'')
 if args[:2]==['docker','inspect']:
  stage=control_stage;name='world-reward-dexycb-'+stage+'-'+rev[:12]
  return subprocess.CompletedProcess(args,0,('FOREIGN'if fault=='foreign_container'else images[stage]+'|/'+name+'|'+entry+'|'+rev).encode())
 if args[:3]==['docker','rm','-f']:
  if fault!='cleanup_fail':state['running']=False
  return subprocess.CompletedProcess(args,1 if fault=='cleanup_fail'else 0,b'')
 raise AssertionError('Unexpected tool '+str(args))
subprocess.run=fake
control_stage=os.environ.get('STAGE','masks')
try:run(sys.argv[1:]);status=0
except Exception as exc:print(type(exc).__name__);status=1
Path(os.environ['CALLS']).write_text(json.dumps(calls));sys.exit(status)
'''
    # Procedural CID belongs to local UID, not a fabricated production root claim.
    source = source.replace('s.st_uid == 0', 's.st_uid == os.getuid()')
    script = tmp_path / 'test_control.py'; script.write_text(source + '\n' + spy)
    log = tmp_path / 'calls.json'; env = dict(PATH='/usr/bin:/bin', HOME=str(tmp_path), WR_ROOT=str(root),
                                            WR_CODE=str(code), WR_CODE_REVISION=revision, CALLS=str(log))
    def args(stage='masks'):
        result = ['--stage', stage, '--manifest-sha256', 'b'*64, '--manifest-bytes', '100',
                  '--acquisition-report-sha256', 'c'*64, '--acquisition-report-bytes', '200']
        if stage == 'tracks': result += ['--masks-report-sha256', 'd'*64, '--masks-report-bytes', '300']
        return result
    def run(arguments=None, fault='', stage='masks'):
        actual = dict(env, FAULT=fault, STAGE=stage)
        result = subprocess.run([sys.executable, '-B', str(script), *(args(stage) if arguments is None else arguments)],
                                env=actual, capture_output=True, text=True, timeout=8)
        calls = json.loads(log.read_text()) if log.exists() else []
        return result, calls
    return dict(root=root,code=code,lock=lock,env=env,run=run,args=args,revision=revision)


@pytest.mark.parametrize('stage,image,budget', [('masks','fd268',300),('tracks','ef12',7200)])
def test_native_stages_narrow_readonly_firewall_lock_and_postseal(runtime,stage,image,budget):
    before=runtime['lock'].read_bytes();result,calls=runtime['run'](stage=stage)
    assert result.returncode==0,(result.stdout,result.stderr)
    command=next(c for c in calls if c[:2]==['docker','run'])
    assert command[command.index('--user')+1]=='0:0' and command[command.index('--network')+1]=='none'
    assert command[command.index('--cap-drop')+1]=='ALL' and '--cap-add'not in command
    assert command[command.index('--entrypoint')+1]=='/usr/bin/env' and '--read-only'in command
    assert any(c.startswith('sha256:'+image)for c in command) and 'HF_HUB_OFFLINE=1'in command
    assert '--host-proof'not in command and '--mounts'not in command and command[-len(runtime['args'](stage)):] == runtime['args'](stage)
    assert calls.index(next(c for c in calls if c[0]=='flock'))<calls.index(command)
    assert any(c[:3]==['docker','rm','-f']for c in calls) and runtime['lock'].read_bytes()==before
    report=json.loads(next((runtime['root']/'results').glob('*/report.json')).read_text())
    assert report['status']=='pass' and report['budget_seconds']==budget and report['owned_cleanup_verified']
    assert report['source_rehashed_after'] and not report['quality_verified'] and not report['private_values_read']
    assert report['native_report_identity']['bytes']>0


@pytest.mark.parametrize('arguments', [[],['--help'],['--stage','masks'],['--stage','bad'],
    ['--stage','masks','--stage','tracks'],['--stage','masks','--resume','true']])
def test_missing_duplicate_unknown_args_reject_before_tools(runtime,arguments):
    result,calls=runtime['run'](arguments);assert result.returncode!=0 and calls==[]


@pytest.mark.parametrize('fault', ['host','gpu','lock_busy','occupied','private_mount'])
def test_preflight_guards_never_invoke_native(runtime,fault):
    if fault=='private_mount':(runtime['root']/'validation/dexycb_identity_v1/report.json').write_text('{}')
    result,calls=runtime['run'](fault=fault)
    assert result.returncode!=0 and not any(c[:2]==['docker','run']for c in calls)


@pytest.mark.parametrize('fault', ['native','native_receipt','bool_as_integer','receipt_writable','timeout','source_post',
                                  'lock_replace','foreign_container','cleanup_fail','cleanup_query','missing_cid'])
def test_native_postfail_retains_owned_outputs_and_never_claims_pass(runtime,fault):
    result,calls=runtime['run'](fault=fault);assert result.returncode!=0
    assert len([c for c in calls if c[:2]==['docker','run']])==1
    report=json.loads(next((runtime['root']/'results').glob('*/report.json')).read_text())
    assert report['status']=='fail' and not report['quality_verified']
    assert list((runtime['root']/'validation/dexycb_identity_v1').glob('identity_masks_*/report.json'))
    if fault=='foreign_container':assert not any(c[:3]==['docker','rm','-f']for c in calls)


def test_docker_rm_auto_exit_is_verified_without_removing_another_container(runtime):
    result,calls=runtime['run'](fault='auto_removed');assert result.returncode==0,(result.stdout,result.stderr)
    assert not any(c[:3]==['docker','rm','-f']for c in calls)
    report=json.loads(next((runtime['root']/'results').glob('*/report.json')).read_text())
    assert report['owned_cleanup_verified'] and report['status']=='pass'


@pytest.mark.parametrize('mutation', ['sha','bytes','duplicate','missing_masks','unexpected_masks'])
def test_independent_pin_types_and_stage_roles_are_exact(runtime,mutation):
    args=runtime['args']('tracks'if mutation=='missing_masks'else'masks')
    if mutation=='sha':args[args.index('--manifest-sha256')+1]='G'*64
    elif mutation=='bytes':args[args.index('--manifest-bytes')+1]='0'
    elif mutation=='duplicate':args+=['--manifest-bytes','100']
    elif mutation=='missing_masks':args=args[:-4]
    else:args+=['--masks-report-sha256','d'*64,'--masks-report-bytes','300']
    result,calls=runtime['run'](args);assert result.returncode!=0 and calls==[]


@pytest.mark.parametrize('target', ['prediction','control','lock_alias'])
def test_occupied_targets_and_alias_lock_are_never_modified(runtime,target):
    if target=='lock_alias':
        original=runtime['lock'].with_name('retained');runtime['lock'].rename(original);runtime['lock'].symlink_to(original)
        preserved=original.read_bytes()
    else:
        parent=runtime['root']/('validation/dexycb_identity_v1'if target=='prediction'else'results')
        name=('identity_masks_'if target=='prediction'else'dexycb-identity-masks-')+runtime['revision']
        occupied=parent/name;occupied.mkdir();(occupied/'retained').write_text('existing work')
    result,calls=runtime['run']();assert result.returncode!=0 and not any(c[:2]==['docker','run']for c in calls)
    if target=='lock_alias':assert original.read_bytes()==preserved and runtime['lock'].is_symlink()
    else:assert {p.name for p in occupied.iterdir()}=={'retained'}


def test_shell_syntax_exact_runtime_namespace_and_literal_closure():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    text=WRAPPER.read_text();assert '# Immutable source closure: /infra/dexycb_identity_infer.py'in text
    assert 'world-reward-ncc-h100-02'in text and 'run_dexycb_identity_infer/code'in text
    assert 'exec /usr/bin/env -i'in text and '/usr/bin/python3 -I -B -'in text
    sys.path.insert(0,str(ROOT/'infra'));import azure_job
    files={str(p.relative_to(ROOT)):p.read_bytes()for folder in ('infra','src','configs')for p in(ROOT/folder).rglob('*')
           if p.is_file()and p.suffix in('.py','.sh','.json','.toml')and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=azure_job.runtime_bundle_paths(files,'infra/run_dexycb_identity_infer.sh')
    assert {'infra/dexycb_identity_infer.py','infra/run_dexycb_identity_infer.sh','infra/bridge_frontend_bindings.py',
            'infra/hand_synthetic_masks.py','infra/robotap_boots_infer.py','src/world_reward/automatic_candidate_bank.py',
            'src/world_reward/relational_motion.py'}<=set(selected)
