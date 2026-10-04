"""Tiny opaque inputs and control spies; no network, Docker, GPU or real labels."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest

REPO=Path(__file__).resolve().parents[1]
WRAPPER=REPO/'infra/run_dexycb_identity_calibrate.sh'


@pytest.fixture
def runtime(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('cpu_wrapper_public_fixture',REPO/'tests/test_dexycb_identity_calibrate.py')
    tests=importlib.util.module_from_spec(spec);spec.loader.exec_module(tests)
    root,code,revision,pins=tests.fixture(tmp_path,monkeypatch)
    (code/'infra').chmod(0o755)
    tests.seal(code/'infra/run_dexycb_identity_calibrate.sh',WRAPPER.read_bytes())
    (code/'infra').chmod(0o555)
    source=WRAPPER.read_text().split("<<'PYWRAPPER'\n",1)[1].rsplit('\nPYWRAPPER',1)[0]
    tree=ast.parse(source);tree.body=tree.body[:-1]
    monkeypatch.setenv('WR_CODE',str(code));monkeypatch.setenv('WR_CODE_REVISION',revision)
    namespace={};exec(compile(ast.unparse(tree).replace('/srv/scenesmith/world-reward',str(root)),'CPU-wrapper-fixture','exec'),namespace)
    infer=namespace['infer'];monkeypatch.setattr(infer,'frontend_proof',lambda *a,**k:({}, {'manufactured_model_proof':'no models executed in this fixture'}))
    monkeypatch.setattr(infer,'boots_proof',lambda *a,**k:{'manufactured_model_proof':'no models executed in this fixture'})
    monkeypatch.setattr(infer,'host_mounts',lambda *a:[code,code.parent/'revision',code.parent/'source-sha256',root/infer.BASE/'inputs'])
    acquisition={'retained_files':{},'producer_revision':'e'*40,'source_before':{}}
    acquired=root/'jobs'/acquisition['producer_revision']/'run_dexycb_acquire/code'
    tests.seal(acquired/'infra/dexycb_acquire.py',b'opaque source proof')
    (root/infer.BASE/'report.json').chmod(0o644)
    pins['acquisition']=tests.write_json(root/infer.BASE/'report.json',acquisition)
    (root/'results').mkdir(exist_ok=True)
    namespace['proof_original']=namespace['proof']
    # Marker/source helpers are real readonly procedural files; native controls
    # below are explicit spies, not source or performance provenance claims.
    return dict(root=root,code=code,revision=revision,pins=pins,ns=namespace,tests=tests,acquisition=acquisition)


def arguments(runtime,stage='public_features'):
    values=['--stage',stage]
    for name,pin in runtime['pins'].items():
        for key,value in pin.items():values+=['--'+name+'-'+key.replace('_','-'),str(value)]
    return values


def make_private(runtime):
    t,ns,pins=runtime['tests'],runtime['ns'],runtime['pins']
    result=t.gate.run('public_features',runtime['code'],runtime['revision'],pins)
    folder=runtime['root']/t.gate.BASE/t.gate.FOLDERS['public_features']
    pins['features']=dict(**t.gate.binding.identity(folder/'report.json'),producer_revision=runtime['revision'],script_sha256=result['script_sha256'])
    clips=ns['infer'].public_inputs(runtime['root']/t.gate.BASE/'inputs',pins['manifest'])[0]
    acquired=runtime['acquisition']
    for clip in clips:
        prefix=clip['subject']+'/'+clip['sequence']
        for suffix in('/meta.yml','/'+ns['infer'].CAMERA+'/labels_000000.npz'):
            name=prefix+suffix;path=runtime['root']/t.gate.BASE/'eval_private'/name
            acquired['retained_files'][name]=t.seal(path,b'opaque initial private annotation; never decoded by host')
    (runtime['root']/t.gate.BASE/'report.json').chmod(0o644)
    pins['acquisition']=t.write_json(runtime['root']/t.gate.BASE/'report.json',acquired)
    # The public feature receipt was made with the original acquisition receipt;
    # set its independent pins coherently before any simulated private evaluation.
    receipt=json.loads((folder/'report.json').read_bytes());receipt['pins']['acquisition']=pins['acquisition']
    (folder/'report.json').chmod(0o644);pins['features'].update(t.write_json(folder/'report.json',receipt))


def install_controls(runtime,monkeypatch,fault=''):
    ns=runtime['ns'];calls=[];state={'live':False};actual_proof=ns['proof']
    if fault=='post_source':
        count={'n':0}
        def changed(stage,pins):
            value,paths=actual_proof(stage,pins);count['n']+=1
            return ('f'*64 if count['n']>1 else value),paths
        ns['proof']=changed
    def fake(args,**kwargs):
        calls.append((args,kwargs))
        if args[:2]==['docker','ps']:
            return subprocess.CompletedProcess(args,0,('d'*64+'\n').encode()if state['live']or fault=='occupied'else b'')
        if args[:2]==['docker','run']:
            assert kwargs['timeout']==150
            cid=Path(args[args.index('--cidfile')+1]);cid.write_text('d'*64+'\n');state['live']=fault!='auto_removed'
            stage=args[args.index('--stage')+1];out=runtime['root']/ns['infer'].BASE/ns['FOLDERS'][stage]
            assert out.is_dir()and list(out.iterdir())==[]
            receipt=dict(stage=stage,status='pass',phase='complete',producer_revision=runtime['revision'],pins=runtime['pins'],
                original_rehashed_after=True,device='cpu',network='none',budget_seconds=120,adoption=False,
                private_annotations_read=stage=='private_calibration')
            own=ns['snapshot'](runtime['code'],runtime['revision'],ns['ENTRY'],ns['HELPERS'])
            receipt.update(source_binding=own,script_sha256=own['helpers']['infra/dexycb_identity_calibrate.py']['sha256'])
            if fault=='native_receipt':receipt['original_rehashed_after']=1
            if fault=='native_source':receipt['source_binding']['closure_sha256']='0'*64
            runtime['tests'].write_json(out/'report.json',receipt)
            if fault=='timeout':raise subprocess.TimeoutExpired(args,150)
            if fault=='interrupt':ns['signal'].getsignal(ns['signal'].SIGTERM)(None,None)
            return subprocess.CompletedProcess(args,7 if fault=='native'else 0,b'')
        if args[:2]==['docker','inspect']:
            name='world-reward-dexycb-cpu-'+('private-calibration'if 'features'in runtime['pins']else'public-features')+'-'+runtime['revision'][:12]
            text='foreign'if fault=='foreign'else ns['IMAGE']+'|/'+name+'|'+ns['ENTRY']+'|'+runtime['revision']
            return subprocess.CompletedProcess(args,0,text.encode())
        if args[:3]==['docker','rm','-f']:
            if fault!='cleanup':state['live']=False
            return subprocess.CompletedProcess(args,1 if fault=='cleanup'else 0,b'')
        raise AssertionError('Unexpected control operation')
    monkeypatch.setattr(ns['subprocess'],'run',fake)
    return calls


@pytest.mark.parametrize('stage',['public_features','private_calibration'])
def test_real_metadata_proof_and_leaf_only_private_whitelist(runtime,stage):
    if stage=='private_calibration':make_private(runtime)
    digest,paths=runtime['ns']['proof'](stage,runtime['pins']);assert len(digest)==64
    base=runtime['root']/runtime['ns']['infer'].BASE
    private=[p for p in paths if p.is_relative_to(base/'eval_private')]
    assert len(private)==(24 if stage=='private_calibration'else 0)and base/'eval_private'not in paths
    assert all(p.name in('meta.yml','labels_000000.npz')for p in private)
    assert base/'report.json'in paths and runtime['code']in paths
    assert not any(p==runtime['root']/'vendor'or p.is_relative_to(runtime['root']/'data')for p in paths)


@pytest.mark.parametrize('stage',['public_features','private_calibration'])
def test_cpu_only_native_output_reservation_and_host_postseal(runtime,monkeypatch,stage):
    if stage=='private_calibration':make_private(runtime)
    calls=install_controls(runtime,monkeypatch);runtime['ns']['run'](arguments(runtime,stage))
    command,kwargs=next(c for c in calls if c[0][:2]==['docker','run'])
    assert '--gpus'not in command and not any(c[0][0]in('flock','nvidia-smi')for c in calls)
    assert 'CUDA_VISIBLE_DEVICES=-1'in command and 'WR_DEXYCB_CPU_OUTPUT_RESERVED=1'in command
    assert command[command.index('--entrypoint')+1]=='/usr/bin/env'and command[command.index('--network')+1]=='none'
    assert command[command.index('--user')+1]=='0:0'and command[command.index('--cap-drop')+1]=='ALL'
    if stage=='private_calibration':assert command[command.index('--cap-add')+1]=='DAC_READ_SEARCH'
    else:assert '--cap-add'not in command
    assert kwargs['timeout']==150 and '--read-only'in command and command[-len(arguments(runtime,stage)):]==arguments(runtime,stage)
    cid=Path(command[command.index('--cidfile')+1]);assert cid.parent.parent==runtime['root']/'results'
    report=json.loads((cid.parent/'report.json').read_bytes())
    assert report['status']=='pass'and report['owned_cleanup_verified']and report['source_rehashed_after']
    assert report['gpu_used']is False and report['private_values_read']is False and report['quality_verified']is False
    assert report['read_capability']==('DAC_READ_SEARCH'if stage=='private_calibration'else None)
    assert report['read_capability_basis']==('readonly_external_initial_annotations_acquired_as_UID1000_mode0400'if stage=='private_calibration'else None)
    assert cid.stat().st_mode&0o777==0o400


@pytest.mark.parametrize('fault',['native','native_receipt','native_source','timeout','interrupt','post_source','foreign','cleanup'])
def test_native_and_postfail_are_retained_without_claim_or_foreign_cleanup(runtime,monkeypatch,fault):
    calls=install_controls(runtime,monkeypatch,fault)
    with pytest.raises(ValueError):runtime['ns']['run'](arguments(runtime))
    assert len([c for c in calls if c[0][:2]==['docker','run']])==1
    receipt=json.loads(next((runtime['root']/'results').glob('*/report.json')).read_bytes())
    assert receipt['status']=='fail'and not receipt['quality_verified']
    if fault=='foreign':assert not any(c[0][:3]==['docker','rm','-f']for c in calls)


def test_already_auto_removed_container_has_cleanup_receipt_without_extra_kill(runtime,monkeypatch):
    calls=install_controls(runtime,monkeypatch,'auto_removed');runtime['ns']['run'](arguments(runtime))
    assert not any(c[0][:3]==['docker','rm','-f']for c in calls)
    receipt=json.loads(next((runtime['root']/'results').glob('*/report.json')).read_bytes())
    assert receipt['owned_cleanup_verified']and receipt['status']=='pass'


@pytest.mark.parametrize('mutation',['empty','duplicate','stage','bytes','sha','missing_features','extra_features','different_native_revision','different_native_script'])
def test_arguments_fail_closed_before_any_control(runtime,monkeypatch,mutation):
    values=arguments(runtime)
    if mutation=='empty':values=[]
    elif mutation=='duplicate':values+=['--stage','public_features']
    elif mutation=='stage':values[1]='tracks'
    elif mutation=='bytes':values[values.index('--manifest-bytes')+1]='0'
    elif mutation=='sha':values[values.index('--manifest-sha256')+1]='G'*64
    elif mutation=='missing_features':values[1]='private_calibration'
    elif mutation=='different_native_revision':values[values.index('--tracks-producer-revision')+1]='d'*40
    elif mutation=='different_native_script':values[values.index('--tracks-script-sha256')+1]='d'*64
    else:values+=['--features-bytes','1','--features-sha256','a'*64,'--features-producer-revision','b'*40,'--features-script-sha256','c'*64]
    calls=install_controls(runtime,monkeypatch)
    with pytest.raises(ValueError):runtime['ns']['run'](values)
    assert calls==[]


@pytest.mark.parametrize('fault',['source','marker','array','private','occupied','existing'])
def test_preflight_actual_metadata_mutations_never_start_native(runtime,monkeypatch,fault):
    if fault=='private':make_private(runtime);path=next((runtime['root']/runtime['ns']['infer'].BASE/'eval_private').rglob('labels_000000.npz'))
    elif fault=='source':path=runtime['code']/'infra/dexycb_identity_infer.py'
    elif fault=='marker':path=runtime['code'].parent/'revision'
    elif fault=='array':path=next((runtime['root']/runtime['ns']['infer'].BASE).glob('identity_masks_*/clip_000.npz'))
    elif fault=='existing':
        path=runtime['root']/runtime['ns']['infer'].BASE/'public_features_v1';path.mkdir();(path/'user.txt').write_text('retain')
    if fault in('source','marker','array','private'):path.chmod(0o644);path.write_bytes(b'tampered');path.chmod(0o444)
    calls=install_controls(runtime,monkeypatch,'occupied'if fault=='occupied'else'')
    with pytest.raises(ValueError):runtime['ns']['run'](arguments(runtime,'private_calibration'if fault=='private'else'public_features'))
    assert not any(c[0][:2]==['docker','run']for c in calls)
    if fault=='existing':assert (path/'user.txt').read_text()=='retain'


def test_shell_syntax_sanitized_environment_and_actual_source_closure():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    text=WRAPPER.read_text();assert '/infra/dexycb_identity_calibrate.py'in text
    assert 'world-reward-ncc-h100-02'in text and 'exec /usr/bin/env -i'in text
    assert 'flock'not in text and 'nvidia-smi'not in text and "'--gpus'"not in text
    sys.path.insert(0,str(REPO/'infra'));import azure_job
    files={str(p.relative_to(REPO)):p.read_bytes()for folder in('infra','src','configs')for p in(REPO/folder).rglob('*')
           if p.is_file()and p.suffix in('.py','.sh','.json','.toml')and '__pycache__'not in p.parts}
    files['pyproject.toml']=(REPO/'pyproject.toml').read_bytes()
    selected=set(azure_job.runtime_bundle_paths(files,'infra/run_dexycb_identity_calibrate.sh'))
    assert {'infra/dexycb_identity_calibrate.py','infra/dexycb_identity_infer.py',
            'src/world_reward/identity_calibration.py','src/world_reward/relational_motion.py'}<=selected
