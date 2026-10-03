"""Tiny v2 wrapper CLI tests; original receipt and geometry are fixture-only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_author_human_feasibility_v2.sh'
IMAGE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
ORIGINAL_SHA='eef678c211714894934d6c0715efed2829cd809999c03e0225f59a9d6021f4c0'
HELPERS=('infra/author_human_feasibility_v2.py','infra/author_human_feasibility.py','infra/run_author_human_feasibility.sh',
    'src/world_reward/author_human_field.py','src/world_reward/cross_surface.py','src/world_reward/solid_queries.py','src/world_reward/__init__.py')


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';revision='a'*40
    code=root/'jobs'/revision/'run_author_human_feasibility_v2/code';(code/'infra').mkdir(parents=True)
    original=root/'validation/author_human_feasibility_v1/report.json';original.parent.mkdir(parents=True)
    # An intentionally non-JSON tiny fixture proves the wrapper hashes only.
    # Replace the literal expected SHA only in this copied test wrapper; no
    # original receipt values or production source constants are fabricated.
    original.write_bytes(b'own failed receipt fixture'.ljust(3655,b' '));original.chmod(0o400)
    fixture_sha=hashlib.sha256(original.read_bytes()).hexdigest()
    (code.parent/'revision').write_text(revision+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    wrapper=code/'infra/run_author_human_feasibility_v2.sh'
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)).replace(ORIGINAL_SHA,fixture_sha))
    for name in HELPERS:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('tiny own source; not executed\n')
    for path in (code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    bin=tmp_path/'bin';bin.mkdir();log=tmp_path/'calls.jsonl'
    timeout=bin/'timeout';timeout.write_text('''#!/bin/sh
printf '%s\\n' "$*" > "$FAKE_TIMEOUT"
shift 3;exec "$@"
''');timeout.chmod(0o755)
    chown=bin/'chown';chown.write_text('#!/bin/sh\nexit 0\n');chown.chmod(0o755)
    docker=bin/'docker';docker.write_text(f'''#!{sys.executable}
import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']:
 count=pathlib.Path(os.environ['FAKE_IMAGE_COUNTER'])
 old=int(count.read_text())if count.exists()else 0;count.write_text(str(old+1))
 if os.environ.get('FAKE_IMAGE_FAIL')=='1':sys.exit(8)
 print(os.environ['FAKE_IMAGE_AFTER']if old else os.environ['FAKE_IMAGE']);sys.exit(0)
if args[0]!='run':sys.exit(9)
out=pathlib.Path(os.environ['WR_ROOT'])/'validation/author_human_feasibility_v2'
status=int(os.environ.get('FAKE_STATUS','0'))
report=out/'report.json';report.write_text(json.dumps({{'status':'pass'if status==0 else 'fail'}}));report.chmod(0o400)
if status==0:
 payload=out/'geometry.npz';payload.write_bytes(b'tiny private fixture; not real geometry');payload.chmod(0o400)
fault=os.environ.get('FAKE_POST_MUTATION','');code=pathlib.Path(os.environ['WR_CODE'])
if fault=='revision':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='archive':(code.parent/'source-sha256').write_text('d'*64+'\\n')
elif fault=='source':
 path=code/'src/world_reward/solid_queries.py';path.chmod(0o644);path.write_text('changed');path.chmod(0o444)
elif fault=='original':
 path=pathlib.Path(os.environ['WR_ROOT'])/'validation/author_human_feasibility_v1/report.json'
 path.chmod(0o600);data=path.read_bytes();path.write_bytes(b'x'+data[1:]);path.chmod(0o400)
sys.exit(status)
''');docker.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,
        FAKE_IMAGE=IMAGE,FAKE_IMAGE_AFTER=IMAGE,FAKE_LOG=str(log),FAKE_TIMEOUT=str(tmp_path/'timeout.log'),
        FAKE_IMAGE_COUNTER=str(tmp_path/'image.count'),PATH=str(bin)+':'+os.environ['PATH'])
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return root,code,original,env,log,wrapper,run


@pytest.mark.parametrize('status',[0,7,124,137])
def test_exact_five_mounts_private_v2_original_failure_unchanged_and_status(runtime,status):
    root,code,original,env,log,wrapper,run=runtime;env['FAKE_STATUS']=str(status)
    old=original.read_bytes();source={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    result=run();assert result.returncode==status,result.stderr
    rows=[json.loads(line)for line in log.read_text().splitlines()]
    assert len(rows)==3 and rows[0]==rows[-1]==['image','inspect',IMAGE,'--format','{{.Id}}']
    args=rows[1];assert args[-3:]==[IMAGE,'-B',str(code/'infra/author_human_feasibility_v2.py')]
    assert '--gpus'not in args and args[args.index('--network')+1]=='none'
    assert args[args.index('--memory')+1]=='12g'and args[args.index('--cpus')+1]=='4'
    assert args[args.index('--user')+1]=='1000'and args[args.index('--entrypoint')+1]=='python'
    out=root/'validation/author_human_feasibility_v2'
    mounts=[args[i+1]for i,item in enumerate(args)if item=='--mount']
    assert mounts==[f'type=bind,src={code},dst={code},readonly',
        f'type=bind,src={code.parent}/revision,dst={code.parent}/revision,readonly',
        f'type=bind,src={code.parent}/source-sha256,dst={code.parent}/source-sha256,readonly',
        f'type=bind,src={original},dst={original},readonly',f'type=bind,src={out},dst={out}']
    assert all(str(original.parent)!=item.split(',src=')[1].split(',')[0]for item in mounts)
    assert out.stat().st_mode&0o777==0o700 and original.read_bytes()==old and original.stat().st_mode&0o777==0o400
    assert {p.name for p in out.iterdir()}==({'report.json','geometry.npz'}if status==0 else {'report.json'})
    assert all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())
    assert source=={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    assert Path(env['FAKE_TIMEOUT']).read_text().startswith('--signal=TERM --kill-after=10s 1203s docker run')


@pytest.mark.parametrize('fault',['missing','writable','wrongbytes','wrongsha','symlink'])
def test_exact_original_receipt_before_any_output_or_docker(runtime,fault):
    root,code,original,env,log,wrapper,run=runtime
    if fault=='missing':original.unlink()
    elif fault=='writable':original.chmod(0o600)
    elif fault=='symlink':original.unlink();original.symlink_to('missing')
    else:
        original.chmod(0o600)
        data=original.read_bytes();original.write_bytes(data+b' 'if fault=='wrongbytes'else b'x'+data[1:]);original.chmod(0o400)
    result=run();assert result.returncode!=0 and not log.exists()
    assert not(root/'validation/author_human_feasibility_v2').exists()


@pytest.mark.parametrize('args',[['--episode','2'],['--resume'],['--budget','1201'],['--original','other'],['--help']])
def test_no_cli_overrides(runtime,args):
    root,code,original,env,log,wrapper,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists()and not(root/'validation/author_human_feasibility_v2').exists()


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_existing_v2_output_never_overwritten_or_removed(runtime,kind):
    root,code,original,env,log,wrapper,run=runtime;out=root/'validation/author_human_feasibility_v2'
    if kind=='directory':out.mkdir();(out/'report.json').write_text('retained')
    elif kind=='file':out.write_text('retained')
    else:out.symlink_to('missing')
    result=run();assert result.returncode!=0 and not log.exists()
    assert out.exists()or out.is_symlink()


@pytest.mark.parametrize('helper',HELPERS)
def test_original_and_new_source_closure_required(runtime,helper):
    root,code,original,env,log,wrapper,run=runtime;path=code/helper;path.parent.chmod(0o755);path.unlink()
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('fault',['root','namespace','revisionenv','revisionmarker','archivemarker','codewritable','codesymlink','entrypoint'])
def test_exact_canonical_readonly_dispatch(runtime,fault,tmp_path):
    root,code,original,env,log,wrapper,run=runtime
    if fault=='root':env['WR_ROOT']=str(tmp_path/'wrong')
    elif fault=='namespace':env['WR_CODE']=str(tmp_path/'wrong')
    elif fault=='revisionenv':env['WR_CODE_REVISION']='A'*40
    elif fault=='revisionmarker':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archivemarker':(code.parent/'source-sha256').write_text('bad')
    elif fault=='codewritable':(code/'src/world_reward/solid_queries.py').chmod(0o644)
    elif fault=='codesymlink':path=code/'src/world_reward/solid_queries.py';path.parent.chmod(0o755);path.unlink();path.symlink_to('cross_surface.py')
    else:
        outside=tmp_path/'outside.sh';outside.write_text(wrapper.read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside)],env=env,capture_output=True,text=True,timeout=8)
        assert result.returncode!=0 and not log.exists();return
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('image',['sha256:'+'0'*64,'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7',''])
def test_wrong_image_fails_before_output_portable_bash(runtime,image):
    root,code,original,env,log,wrapper,run=runtime;env['FAKE_IMAGE']=image
    result=run();assert result.returncode!=0 and len(log.read_text().splitlines())==1
    assert not(root/'validation/author_human_feasibility_v2').exists()


@pytest.mark.parametrize('mutation',['revision','archive','source','original','image'])
def test_after_child_original_receipt_source_markers_and_image_verified(runtime,mutation):
    root,code,original,env,log,wrapper,run=runtime
    if mutation=='image':env['FAKE_IMAGE_AFTER']='sha256:'+'0'*64
    else:env['FAKE_POST_MUTATION']=mutation
    result=run();assert result.returncode!=0
    assert(root/'validation/author_human_feasibility_v2/report.json').is_file()


def test_actual_source_exact_receipt_literal_and_no_original_edits_assets_or_cleanup():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text();assert ORIGINAL_SHA in source and 'st_size!=3655'in source
    assert '1203s'in source and '--kill-after=10s'in source and 'STATUS=$?'in source
    assert 'json.loads'not in source and 'json.load'not in source
    for forbidden in ('--gpus','nvidia-smi','flock','systemctl','src=$ROOT/data','src=$ROOT/vendor','weights'):
        assert forbidden not in source
    assert not re.search(r'(?m)^\s*(?:rm|rmdir)\s',source)
    spec=importlib.util.spec_from_file_location('author_v2_wrapper_archive',ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes()for base in('infra','src','configs')for p in(ROOT/base).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    selected=set(module.runtime_bundle_paths(files,'infra/run_author_human_feasibility_v2.sh'))
    assert selected>={'infra/run_author_human_feasibility_v2.sh',*HELPERS}
    assert 'infra/body_smoke.py'not in selected and 'infra/cari_full_forward.py'not in selected
