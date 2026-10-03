"""Author-only wrapper isolation with tiny fake CLI commands, never geometry."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_author_human_feasibility.sh'
IMAGE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
HELPERS=('infra/author_human_feasibility.py','src/world_reward/author_human_field.py',
         'src/world_reward/cross_surface.py','src/world_reward/__init__.py')


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';revision='a'*40
    code=root/'jobs'/revision/'run_author_human_feasibility/code';(code/'infra').mkdir(parents=True)
    (root/'validation').mkdir();(code.parent/'revision').write_text(revision+'\n')
    (code.parent/'source-sha256').write_text('b'*64+'\n')
    wrapper=code/'infra/run_author_human_feasibility.sh'
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)))
    for name in HELPERS:
        path=code/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('tiny own source; not executed\n')
    for path in (code,*code.rglob('*')):path.chmod(0o555 if path.is_dir() else 0o444)
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
 image=os.environ['FAKE_IMAGE_AFTER']if old else os.environ['FAKE_IMAGE']
 print(image);sys.exit(0)
if args[0]!='run':sys.exit(9)
out=pathlib.Path(os.environ['WR_ROOT'])/'validation/author_human_feasibility_v1'
status=int(os.environ.get('FAKE_STATUS','0'))
report=out/'report.json';report.write_text(json.dumps({{'status':'pass'if status==0 else 'fail'}}));report.chmod(0o400)
if status==0:
 payload=out/'geometry.npz';payload.write_bytes(b'tiny private fixture; not real geometry');payload.chmod(0o400)
fault=os.environ.get('FAKE_POST_MUTATION','')
code=pathlib.Path(os.environ['WR_CODE'])
if fault=='revision':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='archive':(code.parent/'source-sha256').write_text('d'*64+'\\n')
elif fault=='source':
 path=code/'src/world_reward/author_human_field.py';path.chmod(0o644);path.write_text('changed');path.chmod(0o444)
sys.exit(status)
''');docker.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,
        FAKE_IMAGE=IMAGE,FAKE_IMAGE_AFTER=IMAGE,FAKE_LOG=str(log),FAKE_TIMEOUT=str(tmp_path/'timeout.log'),
        FAKE_IMAGE_COUNTER=str(tmp_path/'image.count'),PATH=str(bin)+':'+os.environ['PATH'])
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return root,code,env,log,wrapper,run


@pytest.mark.parametrize('status',[0,7,124,137])
def test_only_author_source_private_output_exact_cpu_runtime_and_child_status(runtime,status):
    root,code,env,log,wrapper,run=runtime;env['FAKE_STATUS']=str(status)
    source={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    result=run();assert result.returncode==status,result.stderr
    rows=[json.loads(line)for line in log.read_text().splitlines()]
    assert rows[0]==rows[-1]==['image','inspect',IMAGE,'--format','{{.Id}}']and len(rows)==3
    args=rows[1];assert args[-3:]==[IMAGE,'-B',str(code/'infra/author_human_feasibility.py')]
    assert '--gpus'not in args and '--network'in args and args[args.index('--network')+1]=='none'
    assert args[args.index('--memory')+1]=='12g'and args[args.index('--cpus')+1]=='4'
    assert args[args.index('--user')+1]=='1000'and args[args.index('--entrypoint')+1]=='python'
    values=[args[i+1]for i,item in enumerate(args)if item=='--env']
    assert set(values)=={f'WR_ROOT={root}',f'WR_CODE={code}',f'WR_CODE_REVISION={env["WR_CODE_REVISION"]}',
        f'WR_IMAGE_ID={IMAGE}',f'PYTHONPATH={code}/src:{code}/infra','PYTHONDONTWRITEBYTECODE=1',
        'OMP_NUM_THREADS=4','OPENBLAS_NUM_THREADS=4','MKL_NUM_THREADS=4'}
    out=root/'validation/author_human_feasibility_v1'
    mounts=[args[i+1]for i,item in enumerate(args)if item=='--mount']
    assert mounts==[f'type=bind,src={code},dst={code},readonly',
        f'type=bind,src={code.parent}/revision,dst={code.parent}/revision,readonly',
        f'type=bind,src={code.parent}/source-sha256,dst={code.parent}/source-sha256,readonly',
        f'type=bind,src={out},dst={out}']
    assert out.stat().st_mode&0o777==0o700
    assert {p.name for p in out.iterdir()}==({'report.json','geometry.npz'}if status==0 else {'report.json'})
    assert all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())
    assert source=={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    assert Path(env['FAKE_TIMEOUT']).read_text().startswith('--signal=TERM --kill-after=10s 1203s docker run')


@pytest.mark.parametrize('args',[['--episode','2'],['--root','/tmp'],['--resume'],['--budget','1201'],['--adopt'],['--help']])
def test_no_arguments_or_controls_before_output_or_docker(runtime,args):
    root,code,env,log,wrapper,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists()and not any((root/'validation').iterdir())


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_no_overwrite_restart_cleanup_existing_private_output(runtime,kind):
    root,code,env,log,wrapper,run=runtime;out=root/'validation/author_human_feasibility_v1'
    if kind=='directory':out.mkdir();(out/'report.json').write_text('retained')
    elif kind=='file':out.write_text('retained')
    else:out.symlink_to('missing')
    result=run();assert result.returncode!=0 and not log.exists()
    if kind=='directory':assert (out/'report.json').read_text()=='retained'
    elif kind=='file':assert out.read_text()=='retained'
    else:assert out.is_symlink()


@pytest.mark.parametrize('fault',['root','namespace','revisionenv','revisionmarker','archivemarker','markersymlink',
    'entrypoint','codewritable','codesymlink','validationmissing','validationsymlink'])
def test_canonical_dispatch_parent_markers_complete_code_readonly(runtime,fault,tmp_path):
    root,code,env,log,wrapper,run=runtime
    if fault=='root':env['WR_ROOT']=str(tmp_path/'other')
    elif fault=='namespace':env['WR_CODE']=str(tmp_path/'other')
    elif fault=='revisionenv':env['WR_CODE_REVISION']='A'*40
    elif fault=='revisionmarker':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archivemarker':(code.parent/'source-sha256').write_text('bad')
    elif fault=='markersymlink':path=code.parent/'revision';path.unlink();path.symlink_to('missing')
    elif fault=='codewritable':(code/'src/world_reward/author_human_field.py').chmod(0o644)
    elif fault=='codesymlink':path=code/'src/world_reward/author_human_field.py';path.parent.chmod(0o755);path.unlink();path.symlink_to('cross_surface.py')
    elif fault=='validationmissing':(root/'validation').rmdir()
    elif fault=='validationsymlink':(root/'validation').rmdir();(root/'validation').symlink_to(tmp_path)
    else:
        outside=tmp_path/'outside.sh';outside.write_text(wrapper.read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside)],env=env,capture_output=True,text=True,timeout=8)
        assert result.returncode!=0 and not log.exists();return
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('helper',HELPERS)
def test_every_pure_source_helper_required_before_reservation(runtime,helper):
    root,code,env,log,wrapper,run=runtime;path=code/helper;path.parent.chmod(0o755);path.unlink()
    result=run();assert result.returncode!=0 and not log.exists()and not any((root/'validation').iterdir())


@pytest.mark.parametrize('image',['sha256:'+'0'*64,'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7','tag:latest',''])
def test_exact_vm02_image_not_vm01_before_output_even_old_bash(runtime,image):
    root,code,env,log,wrapper,run=runtime;env['FAKE_IMAGE']=image
    result=run();assert result.returncode!=0 and not any((root/'validation').iterdir())
    assert len(log.read_text().splitlines())==1 and 'Exact original VM02 CPU image differs'in result.stderr


@pytest.mark.parametrize('mutation',['revision','archive','source','image'])
def test_after_child_actual_markers_source_and_image_rechecked(runtime,mutation):
    root,code,env,log,wrapper,run=runtime
    if mutation=='image':env['FAKE_IMAGE_AFTER']='sha256:'+'0'*64
    else:env['FAKE_POST_MUTATION']=mutation
    result=run();assert result.returncode!=0
    out=root/'validation/author_human_feasibility_v1'
    assert (out/'report.json').is_file()and (out/'geometry.npz').is_file()
    assert all(p.stat().st_mode&0o777==0o400 for p in out.iterdir())


def test_actual_bash_scope_source_archive_and_no_assets():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text()
    assert '1203s' in source and '--kill-after=10s'in source and 'STATUS=$?'in source
    assert '--gpus'not in source and 'nvidia-smi'not in source and 'flock'not in source
    assert 'src=$ROOT/data'not in source and 'src=$ROOT/vendor'not in source and 'weights'not in source
    assert not re.search(r'(?m)^\s*(?:rm|rmdir)\s',source)and 'systemctl'not in source
    spec=importlib.util.spec_from_file_location('author_wrapper_archive',ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes()for base in ('infra','src','configs')for p in (ROOT/base).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    # Tiny source-only fixture for a concurrently authored driver; no execution,
    # no fabricated runtime dependency or source modification in the workspace.
    for name in HELPERS:
        if name not in files:files[name]=b'"""Own pure fixture source."""\n'
    selected=set(module.runtime_bundle_paths(files,'infra/run_author_human_feasibility.sh'))
    assert selected>={'infra/run_author_human_feasibility.sh',*HELPERS}
    assert 'infra/body_smoke.py'not in selected and 'infra/cari_full_forward.py'not in selected
