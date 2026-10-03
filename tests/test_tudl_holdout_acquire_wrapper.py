"""Tiny host wrapper CLI/control tests, never actual acquisition or data."""
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_tudl_holdout_acquire.sh'


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';revision='a'*40
    code=root/'jobs'/revision/'run_tudl_holdout_acquire/code';(code/'infra').mkdir(parents=True)
    (root/'validation').mkdir();(code.parent/'revision').write_text(revision+'\n')
    (code.parent/'source-sha256').write_text('b'*64+'\n')
    wrapper=code/'infra/run_tudl_holdout_acquire.sh'
    # macOS Bash cannot set RLIMIT_AS. Only this tiny local fixture replaces
    # the Linux ulimit builtin; production keeps its fixed16GiB hard preflight.
    source=WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root))
    source=source.replace('ulimit -v 16777216','printf "16777216\\n" > "$FAKE_ULIMIT"')
    wrapper.write_text(source)
    original=code/'infra/tudl_acquire.py';original.write_text('SENTINEL="tiny original helper"\n')
    driver=code/'infra/tudl_holdout_acquire.py'
    driver.write_text('''import json,os,pathlib,sys
import tudl_acquire
assert tudl_acquire.SENTINEL=="tiny original helper"
assert pathlib.Path(tudl_acquire.__file__)==pathlib.Path(os.environ['WR_CODE'])/'infra/tudl_acquire.py'
assert len(sys.argv)==1 and sys.flags.isolated==1 and sys.dont_write_bytecode
assert os.environ['WR_TUDL_HOLDOUT_RESERVED']=='1'
assert os.geteuid()!=0
out=pathlib.Path(os.environ['WR_ROOT'])/'validation/tudl_frame_holdout_v1'
private=out/'eval_private';private.mkdir(mode=0o700)
report=private/'acquisition-report.json';status=int(os.environ.get('FAKE_STATUS','0'))
report.write_text(json.dumps({'status':'pass'if status==0 else 'fail'}));report.chmod(0o400)
code=pathlib.Path(os.environ['WR_CODE']);fault=os.environ.get('FAKE_MUTATION','')
if fault=='revision':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='archive':(code.parent/'source-sha256').write_text('d'*64+'\\n')
elif fault=='source':
 path=code/'infra/tudl_acquire.py';path.chmod(0o644);path.write_text('changed');path.chmod(0o444)
sys.exit(status)
''')
    for path in (code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    bin=tmp_path/'bin';bin.mkdir();log=tmp_path/'calls.jsonl'
    scripts={
        'timeout':f'''#!{sys.executable}
import json,os,subprocess,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['timeout',*sys.argv[1:]])+'\\n')
sys.exit(subprocess.call(sys.argv[4:]))
''',
        'runuser':f'''#!{sys.executable}
import json,os,subprocess,sys
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(['runuser',*sys.argv[1:]])+'\\n')
assert sys.argv[1:5]==['-u','scenesmith','--','env']
args=sys.argv[5:];env=dict(os.environ)
while args and '='in args[0]:
 key,value=args.pop(0).split('=',1);env[key]=value
assert args[:4]==['python3','-I','-B','-']
sys.exit(subprocess.call([sys.executable,*args[1:]],env=env))
''',
        'chown':'#!/bin/sh\nexit 0\n',
    }
    for name,text in scripts.items():path=bin/name;path.write_text(text);path.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,FAKE_LOG=str(log),
        FAKE_ULIMIT=str(tmp_path/'memory-limit'),PATH=str(bin)+':'+os.environ['PATH'])
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return root,code,env,log,wrapper,run


@pytest.mark.parametrize('status',[0,7,124,137])
def test_host_runuser_control_original_import_noargs_and_status(runtime,status):
    root,code,env,log,wrapper,run=runtime;env['FAKE_STATUS']=str(status)
    before={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    result=run();assert result.returncode==status,result.stderr
    rows=[json.loads(line)for line in log.read_text().splitlines()];assert len(rows)==2
    assert rows[0][:5]==['timeout','--signal=TERM','--kill-after=10s','603s','runuser']
    args=rows[1];assert args[:5]==['runuser','-u','scenesmith','--','env']
    assert args[-5:]==['python3','-I','-B','-',str(code)]
    assert set(args[5:-5])=={f'WR_ROOT={root}',f'WR_CODE={code}',f'WR_CODE_REVISION={env["WR_CODE_REVISION"]}',
        'WR_TUDL_HOLDOUT_RESERVED=1','PYTHONDONTWRITEBYTECODE=1'}
    assert Path(env['FAKE_ULIMIT']).read_text()=='16777216\n'
    out=root/'validation/tudl_frame_holdout_v1';assert out.stat().st_mode&0o777==0o755
    private=out/'eval_private';assert private.stat().st_mode&0o777==0o700
    report=private/'acquisition-report.json';assert report.stat().st_mode&0o777==0o400
    assert json.loads(report.read_text())['status']==('pass'if status==0 else 'fail')
    assert before=={str(p.relative_to(code)):p.read_bytes()for p in code.rglob('*')if p.is_file()}
    assert not any(code.rglob('__pycache__'))


@pytest.mark.parametrize('args',[['--help'],['--episode','2'],['--root','/tmp'],['--resume'],['--budget','601'],['--retry']])
def test_no_cli_controls_before_any_output_or_child(runtime,args):
    root,code,env,log,wrapper,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists()and not any((root/'validation').iterdir())


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_absent_output_even_empty_failed_directory_never_cleaned(runtime,kind):
    root,code,env,log,wrapper,run=runtime;out=root/'validation/tudl_frame_holdout_v1'
    if kind=='directory':out.mkdir();(out/'retained').write_text('frozen')
    elif kind=='file':out.write_text('frozen')
    else:out.symlink_to('missing')
    result=run();assert result.returncode!=0 and not log.exists()
    if kind=='directory':assert (out/'retained').read_text()=='frozen'
    elif kind=='file':assert out.read_text()=='frozen'
    else:assert out.is_symlink()


@pytest.mark.parametrize('fault',['root','namespace','revisionenv','revisionmarker','archivemarker','markersymlink',
    'entrypoint','codewritable','codesymlink','validationmissing','validationsymlink','driver','original'])
def test_actual_source_closure_and_markers_before_reservation(runtime,fault,tmp_path):
    root,code,env,log,wrapper,run=runtime
    if fault=='root':env['WR_ROOT']=str(tmp_path/'other')
    elif fault=='namespace':env['WR_CODE']=str(tmp_path/'other')
    elif fault=='revisionenv':env['WR_CODE_REVISION']='A'*40
    elif fault=='revisionmarker':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archivemarker':(code.parent/'source-sha256').write_text('bad')
    elif fault=='markersymlink':path=code.parent/'revision';path.unlink();path.symlink_to('missing')
    elif fault in ('driver','original','codewritable','codesymlink'):
        path=code/'infra'/('tudl_holdout_acquire.py'if fault=='driver'else 'tudl_acquire.py');path.parent.chmod(0o755)
        if fault=='codewritable':path.chmod(0o644)
        elif fault=='codesymlink':path.unlink();path.symlink_to('tudl_holdout_acquire.py')
        else:path.unlink()
    elif fault=='validationmissing':(root/'validation').rmdir()
    elif fault=='validationsymlink':(root/'validation').rmdir();(root/'validation').symlink_to(tmp_path)
    else:
        outside=tmp_path/'outside.sh';outside.write_text(wrapper.read_text())
        result=subprocess.run(['rtk','proxy','bash',str(outside)],env=env,capture_output=True,text=True,timeout=8)
        assert result.returncode!=0 and not log.exists();return
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('mutation',['revision','archive','source'])
def test_postsource_hash_mutation_fails_without_removing_private_receipt(runtime,mutation):
    root,code,env,log,wrapper,run=runtime;env['FAKE_MUTATION']=mutation
    result=run();assert result.returncode!=0
    report=root/'validation/tudl_frame_holdout_v1/eval_private/acquisition-report.json'
    assert report.is_file()and report.stat().st_mode&0o777==0o400


def test_actual_bash_hostonly_scope_and_source_archive():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text()
    assert 'ulimit -v 16777216'in source and '603s runuser -u scenesmith'in source and 'STATUS=$?'in source
    assert 'sys.path.insert(0,str(code/\'infra\'))'in source and 'runpy.run_path'in source
    for command in ('docker run','--gpus','nvidia-smi','flock','systemctl','pip install'):
        assert command not in source
    assert not re.search(r'(?m)^\s*(?:rm|rmdir)\s',source)
    spec=importlib.util.spec_from_file_location('holdout_wrapper_archive',ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes()for base in ('infra','src','configs')for p in (ROOT/base).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    # Source closure fixture only until the peer's driver exists; never executed
    # or written as a fake production driver.
    files.setdefault('infra/tudl_holdout_acquire.py',b'"""Own driver fixture."""\nimport tudl_acquire\n')
    selected=set(module.runtime_bundle_paths(files,'infra/run_tudl_holdout_acquire.sh'))
    assert selected>={'infra/run_tudl_holdout_acquire.sh','infra/tudl_holdout_acquire.py','infra/tudl_acquire.py'}
    assert 'infra/body_smoke.py'not in selected and 'infra/cari_full_forward.py'not in selected
