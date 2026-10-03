"""Tiny public-reader/CLI wrapper tests; no actual RGB/model/GPU/network."""
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile

import pytest

ROOT=Path(__file__).resolve().parents[1]
WRAPPER=ROOT/'infra/run_tudl_anchor_infer.sh'
IMAGE='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3'
WHEEL_SHA='249bb56bbfd3cdc2a004ea0ff4c2b6ddc84d53bc2194761636eb314d5cfa5dfc'
ASSET_DIRS=('weights/cari4d/hf_home/hub/models--Ruicheng--moge-2-vitl-normal','vendor/research/da3_metric_v1','weights/research/da3_metric_v1')
ASSET_FILES=('weights/cari4d/hf_home/hub/blobs/9f/9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c',
 'results/weights-acquisition.json','results/da3-metric-acquisition-v1.json','vendor/research/da3_dependencies_v1/addict-2.4.0-py3-none-any.whl')


@pytest.fixture
def public():
    spec=importlib.util.spec_from_file_location('anchor_actual_public_helper',ROOT/'infra/tudl_holdout_inputs.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module


@pytest.fixture
def runtime(tmp_path,public):
    root=tmp_path/'runtime';revision='a'*40;code=root/'jobs'/revision/'run_tudl_anchor_infer/code'
    (code/'infra').mkdir(parents=True);(code/'configs').mkdir();(code/'src/world_reward').mkdir(parents=True)
    (code.parent/'revision').write_text(revision+'\n');(code.parent/'source-sha256').write_text('b'*64+'\n')
    directory=root/'validation/tudl_frame_holdout_v1/inputs';directory.mkdir(parents=True)
    images=[]
    for i,name in enumerate(public.filenames()):
        path=directory/name;path.write_bytes(('tiny original public bytes '+name).encode());path.chmod(0o444)
        images.append(dict(scene_id=i//4+1,frame_id=public.FRAME_IDS[i//4][i%4],file=name,
            sha256=public.identity(path)['sha256'],width=public.WIDTH,height=public.HEIGHT))
    manifest=dict(schema=public.SCHEMA,revision=public.REVISION,license=public.LICENSE,selection=public.SELECTION,images=images)
    (directory/'manifest.json').write_text(json.dumps(manifest));(directory/'manifest.json').chmod(0o444)
    pins=dict(schema=public.PINS_SCHEMA,acquisition_report=public.ACQUISITION,
        public_files={p.name:public.identity(p)for p in directory.iterdir()})
    (code/'configs/tudl_frame_holdout_input_pins.json').write_text(json.dumps(pins))
    (code/'infra/tudl_holdout_inputs.py').write_bytes((ROOT/'infra/tudl_holdout_inputs.py').read_bytes())
    (code/'infra/tudl_anchor_infer.py').write_text('"""Tiny fixture driver not executed."""\n')
    (code/'src/world_reward/__init__.py').write_text('"""Tiny own fixture."""\n')
    for name in ASSET_DIRS:(root/name).mkdir(parents=True)
    for name in ASSET_FILES:
        path=root/name;path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(b'own tiny wheel fixture'.ljust(3832,b' ')if name.endswith('.whl')else b'tiny original asset bytes')
        path.chmod(0o444)
    fixture_wheel_sha=hashlib.sha256((root/ASSET_FILES[-1]).read_bytes()).hexdigest()
    wrapper=code/'infra/run_tudl_anchor_infer.sh'
    # Only copied fixture root/wheel literals differ; no production asset/data
    # values are read, substituted or manufactured in the real source tree.
    wrapper.write_text(WRAPPER.read_text().replace('/srv/scenesmith/world-reward',str(root)).replace(WHEEL_SHA,fixture_wheel_sha))
    for path in(code,*code.rglob('*')):path.chmod(0o555 if path.is_dir()else 0o444)
    private=directory.parent/'eval_private';private.mkdir(mode=0o700);(private/'forbidden').write_text('must never be mounted')
    bin=tmp_path/'bin';bin.mkdir();log=tmp_path/'calls.jsonl'
    scripts={
        'uname':'#!/bin/sh\nprintf "Linux\\n"\n',
        'chown':'#!/bin/sh\nexit 0\n',
        'flock':'''#!/bin/sh
printf '%s\\n' "$*" >> "$FAKE_LOCK_LOG"
if [ "${FAKE_LOCK_BUSY:-0}" = 1 ];then exit 1;fi
''',
        'nvidia-smi':'''#!/bin/sh
if [ "${FAKE_SMI_FAIL:-0}" = 1 ];then exit 8;fi
if [ "${FAKE_GPU_BUSY:-0}" = 1 ];then printf '1234\\n';fi
''',
        'timeout':'''#!/bin/sh
printf '%s\\n' "$*" > "$FAKE_TIMEOUT"
shift 3;exec "$@"
''',
        'docker':f'''#!{sys.executable}
import json,os,pathlib,sys
args=sys.argv[1:]
with open(os.environ['FAKE_LOG'],'a')as h:h.write(json.dumps(args)+'\\n')
if args[:2]==['image','inspect']:
 count=pathlib.Path(os.environ['FAKE_IMAGE_COUNTER']);old=int(count.read_text())if count.exists()else 0;count.write_text(str(old+1))
 print(os.environ['FAKE_IMAGE_AFTER']if old else os.environ['FAKE_IMAGE']);sys.exit(0)
if args[0]!='run':sys.exit(9)
out=pathlib.Path(os.environ['WR_ROOT'])/'validation/tudl_frame_holdout_v1/anchor_predictions_v1'
status=int(os.environ.get('FAKE_STATUS','0'));report=out/'report.json'
report.write_text(json.dumps({{'status':'pass'if status==0 else 'fail'}}));report.chmod(0o400)
if status==0:
 for i in range(12):p=out/('prediction_%02d.npz'%i);p.write_bytes(b'private tiny fixture; not prediction');p.chmod(0o400)
fault=os.environ.get('FAKE_MUTATION','');root=pathlib.Path(os.environ['WR_ROOT']);code=pathlib.Path(os.environ['WR_CODE'])
if fault=='source':p=code/'infra/tudl_anchor_infer.py';p.chmod(0o644);p.write_text('changed');p.chmod(0o444)
elif fault=='revision':(code.parent/'revision').write_text('c'*40+'\\n')
elif fault=='archive':(code.parent/'source-sha256').write_text('d'*64+'\\n')
elif fault in('wheel','public','receipt'):
 relative={{'wheel':'{ASSET_FILES[-1]}','public':'validation/tudl_frame_holdout_v1/inputs/manifest.json','receipt':'results/weights-acquisition.json'}}[fault]
 p=root/relative;p.chmod(0o644);p.write_bytes(p.read_bytes()+b'changed');p.chmod(0o444)
sys.exit(status)
''',
    }
    for name,source in scripts.items():path=bin/name;path.write_text(source);path.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION=revision,FAKE_IMAGE=IMAGE,FAKE_IMAGE_AFTER=IMAGE,
        FAKE_LOG=str(log),FAKE_TIMEOUT=str(tmp_path/'timeout'),FAKE_LOCK_LOG=str(tmp_path/'lock'),
        FAKE_IMAGE_COUNTER=str(tmp_path/'image.count'),PATH=str(bin)+':'+os.environ['PATH'])
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=8)
    return root,code,directory,pins,env,log,wrapper,run


@pytest.mark.parametrize('status',[0,7,124,137])
def test_public13_exact_assets_and_only_private_output_rw_gpu_runtime_status(runtime,status):
    root,code,directory,pins,env,log,wrapper,run=runtime;env['FAKE_STATUS']=str(status)
    before={name:(directory/name).read_bytes()for name in pins['public_files']}
    result=run();assert result.returncode==status,result.stderr
    rows=[json.loads(line)for line in log.read_text().splitlines()];assert len(rows)==3
    assert rows[0]==rows[-1]==['image','inspect',IMAGE,'--format','{{.Id}}']
    args=rows[1];out=directory.parent/'anchor_predictions_v1'
    assert args[-3:]==[IMAGE,'-B',str(code/'infra/tudl_anchor_infer.py')]
    assert args[args.index('--gpus')+1]=='all'and args[args.index('--network')+1]=='none'
    assert args[args.index('--memory')+1]=='32g'and args[args.index('--cpus')+1]=='4'and args[args.index('--user')+1]=='1000'
    mounts=[args[i+1]for i,item in enumerate(args)if item=='--mount']
    expected=[f'type=bind,src={code},dst={code},readonly',f'type=bind,src={code.parent}/revision,dst={code.parent}/revision,readonly',
        f'type=bind,src={code.parent}/source-sha256,dst={code.parent}/source-sha256,readonly']
    expected+=[f'type=bind,src={directory/name},dst={directory/name},readonly'for name in('manifest.json',*pins['public_files'].keys()-{'manifest.json'})]
    assert len(mounts)==24 and set(mounts)==set(expected+[f'type=bind,src={root/name},dst={root/name},readonly'for name in ASSET_DIRS+ASSET_FILES]+[f'type=bind,src={out},dst={out}'])
    assert [m for m in mounts if not m.endswith(',readonly')]==[f'type=bind,src={out},dst={out}']
    assert all('eval_private'not in m and '/data/'not in m and '/weights/mhr/'not in m for m in mounts)
    values=[args[i+1]for i,item in enumerate(args)if item=='--env']
    assert 'HF_HUB_OFFLINE=1'in values and 'TRANSFORMERS_OFFLINE=1'in values and f'PYTHONPATH={code}/src:{code}/infra:{root/ASSET_FILES[-1]}'in values
    assert out.stat().st_mode&0o777==0o700 and len(list(out.iterdir()))==(13 if status==0 else 1)
    assert before=={name:(directory/name).read_bytes()for name in pins['public_files']}
    assert Path(env['FAKE_TIMEOUT']).read_text().startswith('--signal=TERM --kill-after=10s 903s docker run')
    assert Path(env['FAKE_LOCK_LOG']).read_text()=='--nonblock 9\n'


@pytest.mark.parametrize('args',[['--episode','2'],['--help'],['--resume'],['--budget','901'],['--root','/tmp']])
def test_no_cli_options_before_any_child(runtime,args):
    root,code,directory,pins,env,log,wrapper,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists()and not(directory.parent/'anchor_predictions_v1').exists()


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_private_output_absent_no_overwrite_cleanup(runtime,kind):
    root,code,directory,pins,env,log,wrapper,run=runtime;out=directory.parent/'anchor_predictions_v1'
    if kind=='directory':out.mkdir();(out/'retained').write_text('frozen')
    elif kind=='file':out.write_text('frozen')
    else:out.symlink_to('missing')
    result=run();assert result.returncode!=0 and not log.exists()and(out.exists()or out.is_symlink())


@pytest.mark.parametrize('mode',['GPU_BUSY','LOCK_BUSY','SMI_FAIL'])
def test_gpu_busy_failclosed_before_reservation(runtime,mode):
    root,code,directory,pins,env,log,wrapper,run=runtime;env['FAKE_'+mode]='1'
    result=run();assert result.returncode!=0 and not(directory.parent/'anchor_predictions_v1').exists()
    assert all(json.loads(line)[0]=='image'for line in log.read_text().splitlines())


@pytest.mark.parametrize('name',ASSET_DIRS+ASSET_FILES)
def test_every_exact_asset_path_required_before_docker(runtime,name):
    root,code,directory,pins,env,log,wrapper,run=runtime;path=root/name
    if path.is_dir():path.rmdir()
    else:path.unlink()
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('fault',['wheelbytes','wheelsha','publichash','publicextra','publicsymlink','pinsmissing','pinsschema','driver',
    'codewritable','codesymlink','revision','archive','root','namespace','locksymlink','entrypoint'])
def test_source_public_pins_wheel_and_namespace_fail_before_docker(runtime,fault,tmp_path):
    root,code,directory,pins,env,log,wrapper,run=runtime
    if fault.startswith('wheel'):
        path=root/ASSET_FILES[-1];path.chmod(0o644);data=path.read_bytes();path.write_bytes(data+b'x'if fault=='wheelbytes'else b'x'+data[1:]);path.chmod(0o444)
    elif fault=='publichash':path=directory/'manifest.json';path.chmod(0o644);path.write_text('changed');path.chmod(0o444)
    elif fault=='publicextra':(directory/'forbidden').write_text('extra')
    elif fault=='publicsymlink':path=directory/'manifest.json';path.unlink();path.symlink_to('missing')
    elif fault.startswith('pins'):
        path=code/'configs/tudl_frame_holdout_input_pins.json';path.parent.chmod(0o755)
        if fault=='pinsmissing':path.unlink()
        else:path.chmod(0o644);pins['schema']='wrong';path.write_text(json.dumps(pins));path.chmod(0o444)
    elif fault in('driver','codewritable','codesymlink'):
        path=code/'infra/tudl_anchor_infer.py';path.parent.chmod(0o755)
        if fault=='driver':path.unlink()
        elif fault=='codewritable':path.chmod(0o644)
        else:path.unlink();path.symlink_to('tudl_holdout_inputs.py')
    elif fault=='revision':(code.parent/'revision').write_text('c'*40+'\n')
    elif fault=='archive':(code.parent/'source-sha256').write_text('bad')
    elif fault=='root':env['WR_ROOT']=str(tmp_path/'wrong')
    elif fault=='namespace':env['WR_CODE']=str(tmp_path/'wrong')
    elif fault=='locksymlink':(root/'jobs/.world-reward-h100.lock').symlink_to('missing')
    else:
        outside=tmp_path/'outside.sh';outside.write_text(wrapper.read_text());result=subprocess.run(['rtk','proxy','bash',str(outside)],env=env,capture_output=True,text=True)
        assert result.returncode!=0 and not log.exists();return
    result=run();assert result.returncode!=0 and not log.exists()


@pytest.mark.parametrize('image',['sha256:'+'0'*64,'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7',''])
def test_wrong_vm02_image_fail_before_reservation(runtime,image):
    root,code,directory,pins,env,log,wrapper,run=runtime;env['FAKE_IMAGE']=image
    result=run();assert result.returncode!=0 and not(directory.parent/'anchor_predictions_v1').exists()


@pytest.mark.parametrize('mutation',['source','revision','archive','wheel','public','receipt','image'])
def test_post_original_sources_publicbytes_wheel_markers_image_rehashed(runtime,mutation):
    root,code,directory,pins,env,log,wrapper,run=runtime
    if mutation=='image':env['FAKE_IMAGE_AFTER']='sha256:'+'0'*64
    else:env['FAKE_MUTATION']=mutation
    result=run();assert result.returncode!=0 and(directory.parent/'anchor_predictions_v1/report.json').is_file()


def test_actual_bash_and_runtime_static_closure_encoded_budget():
    subprocess.run(['rtk','proxy','bash','-n',str(WRAPPER)],check=True)
    source=WRAPPER.read_text();assert '903s'in source and 'flock --nonblock 9'in source and WHEEL_SHA in source
    assert 'src=$BASE/inputs,dst='not in source and 'src=$BASE,dst='not in source and 'src=$ROOT/validation,dst='not in source
    assert not re.search(r'(?m)^\s*(?:rm|rmdir)\s',source)
    spec=importlib.util.spec_from_file_location('anchor_archive_actual',ROOT/'infra/azure_job.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    files={str(p.relative_to(ROOT)):p.read_bytes()for base in('infra','src','configs')for p in(ROOT/base).rglob('*')if p.is_file()and '__pycache__'not in p.parts}
    files['pyproject.toml']=(ROOT/'pyproject.toml').read_bytes()
    files.setdefault('infra/tudl_anchor_infer.py',b'"""Own pure fixture driver."""\nimport tudl_holdout_inputs\n')
    selected=module.runtime_bundle_paths(files,'infra/run_tudl_anchor_infer.sh')
    assert {'infra/run_tudl_anchor_infer.sh','infra/tudl_anchor_infer.py','infra/tudl_holdout_inputs.py','configs/tudl_frame_holdout_input_pins.json'}<=set(selected)
    stream=io.BytesIO()
    with tarfile.open(fileobj=stream,mode='w')as archive:
        for name in selected:
            info=tarfile.TarInfo(name);info.size=len(files[name]);archive.addfile(info,io.BytesIO(files[name]))
    encoded,_=module.encoded_runtime_archive(stream.getvalue());assert len(encoded)<=module.MAX_CODE_CONTROL_BYTES
