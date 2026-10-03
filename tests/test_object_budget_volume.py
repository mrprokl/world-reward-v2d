import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import pytest

@pytest.fixture
def driver(monkeypatch):
    infra=Path(__file__).resolve().parents[1]/'infra';monkeypatch.syspath_prepend(str(infra))
    spec=importlib.util.spec_from_file_location('wr_object_volume',infra/'object_budget_volume.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module);return module

def test_backend_reuses_exact_export_packing_with_own_prerequisites(driver):
    source=Path(driver.__file__).read_text()
    assert 'check_prerequisites=prerequisites,simplify=volume.simplify' in source
    assert 'source_shell_volume_relative_limit' in source and 'adoption_performed' in source
    assert 'object_budget_guarded as proposal' in source
    assert driver._argument_parser().parse_args(['--episode','1']).episode==1

def test_output_only_writable_and_old_failures_readonly(driver):
    wrapper=Path(driver.__file__).with_name('run_object_budget_volume.sh').read_text()
    assert '--gpus' not in wrapper and '903s docker run' in wrapper
    assert 'src=$BASE,dst=$BASE,readonly' in wrapper and 'src=$OUT,dst=$OUT"' in wrapper
    assert 'src=$ROOT/outputs,dst=' not in wrapper
    assert 'mesh_volume_qem.cpp' in wrapper and 'mesh_guarded_qem.cpp' in wrapper


@pytest.mark.parametrize('episode',range(30))
def test_explicit_episode_parser_and_unique_namespace(driver,episode):
    parsed=driver._argument_parser().parse_args(['--episode',str(episode)])
    assert type(parsed.episode) is int and parsed.episode==episode
    assert driver.output_relative(episode)==f'outputs/episode_{episode:06d}/object_budget_volume'


@pytest.mark.parametrize('args',[
    [],['--episode'],['--episode','-1'],['--episode','30'],['--episode','true'],['--episode','False'],
    ['--episode','00'],['--episode','01'],['--episode','+1'],['--episode','1.0'],['--episode',' 1'],
    ['--episode','1 '],['--episode','١'],['--episode','2','--episode','2'],['--episode','0','--episode','29'],
    ['--ep','2'],['--episode','2','--adopt'],['--episode','2','--resume'],['--episode','2','--worker'],
    ['--episode','2','--root','/tmp'],['--episode','2','--budget','901'],['--episode','2','extra'],
])
def test_parser_failclosed_without_runtime(driver,args):
    with pytest.raises(SystemExit) as error:driver._argument_parser().parse_args(args)
    assert error.value.code==2


@pytest.mark.parametrize('episode',[True,False,-1,30,1.,'2',None])
def test_output_identity_does_not_coerce_or_accept_bool(driver,episode):
    with pytest.raises(ValueError):driver.output_relative(episode)


@pytest.mark.parametrize('value',[True,False,2,2.,None])
def test_episode_type_direct_rejects_nonstrings(driver,value):
    with pytest.raises(driver.argparse.ArgumentTypeError):driver._episode(value)


@pytest.fixture
def runtime(tmp_path):
    root=tmp_path/'runtime';(root/'outputs').mkdir(parents=True);(root/'results').mkdir()
    image='sha256:'+'b'*64
    (root/'results/image-volume-qem.json').write_text(json.dumps({'status':'pass','image_id':image}))
    code=tmp_path/'code';code.mkdir();bin=tmp_path/'bin';bin.mkdir();log=tmp_path/'calls.jsonl'
    for name,body in {
        'id':'printf "1000\\n"',
        'chown':'exit 0',
        'timeout':'shift 3; exec "$@"',
    }.items():
        path=bin/name;path.write_text('#!/bin/sh\n'+body+'\n');path.chmod(0o755)
    docker=bin/'docker'
    docker.write_text(f'#!{sys.executable}\nimport json,os,sys\nwith open(os.environ["FAKE_LOG"],"a") as h: h.write(json.dumps(sys.argv[1:])+"\\n")\nsys.exit(int(os.environ.get("FAKE_DOCKER_EXIT","0")))\n')
    docker.chmod(0o755)
    env=dict(os.environ,WR_ROOT=str(root),WR_CODE=str(code),WR_CODE_REVISION='a'*40,
        FAKE_LOG=str(log),PATH=str(bin)+':'+os.environ['PATH'])
    wrapper=Path(__file__).resolve().parents[1]/'infra/run_object_budget_volume.sh'
    def run(*args):return subprocess.run(['rtk','proxy','bash',str(wrapper),*args],env=env,capture_output=True,text=True,timeout=5)
    return root,code,image,env,log,run


@pytest.mark.parametrize('episode',range(30))
def test_wrapper_routes_every_episode_readonly_and_fresh(runtime,episode):
    root,code,image,env,log,run=runtime;base=root/f'outputs/episode_{episode:06d}';base.mkdir()
    failed=base/'object_pose_full';failed.mkdir();(failed/'original-failure.txt').write_text('retained failure')
    old=base/'object_budget_guarded';old.mkdir();(old/'report.json').write_text('retained proposal')
    result=run('--episode',str(episode));assert result.returncode==0,result.stderr
    rows=[json.loads(line) for line in log.read_text().splitlines()];assert len(rows)==1
    args=rows[0];out=base/'object_budget_volume'
    assert out.is_dir() and not any(out.iterdir())
    assert args[-4:]==[image,str(code/'infra/object_budget_volume.py'),'--episode',str(episode)]
    assert '--gpus' not in args and args[args.index('--network')+1]=='none'
    assert args[args.index('--memory')+1]=='16g' and args[args.index('--cpus')+1]=='4'
    mounts=[args[i+1] for i,item in enumerate(args) if item=='--mount']
    assert f'type=bind,src={base},dst={base},readonly' in mounts
    assert f'type=bind,src={out},dst={out}' in mounts
    assert [m for m in mounts if not m.endswith(',readonly')]==[f'type=bind,src={out},dst={out}']
    assert (failed/'original-failure.txt').read_text()=='retained failure'
    assert (old/'report.json').read_text()=='retained proposal'


@pytest.mark.parametrize('args',[
    [],['--episode'],['--episode','-1'],['--episode','30'],['--episode','00'],['--episode','01'],
    ['--episode','true'],['--episode','False'],['--episode','+1'],['--episode','1.0'],['--episode','١'],
    ['--episode',' 1'],['--episode=2'],['--episode','2','--episode','2'],['--episode','2','--adopt'],
    ['--episode','2','--resume'],['--ep','2'],['--episode','2','--budget','901'],
])
def test_wrapper_controls_fail_before_output_or_docker(runtime,args):
    root,code,image,env,log,run=runtime;result=run(*args)
    assert result.returncode==2 and not log.exists() and not any((root/'outputs').iterdir())


@pytest.mark.parametrize('kind',['directory','file','broken_symlink'])
def test_wrapper_preserves_existing_candidate_no_overwrite(runtime,kind):
    root,code,image,env,log,run=runtime;base=root/'outputs/episode_000002';base.mkdir();out=base/'object_budget_volume'
    if kind=='directory':out.mkdir();(out/'report.json').write_text('frozen')
    elif kind=='file':out.write_text('frozen')
    else:out.symlink_to('absent')
    result=run('--episode','2');assert result.returncode!=0 and not log.exists()
    if kind=='directory':assert (out/'report.json').read_text()=='frozen'
    elif kind=='file':assert out.read_text()=='frozen'
    else:assert out.is_symlink()


def test_main_new_episode_dispatch_same_algorithm_and_failed_receipt(driver,tmp_path,monkeypatch):
    out=tmp_path/'outputs/episode_000002/object_budget_volume';out.mkdir(parents=True)
    nets=tmp_path/'interfaces';nets.mkdir();(nets/'lo').mkdir()
    monkeypatch.setattr(driver,'Path',lambda value: nets if str(value)=='/sys/class/net' else Path(value))
    monkeypatch.setattr(driver.platform,'system',lambda:'Linux')
    monkeypatch.setenv('WR_ROOT',str(tmp_path));monkeypatch.setenv('WR_CODE_REVISION','a'*40)
    monkeypatch.setenv('WR_IMAGE_ID','sha256:'+'b'*64)
    alarms=[];monkeypatch.setattr(driver.signal,'signal',lambda *args:None)
    monkeypatch.setattr(driver.signal,'alarm',lambda value:alarms.append(value))
    seen=[]
    def produce(root,episode,report,path,**kwargs):
        seen.append((root,episode,path,kwargs))
        assert report['episode_index']==2 and report['budget_seconds']==900
        assert report['source_shell_volume_relative_limit']==.05
        assert report['target_faces']==report['target_vertices']==4096
        assert report['adoption_performed'] is False and report['challenge_performance_verified'] is False
        assert all(report[k] is False for k in ('components_deleted','holes_filled','normals_repaired','frame_poses_changed'))
        raise ValueError('synthetic prerequisite failure')
    monkeypatch.setattr(driver.proposal,'produce',produce)
    with pytest.raises(ValueError,match='synthetic prerequisite failure'):driver.main(['--episode','2'])
    assert seen==[(tmp_path,2,out/'report.json',{'check_prerequisites':driver.prerequisites,'simplify':driver.volume.simplify})]
    assert alarms==[900,0]
    receipt=json.loads((out/'report.json').read_text())
    assert receipt['status']=='fail' and receipt['episode_index']==2 and receipt['adoption_performed'] is False
    assert receipt['error_type']=='ValueError' and receipt['budget_seconds']==900


def test_actual_wrapper_syntax_and_frozen_resource_algorithm_contract(driver):
    wrapper=Path(driver.__file__).with_name('run_object_budget_volume.sh')
    subprocess.run(['rtk','proxy','bash','-n',str(wrapper)],check=True)
    text=wrapper.read_text();source=Path(driver.__file__).read_text()
    assert "printf -v PADDED '%06d'" in text and 'BASE="$ROOT/outputs/episode_$PADDED"' in text
    assert '--signal=TERM --kill-after=5s 903s' in text
    assert 'encoding' not in source and 'simplify=volume.simplify' in source
    assert 'control.get(\'target_faces\')!=4096' in source and 'control.get(\'target_vertices\')!=4096' in source
    assert "'source_shell_volume_relative_limit':.05" in source and 'signal.alarm(900)' in source
