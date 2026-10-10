"""Data-free opt-in consumers: same tracker, canonical paths and numeric math."""
import ast
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from test_cari_prepare_solid import prepare, source_case
from test_object_pose_conditioned import pose
from test_object_pose_solid_wrapper import runtime, runtime_factory, pin

REPO=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('module_name', ['pose','prepare'])
def test_parser_explicit_single_query_flag_without_free_paths(request,module_name):
    module=request.getfixturevalue(module_name)
    assert module._argument_parser().parse_args([]).query_requalification is False
    args=['--mesh-source','solid','--query-requalification']
    assert module._argument_parser().parse_args(args).query_requalification is True
    for extra in (['--query-requalification'],['--query-requalification=1'],['--pins','elsewhere']):
        with pytest.raises(SystemExit):module._argument_parser().parse_args(args+extra)


@pytest.mark.parametrize('module_name,source', [('pose','default'),('pose','volume'),('pose','conditioned'),('prepare','default')])
def test_non_solid_query_flag_rejected_before_public_input_reads(request,module_name,source,monkeypatch):
    module=request.getfixturevalue(module_name)
    monkeypatch.setattr(module.platform,'system',lambda:'Linux')
    monkeypatch.setattr(module.Path,'iterdir',lambda _:iter([Path('lo')]))
    monkeypatch.setattr(module,'_validate_inputs',lambda *args,**kwargs:(_ for _ in ()).throw(AssertionError('must not read inputs')))
    monkeypatch.setattr(sys,'argv',['driver','--mesh-source',source,'--query-requalification'])
    with pytest.raises(ValueError,match='solid-only'):module.main()


def test_pose_active_loader_bool_and_canonical_output_same_no_second_scale(pose,tmp_path,monkeypatch):
    import solid_geometry_loader as loader
    code=tmp_path/'code';pinpath=code/'configs/solid_mesh_000025_pins.json';pinpath.parent.mkdir(parents=True)
    pinpath.write_text('{"synthetic":true}');pinpath.chmod(0o444)
    root=tmp_path/'root';base=root/'outputs/episode_000025';base.mkdir(parents=True)
    obj=base/'object-report.json';alignment=base/'alignment.json';obj.write_text('{}');alignment.write_text('{}')
    glb=base/'object_budget_solid_balanced_fake/object_fixed_canonical.glb';glb.parent.mkdir();glb.write_bytes(b'opaque canonical')
    glb.chmod(0o444);out=base/'object_pose_full_solid';calls=[]
    monkeypatch.setattr(pose,'__file__',str(code/'infra/object_pose_smoke.py'))
    qr={'original_qem_recompiled':False,'active_query':{'source':'synthetic only'}}
    values=(np.ones((4,3)),np.zeros((4,3),np.int64),np.arange(4),{},glb,{'query_requalification':qr})
    def load(*args,**kwargs):calls.append((args,kwargs));return values
    monkeypatch.setattr(loader,'load',load)
    result=pose._load_solid_mesh(root,25,'a'*64,obj,alignment,.375,out,out/'object_fixed_canonical.glb',query_requalification=True)
    assert result[:4]==values[:4] and (out/'object_fixed_canonical.glb').read_bytes()==glb.read_bytes()
    assert calls[0][1]=={'pins':{'synthetic':True},'query_requalification':True}
    assert calls[0][0][-1]==.375 and result[-1]['query_requalification']==qr
    with pytest.raises(ValueError,match='query profile'):
        pose._load_solid_mesh(root,25,'a'*64,obj,alignment,.375,out,out/'fixed.glb',query_requalification=False)


@pytest.mark.parametrize('fault', ['none','missing_report','missing_topology','mismatch','default_relabel','alternate_path','wrong_bool'])
def test_prepare_full_t_same_pose_namespace_and_cpu_profile_binding(prepare,source_case,monkeypatch,fault):
    import solid_geometry_loader as loader
    c=source_case;qr={'active_query':{'source':'synthetic balanced'},'original_qem_recompiled':False}
    receipt={k:c.report['topology_budget'][k] for k in ('cpu_report_sha256','native_report_sha256','cpu_producer_revision','cpu_script_sha256')}
    receipt['query_requalification']=qr
    c.report['query_requalification']=copy.deepcopy(qr);c.report['topology_budget']['query_requalification']=copy.deepcopy(qr)
    if fault=='missing_report':c.report.pop('query_requalification')
    if fault=='missing_topology':c.report['topology_budget'].pop('query_requalification')
    if fault=='mismatch':c.report['query_requalification']={'unqualified':True}
    c.report_path.chmod(0o644);c.report_path.write_text(json.dumps(c.report));c.report_path.chmod(0o444)
    calls=[]
    def load(*args,**kwargs):calls.append((args,kwargs));return c.v,c.f,np.arange(4),{},c.canonical,receipt
    monkeypatch.setattr(loader,'load',load)
    profile=False if fault=='default_relabel' else 'yes' if fault=='wrong_bool' else True
    path=c.pose.with_name('alternate.npz') if fault=='alternate_path' else c.pose
    if fault!='none':
        with pytest.raises(ValueError):prepare._solid_preflight(c.root,c.episode,c.inputs,c.report,path,np,query_requalification=profile)
    else:
        result=prepare._solid_preflight(c.root,c.episode,c.inputs,c.report,path,np,query_requalification=profile)
        assert np.array_equal(result[0],c.v) and np.array_equal(result[3],c.arrays['rotation'])
        assert calls[0][1]['query_requalification'] is True and not (c.pose.parent.parent/'cari_inputs').exists()


def test_shell_prepare_bool_scope_duplicate_and_dependency_stays_canonical(tmp_path):
    script='''source "$1"; shift; wr_parse_cari_arguments prepare "$@" || exit $?;
wr_require_dependency_report(){ printf '%s\\n' "$1"; }; ROOT=/owned;
wr_cari_dependency prepare; printf '%s %s\\n' "$WR_MESH_SOURCE" "$WR_QUERY_REQUALIFICATION"'''
    path=REPO/'infra/cari_wrapper_common.sh'
    for args in (['--episode','25','--mesh-source','solid','--query-requalification'],
                 ['--query-requalification','--mesh-source','solid','--episode','25']):
        result=subprocess.run(['bash','-c',script,'test',str(path),*args],capture_output=True,text=True)
        assert result.returncode==0,result.stderr
        assert result.stdout=='/owned/outputs/episode_000025/object_pose_full_solid/report.json\nsolid 1\n'
    for args in (['--query-requalification'],['--mesh-source','default','--query-requalification'],
                 ['--mesh-source','solid','--query-requalification','--query-requalification']):
        assert subprocess.run(['bash','-c',script,'test',str(path),*args],capture_output=True).returncode==2
    for mode in ('forward','converter','refine','adapter'):
        text=f'source "$1";wr_parse_cari_arguments {mode} --query-requalification'
        assert subprocess.run(['bash','-c',text,'test',str(path)],capture_output=True).returncode==2


def test_prepare_wrapper_forwards_single_optin_after_canonical_dependency(tmp_path):
    bindir=tmp_path/'bin';bindir.mkdir();log=tmp_path/'docker.args'
    for name,body in [('id','printf 1000'),('docker','printf "%s\\n" "$@" > "$SPY_LOG"')]:
        path=bindir/name;path.write_text('#!/bin/bash\n'+body+'\n');path.chmod(0o755)
    root=tmp_path/'root';dep=root/'outputs/episode_000025/object_pose_full_solid/report.json';dep.parent.mkdir(parents=True);dep.write_text('{}')
    env=dict(PATH=str(bindir)+':'+os.defpath,WR_ROOT=str(root),WR_CODE=str(REPO),WR_CODE_REVISION='a'*40,SPY_LOG=str(log))
    args=['--episode','25','--mesh-source','solid','--query-requalification','--no-wait']
    result=subprocess.run(['bash',str(REPO/'infra/run_cari_prepare.sh'),*args],env=env,capture_output=True)
    assert result.returncode==0,result.stderr
    words=log.read_text().splitlines();assert words[-5:]==['--episode','25','--mesh-source','solid','--query-requalification']
    assert words.count('--query-requalification')==1 and not any('balanced' in w for w in words)


def test_pose_wrapper_bad_bool_routes_cannot_reach_docker():
    wrapper=REPO/'infra/run_object_pose_smoke.sh'
    args=['--episode','25','--full-video','--mesh-source','solid','--query-requalification']
    for bad in (args+['--query-requalification'],['--mesh-source','default','--query-requalification'],
                args[:-2]+['conditioned','--query-requalification'],['--query-requalification'],['--query-requalification=1']):
        result=subprocess.run(['bash',str(wrapper),*bad],capture_output=True,env={'PATH':os.defpath,'WR_CODE':'/unused'})
        assert result.returncode==2,result.stderr


@pytest.fixture
def balanced_host(runtime):
    r=runtime;proposal=r['proposal'];qualification=r['qualification'];base=r['base'];code=r['code']
    proposal.chmod(0o755);new=base/('object_budget_solid_balanced_'+'c'*40);proposal.rename(new);proposal=new
    qualification.chmod(0o755);new=r['root']/'results'/('solid-chart-v2-query-requalify-'+'d'*40);qualification.rename(new);qualification=new
    qr={'active_query':{'source':'synthetic'},'original_qem_recompiled':False}
    report=proposal/'report.json';report.chmod(0o644);report.write_text(json.dumps({'query_requalification':qr}));report.chmod(0o444)
    proposal.chmod(0o555);qualification.chmod(0o555)
    pins=json.loads(r['pins'].read_text())
    old_rows=list(pins['files'].items());pins['files']={}
    for name,value in old_rows:
        newname=name.replace('object_budget_solid_','object_budget_solid_balanced_').replace('solid-chart-v2-qualify-','solid-chart-v2-query-requalify-')
        pins['files'][newname]=pin(r['root']/newname)
    pins['report'].update(pin(proposal/'report.json'))
    r['pins'].chmod(0o644);r['pins'].write_text(json.dumps(pins));r['pins'].chmod(0o444)
    config=code/'configs';config.chmod(0o755)
    for name,value in [('solid_chart_v2_balanced_qualification_pins',{'producer_revision':'d'*40}),('certified_solid_balanced_qualification_pins',{'synthetic':True})]:
        p=config/(name+'.json');p.write_text(json.dumps(value));p.chmod(0o444)
    config.chmod(0o555)
    host=(REPO/'infra/run_object_pose_smoke.sh').read_text().split("<<'PYSOLID'\n",1)[1].split('\nPYSOLID',1)[0]
    host=host.replace("dict(bytes=2031,sha256='42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0')",repr(pin(r['official'])))
    host=host.replace('if os.getuid()!=0 or entry!=','if entry!=')
    def call(mode):
        env=dict(os.environ)
        if (r['control']/'proof.json').exists():env['WR_POSE_PROOF_SHA256']=pin(r['control']/'proof.json')['sha256']
        args=[str(r['root']),str(code),r['rev'],'9',str(r['control']),str(r['out']),str(r['lock']),mode,str(code/'infra/run_object_pose_smoke.sh'),'1']
        return subprocess.run(['rtk','proxy',sys.executable,'-I','-B','-',*args],input=host,text=True,capture_output=True,timeout=10,env=env)
    def complete():
        out=r['out'];out.mkdir();(out/'geometry_and_poses.npz').write_bytes(b'opaque numeric fixture')
        (out/'object_fixed_canonical.glb').write_bytes((proposal/'object_fixed_canonical.glb').read_bytes())
        proof=json.loads((r['control']/'proof.json').read_text())
        value=dict(stage='fixed_scale_full_object_pose_initializer',status='pass',episode_index=9,input_sha256=pins['input_sha256'],
            mesh_source='solid',execution_verified=True,original_frame_coverage_verified=True,fixed_shape=True,ground_truth_used=False,
            hand_labeled_test=False,oracle_modes=[],challenge_performance_verified=False,frames=[{'frame_index':i} for i in range(3)],
            temporal_selection={'candidate_indices':[0,1,2]},geometry_and_poses_sha256=pin(out/'geometry_and_poses.npz')['sha256'],
            fixed_canonical_mesh_sha256=pin(out/'object_fixed_canonical.glb')['sha256'],topology_budget={'committed_pins_sha256':pin(r['pins'])['sha256']},
            script_sha256=proof['selected_source'][str(code/'infra/object_pose_smoke.py')]['sha256'])
        (out/'report.json').write_text(json.dumps(value))
    r['complete']=complete
    return r,call,qr


def test_real_host_balanced_narrow_mounts_and_profile_complete_seal(balanced_host):
    r,call,qr=balanced_host;before=call('before');assert before.returncode==0,before.stderr
    mounts=call('mounts');assert mounts.returncode==0,mounts.stderr
    paths=mounts.stdout.splitlines()
    assert str(r['code']/'configs/solid_chart_v2_balanced_qualification_pins.json') in paths
    assert str(r['code']/'configs/certified_solid_balanced_qualification_pins.json') in paths
    assert not any(p.endswith(('certified_solid_query','mesh_conditioned_chart_v2')) for p in paths)
    assert not any(Path(p).name in ('object_budget_solid.py','solid_chart_v2_qualify.py') for p in paths)
    r['complete']();report=r['out']/'report.json';value=json.loads(report.read_text())
    value['query_requalification']=qr;value['topology_budget']['query_requalification']=qr
    report.write_text(json.dumps(value));complete=call('complete');assert complete.returncode==0,complete.stderr
    assert complete.stdout==before.stdout and r['out'].stat().st_mode&0o777==0o555


@pytest.mark.parametrize('fault',['missing','mismatch','changed_pin'])
def test_real_balanced_host_does_not_accept_untraced_query_or_changed_source(balanced_host,fault):
    r,call,qr=balanced_host;assert call('before').returncode==0
    if fault=='changed_pin':
        p=r['code']/'configs/certified_solid_balanced_qualification_pins.json';p.chmod(0o644);p.write_text('{}');p.chmod(0o444)
        assert call('after').returncode!=0
    else:
        r['complete']();p=r['out']/'report.json';value=json.loads(p.read_text());value['query_requalification']=qr
        if fault=='mismatch':value['topology_budget']['query_requalification']={'different':True}
        p.write_text(json.dumps(value));assert call('complete').returncode!=0


def test_all_numeric_helpers_and_saved_pose_literals_unchanged():
    for name,functions in [('cari_prepare',('_solid_compact','_solid_float32_orientation','_solid_topology_equal','_solid_native_sources','_solid_scene_matrix','_solid_serialized_mesh','_solid_camera_roundtrip'))]:
        path='infra/'+name+'.py';old=subprocess.check_output(['rtk','git','show','HEAD:'+path],cwd=REPO)
        before={n.name:ast.dump(n,include_attributes=False) for n in ast.parse(old).body if isinstance(n,ast.FunctionDef)}
        after={n.name:ast.dump(n,include_attributes=False) for n in ast.parse((REPO/path).read_bytes()).body if isinstance(n,ast.FunctionDef)}
        assert all(before[f]==after[f] for f in functions)
    for name in ('object_pose_smoke','cari_prepare'):
        path='infra/'+name+'.py'
        base='5aecc543f03d0f13a6c8c1efa3de6fd50fc8d9be' if name=='object_pose_smoke' else 'HEAD'
        old=ast.parse(subprocess.check_output(['rtk','git','show',base+':'+path],cwd=REPO));new=ast.parse((REPO/path).read_bytes())
        def numerical(tree):
            if name == 'object_pose_smoke':
                from test_object_pose_latent import _default_projection
                tree = _default_projection(tree)
                # The separately tested latent initializer is opt-in. Freeze
                # EVERY native-default main call, not unrelated new helpers.
                tree = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
            return [ast.dump(n,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Call) and
                isinstance(n.func,ast.Name) and n.func.id in ('select_pose_path','align_observed_points','prepare_mhr_wild_export')]
        assert numerical(old)==numerical(new)
        def result(tree):return [ast.dump(n.value,include_attributes=False) for n in ast.walk(tree) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in n.targets)]
        assert result(old)==result(new)
    for name in ('run_cari_prepare.sh','run_object_pose_smoke.sh','cari_wrapper_common.sh'):
        subprocess.run(['bash','-n',str(REPO/'infra'/name)],check=True)
