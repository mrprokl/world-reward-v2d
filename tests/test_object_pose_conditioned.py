"""Opt-in routing/legacy AST checks only; no tracker, RGB or GPU execution."""
import ast
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from test_object_pose_latent import _default_projection

INFRA=Path(__file__).resolve().parents[1]/'infra'


@pytest.fixture
def pose(monkeypatch):
    monkeypatch.syspath_prepend(str(INFRA))
    spec=importlib.util.spec_from_file_location('wr_pose_conditioned_test',INFRA/'object_pose_smoke.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_conditioned_is_explicit_with_no_default_or_volume_policy_change(pose):
    parser=pose._argument_parser()
    assert parser.parse_args([]).mesh_source=='default'
    assert parser.parse_args(['--mesh-source','volume']).mesh_source=='volume'
    for episode in range(30):
        args=parser.parse_args(['--episode',str(episode),'--full-video','--mesh-source','conditioned'])
        assert args.episode==episode and args.full_video and args.mesh_source=='conditioned'
    with pytest.raises(SystemExit):parser.parse_args(['--conditioned-pins','arbitrary.json'])


def test_conditioned_selected_episode_reaches_same_public_input_gate(pose,monkeypatch):
    monkeypatch.setattr(pose.platform,'system',lambda:'Linux')
    monkeypatch.setattr(pose.Path,'iterdir',lambda _:iter([Path('lo')]))
    monkeypatch.setattr(sys,'argv',['object_pose_smoke.py','--episode','9','--full-video','--mesh-source','conditioned'])
    observed=[]
    def gate(root,*,episode_index):
        observed.append(episode_index);raise RuntimeError('Stop before any heavyweight inputs')
    monkeypatch.setattr(pose,'_validate_inputs',gate)
    with pytest.raises(RuntimeError,match='Stop before'):pose.main()
    assert observed==[9]


def conditioned_branch():
    tree=_default_projection(ast.parse((INFRA/'object_pose_smoke.py').read_text()))
    return next(n for n in ast.walk(tree) if isinstance(n,ast.If)
                and ast.unparse(n.test)=="args.mesh_source == 'conditioned'"
                and any(isinstance(x,ast.ImportFrom) and x.module=='conditioned_geometry_loader' for x in n.body))


def test_new_geometry_branch_only_reads_fixed_pins_and_copies_canonical_glb():
    node=conditioned_branch();text=ast.unparse(ast.Module(body=node.body,type_ignores=[]))
    assert 'conditioned_mesh_{args.episode:06d}_pins.json' in text
    assert 'identity(pin_path)' in text and 'pins=pins' in text
    assert 'shutil.copyfile(qualified_glb, fixed_mesh_path)' in text
    for word in ('budget_mesh(', 'fit_topology', 'normalize_degenerate', 'vertices * scale', 'build-info', 'except'):
        assert word not in text
    source=(INFRA/'object_pose_smoke.py').read_text()
    assert "output.with_name(output.name+'_conditioned')" in source
    assert "if args.mesh_source=='conditioned' and np.any(faces[inactive] != 0):" in source
    assert "if args.mesh_source in ('volume','conditioned','solid'):" in source
    assert 'indices = list(range(inputs["total_frames"]))' in source


def test_conditioned_geometry_branch_no_second_scale_and_full_payload_preserved(tmp_path):
    node=conditioned_branch();canonical=tmp_path/'qualified.glb';canonical.write_bytes(b'canonical-fake')
    canonical.chmod(0o444);out=tmp_path/'new_conditioned';pinsfile=tmp_path/'configs/conditioned_mesh_000009_pins.json'
    pinsfile.parent.mkdir();pinsfile.write_text('{}');pinsfile.chmod(0o444)
    vertices=np.arange(12288,dtype=np.float64).reshape(4096,3);faces=np.zeros((4096,3),np.int64);receipt={}
    class Identity:
        def __call__(self,path):return {'bytes':Path(path).stat().st_size,'sha256':'same'}
    events=[]
    import types
    loader=types.ModuleType('conditioned_geometry_loader')
    loader.identity=Identity();loader.strict_json=lambda _:{}
    def load(*args,**kwargs):
        events.append((args,kwargs));return vertices,faces,np.array([],np.int64),{},canonical,receipt
    loader.load=load
    old=sys.modules.get('conditioned_geometry_loader');sys.modules['conditioned_geometry_loader']=loader
    try:
        env={'Path':Path,'__file__':str(tmp_path/'infra/object_pose_smoke.py'),'args':types.SimpleNamespace(mesh_source='conditioned',episode=9),
            'root':tmp_path,'inputs':{'video_sha256':'video'},'sha256':lambda _:'same','object_report_path':Path('object'),
            'alignment_path':Path('align'),'scale':np.array([.375]*3),'output':out,'fixed_mesh_path':out/'object_fixed_canonical.glb'}
        exec(compile(ast.Module(body=node.body,type_ignores=[]),'<isolated conditioned branch>','exec'),env)
    finally:
        if old is None:sys.modules.pop('conditioned_geometry_loader',None)
        else:sys.modules['conditioned_geometry_loader']=old
    assert env['vertices'] is vertices and env['faces'] is faces
    assert events[0][0][-1]==.375 and events[0][1]=={'pins':{}}
    assert (out/'object_fixed_canonical.glb').read_bytes()==canonical.read_bytes()
    assert receipt['committed_pins_sha256']=='same'


def test_default_volume_and_pose_loop_ast_are_unchanged():
    # Independent archived original source is the last committed parent version.
    # git is not invoked by tests; expected AST digests freeze the known blocks.
    import hashlib
    tree=_default_projection(ast.parse((INFRA/'object_pose_smoke.py').read_text()))
    volume=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=="args.mesh_source == 'volume'"
                and any(isinstance(x,ast.ImportFrom) and x.module=='volume_geometry_loader' for x in n.body))
    default=volume
    while len(default.orelse)==1 and isinstance(default.orelse[0],ast.If):default=default.orelse[0]
    loop=next(n for n in ast.walk(tree) if isinstance(n,ast.For) and isinstance(n.target,ast.Name)
              and n.target.id=='index' and isinstance(n.iter,ast.Name) and n.iter.id=='indices')
    digests=[hashlib.sha256(ast.dump(ast.Module(body=nodes,type_ignores=[]),include_attributes=False).encode()).hexdigest()
             for nodes in (volume.body,default.orelse,[loop])]
    assert digests==LEGACY_AST_SHA256


LEGACY_AST_SHA256=['85d19cd4c3fc0475f08b8035b8ad80548e7b2c3b6930e1e2cc3021c6c4db4778',
                  '91835e16afaff3bec3240dd439ccdb910e3fe2dac21347c58004d8077cc43106',
                  '239b78a2e47592647dea2222abf38565edde76da5ec4833798c5967554538ff9']
