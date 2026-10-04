"""Episode/full-video planning only; no geometry assets, labels, or CUDA."""

import importlib.util
import ast
from pathlib import Path
import sys

import pytest


@pytest.fixture
def pose(monkeypatch):
    infra = Path(__file__).resolve().parents[1] / "infra"
    monkeypatch.syspath_prepend(str(infra))
    spec = importlib.util.spec_from_file_location("world_reward_test_object_pose_smoke", infra / "object_pose_smoke.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_sparse_mode_and_all_full_episode_choices(pose):
    parsed = pose._argument_parser().parse_args([])
    assert parsed.episode == 15 and parsed.full_video is False
    assert parsed.mesh_source=='default'
    for episode in range(30):
        parsed = pose._argument_parser().parse_args(["--episode", str(episode), "--full-video"])
        assert parsed.episode == episode and parsed.full_video is True


def test_explicit_solid_reservation_only_allows_empty_owned_directory(pose,tmp_path,monkeypatch):
    output=tmp_path/'object_pose_full_solid'
    assert pose._solid_output_reserved(output) is False
    output.mkdir(mode=0o755);output.chmod(0o755)
    monkeypatch.setenv('WR_POSE_OUTPUT_RESERVED','1')
    assert pose._solid_output_reserved(output) is True
    (output/'foreign').write_text('never resume')
    with pytest.raises(ValueError,match='never resume'):pose._solid_output_reserved(output)
    (output/'foreign').unlink();output.chmod(0o700)
    with pytest.raises(ValueError):pose._solid_output_reserved(output)
    output.chmod(0o755);monkeypatch.setenv('WR_POSE_OUTPUT_RESERVED','0')
    with pytest.raises(ValueError,match='explicit'):pose._solid_output_reserved(output)


@pytest.mark.parametrize("episode", ["-1", "30", "15.0", "false", "../track_2", ""])
def test_invalid_episode_fails_before_heavy_imports(pose, episode):
    with pytest.raises(SystemExit): pose._argument_parser().parse_args(["--episode", episode])


def test_selected_episode_reaches_input_gate_before_trimesh_cuda(pose, monkeypatch):
    monkeypatch.setattr(pose.platform, "system", lambda: "Linux")
    monkeypatch.setattr(pose.Path, "iterdir", lambda self: iter([Path("lo")]))
    monkeypatch.setattr(sys, "argv", ["object_pose_smoke.py", "--episode", "0", "--full-video"])
    received = []
    def input_gate(root, *, episode_index):
        received.append(episode_index)
        raise RuntimeError("synthetic stop before geometry imports")
    monkeypatch.setattr(pose, "_validate_inputs", input_gate)
    with pytest.raises(RuntimeError, match="synthetic stop"): pose.main()
    assert received == [0]


def test_dynamic_full_sparse_indices_and_selected_episode_provenance(pose):
    source = Path(pose.__file__).read_text()
    assert 'outputs/episode_000015' not in source
    assert 'indices = list(range(inputs["total_frames"]))' in source
    assert 'indices = inputs["indices"]' in source
    assert '"episode_index": args.episode' in source
    assert 'full_body.get("episode_index") != args.episode' in source


def test_volume_consumer_keeps_arrays_and_existing_native_chain(pose):
    assert pose._argument_parser().parse_args(['--episode','0','--full-video','--mesh-source','volume']).mesh_source=='volume'
    source=Path(pose.__file__).read_text()
    branch=source.split("if args.mesh_source=='volume':",1)[1].split('    else:',1)[0]
    assert 'fit_topology' not in branch and 'vertices * scale' not in branch
    assert 'shutil.copyfile(qualified_glb,fixed_mesh_path)' in branch
    assert 'body_required={**required' in source and 'Automatic object masks must cover every original frame' in source
    chain=Path(pose.__file__).with_name('run_episode0_volume_chain.sh').read_text()
    assert '--episode 0 --full-video --mesh-source volume' in chain
    for step in ['cari_prepare','cari_forward','cari_converter','final_episode_gate']:
        assert f'run_{step}.sh" --episode 0' in chain
    wrapper=Path(pose.__file__).with_name('run_object_pose_smoke.sh').read_text()
    assert 'src=$ROOT/validation,dst=$ROOT/validation,readonly' in wrapper


@pytest.mark.parametrize("argument", ["--pointmap-directory", "--root", "--manual-mask", "--shape-fit", "--no-gt"])
def test_pose_has_no_new_research_algorithm_or_arbitrary_input_modes(pose, argument):
    with pytest.raises(SystemExit): pose._argument_parser().parse_args([argument])


def test_generic_volume_pins_are_immutable_code_bound_not_cli_or_receipt(pose):
    source=Path(pose.__file__).read_text()
    branch=source.split("if args.mesh_source=='volume':",1)[1].split('    else:',1)[0]
    assert 'volume_mesh_{args.episode:06d}_pins.json' in branch
    assert 'Path(__file__).resolve().parent.parent / "configs"' in branch
    assert 'pins=pins' in branch and 'strict_json(pin_path.read_text())' in branch
    assert "pin_path.stat().st_mode & 0o222" in branch
    assert "identity(pin_path) != pin_identity" in branch
    assert 'committed_pins_sha256' in branch
    assert 'fit_topology_preserving_budget(' not in branch
    with pytest.raises(SystemExit):pose._argument_parser().parse_args(['--volume-pins','arbitrary.json'])


def test_solid_requires_full_video_before_input_io_and_preserves_original_modes(pose,monkeypatch):
    monkeypatch.setattr(pose.platform,'system',lambda:'Linux')
    monkeypatch.setattr(pose.Path,'iterdir',lambda _:iter([Path('lo')]))
    monkeypatch.setattr(pose,'_validate_inputs',lambda *_a,**_k:pytest.fail('No source IO for sparse solid'))
    monkeypatch.setattr(sys,'argv',['object_pose_smoke.py','--mesh-source','solid'])
    with pytest.raises(ValueError,match='complete original video'): pose.main()
    for source in ('default','volume','conditioned','solid'):
        args=pose._argument_parser().parse_args(['--episode','9','--full-video','--mesh-source',source])
        assert args.mesh_source==source and args.full_video


@pytest.mark.parametrize('fault',['none','missing_pins','bad_pins','pins_mutated','source_mutated','copy_changed','preexisting'])
def test_solid_payload_is_same_array_tuple_scale_once_and_copy_byte_exact(pose,tmp_path,monkeypatch,fault):
    import numpy as np
    import solid_geometry_loader as loader
    import shutil
    code=tmp_path/'code'; root=tmp_path/'root'; pin=code/'configs/solid_mesh_000009_pins.json'
    pin.parent.mkdir(parents=True); pin.write_text('{"synthetic":"independent-only"}'); pin.chmod(0o444)
    original=root/'outputs/episode_000009/object_budget_solid/object_fixed_canonical.glb'
    original.parent.mkdir(parents=True); original.write_bytes(b'canonical source unchanged bytes'); original.chmod(0o444)
    reports=[tmp_path/'object.json',tmp_path/'alignment.json']
    for p in reports: p.write_bytes(b'opaque original metadata')
    output=root/'outputs/episode_000009/object_pose_full_solid'; fixed=output/'object_fixed_canonical.glb'
    vertices=np.arange(4096*3,dtype=np.float64).reshape(4096,3)
    faces=np.zeros((4096,3),np.int64); active=np.arange(4,dtype=np.int64); cleanup={'meaningful_faces_removed':0}; receipt={}
    values=(vertices,faces,active,cleanup,original,receipt); calls=[]
    monkeypatch.setattr(pose,'__file__',str(code/'infra/object_pose_smoke.py'))
    def load(*args,**kwargs):
        calls.append((args,kwargs))
        if fault=='pins_mutated': pin.chmod(0o644);pin.write_text('{"changed":true}');pin.chmod(0o444)
        return values
    monkeypatch.setattr(loader,'load',load)
    copy=shutil.copyfile
    def copying(source,target):
        copy(source,target)
        if fault=='source_mutated': original.chmod(0o644);original.write_bytes(b'changed source');original.chmod(0o444)
        elif fault=='copy_changed': Path(target).write_bytes(b'changed target')
    monkeypatch.setattr(shutil,'copyfile',copying)
    if fault=='missing_pins':pin.unlink()
    elif fault=='bad_pins':pin.chmod(0o644);pin.write_text('{"x":1,"x":2}');pin.chmod(0o444)
    elif fault=='preexisting':output.mkdir();(output/'old').write_bytes(b'historical must remain')
    if fault=='none':
        actual=pose._load_solid_mesh(root,9,'a'*64,*reports,.375,output,fixed)
        assert all(a is b for a,b in zip(actual,values))
        assert calls[0][0]==(root,9,'a'*64,pose.sha256(reports[0]),pose.sha256(reports[1]),.375)
        assert calls[0][1]=={'pins':{'synthetic':'independent-only'}}
        assert fixed.read_bytes()==original.read_bytes() and receipt['committed_pins_sha256']==loader.identity(pin)['sha256']
        np.testing.assert_array_equal(vertices,np.arange(4096*3).reshape(4096,3))
    else:
        with pytest.raises((ValueError,FileNotFoundError,FileExistsError)):
            pose._load_solid_mesh(root,9,'a'*64,*reports,.375,output,fixed)
        if fault in ('missing_pins','bad_pins','pins_mutated'):assert not output.exists()
        if fault=='preexisting':assert (output/'old').read_bytes()==b'historical must remain'


def test_original_default_volume_conditioned_and_native_tracker_ast_unchanged(pose):
    import subprocess
    before=subprocess.check_output(['git','show','56ed6fabe4235fe87f356b19a2ce9d65d61b2560:infra/object_pose_smoke.py'],cwd=Path(pose.__file__).resolve().parents[1],text=True)
    old=ast.parse(before); new=ast.parse(Path(pose.__file__).read_text())
    def main(tree):return next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
    def branch(tree,value):
        return next(n for n in ast.walk(main(tree)) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
            and any(isinstance(c,ast.Constant) and c.value==value for c in n.test.comparators)
            and n.body and isinstance(n.body[0],ast.ImportFrom)).body
    for value in ('volume','conditioned'):
        assert ast.dump(ast.Module(branch(old,value),type_ignores=[]))==ast.dump(ast.Module(branch(new,value),type_ignores=[]))
    def default(tree):
        node=next(n for n in ast.walk(main(tree)) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
            and any(isinstance(c,ast.Constant) and c.value=='volume' for c in n.test.comparators))
        while len(node.orelse)==1 and isinstance(node.orelse[0],ast.If):node=node.orelse[0]
        return node.orelse
    assert ast.dump(ast.Module(default(old),type_ignores=[]))==ast.dump(ast.Module(default(new),type_ignores=[]))
    def tracker(tree):
        nodes=main(tree).body; start=next(i for i,n in enumerate(nodes) if isinstance(n,ast.Assign) and
            any(isinstance(t,ast.Tuple) and any(isinstance(e,ast.Name) and e.id=='sampled' for e in t.elts) for t in n.targets))
        return ast.Module(nodes[start:],type_ignores=[])
    assert ast.dump(tracker(old))==ast.dump(tracker(new))
    text=Path(pose.__file__).read_text()
    assert "if args.mesh_source in ('volume','conditioned','solid'):" in text
    for source in ('conditioned','solid'):
        assert f"if args.mesh_source=='{source}' and np.any(faces[inactive] != 0):" in text
    assert "process=False" in text and "sample_surface(mesh, 8192, seed=0)" in text
