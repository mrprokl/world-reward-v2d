"""Six actual public-geometry kernel probes; no fit, GT or model inference.

CPU all-hand/all-face BVH chooses a witness. Unchanged native CUDA first scans
the entire hand/mesh, then independently checks the chosen pair's native value,
point/object-translation gradients and one Adam update. Geometry is preserved.
Passing six probes is NOT broad parity, HOI validation or production adoption.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

import numpy as np

from mediapipe_cpu_runtime_verify import canonical, identity, require, source, strict
from world_reward.contact_bvh import (ContactSearchLimits, NativeContactBVH,
    NATIVE_MIN_TRIANGLE_AREA, NATIVE_SQUARED_DISTANCE_FLOOR)


ROOT = Path('/srv/scenesmith/world-reward')
ENTRY = 'run_contact_bvh_qualification'
IMAGE = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
FRAMES = (0, 47, 95)
BUDGET = 600
NATIVE_REFERENCE_BUDGET = 120
VALUE_ATOL_M = 2e-6
GRAD_ATOL = 2e-6
GRAD_RTOL = 2e-5
ADAM_ATOL_M = 1e-7
ADAM_LR = 1e-4
HAND_SPEC_PIN = dict(bytes=47140, sha256='65e467ae534281c8c5370b76d672c80cf99b95bc73af9c3ac64d5bea6c7f60c8')
HAND_FACES_SHA256 = '7f0898679b24c006e077df9531171a9206ae986d6d5885094cc646669b3784f5'
HAND_TOPOLOGY_SHA256 = '075bf66319bd8fa348e5acf28f838b056952e550419331cf3ed10b68060ea3ba'
HAND_MODEL_SHA256 = '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
NATIVE_PY_PIN = dict(bytes=15972, sha256='92d26429c1eecefe5a183c8b8fb0dc89d5b3d4aed44825132cb9026e30823481')
HELPERS = ('infra/contact_bvh_qualification.py', 'infra/run_contact_bvh_qualification.sh',
           'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/contact_bvh.py')


def _pin(value):
    require(type(value) is dict and set(value) == {'bytes', 'sha256'}
            and type(value['bytes']) is int and 0 < value['bytes'] <= 1 << 30
            and re.fullmatch('[0-9a-f]{64}', str(value['sha256'])), 'Exact artifact pin required')
    return value


def validate_manifest(value, root=ROOT):
    require(type(value) is dict and set(value) == {'schema', 'ground_truth_used', 'private_truth_read',
            'prediction_base', 'input', 'body_report', 'object_report', 'hand_spec'},
            'Exact public kernel qualification manifest required')
    require(value['schema'] == 'world_reward.contact_bvh_qualification_input.v1'
            and value['ground_truth_used'] is False and value['private_truth_read'] is False,
            'Public no-GT manifest required')
    base = Path(value['prediction_base'])
    require(base.is_absolute() and base.parent.parent == root/'results'
            and re.fullmatch('form-hoi-external-predict-[0-9a-f]{40}', base.parent.name)
            and re.fullmatch('[A-Za-z0-9_-]{1,128}', base.name), 'Original prediction namespace required')
    for name in ('input', 'body_report', 'object_report', 'hand_spec'):
        row = value[name]
        require(type(row) is dict and set(row) == {'path', 'pin'}, 'Explicit pinned input row required')
        _pin(row['pin']); path = Path(row['path'])
        require(path.is_absolute() and '..' not in path.parts
                and not {'eval_private', 'track_1', 'gt', '.secrets'}.intersection(path.parts),
                'No private/reference/challenge/secret namespace')
    require(Path(value['body_report']['path']) == base/'body_depth/report.json'
            and Path(value['object_report']['path']) == base/'object/report.json', 'Exact stage reports required')
    input_path = Path(value['input']['path'])
    require(input_path.name == 'input.json' and input_path.parent.name == 'inputs'
            and input_path.parent.parent.name == base.name
            and input_path.is_relative_to(Path('/srv/world-reward-data/form_hoi_external_dev_v1')),
            'Only original external DEV public input metadata allowed')
    require(Path(value['hand_spec']['path']) == root/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'
            and value['hand_spec']['pin'] == HAND_SPEC_PIN, 'Independent authentic whole-hand asset pin required')
    return base


def public_input_identity(path, maximum):
    """One pinned upstream asset may be owner-writable on its acquisition host.

    Dataset inputs/reports/predictions remain sealed. Only this exact canonical
    hand asset is admitted through a hash-controlled regular single-link path;
    no group/world write, no shared chmod, and the container mount stays RO.
    The same admission and independent pin are checked again after the probe.
    """
    path = canonical(path)
    hand_path = ROOT/'weights/cari4d/refinement/mhr_hand_surface_spec.npz'
    if path != hand_path:
        return identity(path, maximum)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o022,
            'Pinned hand asset must be single-link regular and not group/world writable')
    pin = identity(path, maximum, readonly=False)
    after = path.lstat()
    require(pin == HAND_SPEC_PIN and all(getattr(before,key)==getattr(after,key) for key in
        ('st_dev','st_ino','st_mode','st_size','st_mtime_ns','st_ctime_ns','st_nlink','st_uid','st_gid')),
        'Exclusive authentic hand asset changed while admitted')
    return pin


def load_inputs(manifest_path, manifest_pin):
    require(identity(manifest_path, 32768) == manifest_pin, 'Independent manifest pin mismatch')
    value = strict(Path(manifest_path).read_bytes()); base = validate_manifest(value)
    pins = {canonical(manifest_path): manifest_pin}
    reports = {}
    for name in ('input', 'body_report', 'object_report', 'hand_spec'):
        row = value[name]; path = canonical(Path(row['path']))
        require(public_input_identity(path, 1 << 20) == row['pin'], 'Independent input/stage pin mismatch: '+name)
        pins[path] = row['pin']
        if name != 'hand_spec': reports[name] = strict(path.read_bytes())
    public = reports['input']
    require(public.get('schema') == 'world_reward.external_rgb_input.v1'
            and public.get('dataset') == 'nvidia/form-hoi' and public.get('split') == 'development'
            and public.get('sequence_id') == base.name and public.get('total') == 96
            and public.get('original_frame_indices') == list(range(96))
            and public.get('inference_ready') is True and public.get('reference_inputs_present') is False,
            'Original public DEV96 metadata required')
    for name, stage, files in (('body_report', 'body_depth', ('body.npz',)),
                               ('object_report', 'object', ('object.glb', 'transform.json'))):
        report = reports[name]
        require(report.get('status') == 'complete' and report.get('stage') == stage
                and report.get('ground_truth_used') is False and report.get('private_truth_read') is False
                and report.get('input_pin') == value['input']['pin'], 'Sealed same-public-input stage completion required')
        for filename in files:
            pin = _pin(report['artifacts'][filename]); path = canonical(base/stage/filename)
            require(identity(path, 1 << 30) == pin, 'Original whole geometry artifact mismatch')
            pins[path] = pin
    return value, reports, pins


def readonly_mounts(value, manifest_path, code, out):
    base = validate_manifest(value)
    paths = [code.parent, manifest_path, *(Path(value[name]['path']) for name in (
        'input', 'body_report', 'object_report', 'hand_spec')),
        base/'body_depth/body.npz', base/'object/object.glb', base/'object/transform.json']
    require(all(not {'eval_private', 'track_1', 'gt', '.secrets'}.intersection(p.parts) for p in paths),
            'Explicit read-only public geometry mounts only')
    return [(p, True) for p in dict.fromkeys(paths)] + [(out, False)]


def array_sha256(*arrays):
    digest = hashlib.sha256()
    for value in arrays:
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode()); digest.update(np.asarray(array.shape, np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def load_geometry(value):
    import trimesh
    from scipy.spatial.transform import Rotation
    base = Path(value['prediction_base'])
    with np.load(base/'body_depth/body.npz', allow_pickle=False) as z:
        human = np.array(z['vertices_camera_m'], copy=True)
        human_faces = np.array(z['faces'], copy=True)
        frame_index = np.array(z['frame_index'], copy=True)
    require(human.dtype == np.float32 and human.shape == (96,18439,3) and np.isfinite(human).all()
            and human_faces.dtype == np.int64 and human_faces.shape == (36874,3)
            and frame_index.dtype == np.int64 and np.array_equal(frame_index, np.arange(96)),
            'Complete native original Body96 topology/chronology required')
    with np.load(value['hand_spec']['path'], allow_pickle=False) as z:
        keys = ('vertex_indices','sample_local_indices','sample_assignments','hand_faces_left','hand_faces_right')
        arrays = [np.array(z[k],copy=True) for k in keys]
        expected_shapes = ((2,2318),(2,256),(2,2318),(4603,3),(4603,3))
        require(all(a.dtype==np.int32 and a.shape==shape for a,shape in zip(arrays,expected_shapes)),
                'Exact typed whole-hand topology asset required')
        ids = arrays[0]
        require(ids.shape == (2,2318) and ids.min() >= 0 and ids.max() < 18439,
                'All native 2318 hand vertices per side required')
        require(array_sha256(human_faces.astype(np.int32)) == str(z['faces_sha256'].item()) == HAND_FACES_SHA256
                and array_sha256(human_faces.astype(np.int32),*arrays) == str(z['topology_sha256'].item()) == HAND_TOPOLOGY_SHA256
                and str(z['mhr_model_sha256'].item()) == HAND_MODEL_SHA256
                and len(np.unique(ids)) == ids.size,
                'Authentic hand spec/body topology differs')
    mesh = trimesh.load(base/'object/object.glb', force='mesh', process=False)
    require(isinstance(mesh, trimesh.Trimesh), 'One complete unprocessed generated object mesh required')
    transform = strict((base/'object/transform.json').read_bytes())
    scale = np.asarray(transform['scale'], float); q = np.asarray(transform['rotation'], float)
    translation = np.asarray(transform['translation'], np.float32)
    require(scale.shape == (3,) and np.isfinite(scale).all() and (scale > 0).all()
            and q.shape == (4,) and np.isfinite(q).all() and np.isclose(np.linalg.norm(q),1,atol=1e-5)
            and translation.shape == (3,) and np.isfinite(translation).all(), 'Original generated fixed SE3/scale required')
    rotation = Rotation.from_quat([q[1],q[2],q[3],q[0]]).as_matrix().astype(np.float32)
    vertices = (np.asarray(mesh.vertices,float)*scale[None]).astype(np.float32)
    faces = np.asarray(mesh.faces,np.int64)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and np.isfinite(vertices).all()
            and faces.ndim == 2 and faces.shape[1] == 3 and len(faces), 'Complete finite scaled object required')
    return human, ids.astype(np.int64), vertices, faces, rotation, translation


def compare_arrays(reference, candidate, *, atol, rtol=0.):
    a, b = np.asarray(reference), np.asarray(candidate)
    require(a.shape == b.shape and np.isfinite(a).all() and np.isfinite(b).all(), 'Finite equal-shaped parity arrays required')
    return dict(pass_tolerance=bool(np.allclose(a,b,atol=atol,rtol=rtol)),
                max_abs_error=float(np.max(np.abs(a-b))) if a.size else 0., atol=atol, rtol=rtol,
                byte_equal=bool(a.dtype == b.dtype and a.tobytes() == b.tobytes()))


def probe(torch, native_C, point_face_distance, bvh, vertices, faces, world_points, rotation, translation):
    """Full native reference once; chosen pair native narrow phase independently."""
    p = torch.tensor(np.array(world_points,copy=True),device='cuda',dtype=torch.float32)
    r = torch.tensor(rotation,device='cuda',dtype=torch.float32)
    t = torch.tensor(translation,device='cuda',dtype=torch.float32)
    local = ((p-t)@r).contiguous()
    local_cpu = local.detach().cpu().numpy()
    candidate = bvh.minimum(local_cpu)
    v = torch.tensor(vertices,device='cuda',dtype=torch.float32)
    f = torch.tensor(faces,device='cuda',dtype=torch.long)
    triangles = v[f].contiguous(); first = torch.zeros(1,device='cuda',dtype=torch.long)
    torch.cuda.synchronize(); started = time.monotonic()
    distances, native_faces = native_C.point_face_dist_forward(local,first,triangles,first,len(p),NATIVE_MIN_TRIANGLE_AREA)
    torch.cuda.synchronize(); reference_seconds = time.monotonic()-started
    require(reference_seconds <= NATIVE_REFERENCE_BUDGET, 'Single-hand CUDA reference exceeded declared budget')
    require(torch.isfinite(distances).all(), 'Full native reference must be finite')
    clamped = torch.clamp(distances,min=NATIVE_SQUARED_DISTANCE_FLOOR)
    metric_distances = torch.sqrt(clamped)
    ref_point = int(torch.argmin(metric_distances).item()); ref_face = int(native_faces[ref_point].item())
    reference_squared = float(distances[ref_point].item())
    native_point_tied = int((metric_distances == metric_distances.min()).sum().item()) > 1
    # Replay the authentic sqrt(clamp(distance²)).min().square() graph rather
    # than silently replacing its finite-precision derivative with constant1.
    distance_variable = distances.detach().clone().requires_grad_(True)
    reference_loss = torch.sqrt(torch.clamp(distance_variable,min=NATIVE_SQUARED_DISTANCE_FLOOR)).min(dim=0).values.square()
    gradient_weights, = torch.autograd.grad(reference_loss,distance_variable)
    torch.cuda.synchronize(); backward_started = time.monotonic()
    ref_local_gradient, _ = native_C.point_face_dist_backward(local,triangles,native_faces,
        gradient_weights.contiguous(),NATIVE_MIN_TRIANGLE_AREA)
    torch.cuda.synchronize(); backward_seconds = time.monotonic()-backward_started
    ref_world_gradient = ref_local_gradient @ r.T
    ref_translation_gradient = -ref_world_gradient.sum(0)
    # Original geometry remains constant; candidate autograd covers the full hand
    # variable, so unselected points receive exactly zero gradient via indexing.
    variable = p.detach().clone().requires_grad_(True); delta = t.detach().clone().requires_grad_(True)
    chosen = ((variable[candidate.point_index:candidate.point_index+1]-delta)@r).contiguous()
    chosen_triangle = triangles[candidate.face_index:candidate.face_index+1]
    selected_squared = point_face_distance(chosen,first,chosen_triangle,first,1,NATIVE_MIN_TRIANGLE_AREA)[0]
    loss = torch.sqrt(torch.clamp(selected_squared,min=NATIVE_SQUARED_DISTANCE_FLOOR)).square()
    loss.backward(); torch.cuda.synchronize()
    candidate_squared = float(selected_squared.detach().item())
    a = lambda x:x.detach().cpu().numpy()
    gradient = compare_arrays(a(ref_world_gradient),a(variable.grad),atol=GRAD_ATOL,rtol=GRAD_RTOL)
    translation_gradient = compare_arrays(a(ref_translation_gradient),a(delta.grad),atol=GRAD_ATOL,rtol=GRAD_RTOL)
    # A genuine optimizer step on the same two parameter tensors, not a formula
    # approximation. Compare both full-point and translation parameter updates.
    reference_p = p.detach().clone().requires_grad_(True); reference_t = t.detach().clone().requires_grad_(True)
    reference_p.grad = ref_world_gradient.detach().clone(); reference_t.grad = ref_translation_gradient.detach().clone()
    optimizer_ref = torch.optim.Adam([reference_p,reference_t],lr=ADAM_LR)
    optimizer_candidate = torch.optim.Adam([variable,delta],lr=ADAM_LR)
    optimizer_ref.step(); optimizer_candidate.step(); torch.cuda.synchronize()
    adam_points = compare_arrays(a(reference_p),a(variable),atol=ADAM_ATOL_M)
    adam_translation = compare_arrays(a(reference_t),a(delta),atol=ADAM_ATOL_M)
    value_error = abs(np.sqrt(max(reference_squared,NATIVE_SQUARED_DISTANCE_FLOOR))
                      -np.sqrt(max(candidate_squared,NATIVE_SQUARED_DISTANCE_FLOOR)))
    ids_equal = (ref_point,ref_face) == (candidate.point_index,candidate.face_index)
    tie = candidate.tied_witness or candidate.floor_ambiguous or native_point_tied
    passed = bool(value_error <= VALUE_ATOL_M and ids_equal and not tie
                  and gradient['pass_tolerance'] and translation_gradient['pass_tolerance']
                  and adam_points['pass_tolerance'] and adam_translation['pass_tolerance'])
    row = dict(cpu=asdict(candidate),native_reference_seconds=reference_seconds,
        native_backward_seconds=backward_seconds,reference_squared_distance=reference_squared,
        selected_native_squared_distance=candidate_squared,CPU_to_native_squared_abs_error=abs(candidate.squared_distance-reference_squared),
        native_reference_point_index=ref_point,native_reference_face_index=ref_face,
        selected_pair_indices_equal=ids_equal,native_point_tied=native_point_tied,
        distance_abs_error_m=float(value_error),distance_atol_m=VALUE_ATOL_M,
        world_point_gradient=gradient,object_translation_gradient=translation_gradient,
        Adam_points=adam_points,Adam_object_translation=adam_translation,
        Adam_learning_rate=ADAM_LR,kernel_probe_pass=passed,
        native_triangle_gradient_compared=False,native_face_ties_exhaustively_checked=False,
        native_loss_graph='sqrt_clamp_1e-12_min_then_square',production_adopted=False)
    del triangles, v, f, distances, native_faces
    torch.cuda.empty_cache()
    return row


def seal_json(path, value):
    path = canonical(path); require(not path.exists(), 'Exclusive qualification report required')
    with path.open('xb') as stream:
        stream.write((json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())
        stream.flush(); os.fsync(stream.fileno()); os.fchmod(stream.fileno(),0o444)


def main():
    parser = argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument('--manifest',type=Path,required=True)
    parser.add_argument('--manifest-bytes',type=int,required=True)
    parser.add_argument('--manifest-sha256',required=True)
    parser.add_argument('--plan',action='store_true')
    args = parser.parse_args(); manifest_pin = _pin(dict(bytes=args.manifest_bytes,sha256=args.manifest_sha256))
    value,reports,pins = load_inputs(args.manifest,manifest_pin)
    code=canonical(Path(os.environ['WR_CODE'])); revision=os.environ['WR_CODE_REVISION']
    require(Path(os.environ['WR_ROOT']) == ROOT and re.fullmatch('[0-9a-f]{40}',revision), 'Immutable Azure namespace required')
    out=ROOT/'results'/('contact-bvh-qualification-'+revision)
    if args.plan:
        print(json.dumps(dict(image=IMAGE,out=str(out),mounts=[dict(path=str(p),readonly=ro)
            for p,ro in readonly_mounts(value,args.manifest,code,out)]),sort_keys=True));return
    binding=source(ROOT,code,revision,ENTRY,HELPERS)
    require(sys.platform=='linux' and os.geteuid()==0 and os.environ.get('WR_IMAGE_ID')==IMAGE
            and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}, 'Restricted offline original GPU runtime required')
    require(out.is_dir() and {p.name for p in out.iterdir()} <= {'.container.cid'},
            'Exclusive empty qualification output required')
    started=time.monotonic(); report=dict(schema='world_reward.contact_bvh_qualification.v1',status='fail',
        producer_revision=revision,image_id=IMAGE,source_before=binding,input_manifest_pin=manifest_pin,
        ground_truth_used=False,private_truth_read=False,geometry_deleted=False,models_loaded=False,
        fit_run=False,production_adopted=False,broad_native_parity_verified=False,HOI_accuracy_verified=False,
        original_frame_indices=list(range(96)),probe_frames=list(FRAMES),rows=[],
        interpretation='six_single_hand_kernel_qualification_not_HOI_evaluation_or_fullfit',
        geometry_provenance='saved_original_BODY_camera_metres_and_SAM_object_scale_once_native_anchor_SE3')
    def alarm(_signum,_frame):raise TimeoutError('Qualification execution budget exceeded')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(BUDGET)
    try:
        import torch
        import pytorch3d.loss.point_mesh_distance as native_py
        from pytorch3d import _C
        require(torch.cuda.is_available(), 'Actual native GPU reference required')
        native_file=Path(native_py.__file__)
        raw=native_file.read_bytes()
        require(dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())==NATIVE_PY_PIN
                and native_py._DEFAULT_MIN_TRIANGLE_AREA==NATIVE_MIN_TRIANGLE_AREA, 'Pinned unchanged native Python operator required')
        report['native_operator_pin']=NATIVE_PY_PIN
        report['native_binary_pin']=dict(bytes=Path(_C.__file__).stat().st_size,
            sha256=hashlib.sha256(Path(_C.__file__).read_bytes()).hexdigest())
        human,ids,vertices,faces,rotation,translation=load_geometry(value)
        began=time.monotonic();bvh=NativeContactBVH(vertices,faces,limits=ContactSearchLimits())
        report.update(tree_build_seconds=time.monotonic()-began,object_vertices=len(vertices),
            object_faces=len(faces),hand_vertices_per_side=ids.shape[1],all_original_geometry_retained=True,
            tiny_face_fraction=float(np.mean(np.linalg.norm(np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],
                vertices[faces[:,2]]-vertices[faces[:,0]]),axis=1)*.5<NATIVE_MIN_TRIANGLE_AREA)))
        for frame in FRAMES:
            for side in range(2):
                row=probe(torch,_C,native_py.point_face_distance,bvh,vertices,faces,human[frame,ids[side]],rotation,translation)
                report['rows'].append(dict(frame_index=frame,hand_side=side,**row))
        require(all(public_input_identity(path,1<<30)==pin for path,pin in pins.items()),
                'Pinned public geometry changed during qualification')
        after=source(ROOT,code,revision,ENTRY,HELPERS);require(after==binding,'Source closure changed')
        passed=len(report['rows'])==6 and all(row['kernel_probe_pass'] for row in report['rows'])
        report.update(status='pass' if passed else 'parity_rejected',source_after=after,
            inputs_rehashed_after=True,six_probe_parity_verified=passed,
            exhaustive_native_face_tie_qualification=False,
            decision='SIX_PROBES_PASS_NOT_ADOPTED' if passed else 'REJECT_NATIVE_REPLACEMENT_UNQUALIFIED')
    except Exception as error:
        report.update(error_type=type(error).__name__,decision='REJECT_INCOMPLETE_QUALIFICATION')
        if isinstance(error,ValueError):report['error_context']=str(error)[:400]
        raise
    finally:
        signal.alarm(0);report['elapsed_seconds']=time.monotonic()-started
        seal_json(out/'report.json',report)
    print(json.dumps(dict(status=report['status'],probes=len(report['rows']),production_adopted=False)),flush=True)


if __name__=='__main__':main()
