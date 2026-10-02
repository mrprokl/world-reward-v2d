"""Public RGB predictions -> shared official identity and finger-only proposals.

Only public RGB/mask receipts and frozen Body/full predictions are consumed.
The unchanged official mesh converter fits one identity to all six Body
meshes. Independent reference forward fidelity is engineering evidence, not
accuracy. Full decoded controls supply fingers only: no root/body/wrist,
identity, per-frame alignment, hidden truth or automatic adoption is used.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import platform
import re
import secrets
import subprocess
import sys
import time

import numpy as np

import hand_synthetic_infer as infer
from cari_converter import CONVERTER_SHA256, REFERENCE_MODEL_SHA256, require_fidelity, vertex_residual_mm
from world_reward.data import sha256
from world_reward.hand_transfer import transfer_finger_controls
from world_reward.submission import parameters_from_official_converter

STAGE = 'public_synthetic_rgb_official_finger_proposals'
CASES, VERTICES, JOINTS, MAX_SECONDS = 6, 18439, 127, 180.


def require_hash(path, expected):
    if (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
            or path.is_symlink() or not path.is_file() or path.resolve() != path.absolute()
            or sha256(path) != expected):
        raise ValueError('Frozen regular public artifact SHA mismatch')


def finite(value, shape):
    a = np.asarray(value)
    if (np.ma.isMaskedArray(value) or a.shape != shape or a.dtype not in (np.dtype('float32'), np.dtype('float64'))
            or not np.isfinite(a).all()):
        raise ValueError('Require finite fixed-shape float32/64 arrays, without repair')
    return a


def validate_inference(report, records, hashes):
    """Receipt integrity binds actual inference, not truth of its numerical claims."""
    expected = {'stage': infer.STAGE, 'status': 'pass', 'cases': CASES, 'calls_completed': 12,
        'actual_network_inference': True, 'actual_native_parameter_block_forward': True,
        'private_truth_read': False, 'challenge_inputs_used': False, 'hand_labeled_test': False,
        'oracle_modes': [], 'network': 'none', 'native_control_dimension': 204, 'raw266_used': False,
        'rotation_source': 'fresh_native_parameter_block_decode', 'geometry_units': 'metres',
        'geometry_frame': 'SAM_camera_x_right_y_down_z_forward', 'body_checkpoint_sha256': infer.BODY_SHA,
        'prompt_mode': 'automatic_binary_person_mask_and_derived_bbox_no_fallback', 'cam_int': 'None_RGB_size_FOV',
        'script_sha256': sha256(Path(infer.__file__)), 'body_helper_sha256': sha256(Path(infer.body.__file__))}
    if (not isinstance(report, dict) or any(type(report.get(k)) is not type(v) or report.get(k) != v for k,v in expected.items())
            or report.get('public_hashes') != hashes
            or not re.fullmatch(r'[0-9a-f]{40}', str(report.get('producer_revision', '')))
            or not re.fullmatch(r'sha256:[0-9a-f]{64}', str(report.get('image_id', '')))):
        raise ValueError('Require SHA-bound actual public Body/full inference, no labels or raw266')
    model = report.get('model', {})
    if (not isinstance(model, dict) or model.get('body_revision') != infer.body.BODY_REVISION
            or model.get('upstream_revision') != infer.body.UPSTREAM_REVISION
            or model.get('dinov3_revision') != infer.body.DINOV3_REVISION
            or not isinstance(model.get('inference_source_identity'), dict) or not model['inference_source_identity']
            or model.get('checkpoint_loading', {}).get('mode') != 'strict_network_and_head_state_with_explicit_asset_buffer_retention'
            or model.get('checkpoint_loading', {}).get('unexpected_keys') != []
            or model.get('body_assets', {}).get('model.ckpt') != {'sha256': infer.BODY_SHA, 'bytes': infer.BODY_BYTES}):
        raise ValueError('Require original strict pinned model/source/asset receipt')
    maps = model.get('hand_indices', {})
    transfer_finger_controls(np.zeros((CASES,136)), np.zeros((CASES,204)), np.arange(CASES),
                             np.asarray(maps.get('left')), np.asarray(maps.get('right')))
    calls = report.get('calls')
    if not isinstance(calls, list) or len(calls) != 12: raise ValueError('Require twelve complete Body/full calls')
    for i, record in enumerate(records):
        pair = calls[2*i:2*i+2]
        for call, mode in zip(pair, ('body','full')):
            if (not isinstance(call, dict) or type(call.get('case_index')) is not int or call['case_index'] != i
                    or call.get('inference_type') != mode or call.get('image_sha256') != record['image_sha256']
                    or call.get('mask_sha256') != record['mask_sha256'] or call.get('camera_intrinsics_supplied') is not False
                    or not re.fullmatch(r'[0-9a-f]{64}', str(call.get('decoded_rgb_sha256', '')))):
                raise ValueError('Inference calls do not address the same public RGB and automatic mask')
            box = finite(call.get('bbox_xyxy'), (4,))
            if np.any(box < 0) or np.any(box[2:] > [infer.WIDTH,infer.HEIGHT]) or np.any(box[2:]-box[:2] <= 1):
                raise ValueError('Require finite original-grid nonempty automatic mask bbox')
            errors = call.get('native_forward_errors', {})
            if (not isinstance(errors, dict) or set(errors) != {'vertices_m','joints_m','keypoints_m','controls'}
                    or any(type(x) not in (int,float) or not np.isfinite(x) or not 0 <= x <= 1e-5 for x in errors.values())):
                raise ValueError('Fresh native control-forward evidence failed')
        if pair[0]['decoded_rgb_sha256'] != pair[1]['decoded_rgb_sha256'] or pair[0].get('bbox_xyxy') != pair[1].get('bbox_xyxy'):
            raise ValueError('Body/full consumed different RGB or automatic prompt')


def prediction_arrays(archive):
    """Consume all six fresh native blocks/controls, never raw266 or stale rotations."""
    shapes = {'vertices_camera_m': (CASES,VERTICES,3), 'joints_camera_m': (CASES,JOINTS,3),
        'joint_global_rotations': (CASES,JOINTS,3,3), 'mhr_model_params': (CASES,204),
        'global_rot': (CASES,3), 'body_pose_params': (CASES,133), 'hand_pose_params': (CASES,108),
        'scale_params': (CASES,28), 'shape_params': (CASES,45), 'expr_params': (CASES,72),
        'pred_cam_t': (CASES,3), 'focal_length': (CASES,)}
    if set(archive) != set(shapes)|{'faces','frame_index'}: raise ValueError('Require exact fresh prediction archive schema')
    arrays = {k: finite(archive[k], shape) for k,shape in shapes.items()}
    frames, faces = np.asarray(archive['frame_index']), np.asarray(archive['faces'])
    if (np.ma.isMaskedArray(archive['frame_index']) or np.ma.isMaskedArray(archive['faces'])
            or frames.shape != (CASES,) or frames.dtype not in (np.dtype('int32'),np.dtype('int64'))
            or not np.array_equal(frames, np.arange(CASES)) or faces.shape != (36874,3)
            or faces.dtype not in (np.dtype('int32'),np.dtype('int64')) or np.any(faces < 0) or np.any(faces >= VERTICES)
            or np.any(arrays['expr_params']) or np.any(arrays['pred_cam_t'][:,2] <= 0)
            or not np.allclose(arrays['focal_length'], np.hypot(infer.WIDTH,infer.HEIGHT), rtol=1e-6, atol=1e-4)):
        raise ValueError('Prediction coverage, zero expression, camera or topology ABI failed')
    r = arrays['joint_global_rotations'].astype(np.float64)
    if not np.allclose(r@r.swapaxes(-1,-2), np.eye(3), atol=1e-4, rtol=0) or not np.allclose(np.linalg.det(r), 1, atol=1e-4, rtol=0):
        raise ValueError('Fresh native rotations must be proper SO3')
    return arrays|{'frame_index': frames, 'faces': faces}


def public_predictions(root):
    records, hashes = infer.public_inputs(root)
    base = root/'validation/hands_rgb_v1/predictions-v2'; path = base/'report.json'
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError('Require regular in-place public prediction receipt')
    digest = sha256(path); require_hash(path, digest)
    report = json.loads(path.read_text()); validate_inference(report, records, hashes)
    arrays = {}
    for mode in ('body','full'):
        npz = base/f'predictions_{mode}.npz'; require_hash(npz, report.get('prediction_sha256', {}).get(mode))
        with np.load(npz, allow_pickle=False) as archive: arrays[mode] = prediction_arrays(archive)
    if not np.array_equal(arrays['body']['faces'], arrays['full']['faces']): raise ValueError('Body/full topology differs')
    return arrays, report, digest, hashes


def make_proposals(converted, controls, left, right):
    official = parameters_from_official_converter(converted, max_mean_vertex_error_mm=2.)
    for key, shape in (('pose',(CASES,136)), ('scales',(68,)), ('shape',(45,)), ('expression',(72,))):
        finite(getattr(official,key), shape)
    if np.any(official.expression): raise ValueError('Official expressions must remain zero')
    proposal = transfer_finger_controls(official.pose, controls, np.arange(CASES), left, right)
    return official, proposal


def reference_geometry(vertices_mm, joints_m):
    """Official forward already flips axes and includes root: only V mm->m."""
    return finite(vertices_mm,(CASES,VERTICES,3))/1000., finite(joints_m,(CASES,JOINTS,3))


def phase(report, path, name):
    report['phase'] = name; infer._write(path, report)


def run(root, report, path):
    arrays, receipt, receipt_sha, hashes = public_predictions(root)
    if receipt_sha != report['inference_report_sha256'] or hashes != report['public_hashes']:
        raise ValueError('Public prediction chain changed after parent preflight')
    phase(report,path,'assets')
    source = infer.body._source_identity(root); directory, assets = infer.body._body_assets(root)
    if source != receipt['model']['inference_source_identity'] or assets != receipt['model']['body_assets']:
        raise ValueError('Current source or original Body assets differ from frozen inference')
    model_path = root/'weights/mhr/mhr_model.pt'
    tool = root/'vendor/v2d_submission_kit/tools/track1/mesh_to_mhr_params.py'
    require_hash(model_path,REFERENCE_MODEL_SHA256); require_hash(tool,CONVERTER_SHA256)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('CUDA required; no local/CPU conversion fallback')
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True
    payload = torch.load(directory/'model.ckpt', map_location='cpu', weights_only=False)
    state = payload.get('state_dict',payload); maps = []
    for side in ('left','right'):
        tensor = state.get('head_pose.hand_joint_idxs_'+side)
        if not torch.is_tensor(tensor): raise ValueError('Verified checkpoint hand-column map missing')
        value = tensor.detach().cpu().numpy()
        if value.shape != (27,) or value.dtype.kind not in 'iu' or value.tolist() != receipt['model']['hand_indices'][side]:
            raise ValueError('Actual checkpoint hand map differs from full-mode producer')
        maps.append(value.copy())
    del payload,state
    spec = importlib.util.spec_from_file_location('world_reward_public_hand_official_converter',tool)
    if spec is None or spec.loader is None: raise RuntimeError('Verified official tool cannot be imported')
    converter = importlib.util.module_from_spec(spec); spec.loader.exec_module(converter)
    phase(report,path,'official_shared_identity_conversion')
    converted = converter.convert(arrays['body']['vertices_camera_m'],str(model_path),device='cuda',precision='float32',
                                  model_batch=256,log=lambda *args,**kwargs: None)
    official, proposal = make_proposals(converted,arrays['full']['mhr_model_params'],*maps)
    phase(report,path,'independent_reference_fidelity_and_fingers')
    reference = converter.MHR(str(model_path),'cuda',chunk=16,precision='float32')
    reference_faces = reference.model.character_torch.mesh.faces.detach().cpu().numpy()
    if not np.array_equal(reference_faces, arrays['body']['faces']): raise ValueError('Reference and prediction topology differ')
    identity = torch.tensor(np.r_[official.scales,official.shape][None],dtype=torch.float64,device='cuda')
    geometry = []
    for pose in (official.pose,proposal.pose):
        vertices,joints = reference.run(torch.tensor(pose,dtype=torch.float64,device='cuda'),identity)
        geometry.append(reference_geometry(vertices.detach().cpu().numpy(),joints.detach().cpu().numpy()))
    errors, point_max = vertex_residual_mm(geometry[0][0],arrays['body']['vertices_camera_m'])
    fidelity = require_fidelity(errors)
    fidelity['max_point_error_mm_diagnostic_not_gate'] = point_max
    if public_predictions(root)[2] != receipt_sha: raise ValueError('Frozen public inference changed during conversion')
    phase(report,path,'export')
    output = path.parent/'proposals.npz'
    with output.open('xb') as stream:
        np.savez_compressed(stream,frame_index=np.arange(CASES,dtype=np.int64),baseline_pose=official.pose,candidate_pose=proposal.pose,
            scales=official.scales,shape=official.shape,expression=official.expression,faces=arrays['body']['faces'],
            baseline_vertices_camera_m=geometry[0][0],candidate_vertices_camera_m=geometry[1][0],
            baseline_joints_camera_m=geometry[0][1],candidate_joints_camera_m=geometry[1][1],
            hand_indices_left=maps[0],hand_indices_right=maps[1])
    report.update(status='pass',phase='complete',cases=CASES,proposals_sha256=sha256(output),
        nonhand_controls_bit_identical=True,shared_identity_preserved=True,conversion_fidelity_verified=True,
        official_converter_report=official.report,independent_reference_forward=fidelity,transfer=proposal.report,
        reference_control_names_compared=False,rotations_exported=False,geometry_units='metres',
        geometry_frame='camera_x_right_y_down_z_forward',facial_expressions_zero=True)
    infer._write(path,report)


def main():
    p = argparse.ArgumentParser(description=__doc__,allow_abbrev=False); p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args = p.parse_args(); started = time.perf_counter()
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Require isolated Linux CUDA container with network none')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',MOMENTUM_ENABLED='0',WANDB_MODE='disabled')
    root = Path(os.environ['WR_ROOT']); output = root/'validation/hands_rgb_v1/official_proposals'; path = output/'report.json'
    if not root.is_absolute() or output.parent.resolve() != output.parent.absolute():
        raise ValueError('Require absolute in-place public output, no ancestor symlinks')
    if args.worker:
        report = json.loads(path.read_text()); nonce = os.environ.get('WR_HAND_CONVERT_NONCE','')
        if (report.get('status') != 'running' or report.get('stage') != STAGE or not nonce
                or report.get('nonce_sha256') != hashlib.sha256(nonce.encode()).hexdigest()
                or report.get('script_sha256') != sha256(Path(__file__)) or report.get('producer_revision') != os.environ.get('WR_CODE_REVISION')):
            raise RuntimeError('Require parent running receipt; frozen outputs immutable')
        try: run(root,report,path)
        except Exception as exc:
            report.update(status='fail',error_type=type(exc).__name__,error=str(exc)); infer._write(path,report); raise
        return
    reserved = os.environ.get('WR_HAND_CONVERT_OUTPUT_RESERVED') == '1'
    if output.is_symlink() or (output.exists() and (not reserved or not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError('Frozen official proposals exist')
    if reserved and not output.is_dir(): raise ValueError('Reserved output must be a newly created empty mount')
    arrays,receipt,digest,hashes = public_predictions(root); del arrays
    image,revision = os.environ.get('WR_IMAGE_ID',''),os.environ.get('WR_CODE_REVISION','')
    if not re.fullmatch(r'sha256:[0-9a-f]{64}',image) or not re.fullmatch(r'[0-9a-f]{40}',revision):
        raise ValueError('Require immutable image/source revision')
    nonce = secrets.token_hex(32)
    report = {'stage':STAGE,'status':'running','phase':'prerequisites','public_hashes':hashes,
        'inference_report_sha256':digest,'inference_prediction_sha256':receipt['prediction_sha256'],
        'inference_producer_revision':receipt['producer_revision'],'inference_script_sha256':receipt['script_sha256'],
        'body_helper_sha256':receipt['body_helper_sha256'],'body_checkpoint_sha256':infer.BODY_SHA,
        'model_sha256':REFERENCE_MODEL_SHA256,'converter_sha256':CONVERTER_SHA256,
        'image_id':image,'producer_revision':revision,'script_sha256':sha256(Path(__file__)),
        'nonce_sha256':hashlib.sha256(nonce.encode()).hexdigest(),'private_truth_read':False,
        'challenge_inputs_used':False,'hand_labeled_test':False,'oracle_modes':[],'network':'none',
        'accuracy_verified':False,'adoption_performed':False,'max_worker_seconds':MAX_SECONDS,
        'geometry_forward_scope':'official_reference_representation_fidelity_not_accuracy',
        'CUBLAS_WORKSPACE_CONFIG':':4096:8','strict_deterministic_algorithms':True,'TF32':False}
    if not reserved: output.mkdir(exist_ok=False)
    with path.open('x') as stream: stream.write(json.dumps(report,indent=2)+'\n')
    try:
        process = subprocess.run([sys.executable,str(Path(__file__)),'--worker'],check=False,
            timeout=max(.001,MAX_SECONDS-(time.perf_counter()-started)),env=os.environ|{'WR_HAND_CONVERT_NONCE':nonce})
    except subprocess.TimeoutExpired:
        report = json.loads(path.read_text()); report.update(status='fail',error_type='TimeoutError',error='Hard180s conversion budget exceeded',elapsed_seconds=time.perf_counter()-started)
        infer._write(path,report); raise
    report = json.loads(path.read_text()); report['elapsed_seconds'] = time.perf_counter()-started; infer._write(path,report)
    if process.returncode or report['status'] != 'pass': raise RuntimeError('Public official conversion failed: '+report.get('error','incomplete'))
    print(json.dumps({'stage':STAGE,'status':'pass','cases':CASES,'accuracy_verified':False,'adoption_performed':False}))


if __name__ == '__main__': main()
