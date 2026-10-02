"""Private six-case hand evaluation only after public predictions are frozen.

No fitting, correction, model call or adoption. Absolute camera errors and
wrist-relative articulation diagnostics are distinct; these are not V2D scores.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import time

import numpy as np
from world_reward.data import sha256

CASES, VERTICES, JOINTS = 6, 18439, 127
CASE_NAMES = ('neutral', 'left_bend', 'right_bend', 'oblique_bimanual', 'left_occluded', 'edge_crop')
MODEL_SHA = '352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'
CONVERTER_SHA = 'c799ad612fca19620563fcb93bf61e5a4adad0a04251482358746b5f27f8a52e'


def require_hash(path, expected):
    if (not isinstance(expected, str) or not re.fullmatch(r'[0-9a-f]{64}', expected)
            or path.is_symlink() or not path.is_file() or path.resolve() != path.absolute()
            or sha256(path) != expected):
        raise ValueError('Frozen regular artifact SHA mismatch')


def finite(value, shape):
    a = np.asarray(value)
    if a.shape != shape or a.dtype.kind != 'f' or not np.isfinite(a).all():
        raise ValueError('Require finite fixed-shape floating geometry')
    return a.astype(np.float64)


def proposal_contract(report):
    expected = {'stage': 'public_synthetic_rgb_official_finger_proposals', 'status': 'pass', 'cases': CASES,
                'private_truth_read': False, 'challenge_inputs_used': False, 'adoption_performed': False,
                'nonhand_controls_bit_identical': True, 'shared_identity_preserved': True,
                'conversion_fidelity_verified': True, 'model_sha256': MODEL_SHA, 'converter_sha256': CONVERTER_SHA}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k, v in expected.items()):
        raise ValueError('Require frozen public-only official proposals with fidelity, not quality claim')


def hand_partition(names, rig):
    if not isinstance(names, list) or len(names) != JOINTS or len(set(names)) != JOINTS:
        raise ValueError('Require actual unique 127 reference joint names')
    output = {}
    for side, label in (('l', 'left'), ('r', 'right')):
        mask = np.asarray(rig['hand_vertex_mask_' + label])
        if mask.shape != (VERTICES,) or mask.dtype != np.bool_ or mask.sum() < 50:
            raise ValueError('Require actual dominant-LBS private hand regions')
        wrist = names.index(f'{side}_wrist')
        fingers = [i for i, n in enumerate(names) if re.fullmatch(fr'{side}_(thumb|index|middle|ring|pinky)([0-3]|_?null)', n)]
        if len(fingers) < 15: raise ValueError('Named private finger skeleton incomplete')
        output[label] = {'mask': mask, 'wrist': wrist, 'fingers': np.asarray(fingers)}
    if np.any(output['left']['mask'] & output['right']['mask']):
        raise ValueError('Private dominant-LBS regions overlap')
    return output


def errors(pred_v, pred_j, true_v, true_j, region):
    mask, wrist, fingers = region['mask'], region['wrist'], region['fingers']
    distance = np.linalg.norm(pred_v[mask] - true_v[mask], axis=-1) * 1000
    relative_v = (pred_v[mask] - pred_j[wrist]) - (true_v[mask] - true_j[wrist])
    relative_j = (pred_j[fingers] - pred_j[wrist]) - (true_j[fingers] - true_j[wrist])
    return {'hand_pve_mm': float(distance.mean()),
            'finger_mpjpe_mm': float(np.linalg.norm(pred_j[fingers] - true_j[fingers], axis=-1).mean()*1000),
            'wrist_position_error_mm': float(np.linalg.norm(pred_j[wrist]-true_j[wrist])*1000),
            'wrist_relative_hand_pve_mm_diagnostic': float(np.linalg.norm(relative_v, axis=-1).mean()*1000),
            'wrist_relative_finger_mpjpe_mm_diagnostic': float(np.linalg.norm(relative_j, axis=-1).mean()*1000)}, distance


def visible_face_support(face_image, faces, mask):
    """Vertex support of raster-visible body faces, not exact vertex visibility."""
    image = np.asarray(face_image)
    if image.shape != (768, 1024) or image.dtype.kind not in 'iu':
        raise ValueError('Require private full-grid raster face labels')
    visible_faces = np.unique(image[(image >= 0) & (image < len(faces))])
    visible = np.zeros(VERTICES, dtype=np.bool_)
    visible[faces[visible_faces].reshape(-1)] = True
    return visible[mask]


def decision(records):
    """Frozen >=5% mean nonneutral PVE gain; no neutral hand regression >1e-5m."""
    if len(records) != CASES: raise ValueError('Require every frozen original case, not a subset')
    before = np.asarray([r['hands'][s]['baseline']['hand_pve_mm'] for r in records[1:] for s in ('left','right')])
    after = np.asarray([r['hands'][s]['candidate']['hand_pve_mm'] for r in records[1:] for s in ('left','right')])
    if not np.isfinite(np.r_[before, after]).all() or np.any(before < 0) or np.any(after < 0):
        raise ValueError('Require finite nonnegative complete quality metrics')
    neutral = [records[0]['hands'][s]['candidate']['hand_pve_mm']-records[0]['hands'][s]['baseline']['hand_pve_mm'] for s in ('left','right')]
    improvement = float(1-after.mean()/before.mean()) if before.mean() > 0 else None
    return {'mean_nonneutral_hand_pve_baseline_mm': float(before.mean()),
            'mean_nonneutral_hand_pve_candidate_mm': float(after.mean()),
            'mean_nonneutral_hand_pve_relative_gain': improvement,
            'neutral_side_regression_mm': neutral,
            'synthetic_hypothesis_supported': improvement is not None and improvement >= .05 and max(neutral) <= .01,
            'selection_scope': 'synthetic engineering only, not real-data validation or adoption'}


def run(root, report):
    base = root/'validation/hands_rgb_v1'; proposal = base/'official_proposals'; private = base/'eval_private'
    receipt_path = proposal/'report.json'; receipt_sha = sha256(receipt_path)
    require_hash(receipt_path, receipt_sha); receipt = json.loads(receipt_path.read_text()); proposal_contract(receipt)
    npz_path = proposal/'proposals.npz'; require_hash(npz_path, receipt.get('proposals_sha256'))
    infer_path = base/'predictions/report.json'; require_hash(infer_path, receipt.get('inference_report_sha256'))
    infer = json.loads(infer_path.read_text())
    if infer.get('status') != 'pass' or infer.get('private_truth_read') is not False:
        raise ValueError('Original public inference must pass with no private truth access')
    for mode in ('body', 'full'):
        digest = receipt.get('inference_prediction_sha256', {}).get(mode)
        if infer.get('prediction_sha256', {}).get(mode) != digest: raise ValueError('Inference artifact chain mismatch')
        require_hash(base/f'predictions/predictions_{mode}.npz', digest)
    public_hashes = receipt.get('public_hashes', {})
    for key, relative in (('input_manifest', 'inputs/manifest.json'), ('mask_report', 'automatic_masks/report.json')):
        require_hash(base/relative, public_hashes.get(key))
    with np.load(npz_path, allow_pickle=False) as source:
        values = {k:source[k].copy() for k in source.files}
    for mode in ('baseline', 'candidate'):
        values[mode+'_vertices_camera_m'] = finite(values[mode+'_vertices_camera_m'], (CASES, VERTICES, 3))
        values[mode+'_joints_camera_m'] = finite(values[mode+'_joints_camera_m'], (CASES, JOINTS, 3))
        finite(values[mode+'_pose'], (CASES, 136))
    fixed = np.r_[np.arange(68), np.arange(122,136)]
    a,b = values['baseline_pose'],values['candidate_pose']
    if a.dtype != b.dtype or a[:,fixed].tobytes() != b[:,fixed].tobytes():
        raise ValueError('Proposal changed fixed nonhand pose bytes')
    if (np.asarray(values['frame_index']).dtype.kind not in 'iu' or not np.array_equal(values['frame_index'], np.arange(CASES))
            or np.any(finite(values['expression'], (72,))) or finite(values['scales'], (68,)).shape != (68,)):
        raise ValueError('Require all cases, zero expressions and one shared identity')
    finite(values['shape'], (45,))
    render_path = private/'render-report.json'; render_sha = sha256(render_path)
    require_hash(render_path, render_sha); render = json.loads(render_path.read_text())
    if (render.get('stage') != 'own_mhr_rendered_rgb_validation' or render.get('status') != 'pass'
            or render.get('model_sha256') != MODEL_SHA or render.get('inference_performed') is not False
            or render.get('fixture', {}).get('case_names') != list(CASE_NAMES)
            or [c.get('case_index') for c in render.get('cases', [])] != list(range(CASES))):
        raise ValueError('Require complete own private renderer provenance')
    rig_path = private/'rig.npz'; require_hash(rig_path, render.get('rig_sha256'))
    with np.load(rig_path, allow_pickle=False) as source: rig = {k:source[k].copy() for k in source.files}
    faces = np.asarray(rig['faces'])
    if (faces.shape != (36874,3) or faces.dtype.kind not in 'iu' or np.any(faces<0) or np.any(faces>=VERTICES)
            or not np.array_equal(faces, values['faces'])):
        raise ValueError('Private/public topology differs')
    regions = hand_partition(render['joint_names'], rig)
    report.update(proposals_sha256=receipt['proposals_sha256'], proposal_report_sha256=receipt_sha,
                  inference_report_sha256=sha256(infer_path), render_report_sha256=render_sha, rig_sha256=render['rig_sha256'],
                  predictions_frozen_before_private_truth_read=True, photorealism_verified=False, cases=[])
    for i, record in enumerate(render['cases']):
        truth_path = private/f'case_{i:03d}.npz'; require_hash(truth_path, record.get('truth_sha256'))
        require_hash(base/f'inputs/case_{i:03d}.png', record.get('rgb_sha256'))
        with np.load(truth_path, allow_pickle=False) as source: truth = {k:source[k].copy() for k in source.files}
        tv, tj = finite(truth['vertices_camera_m'], (VERTICES,3)), finite(truth['joints_camera_m'], (JOINTS,3))
        result = {'case_index':i, 'case_name':CASE_NAMES[i], 'truth_sha256':record['truth_sha256'], 'hands':{}}
        for side, region in regions.items():
            hand = {}; visible = visible_face_support(truth['visible_face_indices'], faces, region['mask'])
            for mode in ('baseline', 'candidate'):
                metrics, distance = errors(values[mode+'_vertices_camera_m'][i], values[mode+'_joints_camera_m'][i], tv, tj, region)
                metrics['visible_face_support_pve_mm'] = float(distance[visible].mean()) if visible.any() else None
                metrics['without_visible_face_support_pve_mm'] = float(distance[~visible].mean()) if (~visible).any() else None
                hand[mode] = metrics
            hand['vertices'] = int(region['mask'].sum()); hand['visible_face_supported_vertices'] = int(visible.sum())
            result['hands'][side] = hand
        report['cases'].append(result)
    report['decision'] = decision(report['cases'])
    require_hash(receipt_path, receipt_sha); require_hash(npz_path, receipt['proposals_sha256']); require_hash(render_path, render_sha)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Require remote isolated CPU evaluation')
    root = Path(os.environ['WR_ROOT']); revision = os.environ.get('WR_CODE_REVISION',''); image = os.environ.get('WR_IMAGE_ID','')
    if not re.fullmatch(r'[0-9a-f]{40}',revision) or not re.fullmatch(r'sha256:[0-9a-f]{64}',image):
        raise ValueError('Require immutable evaluation source/image')
    output = root/'validation/hands_rgb_v1/quality'; path = output/'report.json'
    reserved = os.environ.get('WR_HAND_EVAL_RESERVED') == '1'
    if output.is_symlink() or (output.exists() and (not reserved or not output.is_dir() or any(output.iterdir()))):
        raise FileExistsError('Quality evaluation output is frozen')
    if not reserved: output.mkdir()
    report = {'stage':'private_synthetic_hand_quality', 'status':'fail', 'code_revision':revision, 'image_id':image,
              'script_sha256':sha256(Path(__file__)), 'network':'none', 'gpu_used':False,
              'challenge_inputs_used':False, 'evaluation_truth_used':True, 'fitting_performed':False,
              'adoption_performed':False, 'challenge_performance_verified':False,
              'metrics_scope':'absolute camera hand PVE/MPJPE; wrist-relative articulation diagnostic, not scorer alignment',
              'visibility_scope':'vertices incident on raster-visible body faces, not exact per-vertex occlusion',
              'preregistered_gate':{'mean_nonneutral_pve_gain':.05,'neutral_side_max_regression_mm':.01}}
    started = time.perf_counter()
    try:
        run(root, report); report['status']='pass'
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)); raise
    finally:
        report['elapsed_seconds']=time.perf_counter()-started
        with path.open('x') as stream:json.dump(report,stream,indent=2,allow_nan=False)
    print(json.dumps({'stage':report['stage'],'status':report['status'],'decision':report['decision']}))


if __name__ == '__main__':main()
