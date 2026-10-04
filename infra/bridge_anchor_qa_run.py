"""Authenticated CPU-only QA of four frozen model observations, never truth.

Silhouette/hand-mask support is an engineering proxy, not independent anatomical,
metric, contact or temporal validation. No model, weights or renderer recipe IO.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bridge_frontend_bindings as binding

ENTRY = 'run_bridge_anchor_geometry_qa'
BASE = binding.BASE
OUTPUT = BASE + '/geometry_qa_v1'
OBSERVATIONS = BASE + '/observations_anchor_v1'
PINS = 'configs/bridge_rgb_anchor_observation_pins.json'
MASK_PINS = 'configs/bridge_rgb_anchor_mask_pins.json'
BUDGET = 180
HELPERS = ('infra/bridge_anchor_qa_run.py', 'infra/run_bridge_anchor_geometry_qa.sh',
    'infra/bridge_anchor_geometry_qa.py', 'infra/triangle_ray_gate.py', 'infra/run_triangle_ray_gate.sh',
    'infra/bridge_frontend_bindings.py', 'infra/frontend_selected_assets.py',
    'infra/frontend_sam2_kernel_gate.py', binding.CONFIG, binding.selected.PINS,
    binding.INPUT_PINS, MASK_PINS, PINS)
FILENAMES = tuple(Path(n).stem + '.npz' for n in binding.FILENAMES)
ARRAYS = {'global_rot', 'body_pose_params', 'hand_pose_params', 'scale_params',
    'shape_params', 'expr_params', 'vertices_camera_m', 'joints_camera_m',
    'joint_global_rotations_native', 'mhr_model_params', 'pred_cam_t', 'focal_length',
    'faces', 'depth', 'points', 'depth_validity', 'intrinsics_normalized', 'camera_K',
    'person_mask', 'object_mask', 'clip_id', 'frame_id'}
require = binding.require


def observations(root, pins_path):
    """Authenticate ALL five exact private-mode artifacts before receipt JSON."""
    directory = binding.canonical(root/OBSERVATIONS)
    ownpin = binding.identity(pins_path, 200000)
    pins = binding.strict_json(Path(pins_path).read_bytes())
    require(pins.get('schema') == 'world_reward.bridge_rgb_anchor_observation_pins.v1'
        and re.fullmatch('[0-9a-f]{40}', str(pins.get('producer_revision')))
        and re.fullmatch('[0-9a-f]{64}', str(pins.get('script_sha256'))), 'Frozen observation producer/script required')
    files = pins.get('files'); binding.validate_file_pins(files, {'report.json', *FILENAMES}, 20_000_000)
    require({p.name for p in directory.iterdir()} == set(files), 'Exact five original observations required')
    frozen = {Path(pins_path): ownpin}
    for name, pin in files.items():
        path = directory/name
        require(stat.S_IMODE(path.lstat().st_mode) == 0o400 and binding.identity(path, 20_000_000) == pin,
            'Original observations must be independently pinned readonly0400 bytes')
        frozen[path] = pin
    receipt = binding.strict_json((directory/'report.json').read_bytes())
    expected = dict(schema='world_reward.bridge_rgb_anchor_observations.v1', stage='bridge_rgb_anchor_model_observations',
        status='pass', phase='complete', producer_revision=pins['producer_revision'], script_sha256=pins['script_sha256'],
        image_id=binding.IMAGE, sources_and_inputs_rechecked=True, ground_truth_used=False,
        challenge_inputs_used=False, hand_labeled_test=False, oracle_modes=[], network='none',
        body_model_loads=1, MoGe_model_loads=1, body_attempts=4, body_returns=4,
        native_decode_returns=4, MoGe_attempts=4, MoGe_returns=4,
        geometry_frame='opencv_x_right_y_down_z_forward',
        joint_rotation_frame='native_MHR_local_axes_in_native_global_frame_NOT_camera',
        focal_prior='RGB_hypot_640_480_800_not_source_calibration', model_parameters_modified=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items()), 'Genuine completed original observation receipt required')
    binding.recheck(frozen)
    return pins, receipt, frozen


def controls(root, code, live=False):
    pins, receipt, frozen = observations(root, code/PINS)
    old_code = root/'jobs'/pins['producer_revision']/'run_bridge_rgb_anchor_infer/code'
    proof = binding.authenticate(root, old_code, 'run_bridge_rgb_anchor_infer', live=live)
    require(receipt.get('source_binding') == proof['source_binding']
        and proof['source_binding']['helpers']['infra/bridge_rgb_anchor_infer.py']['sha256'] == pins['script_sha256']
        and all(receipt.get(k) == proof[k] for k in ('build_report_identity', 'kernel_report_identity')), 'Original native inference source/proofs differ')
    kernel = binding.kernel_helper()
    current = kernel.closure(code, code.parent.parent.name, ENTRY, HELPERS)
    records, public = binding.public_inputs(root, code/binding.INPUT_PINS)
    masks, mask_files = binding.load_masks(root, code/MASK_PINS, records)
    original_public = {}
    for path, pin in (public | mask_files).items():
        original = old_code/path.relative_to(code) if path.is_relative_to(code) else path
        require(binding.identity(original) == pin, 'Current public pin config differs from original inference bytes')
        original_public[str(original)] = pin
    require(receipt.get('public_input_pins') == original_public, 'Original inference public RGB/mask pins differ')
    rows = receipt.get('frames'); solver = receipt.get('native_focal_solver')
    require(type(rows) is list and len(rows) == 4 and type(solver) is list and len(solver) == 4, 'All four original observations/solver returns required')
    for clip, (row, rgb, diagnostic) in enumerate(zip(rows, records, solver)):
        require(type(row) is dict and type(row.get('clip_id')) is int and row['clip_id'] == clip
            and type(row.get('frame_id')) is int and row['frame_id'] == 0 and row.get('completed') is True
            and row.get('rgb_sha256') == rgb['sha256'] and row.get('output') == dict(file=FILENAMES[clip], **pins['files'][FILENAMES[clip]])
            and diagnostic.get('clip_id') == clip and diagnostic.get('original_returned') is True
            and type(diagnostic.get('nearest64_valid_pixels')) is int and diagnostic['nearest64_valid_pixels'] >= 2,
            'Original frame identities/output pins or native solver differ')
    names = receipt.get('body_model', {}).get('named_metadata', {}).get('joint_names')
    require(type(names) is list and len(names) == 127 and all(type(n) is str and n for n in names)
        and len(set(names)) == 127, 'Same-model actual 127 unique joint names required')
    frozen.update(public | mask_files); binding.recheck(frozen)
    return dict(current_source=current, original_source=proof['source_binding'], image_id=binding.IMAGE,
        build_report_identity=proof['build_report_identity'], kernel_report_identity=proof['kernel_report_identity'],
        files={str(p): v for p, v in frozen.items()}), records, masks, names, frozen, old_code


def readonly_mounts(root, code, old_code):
    paths = (code.parent, old_code.parent, *binding.control_paths(),
        *(root/BASE/n for n in ('inputs', 'automatic_masks', 'observations_anchor_v1')))
    for path in paths:
        binding.canonical(path); require(path.exists(), 'Narrow CPU observation/proof mount absent')
    return list(dict.fromkeys(str(p) for p in paths))


def run(root, code, report, persist):
    import numpy as np
    from PIL import Image
    proof, _, masks, names, frozen, _ = controls(root, code)
    report.update(provenance=proof, phase='four_record_geometry_qa'); persist()
    import bridge_anchor_geometry_qa as quality
    require(Path(quality.__file__).resolve() == code/'infra/bridge_anchor_geometry_qa.py', 'Actual authenticated pure QA helper required')
    deadline = time.monotonic() + BUDGET
    for clip, mask_row in enumerate(masks):
        row = dict(clip_id=clip, frame_id=0, completed=False); report['frames'].append(row); persist()
        part = mask_row['person']
        with Image.open(root/BASE/'automatic_masks'/part['mask_file']) as png:
            require(png.format == 'PNG' and png.mode == 'L' and png.size == (640, 480), 'Original automatic person PNG required')
            mask = np.asarray(png).copy()
        require(np.isin(mask, [0, 255]).all() and np.count_nonzero(mask) == part['mask_pixels'], 'Original automatic mask binary values/area differ')
        with np.load(root/OBSERVATIONS/FILENAMES[clip], allow_pickle=False) as prediction:
            require(len(prediction.files) == len(ARRAYS) and set(prediction.files) == ARRAYS, 'Original exact native observation array ABI required')
            for name, value in (('clip_id', clip), ('frame_id', 0)):
                array = prediction[name]
                require(array.shape == () and array.dtype == np.int64 and int(array) == value, 'Original NPZ frame identity changed')
            require(prediction['person_mask'].dtype == np.bool_ and np.array_equal(prediction['person_mask'], mask > 0), 'Original observation automatic person mask differs')
            require(prediction['vertices_camera_m'].shape == (18439, 3) and prediction['joints_camera_m'].shape == (127, 3)
                and prediction['faces'].shape == (36874, 3) and prediction['faces'].dtype == np.int64,
                'Original native Body vertex/joint/face topology ABI required')
            result = quality.evaluate_geometry(prediction['vertices_camera_m'], prediction['joints_camera_m'],
                prediction['faces'], prediction['camera_K'], mask > 0, names,
                deadline=float(min(deadline, time.monotonic() + quality.BUDGET_SECONDS)))
        row.update(diagnostics=result, completed=True); persist()  # Every gate diagnostic survives FAIL.
    binding.recheck(frozen)
    require(controls(root, code)[0] == proof, 'Current/original source or observations changed during QA')
    report.update(sources_and_inputs_rechecked=True, all_frames_completed=True)
    require(all(r['diagnostics']['status'] == 'pass' for r in report['frames']), 'Frozen four-anchor engineering geometry gate failed; no rescue')
    report.update(status='pass', phase='complete')


def persist_run(out, report, operation):
    path = out/'report.json'; started = time.monotonic()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
    with os.fdopen(fd, 'w') as stream:
        def persist():
            report['elapsed_seconds'] = time.monotonic() - started
            stream.seek(0); json.dump(report, stream, allow_nan=False); stream.write('\n')
            stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError('CPU geometry QA budget exceeded')
        before = {s: signal.getsignal(s) for s in (signal.SIGALRM, signal.SIGTERM)}
        for s in before: signal.signal(s, expired)
        signal.alarm(BUDGET)
        try: persist(); operation(persist)
        except Exception as error:
            report.update(status='fail', error_type=type(error).__name__); raise
        finally:
            signal.alarm(0)
            try: persist()
            finally:
                for s, handler in before.items(): signal.signal(s, handler)
                require(stat.S_IMODE(path.lstat().st_mode) == 0o400, 'Owned QA report mode changed')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    modes = parser.add_mutually_exclusive_group(required=True)
    for mode in ('preflight', 'mounts', 'verify', 'run'): modes.add_argument('--'+mode, action='store_true')
    args = parser.parse_args(argv); root = binding.ROOT; code = binding.canonical(Path(os.environ['WR_CODE']))
    require(Path(__file__).resolve() == code/'infra/bridge_anchor_qa_run.py', 'Actual immutable QA driver required')
    if not args.run:
        proof, _, _, _, _, old = controls(root, code, live=True)
        if args.mounts:
            for path in readonly_mounts(root, code, old): print(path)
        else: print(json.dumps(proof, sort_keys=True, separators=(',', ':')))
        return
    require(sys.platform == 'linux' and os.geteuid() == 0 and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}
        and os.environ.get('WR_IMAGE_ID') == binding.IMAGE, 'Authenticated offline root-owned CPU container required')
    out = binding.canonical(root/OUTPUT)
    require(out.is_dir() and stat.S_IMODE(out.stat().st_mode) == 0o700 and not any(out.iterdir()), 'Fresh private CPU QA output required')
    report = dict(schema='world_reward.bridge_anchor_geometry_qa_run.v1', stage='bridge_anchor_geometry_qa', status='fail', phase='integrity',
        producer_revision=code.parent.parent.name, script_sha256=binding.identity(Path(__file__), 200000)['sha256'],
        budget_seconds=BUDGET, frames=[], network='none', models_loaded=False, ground_truth_used=False,
        challenge_inputs_used=False, independent_anatomical_keypoints_verified=False, metric_accuracy_verified=False,
        contact_verified=False, temporal_accuracy_verified=False, bridge_adoption_performed=False, submission_eligible=False)
    persist_run(out, report, lambda persist: run(root, code, report, persist))
    print(json.dumps(dict(stage=report['stage'], status=report['status'], frames=len(report['frames']), elapsed_seconds=report['elapsed_seconds'])))


if __name__ == '__main__': main()
