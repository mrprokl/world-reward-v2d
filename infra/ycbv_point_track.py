"""One frozen RGB-only YCBV native25-pool versus Boots-association experiment.

All stage receipts/artifacts must first be independently pinned after real PASS.
No acquisition recipes, sensor calibration, reference geometry or private labels
enter the GPU. Missing provenance/budgets stop before model reads; no rerun path.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import signal
import stat
import time

import ycbv_point_depth as depth

ROOT, BASE, SCENES = depth.ROOT, depth.BASE, depth.SCENES
JOB, OUTPUT = 'run_ycbv_point_track', 'comparison_v1'
PIN_FILE = 'configs/ycbv_point_track_pins.json'
IMAGE = 'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
STAGE, TOTAL_BUDGET, FRAMES = 'public_ycbv_same_native25_pool_boots_point_comparison', 3600, 96
BOOTS_BASE = 'validation/robotap_boots_v1/assets'
BOOTS_RUNTIME = {'bytes': 3051, 'sha256': '794b6e5313e992c2c86610f855f4a86dc8c9a994ede37c22e33093e6ad9fc5e7',
    'producer_revision': 'f3cfde1992c6cfc9a6504b393d8a1f24735dc94d',
    'script_sha256': '5b2974b3e36ced5b8cbf67f6be02d6e9ee6e773403cd9575397f4a940206de9c'}
BOOTS_RUNTIME_PATH = 'results/bootstapir-runtime-verify-' + BOOTS_RUNTIME['producer_revision'] + '/report.json'
BOOTS_CHECKPOINT = {'bytes': 218886140, 'sha256': '8493c7a69e02c85b9382fbb3c7b8b539b36bc08ede744b9e99feb739a0129f4b'}
BOOTS_SOURCE = {
    'README.md': {'bytes': 22951, 'sha256': 'c0cda10a66a91513fd78e1638ff6db7789d4320dc24e5897ce6bc1bd567c15c3'},
    'LICENSE': {'bytes': 11358, 'sha256': 'cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30'},
    'tapnet/__init__.py': {'bytes': 1350, 'sha256': '35b29d7968272465e0f9cbb0d2c1035003ef96b140f4de7516eb85a047ec9ab3'},
    'tapnet/torch/__init__.py': {'bytes': 656, 'sha256': 'f7c8d8c66e2f93ac350ce82c03f02152ffc82524cd120ebc944c931b6ea879e1'},
    'tapnet/torch/tapir_model.py': {'bytes': 27917, 'sha256': '87c05fb17d48d0f4353a00b67c9d40d3a264e13e43d88e3f7b5829119845e3ba'},
    'tapnet/torch/nets.py': {'bytes': 12032, 'sha256': '4961a8e4882e8c9ac219c4714f6f2c811cc1ab665a439d4d2e79942002a9517c'},
    'tapnet/torch/utils.py': {'bytes': 10477, 'sha256': 'a4267b3a6bd8ecf2dfe5effacab9f7a43664dc5d08966c98f6a0cc68f5ccedd3'}}
SOURCE_FILES = ('infra/ycbv_point_track.py', 'infra/run_ycbv_point_track.sh', 'infra/ycbv_point_depth.py',
    'infra/tudl_holdout_inputs.py', 'infra/object_synthetic_observations.py', 'infra/camera_render.py',
    'infra/robotap_boots_infer.py', 'infra/robotap_boots_acquire.py', 'src/world_reward/__init__.py',
    'src/world_reward/data.py', 'src/world_reward/pointmap.py', 'src/world_reward/point_surface_queries.py',
    'src/world_reward/point_pose_cost.py', 'src/world_reward/point_candidate_pool.py',
    'src/world_reward/rigid_alignment.py', 'src/world_reward/point_pose_comparison.py', 'src/world_reward/pose_selection.py')
FROZEN = {**depth.FROZEN,
    'infra/ycbv_point_depth.py': '5a255115b2989dac6050b0e335a02d3f64722ecec2b491ecee8f0b83e69a55e2',
    'infra/robotap_boots_infer.py': '2339d6660ec2b751400b7ffe07a75b6abe9c4d48c3cd75a1725b92be65a6f826',
    'infra/robotap_boots_acquire.py': 'd4071356e04b086266cf3b9f1ca9f3359e0b2dff75ef0101d76d092f1eae0171',
    'infra/camera_render.py': '325fa1c42b6e620f93fe59d6e4f63852557a08343b13e2bea60eee8c88f4743c',
    'src/world_reward/point_surface_queries.py': 'aef81ccb023907cf6478637dd6a3f58fcb8ed9a0f323a47586e11b1d0a65936b',
    'src/world_reward/point_candidate_pool.py': '973d774cb5e85eccbf86f2acd224756fd2370abdff9751a6f18dc91ef0f69f8a',
    'src/world_reward/point_pose_comparison.py': 'c6c46a1968359812869a282041ef600abdfc071dc7865626af68b1bbd810a909',
    'src/world_reward/point_pose_cost.py': '2db8767ee43110c38370620ebc249e97e023f55cbb98111542e407709823142d',
    'src/world_reward/pose_selection.py': '663636ad7bb84ecf2093169313de0301bd2eef0f3a568dd90f84f05a1958549a',
    'src/world_reward/rigid_alignment.py': 'd165a6a353e5d5557fc6f3256bdb694d72979c64c9779e29627954907b2b3f51'}
STAGES = {
    'depth_init': ('depth_init_v1', depth.STAGE, 'run_ycbv_point_depth', 'infra/ycbv_point_depth.py', 300, 'GPU_budget_elapsed_seconds'),
    'masks': ('automatic_masks_v1', 'public_ycbv_point_native_object_masks', 'run_ycbv_point_masks', 'infra/ycbv_point_masks.py', 600, 'elapsed_seconds'),
    'objects': ('objects_init_v1', 'external_ycbv_three_anchor_native_Objects_initializer', 'run_ycbv_point_objects', 'infra/ycbv_point_objects.py', 900, 'GPU_budget_elapsed_seconds')}


def require(condition, message):
    if not condition: raise ValueError(message)


def pin(value, producer=False):
    keys = {'bytes', 'sha256'} | ({'producer_revision', 'script_sha256'} if producer else set())
    require(type(value) is dict and set(value) == keys, 'Exact independent identity keys required')
    depth.files._identity_record({k: value[k] for k in ('bytes', 'sha256')})
    if producer:
        for key, length in (('producer_revision', 40), ('script_sha256', 64)):
            require(type(value[key]) is str and re.fullmatch(fr'[0-9a-f]{{{length}}}', value[key]), 'Exact producer source identity required')


def stage_names(kind):
    if kind == 'depth_init': return {f'scene_{s:06d}_frame_000000.npz' for s in SCENES}
    if kind == 'masks': return {f'scene_{s:06d}/masks/1/{f:06d}.png' for s in SCENES for f in range(FRAMES)}
    return {f'scene_{s:06d}/{name}' for s in SCENES for name in ('object.glb', 'transform.json', 'intrinsics.json', 'canonical.npz')}


def validate_track_pins(value):
    require(type(value) is dict and set(value) == {'schema', 'input_pins', *STAGES}
        and value['schema'] == 'world-reward-ycbv-point-track-pins-v1', 'Actual independently frozen prerequisite pins required')
    pin(value['input_pins'])
    for kind in STAGES:
        row = value[kind]; require(type(row) is dict and set(row) == {'report', 'files'}, 'Exact prerequisite report/artifacts required'); pin(row['report'], True)
        require(type(row['files']) is dict and set(row['files']) == stage_names(kind), 'Full exact prerequisite artifact coverage required')
        for artifact in row['files'].values(): pin(artifact)
    return value


def checked_json(path, wanted):
    require(depth.files.identity(path) == {k: wanted[k] for k in ('bytes', 'sha256')}, 'Independent original bytes differ before JSON')
    raw = path.read_bytes()
    require({'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()} == {k: wanted[k] for k in ('bytes', 'sha256')}, 'Bound receipt changed before JSON')
    return depth.files.strict_json(raw)


def remaining_budget(reports):
    durations = {}
    for kind, (_, _, _, _, cap, field) in STAGES.items():
        value = reports[kind].get(field)
        require(type(value) in (int, float) and math.isfinite(value) and 0 < value <= cap
            and type(reports[kind].get('budget_seconds')) is int and reports[kind]['budget_seconds'] == cap, 'Missing/unsafe/overrun original GPU duration')
        durations[kind] = float(value)
    remaining = TOTAL_BUDGET - sum(durations.values())
    require(remaining > 0, 'No remaining preregistered whole-pilot GPU budget')
    return durations, remaining


def prerequisites(root, code):
    pins = validate_track_pins(depth.files.strict_json((code / PIN_FILE).read_bytes()))
    require(depth.files.identity(code / depth.PIN_FILE) == pins['input_pins'], 'Actual public input-pin config differs')
    public_pins = checked_json(code / depth.PIN_FILE, pins['input_pins']); records, public = depth.public_inputs(root / BASE / 'inputs', public_pins)
    reports = {}; artifacts = {}
    # ALL artifacts in ALL stages are hashed before ANY stage receipt is read.
    for kind, (folder, _, _, _, _, _) in STAGES.items():
        artifacts[kind] = {name: depth.files.identity(root / BASE / folder / name) for name in pins[kind]['files']}
        require(artifacts[kind] == pins[kind]['files'], 'Pinned prerequisite artifact changed')
        require(depth.files.identity(root / BASE / folder / 'report.json') == {k: pins[kind]['report'][k] for k in ('bytes', 'sha256')}, 'Pinned prerequisite receipt changed')
    for kind, (folder, stage, _, _, _, _) in STAGES.items():
        report = checked_json(root / BASE / folder / 'report.json', pins[kind]['report']); spec = pins[kind]['report']
        expected = dict(stage=stage, status='pass', phase='complete', producer_revision=spec['producer_revision'], script_sha256=spec['script_sha256'])
        require(type(report) is dict and all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()), 'Actual completed prerequisite producer required')
        require(report.get('challenge_inputs_used') is False and report.get('hand_labeled_test') is False and report.get('oracle_modes') == [], 'Automatic no-challenge/no-oracle prerequisite required')
        reports[kind] = report
    init, masks, objects = (reports[k] for k in ('depth_init', 'masks', 'objects'))
    require(init.get('input_pins') == public_pins and init.get('input_manifest') == public['manifest']
        and init.get('public_RGB_identities') == public['RGB_identities'] and init.get('private_truth_read') is False
        and all(init.get(k) is True for k in ('sources_after_reverified', 'all_public_RGB_after_reverified', 'model_assets_after_reverified', 'saved_native_arrays_byte_exact'))
        and all(type(init.get(k)) is int and init[k] == 3 for k in ('native_calls_attempted', 'native_calls_returned', 'native_calls_completed', 'outputs_completed')), 'Original native three-frame-zero evidence required')
    require(masks.get('public_manifest') == public['manifest'] and masks.get('public_RGB_identities') == public['RGB_identities']
        and masks.get('private_annotations_read') is False and masks.get('query') == 'object.'
        and masks.get('all_inputs_sources_assets_outputs_rehashed') is True and masks.get('disposable_RGB_stages_removed') is True
        and all(type(masks.get(k)) is int and masks[k] == 3 for k in ('detector_attempts', 'detector_calls', 'sam2_attempts', 'sam2_calls'))
        and type(masks.get('frames_completed')) is int and masks['frames_completed'] == 288, 'Original automatic full96 masks required')
    require(objects.get('ground_truth_used') is False and objects.get('private_annotations_read') is False
        and all(objects.get(k) is False for k in ('human_scalar_used', 'human_scale_used', 'sensor_depth_used', 'source_camera_calibration_used', 'hand_observations_used'))
        and objects.get('source_rehashed_after') is True
        and all(type(objects.get(k)) is int and objects[k] == 3 for k in ('native_calls_attempted', 'native_calls_returned', 'native_calls_completed', 'outputs_completed')), 'Original public native Objects evidence required')
    for kind, rows, field in (('depth_init', init.get('outputs'), 'file'), ('masks', masks.get('masks'), 'file')):
        require(type(rows) is list and len(rows) == len(stage_names(kind)), 'Prerequisite output inventory cardinality differs')
        for row, expected_record in zip(rows, records[::96] if kind == 'depth_init' else records):
            name = (Path(expected_record['file']).stem + '.npz') if kind == 'depth_init' else f"scene_{expected_record['scene_id']:06d}/masks/1/{expected_record['frame_id']:06d}.png"
            require(row.get(field) == name and type(row.get('scene_id')) is int and row['scene_id'] == expected_record['scene_id']
                and type(row.get('frame_id')) is int and row['frame_id'] == expected_record['frame_id']
                and row.get('rgb_sha256') == expected_record['sha256'] and {k: row.get(k) for k in ('bytes', 'sha256')} == artifacts[kind][name], 'Original RGB/artifact output linkage differs')
    require(type(objects.get('outputs')) is list and len(objects['outputs']) == 3, 'All original Objects outputs required')
    for scene, row in zip(SCENES, objects['outputs']):
        bundle = row.get('input_bundle'); require(type(bundle) is dict and bundle.get('scene_id') == scene, 'Original Objects triplet required')
        expected_paths = dict(rgb=f'{BASE}/inputs/scene_{scene:06d}_frame_000000.png', mask=f'{BASE}/automatic_masks_v1/scene_{scene:06d}/masks/1/000000.png', depth=f'{BASE}/depth_init_v1/scene_{scene:06d}_frame_000000.npz')
        for kind, relative in expected_paths.items():
            require(type(bundle.get(kind)) is dict and bundle[kind].get('path') == relative
                and {k: bundle[kind].get(k) for k in ('bytes', 'sha256')} == depth.files.identity(root / relative), 'Objects input was not exact native public RGB/mask/depth')
        require(row.get('scene_id') == scene and row.get('frame_id') == 0 and row.get('camera_unchanged') is True
            and row.get('files') == {name: artifacts['objects'][f'scene_{scene:06d}/{name}'] for name in ('object.glb', 'transform.json', 'intrinsics.json', 'canonical.npz')}
            and row.get('geometry', {}).get('scale_baked_once') is True and row.get('geometry', {}).get('source_triangles_retained') is True
            and row.get('geometry', {}).get('pose_applied_to_mesh') is False, 'Fixed original native geometry/camera proof required')
    durations, remaining = remaining_budget(reports)
    return records, pins, reports, {'public': public, 'artifacts': artifacts, 'prior_GPU_seconds': durations, 'remaining_GPU_seconds': remaining}


def source_binding(code, revision, container=False):
    depth.files._canonical(code); require(code == ROOT / 'jobs' / revision / JOB / 'code' and re.fullmatch(r'[0-9a-f]{40}', revision)
        and Path(__file__).resolve() == code / SOURCE_FILES[0] and Path(depth.__file__).resolve() == code / 'infra/ycbv_point_depth.py', 'Actual immutable track source required')
    wanted = {*SOURCE_FILES, PIN_FILE, depth.PIN_FILE}
    paths = [code / n for n in sorted(wanted)] if container else sorted(code.rglob('*'))
    if container: require({str(p.relative_to(code)) for p in code.rglob('*') if p.is_file()} == wanted, 'Only exact blind numeric/public pin closure permitted')
    values = {}
    for path in paths:
        depth.files._canonical(path); mode = path.lstat().st_mode
        require(not mode & 0o222 and (stat.S_ISDIR(mode) or stat.S_ISREG(mode)), 'Complete readonly canonical source required')
        if path.is_file(): values[str(path.relative_to(code))] = depth.files.identity(path)
    require(wanted <= values.keys() and all(values[n]['sha256'] == sha for n, sha in FROZEN.items()), 'Frozen original native numeric helper mismatch')
    markers = {n: depth.marker_identity(code.parent / n) for n in ('revision', 'source-sha256')}
    require((code.parent / 'revision').read_bytes() == (revision + '\n').encode() and re.fullmatch(b'[0-9a-f]{64}\n', (code.parent / 'source-sha256').read_bytes()), 'Original dispatch markers differ')
    return {'files': values, 'markers': markers}


def producer_proofs(root, pins, reports):
    result = {}
    for kind, (_, _, job, script, _, _) in STAGES.items():
        report = reports[kind]; p = pins[kind]['report']; original = root / 'jobs' / p['producer_revision'] / job / 'code'
        helpers = report.get('source_helpers', {}).get('files') if kind == 'depth_init' else report.get('source_binding', {}).get('helpers') if kind == 'masks' else report.get('source_helpers')
        require(type(helpers) is dict and script in helpers and helpers[script].get('sha256') == p['script_sha256'], 'Actual prerequisite producer source inventory missing')
        actual = {}
        for name, identity in helpers.items():
            require(type(name) is str and re.fullmatch(r'(?:infra/[a-z0-9_]+\.(?:py|sh)|configs/[a-z0-9_]+\.json|src/world_reward/[a-z0-9_/]+\.py)', name), 'Canonical code-only producer path required')
            pin(identity); actual[name] = depth.files.identity(original / name); require(actual[name] == identity, 'Original independently pinned producer helper changed')
        markers = {n: depth.marker_identity(original.parent / n) for n in ('revision', 'source-sha256')}
        require((original.parent / 'revision').read_bytes() == (p['producer_revision'] + '\n').encode()
            and re.fullmatch(b'[0-9a-f]{64}\n', (original.parent / 'source-sha256').read_bytes()), 'Original stage producer markers required')
        result[kind] = {'files': actual, 'markers': markers}
    return result


def boots_assets(root):
    native = {name: depth.files.identity(root / BOOTS_BASE / 'tapnet_source' / name) for name in BOOTS_SOURCE}
    require(native == BOOTS_SOURCE and depth.files.identity(root / BOOTS_BASE / 'bootstapir_checkpoint_v2.pt') == BOOTS_CHECKPOINT, 'Original licensed Boots source/checkpoint byte pins differ')
    return {'sources': native, 'checkpoint': BOOTS_CHECKPOINT}


def host_proof(root, code, revision):
    source = source_binding(code, revision); _, pins, reports, public = prerequisites(root, code)
    producer = producer_proofs(root, pins, reports); assets = boots_assets(root)
    runtime = checked_json(root / BOOTS_RUNTIME_PATH, BOOTS_RUNTIME); cpu = runtime.get('cpu_import', {})
    expected = dict(stage='bootstapir_runtime_verify', status='pass', phase='complete', producer_revision=BOOTS_RUNTIME['producer_revision'], child_image_id=IMAGE,
        native_source_revision='730cda1c730877cfedbe01bf87fb1cadb78a565d', original_build_status='fail', original_build_failure_preserved=True,
        source_rehashed_after=True, gpu_execution=False, checkpoint_read=False, rgb_or_labels_read=False)
    require(all(type(runtime.get(k)) is type(v) and runtime[k] == v for k, v in expected.items()) and runtime.get('native_sources') == BOOTS_SOURCE
        and runtime.get('source_helpers', {}).get('infra/run_bootstapir_runtime_verify.sh', {}).get('sha256') == BOOTS_RUNTIME['script_sha256']
        and cpu.get('python') == '3.11' and cpu.get('torch') == '2.5.1+cu124' and cpu.get('numpy') == '1.26.3'
        and all(cpu.get(k) is True for k in ('tree_cpu_verified', 'source_native_einshape_cpu_verified', 'native_bilinear_cpu_verified', 'native_tapir_modules_imported'))
        and cpu.get('models_instantiated') is False and cpu.get('cuda_initialized') is False, 'Actual native Boots CPU runtime proof required')
    return hashlib.sha256(json.dumps(dict(source=source, prerequisites=public, pins=pins, producer_sources=producer,
        boots_assets=assets, runtime=BOOTS_RUNTIME), sort_keys=True).encode()).hexdigest()


def freeze(array):
    import numpy as np
    value = np.array(array, copy=True); value.flags.writeable = False; return value


def native_initial_camera(initial):
    """Pixel K from original normalized FP32 output, never rounded to its prior."""
    import numpy as np
    checks = depth.validate_prediction_arrays(initial)
    return freeze(np.asarray(checks['pixel_K'], dtype=np.float64))


def verify_output_inventory(out, outputs):
    expected = {f'scene_{scene:06d}.npz' for scene in SCENES}
    require(type(outputs) is list and len(outputs) == 3 and {row.get('file') for row in outputs} == expected,
        'All three complete prediction artifact receipts required')
    require({p.name for p in out.iterdir()} == {'.container.cid', 'report.json', *expected}, 'Exact three frozen prediction output inventory required')
    for scene, row in zip(SCENES, outputs):
        require(row.get('file') == f'scene_{scene:06d}.npz' and type(row.get('scene_id')) is int and row['scene_id'] == scene
            and type(row.get('frames')) is int and row['frames'] == FRAMES
            and depth.files.identity(out / row['file']) == {k: row.get(k) for k in ('bytes', 'sha256')}, 'Original complete prediction bytes changed')


def load_geometry(data, K):
    import numpy as np
    require(type(data) is dict and set(data) == {'vertices', 'faces', 'R0', 't0', 'K', 'scale'}, 'Original canonical native geometry schema required')
    for name in ('vertices', 'R0', 't0', 'K', 'scale'):
        value = data[name]; require(type(value) is np.ndarray and value.dtype == np.float64 and np.isfinite(value).all(), 'Original canonical float64 arrays required')
    v, f = data['vertices'], data['faces']
    require(v.ndim == 2 and v.shape[1:] == (3,) and len(v) >= 4 and type(f) is np.ndarray and f.dtype == np.int64
        and f.ndim == 2 and f.shape[1:] == (3,) and len(f) >= 4 and (f >= 0).all() and (f < len(v)).all(), 'Full original native triangles required')
    with np.errstate(over='ignore', invalid='ignore', under='ignore'):
        triangles = v[f]; normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    require(np.isfinite(normals).all() and not np.any(np.all(normals == 0, axis=1)), 'Every original triangle must remain finite and nondegenerate; no deletion')
    require(data['R0'].shape == (3, 3) and data['t0'].shape == (3,) and data['K'].shape == (3, 3)
        and np.array_equal(data['K'], K) and data['scale'].shape == () and data['scale'].item() == 1.
        and np.allclose(data['R0'] @ data['R0'].T, np.eye(3), atol=1e-6, rtol=0) and np.isclose(np.linalg.det(data['R0']), 1., atol=1e-6, rtol=0), 'Clip-constant original camera/proper pose/scale1 required')
    return {name: freeze(value) for name, value in data.items()}


def compose_scene(geometry, initial, frames, masks, *, sample_surface, mesh_centroid, raster, track, align=None):
    """Pure injectable composition; production supplies unchanged native backends.

    Freeze original surface queries once; build all96 native25 slots before
    Boots/ranking. The iterator yields (original index, RGB, native pointmap).
    Observed tracks never enter ICP seeds, sample selection or pool generation.
    """
    import numpy as np
    from world_reward.point_surface_queries import canonical_surface_queries
    from world_reward.point_candidate_pool import NativeCandidatePool
    from world_reward.point_pose_comparison import compare
    K = geometry['K']; canonical = canonical_surface_queries(geometry['vertices'], geometry['faces'], geometry['R0'], geometry['t0'], K,
        masks[0], initial['mask'], image_width=depth.WIDTH, image_height=depth.HEIGHT)
    builder = NativeCandidatePool(np.arange(FRAMES, dtype=np.int64), geometry['vertices'], geometry['faces'], sample_surface,
        mesh_centroid, geometry['R0'], geometry['t0'], K, depth.WIDTH, depth.HEIGHT)
    video = []; coverage = []
    for index, rgb, points in frames:
        require(type(index) is int and index == len(video) and index < FRAMES and rgb.dtype == np.uint8 and rgb.shape == (depth.HEIGHT, depth.WIDTH, 3), 'All original ordered RGB/point frames required')
        builder.add_frame(index, points, masks[index], raster, align=align); video.append(rgb); coverage.append(index)
    require(coverage == list(range(FRAMES)), 'Full original frame coverage required before tracking')
    pool = builder.finalize()  # Both rankings consume this same frozen pool.
    tracks_256, visible, native_detail = track(np.stack(video), canonical.query_points)
    comparison = compare(pool, canonical.canonical_points, freeze(tracks_256), freeze(visible), K, depth.WIDTH, depth.HEIGHT)
    return comparison, canonical, native_detail


def native_boots(torch, native, utils, model, video, queries, report):
    import numpy as np
    before = depth.array_identities({'queries': queries})
    data = dict(video=video, query_points=queries, point_indices=np.arange(len(queries), dtype=np.int64))
    arrays = native.native_prediction(torch, utils, model, data, report); torch.cuda.synchronize(); report['native_calls_completed'] += 1
    row = dict(query_count=len(queries), frames=FRAMES, width=depth.WIDTH, height=depth.HEIGHT, point_indices=data['point_indices'].tolist())
    native.validate_prediction(arrays, row, queries, data['point_indices'])
    require(depth.array_identities({'queries': queries}) == before, 'Original automatic query order changed')
    return arrays['tracks_256'].transpose(1, 0, 2), arrays['visible'].T, {k: arrays[k] for k in ('tracks_256', 'visible', 'occlusion', 'expected_dist')}


def run(root, code, revision, out, report, persist):
    import numpy as np
    records, pins, reports, before = prerequisites(root, code); source = source_binding(code, revision, True)
    remaining = before['remaining_GPU_seconds']; started = time.monotonic(); deadline = started + remaining
    signal.alarm(math.ceil(remaining)); binding = None
    try:
        report.update(prerequisite_pins=pins, prerequisite_bindings=before, budget_remaining_seconds=remaining, phase='model_read'); persist()
        model_path, binding = depth.moge_bindings(root, code)
        require(binding == reports['depth_init'].get('MoGe_bindings'), 'Original MoGe checkpoint/source differs from initialization')
        boots_before = boots_assets(root)
        import torch
        import trimesh
        import camera_render
        import robotap_boots_infer as native
        from moge.model import v2 as moge
        from moge.utils import geometry_torch
        require(torch.cuda.is_available() and 'H100' in torch.cuda.get_device_name() and torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.26.3', 'Actual original H100 numeric runtime required')
        require(moge.recover_focal_shift is geometry_torch.recover_focal_shift, 'Original focal solver alias required')
        import pytorch3d
        require(pytorch3d.__version__ == '0.7.9', 'Original native PyTorch3D scalar raster required')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(False, warn_only=False)
        net = moge.MoGeModel.from_pretrained(str(model_path)).cuda().eval()
        tapir, utils = native.native_modules(root / BOOTS_BASE / 'tapnet_source')
        boots = native.load_model(torch, tapir, root / BOOTS_BASE / 'bootstapir_checkpoint_v2.pt')
        report.update(MoGe_bindings=binding, Boots_assets=boots_before, torch_version=torch.__version__, numpy_version=np.__version__,
            pytorch3d_version=pytorch3d.__version__, trimesh_version=trimesh.__version__, phase='full_native_pool'); persist()
        fov = float(np.degrees(2 * np.arctan(depth.WIDTH / (2 * float(np.hypot(depth.WIDTH, depth.HEIGHT))))))
        from PIL import Image
        def rgb(record):
            original = depth.files.identity(record['path'])
            require(original['sha256'] == record['sha256'], 'Original RGB changed before decode')
            with Image.open(record['path']) as image:
                require(image.format == 'PNG' and image.mode == 'RGB' and image.size == (depth.WIDTH, depth.HEIGHT), 'Original public RGB PNG required'); value = np.asarray(image).copy()
            require(depth.files.identity(record['path']) == original, 'Original RGB changed during decode')
            return value
        def raster(v, f, K, width, height): return camera_render.raster_camera_mesh(v, f, K, width, height)[0].cpu().numpy()
        for scene_index, scene in enumerate(SCENES):
            require(time.monotonic() < deadline, 'Remaining whole-pilot GPU budget exhausted')
            rows = records[scene_index * FRAMES:(scene_index + 1) * FRAMES]
            init_path = root / BASE / 'depth_init_v1' / (Path(rows[0]['file']).stem + '.npz')
            with np.load(init_path, allow_pickle=False) as archive: initial = {k: archive[k] for k in archive.files}
            K = native_initial_camera(initial)
            with np.load(root / BASE / 'objects_init_v1' / f'scene_{scene:06d}/canonical.npz', allow_pickle=False) as archive: geometry = load_geometry({k: archive[k] for k in archive.files}, K)
            mesh = trimesh.Trimesh(geometry['vertices'], geometry['faces'], process=False)
            mesh_audit = dict(watertight=bool(mesh.is_watertight), winding_consistent=bool(mesh.is_winding_consistent),
                signed_volume=float(mesh.volume) if np.isfinite(mesh.volume) else None,
                closedness_required=False, mesh_repaired=False, original_triangles_retained=True)
            sampled, _ = trimesh.sample.sample_surface(mesh, 8192, seed=0); masks = []
            for frame in range(FRAMES):
                with Image.open(root / BASE / 'automatic_masks_v1' / f'scene_{scene:06d}/masks/1/{frame:06d}.png') as image:
                    require(image.format == 'PNG' and image.mode == 'L' and image.size == (depth.WIDTH, depth.HEIGHT), 'Original automatic mask PNG required'); value = np.asarray(image).copy()
                require(value.dtype == np.uint8 and np.all((value == 0) | (value == 255)) and value.any(), 'Original native nonempty binary mask required'); masks.append(value > 0)
            def frames():
                for index, record in enumerate(rows):
                    value = rgb(record)
                    if index == 0:
                        expected = reports['depth_init']['outputs'][scene_index]['decoded_RGB_sha256']
                        require(hashlib.sha256(value.tobytes()).hexdigest() == expected, 'Initialization consumed different original RGB pixels')
                        arrays = initial  # Exactly reuse native initialization; never infer twice.
                    else:
                        tensor = torch.from_numpy(value).cuda().permute(2, 0, 1).float() / 255
                        report['MoGe_calls_attempted'] += 1; persist()
                        with torch.inference_mode(): prediction = net.infer(tensor[None], fov_x=fov)
                        report['MoGe_calls_returned'] += 1
                        arrays = {k: prediction[k][0].cpu().numpy() for k in ('depth', 'points', 'mask', 'intrinsics')}
                        from world_reward.pointmap import validate_camera_pointmap
                        require(all(arrays[k].dtype == np.float32 for k in ('depth', 'points', 'intrinsics'))
                            and arrays['mask'].dtype == np.bool_, 'Original native output dtypes required')
                        validate_camera_pointmap(arrays['depth'], arrays['points'], arrays['mask'], arrays['intrinsics'], K)
                        torch.cuda.synchronize(); report['MoGe_calls_completed'] += 1
                        del tensor, prediction
                    # Preserve every native valid/invalid XYZ; builder applies its ORIGINAL support rule.
                    yield index, value, arrays['points']
            def tracking(video, queries): return native_boots(torch, native, utils, boots, video, queries, report)
            comparison, canonical, tracks = compose_scene(geometry, initial, frames(), masks,
                sample_surface=sampled, mesh_centroid=np.asarray(mesh.centroid), raster=raster, track=tracking)
            arrays = {f'pool_{name}': value for name, value in vars(comparison.pool).items()}
            for branch in ('baseline', 'candidate'):
                path = getattr(comparison, branch)
                arrays.update({f'{branch}_{name}': getattr(path, name) for name in ('candidate_indices', 'rotations', 'translations')})
                poses = np.broadcast_to(np.eye(4), (FRAMES, 4, 4)).copy()
                poses[:, :3, :3] = path.rotations; poses[:, :3, 3] = path.translations
                arrays[f'{branch}_poses'] = poses
            arrays['frame_index'] = comparison.pool.frame_index
            arrays.update(point_costs=comparison.point_costs, candidate_costs=comparison.candidate_costs, visible_count=comparison.visible_count,
                no_visible_evidence=comparison.no_visible_evidence, canonical_points=canonical.canonical_points, query_points=canonical.query_points,
                query_grid_indices=canonical.grid_indices, query_face_indices=canonical.face_indices, **tracks)
            target = out / f'scene_{scene:06d}.npz'
            with depth.exclusive(target, 0o400) as stream: np.savez_compressed(stream, **arrays)
            identities = depth.array_identities(arrays)
            with np.load(target, allow_pickle=False) as archive: require(depth.array_identities({k: archive[k] for k in archive.files}) == identities, 'Saved full original native pool/tracks/paths changed')
            report['outputs'].append(dict(file=target.name, scene_id=scene, frames=FRAMES, queries=len(canonical.query_points),
                native_pixel_K=K.tolist(), raw_surface_audit=mesh_audit, **depth.files.identity(target), arrays=identities)); persist()
            del mesh, sampled, masks, initial, geometry, comparison, canonical, tracks, arrays; gc.collect(); torch.cuda.empty_cache()
        torch.cuda.synchronize()
        require(report['MoGe_calls_attempted'] == report['MoGe_calls_returned'] == report['MoGe_calls_completed'] == 285
            and report['native_calls_attempted'] == report['native_calls_returned'] == report['native_calls_completed'] == 3 and len(report['outputs']) == 3, 'All285 remaining depths/3full96 Boots/comparisons required')
        report.update(status='pass', phase='complete', full_original_frame_coverage=True, same_native_valid_pool=True,
            native_pool_frozen_before_tracking_and_rankings=True)
    finally:
        verified = True
        try:
            verified = source_binding(code, revision, True) == source and prerequisites(root, code)[3] == before
            if binding is not None: verified &= depth.moge_bindings(root, code)[1] == binding
            verified &= boots_assets(root) == {'sources': BOOTS_SOURCE, 'checkpoint': BOOTS_CHECKPOINT}
            if report['status'] == 'pass': verify_output_inventory(out, report['outputs'])
            else:
                for row in report['outputs']:
                    require(depth.files.identity(out / row['file']) == {k: row[k] for k in ('bytes', 'sha256')}, 'Completed partial output changed before failure seal')
        except Exception: verified = False
        report['all_inputs_models_sources_after_reverified'] = bool(verified)
        report['GPU_budget_elapsed_seconds'] = time.monotonic() - started
        report['whole_pilot_GPU_elapsed_seconds'] = sum(before['prior_GPU_seconds'].values()) + report['GPU_budget_elapsed_seconds']; signal.alarm(0)
        require(verified, 'Original source/public/model bindings failed post-verification')
        require(report['whole_pilot_GPU_elapsed_seconds'] <= TOTAL_BUDGET, 'Frozen3600s whole-pilot GPU budget exceeded')


def main(argv=None):
    parser = argparse.ArgumentParser(allow_abbrev=False); parser.add_argument('--host-proof', action='store_true'); args = parser.parse_args(argv)
    root, code, revision = Path(os.environ['WR_ROOT']), Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION']
    if args.host_proof: print(host_proof(root, code, revision)); return
    require(root == ROOT and os.geteuid() == 0 and os.uname().sysname == 'Linux' and os.environ.get('WR_IMAGE_ID') == IMAGE
        and os.environ.get('WR_AZURE_VM02_VERIFIED') == '1' and re.fullmatch(r'[0-9a-f]{64}', os.environ.get('WR_YCBV_HOST_PROOF_SHA256', ''))
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'}, 'Actual host-proven offline ROOT narrow CUDA container required')
    source = source_binding(code, revision, True)
    require(not any((root / n).exists() for n in (BASE + '/eval_private', BASE + '/report.json', 'data', 'vendor', 'validation/robotap_boots_v1/eval_private')), 'No challenge/private/calibration/recipe mounts permitted')
    out = depth.files._canonical(root / BASE / OUTPUT)
    require(out.is_dir() and {p.name for p in out.iterdir()} == {'.container.cid'} and out.stat().st_uid == 0 and out.stat().st_mode & 0o777 == 0o700, 'Fresh exclusive root-owned output required')
    report = dict(stage=STAGE, status='fail', phase='prerequisites', producer_revision=revision, script_sha256=source['files'][SOURCE_FILES[0]]['sha256'], source_helpers=source,
        image_id=IMAGE, execution_uid=0, network='none', host_proof_sha256=os.environ['WR_YCBV_HOST_PROOF_SHA256'],
        ground_truth_used=False, private_annotations_read=False, sensor_depth_used=False, source_camera_calibration_used=False, human_scale_used=False,
        hand_labeled_test=False, oracle_initial_queries=False, oracle_modes=[], challenge_inputs_used=False, training_overlap_verified=False,
        accuracy_verified=False, full_HOI_verified=False, CARI4D_victory_verified=False, adoption_performed=False, baseline_native_not_packed_track1=True,
        budget_seconds=TOTAL_BUDGET, MoGe_calls_attempted=0, MoGe_calls_returned=0, MoGe_calls_completed=0,
        native_calls_attempted=0, native_calls_returned=0, native_calls_completed=0, outputs=[])
    previous = {}
    def expired(*_): raise TimeoutError('Frozen remaining whole-pilot budget exhausted')
    with depth.exclusive(out / 'report.json', 0o400) as stream:
        def persist(): stream.seek(0); stream.write((json.dumps(report, allow_nan=False) + '\n').encode()); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        try:
            previous = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
            persist(); run(root, code, revision, out, report, persist)
        except Exception as error: report.update(status='fail', error_type=type(error).__name__, error='Frozen automatic point-pose comparison failed; private values omitted')
        finally:
            signal.alarm(0)
            for s, handler in previous.items(): signal.signal(s, handler)
            try: report['source_after_reverified'] = source_binding(code, revision, True) == source
            except Exception: report['source_after_reverified'] = False
            if not report['source_after_reverified']: report['status'] = 'fail'
            persist()
    if report['status'] != 'pass': raise SystemExit(1)


if __name__ == '__main__': main()
