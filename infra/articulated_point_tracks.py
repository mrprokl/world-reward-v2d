"""Six full authored RGB clips through unchanged native BootsTAPIR, Azure only.

The parent authenticates its complete immutable source, leased offline runtime
and previous manufacture stage, and seals the returned receipt. This stage reads
only six public RGB/query artifacts and their sealed report, never private truth
or an authored fit bundle. Model/runtime ancestry is reused, not a RoboTAP data
adapter. Raw model outputs and bound evidence retain every original slot/frame.
"""
from __future__ import annotations

import gc
import os
from pathlib import Path
import random
import sys
import time

IMAGE = 'sha256:ef12f589dd270e56be3a2d2e2f33ccd356e5b160a5c6ca03b8a9449ccc10d1e4'
BUDGET_SECONDS = 900
SOURCE_HELPERS = ('infra/articulated_point_tracks.py', 'infra/articulated_point_native.py',
    'infra/articulated_point_study.py', 'configs/articulated_runtime_replica_pins.json',
    'infra/tracker_noise_experiment.py', 'infra/robotap_boots_infer.py', 'infra/robotap_boots_acquire.py',
    'configs/robotap_boots_protocol.json', 'configs/robotap_boots_inference_pins.json',
    'src/world_reward/articulated_point_cohort.py', 'src/world_reward/point_surface_queries.py',
    'src/world_reward/fixed_shape_point_pose.py', 'src/world_reward/joint_point_evidence.py',
    'src/world_reward/joint_point_objective.py')


def _owned(array):
    import numpy as np
    return np.frombuffer(array.tobytes(order='C'), dtype=array.dtype).reshape(array.shape)


def _public_arrays(np, rt, path):
    """Strict original numerical public ABI; no pickle, resampling or query fit."""
    from world_reward.articulated_point_cohort import FRAMES
    from world_reward.contracts import require_rigid_transforms
    from world_reward.point_surface_queries import SurfaceQueries
    from world_reward.fixed_shape_point_pose import PointTrackEvidence
    from world_reward.joint_point_evidence import bind_joint_point_evidence
    with np.load(path, allow_pickle=False) as saved:
        expected = {'rgb', 'frame_index', 'source_frame_ids', 'vertices', 'faces', 'K', 'initial_pose'}
        expected |= set(SurfaceQueries.__dataclass_fields__)
        rt.require(set(saved.files) == expected and len(saved.files) == len(expected),
                   'Exact public RGB/initial-attachment array inventory required')
        arrays = {name: _owned(saved[name]) for name in saved.files}
    timeline = np.arange(FRAMES, dtype=np.int64)
    rt.require(arrays['rgb'].dtype == np.uint8 and arrays['rgb'].shape == (24, 480, 640, 3)
        and all(arrays[k].dtype == np.int64 and np.array_equal(arrays[k], timeline)
                for k in ('frame_index', 'source_frame_ids')),
        'All24 original uint8 RGB/full frame IDs required')
    rt.require(arrays['initial_pose'].dtype == np.float32 and arrays['initial_pose'].shape == (24, 4, 4)
        and arrays['K'].dtype == np.float64 and arrays['K'].shape == (3, 3),
        'Original complete FP32 object pose and full-image F64 camera required')
    require_rigid_transforms(arrays['initial_pose'], FRAMES)
    queries = SurfaceQueries(**{k: arrays[k] for k in SurfaceQueries.__dataclass_fields__})
    n = len(queries.face_indices)
    rt.require(8 <= n <= 32, 'All available original[8..32] query slots required; no refill')
    dummy = PointTrackEvidence(timeline, timeline, np.arange(n, dtype=np.int64), queries.query_points,
        np.broadcast_to(queries.query_points[:, [2, 1]] * np.array([256/640, 256/480]), (24, n, 2)).copy(),
        np.zeros((24, n), dtype=np.bool_))
    # This numerical preflight validates ALL supplied query/mesh associations;
    # dummy support is never used for inference or exported as a prediction.
    bind_joint_point_evidence(queries, dummy, native_vertices=arrays['vertices'], native_faces=arrays['faces'],
        K=arrays['K'], image_size=(480, 640), frame_index=timeline, source_frame_ids=timeline,
        native_frame_names=tuple(f'{i:06d}' for i in range(24)), source_references=('frozen-authored-public-preflight',))
    return arrays, queries


def _bind(arrays, queries, predicted, references):
    import numpy as np
    from world_reward.fixed_shape_point_pose import PointTrackEvidence
    from world_reward.joint_point_evidence import bind_joint_point_evidence
    tracks = PointTrackEvidence(predicted['frame_index'], arrays['source_frame_ids'],
        predicted['point_indices'], predicted['query_points'],
        np.transpose(predicted['tracks_256'], (1, 0, 2)).copy(), predicted['visible'].T.copy())
    bound = bind_joint_point_evidence(queries, tracks, native_vertices=arrays['vertices'], native_faces=arrays['faces'],
        K=arrays['K'], image_size=(480, 640), frame_index=arrays['frame_index'],
        source_frame_ids=arrays['source_frame_ids'], native_frame_names=references[0],
        source_references=references[1])
    return tracks, bound


def _save_arrays(np, rt, path, arrays):
    """Exclusive sealed numeric artifact, then lossless dtype/shape/byte reload."""
    with path.open('xb') as stream:
        os.fchmod(stream.fileno(), 0o444)
        np.savez(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    with np.load(path, allow_pickle=False) as saved:
        rt.require(set(saved.files) == set(arrays) and len(saved.files) == len(arrays), 'Saved inventory differs')
        for name, original in arrays.items():
            actual = saved[name]
            rt.require(actual.dtype == original.dtype and actual.shape == original.shape
                       and actual.tobytes(order='C') == original.tobytes(order='C'),
                       'Original native output/evidence bytes changed in serialization')
    return rt.identity(path, 1_000_000_000)


def native(rt, code, out, c):
    """One model load/six native calls; parent owns stage/provenance publication."""
    started = time.monotonic()
    check = lambda: rt.require(time.monotonic()-started <= BUDGET_SECONDS, 'Inclusive authored tracking deadline exhausted')
    study = c['articulated_study']
    rt.require(study['stage'] == 'tracks' and type(study['budgets_seconds']['tracks']) is int
        and study['budgets_seconds']['tracks'] == BUDGET_SECONDS, 'Explicit prospective900s authored tracks stage required')
    code, out = rt.canonical(Path(code)), rt.canonical(Path(out))
    rt.require(out.is_dir() and not tuple(out.iterdir()), 'Fresh empty owned tracking output required')
    import articulated_point_native as authored
    import robotap_boots_infer as boots
    import tracker_noise_experiment as ancestry
    for module, name in ((authored, 'articulated_point_native'), (boots, 'robotap_boots_infer'),
                         (ancestry, 'tracker_noise_experiment')):
        rt.require(Path(module.__file__).resolve() == code/'infra'/f'{name}.py', 'Actual bound stage helper origin differs')
    import articulated_point_study as controller
    rt.require(Path(controller.__file__).resolve() == code/'infra/articulated_point_study.py',
        'Actual source-bound runtime controller required')
    actual_image = controller.runtime_image_id(rt, code, c['_articulated_proof'], 'tracks')
    manufacture_path, manufacture = authored.authenticate_manufacture(rt, c)
    runtime = ancestry.runtime_evidence(code)
    check()
    rt.require(sys.platform == 'linux' and os.geteuid() == 1000 and 'torch' not in sys.modules
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'} and os.environ.get('WR_IMAGE_ID') == actual_image
        and os.environ.get('CUBLAS_WORKSPACE_CONFIG') == ':4096:8',
        'Fresh exact native Boots runtime/workspace required before Torch import')
    import numpy as np
    import torch
    rt.require(torch.cuda.is_available() and torch.__version__ == '2.5.1+cu124' and np.__version__ == '1.26.3',
               'Actual qualified FP32 CUDA runtime required; no fallback')
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4); torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False; torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(False, warn_only=False)
    from world_reward.articulated_point_cohort import SCENES
    views = manufacture['public_views']
    rt.require(len(views) == len(SCENES) == 6 and [r['scene_id'] for r in views] == [r[0] for r in SCENES]
        and [r['scene_index'] for r in views] == list(range(6)), 'All six fixed original public scenes required')
    public = []
    # Validate ALL six frozen scenes before the first model load/native call.
    for i, view in enumerate(views):
        path = manufacture_path/view['path']
        rt.require(view['path'] == f'scene_{i:02d}_public.npz' and rt.identity(path, 1_000_000_000) == view['identity'],
                   'Exact sealed public scene identity required')
        arrays, queries = _public_arrays(np, rt, path); public.append((path, view['identity'], len(queries.face_indices)))
        del arrays, queries; gc.collect(); check()
    report = dict(stage='articulated_point_tracks_native_v1', status='fail', phase='model_load',
        image_id=actual_image, qualified_config_id=IMAGE, frames=144, scene_ids=[r[0] for r in SCENES], model_loads=0,
        native_calls_attempted=0, native_calls_returned=0, native_calls_completed=0,
        manufacture_report_identity=c['articulated_study']['manufacture']['report'],
        policy_sha256=authored.policy_sha256(c), source_binding=c['_articulated_proof']['source_binding'],
        runtime_ancestry=runtime[1], outputs={}, scenes=[], calibration_performed=False,
        synthetic_queries_known=True, challenge_inputs_used=False, private_truth_read=False,
        conditional_not_end_to_end=True, quality_verified=False, adoption=False,
        numerical_runtime=dict(torch=str(torch.__version__), numpy=np.__version__, seed=0, TF32=False,
            deterministic_algorithms=False, exact_replay_verified=False),
        inference_config=dict(resolution=[256, 256], pyramid_level=1, query_chunk_size=32,
            is_training=False, compute_dtype='float32', AMP=False,
            preprocessing='unchanged native utils.bilinear/convert_grid_coordinates',
            visibility='(1-sigmoid(occlusion))*(1-sigmoid(expected_dist))>0.5',
            static_control_is_valid_motion_prediction=False))
    failure = None
    try:
        tapir, utils = boots.native_modules(boots.ROOT/boots.BASE/'assets/tapnet_source')
        model = boots.load_model(torch, tapir, boots.ROOT/boots.BASE/'assets/bootstapir_checkpoint_v2.pt')
        report['model_loads'] = 1; check()
        for i, (path, pin, n) in enumerate(public):
            report['phase'] = f'scene_{i:02d}'; check()
            rt.require(rt.identity(path, 1_000_000_000) == pin, 'Frozen RGB/query artifact changed')
            arrays, queries = _public_arrays(np, rt, path)
            row = dict(query_count=n, frames=24, height=480, width=640,
                       point_indices=list(range(n)), unavailable_original_indices=[])
            data = dict(video=arrays['rgb'], query_points=queries.query_points, point_indices=np.arange(n, dtype=np.int64))
            boots.validate_video(data, row)
            predicted = boots.native_prediction(torch, utils, model, data, report)
            torch.cuda.synchronize(); check()
            boots.validate_prediction(predicted, row, queries.query_points, data['point_indices'])
            raw_name, evidence_name = f'scene_{i:02d}_tracks.npz', f'scene_{i:02d}_evidence.npz'
            report['outputs'][raw_name] = _save_arrays(np, rt, out/raw_name, predicted)
            names = tuple(f'{t:06d}' for t in range(24))
            refs = ('authored conditional known geometry and camera', 'Boots native fullT automatic tracks')
            tracks, evidence = _bind(arrays, queries, predicted, (names, refs))
            report['outputs'][evidence_name] = _save_arrays(np, rt, out/evidence_name, vars(tracks))
            counts = evidence.support_counts()
            report['scenes'].append(dict(scene_index=i, scene_id=SCENES[i][0], split=SCENES[i][1],
                public_identity=pin, query_count=n, frames=24, frame_indices=list(range(24)),
                source_frame_ids=list(range(24)), raw_file=raw_name, evidence_file=evidence_name,
                evidence_sha256=evidence.evidence_sha256, source_references=list(refs),
                native_frame_names=list(names), image_size=list(evidence.image_size),
                frame_convention=evidence.frame_convention, after_initializer_support=counts.tolist()))
            rt.require(np.all(counts > 0), 'Every original query needs native-visible support after t0; no drop/refill')
            report['native_calls_completed'] += 1
            rt.require(rt.identity(path, 1_000_000_000) == pin, 'Original public input changed during tracking')
            del arrays, queries, predicted, tracks, evidence, data; gc.collect(); check()
        rt.require(report['native_calls_attempted'] == report['native_calls_returned'] == report['native_calls_completed'] == 6,
                   'Exactly six full-T native tracking calls required')
    except BaseException as error:
        failure = error; report.update(error_type=type(error).__name__, failure=str(error)[:400])
    finally:
        try:
            rt.require(authored.authenticate_manufacture(rt, c) == (manufacture_path, manufacture)
                and ancestry.runtime_evidence(code) == runtime, 'Frozen manufacture/runtime ancestry changed')
            report.update(manufacture_rehashed_after=True, runtime_rehashed_after=True)
            for name, pin in report['outputs'].items():
                rt.require(rt.identity(out/name, 1_000_000_000) == pin, 'Native prediction/evidence changed before stage seal')
            report['outputs_rehashed_after'] = True; check()
        except BaseException as error:
            failure = failure or error; report['post_error_type'] = type(error).__name__
        report.update(status='pass' if failure is None else 'fail', elapsed_seconds=time.monotonic()-started,
                      budget_seconds=BUDGET_SECONDS)
        if failure is None: report['phase'] = 'complete'
    return report
