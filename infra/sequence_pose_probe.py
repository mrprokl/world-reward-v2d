"""Azure CPU saved-only RGB pose ablation; never replaces a baseline/export.

Frozen external manufactured DEV and RESERVED diagnostics precede challenge QA.
Only original legal Track1 RGB and SHA-bound saved predictions are consumed.
No native model calls, source calibration, hidden labels, mesh or scale edits.
"""
from dataclasses import asdict
import hashlib
import json
import os
import re
import stat
import sys
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial import cKDTree
from world_reward.sequence_pose import SequencePoseConfig, initialize_missing_poses, refine_sequence
from mediapipe_cpu_runtime_verify import canonical, identity, source, strict

ROOT = Path('/srv/scenesmith/world-reward')
SOURCE = '052ba1554e9a573d566713a99a61d89a5f27681c'
ENTRY = 'run_sequence_pose_probe'
FRONTEND = 'b658881079871508c6b3ec14d001dc1299956996'
FRONTEND_NATIVE_PIN = dict(bytes=23225, sha256='6d1dea3a41e004d17bf6134bb74e78a2710f70607d130fdec5f051ed265c7813')
HELPERS = ('infra/sequence_pose_probe.py', 'infra/run_sequence_pose_probe.sh',
    'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/sequence_pose.py',
    'src/world_reward/point_surface_queries.py', 'src/world_reward/point_pose_cost.py',
    'src/world_reward/mesh_geometry.py', 'src/world_reward/__init__.py',
    'configs/sequence_pose_v2.json')
CFG = SequencePoseConfig(1.5, .05, .15, 20., 100., .1, .02, 60,
    'manufactured_nonchallenge_sequence_DEV_20261010_v1')


class ArtifactLedger:
    """Read-only source ledger; stable identity excludes read-induced access time."""
    def __init__(self):
        self.records = {}
        self.stats = {}

    @staticmethod
    def stable_stat(path):
        value = canonical(path).lstat()
        if not stat.S_ISREG(value.st_mode) or value.st_nlink != 1:
            raise ValueError('Canonical single-link regular artifact required')
        return tuple(getattr(value, key) for key in ('st_dev', 'st_ino', 'st_mode',
            'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink', 'st_uid', 'st_gid'))

    def record(self, path, expected=None, maximum=2 << 30):
        path = canonical(path); before = self.stable_stat(path)
        result = identity(path, maximum, readonly=False)
        if before != self.stable_stat(path):
            raise ValueError('Source changed while hashing')
        if expected is not None and result != expected:
            raise ValueError('Source SHA mismatch')
        key = str(path)
        if key in self.records and (result != self.records[key] or before != self.stats[key]):
            raise ValueError('Previously bound artifact changed')
        self.records[key], self.stats[key] = result, before
        return result

    def read(self, path, expected=None, maximum=4 << 20):
        path = canonical(path)
        pin = self.record(path, expected, maximum)
        raw = path.read_bytes()
        if (self.stable_stat(path) != self.stats[str(path)]
                or dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest()) != pin):
            raise ValueError('Source changed while reading')
        return raw

    def verify(self):
        for path, record in tuple(self.records.items()):
            self.record(Path(path), record)


def saved_frontend(ledger):
    """Independent native aggregate SHA authenticates clip-report descendants."""
    base = ROOT / 'results' / ('gemini-sam31-' + FRONTEND)
    aggregate = strict(ledger.read(base/'native-report.json', FRONTEND_NATIVE_PIN))
    if (aggregate.get('schema') != 'world_reward.gemini_sam31_tracking.v1'
            or aggregate.get('producer_revision') != FRONTEND
            or aggregate.get('status') != 'complete_diagnostic_not_quality_pass'
            or aggregate.get('ground_truth_used') is not False
            or aggregate.get('manual_labels') is not False
            or aggregate.get('original_frame_grid_preserved') is not True
            or [row.get('episode_index') for row in aggregate.get('episodes', [])] != [9, 1, 14, 7]):
        raise ValueError('Exact independently pinned full-T automatic frontend required')
    return base, {row['episode_index']: row for row in aggregate['episodes']}


def authenticated_masks(ledger, frontend_base, aggregate_row, base, episode, count):
    """Copied reports/inventory must match the independently pinned SAM producer."""
    original = frontend_base/f'episode_{episode:06d}'/'automatic_masks'
    if (aggregate_row.get('status') != 'pass' or aggregate_row.get('frames') != count
            or aggregate_row.get('automatic_masks') != str(original.relative_to(frontend_base))):
        raise ValueError('Saved automatic clip namespace/timeline differs')
    raw = ledger.read(original/'report.json', aggregate_row['report'])
    if ledger.read(base/'automatic_masks/report.json', aggregate_row['report']) != raw:
        raise ValueError('Copied frontend report differs')
    report = strict(raw)
    required = dict(stage='automatic_masks', status='pass', episode_index=episode,
        frames=count, producer_revision=FRONTEND, input_track='track_1', ground_truth_used=False,
        hand_labeled_test=False, manual_labels=False, oracle_modes=[], full_original_grid=True,
        fixed_native_ids=[0, 1], empty_mask_interpolation=False)
    if any(type(report.get(key)) is not type(value) or report[key] != value for key, value in required.items()):
        raise ValueError('Original full-T oracle-disabled native masks required')
    original_raw = ledger.read(original/'mask-inventory.json')
    copied_raw = ledger.read(base/'automatic_masks/mask-inventory.json')
    inventory = strict(original_raw)
    expected = {f'{role}/{index:06d}.png' for role in (0, 1) for index in range(count)}
    if (type(inventory) is not dict or set(inventory) != expected or copied_raw != original_raw
            or any(type(row) is not dict or set(row) != {'bytes', 'sha256'}
                or type(row['bytes']) is not int or not 0 < row['bytes'] <= 16 << 20
                or type(row['sha256']) is not str or not re.fullmatch('[0-9a-f]{64}', row['sha256'])
                for row in inventory.values())
            or report['mask_inventory'] != dict(files=2*count,
                bytes=sum(row['bytes'] for row in inventory.values()),
                sha256=hashlib.sha256(original_raw).hexdigest())):
        raise ValueError('Original complete mask inventory SHA differs')
    return inventory, report


def saved_numerical_frontend(ledger, experiment, frontend_rows):
    """Bind the original 052 transport receipt to the independent native masks."""
    report = strict(ledger.read(experiment/'report.json', maximum=8 << 20))
    required = dict(schema='world_reward.gemini_full4d.v1', producer_revision=SOURCE,
        status='complete_diagnostic_not_quality_pass', frontend_producer=FRONTEND,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        baseline_modified=False, per_frame_alignment=False, per_frame_scale=False,
        object_geometry_clip_constant=True)
    if any(type(report.get(key)) is not type(value) or report[key] != value for key, value in required.items()):
        raise ValueError('Original completed video-only numerical transport required')
    rows = report.get('episodes')
    if type(rows) is not list or [row.get('episode') for row in rows] != [9, 1, 14, 7]:
        raise ValueError('Original unreplaced numerical cohort required')
    inputs = report.get('inputs')
    if type(inputs) is not list or [item.get('episode') for item in inputs] != [9, 1, 14, 7]:
        raise ValueError('Original numerical RGB input ledger required')
    inputs = {item['episode']: item for item in inputs}
    selected = {}
    for row in rows:
        episode = row['episode']
        if episode not in (9, 14): continue
        frontend = row.get('frontend', {})
        native = frontend_rows[episode]
        original_input = inputs[episode]
        video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{episode:06d}.mp4'
        if (not row.get('status', '').startswith('complete_full4d')
                or row.get('original_frames') != native['frames']
                or original_input.get('total') != native['frames'] or original_input.get('video') != str(video)
                or type(original_input.get('video_pin')) is not dict
                or set(original_input['video_pin']) != {'bytes', 'sha256'}
                or type(original_input['video_pin']['bytes']) is not int or original_input['video_pin']['bytes'] <= 0
                or type(original_input['video_pin']['sha256']) is not str
                or not re.fullmatch('[0-9a-f]{64}', original_input['video_pin']['sha256'])
                or frontend.get('producer_revision') != FRONTEND
                or frontend.get('source_directory') != str(ROOT/'results'/('gemini-sam31-'+FRONTEND)
                    /f'episode_{episode:06d}'/'automatic_masks')
                or frontend.get('files', {}).get('report.json') != native['report']
                or frontend.get('aggregate_report') != FRONTEND_NATIVE_PIN
                or frontend.get('byte_identical_copy') is not True
                or frontend.get('hand_modified_masks') is not False
                or frontend.get('model_calls') != 0):
            raise ValueError('Original numerical frontend report/native aggregate differs')
        selected[episode] = dict(frontend, video_pin=original_input['video_pin'])
    if set(selected) != {9, 14}: raise ValueError('Both original completed saved exports required')
    return selected


def actual_fps(capture, source_fps=None):
    import cv2
    value = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(value) or value <= 0 or (source_fps is not None and not np.isclose(value, source_fps, atol=1e-6, rtol=1e-6)):
        raise ValueError('Positive decoder frame rate must match original source timing')
    return value


def lk_step(cv2, previous, gray, current):
    """Literal finite/status-supported FB subset; normal tracking loss abstains."""
    current = np.asarray(current)
    if current.ndim != 2 or current.shape[1:] != (2,) or current.dtype.kind != 'f' or not np.isfinite(current).all():
        raise ValueError('Finite supported OpenCV points required')
    current = current.astype(np.float32, copy=False)
    n = len(current); good = np.zeros(n, bool); output = np.full((n, 2), np.nan, np.float32)
    if not n: return output, good
    q, status, _ = cv2.calcOpticalFlowPyrLK(previous, gray, current.reshape(-1, 1, 2), None)
    if q is None or status is None: return output, good
    q, status = np.asarray(q), np.asarray(status)
    if q.shape != (n, 1, 2) or q.dtype.kind != 'f' or status.shape != (n, 1) or not np.isin(status, [0, 1]).all():
        raise ValueError('Invalid forward LK result')
    q = q[:, 0].astype(np.float32, copy=False); output[:] = q
    ids = np.flatnonzero(status[:, 0].astype(bool) & np.isfinite(q).all(1))
    if not len(ids): return output, good
    back, backward_status, _ = cv2.calcOpticalFlowPyrLK(gray, previous, q[ids].reshape(-1, 1, 2), None)
    if back is None or backward_status is None: return output, good
    back, backward_status = np.asarray(back), np.asarray(backward_status)
    if back.shape != (len(ids), 1, 2) or back.dtype.kind != 'f' or backward_status.shape != (len(ids), 1) or not np.isin(backward_status, [0, 1]).all():
        raise ValueError('Invalid backward LK result')
    good[ids] = backward_status[:, 0].astype(bool) & np.isfinite(back[:, 0]).all(1)
    good[ids] &= np.linalg.norm(back[:, 0]-current[ids], axis=1) <= 1.
    good &= np.isfinite(output).all(1)
    h, w = gray.shape
    good &= (output[:, 0] >= 0) & (output[:, 0] < w) & (output[:, 1] >= 0) & (output[:, 1] < h)
    return output, good


def pose_motion_summary(vertices, rotations, translations, fps):
    """Saved-prediction motion only; neither RGB truth nor an anti-jitter score."""
    centroid = np.einsum('tij,j->ti', rotations, vertices.mean(0))+translations
    linear = np.linalg.norm(np.diff(centroid, axis=0), axis=1)*fps
    angular = np.linalg.norm(Rotation.from_matrix(rotations[1:] @ rotations[:-1].swapaxes(-1, -2)).as_rotvec(), axis=1)*fps
    return dict(full_centroid_path_m=float(linear.sum()/fps),
        centroid_speed_m_s_median=float(np.median(linear)),
        centroid_speed_m_s_p95=float(np.quantile(linear, .95)),
        angular_speed_rad_s_median=float(np.median(angular)),
        angular_speed_rad_s_p95=float(np.quantile(angular, .95)),
        full_frames=len(translations), scope='motion_proxy_not_accuracy_or_quality_gate')


def fresh_runtime_output(out):
    """Only Docker's exact owned CID control file may precede native outputs."""
    out = canonical(out)
    if not out.is_dir() or {path.name for path in out.iterdir()} != {'.container.cid'}:
        raise ValueError('Fresh isolated output directory with owned CID required')
    cid = out/'.container.cid'
    identity(cid, 65, readonly=False)
    if re.fullmatch(b'[0-9a-f]{64}\n?', cid.read_bytes()) is None:
        raise ValueError('Exact owned Docker CID required')


def motion_gate(truth, old, new):
    """Full material-point path and displacement increments, not just endpoints."""
    delta = lambda value: np.diff(value, axis=0)
    path = lambda value: float(np.linalg.norm(delta(value), axis=-1).mean(axis=1).sum())
    true_path, new_path = path(truth), path(new)
    old_error = float(np.linalg.norm(delta(old)-delta(truth), axis=-1).mean())
    new_error = float(np.linalg.norm(delta(new)-delta(truth), axis=-1).mean())
    retention = None if true_path < 1e-8 else new_path/true_path
    return dict(full_surface_path_m=true_path, recovered_surface_path_m=new_path,
        full_path_retention=retention, old_motion_increment_error_m=old_error,
        new_motion_increment_error_m=new_error,
        passed=new_error <= old_error*1.05+1e-12 and (retention is None or .8 <= retention <= 1.2))


def project(p, r, t, k):
    xyz = p[None] @ r.swapaxes(-1, -2)+t[:, None]
    return xyz[..., :2]/xyz[..., 2, None]*k.diagonal()[:2]+k[:2, 2]


def manufactured_gate(seed, reserved, config=None):
    """Independent moving/static/occlusion/turn cases; truth never enters fit."""
    rng = np.random.default_rng(seed); rows = []
    for case in ('static', 'linear', 'acceleration', 'rotation', 'turn', 'occlusion'):
        n = 24; a = np.arange(n)/30
        v = rng.uniform(-.15, .15, (24, 3)); k = np.array([[256., 0, 128], [0, 256., 128], [0, 0, 1.]])
        t = np.zeros((n, 3)); t[:, 2] = 2.
        angle = np.zeros(n)
        if case in ('linear', 'occlusion'): t[:, 0] = .15*a
        if case == 'acceleration': t[:, 0] = .3*a*a
        if case == 'turn': t[:, 0] = .16*np.sin(5*a); t[:, 1] = .06*np.cos(5*a)
        if case == 'rotation': angle = 1.2*a
        if case == 'occlusion': t[:, 1] = .04*np.sin(7*a); angle = .5*a
        r = Rotation.from_rotvec(np.column_stack((a*0, angle, a*0))).as_matrix()
        noisy_t = t+rng.normal(0, .008, t.shape); noisy_t[0] = t[0]
        noisy_r = Rotation.from_rotvec(rng.normal(0, .02, (n, 3))).as_matrix() @ r; noisy_r[0] = r[0]
        xy = project(v, r, t, k)+rng.normal(0, .3 if not reserved else .5, (n, len(v), 2))
        support = np.ones(xy.shape[:2], bool)
        if case == 'occlusion': support[9:15] = False; xy[~support] = np.nan
        out = refine_sequence(v, v, xy, support, noisy_r, noisy_t, np.ones(n, bool), k,
                              np.arange(n), 30, CFG if config is None else config)
        truth = v[None] @ r.swapaxes(-1, -2)+t[:, None]
        old = v[None] @ noisy_r.swapaxes(-1, -2)+noisy_t[:, None]
        new = v[None] @ out.rotations.swapaxes(-1, -2)+out.translations[:, None]
        old_error = float(np.linalg.norm(old-truth, axis=2).mean())
        new_error = float(np.linalg.norm(new-truth, axis=2).mean())
        excursion = float(np.linalg.norm(truth[-1]-truth[0], axis=1).mean())
        recovered = float(np.linalg.norm(new[-1]-new[0], axis=1).mean())
        # Anti-collapse and actual error gates; no judging by low acceleration alone.
        passed = new_error <= old_error*.9 and (excursion < 1e-6 or .8 <= recovered/excursion <= 1.2)
        motion = motion_gate(truth, old, new)
        passed = passed and motion['passed']
        rows.append(dict(case=case, old_error_m=old_error, new_error_m=new_error,
                         motion_retention=None if excursion < 1e-6 else recovered/excursion,
                         pass_gate=bool(passed), converged=out.diagnostics['converged'], full_motion=motion))
    return dict(seed=seed, reserved=reserved, cases=rows, passed=all(r['pass_gate'] for r in rows),
        frozen_cfg_unchanged=True, original_seed_unchanged=True,
        converged=all(r['converged'] for r in rows),
        scope='manufactured_numerical_diagnostic_not_quality_or_probability_calibration')


def manufactured_missing_pose_gate(seed, config=None):
    """New v2 missing-pose protocol; never relabel old DEV/RESERVED receipts."""
    rng = np.random.default_rng(seed); n = 24; u = np.arange(n)/30
    vertices = rng.uniform(-.15, .15, (24, 3))
    camera = np.array([[256., 0, 128], [0, 256., 128], [0, 0, 1.]])
    rotation = Rotation.from_rotvec(np.column_stack((u*0, .5*u, u*0))).as_matrix()
    translation = np.column_stack((.15*u, .025*np.sin(3*u), 2.+u*0))
    proposal_t = translation+rng.normal(0, .008, translation.shape); proposal_t[0] = translation[0]
    proposal_r = Rotation.from_rotvec(rng.normal(0, .02, (n, 3))).as_matrix() @ rotation; proposal_r[0] = rotation[0]
    observed = np.ones(n, bool); observed[9:15] = False
    missing_r, missing_t = proposal_r.copy(), proposal_t.copy()
    missing_r[~observed] = np.nan; missing_t[~observed] = np.nan
    initial_r, initial_t = initialize_missing_poses(missing_r, missing_t, observed)
    tracks = project(vertices, rotation, translation, camera)+rng.normal(0, .5, (n, len(vertices), 2))
    visible = np.broadcast_to(observed[:, None], tracks.shape[:2]).copy(); tracks[~visible] = np.nan
    result = refine_sequence(vertices, vertices, tracks, visible, initial_r, initial_t, observed,
                             camera, np.arange(n), 30, CFG if config is None else config)
    surface = lambda r, t: vertices[None] @ r.swapaxes(-1, -2)+t[:, None]
    truth, old, new = surface(rotation, translation), surface(initial_r, initial_t), surface(result.rotations, result.translations)
    motion = motion_gate(truth, old, new)
    old_error = float(np.linalg.norm(old-truth, axis=-1).mean())
    new_error = float(np.linalg.norm(new-truth, axis=-1).mean())
    return dict(protocol='new_manufactured_missing_pose_full_motion_v2', seed=seed,
        original_DEV_RESERVED_unchanged=True, full_original_frames=n, missing_pose_frames=6,
        missing_observations_remain_nan=True, inferred_pose_not_observation=True,
        old_error_m=old_error, new_error_m=new_error, full_motion=motion,
        converged=result.diagnostics['converged'], passed=new_error <= .9*old_error and motion['passed'],
        external_quality_or_probability_calibration=False)


def profile_config(code, profile):
    """Explicit frozen external-development profiles; no challenge retuning."""
    if profile == 'v1': return CFG, (20261010, 20261011, 20261012)
    if profile != 'v2': raise ValueError('Only frozen v1/v2 profiles are allowed')
    value = strict((code/'configs/sequence_pose_v2.json').read_bytes())
    if (value['schema'] != 'world_reward.sequence_pose_v2.v1'
            or value['development_seed'] != 20261020
            or value['reserved_seed'] != 20261022
            or value['missing_pose_reserved_seed'] != 20261023
            or value['challenge_used_for_selection'] is not False):
        raise ValueError('Frozen non-challenge v2 protocol required')
    return SequencePoseConfig(**value['config']), (value['development_seed'],
        value['reserved_seed'], value['missing_pose_reserved_seed'])


def run(profile='v1'):
    started = time.monotonic(); revision = os.environ['WR_CODE_REVISION']; code = Path(os.environ['WR_CODE'])
    if ROOT != Path(os.environ['WR_ROOT']) or code != ROOT/'jobs'/revision/'run_sequence_pose_probe/code':
        raise ValueError('Exact Azure code/runtime source required')
    out = ROOT/'results'/('sequence-pose-probe-'+revision+('-v2' if profile == 'v2' else ''))
    fresh_runtime_output(out)
    ledger = ArtifactLedger(); pin = ledger.read; bound = ledger.records
    fit_config, seeds = profile_config(code, profile)
    report = dict(status='fail', profile=profile, producer_revision=revision, numerical_source=SOURCE, ground_truth_used=False,
                  manual_labels=False, full_4D_export_replaced=False, model_calls=0, config=asdict(fit_config))
    try:
        binding = source(ROOT, code, revision, ENTRY, HELPERS)
        report['source_binding'] = binding
        for name in HELPERS: pin(code/name)
        dev = manufactured_gate(seeds[0], False, fit_config); report['development'] = dev
        if not dev['passed']: raise ValueError('External DEV motion/error gate rejected; do not touch challenge')
        reserved = manufactured_gate(seeds[1], True, fit_config); report['reserved'] = reserved
        if not reserved['passed']: raise ValueError('External RESERVED gate rejected; no challenge adaptation')
        missing = manufactured_missing_pose_gate(seeds[2], fit_config)
        report['missing_pose_validation'] = missing
        if not missing['passed']: raise ValueError('Fresh missing-pose full-motion gate rejected')
        report['validation_convergence'] = 'converged' if dev['converged'] and reserved['converged'] and missing['converged'] else 'provisional_budget_limited_not_quality_qualified'
        frontend_base, frontend_rows = saved_frontend(ledger)
        experiment = ROOT/'experiments'/('full4d-v1-'+SOURCE)
        numerical_frontend = saved_numerical_frontend(ledger, experiment, frontend_rows)
        import cv2
        from PIL import Image
        report['episodes'] = []
        for ep in (9, 14):
            exp = ROOT/'experiments'/('full4d-v1-'+SOURCE); base = exp/'outputs'/f'episode_{ep:06d}'
            pins = json.loads(pin(exp/'pins'/f'cari_clip_{ep:06d}_shared_export_pins.json'))['export_files']
            directory = base/'cari_shared_export_v1'
            export_report = strict(pin(directory/'report.json', pins['report.json']))
            if (export_report['ground_truth_used'] is not False or export_report['oracle_modes'] != []
                    or export_report['status'] != 'pass' or export_report['producer_revision'] != SOURCE):
                raise ValueError('Original video-only source required')
            pin(directory/'trajectory.npz', pins['trajectory.npz'])
            with np.load(directory/'trajectory.npz', allow_pickle=False) as z: a = {k:z[k] for k in z.files}
            r, t, k = a['object_rotation'], a['object_translation'], a['camera_K']; n = len(t)
            if not np.array_equal(a['frame_index'], np.arange(n)) or float(a['object_scale']) != 1:
                raise ValueError('Fixed original full timeline/geometry required')
            inventory, mask_report = authenticated_masks(ledger, frontend_base, frontend_rows[ep], base, ep, n)
            if numerical_frontend[ep]['mask_inventory'] != mask_report['mask_inventory']:
                raise ValueError('052 frontend inventory differs from native SAM report')
            def original_mask(index):
                p = base/f'automatic_masks/masks/1/{index:06d}.png'
                pin(p,inventory[f'1/{index:06d}.png'])
                with Image.open(p) as image:
                    if image.mode != 'L' or image.size != (width, height):
                        raise ValueError('Literal original-resolution binary mask required')
                    array = np.asarray(image)
                    if not np.isin(array, [0, 255]).all(): raise ValueError('Mask is not native binary encoding')
                    return array > 0
            first_mask = base/'automatic_masks/masks/1/000000.png'
            pin(first_mask, inventory['1/000000.png'])
            with Image.open(first_mask) as image: width, height = image.size
            mask = original_mask(0)
            video = ROOT/'data/track_1/videos/chunk-000/observation.images.exo_camera'/f'episode_{ep:06d}.mp4'
            # Actual whole-video SHA, streamed remotely; never local video transit.
            video_pin = ledger.record(video)
            if video_pin['sha256'] != mask_report['input_sha256'] or numerical_frontend[ep]['video_pin'] != video_pin:
                raise ValueError('Original RGB differs from SAM/export source')
            capture = cv2.VideoCapture(str(video)); ok, frame = capture.read()
            if not ok or int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != n or frame.shape[:2] != mask.shape:
                raise ValueError('Original decoder count/grid differs')
            fps = actual_fps(capture)
            h, w = mask.shape
            if (export_report.get('frames') != n or export_report.get('original_frame_indices') != list(range(n))
                    or export_report.get('clip_spec') != dict(episode_index=ep, total_frames=n,
                        camera_name='front_stereo_camera_left', height=h, width=w)):
                raise ValueError('Original saved export RGB resolution/timeline differs')
            # Fixed 640x480 diagnostic, K scaled once. Native centres -.5 become OpenCV integer centres.
            size = (640, 480); scale = np.array([size[0]/w, size[1]/h])
            kk = k.copy(); kk[0] *= scale[0]; kk[1] *= scale[1]; kk[:2, 2] -= .5
            gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
            m = cv2.resize(mask.astype('uint8'), size, interpolation=cv2.INTER_NEAREST)
            m = cv2.erode(m, np.ones((3, 3), np.uint8))
            queries = cv2.goodFeaturesToTrack(gray, maxCorners=64, qualityLevel=.01, minDistance=5, mask=m)
            if queries is None or len(queries) < 8: raise ValueError('Insufficient RGB texture; no invented observations')
            queries = queries.reshape(-1, 2).astype(np.float32); count = len(queries)
            # Bind features to first-hit actual exported mesh triangles, not raw depth samples.
            from world_reward.point_surface_queries import _ray_triangle_hits
            v, faces = a['object_vertices'], a['object_faces'].astype(np.int64)
            original = (queries+.5)/scale-.5
            grid = np.column_stack((original[:, 1], original[:, 0]))
            from world_reward.mesh_geometry import normalize_degenerate_faces
            active_faces, face_diagnostics = normalize_degenerate_faces(v,faces)
            render_faces=faces[active_faces]
            _, depth, face_ids, bary = _ray_triangle_hits(v, render_faces, r[0], t[0], k, grid, np.ones(count, bool))
            keep = face_ids >= 0
            points = (v[render_faces[face_ids[keep]]]*bary[keep, :, None]).sum(1); queries = queries[keep]
            if len(points) < 8: raise ValueError('Insufficient canonical RGB attachments')
            tracks = np.full((n, len(points), 2), np.nan); visible = np.zeros(tracks.shape[:2], bool)
            tracks[0] = queries; visible[0] = True; active = np.arange(len(points)); previous = gray; current = queries.copy()
            for index in range(1, n):
                ok, frame = capture.read()
                if not ok: raise ValueError('Full original decoder coverage required')
                gray = cv2.cvtColor(cv2.resize(frame, size), cv2.COLOR_BGR2GRAY)
                if len(active):
                    q, good = lk_step(cv2, previous, gray, current)
                    current_mask = original_mask(index)
                    if current_mask.any():
                        small = cv2.resize(current_mask.astype('uint8'),size,interpolation=cv2.INTER_NEAREST)
                        supported = np.flatnonzero(good)
                        safe = np.floor(q[supported]+.5).astype(int)
                        inside = (safe[:, 0] < 640) & (safe[:, 1] < 480)
                        good[supported] = False
                        supported, safe = supported[inside], safe[inside]
                        good[supported] = small[safe[:,1],safe[:,0]]>0
                    else: good[:] = False  # No RGB reacquisition implemented in this saved-only diagnostic.
                    active, current = active[good], q[good]
                    tracks[index, active] = current; visible[index, active] = True
                previous = gray
            if capture.read()[0]: raise ValueError('Decoder has unexpected extra original frames')
            actual_fps(capture, fps)
            capture.release(); ledger.record(video, video_pin)
            # Predeclared visibility filter uses RGB support only, before any fitting/QA.
            retain = visible[1:].sum(0)>0; points=points[retain]; tracks=tracks[:,retain]; visible=visible[:,retain]
            if len(points)<8: raise ValueError('Insufficient persistent material tracks')
            fitted = refine_sequence(v, points, tracks, visible, r, t, np.ones(n,bool), kk,
                                     np.arange(n),fps,fit_config)
            path = out/f'episode_{ep:06d}.npz'
            np.savez_compressed(path, rotation=fitted.rotations, translation=fitted.translations,
                                frame_index=fitted.frame_index, object_vertices=v, object_faces=faces,
                                object_scale=a['object_scale'], camera_K=k, points=points,
                                tracks_xy=tracks, RGB_visible=visible)
            path.chmod(0o444)
            before_xy = project(points,r,t,kk); after_xy = project(points,fitted.rotations,fitted.translations,kk)
            before_error = float(np.linalg.norm(before_xy[visible]-tracks[visible],axis=1).mean())
            after_error = float(np.linalg.norm(after_xy[visible]-tracks[visible],axis=1).mean())
            target_path=directory/'target.npy'; ledger.record(target_path,pins['target.npy'])
            target=np.load(target_path,mmap_mode='r',allow_pickle=False)
            # Geometry-only human/object nearest-distance proxy on nine uniform frames; not true contact/penetration.
            contact=[]
            for index in np.linspace(0,n-1,9).astype(int):
                tree=cKDTree(np.asarray(target[index]))
                old=v@r[index].T+t[index]; new=v@fitted.rotations[index].T+fitted.translations[index]
                contact.append([int(index),float(tree.query(old)[0].min()),float(tree.query(new)[0].min())])
            report['episodes'].append(dict(episode=ep,frames=n,points=len(points),RGB_supported_frames=int(visible.any(1).sum()),
                before_reprojection_px=before_error,after_reprojection_px=after_error,
                nearest_human_surface_distance_proxy_m=contact,fit=fitted.diagnostics,output=ledger.record(path),
                original_motion=pose_motion_summary(v,r,t,fps),
                fitted_motion=pose_motion_summary(v,fitted.rotations,fitted.translations,fps),
                query_only_face_diagnostics=face_diagnostics,
                production_adopted=False,heldout_4D_accuracy_verified=False, fps=fps,
                occlusion_reacquisition_implemented=False))
        ledger.verify()
        if source(ROOT, code, revision, ENTRY, HELPERS) != binding: raise ValueError('Own complete source closure changed')
        report.update(status='complete_saved_RGB_diagnostic_not_quality_pass',sources=bound)
    except Exception as exc:
        report.update(error_type=type(exc).__name__,error=str(exc)[:400])
    report['elapsed_seconds']=time.monotonic()-started
    receipt=out/'report.json'; receipt.write_text(json.dumps(report,allow_nan=False,sort_keys=True)+'\n'); receipt.chmod(0o444)
    print(json.dumps({k:report.get(k) for k in ('status','error_type','error','elapsed_seconds','episodes','development','reserved')},allow_nan=False))
    if report['status']=='fail': raise SystemExit(1)


if __name__ == '__main__':
    if len(sys.argv) not in (1, 2): raise SystemExit('Only a frozen profile name is accepted')
    run('v1' if len(sys.argv) == 1 else sys.argv[1])
