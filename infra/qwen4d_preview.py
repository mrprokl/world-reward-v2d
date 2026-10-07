"""Saved-only Azure QA: frozen historical 4D versus new three-frame initializers.

No model, fitting, camera replacement or prediction write occurs. The candidate
object is shown at frame zero only: it does not yet have a rigid trajectory.
The candidate human remains a frame-independent initializer, not shared 4D.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import stat
import time

import numpy as np

from reconstruction_preview import overlay, raster_depth, HUMAN_RGB, OBJECT_RGB
from world_reward.mesh_geometry import normalize_degenerate_faces

ROOT = Path('/srv/scenesmith/world-reward')
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
MODEL = 'Qwen/Qwen3-VL-8B-Instruct'
MODEL_REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
WIDTH, HEIGHT, MAX_JPEG_BYTES = 320, 240, 180_000


def require(value, message):
    if not value:
        raise ValueError(message)


def canonical(path):
    path = Path(path)
    require(path.is_absolute() and path.resolve() == path
        and not any(p.is_symlink() for p in (path, *path.parents)), 'Canonical path required')
    return path


def identity(path, maximum=1_000_000_000):
    path = canonical(path)
    before = path.lstat()
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1
        and 0 < before.st_size <= maximum, 'Bounded single-link ordinary file required')
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    after = path.lstat()
    fields = ('st_dev', 'st_ino', 'st_mode', 'st_size', 'st_mtime_ns', 'st_ctime_ns', 'st_nlink')
    require(all(getattr(before, k) == getattr(after, k) for k in fields), 'Source changed while hashing')
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def strict_json(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, 'Duplicate JSON field')
            result[key] = value
        return result
    def invalid(_):
        raise ValueError('Nonfinite JSON value')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


class Sources:
    """Bind only files actually consumed, before reading and again after QA."""
    def __init__(self):
        self.files = {}

    def bind(self, path, pin=None, maximum=1_000_000_000):
        path = canonical(path)
        actual = identity(path, maximum)
        require(pin is None or actual == pin, 'Frozen source identity differs')
        require(path not in self.files or self.files[path] == actual, 'Previously bound source changed')
        self.files[path] = actual
        return path

    def json(self, path, pin=None):
        path = self.bind(path, pin, 4 << 20)
        return strict_json(path.read_bytes())

    def verify(self):
        require(all(identity(path) == pin for path, pin in self.files.items()),
            'Read-only source changed during QA')


def frame_indices(total):
    require(type(total) is int and total >= 3, 'At least three original frames required')
    return [0, total // 2, total - 1]


def candidate_base(root, prefix, revision, episode):
    require(type(episode) is int and episode in (8, 9, 26), 'Frozen three-clip pilot only')
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision), 'Exact code revision required')
    require(prefix == f'experiments/qwen4d-v1-{revision}/outputs',
        'Explicit revision-isolated candidate outputs required, never baseline outputs')
    return canonical(canonical(root)/prefix/f'episode_{episode:06d}')


def report_contract(report, stage, episode, video_sha):
    expected = dict(stage=stage, status='pass', episode_index=episode, input_track='track_1',
        input_sha256=video_sha, ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()),
        'Exact video-only producer provenance required')


def camera_matrix(intrinsics, width, height):
    require(type(intrinsics) is dict and type(intrinsics.get('width')) is int
        and type(intrinsics.get('height')) is int
        and (intrinsics['width'], intrinsics['height']) == (width, height),
        'Original-resolution camera required')
    vals = [intrinsics.get(k) for k in ('fx', 'fy', 'cx', 'cy')]
    require(all(type(v) in (int, float) and np.isfinite(v) for v in vals)
        and vals[0] > 0 and vals[1] > 0, 'Finite positive camera intrinsics required')
    return np.array([[vals[0], 0, vals[2]], [0, vals[1], vals[3]], [0, 0, 1]], dtype=np.float64)


def shared_camera(focals, evidence, object_intrinsics, indices, width, height):
    focals = np.asarray(focals)
    require(focals.shape == (3,) and focals.dtype.kind in 'iuf'
        and np.isfinite(focals).all() and (focals > 0).all(), 'Three native focal lengths required')
    matrices = [np.array([[f, 0, width/2], [0, f, height/2], [0, 0, 1]], dtype=np.float64)
        for f in focals]
    require(type(evidence) is list and len(evidence) == 3
        and [r.get('frame_index') for r in evidence] == indices,
        'Ordered original-frame alignment evidence required')
    require(all(np.array_equal(matrices[0], K) for K in matrices)
        and all(np.array_equal(matrices[p], r.get('estimated_body_K')) for p, r in enumerate(evidence))
        and np.array_equal(matrices[0], camera_matrix(object_intrinsics, width, height)),
        'Shared camera differs; never override or realign it')
    return matrices[0]


def object_camera_vertices(vertices, transform):
    """Exact stored canonical scale once, then native wxyz rotation and camera translation."""
    from scipy.spatial.transform import Rotation
    vertices = np.asarray(vertices)
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 3
        and vertices.dtype.kind in 'iuf' and np.isfinite(vertices).all(), 'Finite canonical object required')
    require(type(transform) is dict, 'Stored object transform required')
    q, scale, translation = (np.asarray(transform.get(k), dtype=float)
        for k in ('rotation', 'scale', 'translation'))
    require(q.shape == (4,) and scale.shape == (3,) and translation.shape == (3,)
        and np.isfinite(np.concatenate((q, scale, translation))).all()
        and np.isclose(np.linalg.norm(q), 1., atol=1e-5, rtol=0) and (scale > 0).all(),
        'Finite native unit-wxyz transform and positive fixed scale required')
    rotation = Rotation.from_quat([q[1], q[2], q[3], q[0]]).as_matrix()
    result = (vertices * scale) @ rotation.T + translation
    require(np.isfinite(result).all() and (result[:, 2] > 1e-4).all(),
        'Complete object must remain in front of the camera; no clipping')
    return result


def render_depth(vertices, faces, camera):
    active, receipt = normalize_degenerate_faces(vertices, faces)
    require(len(active) > 0, 'Nonempty unchanged object/human surface required')
    return raster_depth(vertices, np.asarray(faces)[active], camera, WIDTH, HEIGHT), {
        key: receipt[key] for key in ('input_vertices', 'input_faces', 'active_faces',
        'excluded_faces', 'input_arrays_modified', 'components_selected_or_removed')}


def mask_overlay(rgb, human, obj):
    require(human.dtype == np.bool_ and obj.dtype == np.bool_ and human.shape == obj.shape
        and rgb.shape == (*human.shape, 3), 'Same-viewport boolean masks required')
    color = (human[..., None] * np.asarray(HUMAN_RGB) + obj[..., None] * np.asarray(OBJECT_RGB))
    count = human.astype(int) + obj.astype(int)
    color = color / np.maximum(count[..., None], 1)
    output = rgb.astype(float).copy()
    present = count > 0
    output[present] = .48 * output[present] + .52 * color[present]
    return np.rint(output).astype(np.uint8)


def load_baseline(root, code, episode, total, camera, sources):
    """Only independently frozen safe NumPy exports; no pickle or producer replay."""
    pins = sources.json(code/f'configs/cari_clip_{episode:06d}_shared_export_pins.json')
    spec = pins.get('clip_spec', {})
    require(pins.get('schema') == 'world-reward-cari-shared-export-pins-v1'
        and spec == dict(episode_index=episode, total_frames=total,
            camera_name='front_stereo_camera_left', height=1152, width=1536),
        'Exact historical full-video export pins required')
    base = canonical(root/f'outputs/episode_{episode:06d}/cari_shared_export_v1')
    require(set(pins.get('export_files', {})) == {'report.json', 'target.npy',
        'trajectory.npz', 'native_parameters.npz', 'object_aligned.glb'}, 'Exact baseline export inventory required')
    for name, pin in pins['export_files'].items():
        sources.bind(base/name, pin)
    report = sources.json(base/'report.json')
    expected = dict(stage='world_reward_native_cari_shared_full_video_direct_export',
        status='pass', phase='complete', episode_index=episode, clip_spec=spec,
        input_track='track_1', ground_truth_used=False, ground_truth_read=False,
        private_truth_read=False, hand_labeled_test=False, oracle_modes=[],
        unchanged_refined_predictions_verified=True, full_original_native_export_verified=True)
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items()),
        'Frozen original baseline export provenance required')
    target = np.load(base/'target.npy', mmap_mode='r', allow_pickle=False)
    with np.load(base/'trajectory.npz', allow_pickle=False) as data:
        trajectory = {k: data[k] for k in ('object_vertices', 'object_faces', 'object_rotation',
            'object_translation', 'object_scale', 'camera_K', 'frame_index')}
    with np.load(base/'native_parameters.npz', allow_pickle=False) as data:
        faces, indices = data['human_faces'], data['frame_index']
    require(target.shape == (total, 18439, 3) and not target.flags.writeable
        and np.array_equal(indices, np.arange(total))
        and np.array_equal(trajectory['frame_index'], np.arange(total))
        and trajectory['object_scale'].shape == () and float(trajectory['object_scale']) == 1.
        and np.array_equal(trajectory['camera_K'], camera),
        'Original full-T geometry and exact common camera required; no alignment')
    return target, faces, trajectory


def run(root, code, revision, prefix, episode, output):
    started = time.perf_counter()
    root, code, output = map(canonical, (root, code, output))
    base = candidate_base(root, prefix, revision, episode)
    require(output.is_dir() and output.stat().st_uid == os.getuid() and not any(output.iterdir())
        and output.is_relative_to(root/'experiments'/f'qwen4d-v1-{revision}'),
        'Fresh empty owned preview directory inside the candidate experiment required')
    sources = Sources()
    require(Path(__file__).resolve() == code/'infra/qwen4d_preview.py'
        and Path(raster_depth.__code__.co_filename).resolve() == code/'infra/reconstruction_preview.py'
        and Path(normalize_degenerate_faces.__code__.co_filename).resolve()
            == code/'src/world_reward/mesh_geometry.py', 'Actual source helper origins differ')
    sources.bind(Path(__file__).resolve(), maximum=1 << 20)
    sources.bind(code/'infra/reconstruction_preview.py', maximum=1 << 20)
    sources.bind(code/'infra/camera_render.py', maximum=1 << 20)
    sources.bind(code/'src/world_reward/mesh_geometry.py', maximum=1 << 20)
    manifest = sources.json(root/'results/input-manifest.json')
    require((manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) ==
        ('track_1', 'nvidia/video_to_data_challenge', DATASET), 'Pinned original Track1-only manifest required')

    def original(relative):
        records = [r for r in manifest['files'] if r.get('path') == relative]
        require(len(records) == 1, 'Unique original input manifest record required')
        return sources.bind(root/'data'/relative, {k: records[0][k] for k in ('bytes', 'sha256')})

    episodes = original('track_1/meta/episodes.jsonl')
    rows = [strict_json(line) for line in episodes.read_bytes().splitlines() if line.strip()]
    rows = [r for r in rows if type(r.get('episode_index')) is int and r['episode_index'] == episode]
    require(len(rows) == 1, 'Unique official episode record required')
    total = rows[0]['length']; indices = frame_indices(total)
    video = original(f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4')
    video_sha = sources.files[video]['sha256']
    mask_dir = base/'automatic_masks'
    mask_report = sources.json(mask_dir/'report.json')
    body_report_path = base/'body_smoke/report.json'
    body = sources.json(body_report_path)
    scale_path = base/'scale_smoke/report.json'; scale = sources.json(scale_path)
    object_dir = base/'object_grounded'; obj_report = sources.json(object_dir/'report.json')
    for report, stage in ((mask_report, 'automatic_masks'), (body, 'sam3d_body_three_frame_smoke'),
            (scale, 'predicted_human_anchored_moge2_pointmaps'), (obj_report, 'sam3d_objects_grounded_fixed_frame')):
        report_contract(report, stage, episode, video_sha)
    for report, script in ((body, 'body_smoke.py'), (scale, 'scale_smoke.py'),
            (obj_report, 'object_smoke.py')):
        path = sources.bind(code/'infra'/script, maximum=1 << 20)
        require(report.get('script_sha256') == sources.files[path]['sha256'],
            'Candidate producer code differs from the immutable source')
    require(mask_report.get('frames') == total and mask_report.get('grounding_model') == MODEL
        and mask_report.get('grounding_model_revision') == MODEL_REVISION
        and mask_report.get('adapter_only') is True and mask_report.get('quality_verified') is False,
        'Original Qwen-grounded full-T automatic masks required')
    inventory_path = mask_dir/'mask-inventory.json'
    inventory = sources.json(inventory_path, mask_report['mask_inventory_identity'])
    prompts_path = sources.bind(mask_dir/'prompts.json', mask_report['prompts_identity'])
    require(body.get('geometry_units') == 'metres' and body.get('mhr_geometry_forward_verified') is True
        and body.get('frame_indices') == indices and body.get('total_video_frames') == total
        and body.get('mask_report_sha256') == sources.files[mask_dir/'report.json']['sha256']
        and body.get('prompts_sha256') == sources.files[prompts_path]['sha256']
        and body.get('human_identity_clip_constant') is False
        and body.get('frame_independent_initializer_only') is True
        and scale.get('frame_indices') == indices
        and scale.get('body_report_sha256') == sources.files[body_report_path]['sha256'],
        'Verified new independent human initialization and shared-scale lineage required')
    grounding = obj_report.get('pointmap_grounding') or {}
    require(obj_report.get('frame_index') == 0
        and obj_report.get('scale_source') == 'already_human_anchored_MoGe2_no_second_scalar'
        and grounding.get('alignment_report_sha256') == sources.files[scale_path]['sha256']
        and scale.get('coordinate_frame') == 'OpenCV_x_right_y_down_z_forward'
        and scale.get('pointmap_scale_application') == 'one_clip_scalar_to_MoGe2_XYZ_already_applied',
        'One already-human-anchored object gauge; no second scale permitted')
    require(grounding.get('intrinsics_path') == str(base/'scale_smoke/000000_intrinsics.json')
        and grounding.get('pointmap_path') == str(base/'scale_smoke/000000.npy'),
        'Object grounding must use the new candidate namespace, never old target-conditioned assets')
    prediction = sources.bind(base/'body_smoke/predictions.npz')
    require(sources.files[prediction]['sha256'] == body['predictions_sha256'], 'Verified body prediction bytes differ')
    with np.load(prediction, allow_pickle=False) as data:
        vertices, faces, focals = data['vertices_camera_m'], data['faces'], data['focal_length']
        require(vertices.shape == (3, 18439, 3) and np.array_equal(data['frame_index'], indices),
            'Exact three original-frame body geometries required')
    for name, key in (('object.glb', 'object_sha256'), ('transform.json', 'transform_sha256'),
            ('intrinsics.json', 'intrinsics_sha256')):
        path = sources.bind(object_dir/name)
        require(sources.files[path]['sha256'] == obj_report[key], 'Verified object artifact bytes differ')
    transform = sources.json(object_dir/'transform.json')
    require(transform == obj_report['transform'], 'Stored native object transform differs')
    intrinsics = sources.json(object_dir/'intrinsics.json')
    source_intrinsics = sources.json(Path(grounding['intrinsics_path']))
    require(sources.files[Path(grounding['intrinsics_path'])]['sha256'] == grounding.get('intrinsics_sha256')
        and intrinsics == source_intrinsics, 'Object must preserve the new grounding camera exactly')
    width, height = 1536, 1152
    camera = shared_camera(focals, scale['human_evidence'], intrinsics, indices, width, height)
    import trimesh
    mesh = trimesh.load(object_dir/'object.glb', force='mesh', process=False)
    require(isinstance(mesh, trimesh.Trimesh), 'Unchanged canonical GLB mesh required')
    obj_vertices = object_camera_vertices(mesh.vertices, transform)
    baseline, baseline_faces, trajectory = load_baseline(root, code, episode, total, camera, sources)
    display_camera = camera.copy()
    display_camera[0] *= WIDTH/width; display_camera[1] *= HEIGHT/height
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    sheet = Image.new('RGB', (WIDTH*3, 84 + (HEIGHT+26)*4), (24, 26, 31))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 14)
    draw.text((8, 6), f'World Reward | episode {episode:02d} | FIRST 3D QA GATE - NOT full 4D / NOT a score', font=font, fill=(235,235,235))
    draw.text((8, 29), 'Cyan: human | Orange: object | Historical full 4D vs NEW frame-independent initializers', font=font, fill=(235,235,235))
    draw.text((8, 52), 'NEW object: frame 0 ONLY; absent later until rigid tracking. No fitting / no camera realignment.', font=font, fill=(235,235,235))
    body_frames = body['frames']
    require([r['frame_index'] for r in body_frames] == indices, 'Ordered original body evidence required')
    render_records = []
    capture = cv2.VideoCapture(str(video))
    try:
        require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total,
            'Decoder original frame count differs')
        for position, index in enumerate(indices):
            require(capture.set(cv2.CAP_PROP_POS_FRAMES, index), 'Exact original frame seek failed')
            ok, bgr = capture.read()
            require(ok and bgr.shape == (height, width, 3)
                and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == index+1, 'Exact original frame decode failed')
            full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb_sha = hashlib.sha256(full_rgb.tobytes()).hexdigest()
            require(rgb_sha == body_frames[position]['decoded_rgb_sha256']
                == scale['human_evidence'][position]['decoded_rgb_sha256'], 'Saved evidence original RGB differs')
            if index == 0:
                require(rgb_sha == obj_report['decoded_rgb_sha256'], 'Object original RGB differs')
            masks = []
            for object_id in ('0', '1'):
                key = f'{object_id}/{index:06d}.png'
                path = sources.bind(mask_dir/'masks'/key, inventory[key], 16 << 20)
                with Image.open(path) as image:
                    array = np.asarray(image)
                require(array.shape == (height, width) and np.isin(array, [0, 255]).all()
                    and (array > 0).any(), 'Nonempty original-resolution binary automatic mask required')
                masks.append(np.asarray(Image.fromarray(array).resize((WIDTH, HEIGHT),
                    Image.Resampling.NEAREST)) > 0)
            human_path = mask_dir/f'masks/0/{index:06d}.png'
            require(sources.files[human_path]['sha256'] == body_frames[position]['mask_sha256'],
                'New human mask differs from body inference')
            if index == 0:
                require(sources.files[mask_dir/'masks/1/000000.png']['sha256'] == obj_report['mask_sha256'],
                    'New object mask differs from object inference')
            rgb = cv2.resize(full_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
            old_obj = trajectory['object_vertices'] @ trajectory['object_rotation'][index].T + trajectory['object_translation'][index]
            old_human_depth, old_human_view = render_depth(baseline[index], baseline_faces, display_camera)
            old_obj_depth, old_obj_view = render_depth(old_obj, trajectory['object_faces'], display_camera)
            new_human_depth, new_human_view = render_depth(vertices[position], faces, display_camera)
            new_obj_depth = np.full((HEIGHT, WIDTH), np.inf)
            new_obj_view = None
            if index == 0:
                new_obj_depth, new_obj_view = render_depth(obj_vertices, mesh.faces, display_camera)
            images = [(rgb, 'Original RGB'), (mask_overlay(rgb, *masks), 'NEW Qwen3-VL + SAM2 masks'),
                (overlay(rgb, old_human_depth, old_obj_depth), 'Historical full 4D'),
                (overlay(rgb, new_human_depth, new_obj_depth),
                    'NEW human + object init' if index == 0 else 'NEW human - object untracked')]
            for row, (image, label) in enumerate(images):
                x, y = position*WIDTH, 84 + row*(HEIGHT+26)
                draw.text((x+5, y), f'{label} | f{index}', font=font, fill=(235,235,235))
                sheet.paste(Image.fromarray(image), (x, y+24))
            render_records.append(dict(frame_index=index, candidate_object_rendered=index == 0,
                historical_human_view=old_human_view, historical_object_view=old_obj_view,
                candidate_human_view=new_human_view, candidate_object_view=new_obj_view))
    finally:
        capture.release()
    encoded = BytesIO(); sheet.save(encoded, format='JPEG', quality=60, optimize=True)
    raw = encoded.getvalue()
    require(len(raw) <= MAX_JPEG_BYTES, 'Tiny JPEG cap exceeded; no automatic quality rescue')
    sources.verify()
    image_path = output/f'episode-{episode}-qwen4d-initializers.jpg'
    with image_path.open('xb') as stream:
        stream.write(raw)
    image_path.chmod(0o444)
    report = dict(schema='world_reward.qwen4d_initializers_preview.v1', status='pass',
        producer_revision=revision, episode_index=episode, original_frames=total, frame_indices=indices,
        input_track='track_1', ground_truth_used=False, manual_annotation=False,
        fitting=False, model_execution=False, gpu_used=False, metric_evaluation=False,
        full_4d_candidate=False, candidate_accuracy_validated=False,
        candidate_human_identity_clip_constant=False, candidate_object_trajectory_available=False,
        candidate_object_rendered_frames=[0], geometry_unchanged=True,
        original_mask_bytes_independently_pinned=mask_report.get('source_mask_bytes_independently_pinned'),
        camera_realignment=False, exact_shared_camera_verified=True,
        second_object_scale_applied=False, historical_reference_not_controlled_ablation=True,
        viewport_hw=[HEIGHT, WIDTH], image_hw=[sheet.height, sheet.width], jpeg_quality=60,
        image=identity(image_path), render_records=render_records,
        sources={str(p): pin for p, pin in sources.files.items()},
        sources_rehashed_after=True, elapsed_seconds=time.perf_counter()-started)
    with (output/'report.json').open('x') as stream:
        stream.write(json.dumps(report, sort_keys=True, allow_nan=False)+'\n')
    (output/'report.json').chmod(0o444)
    return report


def main():
    require(platform.system() == 'Linux'
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
        'Saved data/rendering stays in an offline Azure Linux CPU container')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--episode', type=int, choices=(8, 9, 26), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(Path(os.environ.get('WR_ROOT', str(ROOT))) == ROOT, 'Original Azure root required')
    report = run(ROOT, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
        os.environ['WR_OUTPUT_PREFIX'], args.episode, args.output)
    print(json.dumps({k: report[k] for k in ('schema', 'status', 'episode_index', 'image', 'elapsed_seconds')}))


if __name__ == '__main__':
    main()
