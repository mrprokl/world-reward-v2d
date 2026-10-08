"""Azure-only, saved full-timeline 4D comparison; never a fitting stage.

Left: original RGB. Right: the exact exported human and rigid object, with one
fixed inferred camera and a display-only virtual floor. There is no per-frame
alignment, centring, scale adjustment, prediction repair or frame selection.
The horizontal floor is an explicitly uncertain visualization assumption, not
source calibration, ground truth, or evidence that absolute scale is correct.
"""
from __future__ import annotations

import argparse
from io import BytesIO
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import re
import shutil
import subprocess
import sys
import time

# The managed host has only stdlib; rendering dependencies load in CPU child.
if sys.argv[1:] == ['--initialization-host']:
    from mediapipe_cpu_runtime_verify import canonical, identity, require
else:
    import numpy as np
    from camera_render import _intrinsics, _opencv_camera
    from qwen4d_preview import Sources, canonical, identity, require
    from world_reward.mesh_geometry import normalize_degenerate_faces

ROOT = Path('/srv/scenesmith/world-reward')
DATASET = '5f68335f3acc802033d1e80728c1633197521de8'
WIDTH, HEIGHT, BAR, FPS = 320, 240, 20, 30
MAX_VIDEO_BYTES, MAX_POSTER_BYTES = 2_000_000, 100_000
GRID_M = .5
EXPORT_FILES = {'report.json', 'target.npy', 'native_parameters.npz',
    'trajectory.npz', 'object_aligned.glb'}
HUMAN_RGB, OBJECT_RGB = (48, 205, 220), (245, 148, 56)
INITIALIZATION_ENTRY = 'run_full4d_video_initialization'
INITIALIZATION_HELPERS = ('infra/full4d_video.py', 'infra/run_full4d_video_initialization.sh',
    'infra/qwen4d_preview.py', 'infra/reconstruction_preview.py', 'infra/camera_render.py',
    'infra/mediapipe_cpu_runtime_verify.py', 'src/world_reward/mesh_geometry.py',
    'src/world_reward/seeded_tracking.py')
INITIALIZATION_STAGES = {'body_smoke': ('sam3d_body_three_frame_smoke', 'infra/body_smoke.py'),
    'scale_smoke': ('predicted_human_anchored_moge2_pointmaps', 'infra/scale_smoke.py'),
    'object_grounded': ('sam3d_objects_grounded_fixed_frame', 'infra/object_smoke.py')}


def experiment_paths(root, revision, prefix, pin_root, episode):
    require(type(revision) is str and re.fullmatch('[0-9a-f]{40}', revision),
        'Exact experiment revision required')
    require(type(episode) is int and 0 <= episode < 30, 'Original Track1 episode required')
    experiment = canonical(root) / 'experiments' / f'full4d-v1-{revision}'
    require(prefix == f'experiments/full4d-v1-{revision}/outputs'
        and canonical(pin_root) == experiment / 'pins',
        'New isolated experiment and matching runtime pins required; never baseline outputs')
    return (canonical(experiment / 'outputs' / f'episode_{episode:06d}' / 'cari_shared_export_v1'),
        canonical(Path(pin_root) / f'cari_clip_{episode:06d}_shared_export_pins.json'), experiment)


def checked_geometry(target, human_faces, trajectory, native_indices, total):
    """Validate safe saved arrays without modifying, retiming or rescaling them."""
    require(type(total) is int and total >= 3, 'Complete original timeline required')
    require(target.ndim == 3 and target.shape[0] == total and target.shape[2] == 3
        and target.shape[1] >= 3 and target.dtype.kind in 'f'
        and np.isfinite(target).all() and (target[..., 2] > 1e-4).all(),
        'Finite complete human camera geometry in front of camera required')
    required = {'object_vertices', 'object_faces', 'object_rotation', 'object_translation',
        'object_scale', 'camera_K', 'frame_index'}
    require(set(trajectory) == required, 'Exact safe trajectory fields required')
    require(np.array_equal(native_indices, np.arange(total))
        and np.array_equal(trajectory['frame_index'], np.arange(total)),
        'All original frame indices, including the tail and occlusions, required')
    require(trajectory['object_scale'].shape == () and float(trajectory['object_scale']) == 1.,
        'Object canonical metric scale already applied; no second scaling')
    vertices, rotations, translations = (trajectory[k] for k in
        ('object_vertices', 'object_rotation', 'object_translation'))
    require(vertices.ndim == 2 and vertices.shape[1] == 3 and len(vertices) >= 3
        and vertices.dtype.kind in 'f' and np.isfinite(vertices).all()
        and rotations.shape == (total, 3, 3) and translations.shape == (total, 3)
        and np.isfinite(rotations).all() and np.isfinite(translations).all(),
        'One nonempty object geometry and every rigid pose required')
    require(np.allclose(rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-4, rtol=0)
        and np.allclose(np.linalg.det(rotations), 1., atol=1e-4, rtol=0),
        'Original proper rigid rotations required; no pose repair')
    _intrinsics(trajectory['camera_K'])
    normalize_degenerate_faces(target[0], human_faces)
    normalize_degenerate_faces(vertices, trajectory['object_faces'])


def display_intrinsics(camera, width, height):
    require(type(width) is int and type(height) is int and width > 0 and height > 0
        and width * HEIGHT == height * WIDTH, 'Original4:3 aspect ratio required')
    result = _intrinsics(camera).copy()
    result[0] *= WIDTH / width
    result[1] *= HEIGHT / height
    return result


def fixed_floor_height(target):
    """Median of each frame's lowest body vertex, in camera Y-down metres.

    Computed ONCE over the complete saved clip. It only draws the background;
    neither the human nor object is moved to this plane. Camera pitch is not
    recovered here, so the horizontal-plane assumption is reported explicitly.
    """
    require(target.ndim == 3 and target.shape[2] == 3 and len(target) >= 3
        and np.isfinite(target).all(), 'Full finite human timeline required for virtual floor')
    return float(np.median(np.max(target[..., 1], axis=1)))


def floor_background(camera, floor_y_m):
    """Analytic fixed-camera ray/plane view with a world-origin0.5m grid."""
    camera = _intrinsics(camera)
    require(type(floor_y_m) in (int, float) and np.isfinite(floor_y_m),
        'Finite fixed virtual floor required')
    yy, xx = np.mgrid[:HEIGHT, :WIDTH]
    rays_x = (xx + .5 - camera[0, 2]) / camera[0, 0]
    rays_y = (yy + .5 - camera[1, 2]) / camera[1, 1]
    z = np.full((HEIGHT, WIDTH), np.nan)
    np.divide(floor_y_m, rays_y, out=z, where=np.abs(rays_y) > 1e-9)
    visible = np.isfinite(z) & (z > 1e-4)
    x = np.where(visible, rays_x * z, 0.)
    safe_z = np.where(visible, z, 0.)
    # Avoid integer overflow near the horizon; fade the distant plane instead.
    near = visible & (safe_z < 1000.)
    checker = ((np.floor(x / GRID_M) + np.floor(safe_z / GRID_M)) % 2) == 0
    background = np.full((HEIGHT, WIDTH, 3), (29, 33, 40), np.float64)
    checker_color = np.where(checker[..., None], np.array([82, 88, 96]), np.array([72, 78, 86]))
    fade = np.exp(-np.maximum(safe_z, 0) / 35.)[..., None]
    plane = checker_color * fade + np.array([39, 44, 52]) * (1 - fade)
    background[visible] = plane[visible]
    # Physical fixed width, softened by distance; no grid tied to actor motion.
    x_edge = np.minimum(np.mod(x, GRID_M), GRID_M - np.mod(x, GRID_M))
    z_edge = np.minimum(np.mod(safe_z, GRID_M), GRID_M - np.mod(safe_z, GRID_M))
    edge = near & ((x_edge < .01) | (z_edge < .01))
    background[edge] *= .78
    return np.rint(background).astype(np.uint8)


def object_at_frame(trajectory, index):
    result = trajectory['object_vertices'] @ trajectory['object_rotation'][index].T \
        + trajectory['object_translation'][index]
    require(np.isfinite(result).all() and (result[:, 2] > 1e-4).all(),
        'Entire original object must remain in front of camera; no clipping')
    return result


class SavedSceneRenderer:
    """Full original surfaces, one shared z-buffer, fixed light and camera."""
    def __init__(self, camera, human_faces, object_faces, floor_y_m):
        require(platform.system() == 'Linux', 'Rendering stays on Azure Linux')
        import torch
        import pytorch3d
        from pytorch3d.renderer import MeshRasterizer, RasterizationSettings
        require(torch.cuda.is_available() and pytorch3d.__version__ == '0.7.9',
            'Audited Azure CUDA PyTorch3D0.7.9 required')
        self.torch = torch
        self.MeshRasterizer, self.RasterizationSettings = MeshRasterizer, RasterizationSettings
        self.camera = _opencv_camera(torch, camera, WIDTH, HEIGHT)
        self.human_faces, self.object_faces = human_faces, object_faces
        self.background = torch.tensor(floor_background(camera, floor_y_m), device='cuda', dtype=torch.float32)
        self.light = torch.tensor([-.35, -.55, -1.], device='cuda', dtype=torch.float32)
        self.light /= torch.linalg.norm(self.light)
        self.frames_rendered = 0
        self.max_human_collapsed, self.max_object_collapsed = 0, 0

    def frame(self, human, obj):
        from pytorch3d.structures import Meshes
        torch = self.torch
        human_active, hr = normalize_degenerate_faces(human, self.human_faces)
        object_active, ore = normalize_degenerate_faces(obj, self.object_faces)
        require(len(human_active) and len(object_active), 'Both original surfaces must be visible mesh inputs')
        self.max_human_collapsed = max(self.max_human_collapsed, hr['excluded_faces'])
        self.max_object_collapsed = max(self.max_object_collapsed, ore['excluded_faces'])
        vertices = np.concatenate((human, obj)).astype(np.float32, copy=False)
        require(np.isfinite(vertices).all() and (vertices[:, 2] > 1e-4).all(),
            'Whole unchanged geometry required; near-plane crossings fail')
        hf = self.human_faces[human_active]
        of = self.object_faces[object_active] + len(human)
        faces = np.concatenate((hf, of)).astype(np.int64, copy=False)
        with torch.inference_mode():
            v = torch.as_tensor(vertices, device='cuda')
            f = torch.as_tensor(faces, device='cuda')
            settings = self.RasterizationSettings(image_size=(HEIGHT, WIDTH), blur_radius=0.,
                faces_per_pixel=1, perspective_correct=True, clip_barycentric_coords=False,
                cull_backfaces=False, cull_to_frustum=False, z_clip_value=None,
                max_faces_per_bin=len(faces))
            fragments = self.MeshRasterizer(cameras=self.camera, raster_settings=settings)(
                Meshes(verts=[v], faces=[f]))
            face_index = fragments.pix_to_face[0, ..., 0]
            triangles = v[f]
            normals = torch.linalg.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
            normals = normals / torch.linalg.norm(normals, dim=-1, keepdim=True).clamp_min(1e-20)
            # Two-sided shading preserves unknown mesh winding; no topology repair.
            illumination = .5 + .5 * torch.abs(normals @ self.light)
            colors = torch.empty((len(faces), 3), device='cuda', dtype=torch.float32)
            colors[:len(hf)] = torch.tensor(HUMAN_RGB, device='cuda')
            colors[len(hf):] = torch.tensor(OBJECT_RGB, device='cuda')
            colors *= illumination[:, None]
            output = self.background.clone()
            visible = face_index >= 0
            output[visible] = colors[face_index[visible]]
            require(torch.isfinite(output).all(), 'Finite raster output required')
            rgb = output.round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        self.frames_rendered += 1
        return rgb


def ffmpeg_command(executable, output, total):
    require(type(total) is int and total >= 3, 'Original frame count required')
    # Determined ONLY by clip duration and transfer cap, never observed quality.
    ceiling = min(650_000, int((MAX_VIDEO_BYTES - 150_000) * 8 * FPS / total * .85))
    require(ceiling >= 50_000, 'Clip too long for the frozen full-timeline lightweight preset')
    return [executable, '-hide_banner', '-loglevel', 'error', '-nostdin', '-n',
        '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{WIDTH*2}x{HEIGHT+BAR*2}',
        '-r', str(FPS), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast',
        '-crf', '30', '-maxrate', str(ceiling), '-bufsize', str(ceiling),
        '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)]


def verify_encoded(probe, total):
    streams = probe.get('streams')
    require(type(streams) is list and len(streams) == 1, 'Exactly one video stream required')
    stream = streams[0]
    require(stream.get('codec_type') == 'video' and stream.get('codec_name') == 'h264'
        and stream.get('width') == WIDTH*2 and stream.get('height') == HEIGHT+BAR*2
        and stream.get('r_frame_rate') == f'{FPS}/1' and stream.get('avg_frame_rate') == f'{FPS}/1'
        and str(stream.get('nb_read_frames')) == str(total),
        'Encoded full original frame count,30fps and viewport required; no dropped frames')


def run(root, code, revision, prefix, pin_root, episode, output):
    started = time.perf_counter()
    root, code, output = map(canonical, (root, code, output))
    base, pins_path, experiment = experiment_paths(root, revision, prefix, pin_root, episode)
    require(output.is_relative_to(experiment) and output.is_dir() and not any(output.iterdir())
        and output.stat().st_uid == os.getuid(), 'Fresh owned video output inside new experiment required')
    sources = Sources()
    for name in ('infra/full4d_video.py', 'infra/camera_render.py', 'infra/qwen4d_preview.py',
            'src/world_reward/mesh_geometry.py'):
        sources.bind(code/name, maximum=1 << 20)
    require(Path(__file__).resolve() == code/'infra/full4d_video.py', 'Actual immutable viewer source required')
    pins = sources.json(pins_path)
    require(not pins_path.stat().st_mode & 0o222, 'Read-only producer export pins required')
    spec = pins.get('clip_spec', {})
    require(pins.get('schema') == 'world-reward-cari-shared-export-pins-v1'
        and set(spec) == {'episode_index', 'total_frames', 'camera_name', 'height', 'width'}
        and spec.get('episode_index') == episode and type(spec.get('episode_index')) is int
        and (spec.get('camera_name'), spec.get('height'), spec.get('width')) ==
            ('front_stereo_camera_left', 1152, 1536)
        and set(pins.get('export_files', {})) == EXPORT_FILES, 'Exact original full-video export pins required')
    for name, pin in pins['export_files'].items():
        sources.bind(base/name, pin)
        require(not (base/name).stat().st_mode & 0o222, 'Frozen read-only exports required')
    producer = sources.json(base/'report.json')
    expected = dict(stage='world_reward_native_cari_shared_full_video_direct_export', status='pass',
        phase='complete', episode_index=episode, clip_spec=spec, producer_revision=revision,
        input_track='track_1', ground_truth_used=False, ground_truth_read=False, private_truth_read=False,
        hand_labeled_test=False, oracle_modes=[], unchanged_refined_predictions_verified=True,
        full_original_native_export_verified=True)
    require(all(type(producer.get(k)) is type(v) and producer[k] == v for k, v in expected.items()),
        'Complete video-only new full4D producer provenance required')
    export_pin = pins.get('export', {})
    require(all(export_pin.get(k) == sources.files[base/'report.json'][k] for k in ('bytes', 'sha256'))
        and export_pin.get('producer_revision') == revision
        and export_pin.get('script_sha256') == producer.get('script_sha256'), 'Independent full4D producer binding differs')
    manifest = sources.json(root/'results/input-manifest.json')
    require((manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) ==
        ('track_1', 'nvidia/video_to_data_challenge', DATASET), 'Original Track1-only manifest required')
    relative = f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{episode:06d}.mp4'
    records = [row for row in manifest['files'] if row.get('path') == relative]
    require(len(records) == 1, 'Unique original RGB source required')
    video = sources.bind(root/'data'/relative, {k: records[0][k] for k in ('bytes', 'sha256')})
    target = np.load(base/'target.npy', mmap_mode='r', allow_pickle=False)
    with np.load(base/'trajectory.npz', allow_pickle=False) as archive:
        trajectory = {k: archive[k] for k in ('object_vertices', 'object_faces', 'object_rotation',
            'object_translation', 'object_scale', 'camera_K', 'frame_index')}
    with np.load(base/'native_parameters.npz', allow_pickle=False) as archive:
        human_faces, native_indices = archive['human_faces'], archive['frame_index']
    total = spec['total_frames']
    require(target.shape == (total, 18439, 3) and not target.flags.writeable,
        'Original frozen all18439 human vertices required')
    checked_geometry(target, human_faces, trajectory, native_indices, total)
    camera = display_intrinsics(trajectory['camera_K'], spec['width'], spec['height'])
    floor_y = fixed_floor_height(target)
    renderer = SavedSceneRenderer(camera, human_faces, trajectory['object_faces'], floor_y)
    import cv2
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg, ffprobe = shutil.which('ffmpeg'), shutil.which('ffprobe')
    require(ffmpeg and ffprobe, 'Existing native ffmpeg/ffprobe required; no download')
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 11)
    template = Image.new('RGB', (WIDTH*2, HEIGHT+BAR*2), (24, 27, 33))
    draw = ImageDraw.Draw(template)
    draw.text((6, 3), 'World Reward | Original RGB', font=font, fill=(238, 238, 238))
    draw.text((WIDTH+6, 3), 'Reconstructed4D | SAME inferred scale', font=font, fill=(238, 238, 238))
    draw.text((WIDTH+6, HEIGHT+BAR+3), 'Virtual floor (assumed) | grid0.5m', font=font, fill=(238, 238, 238))
    capture = cv2.VideoCapture(str(video))
    require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
        and abs(capture.get(cv2.CAP_PROP_FPS)-FPS) < .003,
        'Original full clip decoder count and30fps required')
    destination = output/f'episode_{episode:06d}.mp4'
    command = ffmpeg_command(ffmpeg, destination, total)
    count, poster = 0, None
    with (output/'encode.log').open('xb') as log:
        proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
        try:
            for index in range(total):
                ok, bgr = capture.read()
                require(ok and bgr.shape == (spec['height'], spec['width'], 3)
                    and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == index+1,
                    'Sequential exact original frame decode failed; no frame skipping')
                rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
                scene = renderer.frame(target[index], object_at_frame(trajectory, index))
                frame = template.copy()
                frame.paste(Image.fromarray(rgb), (0, BAR))
                frame.paste(Image.fromarray(scene), (WIDTH, BAR))
                ImageDraw.Draw(frame).text((6, HEIGHT+BAR+3),
                    f'Episode{episode:02d} | frame{index:04d}/{total-1:04d} | {index/FPS:.2f}s',
                    font=font, fill=(238, 238, 238))
                proc.stdin.write(np.asarray(frame).tobytes())
                count += 1
                if index == 0:
                    encoded = BytesIO(); frame.save(encoded, format='JPEG', quality=72, optimize=True)
                    poster = encoded.getvalue()
            ok, _ = capture.read()
            require(not ok, 'Extra source frames beyond original metadata')
            proc.stdin.close()
            require(proc.wait(timeout=60) == 0, 'Native H264 encode failed')
        finally:
            capture.release()
            if proc.poll() is None:
                proc.kill(); proc.wait(timeout=15)
    require(count == total == renderer.frames_rendered and destination.stat().st_size <= MAX_VIDEO_BYTES,
        'Complete full-T video within frozen byte cap required')
    probe = json.loads(subprocess.check_output([ffprobe, '-v', 'error', '-count_frames',
        '-show_streams', '-show_format', '-of', 'json', str(destination)], timeout=60))
    verify_encoded(probe, total)
    require(poster and len(poster) <= MAX_POSTER_BYTES, 'Tiny first-frame poster byte cap exceeded')
    poster_path = output/f'episode_{episode:06d}.jpg'; poster_path.write_bytes(poster)
    destination.chmod(0o444); poster_path.chmod(0o444)
    sources.verify()
    # Successful empty encoder logs are disposable; failures retain diagnostics.
    (output/'encode.log').unlink()
    result = dict(schema='world_reward.full4d_video.v1', status='pass', producer_revision=revision,
        episode_index=episode, original_frames=total, frames_encoded=count, fps=FPS,
        original_frame_indices=list(range(total)), video=identity(destination), poster=identity(poster_path),
        viewport_hw=[HEIGHT, WIDTH], video_hw=[HEIGHT+BAR*2, WIDTH*2], input_track='track_1',
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[], model_execution=False,
        optimizer_execution=False, metric_evaluation=False, quality_verified=False,
        original_geometry_unchanged=True, object_scale_applied_again=False,
        per_frame_alignment=False, per_frame_camera=False, per_frame_centring=False,
        display_camera_K=camera.tolist(), camera_frame='OpenCV_x_right_y_down_z_forward',
        scale='unchanged_upstream_inferred_metres_not_ground_truth',
        virtual_floor=dict(camera_y_m=floor_y, grid_spacing_m=GRID_M,
            method='whole_clip_median_of_per_frame_maximum_human_camera_Y', fixed_over_clip=True,
            horizontal_camera_assumption=True, physical_ground_verified=False,
            occludes_predictions=False, reconstruction_moved_to_floor=False),
        raster=dict(device='cuda', renderer='pytorch3d0.7.9_shared_scene_zbuffer',
            human_max_collapsed_render_only_faces=renderer.max_human_collapsed,
            object_max_collapsed_render_only_faces=renderer.max_object_collapsed,
            meaningful_surface_or_components_removed=False),
        encoder=dict(preset='fast', crf=30, command=command, output_byte_cap=MAX_VIDEO_BYTES),
        sources={str(p): row for p, row in sources.files.items()}, source_rehashed_after=True,
        elapsed_seconds=time.perf_counter()-started)
    receipt = output/'report.json'
    receipt.write_text(json.dumps(result, sort_keys=True, allow_nan=False)+'\n'); receipt.chmod(0o444)
    print('FULL4D_VIDEO_PASS', episode, count, result['video']['bytes'], result['video']['sha256'])
    return result


def initialization_contract(body, scale, obj, masks, episode, total, video_sha):
    """Completed saved initializers only; PASS here never means 4D/quality PASS."""
    from qwen4d_preview import report_contract
    require(type(episode) is int and episode in random.Random(20261008).sample(range(30), 4)
        and type(total) is int and total >= 30, 'Original frozen random cohort required')
    for report, stage in ((body, INITIALIZATION_STAGES['body_smoke'][0]),
            (scale, INITIALIZATION_STAGES['scale_smoke'][0]),
            (obj, INITIALIZATION_STAGES['object_grounded'][0]), (masks, 'automatic_masks')):
        report_contract(report, stage, episode, video_sha)
    indices = [0, total//2, total-1]
    require(body.get('frame_indices') == indices and body.get('total_video_frames') == total
        and body.get('geometry_units') == 'metres' and body.get('mhr_geometry_forward_verified') is True
        and body.get('frame_independent_initializer_only') is True
        and body.get('human_identity_clip_constant') is False
        and body.get('input_dataset_revision') == DATASET and scale.get('frame_indices') == indices
        and scale.get('coordinate_frame') == 'OpenCV_x_right_y_down_z_forward'
        and scale.get('pointmap_scale_application') == 'one_clip_scalar_to_MoGe2_XYZ_already_applied'
        and obj.get('frame_index') == 0
        and obj.get('scale_source') == 'already_human_anchored_MoGe2_no_second_scalar'
        and masks.get('frames') == total and masks.get('full_original_grid') is True
        and masks.get('grounding_model') == 'gemini-3.5-flash'
        and masks.get('mask_model') == 'facebook/sam3.1'
        and masks.get('source_revision') == '2345a4ad109ac29c569da749c91d84f10dc08c40'
        and masks.get('model_revision') == 'daa63191845a41281374e725f4c9e51c7a824460'
        and masks.get('empty_mask_interpolation') is False and masks.get('manual_labels') is False,
        'Exact native frame0/new automatic frontend and inferred gauge required')


def initialization_scene(human, human_faces, obj, object_faces, camera):
    """Stateless CPU preview: unchanged surfaces, joint depth order, fixed floor."""
    from qwen4d_preview import render_depth
    human_depth, human_receipt = render_depth(human, human_faces, camera)
    object_depth, object_receipt = render_depth(obj, object_faces, camera)
    floor_y = float(np.max(human[:, 1]))
    scene = floor_background(camera, floor_y)
    h = np.isfinite(human_depth) & (human_depth <= object_depth)
    o = np.isfinite(object_depth) & (object_depth < human_depth)
    scene[h] = HUMAN_RGB; scene[o] = OBJECT_RGB
    return scene, dict(device='cpu', renderer='perspective_correct_cpu_joint_scene_zbuffer',
        human=human_receipt, object=object_receipt,
        virtual_floor=dict(camera_y_m=floor_y, grid_spacing_m=GRID_M,
            method='frame0_maximum_human_camera_Y_display_only', horizontal_camera_assumption=True,
            physical_ground_verified=False, occludes_predictions=False, reconstruction_moved_to_floor=False))


def initialization_sheet(rgb, masks, scene, episode, total):
    from PIL import Image, ImageDraw, ImageFont
    from qwen4d_preview import mask_overlay
    require(rgb.shape == scene.shape == (HEIGHT, WIDTH, 3)
        and len(masks) == 2, 'Same three-panel viewport required')
    sheet = Image.new('RGB', (WIDTH*3, HEIGHT+90), (24, 27, 33))
    draw = ImageDraw.Draw(sheet); font = ImageFont.load_default()
    draw.text((6, 5), f'World Reward | episode {episode:02d} | INITIAL 3D - NOT full 4D / temporal validation / score',
        font=font, fill='white')
    draw.text((6, 23), f'Original frame 0/{total-1} | cyan: human | orange: object | unchanged inferred scale, not calibrated',
        font=font, fill='white')
    for column, (image, label) in enumerate(((rgb, 'Original RGB'),
            (mask_overlay(rgb, *masks), 'NEW Gemini + SAM3.1 masks'),
            (scene, 'NEW SAM3D Body + Objects INITIAL 3D'))):
        draw.text((column*WIDTH+6, 43), label, font=font, fill='white')
        sheet.paste(Image.fromarray(image), (column*WIDTH, 62))
    draw.text((6, HEIGHT+67), 'Virtual floor ASSUMED, grid0.5m | frame0 only: NO motion / NO fitting / NO geometry repair',
        font=font, fill='white')
    encoded = BytesIO(); sheet.save(encoded, format='JPEG', quality=72, optimize=True)
    require(len(encoded.getvalue()) <= MAX_POSTER_BYTES, 'Fixed initialization JPEG byte cap exceeded')
    return encoded.getvalue()


def run_initialization(root, code, revision, producer_revision, output, episode=None):
    """Observe completed stage files while the numerical experiment continues."""
    from mediapipe_cpu_runtime_verify import source
    from qwen4d_preview import (strict_json, object_camera_vertices, shared_camera, render_depth)
    from reconstruction_preview import raster_depth
    import cv2
    from PIL import Image
    import trimesh
    started = time.perf_counter()
    root, code, output = map(canonical, (root, code, output))
    require(root == ROOT and re.fullmatch('[0-9a-f]{40}', str(producer_revision))
        and revision != producer_revision and Path(__file__).resolve() == code/'infra/full4d_video.py'
        and output == root/'results'/('full4d-initialization-'+revision)/'group'
        and output.is_dir() and output.stat().st_uid == os.getuid() and not any(output.iterdir()),
        'Fresh separate owned publication output and exact original numerical producer required')
    require(Path(raster_depth.__code__.co_filename).resolve() == code/'infra/reconstruction_preview.py'
        and Path(render_depth.__code__.co_filename).resolve() == code/'infra/qwen4d_preview.py'
        and Path(normalize_degenerate_faces.__code__.co_filename).resolve() == code/'src/world_reward/mesh_geometry.py',
        'Actual CPU renderer helper origins differ')
    publisher = source(root, code, revision, INITIALIZATION_ENTRY, INITIALIZATION_HELPERS)
    numerical_code = root/'jobs'/producer_revision/'run_gemini_full4d'/'code'
    numerical_helpers = ('infra/gemini_full4d.py', 'configs/gemini_full4d_v1.json',
        *(row[1] for row in INITIALIZATION_STAGES.values()))
    numerical = source(root, numerical_code, producer_revision, 'run_gemini_full4d', numerical_helpers)
    sources = Sources()
    for name in INITIALIZATION_HELPERS: sources.bind(code/name, maximum=2 << 20)
    cfg = sources.json(numerical_code/'configs/gemini_full4d_v1.json')
    cohort = random.Random(20261008).sample(range(30), 4)
    require(cfg.get('schema') == 'world_reward.gemini_full4d.v1' and cfg.get('episodes') == cohort
        and cfg.get('resample_failed_clips') is False and cfg.get('ground_truth_used') is False
        and cfg.get('manual_labels') is False and cfg.get('challenge_performance_verified') is False,
        'Frozen numerical population, no replacement/manual/GT inputs required')
    require(episode is None or type(episode) is int and episode in cohort, 'Only original frozen cohort allowed')
    frontend_revision = cfg['frontend_producer_revision']
    frontend_code = root/'jobs'/frontend_revision/'run_gemini_sam31_track'/'code'
    original_frontend = root/'results'/('gemini-sam31-'+frontend_revision)
    host = sources.json(original_frontend/'report.json')
    frontend = source(root, frontend_code, frontend_revision, 'run_gemini_sam31_track',
        tuple(host.get('source_binding', {}).get('helpers', {})))
    aggregate = sources.json(original_frontend/'native-report.json', host.get('native_report'))
    require(host.get('source_binding') == frontend and host.get('producer_revision') == frontend_revision
        and host.get('status') == aggregate.get('status') == 'complete_diagnostic_not_quality_pass'
        and aggregate.get('producer_revision') == frontend_revision
        and aggregate.get('ground_truth_used') is False and aggregate.get('manual_labels') is False,
        'Exact actual saved SAM3.1 host/native/source closure required')
    frontend_cfg = sources.json(frontend_code/'configs/gemini_sam31_v1.json')
    gemini = sources.json(root/'results'/('gemini-initial-'+frontend_cfg['gemini_producer'])/'native-report.json',
        frontend_cfg['gemini_native_report'])
    require(gemini.get('status') == 'complete_diagnostic_not_quality_pass'
        and gemini.get('ground_truth_used') is False and gemini.get('manual_labels') is False,
        'Exact saved automatic Gemini producer required')
    manifest = sources.json(root/'results/input-manifest.json')
    require((manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) ==
        ('track_1', 'nvidia/video_to_data_challenge', DATASET), 'Original Track1-only manifest required')
    def original(relative):
        records = [r for r in manifest['files'] if r.get('path') == relative]
        require(len(records) == 1, 'Unique original manifest record required')
        return sources.bind(root/'data'/relative, {k: records[0][k] for k in ('bytes', 'sha256')})
    metadata = original('track_1/meta/episodes.jsonl')
    records = [strict_json(line) for line in metadata.read_bytes().splitlines() if line.strip()]
    availability, results = [], []
    for ep in cohort:
        base = root/'experiments'/('full4d-v1-'+producer_revision)/'outputs'/f'episode_{ep:06d}'
        paths = {stage: base/stage/'report.json' for stage in INITIALIZATION_STAGES}
        missing = [stage for stage, path in paths.items() if not path.is_file()]
        if missing:
            availability.append(dict(episode=ep, status='not_ready', missing_stages=missing)); continue
        stage_reports = {stage: sources.json(path) for stage, path in paths.items()}
        failed = [stage for stage, r in stage_reports.items() if r.get('status') != 'pass']
        if failed:
            availability.append(dict(episode=ep, status='not_ready', nonpassing_stages=failed)); continue
        availability.append(dict(episode=ep, status='three_saved_initializers_pass'))
        if episode is not None and ep != episode: continue
        ep_records = [r for r in records if type(r.get('episode_index')) is int and r['episode_index'] == ep]
        require(len(ep_records) == 1, 'Unique original episode count required')
        total = ep_records[0]['length']; indices = [0, total//2, total-1]
        video = original(f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{ep:06d}.mp4')
        video_sha = sources.files[video]['sha256']
        mask_dir = base/'automatic_masks'; masks_report = sources.json(mask_dir/'report.json')
        body, scale, obj = (stage_reports[k] for k in ('body_smoke', 'scale_smoke', 'object_grounded'))
        initialization_contract(body, scale, obj, masks_report, ep, total, video_sha)
        for stage, report in stage_reports.items():
            script = sources.bind(numerical_code/INITIALIZATION_STAGES[stage][1], maximum=2 << 20)
            require(report.get('script_sha256') == sources.files[script]['sha256'], 'Original numerical script SHA differs')
        frontend_rows = [r for r in aggregate.get('episodes', []) if r.get('episode_index') == ep]
        require(len(frontend_rows) == 1 and frontend_rows[0].get('status') == 'pass'
            and frontend_rows[0].get('frames') == total
            and frontend_rows[0].get('report') == sources.files[mask_dir/'report.json']
            and masks_report.get('producer_revision') == frontend_revision
            and masks_report.get('script_sha256') == identity(frontend_code/'infra/gemini_sam31_track.py')['sha256'],
            'Copied masks must retain exact original new native producer')
        source_mask_dir = original_frontend/f'episode_{ep:06d}'/'automatic_masks'
        for name in ('report.json', 'prompts.json', 'mask-inventory.json', 'grounding.json', 'tracking.json'):
            copied = sources.bind(mask_dir/name)
            original_file = sources.bind(source_mask_dir/name, sources.files[copied])
            require(not original_file.stat().st_mode & 0o222 and not copied.stat().st_mode & 0o222,
                'Native masks/metadata must be original read-only byte copies')
        inventory = sources.json(mask_dir/'mask-inventory.json')
        require(set(inventory) == {f'{role}/{i:06d}.png' for role in ('0', '1') for i in range(total)},
            'Complete original two-ID mask inventory required')
        inventory_identity = sources.files[mask_dir/'mask-inventory.json']
        require(masks_report.get('mask_inventory') == dict(files=2*total,
            bytes=sum(r['bytes'] for r in inventory.values()), sha256=inventory_identity['sha256']),
            'Copied full-T PNG inventory must match original native report')
        prompts = sources.json(mask_dir/'prompts.json').get('prompts')
        require(type(prompts) is list and len(prompts) == 2 and {r.get('object_id') for r in prompts} == {0, 1}
            and all(r.get(k) is None for r in prompts for k in ('points', 'point_labels', 'mask_path')),
            'Only automatically derived pair boxes required; no manual prompts')
        require(body.get('mask_report_sha256') == sources.files[mask_dir/'report.json']['sha256']
            and body.get('prompts_sha256') == sources.files[mask_dir/'prompts.json']['sha256']
            and scale.get('body_report_sha256') == sources.files[paths['body_smoke']]['sha256'],
            'Body/scale lineage differs from new masks')
        body_path = sources.bind(base/'body_smoke/predictions.npz')
        require(sources.files[body_path]['sha256'] == body.get('predictions_sha256'), 'Saved body arrays changed')
        with np.load(body_path, allow_pickle=False) as data:
            require(data['vertices_camera_m'].shape == (3, 18439, 3)
                and np.array_equal(data['frame_index'], indices), 'Exact saved three-frame body geometry required')
            human, human_faces, focals = data['vertices_camera_m'][0].copy(), data['faces'].copy(), data['focal_length'].copy()
        object_dir = base/'object_grounded'
        for name, key in (('object.glb', 'object_sha256'), ('transform.json', 'transform_sha256'),
                ('intrinsics.json', 'intrinsics_sha256')):
            path = sources.bind(object_dir/name)
            require(sources.files[path]['sha256'] == obj.get(key), 'Exact saved object artifact required')
        transform = sources.json(object_dir/'transform.json'); intrinsics = sources.json(object_dir/'intrinsics.json')
        require(transform == obj.get('transform'), 'Native object transform changed')
        grounding = obj.get('pointmap_grounding') or {}
        require(grounding.get('alignment_report_sha256') == sources.files[paths['scale_smoke']]['sha256']
            and grounding.get('pointmap_path') == str(base/'scale_smoke/000000.npy')
            and grounding.get('intrinsics_path') == str(base/'scale_smoke/000000_intrinsics.json'),
            'Original same-experiment grounding, never historical target assets, required')
        source_intrinsics = sources.json(Path(grounding['intrinsics_path']))
        require(source_intrinsics == intrinsics
            and sources.files[Path(grounding['intrinsics_path'])]['sha256'] == grounding.get('intrinsics_sha256'),
            'Original shared inferred camera differs; no camera replacement')
        pointmap = sources.bind(Path(grounding['pointmap_path']))
        require(sources.files[pointmap]['sha256'] == grounding.get('pointmap_sha256'), 'Original grounding pointmap changed')
        mesh = trimesh.load(object_dir/'object.glb', force='mesh', process=False)
        require(isinstance(mesh, trimesh.Trimesh) and len(mesh.vertices) >= 3
            and len(mesh.faces) == obj.get('faces'), 'Whole original canonical object mesh required')
        object_vertices = object_camera_vertices(mesh.vertices, transform)
        camera = shared_camera(focals, scale['human_evidence'], intrinsics, indices, 1536, 1152)
        display_camera = display_intrinsics(camera, 1536, 1152)
        capture = cv2.VideoCapture(str(video))
        try:
            require(capture.isOpened() and int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) == total
                and abs(capture.get(cv2.CAP_PROP_FPS)-FPS) < .003, 'Full original RGB decoder count/rate required')
            ok, bgr = capture.read()
            require(ok and bgr.shape == (1152, 1536, 3) and int(capture.get(cv2.CAP_PROP_POS_FRAMES)) == 1,
                'Exact original frame0 decode required')
            full_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        finally: capture.release()
        rgb_sha = hashlib.sha256(full_rgb.tobytes()).hexdigest()
        require(body['frames'][0]['frame_index'] == 0 and scale['human_evidence'][0]['frame_index'] == 0
            and rgb_sha == body['frames'][0]['decoded_rgb_sha256']
            == scale['human_evidence'][0]['decoded_rgb_sha256'] == obj.get('decoded_rgb_sha256'),
            'Body/object/scale must refer to this exact original decoded frame0')
        tracking = sources.json(mask_dir/'tracking.json'); seeds = sources.json(mask_dir/'grounding.json')
        require(tracking.get('original_frame_indices') == list(range(total))
            and tracking['records'][0]['frame_index'] == 0
            and tracking['records'][0]['decoded_rgb_sha256'] == rgb_sha
            and seeds.get('ground_truth_used') is False and seeds.get('hand_labeled_test') is False
            and seeds.get('manual_points') is False
            and seeds.get('saved_producer') == masks_report.get('grounding_saved_producer')
            and all(r['rgb_sha256'] == rgb_sha for r in seeds['records'] if r['frame_index'] == 0),
            'Saved Gemini seeds and native original timeline must match RGB')
        from world_reward.seeded_tracking import seed_rows
        saved_hashes = {i: tracking['records'][i]['decoded_rgb_sha256'] for i in (0, 14, 29)}
        require(seeds.get('records') == seed_rows(gemini['rows'], ep, [0, 14, 29], 1536, 1152, saved_hashes),
            'Native prompt records must be exact saved automatic Gemini predictions')
        mask_arrays = []
        for role in ('0', '1'):
            key = role+'/000000.png'; path = sources.bind(mask_dir/'masks'/key, inventory[key], 16 << 20)
            sources.bind(source_mask_dir/'masks'/key, inventory[key], 16 << 20)
            with Image.open(path) as image: array = np.asarray(image)
            require(array.shape == (1152, 1536) and np.isin(array, [0, 255]).all() and (array > 0).any(),
                'Original-resolution nonempty binary initializer masks required')
            mask_arrays.append(np.asarray(Image.fromarray(array).resize((WIDTH, HEIGHT), Image.Resampling.NEAREST)) > 0)
        require(body['frames'][0]['mask_sha256'] == inventory['0/000000.png']['sha256']
            and obj.get('mask_sha256') == inventory['1/000000.png']['sha256'], 'Initializer masks changed')
        rgb = cv2.resize(full_rgb, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA)
        scene, raster = initialization_scene(human, human_faces, object_vertices, mesh.faces, display_camera)
        raw = initialization_sheet(rgb, mask_arrays, scene, ep, total)
        path = output/f'episode_{ep:06d}.jpg'
        with path.open('xb') as handle: os.fchmod(handle.fileno(), 0o444); handle.write(raw)
        results.append(dict(episode=ep, frame_index=0, original_frames=total, jpeg=identity(path, MAX_POSTER_BYTES),
            decoded_rgb_sha256=rgb_sha, display_camera_K=display_camera.tolist(), raster=raster,
            raw_glb_vertices=len(mesh.vertices), upstream_inspector_vertices=obj.get('vertices'),
            mesh_load_process=False))
    require(results, 'No completed saved initializer in the original frozen cohort; no substitution')
    sources.verify()
    require(source(root, code, revision, INITIALIZATION_ENTRY, INITIALIZATION_HELPERS) == publisher
        and source(root, numerical_code, producer_revision, 'run_gemini_full4d', numerical_helpers) == numerical
        and source(root, frontend_code, frontend_revision, 'run_gemini_sam31_track',
            tuple(host['source_binding']['helpers'])) == frontend, 'Immutable source changed during saved-only QA')
    result = dict(schema='world_reward.full4d_initialization_visual.v1', status='pass_saved_initialization_visual_only',
        publisher_revision=revision, numerical_producer_revision=producer_revision,
        publisher_source=publisher, numerical_source=numerical, frontend_source=frontend,
        frozen_cohort=cohort, cohort_denominator=len(cohort), availability=availability, episodes=results,
        input_track='track_1', ground_truth_used=False, hand_labeled_test=False, oracle_modes=[],
        model_execution=False, optimizer_execution=False, gpu_used=False, full_4D_produced=False,
        temporal_validation=False, metric_evaluation=False, quality_verified=False,
        original_geometry_unchanged=True, object_stored_scale_applied_once=True, camera_replaced=False,
        per_frame_alignment=False, per_frame_centring=False, sources_mount_readonly=True,
        original_stage_permission_immutability_claimed=False,
        scale='upstream_human_anchored_inferred_metres_not_independent_calibration',
        sources={str(p): row for p, row in sources.files.items()}, source_rehashed_after=True,
        elapsed_seconds=time.perf_counter()-started)
    with (output/'report.json').open('x') as handle:
        os.fchmod(handle.fileno(), 0o444); json.dump(result, handle, sort_keys=True, allow_nan=False); handle.write('\n')
    print('INITIAL_3D_VISUAL_ONLY_PASS', len(results), producer_revision, flush=True)
    return result


def initialization_host():
    """Bounded CPU observer, independently owned; never takes the GPU lease."""
    from mediapipe_cpu_runtime_verify import source, strict
    require(platform.system() == 'Linux' and os.geteuid() == 0
        and os.uname().nodename == 'scenesmith-ncc-h100-01', 'Azure host observer only')
    root, code = canonical(Path(os.environ['WR_ROOT'])), canonical(Path(os.environ['WR_CODE']))
    revision, producer = os.environ['WR_CODE_REVISION'], os.environ['WR_PRODUCER_REVISION']
    before = source(root, code, revision, INITIALIZATION_ENTRY, INITIALIZATION_HELPERS)
    require(root == ROOT and re.fullmatch('[0-9a-f]{40}', producer) and revision != producer,
        'Separate exact numerical/publication revisions required')
    numeric = root/'jobs'/producer/'run_gemini_full4d'/'code'
    source(root, numeric, producer, 'run_gemini_full4d', ('configs/gemini_full4d_v1.json',))
    cfg = strict((numeric/'configs/gemini_full4d_v1.json').read_bytes())
    image = 'sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7'
    require(cfg.get('body_image') == image and strict(subprocess.check_output(['docker', 'image', 'inspect', image]))[0]['Id'] == image,
        'Existing audited CPU-capable body image required; no build/download')
    base = root/'results'/('full4d-initialization-'+revision)
    require(not base.exists(), 'Fresh initialization publication namespace required')
    base.mkdir(mode=0o755); output = base/'group'; output.mkdir(mode=0o755); os.chown(output, 1000, 1000)
    name = 'wr-initial3d-'+revision[:12]
    cmd = ['docker', 'run', '--rm', '--name', name, '--label', 'world_reward.initial3d.owner='+revision,
        '--network', 'none', '--read-only', '--user', '1000:1000', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '256', '--memory', '8g', '--cpus', '2',
        '--tmpfs', '/tmp:rw,nosuid,size=256m', '--mount', f'type=bind,src={root},dst={root},readonly',
        '--mount', f'type=bind,src={output},dst={output}', '--entrypoint', '/usr/bin/env', image,
        '-i', 'PATH=/opt/conda/bin:/usr/local/bin:/usr/bin:/bin', 'HOME=/tmp',
        'CUDA_VISIBLE_DEVICES=-1', 'PYTHONDONTWRITEBYTECODE=1', 'OMP_NUM_THREADS=2', 'OPENBLAS_NUM_THREADS=2',
        'MKL_NUM_THREADS=2', f'PYTHONPATH={code}/src:{code}/infra', f'WR_ROOT={root}', f'WR_CODE={code}',
        f'WR_CODE_REVISION={revision}', 'python', '-B', str(code/'infra/full4d_video.py'),
        '--initialization', '--producer-revision', producer, '--output', str(output)]
    try:
        with (base/'native.log').open('xb') as log:
            os.fchmod(log.fileno(), 0o400)
            completed = subprocess.run(cmd, stdout=log, stderr=log, timeout=600)
        require(completed.returncode == 0, 'Saved initialization CPU observer failed; inspect Azure-only log')
    finally:
        found = subprocess.check_output(['docker', 'ps', '-aq', '--filter', 'name=^/'+name+'$']).strip()
        if found:
            owner = subprocess.check_output(['docker', 'inspect', name, '--format', '{{index .Config.Labels "world_reward.initial3d.owner"}}']).decode().strip()
            require(owner == revision, 'Never clean a foreign container')
            subprocess.run(['docker', 'rm', '-f', name], check=True, stdout=subprocess.DEVNULL, timeout=20)
    require(source(root, code, revision, INITIALIZATION_ENTRY, INITIALIZATION_HELPERS) == before,
        'Observer immutable source changed')
    # Success is fully represented by the concise source-bound visual receipt.
    # Failed observers retain their Azure-only native log for diagnosis.
    (base/'native.log').unlink()
    print('INITIAL_3D_CPU_OBSERVER_COMPLETE', revision, flush=True)


def main():
    if sys.argv[1:] == ['--initialization-host']:
        initialization_host(); return
    require(platform.system() == 'Linux' and os.geteuid() == 1000
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
        'Owned offline Azure renderer container required')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--episode', type=int, choices=range(30))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--initialization', action='store_true')
    parser.add_argument('--producer-revision')
    args = parser.parse_args()
    root = canonical(Path(os.environ['WR_ROOT']))
    require(root == ROOT, 'Original asset/data root required')
    if args.initialization:
        run_initialization(root, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
            args.producer_revision, args.output, args.episode); return
    require(args.episode is not None and args.producer_revision is None,
        'Full4D viewer requires an episode and no initialization producer override')
    run(root, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
        os.environ['WR_OUTPUT_PREFIX'], Path(os.environ['WR_PIN_ROOT']), args.episode, args.output)


if __name__ == '__main__':
    main()
