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
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import time

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


def main():
    require(platform.system() == 'Linux' and os.geteuid() == 1000
        and {p.name for p in Path('/sys/class/net').iterdir()} == {'lo'},
        'Owned offline Azure renderer container required')
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--episode', type=int, choices=range(30), required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root = canonical(Path(os.environ['WR_ROOT']))
    require(root == ROOT, 'Original asset/data root required')
    run(root, Path(os.environ['WR_CODE']), os.environ['WR_CODE_REVISION'],
        os.environ['WR_OUTPUT_PREFIX'], Path(os.environ['WR_PIN_ROOT']), args.episode, args.output)


if __name__ == '__main__':
    main()
