"""Bounded Azure-only visual QA of frozen predictions, never a fitting stage.

Show original RGB, a perspective-correct CPU depth overlay, and an automatic
object-centred detail at the first/middle/last original frame. No labels,
optimisation, translation, camera replacement or prediction writes occur.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import time

import numpy as np

from camera_render import _mesh_inputs, project_camera_points

ROOT = Path('/srv/scenesmith/world-reward')
WIDTH, HEIGHT = 320, 240
MAX_JPEG_BYTES = 240_000
HUMAN_RGB, OBJECT_RGB = (30, 210, 230), (255, 145, 40)


def frame_indices(total):
    if type(total) is not int or total < 3:
        raise ValueError('At least three original frames required')
    return [0, (total - 1) // 2, total - 1]


def identity(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError('Ordinary source file required')
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return dict(bytes=path.stat().st_size, sha256=h.hexdigest())


def raster_depth(vertices, faces, matrix, width, height):
    """All original triangles, pixel centres, harmonic camera-Z and z-buffer.

    Preview only; it does not replace the native CUDA inference rasterizer.
    Degenerate screen projections occupy no pixels; 3D invalidity still fails.
    """
    vertices, faces, matrix = _mesh_inputs(vertices, faces, matrix, width, height, 1e-4)
    uv = project_camera_points(vertices, matrix)
    depth = np.full((height, width), np.inf)
    for face in faces:
        tri = uv[face]
        x0, y0 = np.maximum(np.ceil(tri.min(axis=0) - .5).astype(int), [0, 0])
        x1, y1 = np.minimum(np.floor(tri.max(axis=0) - .5).astype(int), [width-1, height-1])
        if x1 < x0 or y1 < y0:
            continue
        a, b, c = tri
        den = (b[1]-c[1])*(a[0]-c[0]) + (c[0]-b[0])*(a[1]-c[1])
        if abs(den) < 1e-12:
            continue
        yy, xx = np.mgrid[y0:y1+1, x0:x1+1]
        xx = xx + .5; yy = yy + .5
        w0 = ((b[1]-c[1])*(xx-c[0]) + (c[0]-b[0])*(yy-c[1])) / den
        w1 = ((c[1]-a[1])*(xx-c[0]) + (a[0]-c[0])*(yy-c[1])) / den
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        reciprocal = w0/vertices[face[0], 2] + w1/vertices[face[1], 2] + w2/vertices[face[2], 2]
        z = np.full_like(reciprocal, np.inf)
        np.divide(1., reciprocal, out=z, where=inside & (reciprocal > 0))
        patch = depth[y0:y1+1, x0:x1+1]
        np.minimum(patch, z, out=patch)
    return depth


def overlay(rgb, human_depth, object_depth):
    if rgb.shape != (*human_depth.shape, 3) or object_depth.shape != human_depth.shape:
        raise ValueError('Same viewport required')
    result = rgb.astype(float).copy()
    human = np.isfinite(human_depth) & (human_depth <= object_depth)
    obj = np.isfinite(object_depth) & (object_depth < human_depth)
    for mask, color in ((human, HUMAN_RGB), (obj, OBJECT_RGB)):
        result[mask] = .48 * result[mask] + .52 * np.asarray(color)
    return np.rint(result).astype(np.uint8)


def detail_box(projected_object, width, height):
    """Display-only aspect-preserving crop; identical rule for every clip."""
    low, high = projected_object.min(axis=0), projected_object.max(axis=0)
    centre = (low + high) / 2
    # Integer4:3 rectangles preserve the original image aspect, even at edges.
    unit = min(int(np.ceil(max((high[0]-low[0])*2.5/4,
        (high[1]-low[1])*2.5/3, 64/3))), width//4, height//3)
    size = np.array([4*unit, 3*unit])
    lo = np.clip(np.rint(centre-size/2), [0, 0], [width,height]-size).astype(int)
    return tuple(int(x) for x in (*lo, *(lo+size)))


def main():
    if platform.system() != 'Linux' or {p.name for p in Path('/sys/class/net').iterdir()} != {'lo'}:
        raise RuntimeError('Azure offline CPU container required')
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('--episode', type=int, choices=range(30), required=True)
    args = parser.parse_args()
    code = Path(os.environ['WR_CODE']); revision = os.environ['WR_CODE_REVISION']
    output = Path(os.environ['WR_PREVIEW_OUTPUT'])
    if output.resolve() != output or not output.is_dir() or any(output.iterdir()):
        raise ValueError('Exclusive empty preview directory required')
    started = time.perf_counter()
    pin_path = code/f'configs/cari_clip_{args.episode:06d}_shared_export_pins.json'
    pins = json.loads(pin_path.read_text()); spec = pins['clip_spec']
    if pins['schema'] != 'world-reward-cari-shared-export-pins-v1' or spec['episode_index'] != args.episode:
        raise ValueError('Exact shared export pins required')
    base = ROOT/f'outputs/episode_{args.episode:06d}/cari_shared_export_v1'
    files = {base/name: pin for name, pin in pins['export_files'].items()}
    if any(identity(p) != pin for p, pin in files.items()):
        raise ValueError('Frozen export bytes differ')
    report = json.loads((base/'report.json').read_text())
    expected = dict(stage='world_reward_native_cari_shared_full_video_direct_export',
        status='pass', phase='complete', episode_index=args.episode, clip_spec=spec,
        input_track='track_1', ground_truth_used=False, ground_truth_read=False,
        private_truth_read=False, hand_labeled_test=False, oracle_modes=[],
        unchanged_refined_predictions_verified=True, full_original_native_export_verified=True)
    if any(type(report.get(k)) is not type(v) or report[k] != v for k,v in expected.items()):
        raise ValueError('Complete original export required')
    manifest_path = ROOT/'results/input-manifest.json'
    manifest = json.loads(manifest_path.read_text())
    relative = f'track_1/videos/chunk-000/observation.images.exo_camera/episode_{args.episode:06d}.mp4'
    records = [v for v in manifest['files'] if v['path'] == relative]
    if len(records) != 1 or (manifest.get('track'), manifest.get('repo_id'), manifest.get('revision')) != (
            'track_1', 'nvidia/video_to_data_challenge', '5f68335f3acc802033d1e80728c1633197521de8'):
        raise ValueError('Track1-only manifest required')
    video = ROOT/'data'/relative
    video_pin = {k: records[0][k] for k in ('bytes', 'sha256')}
    if identity(video) != video_pin:
        raise ValueError('Original video bytes differ')
    files.update({video: video_pin, pin_path: identity(pin_path), manifest_path: identity(manifest_path)})
    from PIL import Image, ImageDraw, ImageFont
    import cv2
    target = np.load(base/'target.npy', mmap_mode='r', allow_pickle=False)
    with np.load(base/'trajectory.npz', allow_pickle=False) as archive:
        trajectory = {k: archive[k] for k in ('object_vertices', 'object_faces', 'object_rotation',
            'object_translation', 'object_scale', 'camera_K', 'frame_index')}
    with np.load(base/'native_parameters.npz', allow_pickle=False) as archive:
        human_faces = archive['human_faces']; native_indices = archive['frame_index']
    total = spec['total_frames']; indices = frame_indices(total)
    if target.shape != (total, 18439, 3) or not np.array_equal(trajectory['frame_index'], np.arange(total)) \
            or not np.array_equal(native_indices, np.arange(total)) or float(trajectory['object_scale']) != 1.:
        raise ValueError('Full original frame indices and unchanged exported geometry required')
    matrix = trajectory['camera_K'].copy()
    matrix[0] *= WIDTH/spec['width']; matrix[1] *= HEIGHT/spec['height']
    sheet = Image.new('RGB', (WIDTH*3, 72 + (HEIGHT+26)*3), (24, 26, 31))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 15)
    draw.text((12, 8), f'World Reward | episode {args.episode:02d} | frozen baseline candidate', font=font, fill=(235,235,235))
    draw.text((12, 32), 'Cyan: human   Orange: object | NO ground truth / NO quality score', font=font, fill=(235,235,235))
    capture = cv2.VideoCapture(str(video))
    if int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) != total:
        raise ValueError('Decoder original frame count differs')
    boxes = []
    for column, index in enumerate(indices):
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, index):
            raise ValueError('Exact original frame seek failed')
        ok, bgr = capture.read()
        if not ok or bgr.shape != (spec['height'], spec['width'], 3) or int(capture.get(cv2.CAP_PROP_POS_FRAMES)) != index+1:
            raise ValueError('Exact original frame decode failed')
        rgb = cv2.cvtColor(cv2.resize(bgr, (WIDTH, HEIGHT), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2RGB)
        obj = trajectory['object_vertices'] @ trajectory['object_rotation'][index].T + trajectory['object_translation'][index]
        human_depth = raster_depth(target[index], human_faces, matrix, WIDTH, HEIGHT)
        object_depth = raster_depth(obj, trajectory['object_faces'], matrix, WIDTH, HEIGHT)
        rendered = overlay(rgb, human_depth, object_depth)
        box = detail_box(project_camera_points(obj, matrix), WIDTH, HEIGHT); boxes.append(list(box))
        detail = Image.fromarray(rendered).crop(box).resize((WIDTH, HEIGHT), Image.Resampling.LANCZOS)
        for row, (image, name) in enumerate(((Image.fromarray(rgb), 'Original RGB'),
                (Image.fromarray(rendered), 'Predicted mesh overlay'), (detail, 'Automatic object detail'))):
            x, y = column*WIDTH, 72 + row*(HEIGHT+26)
            draw.text((x+6, y), f'{name} | frame {index}/{total-1}', font=font, fill=(235,235,235))
            sheet.paste(image, (x, y+24))
    capture.release()
    from io import BytesIO
    encoded = BytesIO(); sheet.save(encoded, format='JPEG', quality=70, optimize=True)
    if len(encoded.getvalue()) > MAX_JPEG_BYTES:
        raise ValueError('Tiny preview byte cap exceeded; no automatic quality rescue')
    image_path = output/f'episode_{args.episode:06d}.jpg'
    image_path.write_bytes(encoded.getvalue()); image_path.chmod(0o444)
    if any(identity(p) != pin for p, pin in files.items()):
        raise ValueError('Read-only input changed during QA')
    result = dict(schema='world_reward.reconstruction_preview.v1', status='pass', producer_revision=revision,
        episode_index=args.episode, original_frames=total, frame_indices=indices, input_track='track_1',
        ground_truth_used=False, manual_annotation=False, fitting=False, metric_evaluation=False,
        model_execution=False, gpu_used=False, render='cpu_perspective_correct_camera_z',
        original_geometry_unchanged=True, object_detail_boxes_xyxy=boxes,
        viewport_hw=[HEIGHT, WIDTH], image_hw=[sheet.height, sheet.width], image=identity(image_path),
        sources={str(p): v for p,v in files.items()}, elapsed_seconds=time.perf_counter()-started)
    path = output/'report.json'; path.write_text(json.dumps(result, sort_keys=True)+'\n'); path.chmod(0o444)
    print('PREVIEW_PASS', args.episode, result['image']['bytes'], result['image']['sha256'])


if __name__ == '__main__':
    main()
