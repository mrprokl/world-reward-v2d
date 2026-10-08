"""Tiny manufactured QA contracts only; no assets, checkpoints or inference."""
import ast
import copy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'infra'))
import full4d_video as video


def reports():
    common = dict(status='pass', episode_index=9, input_track='track_1', input_sha256='a'*64,
        ground_truth_used=False, hand_labeled_test=False, oracle_modes=[])
    body = dict(common, stage='sam3d_body_three_frame_smoke', frame_indices=[0, 20, 39],
        total_video_frames=40, geometry_units='metres', mhr_geometry_forward_verified=True,
        frame_independent_initializer_only=True, human_identity_clip_constant=False,
        input_dataset_revision=video.DATASET)
    scale = dict(common, stage='predicted_human_anchored_moge2_pointmaps', frame_indices=[0, 20, 39],
        coordinate_frame='OpenCV_x_right_y_down_z_forward',
        pointmap_scale_application='one_clip_scalar_to_MoGe2_XYZ_already_applied')
    obj = dict(common, stage='sam3d_objects_grounded_fixed_frame', frame_index=0,
        scale_source='already_human_anchored_MoGe2_no_second_scalar')
    mask = dict(common, stage='automatic_masks', frames=40, full_original_grid=True,
        grounding_model='gemini-3.5-flash', mask_model='facebook/sam3.1',
        source_revision='2345a4ad109ac29c569da749c91d84f10dc08c40',
        model_revision='daa63191845a41281374e725f4c9e51c7a824460',
        empty_mask_interpolation=False, manual_labels=False)
    return [body, scale, obj, mask]


def test_pass_means_saved_initialization_only_and_preserves_receipts():
    rows = reports(); before = copy.deepcopy(rows)
    video.initialization_contract(*rows, 9, 40, 'a'*64)
    assert rows == before


@pytest.mark.parametrize('index,key,value', [(0, 'status', 'running'),
    (0, 'ground_truth_used', True), (1, 'oracle_modes', ['GT']),
    (2, 'frame_index', 14), (2, 'scale_source', 'independent_MoGe1'),
    (3, 'mask_model', 'facebook/sam2'), (3, 'manual_labels', True),
    (3, 'empty_mask_interpolation', True), (0, 'frame_indices', [0, 14, 29])])
def test_wrong_or_incomplete_producer_is_not_promoted(index, key, value):
    rows = reports(); rows[index][key] = value
    with pytest.raises(ValueError): video.initialization_contract(*rows, 9, 40, 'a'*64)


def test_no_replacement_clip_or_boolean_episode():
    for episode in (0, True):
        with pytest.raises(ValueError): video.initialization_contract(*reports(), episode, 40, 'a'*64)


def test_cpu_scene_is_stateless_joint_depth_and_floor_only():
    camera = np.array([[400., 0, 160.], [0, 400., 120.], [0, 0, 1.]])
    human = np.array([[-.2, -.2, 3.], [.2, -.2, 3.], [0., .2, 3.]])
    obj = human.copy(); obj[:, :2] *= .5; obj[:, 2] = 2.
    faces = np.array([[0, 1, 2]], np.int64)
    before = [x.copy() for x in (human, obj, faces, camera)]
    first, receipt = video.initialization_scene(human, faces, obj, faces, camera)
    second, receipt2 = video.initialization_scene(human, faces, obj, faces, camera)
    np.testing.assert_array_equal(first, second); assert receipt == receipt2
    assert tuple(first[120, 160]) == video.OBJECT_RGB
    assert receipt['device'] == 'cpu' and receipt['virtual_floor']['camera_y_m'] == .2
    assert receipt['virtual_floor']['reconstruction_moved_to_floor'] is False
    for old, new in zip(before, (human, obj, faces, camera)): np.testing.assert_array_equal(old, new)
    back = obj.copy(); back[:, 2] = 4.
    behind, _ = video.initialization_scene(human, faces, back, faces, camera)
    assert tuple(behind[120, 160]) == video.HUMAN_RGB


def test_initialization_sheet_labels_and_bytes_are_bounded(monkeypatch):
    from PIL import ImageDraw
    texts = []; original = ImageDraw.ImageDraw.text
    def record(self, xy, text, *args, **kwargs):
        texts.append(text); return original(self, xy, text, *args, **kwargs)
    monkeypatch.setattr(ImageDraw.ImageDraw, 'text', record)
    rgb = np.full((video.HEIGHT, video.WIDTH, 3), 30, np.uint8)
    masks = [np.zeros((video.HEIGHT, video.WIDTH), bool) for _ in range(2)]
    raw = video.initialization_sheet(rgb, masks, rgb.copy(), 9, 40)
    assert 0 < len(raw) <= 100_000 and raw[:2] == b'\xff\xd8'
    label = ' '.join(texts)
    assert 'INITIAL 3D' in label and 'NOT full 4D / temporal validation / score' in label
    assert 'frame0 only' in label and 'ASSUMED' in label and 'SAM3.1' in label


def test_host_route_has_no_numpy_dependency_and_fails_outside_azure():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, '-S', str(root/'infra/full4d_video.py'),
        '--initialization-host'], text=True, capture_output=True, timeout=10)
    assert result.returncode != 0 and 'Azure host observer only' in result.stderr
    assert 'ModuleNotFoundError' not in result.stderr


def test_observer_source_never_executes_learned_pipeline_or_takes_gpu_lease():
    path = Path(video.__file__)
    tree = ast.parse(path.read_text())
    methods = {n.name: ast.get_source_segment(path.read_text(), n) for n in tree.body if isinstance(n, ast.FunctionDef)}
    observer = methods['run_initialization'] + methods['initialization_host']
    assert '--gpus' not in observer and 'torch' not in observer and 'flock' not in observer
    assert 'model_execution=False' in observer and 'full_4D_produced=False' in observer
    assert "--network', 'none'" in observer and "--pids-limit', '256'" in observer
    assert "'--memory', '8g'" in observer and "'--cpus', '2'" in observer
    assert "source_mask_dir/'masks'/key" in observer and "seed_rows(gemini['rows']" in observer
    assert 'Path(grounding[\'pointmap_path\'])' in observer and 'sources.verify()' in observer


def test_dispatch_wrapper_consumes_only_one_validated_positional_producer():
    wrapper = Path(video.__file__).with_name('run_full4d_video_initialization.sh').read_text()
    assert '[[ $# == 1 ]]' in wrapper and 'PRODUCER="$1"' in wrapper
    assert '"$PRODUCER" =~ ^[0-9a-f]{40}$' in wrapper
    assert 'WR_PRODUCER_REVISION="$PRODUCER"' in wrapper
    assert '${WR_PRODUCER_REVISION:?' not in wrapper
