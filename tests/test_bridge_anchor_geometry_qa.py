"""Procedural tiny predicted geometry, never challenge labels/media/assets."""
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
import pytest


@pytest.fixture
def qa():
    path = Path(__file__).parents[1]/'infra/bridge_anchor_geometry_qa.py'
    spec = importlib.util.spec_from_file_location('bridge_geometry_qa_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture():
    K = np.array([[16., 0., 8.], [0., 16., 8.], [0., 0., 1.]])
    uv = np.array([[2., 2.], [14., 2.], [14., 14.], [2., 14.]])
    vertices = np.column_stack(((uv-8)*2/16, np.full(4, 2.))).astype(np.float32)
    faces = np.array([[0, 1, 2], [0, 2, 3]], np.int64)
    mask = np.zeros((16, 16), bool); mask[2:14, 2:14] = True
    names = [f'other{i}' for i in range(127)]
    # Deliberately non-ordinal/interleaved semantic positions.
    slots = {'l_wrist':91, 'l_index1':7, 'l_middle1':112,
             'r_wrist':37, 'r_index1':122, 'r_middle1':4}
    joints = np.tile([0., 0., 2.], (127, 1)).astype(np.float32)
    for name, index in slots.items(): names[index] = name
    for side, wrist_x in (('l', -.25), ('r', .25)):
        joints[slots[side+'_wrist']] = [wrist_x, 0., 2.]
        joints[slots[side+'_index1']] = [wrist_x+.08, -.15, 2.]
        joints[slots[side+'_middle1']] = [wrist_x, -.2, 2.]
    return vertices, joints, faces, K, mask, names


def test_perfect_proxy_pass_named_camera_frame_and_no_claims_or_mutation(qa):
    args = fixture(); copies = [x.copy() if isinstance(x, np.ndarray) else x.copy() for x in args]
    for x in args:
        if isinstance(x, np.ndarray): x.flags.writeable = False
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'pass' and result['silhouette']['IoU'] == 1.
    assert result['silhouette']['predicted_pixels'] == 144
    assert result['left']['valid'] and result['right']['valid']
    assert result['left']['mask_support_proxy_passed']
    assert not result['independent_RGB_hand_support_verified']
    assert not result['accuracy_verified'] and not result['contact_verified'] and not result['adoption']
    assert result['operator_only'] and not result['geometry_mutated']
    np.testing.assert_array_equal(result['left']['origin_camera_m'], args[1][91])
    for actual, original in zip(args, copies):
        np.testing.assert_array_equal(actual, original)
    json.dumps(result, allow_nan=False)


def test_anatomical_frame_is_camera_equivariant_proper_and_no_native_axes(qa):
    _, joints, _, _, _, names = fixture()
    original = qa.anatomical_frame(joints, names, 'l')
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    moved = joints.astype(np.float64)@rotation.T + [.1, .2, .3]
    result = qa.anatomical_frame(moved, names, 'l')
    R = np.array(result['R_frame_to_camera'])
    np.testing.assert_allclose(R, rotation@original['R_frame_to_camera'], atol=1e-14)
    np.testing.assert_allclose(R.T@R, np.eye(3), atol=1e-14)
    assert np.linalg.det(R) == pytest.approx(1., abs=1e-14)
    np.testing.assert_allclose(result['origin_camera_m'], rotation@joints[91]+[.1,.2,.3])


@pytest.mark.parametrize('fault', ['zero', 'collinear', 'near_collinear', 'nan', 'behind', 'missing'])
def test_invalid_named_left_frame_never_repaired(qa, fault):
    args = list(fixture()); j, names = args[1], args[5]
    if fault == 'zero': j[7] = j[91]
    elif fault == 'collinear': j[7] = j[91]+(j[112]-j[91])*.5
    elif fault == 'near_collinear': j[7] = j[91]+(j[112]-j[91])*.5; j[7,0] += 1e-6
    elif fault == 'nan': j[7,0] = np.nan
    elif fault == 'behind': j[7,2] = -1
    else: names[7] = 'not_index'
    frame = qa.anatomical_frame(j, names, 'l')
    assert not frame['valid'] and frame['R_frame_to_camera'] is None and frame['reasons']
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'fail' and (result['gate_reasons'] or result['invalid_reasons'])


def test_right_only_degeneracy_diagnostic_does_not_rescue_or_block_left_gate(qa):
    args = list(fixture()); args[1][122] = args[1][37]
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'pass' and not result['right']['valid'] and result['left']['valid']
    assert result['gates']['right_diagnostic_only']


def test_mask_support_exact_euclidean_pixel_centres_radius5_and_no_dilation(qa):
    K = np.array([[16.,0.,8.],[0.,16.,8.],[0.,0.,1.]])
    mask = np.zeros((16,16),bool); mask[8,13] = True
    def project(u,v): return np.array([(u-8)*2/16,(v-8)*2/16,2.])
    on = qa._mask_support(project(8.5,8.5),K,mask,'named')
    assert on['valid'] and on['nearest_foreground_within_radius_px'] == 5.
    outside = qa._mask_support(project(8.49,8.5),K,mask,'named')
    assert not outside['valid'] and outside['nearest_foreground_within_radius_px'] is None
    diagonal = qa._mask_support(project(8.5,3.5),K,mask,'named')
    assert not diagonal['valid']  # A square/L-infinity neighborhood would falsely pass.
    assert mask.sum() == 1


@pytest.mark.parametrize('point', [[-1.01,0.,2.],[1.,0.,2.],[0.,0.,-2.],[np.nan,0.,2.]])
def test_outside_nonfinite_and_behind_projection_explicitly_invalid(qa, point):
    _, _, _, K, mask, _ = fixture()
    result = qa._mask_support(point,K,mask,'named')
    assert not result['valid'] and not result['mask_support_proxy'] and result['reasons']


def test_silhouette_threshold_exactly_point7_not_fitted_or_relaxed(qa):
    args = list(fixture())
    # 144 original prediction pixels; independent foreground subsets, not fitted geometry.
    args[4][:] = False; args[4].ravel()[np.array([y*16+x for y in range(2,14) for x in range(2,14)])[:100]] = True
    result = qa.evaluate_geometry(*args)
    assert result['silhouette']['IoU'] == 100/144 < .7 and result['status'] == 'fail'
    args[4][10,6] = True  # 101/144 > .7, with all named landmarks still supported.
    passed = qa.evaluate_geometry(*args)
    assert passed['silhouette']['IoU'] == 101/144 and passed['status'] == 'pass'
    assert passed['gates']['silhouette_IoU_min'] == .7


def test_exact_inclusive_iou_boundary_point7(qa):
    args = list(fixture()); args[0][:,:2] *= 5/6
    args[4][:] = False; args[4][3:10,3:13] = True
    at_boundary = qa.evaluate_geometry(*args)
    assert at_boundary['silhouette']['predicted_pixels'] == 100
    assert at_boundary['silhouette']['IoU'] == .7 and at_boundary['status'] == 'pass'
    args[4][3,3] = False
    below = qa.evaluate_geometry(*args)
    assert below['silhouette']['IoU'] == .69 and below['status'] == 'fail'


def test_unsupported_left_landmark_fails_even_if_silhouette_perfect(qa):
    args = list(fixture())
    # Keep noncollinear left joints positive/in-image but more than5px outside person foreground.
    args[1][91] = [-.9375,-.9375,2.]
    args[1][7] = [-.9,-.9375,2.]
    args[1][112] = [-.9375,-.9,2.]
    # At the top-left image corner distance to nearest foreground is~2.83: use a smaller person.
    args[0][:,:2] *= .25
    args[4][:] = False; args[4][6:10,6:10] = True
    result = qa.evaluate_geometry(*args)
    assert result['silhouette']['IoU'] == 1 and result['left']['valid']
    assert not result['left']['mask_support_proxy_passed'] and result['status'] == 'fail'


@pytest.mark.parametrize('fault', ['vertex_nan','vertex_behind','degenerate','faces_float','camera',
                                 'joint_nan_elsewhere','joint_behind_elsewhere','mask_float','mask_array','names_duplicate','names_count'])
def test_original_invalid_inputs_fail_without_repair(qa,fault):
    args = list(fixture())
    if fault == 'vertex_nan': args[0][0,0] = np.nan
    elif fault == 'vertex_behind': args[0][0,2] = -1
    elif fault == 'degenerate': args[0][1] = args[0][0]
    elif fault == 'faces_float': args[2] = args[2].astype(float)
    elif fault == 'camera': args[3][0,2] = 8.5
    elif fault == 'joint_nan_elsewhere': args[1][0,0] = np.nan
    elif fault == 'joint_behind_elsewhere': args[1][0,2] = -1
    elif fault == 'mask_float': args[4] = args[4].astype(float)
    elif fault == 'mask_array': args[4] = np.ma.array(args[4],mask=False)
    elif fault == 'names_duplicate': args[5][0] = args[5][1]
    else: args[5].pop()
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'fail' and result['invalid_reasons']


def test_empty_person_or_empty_offscreen_silhouette_never_passes(qa):
    args = list(fixture()); args[4][:] = False
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'fail' and result['silhouette']['IoU'] == 0.
    args[0][:,0] += 100
    empty = qa.evaluate_geometry(*args)
    assert empty['status'] == 'fail' and empty['silhouette']['IoU'] is None


def test_expired_deadline_and_modified_renderer_pin_fail_closed(qa,monkeypatch):
    result = qa.evaluate_geometry(*fixture(),deadline=float(time.monotonic()-1))
    assert result['status'] == 'fail' and 'deadline' in result['invalid_reasons'][0]
    monkeypatch.setattr(qa,'RENDERER_SHA256','0'*64)
    result = qa.evaluate_geometry(*fixture())
    assert result['status'] == 'fail' and 'source bytes/SHA' in result['invalid_reasons'][0]


def test_renderer_is_standalone_only_has_fixed_budget_and_no_fixture_calls(qa):
    renderer = qa._load_renderer()
    assert renderer.BUDGET == qa.BUDGET_SECONDS == 100
    assert renderer.MAX_CANDIDATES == qa.MAX_CANDIDATES == 20_000_000
    source = Path(qa.__file__).read_text()
    assert 'renderer.render_triangles(' in source
    assert not any(word in source for word in ('micro_fixture(', 'independent_reference(', 'validate_micro_reference(',
        'torch', 'np.load(', 'np.save', 'PIL', 'slerp', 'readGT', 'alpha_fit'))


def test_overflowing_hand_vectors_are_invalid_not_nan_json(qa):
    _, joints, _, _, _, names = fixture(); joints = joints.astype(np.float64)
    joints[7,0] = 1e308
    result = qa.anatomical_frame(joints,names,'l')
    assert not result['valid'] and result['normalized_cross_length'] is None
    json.dumps(result,allow_nan=False)


def test_fixed_renderer_work_guard_failure_preserves_geometry(qa,monkeypatch):
    renderer = qa._load_renderer(); renderer.MAX_CANDIDATES = 1
    monkeypatch.setattr(qa,'_load_renderer',lambda: renderer)
    args = fixture(); before = args[0].copy()
    result = qa.evaluate_geometry(*args)
    assert result['status'] == 'fail' and 'ceiling' in result['invalid_reasons'][0]
    np.testing.assert_array_equal(args[0],before)
