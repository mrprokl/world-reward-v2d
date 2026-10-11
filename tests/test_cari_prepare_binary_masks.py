"""Binary observation admission, not mask recovery or prediction fabrication."""
import ast
from pathlib import Path

import numpy as np
import pytest

import cari_prepare as prepare


def mask(value=0, dtype=np.uint8):
    # A tiny underlying allocation with the real required image dimensions.
    return np.broadcast_to(np.array([[value]], dtype=dtype), (1152, 1536))


@pytest.mark.parametrize('role', [0, 1])
@pytest.mark.parametrize('value', [255])
def test_existing_nonempty_binary_masks_remain_valid(role, value):
    array = mask(value)
    before = array.tobytes()
    prepare._validate_automatic_mask(array, role, np=np)
    assert array.tobytes() == before


@pytest.mark.parametrize('flag', [False, np.bool_(False)])
def test_occluded_object_retains_all_zero_original_pixels_and_full_shape(flag):
    array = mask()
    before = array.tobytes()
    prepare._validate_automatic_mask(array, 1, allow_unobserved_poses=True, pose_observed=flag, np=np)
    assert array.shape == (1152, 1536) and array.tobytes() == before and not array.any()


@pytest.mark.parametrize('role,optin,flag', [
    (0, False, None), (0, True, False), (1, False, None), (1, False, False),
    (1, True, None), (1, True, True), (1, True, np.bool_(True)),
])
def test_empty_human_or_unqualified_object_is_still_rejected(role, optin, flag):
    with pytest.raises(RuntimeError, match='explicitly unobserved object'):
        prepare._validate_automatic_mask(mask(), role, allow_unobserved_poses=optin,
                                         pose_observed=flag, np=np)


@pytest.mark.parametrize('array', [mask(1), mask(254), mask(0, np.float64),
    mask(255, np.int64), np.zeros((3, 3), np.uint8), np.zeros((1152, 1536, 1), np.uint8)])
def test_optin_never_admits_nonbinary_nonoriginal_storage_or_dimensions(array):
    with pytest.raises(RuntimeError, match='binary full-resolution'):
        prepare._validate_automatic_mask(array, 1, allow_unobserved_poses=True,
                                         pose_observed=False, np=np)


@pytest.mark.parametrize('role,optin,flag', [(True, True, False), (2, True, False),
    (1, 1, False), (1, True, 0), (1, True, 'false')])
def test_ambiguous_policy_and_flags_rejected(role, optin, flag):
    with pytest.raises(ValueError, match='boolean pose observation'):
        prepare._validate_automatic_mask(mask(), role, allow_unobserved_poses=optin,
                                         pose_observed=flag, np=np)


def test_actual_hdf5_export_calls_gate_after_original_hash_check_without_pixel_edit():
    tree = ast.parse(Path(prepare.__file__).read_text())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'main')
    segment = ast.get_source_segment(Path(prepare.__file__).read_text(), main)
    start = segment.index('masks_path = output / "automatic_masks.h5"')
    end = segment.index('export_seq = prepare_mhr_wild_export', start)
    writer = segment[start:end]
    assert writer.index('if sha256(path) != expected_hash:') < writer.index('_validate_automatic_mask(')
    assert 'pose_observed=None if pose_observed is None else pose_observed[index]' in writer
    assert 'allow_unobserved_poses=args.allow_unobserved_poses' in writer
    assert 'data=array, compression="lzf"' in writer
    assert 'np.where' not in writer and 'fill' not in writer and 'continue' not in writer
