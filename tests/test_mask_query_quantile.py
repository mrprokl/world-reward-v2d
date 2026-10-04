"""Frozen new8-case CPU availability control, not native inference or accuracy.

Fixture hashes were computed before any operator measurement. Old EP21/R35/etc
remain closed; failed assertions never authorize changed fixtures or margins.
"""
from dataclasses import fields
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pytest

import world_reward.point_surface_queries as operator
from world_reward.point_surface_queries import (
    MaskQueryError, SurfaceQueries, canonical_mask_quantile_queries, canonical_surface_queries,
)

CASE_SHA256 = (
    'c8c6f0829dbfb7505b0fb7313dddd96e4e1a3ee1fb7b2dc7e5f46c51b89434bb',
    'e5628a53a4aadcdd9a307e0a63a513ceb4879022ff2a725421abdc463184186f',
    '20730b362e83234f5e706e2cc6deb151cc5e9b67aeda2e04b85de4e3ed81c360',
    '96108f1381de11cfd01aacd5f5b65dfbeba53c97fb7db13173a96ebba819ad69',
    'cd5297af7f96f92ededdf541333c682b08a985dd7477f96473e8406624743ba3',
    'e32009eadbbabc433fdd1393fdce3f717e8be9e4c37b32e528f4cb7b3458d5de',
    '5b6354f4a0439e557619a948785d84bf7c399e8457d528181c3161c2ef208156',
    '4f550a649ad402c8ed58b0d4e40dccf135437b9e4e9ea8ea128d645246f38580',
)
COHORT_SHA256 = '13c6a154bd278946128b7c255f1313a2d5b1c65f079bfb88fe8cc61776d3ebac'
PROTOCOL_SHA256 = '38c420d483ea023a102b6228977be28c1e0b34a70e29da53f3dfd098719e17e0'


def fixtures():
    y, x = np.mgrid[:48, :64]
    p3 = ((x >= 8) & (x < 52) & (((y >= 4) & (y < 12)) | ((y >= 32) & (y < 40))))
    p3 |= (x >= 8) & (x < 16) & (y >= 12) & (y < 32)
    masks = [(y >= 7) & (y < 11) & (x >= 9) & (x < 17),
             (y == 19) & (x >= 8) & (x < 56), p3,
             ((y >= 4) & (y < 44) & (x >= 4) & (x < 60))
             & ~((y >= 16) & (y < 32) & (x >= 16) & (x < 48)),
             np.zeros((48, 64), bool), p3, p3, p3]
    result = []
    for i, mask in enumerate(masks):
        row = dict(vertices=np.array([[-5., -5, 2], [11, -5, 2], [-5, 11, 2]]),
                   faces=np.array([[0, 1, 2]], np.int64), R0=np.eye(3), t0=np.zeros(3),
                   K=np.array([[64., 0, 32], [0, 48, 24], [0, 0, 1]]),
                   automatic_mask=mask.copy(), inferred_depth_valid=np.ones((48, 64), bool))
        if i == 5: row['inferred_depth_valid'][:] = False
        if i == 6: row['t0'] = np.array([40., 0, 0])
        if i == 7: row['faces'] = np.array([[0, 1, 2], [0, 1, 2]], np.int64)
        result.append(row)
    return result


def manifest(rows):
    result = []
    for i, row in enumerate(rows):
        arrays = {key: dict(dtype=a.dtype.str, shape=list(a.shape),
                            sha256=hashlib.sha256(a.tobytes()).hexdigest()) for key, a in sorted(row.items())}
        digest = hashlib.sha256(json.dumps(arrays, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        result.append(dict(case='P'+str(i+1) if i < 4 else 'N'+str(i-3), arrays=arrays, sha256=digest))
    return result


def arguments(row):
    return row | dict(image_width=64, image_height=48)


def scalar_quantiles(mask):
    pixels = [(y, x) for y in range(48) for x in range(64) if mask[y, x]]
    count = min(32, len(pixels))
    return np.array([pixels[(2*i+1)*len(pixels)//(2*count)] for i in range(count)], np.int64).reshape(-1, 2)


def independent_plane_reference(indices):
    points, bary = [], []
    for y, x in indices:
        px, py = (float(x)+.5-32)/32, (float(y)+.5-24)/24
        points.append([px, py, 2.])
        bary.append([(6-px-py)/16, (px+5)/16, (py+5)/16])
    return np.array(points).reshape(-1, 3), np.array(bary).reshape(-1, 3)


def call(method, row):
    try:
        return method(**arguments(row))
    except ValueError as error:
        return error


def signature(value):
    if isinstance(value, ValueError):
        result = (type(value).__name__, str(value))
        if isinstance(value, MaskQueryError):
            d = value.diagnostics
            result += (json.dumps(d.scalar_report(), sort_keys=True),)
            result += tuple(getattr(d, key).tobytes() for key in
                            ('candidate_indices', 'depth_supported', 'surface_hit',
                             'boundary_ambiguous', 'depth_tied', 'qualified'))
        return result
    result = value.queries if hasattr(value, 'diagnostics') else value
    signature_ = tuple((getattr(result, key.name).dtype.str, getattr(result, key.name).shape,
                        getattr(result, key.name).tobytes()) for key in fields(SurfaceQueries))
    if hasattr(value, 'diagnostics'):
        signature_ += (json.dumps(value.diagnostics.scalar_report(), sort_keys=True),)
    return signature_


def test_frozen_eight_case_control_before_measurement_and_inclusive30s_gate():
    started = time.monotonic()
    source_sha = hashlib.sha256(Path(operator.__file__).read_bytes()).hexdigest()
    protocol = Path(__file__).resolve().parents[1]/'docs/mask_query_quantile_protocol.md'
    assert hashlib.sha256(protocol.read_bytes()).hexdigest() == PROTOCOL_SHA256
    rows = fixtures(); ledger = manifest(rows)
    assert tuple(row['sha256'] for row in ledger) == CASE_SHA256
    assert hashlib.sha256(json.dumps(ledger, sort_keys=True, separators=(',', ':')).encode()).hexdigest() == COHORT_SHA256
    # ALL full arrays frozen before the first A or B call. No fixture replacement.
    results = []
    for i, row in enumerate(rows):
        a = [call(canonical_surface_queries, row) for _ in range(2)]
        b = [call(canonical_mask_quantile_queries, row) for _ in range(2)]
        assert signature(a[0]) == signature(a[1]) and signature(b[0]) == signature(b[1])
        expected = scalar_quantiles(row['automatic_mask'])
        diag = b[0].diagnostics
        np.testing.assert_array_equal(diag.candidate_indices, expected)
        assert diag.raycast_complete and diag.depth_support_known
        if i < 4:
            assert not isinstance(b[0], ValueError)
            result = b[0].queries
            assert len(result.canonical_points) == diag.distinct_witnesses == 32
            assert diag.qualified.all() and not diag.boundary_ambiguous.any() and not diag.depth_tied.any()
            np.testing.assert_array_equal(result.grid_indices, expected)
            np.testing.assert_array_equal(result.query_points, np.column_stack((np.zeros(32), expected+.5)))
            points, bary = independent_plane_reference(result.grid_indices)
            np.testing.assert_allclose(result.canonical_points, points, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result.camera_depth_m, 2, atol=1e-12, rtol=0)
            np.testing.assert_allclose(result.barycentric, bary, atol=1e-12, rtol=0)
            if i < 2: assert isinstance(a[0], ValueError) and 'Fewer than8' in str(a[0])
            else:
                assert isinstance(a[0], SurfaceQueries)
                pa, ba = independent_plane_reference(a[0].grid_indices)
                np.testing.assert_allclose(a[0].canonical_points, pa, atol=1e-12, rtol=0)
                np.testing.assert_allclose(a[0].barycentric, ba, atol=1e-12, rtol=0)
        else:
            assert isinstance(a[0], ValueError) and isinstance(b[0], MaskQueryError)
            assert 'depth tie' in str(b[0]) if i == 7 else 'Fewer than8' in str(b[0])
            if i == 7: assert diag.depth_tied.all() and not diag.qualified.any()
        yy, xx = np.meshgrid(np.arange(2, 48, 4), np.arange(2, 64, 4), indexing='ij')
        a_mask_count = int(np.count_nonzero(row['automatic_mask'][yy, xx]))
        results.append(dict(case=ledger[i]['case'], A_candidates_in_mask=a_mask_count,
                            A_status='fail' if isinstance(a[0], ValueError) else 'pass',
                            B_status='fail' if isinstance(b[0], ValueError) else 'pass',
                            B=diag.scalar_report()))
    assert manifest(rows) == ledger
    assert hashlib.sha256(Path(operator.__file__).read_bytes()).hexdigest() == source_sha
    elapsed = time.monotonic()-started
    assert elapsed <= 30
    print(json.dumps(dict(status='pass', source_sha256=source_sha, cohort_sha256=COHORT_SHA256,
                          elapsed_seconds=elapsed, paired_cases=8, original_operator_calls=16,
                          mask_quantile_operator_calls=16, adoption_authorized=False, results=results), sort_keys=True))


def test_partial_depth_support_retains32_frozen_slots_no_refill():
    row = fixtures()[2]; pixels = scalar_quantiles(row['automatic_mask'])
    row['inferred_depth_valid'][pixels[:5, 0], pixels[:5, 1]] = False
    result = canonical_mask_quantile_queries(**arguments(row))
    assert len(result.queries.canonical_points) == 27
    np.testing.assert_array_equal(result.diagnostics.candidate_indices, pixels)
    assert result.diagnostics.depth_supported.tolist() == [False]*5+[True]*27
    row['inferred_depth_valid'][pixels[5:26, 0], pixels[5:26, 1]] = False
    with pytest.raises(MaskQueryError, match='Fewer than8') as captured:
        canonical_mask_quantile_queries(**arguments(row))
    diag = captured.value.diagnostics
    assert diag.qualified.sum() == 6 and len(diag.candidate_indices) == 32


def test_masks_with_less_than32_pixels_keep_all_original_distinct_slots():
    row = fixtures()[0]; row['automatic_mask'][:] = False; row['automatic_mask'][10, 8:17] = True
    result = canonical_mask_quantile_queries(**arguments(row))
    assert len(result.queries.canonical_points) == 9
    np.testing.assert_array_equal(result.queries.grid_indices, [[10, x] for x in range(8, 17)])


def test_new_results_and_error_diagnostics_own_immutable_copies():
    row = fixtures()[0]; result = canonical_mask_quantile_queries(**arguments(row))
    for a in (*vars(result.queries).values(), *(getattr(result.diagnostics, k) for k in
               ('candidate_indices', 'depth_supported', 'surface_hit', 'boundary_ambiguous', 'depth_tied', 'qualified'))):
        assert not a.flags.writeable
        with pytest.raises(ValueError): a.setflags(write=True)
    row['automatic_mask'][:] = False; row['inferred_depth_valid'][:] = False
    assert result.diagnostics.depth_supported.all() and len(result.queries.canonical_points) == 32


@pytest.mark.parametrize('key,value', [
    ('inferred_depth_valid', np.ones((48, 64), np.uint8)),
    ('vertices', np.zeros((3, 3))), ('vertices', np.full((3, 3), np.nan)),
    ('faces', np.array([[0, 1, 9]], np.int64)), ('faces', np.array([[0, 1, 2]], np.int32)),
    ('R0', np.diag([1., 1., -1.])), ('K', np.eye(3, dtype=np.int64)),
])
def test_invalid_input_retains_frozen_candidates_without_false_missing_evidence(key, value):
    row = fixtures()[0]; expected = scalar_quantiles(row['automatic_mask']); row[key] = value
    with pytest.raises(MaskQueryError) as captured: canonical_mask_quantile_queries(**arguments(row))
    d = captured.value.diagnostics
    np.testing.assert_array_equal(d.candidate_indices, expected)
    assert not d.raycast_complete
    if key == 'inferred_depth_valid': assert not d.depth_support_known


def test_boundary_ambiguity_is_whole_call_failure_not_pixel_shift_or_refill():
    row = fixtures()[0]; pixels = scalar_quantiles(row['automatic_mask']); y, x = pixels[0]
    p, _ = independent_plane_reference([[y, x]])
    row['vertices'] = np.array([p[0], [p[0, 0]+5, p[0, 1]-5, 2], [p[0, 0]+5, p[0, 1]+5, 2]])
    with pytest.raises(MaskQueryError, match='ambiguous') as captured:
        canonical_mask_quantile_queries(**arguments(row))
    assert captured.value.diagnostics.boundary_ambiguous[0]
    np.testing.assert_array_equal(captured.value.diagnostics.candidate_indices, pixels)


def test_ray_triangle_arithmetic_shared_by_legacy_and_mask_quantile(monkeypatch):
    original = operator._ray_triangle_hits; grids = []
    def spy(*args, **kwargs):
        grids.append((args[5].copy(), kwargs.get('progress') is not None))
        return original(*args, **kwargs)
    monkeypatch.setattr(operator, '_ray_triangle_hits', spy)
    row = fixtures()[2]
    canonical_surface_queries(**arguments(row)); canonical_mask_quantile_queries(**arguments(row))
    assert [(len(g), p) for g, p in grids] == [(192, False), (32, True)]
    np.testing.assert_array_equal(grids[1][0], scalar_quantiles(row['automatic_mask']))
