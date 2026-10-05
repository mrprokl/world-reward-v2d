import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "infra"))
from world_reward.surface_pose_geometry import (
    compact_surface, serialized_mesh, camera_roundtrip, transform_points, surface_signature,
)


def mesh():
    # An open sheet plus an orphan, not a closed solid.
    return np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [2., 2., 2.]]), np.array([[0, 1, 2]], np.int64)


def padded(v, f):
    return np.vstack((v, np.repeat(v[:1], 4096-len(v), axis=0))), np.vstack((f, np.zeros((4096-len(f), 3), np.int64)))


def serialized(v=None, f=None, a=None, b=None):
    v, f = mesh() if v is None else (v, f)
    a = np.eye(4) if a is None else a
    b = a.copy() if b is None else b
    local = v.astype(np.float32).astype(np.float64)
    loaded = transform_points(local, b)
    return dict(source_vertices=v, source_faces=f, local_vertices=local, local_faces=f.copy(),
        raw_world_triangles=transform_points(local, a)[f], loaded_vertices=loaded,
        loaded_faces=f.copy(), native_vertices=loaded.astype(np.float32), native_faces=f.copy(),
        raw_matrix=a, effective_matrix=b, projected_matrix=b.copy())


def test_open_orphan_prefix_preserved_owned_and_full_boundary():
    v, f = mesh(); pv, pf = padded(v, f)
    out, faces, active, top = compact_surface(pv, pf, reference_vertices=v, reference_faces=f)
    assert out.shape == (4, 3) and faces.shape == (1, 3) and active.tolist() == [0]
    assert top['canonical_unused_vertices_preserved'] == 1 and not top['diagnostics']['closed']
    assert top['diagnostics']['boundary_edges'] == 3
    pv[0] = 9
    assert out[0].tolist() == [0., 0., 0.]
    for a in (out, faces, active):
        assert not a.flags.writeable
        with pytest.raises(ValueError): a.setflags(write=True)


def test_missing_authority_no_ambiguous4096_unpadding():
    with pytest.raises(ValueError, match='independently bound'):
        compact_surface(*padded(*mesh()))


@pytest.mark.parametrize('which', ['vertex', 'face', 'originalface', 'dtype', 'count'])
def test_padding_no_repair(which):
    v, f = mesh(); pv, pf = padded(v, f)
    kwargs = dict(reference_vertices=v, reference_faces=f)
    if which == 'vertex': pv[-1] = 7
    elif which == 'face': pf[-1] = [0, 1, 1]
    elif which == 'originalface': pf[0] = [0, 2, 1]
    elif which == 'dtype': pf = pf.astype(np.int32)
    else: kwargs['canonical_vertex_count'] = 3
    with pytest.raises(ValueError): compact_surface(pv, pf, **kwargs)


def test_zero_meaningful_face_not_padding_and_signedzero_lineage():
    v, f = mesh(); f[0] = 0
    with pytest.raises(ValueError): compact_surface(v, f)
    v, f = mesh(); pv, pf = padded(v, f); pv[1, 2] = -0.
    with pytest.raises(ValueError, match='changed'):
        compact_surface(pv, pf, reference_vertices=v, reference_faces=f)


def test_raw_and_effective_distinct_source_supplied_projection():
    a = np.eye(4); a[0, 0] += 1e-8
    b = np.eye(4)
    kwargs = serialized(a=a, b=b)
    nv, nf, proof = serialized_mesh(**kwargs)
    assert proof['native_rigid_projection_used'] is True
    assert proof['raw_world_exact'] and proof['effective_loader_exact']
    assert not proof['local_geometry_repaired'] and not proof['embedding_verified']
    assert nv.shape == (4, 3) and nf.tolist() == [[0, 1, 2]] and not nv.flags.writeable


@pytest.mark.parametrize('field', ['local_vertices', 'local_faces', 'raw_world_triangles',
    'loaded_vertices', 'loaded_faces', 'native_vertices', 'native_faces', 'projected_matrix'])
def test_serialization_no_dropped_rows_faces_or_guessed_transform(field):
    kwargs = serialized()
    kwargs[field] = kwargs[field].copy()
    if field.endswith('faces'):
        kwargs[field][0] = [0, 2, 1]
    elif field == 'projected_matrix': kwargs[field][0, 3] = .1
    else: kwargs[field].flat[0] += .1
    with pytest.raises(ValueError): serialized_mesh(**kwargs)


def test_native_orphan_loss_rejected_even_if_triangles_unchanged():
    kwargs = serialized(); kwargs['native_vertices'] = kwargs['native_vertices'][:3]
    with pytest.raises(ValueError): serialized_mesh(**kwargs)


def test_quantization_collapse_is_not_hidden_by_triangle_signature():
    v = np.array([[1., 0., 0.], [1.+2**-30, 0., 0.], [1., 1., 0.]])
    f = np.array([[0, 1, 2]], np.int64)
    with pytest.raises(ValueError, match='collapsed'):
        serialized_mesh(**serialized(v, f))


def test_component_boundary_signature_ignores_row_permutation_not_orientation():
    v, f = mesh(); order = np.array([2, 0, 3, 1]); inverse = np.argsort(order)
    assert surface_signature(v, f) == surface_signature(v[order], inverse[f])
    assert surface_signature(v, f) != surface_signature(v, f[:, ::-1])


def test_camera_fullT_allcanonicalrows_and_savedF32_semantics():
    v, f = mesh(); r = np.repeat(np.eye(3)[None], 3, axis=0)
    t = np.array([[0.,0.,0.], [.1,0.,0.], [.2,0.,0.]])
    p = np.repeat(np.eye(4)[None], 3, axis=0).astype(np.float32); p[:, :3, 3] = t
    assert camera_roundtrip(v, f, r, t, p, v.astype(np.float32), f) < 1e-5
    nv = v.astype(np.float32); nv[-1, 0] += .01
    with pytest.raises(ValueError, match='roundtrip'):
        camera_roundtrip(v, f, r, t, p, nv, f)
    p[2, 0, 0] = 2
    with pytest.raises(ValueError, match='proper rigid'):
        camera_roundtrip(v, f, r, t, p, v.astype(np.float32), f)


def test_camera_rejects_missing_pose_wrongdtype_nonfinite_and_empty():
    v, f = mesh(); r=np.eye(3)[None]; t=np.zeros((1,3)); p=np.eye(4)[None].astype(np.float32)
    for rr,tt,pp in ((r,t,p.astype(np.float64)), (r,t,p[:0]), (r[:0],t[:0],p[:0])):
        with pytest.raises(ValueError): camera_roundtrip(v,f,rr,tt,pp,v.astype(np.float32),f)
    p[0,0,3]=np.nan
    with pytest.raises(ValueError): camera_roundtrip(v,f,r,t,p,v.astype(np.float32),f)


def test_transform_arithmetic_matches_source_verified_primary_fixture():
    # Literal numerical body of pinned Trimesh5.1.0 transformations.py
    # (74801B SHA644b112736124b7803c028a248279926d10f748649634006d493a90722eae360).
    # Independent test fixture, not a local Trimesh installation/model download.
    def native(points,matrix):
        points=np.asanyarray(points,dtype=np.float64)
        matrix=np.asanyarray(matrix,dtype=np.float64)
        count,dim=points.shape
        if np.abs(matrix-np.eye(4)[:dim+1,:dim+1]).max()<1e-8:
            return np.ascontiguousarray(points.copy())
        stack=np.column_stack((points,np.ones(count)))
        return np.dot(matrix,stack.T).T[:,:dim]
    points=np.random.default_rng(934).normal(size=(37,3))
    points[0,1]=-0.
    near=np.eye(4);near[0,3]=2**-30
    assert transform_points(points,near).tobytes()==points.tobytes()
    far=np.eye(4);far[:3,3]=[.125,-.25,.5]
    angle=.371;far[:3,:3]=[[np.cos(angle),-np.sin(angle),0],[np.sin(angle),np.cos(angle),0],[0,0,1]]
    for matrix in (np.eye(4),near,far):
        assert transform_points(points,matrix).tobytes()==native(points,matrix).tobytes()
