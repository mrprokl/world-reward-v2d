"""New explicit precision backend; historical helpers are never modified.

Only the global face-area guard is replaced by exact non-collinearity of stored
positions. Topology, volume, birth maps, sampling, containment and official
packing math below remain static copies, checked against the frozen originals.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import re
import numpy as np
from scipy.spatial import cKDTree
from world_reward.exact_triangle_predicates import validate_exact_triangle_non_degeneracy
from mesh_link_gate import _surface
from guarded_mesh_gate import component_labels
from mesh_endpoint_gate import _components, solid_winding_and_distances, WINDING_ATOL
from object_budget_endpoint import exact_weld, _triangle_rows

LEGACY_HASHES = {
    "mesh_link_gate.py": "48fa8c41e8c05e60d7fe492cb50ed7f94fba937d94dbb9fb685fc88d6925c0b0",
    "guarded_mesh_gate.py": "7e440b6b51cf4a7c1c72a0f616b72e9a62528f1e920d40e21751a87d3d8606da",
    "mesh_endpoint_gate.py": "6684203fdd673752f4167d1926c153784407e07d8eb5c7e2c9cab7d6a078b7a5",
    "object_budget_endpoint.py": "a541adf8a89823733e02150e49063c8adef5766e46df1af3ccfb6373bd37a119",
    "object_budget_guarded.py": "40dc5a85b51fc5edc76d7cfde79de20c90d2479d549f3da31fbe859e27d72fea",
    "volume_mesh_gate.py": "53faaf1c913c4296251680a598414d1b918d36b795d8a44267037e6d26937187",
    "mesh_volume_qem.cpp": "7e3e55791cf66a89080c327a976cb0e04b5f0af511fef9322885b49d93b9e273",
    "mesh_guarded_qem.cpp": "964d89560cbbae6fa2dd8b76e480a6ab7dae3073e0fa419604707e4a0d9af081",
}


def validate_legacy_sources():
    for name, expected in LEGACY_HASHES.items():
        path = Path(__file__).with_name(name)
        if path.is_symlink() or not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("Frozen legacy helper/source changed")
    return dict(LEGACY_HASHES)


def _exact_faces(vertices, faces):
    # Promotion of stored float32 to a private float64 predicate view is exact;
    # all subsequent topology/volume math still uses the original dtype/values.
    if vertices.dtype.itemsize not in (4, 8):
        raise ValueError("Require stored float32 or float64 positions")
    predicate_view = vertices.astype(np.float64, copy=False)
    if not np.array_equal(predicate_view, vertices):
        raise ValueError("Predicate view changed represented coordinates")
    return validate_exact_triangle_non_degeneracy(predicate_view, faces)


def exact_mesh_topology(vertices, faces) -> dict:
    """Exact face predicates; unchanged oriented manifold and volume checks.

    Disconnected inward components are legitimate cavity boundaries. Sign and
    Euler signatures are compared separately; positive total volume alone is
    insufficient. No arrays are mutated or faces deleted to obtain a pass.
    """
    v, f = np.asarray(vertices), np.asarray(faces)
    if (np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces)
            or v.ndim != 2 or v.shape[1] != 3 or len(v) < 4
            or f.ndim != 2 or f.shape[1] != 3 or len(f) < 4
            or v.dtype.kind != 'f' or f.dtype.kind not in 'iu'
            or not np.isfinite(v).all() or np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError("Mesh arrays must be finite floating Vx3 and valid integer Fx3")
    if np.any(np.sort(f, axis=1)[:, 1:] == np.sort(f, axis=1)[:, :-1]):
        raise ValueError("Repeated face indices are not a manifold surface")
    extent = float(np.linalg.norm(np.ptp(v, axis=0)))
    if extent <= 0:
        raise ValueError("Collapsed global extent is forbidden")
    _exact_faces(v, f)
    directed = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges, inverse, counts = np.unique(np.sort(directed, axis=1), axis=0, return_inverse=True, return_counts=True)
    orientations = np.where(directed[:, 0] < directed[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(inverse, weights=orientations) != 0):
        raise ValueError("Edges must each have exactly two opposite incidences")
    active = np.unique(f)
    adjacency = [[] for _ in range(len(v))]
    links = [[] for _ in range(len(v))]
    for a, b in edges:
        adjacency[a].append(int(b)); adjacency[b].append(int(a))
    for a, b, c in f:
        links[a].append((int(b), int(c)))
        links[b].append((int(c), int(a)))
        links[c].append((int(a), int(b)))
    for vertex in active:
        graph = {}
        for a, b in links[vertex]:
            graph.setdefault(a, []).append(b); graph.setdefault(b, []).append(a)
        if any(len(neighbors) != 2 for neighbors in graph.values()):
            raise ValueError("Every vertex link must be a cycle")
        seen, pending = set(), [next(iter(graph))]
        while pending:
            point = pending.pop()
            if point not in seen:
                seen.add(point); pending.extend(graph[point])
        if len(seen) != len(graph):
            raise ValueError("Disconnected vertex link is non-manifold")
    labels = np.full(len(v), -1, dtype=np.int64)
    component = 0
    for vertex in active:
        if labels[vertex] != -1:
            continue
        labels[vertex] = component
        pending = [int(vertex)]
        while pending:
            point = pending.pop()
            for neighbor in adjacency[point]:
                if labels[neighbor] == -1:
                    labels[neighbor] = component; pending.append(neighbor)
        component += 1
    signatures = []
    for index in range(component):
        vf = np.flatnonzero(labels == index)
        ff = f[labels[f[:, 0]] == index]
        ef = edges[labels[edges[:, 0]] == index]
        # Translation-invariant volume evaluation improves conditioning.
        x = v[ff] - np.mean(v[vf], axis=0)
        volume = float(np.einsum('ij,ij->i', x[:, 0], np.cross(x[:, 1], x[:, 2])).sum() / 6.)
        if not np.isfinite(volume) or abs(volume) <= np.finfo(v.dtype).eps * extent ** 3 * 32:
            raise ValueError("Each closed component must have nonzero signed volume")
        signatures.append({"euler": int(len(vf) - len(ef) + len(ff)), "volume_sign": 1 if volume > 0 else -1,
                           "signed_volume": volume, "vertices": int(len(vf)), "faces": int(len(ff))})
    signatures.sort(key=lambda row: (row['euler'], row['volume_sign'], abs(row['signed_volume'])))
    if sum(row['signed_volume'] for row in signatures) <= 0:
        raise ValueError("Complete oriented solid must have positive net volume")
    return {"vertices": len(v), "active_vertices": len(active), "faces": len(f),
            "components": signatures, "closed_oriented_vertex_manifold": True,
            "diagonal": extent}


def write_obj(path, vertices, faces):
    exact_mesh_topology(vertices, faces)
    with Path(path).open('x') as stream:
        for row in vertices:stream.write('v '+' '.join(format(float(x), '.17g') for x in row)+'\n')
        for row in faces:stream.write('f '+' '.join(str(int(x)+1) for x in row)+'\n')


def read_obj(path):
    vertices, faces = [], []
    for line in Path(path).read_text().splitlines():
        parts=line.split()
        if not parts or parts[0].startswith('#'):continue
        if len(parts)!=4 or parts[0] not in ('v','f'):raise ValueError('Require pure triangular OBJ, no implicit transforms')
        if parts[0]=='v':vertices.append([float(x) for x in parts[1:]])
        else:
            if any(not re.fullmatch('[1-9][0-9]*',x) for x in parts[1:]):raise ValueError('OBJ faces require positive bare indices')
            faces.append([int(x)-1 for x in parts[1:]])
    v,f=np.asarray(vertices,np.float64),np.asarray(faces,np.int64)
    exact_mesh_topology(v,f)
    return v,f


def mapped_geometry(source,candidate,mapping,diagnostics=None):
    """Birthface provenance matches actual shells, never volume-sorted pairing."""
    sv,sf=source;cv,cf=candidate;before=exact_mesh_topology(sv,sf);after=exact_mesh_topology(cv,cf)
    expected={'source_vertices':len(sv),'source_faces':len(sf),'output_vertices':len(cv),'output_faces':len(cf),
              'target_reached':True,'mapping_complete':True}
    if any(type(mapping.get(k)) is not type(v) or mapping[k]!=v for k,v in expected.items()):raise ValueError('Native birth mapping counts/target mismatch')
    J,I=(np.asarray(mapping.get(key)) for key in ('J','I'))
    if (J.shape!=(len(cf),) or I.shape!=(len(cv),) or J.dtype.kind not in 'iu' or I.dtype.kind not in 'iu'
            or np.any(J<0) or np.any(J>=len(sf)) or np.any(I<0) or np.any(I>=len(sv))):
        raise ValueError('Native birth maps invalid')
    if len(cv)>4096 or len(cf)>4096:raise ValueError('Native fixed budget not met')
    sl,cl=component_labels(sv,sf),component_labels(cv,cf);matched=[];seen=set()
    for label in np.unique(cl):
        selected=cl==label;birth=np.unique(sl[J[selected]])
        if len(birth)!=1 or int(birth[0]) in seen:raise ValueError('Native QEM merged or split an original shell')
        source_label=int(birth[0]);seen.add(source_label)
        # Inward shells are valid but mesh_topology requires positive whole
        # volume. Count/sign/Euler directly per shell; full meshes validated above.
        def shell(v,f):
            ids=np.unique(f);edges=np.unique(np.sort(np.concatenate([f[:,[0,1]],f[:,[1,2]],f[:,[2,0]]]),axis=1),axis=0)
            t=v[f]-v[ids].mean(0);volume=float(np.einsum('ij,ij->i',t[:,0],np.cross(t[:,1],t[:,2])).sum()/6)
            return len(ids)-len(edges)+len(f),volume
        ae,av=shell(sv,sf[sl==source_label]);be,bv=shell(cv,cf[selected])
        if ae!=be or np.sign(av)!=np.sign(bv) or not av or not bv:raise ValueError('Original shell Euler/orientation changed')
        # Vertex birth provenance must remain in its corresponding source shell.
        active=np.unique(cf[selected]);source_vertices=np.unique(sf[sl==source_label])
        if not np.isin(I[active],source_vertices).all():raise ValueError('Native vertex birth crosses source shells')
        error=abs(bv-av)/abs(av)
        matched.append({'source_component':source_label,'euler':int(ae),'volume_sign':1 if av>0 else -1,
                        'source_volume':av,'candidate_volume':bv,'relative_volume_error':error})
    if seen!=set(map(int,np.unique(sl))):raise ValueError('Native QEM removed an original shell')
    p,q=_surface(*source),_surface(*candidate)
    cd=float(.5*(cKDTree(p).query(q)[0].mean()+cKDTree(q).query(p)[0].mean())/before['diagonal'])
    av=sum(x['source_volume'] for x in matched);bv=sum(x['candidate_volume'] for x in matched);net=abs(bv-av)/av
    result={'sampled_bidirectional_chamfer_diagonal_ratio':cd,'net_volume_relative_error':net,'birthface_matched_shells':matched,
            'source_topology':before,'candidate_topology':after,'scale_or_pose_fitted':False}
    if diagnostics is not None:diagnostics.update(result)
    if cd>.01 or net>.05 or any(x['relative_volume_error']>.05 for x in matched):raise ValueError('Frozen1% geometry/5% net and per-shell volume gates failed')
    return result


def verify_pack_fidelity(exported, packed_vertices, packed_faces):
    """Prove official process/weld only reordered equivalent exported triangles.

    Padding must be zero faces and repetitions of the first vertex. A meaningful
    face deleted, nonexact positional merge, flip or second simplification fails.
    """
    v, f = np.asarray(packed_vertices), np.asarray(packed_faces)
    if (v.shape != (4096, 3) or f.shape != (4096, 3) or v.dtype.kind != 'f'
            or f.dtype.kind not in 'iu' or not np.isfinite(v).all() or np.min(f) < 0 or np.max(f) >= len(v)):
        raise ValueError('Official fixed 4096 arrays are invalid')
    zero = np.all(f == 0, axis=1)
    active = f[~zero]
    if not np.array_equal(_triangle_rows(*exported), _triangle_rows(v, active)):
        raise ValueError('Official export/weld altered meaningful oriented triangle geometry')
    used = np.unique(active)
    unused = np.setdiff1d(np.arange(len(v)), used)
    if len(unused) and not np.array_equal(v[unused], np.repeat(v[:1], len(unused), axis=0)):
        raise ValueError('Nonpadding unused vertices in official output')
    compact, faces, _ = exact_weld(v[used], np.searchsorted(used, active))
    exact_mesh_topology(compact, faces)
    return (compact, faces), {'active_faces': len(active), 'padding_faces': int(zero.sum()),
        'padding_vertices': len(unused), 'oriented_triangles_exact': True,
        'official_helper_simplification_invoked': False, 'nonexact_merge_or_face_deletion': False}


def true_hollow_containment(vertices, faces, *, self_intersecting_faces, diagnostics=None):
    """Two closed, nonintersecting shells; all inward vertices wind inside outer.

    Query the positive component ALONE: odd parity of combined nested shells
    would wrongly mark the actual cavity interior outside. Winding error and
    sampled boundary distance never replace global pair self-intersections.
    """
    if type(self_intersecting_faces) is not int or self_intersecting_faces != 0:
        raise ValueError('Containment requires zero pair self-intersections')
    exact_mesh_topology(vertices, faces)
    components = _components(vertices, faces)
    positive = [c for c in components if c[2] > 0]
    negative = [c for c in components if c[2] < 0]
    if len(components) != 2 or len(positive) != 1 or len(negative) != 1:
        raise ValueError('Hollow control requires one outward and one inward shell')
    ids, _, _ = negative[0]
    winding, distance = solid_winding_and_distances(np.asarray(vertices)[ids], positive[0][1])
    diagonal = float(np.linalg.norm(np.ptp(vertices, axis=0)))
    margin = 64 * np.finfo(np.float64).eps * diagonal
    errors = np.abs(winding - 1.)
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(winding_max_error=float(errors.max()), winding_atol=WINDING_ATOL,
                       inner_vertex_count=len(ids), boundary_margin=margin,
                       min_inner_vertex_to_outer_surface_distance=float(distance.min()),
                       min_surface_shell_separation=None,
                       continuous_surface_separation_computed=False,
                       vertex_distance_is_upper_bound_on_surface_separation=True,
                       global_pair_self_intersections=0,
                       true_containment_verified=False)
    if np.max(errors) > WINDING_ATOL or np.min(distance) <= margin:
        raise ValueError('Inner shell escapes or touches the outer boundary, or winding is numerically ambiguous')
    diagnostics['true_containment_verified'] = True
    return diagnostics


