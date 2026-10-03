"""Conservative full-surface certificates for own private procedural geometry.

No mesh repair, sampling-based collision test, actor rescaling or biomechanics.
Inputs must be outward, closed, embedded two-manifolds. All original faces are
retained. Float64 predicates fail closed within a declared metric tolerance;
exact touching is NOT certified as a nonpenetrating grasp. A positive gap up to
``proximity_m`` is merely surface proximity, never force closure or true contact.
"""
from __future__ import annotations

import hashlib
import math
import time

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


def _positive(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive scalar")
    return float(value)


def _hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256(str(value.dtype).encode() + repr(value.shape).encode())
    digest.update(value.tobytes())
    return digest.hexdigest()


def validate_closed_mesh(vertices, faces, *, tolerance_m=1e-8):
    """Validate every original vertex/face; closure does not prove embedding.

    Multiple disjoint outward components are allowed, but pinched vertex links,
    inward shells, tiny/ill-conditioned triangles and unused vertices fail.
    Self intersections and nested component containment are audited separately.
    """
    tolerance = _positive(tolerance_m, "tolerance_m")
    if np.ma.isMaskedArray(vertices) or np.ma.isMaskedArray(faces):
        raise ValueError("Masked geometry forbidden")
    original_v, original_f = np.asarray(vertices), np.asarray(faces)
    if (original_v.dtype not in (np.dtype("float32"), np.dtype("float64"))
            or original_v.ndim != 2 or original_v.shape[1:] != (3,) or len(original_v) < 4
            or not np.isfinite(original_v).all()):
        raise ValueError("Finite nonempty float32/64 vertices required")
    if (original_f.dtype.kind not in "iu" or original_f.ndim != 2 or original_f.shape[1:] != (3,)
            or len(original_f) < 4 or np.any(original_f < 0) or np.any(original_f >= len(original_v))):
        raise ValueError("Complete integer indexed triangle mesh required")
    v, f = original_v.astype(np.float64, copy=True), original_f.astype(np.int64, copy=True)
    if not np.array_equal(np.unique(f), np.arange(len(v))):
        raise ValueError("All original vertices must be covered; no deletion/repair")
    if np.any(np.diff(np.sort(f, axis=1), axis=1) == 0):
        raise ValueError("Repeated-index triangle forbidden")
    if len(np.unique(np.sort(f, axis=1), axis=0)) != len(f):
        raise ValueError("Duplicate original triangle forbidden")
    precision = 128 * np.finfo(np.float64).eps * max(1., float(np.abs(v).max()))
    if precision > tolerance / 8:
        raise ValueError("Coordinates insufficiently conditioned for declared tolerance")
    triangles = v[f]
    edge = np.roll(triangles, -1, axis=1) - triangles
    lengths = np.linalg.norm(edge, axis=-1)
    normals = np.cross(edge[:, 0], -edge[:, 2])
    norm = np.linalg.norm(normals, axis=1)
    altitude = norm / lengths.max(axis=1)
    if (not np.isfinite(lengths).all() or not np.isfinite(norm).all()
            or np.any(lengths <= 8 * tolerance) or np.any(altitude <= 8 * tolerance)):
        raise ValueError("Degenerate/ill-conditioned triangles forbidden; no epsilon repair")
    edges = np.concatenate((f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]))
    _, inverse, counts = np.unique(np.sort(edges, axis=1), axis=0, return_inverse=True, return_counts=True)
    directions = np.where(edges[:, 0] < edges[:, 1], 1, -1)
    if np.any(counts != 2) or np.any(np.bincount(inverse, weights=directions) != 0):
        raise ValueError("Every edge needs exactly two opposite oriented incidences")
    order = np.argsort(inverse, kind="stable")
    incident = np.tile(np.arange(len(f)), 3)[order].reshape(-1, 2)
    graph = coo_matrix((np.ones(2 * len(incident), np.uint8),
        (np.r_[incident[:, 0], incident[:, 1]], np.r_[incident[:, 1], incident[:, 0]])), shape=(len(f), len(f)))
    count, labels = connected_components(graph, directed=False)
    # The link of every vertex must be one connected cycle, not two pinched fans.
    flat = f.ravel()
    order = np.argsort(flat, kind="stable")
    cuts = np.r_[0, np.flatnonzero(np.diff(flat[order])) + 1, len(flat)]
    for lo, hi in zip(cuts[:-1], cuts[1:]):
        vertex = flat[order[lo]]
        links = f[order[lo:hi] // 3]
        adjacency = {}
        for row in links:
            first, last = row[row != vertex]
            adjacency.setdefault(int(first), set()).add(int(last))
            adjacency.setdefault(int(last), set()).add(int(first))
        if any(len(neighbors) != 2 for neighbors in adjacency.values()):
            raise ValueError("Vertex link is not a closed manifold cycle")
        seen, pending = set(), [next(iter(adjacency))]
        while pending:
            item = pending.pop()
            if item not in seen:
                seen.add(item); pending.extend(adjacency[item] - seen)
        if len(seen) != len(adjacency):
            raise ValueError("Pinched/nonmanifold vertex fan forbidden")
    components, volumes = [], []
    for component in range(count):
        ids = np.flatnonzero(labels == component)
        vertex_ids = np.unique(f[ids])
        local = triangles[ids] - v[vertex_ids].mean(axis=0)
        volume = float(np.einsum("ij,ij->i", local[:, 0], np.cross(local[:, 1], local[:, 2])).sum() / 6)
        if not math.isfinite(volume) or volume <= tolerance * norm[ids].sum():
            raise ValueError("Each component must have well-conditioned positive outward volume")
        components.append(ids); volumes.append(volume)
    return dict(vertices=v, faces=f, triangles=triangles, normals=normals / norm[:, None],
        components=components, face_components=labels,
        report=dict(vertices=len(v), faces=len(f), components=count, signed_volumes_m3=volumes,
            original_vertices_sha256=_hash(original_v), original_faces_sha256=_hash(original_f),
            original_face_coverage=len(f), topology_closed_oriented=True, embedding_verified=False))


def _check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError("Cross-surface whole-stage deadline exceeded; no certificate")


def _candidate_pairs(left, right, margin, limit, deadline, *, same=False):
    """Complete sphere/AABB broadphase; bounded lists, not dense Fleft*Fright."""
    a, b = left["triangles"], right["triangles"]
    centers = a.mean(axis=1)
    radii = np.linalg.norm(a - centers[:, None], axis=-1).max(axis=1)
    tree = cKDTree(centers)
    amin, amax = a.min(axis=1), a.max(axis=1)
    total, maximum_radius = 0, radii.max()
    for index, triangle in enumerate(b):
        _check_deadline(deadline)
        center = triangle.mean(axis=0)
        radius = np.linalg.norm(triangle - center, axis=1).max() + maximum_radius + margin
        selected = np.asarray(tree.query_ball_point(center, radius), np.int64)
        if same:
            selected = selected[selected < index]
        if not len(selected):
            continue
        overlap = np.all(amax[selected] + margin >= triangle.min(axis=0), axis=1)
        overlap &= np.all(amin[selected] - margin <= triangle.max(axis=0), axis=1)
        selected = selected[overlap]
        total += len(selected)
        if total > limit:
            raise RuntimeError("Cross-surface candidate-pair budget exceeded; no certificate")
        for other in selected:
            yield int(other), index


def _point_triangle(point, triangle):
    """Continuous closest point on a nondegenerate closed triangle."""
    a, b, c = triangle
    normal = np.cross(b - a, c - a)
    projection = point - normal * (np.dot(point - a, normal) / np.dot(normal, normal))
    tests = [np.dot(np.cross(end - start, projection - start), normal)
        for start, end in ((a, b), (b, c), (c, a))]
    candidates = [projection] if min(tests) >= 0 else []
    for first, last in ((a, b), (b, c), (c, a)):
        edge = last - first
        parameter = float(np.clip(np.dot(point - first, edge) / np.dot(edge, edge), 0., 1.))
        candidates.append(first + parameter * edge)
    distances = [float(np.dot(point - candidate, point - candidate)) for candidate in candidates]
    chosen = int(np.argmin(distances))
    return distances[chosen], candidates[chosen]


def _segment_distance(a, b, c, d):
    """All endpoint/segment regions and the interior line-line solution."""
    u, v, w = b - a, d - c, a - c
    uu, vv, uv = np.dot(u, u), np.dot(v, v), np.dot(u, v)
    values = []
    for point, start, edge, length in ((a, c, v, vv), (b, c, v, vv), (c, a, u, uu), (d, a, u, uu)):
        parameter = float(np.clip(np.dot(point - start, edge) / length, 0., 1.))
        delta = point - start - parameter * edge
        values.append(float(np.dot(delta, delta)))
    # Cross-product formulation avoids catastrophic uu*vv-uv**2 cancellation
    # for almost parallel segments with a valid interior/interior closest pair.
    u_long, v_long, w_long = u.astype(np.longdouble), v.astype(np.longdouble), w.astype(np.longdouble)
    normal = np.cross(u_long, v_long)
    determinant = np.dot(normal, normal)
    if determinant > 0:
        s = float(np.dot(np.cross(v_long, w_long), normal) / determinant)
        t = float(np.dot(np.cross(u_long, w_long), normal) / determinant)
        if 0 <= s <= 1 and 0 <= t <= 1:
            delta = w + s * u - t * v
            values.append(float(np.dot(delta, delta)))
    return min(values)


def _triangle_distance(a, b):
    values = [_point_triangle(point, b)[0] for point in a]
    values += [_point_triangle(point, a)[0] for point in b]
    values += [_segment_distance(a[i], a[(i + 1) % 3], b[j], b[(j + 1) % 3])
        for i in range(3) for j in range(3)]
    return math.sqrt(max(0., min(values)))


def _clip_to_plane(triangle, signed, tolerance):
    points = [triangle[i] for i in range(3) if abs(signed[i]) <= tolerance]
    for i in range(3):
        j = (i + 1) % 3
        if signed[i] < -tolerance and signed[j] > tolerance or signed[j] < -tolerance and signed[i] > tolerance:
            points.append(triangle[i] + signed[i] / (signed[i] - signed[j]) * (triangle[j] - triangle[i]))
    return points


def _coplanar_polygon(a, b, normal):
    """Sutherland-Hodgman in the best-conditioned plane, preserving 3D points."""
    drop = int(np.argmax(np.abs(normal)))
    keep = [i for i in range(3) if i != drop]
    projected = b[:, keep]
    def cross2(first, last):
        return first[0] * last[1] - first[1] * last[0]
    orientation = cross2(projected[1] - projected[0], projected[2] - projected[0])
    sign = 1. if orientation > 0 else -1.
    polygon = list(a)
    for i in range(3):
        first, last = projected[i], projected[(i + 1) % 3]
        edge = last - first
        def signed(point):
            return sign * float(cross2(edge, point[keep] - first))
        output = []
        if not polygon:
            break
        previous = polygon[-1]; previous_value = signed(previous)
        for point in polygon:
            value = signed(point)
            if (value >= 0) != (previous_value >= 0):
                output.append(previous + previous_value / (previous_value - value) * (point - previous))
            if value >= 0:
                output.append(point)
            previous, previous_value = point, value
        polygon = output
    if len(polygon) < 3:
        area = 0.
    else:
        p = np.asarray(polygon)[:, keep]
        area = abs(float(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1)))) / 2
    return polygon, area


def triangle_relation(a, b, *, tolerance_m=1e-8):
    """Full closed triangle intersection classification, including coplanarity.

    ``proper_intersection`` is a transverse interior/interior segment. All
    other surface intersections/tolerance contacts are distinct and still
    prohibit a positive certificate. Separated triangles include edge-edge
    interior minima, not only vertices sampled against the other surface.
    """
    tolerance = _positive(tolerance_m, "tolerance_m")
    if np.ma.isMaskedArray(a) or np.ma.isMaskedArray(b):
        raise ValueError("Masked triangles forbidden")
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    if a.shape != (3, 3) or b.shape != (3, 3) or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Two finite original triangles required")
    precision = 128 * np.finfo(np.float64).eps * max(1., float(np.abs(a).max()), float(np.abs(b).max()))
    if precision > tolerance / 8:
        raise ValueError("Triangle coordinates insufficiently conditioned for declared tolerance")
    origin = a[0].copy(); a, b = a - origin, b - origin
    ea, eb = np.roll(a, -1, axis=0) - a, np.roll(b, -1, axis=0) - b
    na, nb = np.cross(ea[0], -ea[2]), np.cross(eb[0], -eb[2])
    la, lb = np.linalg.norm(na), np.linalg.norm(nb)
    lengths_a, lengths_b = np.linalg.norm(ea, axis=1), np.linalg.norm(eb, axis=1)
    if (not math.isfinite(la) or not math.isfinite(lb) or not np.isfinite(lengths_a).all()
            or not np.isfinite(lengths_b).all() or la <= 8*tolerance*lengths_a.max()
            or lb <= 8*tolerance*lengths_b.max() or min(lengths_a.min(), lengths_b.min()) <= 8*tolerance):
        raise ValueError("Well-conditioned nondegenerate original triangles required")
    na, nb = na / la, nb / lb
    da, db = (a - b[0]) @ nb, b @ na
    direction = np.cross(na, nb); cross_length = np.linalg.norm(direction)
    axes = [na, nb]
    axes += [np.cross(first, last) for first in ea for last in eb]
    axes += [np.cross(na, edge) for edge in ea] + [np.cross(nb, edge) for edge in eb]
    separated = False
    for axis in axes:
        norm = np.linalg.norm(axis)
        if norm <= 64 * np.finfo(float).eps:
            continue
        x, y = a @ (axis / norm), b @ (axis / norm)
        if max(x.min() - y.max(), y.min() - x.max()) > tolerance:
            separated = True; break
    if separated:
        distance = _triangle_distance(a, b)
        return dict(kind="separated" if distance > tolerance else "ambiguous", distance_m=distance, points=[])
    if max(np.abs(da).max(), np.abs(db).max()) <= tolerance:
        polygon, area = _coplanar_polygon(a, b, na)
        if area > tolerance * tolerance:
            return dict(kind="coplanar_overlap", distance_m=0., points=[(p + origin).tolist() for p in polygon])
        distance = _triangle_distance(a, b)
        kind = "boundary_contact" if polygon and distance <= tolerance else "ambiguous"
        return dict(kind=kind, distance_m=distance, points=[(p + origin).tolist() for p in polygon])
    if cross_length <= 64 * np.finfo(float).eps:
        return dict(kind="ambiguous", distance_m=_triangle_distance(a, b), points=[])
    direction /= cross_length
    pa, pb = _clip_to_plane(a, da, tolerance), _clip_to_plane(b, db, tolerance)
    if not pa or not pb:
        distance = _triangle_distance(a, b)
        return dict(kind="ambiguous", distance_m=distance, points=[])
    sa, sb = np.asarray(pa) @ direction, np.asarray(pb) @ direction
    lo, hi = max(sa.min(), sb.min()), min(sa.max(), sb.max())
    if hi < lo - tolerance:
        distance = _triangle_distance(a, b)
        return dict(kind="ambiguous", distance_m=distance, points=[])
    base = np.asarray(pa)[np.argmin(sa)]
    points = [base + (value - sa.min()) * direction + origin for value in (lo, hi)]
    proper = da.min() < -tolerance and da.max() > tolerance and db.min() < -tolerance and db.max() > tolerance and hi - lo > tolerance
    return dict(kind="proper_intersection" if proper else "boundary_contact", distance_m=0., points=[p.tolist() for p in points])


def _shared_simplex_only(a, b, ia, ib, tolerance):
    common = set(map(int, ia)) & set(map(int, ib))
    if not len(common):
        return False
    shared = a[[i for i, vertex in enumerate(ia) if int(vertex) in common]]
    na, nb = np.cross(a[1] - a[0], a[2] - a[0]), np.cross(b[1] - b[0], b[2] - b[0])
    da, db = (a - b[0]) @ (nb / np.linalg.norm(nb)), (b - a[0]) @ (na / np.linalg.norm(na))
    if len(common) == 1:
        # A strict plane supporting exactly the shared vertex proves that no
        # other point of the triangles intersects; near-zero signs fail closed.
        x = da[[i for i, vertex in enumerate(ia) if int(vertex) not in common]]
        y = db[[i for i, vertex in enumerate(ib) if int(vertex) not in common]]
        if min(x) > tolerance or max(x) < -tolerance or min(y) > tolerance or max(y) < -tolerance:
            return True
        x = a[[i for i, vertex in enumerate(ia) if int(vertex) not in common]] - shared[0]
        y = b[[i for i, vertex in enumerate(ib) if int(vertex) not in common]] - shared[0]
        directions = [x.mean(axis=0) - y.mean(axis=0)]
        directions += [first / np.linalg.norm(first) - last / np.linalg.norm(last) for first in x for last in y]
        for direction in directions:
            norm = np.linalg.norm(direction)
            if norm > 0 and min(x @ (direction/norm)) > tolerance and max(y @ (direction/norm)) < -tolerance:
                return True
        return False  # Unproven adjacent geometry is never exempted by proximity.
    # Robust noncoplanar shared edge can intersect only on that original edge.
    if len(common) == 2:
        if np.max(np.abs(da)) > tolerance and np.max(np.abs(db)) > tolerance:
            return True
        # Opposite strict halfplanes around the shared edge prove that coplanar
        # faces meet only on that edge. No tolerance-sized overlap is excused.
        edge = shared[1] - shared[0]
        first = a[[i for i, vertex in enumerate(ia) if int(vertex) not in common]][0]
        last = b[[i for i, vertex in enumerate(ib) if int(vertex) not in common]][0]
        axis = np.cross(edge, np.cross(edge, first - shared[0]))
        axis /= np.linalg.norm(axis)
        return np.dot(first-shared[0], axis) < -tolerance and np.dot(last-shared[0], axis) > tolerance
    return False


def _winding(point, triangles, tolerance):
    delta = triangles - point
    length = np.linalg.norm(delta, axis=-1)
    if np.any(length <= tolerance):
        return None
    first, second, third = delta[:, 0], delta[:, 1], delta[:, 2]
    numerator = np.einsum("ij,ij->i", first, np.cross(second, third))
    denominator = np.prod(length, axis=1) + np.einsum("ij,ij->i", first, second) * length[:, 2]
    denominator += np.einsum("ij,ij->i", second, third) * length[:, 0]
    denominator += np.einsum("ij,ij->i", third, first) * length[:, 1]
    value = float(np.sum(2 * np.arctan2(numerator, denominator)) / (4 * np.pi))
    if not math.isfinite(value) or min(abs(value), abs(value - 1.)) > 1e-6:
        return None
    return value


def _embedding(mesh, tolerance, limit, deadline):
    failures, checked, shared = [], 0, 0
    triangles, faces = mesh["triangles"], mesh["faces"]
    for first, last in _candidate_pairs(mesh, mesh, tolerance, limit, deadline, same=True):
        checked += 1
        a, b = triangles[first], triangles[last]
        if _shared_simplex_only(a, b, faces[first], faces[last], tolerance):
            shared += 1; continue
        na, nb = mesh["normals"][first], mesh["normals"][last]
        da, db = (a - b[0]) @ nb, (b - a[0]) @ na
        if np.min(da) > tolerance or np.max(da) < -tolerance or np.min(db) > tolerance or np.max(db) < -tolerance:
            continue
        relation = triangle_relation(a, b, tolerance_m=tolerance)
        if relation["kind"] != "separated":
            failures.append(dict(faces=[first, last], kind=relation["kind"]))
    nested = []
    for index, component in enumerate(mesh["components"]):
        _check_deadline(deadline)
        other = np.flatnonzero(mesh["face_components"] != index)
        if not len(other):
            continue
        point = mesh["vertices"][mesh["faces"][component[0], 0]]
        winding = _winding(point, mesh["triangles"][other], tolerance)
        if winding is None or winding > .5:
            nested.append(index)
    return dict(verified=not failures and not nested, candidate_pairs=checked,
        permitted_shared_simplex_pairs=shared, forbidden_intersections=failures,
        nested_or_ambiguous_components=nested)


def audit_closed_surface(vertices, faces, *, tolerance_m=1e-8,
        max_candidate_pairs=2_000_000, budget_seconds=30.):
    """Independent full closed/embedded surface prerequisite before fitting."""
    started = time.monotonic()
    tolerance = _positive(tolerance_m, "tolerance_m")
    if type(max_candidate_pairs) is not int or max_candidate_pairs <= 0:
        raise ValueError("Positive integer candidate-pair budget required")
    deadline = started + _positive(budget_seconds, "budget_seconds")
    mesh = validate_closed_mesh(vertices, faces, tolerance_m=tolerance)
    embedding = _embedding(mesh, tolerance, max_candidate_pairs, deadline)
    _check_deadline(deadline)
    mesh["report"]["embedding_verified"] = embedding["verified"]
    return dict(schema="world-reward-own-closed-surface-certificate-v1",
        status="pass" if embedding["verified"] else "fail", mesh=mesh["report"], embedding=embedding,
        tolerance_m=tolerance, full_original_faces_retained=True, exact_arithmetic_proof=False,
        max_candidate_pairs=max_candidate_pairs, budget_seconds=budget_seconds,
        elapsed_seconds=time.monotonic() - started)


def audit_cross_surfaces(vertices_a, faces_a, vertices_b, faces_b, *, tolerance_m=1e-8,
        proximity_m=.002, max_candidate_pairs=2_000_000, budget_seconds=30.):
    """Audit full meshes, embeddings, every possible near/cross pair and solids.

    Invalid mesh topology raises ValueError; exceeded compute budgets raise and
    never produce PASS. A returned FAIL preserves complete diagnostics. Positive
    certificates require strictly separated surfaces (>tolerance) and no nested
    solids, even if gaps are within the declared 2mm proximity band.
    """
    started = time.monotonic()
    tolerance, proximity = _positive(tolerance_m, "tolerance_m"), _positive(proximity_m, "proximity_m")
    if proximity <= tolerance:
        raise ValueError("Proximity band must exceed conservative tolerance")
    if type(max_candidate_pairs) is not int or max_candidate_pairs <= 0:
        raise ValueError("Positive integer candidate-pair budget required")
    deadline = started + _positive(budget_seconds, "budget_seconds")
    a = validate_closed_mesh(vertices_a, faces_a, tolerance_m=tolerance)
    _check_deadline(deadline)
    b = validate_closed_mesh(vertices_b, faces_b, tolerance_m=tolerance)
    embeddings = [_embedding(mesh, tolerance, max_candidate_pairs, deadline) for mesh in (a, b)]
    kinds = {name: 0 for name in ("proper_intersection", "coplanar_overlap", "boundary_contact", "ambiguous")}
    interactions, checked, near, minimum = [], 0, 0, None
    for first, last in _candidate_pairs(a, b, proximity, max_candidate_pairs, deadline):
        checked += 1
        relation = triangle_relation(a["triangles"][first], b["triangles"][last], tolerance_m=tolerance)
        if relation["kind"] != "separated":
            kinds[relation["kind"]] += 1
            interactions.append(dict(faces=[first, last], kind=relation["kind"]))
        distance = relation["distance_m"]
        if distance <= proximity:
            minimum = distance if minimum is None else min(minimum, distance)
            if relation["kind"] == "separated":
                near += 1
    containment = []
    for mesh, other in ((a, b), (b, a)):
        result = []
        for component in mesh["components"]:
            _check_deadline(deadline)
            point = mesh["vertices"][mesh["faces"][component[0], 0]]
            winding = _winding(point, other["triangles"], tolerance)
            result.append(dict(winding=winding, state="ambiguous" if winding is None else "inside" if winding > .5 else "outside"))
        containment.append(result)
    certificate = all(row["verified"] for row in embeddings) and not interactions
    certificate &= all(item["state"] == "outside" for row in containment for item in row)
    for mesh, embedding in zip((a, b), embeddings):
        mesh["report"]["embedding_verified"] = embedding["verified"]
    _check_deadline(deadline)
    return dict(schema="world-reward-own-cross-surface-certificate-v1", status="pass" if certificate else "fail",
        nonpenetration_certified=bool(certificate), meshes=[a["report"], b["report"]],
        embedding=embeddings, intermesh_candidate_pairs=checked, intersection_counts=kinds,
        intersections=interactions, containment=containment, proximity_pairs=near,
        minimum_surface_distance_m=minimum, minimum_distance_lower_bound_m=proximity if minimum is None else minimum,
        minimum_distance_exact=minimum is not None,
        full_original_faces_retained=True, broadphase_complete=True, sampled_collision_test_used=False,
        tolerance_m=tolerance, proximity_m=proximity, max_candidate_pairs=max_candidate_pairs,
        touching_certified=False, force_closure_verified=False, biomechanics_verified=False,
        budget_seconds=budget_seconds, exact_arithmetic_proof=False, elapsed_seconds=time.monotonic() - started)
