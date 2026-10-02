"""Fixed-budget simplification which rejects topology or cavity-orientation loss.

This is geometry regeneration, not deletion of simplification artifacts. No
hole filling, normal inversion, component removal or metric-driven shrinking.
Optional trimesh/fast_simplification imports are lazy and run only remotely.
"""

from __future__ import annotations

from numbers import Integral

import numpy as np


def _budget(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 4:
        raise ValueError(f"{name} must be an integer >= 4")
    return int(value)


def _inspect(mesh):
    vertices, faces = np.asarray(mesh.vertices), np.asarray(mesh.faces)
    if (vertices.ndim != 2 or vertices.shape[1] != 3 or len(vertices) < 4
            or vertices.dtype.kind not in "iuf" or not np.isfinite(vertices).all()
            or faces.ndim != 2 or faces.shape[1] != 3 or len(faces) < 4
            or faces.dtype.kind not in "iu" or (faces < 0).any() or (faces >= len(vertices)).any()):
        raise ValueError("Invalid finite triangular mesh arrays")
    triangles = vertices[faces]
    if np.any(np.all(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]) == 0, axis=1)):
        raise ValueError("Candidate contains collapsed triangles; regenerate instead of deleting faces")
    incidence = np.bincount(mesh.edges_unique_inverse)
    if not len(incidence) or not np.all(incidence == 2):
        raise ValueError("Candidate has boundary or nonmanifold edges")
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError("Candidate is not closed with consistent winding")
    volume = float(mesh.volume)
    if not np.isfinite(volume) or volume <= 0:
        raise ValueError("Whole closed mesh must have finite positive signed volume")
    # Trimesh split defaults repair=True, which can fill holes. Explicitly
    # disable both repair and component filtering at every validation step.
    components = list(mesh.split(only_watertight=False, repair=False))
    if not components or sum(len(component.faces) for component in components) != len(faces):
        raise ValueError("Splitting lost source triangles")
    stats = []
    for component in components:
        counts = np.bincount(component.edges_unique_inverse)
        component_volume = float(component.volume)
        if (len(component.faces) < 4 or not len(counts) or not np.all(counts == 2)
                or not component.is_watertight or not component.is_winding_consistent
                or not np.isfinite(component_volume) or component_volume == 0):
            raise ValueError("A component is open, nonmanifold, inconsistently wound or zero-volume")
        stats.append({"faces": len(component.faces), "vertices": len(component.vertices),
                      "signed_volume": component_volume, "volume_sign": 1 if component_volume > 0 else -1,
                      "euler_characteristic": int(component.euler_number)})
    return components, {"vertices": len(vertices), "faces": len(faces), "signed_volume": volume,
                        "components": stats, "boundary_edges": 0, "nonmanifold_edges": 0,
                        "closed": True, "winding_consistent": True}


def _signature(stats):
    return sorted((component["volume_sign"], component["euler_characteristic"])
                  for component in stats["components"])


def _allocations(face_counts, target):
    """Even proportional allocation with >=4 faces per retained component."""
    counts = np.asarray(face_counts, dtype=np.int64)
    if target < 4 * len(counts):
        raise ValueError("Target cannot retain at least four faces per component")
    # Closed triangular manifolds have even face counts. Do not round upwards
    # past either the requested total or each component's original count.
    capacities = counts // 2
    units = min(target // 2, int(capacities.sum()))
    quotas = units * counts / counts.sum()
    allocation = np.minimum(capacities, np.maximum(2, np.floor(quotas).astype(np.int64)))
    while int(allocation.sum()) > units:
        choices = np.flatnonzero(allocation > 2)
        if not len(choices):
            raise ValueError("Insufficient face budget for retained components")
        index = max(choices.tolist(), key=lambda i: (allocation[i] - quotas[i], -i))
        allocation[index] -= 1
    while int(allocation.sum()) < units:
        choices = np.flatnonzero(allocation < capacities)
        if not len(choices):
            break
        index = max(choices.tolist(), key=lambda i: (quotas[i] - allocation[i], -i))
        allocation[index] += 1
    return (allocation * 2).tolist()


def fit_topology_preserving_budget(mesh, face_budget=4096, vertex_budget=4096):
    """Return first validated (mesh, diagnostics), or fail all fixed attempts.

    First try global targets [4096,4080,4000,3840] at the default budget; custom
    budgets use the same fixed ratios, rounded down to even face counts. Then
    retry those targets componentwise, allocated in proportion to original face
    counts with minimum four faces per component. Source components are never
    filtered, inverted or filled. Both modes start afresh from original arrays.

    Candidates must fit both budgets, retain closed consistently wound manifold
    edges, connected-component count, signed-orientation/Euler signature, and
    positive total volume. Componentwise candidates additionally retain each
    original component's own sign and Euler characteristic. No error is repaired
    by deleting positive-area triangles. Physical/shape accuracy and cavity
    containment are not established by these necessary topology checks.
    """
    face_budget, vertex_budget = _budget(face_budget, "face_budget"), _budget(vertex_budget, "vertex_budget")
    import trimesh

    if not isinstance(mesh, trimesh.Trimesh):
        raise ValueError("A preloaded canonical trimesh.Trimesh is required, not a Scene")
    components, original = _inspect(mesh)
    reference_signature = _signature(original)
    targets = list(dict.fromkeys(max(4, int(face_budget * numerator // 4096) // 2 * 2)
                                 for numerator in (4096, 4080, 4000, 3840)))
    diagnostic = {"schema": "world-reward-topology-preserving-budget-v1",
                  "face_budget": face_budget, "vertex_budget": vertex_budget, "target_faces": targets,
                  "original": original, "attempts": [], "input_modified": False,
                  "holes_filled": False, "components_removed": False, "normals_inverted": False,
                  "meaningful_triangles_deleted_to_repair": False, "shape_accuracy_verified": False,
                  "cavity_containment_verified": False}
    if original["faces"] <= face_budget and original["vertices"] <= vertex_budget:
        diagnostic.update(selected_mode="unchanged", selected_target=original["faces"], selected=original)
        return mesh.copy(), diagnostic
    import fast_simplification

    def simplified(source, target):
        if len(source.faces) <= target:
            return source.copy()
        vertices, faces = fast_simplification.simplify(
            np.array(source.vertices, dtype=np.float64, copy=True),
            np.array(source.faces, dtype=np.int64, copy=True), target_count=int(target),
        )
        return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)

    for mode in ("global", "componentwise"):
        for target in targets:
            attempt = {"mode": mode, "target_faces": target}
            try:
                if mode == "global":
                    candidate = simplified(mesh, target)
                else:
                    allocation = _allocations([len(component.faces) for component in components], target)
                    attempt["component_target_faces"] = allocation
                    candidates = []
                    for component, component_target, reference in zip(components, allocation, original["components"], strict=True):
                        reduced = simplified(component, component_target)
                        # A negative cavity cannot pass _inspect by itself (it
                        # rightly requires positive whole volume). Validate it
                        # in a combined mesh below; check its own sign/Euler now.
                        if (np.sign(reduced.volume) != reference["volume_sign"]
                                or int(reduced.euler_number) != reference["euler_characteristic"]):
                            raise ValueError("Simplification changed an original component orientation or Euler characteristic")
                        candidates.append(reduced)
                    # Concatenation uses original component order and no welding,
                    # topology repair or source transforms.
                    candidate = trimesh.util.concatenate(candidates)
                _, stats = _inspect(candidate)
                if stats["faces"] > target or stats["faces"] > face_budget or stats["vertices"] > vertex_budget:
                    raise ValueError("Simplifier did not meet face/vertex budgets")
                if len(stats["components"]) != len(original["components"]) or _signature(stats) != reference_signature:
                    raise ValueError("Simplification changed component count, signed orientation or Euler characteristic")
                attempt.update(status="pass", result=stats)
                diagnostic["attempts"].append(attempt)
                diagnostic.update(selected_mode=mode, selected_target=target, selected=stats)
                return candidate, diagnostic
            except (ValueError, RuntimeError) as exc:
                attempt.update(status="fail", reason=str(exc))
                diagnostic["attempts"].append(attempt)
    reasons = "; ".join(f"{attempt['mode']}:{attempt['target_faces']}:{attempt['reason']}"
                        for attempt in diagnostic["attempts"])
    raise ValueError(f"No topology-preserving approximation fits both budgets; {reasons}")
