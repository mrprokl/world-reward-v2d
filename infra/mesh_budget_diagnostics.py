"""Remote-only topology diagnostics before any physical/contact claim."""

import json
import os
from pathlib import Path
import platform
import sys

from world_reward.data import sha256


def main():
    if platform.system() != "Linux":
        raise RuntimeError("Dense geometry stays on Azure")
    import numpy as np
    import trimesh
    root = Path(os.environ["WR_ROOT"])
    sys.path.insert(0, str(root / "vendor/v2d_submission_kit"))
    from v2dlb.mesh_budget import budget_mesh
    path = root / "outputs/episode_000015/object_grounded/object.glb"
    original = trimesh.load(path, force="mesh")
    vertices, faces = budget_mesh(str(path), faces=4096, vertices=4096)
    active = (faces[:, 0] != faces[:, 1]) & (faces[:, 0] != faces[:, 2]) & (faces[:, 1] != faces[:, 2])
    meshes = {"original": original, "budget_unprocessed": trimesh.Trimesh(vertices, faces[active], process=False),
              "budget_processed": trimesh.Trimesh(vertices, faces[active], process=True)}
    result = {"stage": "mesh_budget_topology_diagnostics", "object_sha256": sha256(path),
              "meshes": {}, "ground_truth_used": False, "challenge_performance_verified": False}
    for name, mesh in meshes.items():
        incidence = np.bincount(mesh.edges_unique_inverse)
        components = mesh.split(only_watertight=False)
        result["meshes"][name] = {"vertices": len(mesh.vertices), "faces": len(mesh.faces),
                                  "watertight": bool(mesh.is_watertight), "winding_consistent": bool(mesh.is_winding_consistent),
                                  "signed_volume": float(mesh.volume), "extents": mesh.extents.tolist(),
                                  "boundary_edges": int((incidence == 1).sum()), "nonmanifold_edges": int((incidence > 2).sum()),
                                  "components": len(components), "component_volumes": [float(m.volume) for m in components[:10]]}
    small_components = meshes["budget_unprocessed"].split(only_watertight=False)
    result["small_budget_components"] = []
    for component in small_components:
        if len(component.faces) <= 10:
            result["small_budget_components"].append({"vertices": component.vertices.tolist(),
                                                      "faces": component.faces.tolist(),
                                                      "triangle_areas": component.area_faces.tolist()})
    target = root / "results/mesh-budget-diagnostics-detail.json"
    with target.open("x") as handle:
        handle.write(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
