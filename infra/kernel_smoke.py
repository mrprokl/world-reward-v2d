"""Small actual CUDA/EGL kernel gates inside a remote inference image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import platform
import time


def main() -> None:
    if platform.system() != "Linux":
        raise RuntimeError("CUDA validation is Azure-only")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--require-egl", action="store_true",
                        help="Require optional pyrender visualization, not a core reconstruction dependency")
    args = parser.parse_args()
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    records = {}

    def gate(name, function):
        started = time.perf_counter()
        try:
            function()
            torch.cuda.synchronize()
            records[name] = {"status": "pass", "seconds": time.perf_counter() - started}
        except Exception as exc:
            records[name] = {"status": "fail", "error_type": type(exc).__name__, "error": str(exc)}

    def pytorch3d():
        from pytorch3d.ops import knn_points
        p = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device="cuda")
        out = knn_points(p, p, K=1)
        if not torch.equal(out.idx, torch.tensor([[[0], [1]]], device="cuda")) or not torch.equal(out.dists, torch.zeros_like(out.dists)):
            raise RuntimeError("CUDA KNN returned incorrect tiny reference")

    def kaolin():
        from kaolin.metrics.pointcloud import chamfer_distance
        p = torch.tensor([[[0., 0., 0.], [1., 0., 0.]]], device="cuda")
        result = chamfer_distance(p, p)
        if not torch.isfinite(result).all() or not torch.equal(result, torch.zeros_like(result)):
            raise RuntimeError("Kaolin CUDA Chamfer tiny reference failed")

    def nvdiffrast():
        import nvdiffrast.torch as dr
        context = dr.RasterizeCudaContext()
        v = torch.tensor([[[-.5, -.5, .1, 1.], [.5, -.5, .1, 1.], [0., .5, .1, 1.]]], device="cuda")
        f = torch.tensor([[0, 1, 2]], device="cuda", dtype=torch.int32)
        raster, _ = dr.rasterize(context, v, f, resolution=[32, 32])
        if not torch.isfinite(raster).all() or not (raster[..., 3] > 0).any():
            raise RuntimeError("Nvdiffrast CUDA raster produced no triangle")

    def egl():
        import numpy as np
        import pyrender
        import trimesh
        scene = pyrender.Scene()
        scene.add(pyrender.Mesh.from_trimesh(trimesh.creation.box()))
        pose = np.eye(4)
        pose[2, 3] = 3.
        scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3), pose=pose)
        renderer = pyrender.OffscreenRenderer(32, 32)
        try:
            _, depth = renderer.render(scene)
            if not np.isfinite(depth).all() or not (depth > 0).any():
                raise RuntimeError("EGL renderer produced no box depth")
        finally:
            renderer.delete()

    def flash_attention():
        from flash_attn import flash_attn_func
        q = torch.zeros((1, 8, 2, 64), device="cuda", dtype=torch.float16)
        v = torch.ones_like(q)
        out = flash_attn_func(q, q, v, dropout_p=0.)
        if not torch.allclose(out, v, atol=1e-3, rtol=0):
            raise RuntimeError("FlashAttention CUDA tiny reference failed")

    for name, function in (("pytorch3d_knn", pytorch3d), ("kaolin_chamfer", kaolin),
                           ("nvdiffrast_raster", nvdiffrast), ("egl_depth", egl)):
        gate(name, function)
    if args.flash_attention:
        gate("flash_attention", flash_attention)
    required = [name for name in records if name != "egl_depth" or args.require_egl]
    result = {"stage": "actual_runtime_kernels", "torch": torch.__version__,
              "gpu": torch.cuda.get_device_name(), "gates": records,
              "required_gates": required,
              "optional_egl_reason": "Official minimal Objects and CARI paths use PyTorch3D/CUDA raster; pyrender is visualization only",
              "status": "pass" if all(records[name]["status"] == "pass" for name in required) else "fail"}
    args.report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if result["status"] != "pass":
        raise RuntimeError("One or more actual runtime kernels failed; inspect scalar report")


if __name__ == "__main__":
    main()
