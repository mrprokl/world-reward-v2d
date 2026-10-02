"""Azure-only differentiable silhouette capability, not reconstruction quality.

One closed procedural cuboid and a fixed synthetic translated alpha reference
exercise the actual PyTorch3D rasterizer/SoftSilhouetteShader. A single frozen
0.5mm negative-gradient translation must decrease the same full-grid MSE.
No model, challenge input, labels, adaptation, geometry scaling or alignment.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np

import camera_render as camera
from world_reward.data import sha256

STAGE = "own_synthetic_soft_silhouette_autograd_capability"
IMAGE = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
BASE = "results/soft-silhouette-probe-v1"
WIDTH, HEIGHT, SOURCE_WIDTH, SOURCE_HEIGHT, BUDGET = 256, 192, 1024, 768, 180
SIGMA, GAMMA, FACES_PER_PIXEL = 1e-4, 1e-4, 8
BLUR_RADIUS = math.log(1./1e-4 - 1.)*SIGMA
STEP_M, PROJECTION_ATOL_PX = .0005, 1e-4
TARGET_TRANSLATION_M = (.008, -.006, .014)


def scaled_intrinsics(matrix, source_width, source_height, width, height):
    """Scale edge-coordinate OpenCV K, preserving pixel-centre cell convention."""
    if np.ma.isMaskedArray(matrix): raise ValueError("No masked camera matrices")
    K = camera._intrinsics(matrix).copy()
    for value in (source_width, source_height, width, height):
        if type(value) is not int or value <= 0: raise ValueError("Positive integer image grids required")
    scaled = np.diag([width/source_width, height/source_height, 1.])@K
    camera._intrinsics(scaled)
    return scaled


def fixture():
    """Eight vertices, twelve outward triangles, camera metres, all Z positive."""
    centre = np.array([.13, -.08, 2.3], np.float64)
    extents = np.array([.62, .48, .38], np.float64)
    signs = np.array([[-1,-1,-1], [1,-1,-1], [1,1,-1], [-1,1,-1],
                      [-1,-1,1], [1,-1,1], [1,1,1], [-1,1,1]], np.float64)
    vertices = centre+signs*extents/2.
    faces = np.array([[0,2,1], [0,3,2], [4,5,6], [4,6,7], [0,1,5], [0,5,4],
                      [3,7,6], [3,6,2], [0,4,7], [0,7,3], [1,2,6], [1,6,5]], np.int64)
    source_K = np.array([[960., 0., SOURCE_WIDTH/2.], [0., 1040., SOURCE_HEIGHT/2.], [0., 0., 1.]], np.float64)
    K = scaled_intrinsics(source_K, SOURCE_WIDTH, SOURCE_HEIGHT, WIDTH, HEIGHT)
    vertices32, faces, _ = camera._mesh_inputs(vertices, faces, K, WIDTH, HEIGHT, 1e-4)
    return vertices32, faces, source_K, K


def negative_gradient_step(gradient):
    if (not isinstance(gradient, np.ndarray) or np.ma.isMaskedArray(gradient) or gradient.dtype.kind != "f"
            or gradient.shape != (3,) or not np.isfinite(gradient).all() or np.any(gradient == 0)):
        raise ValueError("All three translation derivatives must be finite and nonzero")
    norm = float(np.linalg.norm(gradient.astype(np.float64)))
    if not np.isfinite(norm) or norm <= 0: raise ValueError("Finite nonzero gradient norm required")
    return -STEP_M*gradient.astype(np.float64)/norm


def validate_measurement(before, after, gradient, step):
    expected = negative_gradient_step(gradient)
    if (not isinstance(step, np.ndarray) or step.shape != (3,) or not np.isfinite(step).all()
            or not np.allclose(step, expected, rtol=1e-6, atol=1e-10)
            or not np.isclose(np.linalg.norm(step), STEP_M, rtol=1e-6, atol=1e-10)):
        raise ValueError("Exactly one fixed normalized0.5mm negative-gradient step required")
    if (type(before) is not float or type(after) is not float or not np.isfinite([before, after]).all()
            or before <= 0 or after < 0 or not after < before):
        raise ValueError("One frozen step must strictly decrease the identical full-grid scalar loss")


def regular(path):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file(): raise ValueError("Regular non-symlink source required")
    return path


def strict_torch():
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8": raise ValueError("Configure CUBLAS before torch import")
    import random
    import torch
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4); torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False
    if not torch.cuda.is_available(): raise RuntimeError("Require actual Azure CUDA; no CPU/mock renderer fallback")
    return torch


def run(report, persist):
    vertices, faces, source_K, K = fixture()
    frozen = [(regular(Path(__file__)), report["script_sha256"]),
              (regular(Path(camera.__file__)), report["camera_helper_sha256"])]
    vertices_hash = hashlib.sha256(vertices.tobytes()).hexdigest()
    torch = strict_torch()
    import pytorch3d
    from pytorch3d.renderer import BlendParams, MeshRasterizer, RasterizationSettings, SoftSilhouetteShader
    from pytorch3d.structures import Meshes
    if pytorch3d.__version__ != "0.7.9": raise ValueError("Pinned PyTorch3D0.7.9 required")
    sources = {}
    for value in (MeshRasterizer, SoftSilhouetteShader, BlendParams):
        path = regular(Path(sys.modules[value.__module__].__file__))
        sources[value.__name__] = {"file": str(path), "sha256": sha256(path)}; frozen.append((path, sha256(path)))
    report.update(phase="projection_and_hard_parity", torch_version=str(torch.__version__),
        cuda_version=torch.version.cuda, pytorch3d_version=pytorch3d.__version__, renderer_sources=sources,
        camera_K_source=source_K.tolist(), camera_K_scaled=K.tolist(), vertices_sha256=vertices_hash,
        vertex_count=len(vertices), face_count=len(faces)); persist()
    base = torch.tensor(vertices, device="cuda", dtype=torch.float32)
    triangles = torch.tensor(faces, device="cuda", dtype=torch.int64)
    cameras = camera._opencv_camera(torch, K, WIDTH, HEIGHT)
    expected_pixels = camera.project_camera_points(vertices, K)
    screen = cameras.transform_points_screen(base[None])[0, :, :2].detach().cpu().numpy()
    projection_error = float(np.abs(screen-expected_pixels).max())
    expected_source = camera.project_camera_points(vertices, source_K)*[WIDTH/SOURCE_WIDTH, HEIGHT/SOURCE_HEIGHT]
    scaling_error = float(np.abs(expected_source-expected_pixels).max())
    report.update(projection_max_error_px=projection_error, grid_scaling_max_error_px=scaling_error); persist()
    if projection_error > PROJECTION_ATOL_PX or scaling_error > 1e-10:
        raise ValueError("Soft and hard camera projection/pixel-centre scaling differs")
    hard_settings = RasterizationSettings(image_size=(HEIGHT, WIDTH), blur_radius=0., faces_per_pixel=1,
        perspective_correct=True, clip_barycentric_coords=False, cull_backfaces=False,
        cull_to_frustum=False, z_clip_value=None, max_faces_per_bin=len(faces))
    own_hard = MeshRasterizer(cameras=cameras, raster_settings=hard_settings)(Meshes(verts=[base], faces=[triangles]))
    hard_mask = own_hard.pix_to_face[0, ..., 0] >= 0
    reference_hard, reference_depth = camera.raster_camera_mesh(vertices, faces, K, WIDTH, HEIGHT)
    report["actual_raster_calls"] += 2
    report.update(hard_mask_bit_identical=bool(torch.equal(hard_mask, reference_hard)),
        hard_mask_pixels=int(hard_mask.sum().item())); persist()
    if not report["hard_mask_bit_identical"] or report["hard_mask_pixels"] <= 64:
        raise ValueError("Independent hard raster differs from existing camera helper")
    del own_hard, reference_hard, reference_depth
    soft_settings = RasterizationSettings(image_size=(HEIGHT, WIDTH), blur_radius=BLUR_RADIUS,
        faces_per_pixel=FACES_PER_PIXEL, perspective_correct=True, clip_barycentric_coords=False,
        cull_backfaces=False, cull_to_frustum=False, z_clip_value=None, max_faces_per_bin=len(faces))
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=soft_settings)
    shader = SoftSilhouetteShader(blend_params=BlendParams(sigma=SIGMA, gamma=GAMMA))

    def alpha(translation):
        mesh = Meshes(verts=[base+translation[None]], faces=[triangles])
        fragments = rasterizer(mesh); report["actual_raster_calls"] += 1
        result = shader(fragments, mesh)[0, ..., 3]
        if (result.shape != (HEIGHT, WIDTH) or not torch.isfinite(result).all()
                or bool((result < 0).any()) or bool((result > 1).any())):
            raise ValueError("Native SoftSilhouetteShader alpha contract failed")
        return result

    report.update(phase="translation_autograd", synthetic_target_translation_m=list(TARGET_TRANSLATION_M)); persist()
    with torch.no_grad():
        target = alpha(torch.tensor(TARGET_TRANSLATION_M, device="cuda", dtype=torch.float32)).clone()
    target_hash = hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest()
    report.update(target_frozen_before_optimization=True, synthetic_target_alpha_sha256=target_hash); persist()
    translation = torch.zeros(3, device="cuda", dtype=torch.float32, requires_grad=True)
    prediction = alpha(translation)
    loss = ((prediction-target)**2).mean()
    before = float(loss.detach().cpu())
    loss.backward(); report["actual_backward_calls"] += 1
    if translation.grad is None: raise ValueError("Raster/shader detached translation autograd")
    gradient = translation.grad.detach().cpu().numpy().copy()
    report.update(translation_gradient=gradient.tolist(), loss_before=before); persist()
    step = negative_gradient_step(gradient)
    report.update(translation_gradient=gradient.tolist(), translation_gradient_norm=float(np.linalg.norm(gradient.astype(np.float64))),
        actual_translation_step_m=step.tolist(), actual_translation_step_norm_m=float(np.linalg.norm(step)),
        loss_before=before, phase="fixed_step_replay"); persist()
    with torch.no_grad():
        step_tensor = torch.tensor(step, device="cuda", dtype=torch.float32)
        after = float(((alpha(step_tensor)-target)**2).mean().cpu())
        moved = (base+step_tensor[None]).cpu().numpy()
    report.update(loss_after=after, actual_translation_steps=1); persist()
    validate_measurement(before, after, gradient, step)
    if (not np.allclose(np.ptp(moved, axis=0), np.ptp(vertices, axis=0), rtol=0., atol=5e-7)
            or not np.allclose(moved-vertices, step[None], rtol=0., atol=2e-7)
            or not np.isfinite(moved).all() or (moved[:, 2] <= 1e-4).any()
            or hashlib.sha256(vertices.tobytes()).hexdigest() != vertices_hash
            or hashlib.sha256(target.cpu().numpy().tobytes()).hexdigest() != target_hash):
        raise ValueError("Fixed target/base geometry changed, scaled, clipped or translated inconsistently")
    for path, digest in frozen:
        if sha256(regular(path)) != digest: raise ValueError("Pinned source changed during capability probe")
    if report["actual_raster_calls"] != 5 or report["actual_backward_calls"] != 1:
        raise ValueError("Exactly2 hard+3 soft rasters/1 backward required")
    report.update(status="pass", phase="complete", gradient_capability_verified=True,
        loss_decrease_verified=True, physical_extent_retained=True, source_rehashed_after_run=True,
        peak_gpu_allocated_bytes=int(torch.cuda.max_memory_allocated()))


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ["WR_ROOT"]); revision = os.environ["WR_CODE_REVISION"]; image = os.environ["WR_IMAGE_ID"]
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root
            or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or not re.fullmatch("[0-9a-f]{40}", revision) or image != IMAGE):
        raise ValueError("Canonical Azure offline root/revision/pinned image required")
    out = root/BASE
    if out.resolve() != out.absolute() or not out.is_dir() or any(out.iterdir()):
        raise FileExistsError("Exclusive empty new capability output required")
    start = time.perf_counter()
    report = {"stage": STAGE, "status": "fail", "phase": "provenance", "producer_revision": revision, "image_id": image,
        "script_sha256": sha256(Path(__file__)), "camera_helper_sha256": sha256(Path(camera.__file__)),
        "pytorch3d_source_revision": camera.PYTORCH3D_REVISION, "pytorch3d_revision_assurance": "audited_pinned_image_not_runtime_git_attestation",
        "budget_seconds": BUDGET, "network": "none", "challenge_inputs_used": False, "ground_truth_used": False,
        "synthetic_reference_used": True, "hand_labeled_test": False, "oracle_modes": [], "accuracy_verified": False,
        "adoption_authorized": False, "submission_produced": False, "native_model_used": False,
        "alignment_calls": 0, "planned_translation_steps": 1, "actual_translation_steps": 0,
        "adaptive_line_search_used": False, "geometry_rescaled": False,
        "width": WIDTH, "height": HEIGHT, "source_width": SOURCE_WIDTH, "source_height": SOURCE_HEIGHT,
        "sigma": SIGMA, "gamma": GAMMA, "blur_radius": BLUR_RADIUS, "faces_per_pixel": FACES_PER_PIXEL,
        "translation_step_norm_m": STEP_M, "projection_atol_px": PROJECTION_ATOL_PX,
        "loss": "full_grid_mean_squared_soft_alpha_against_fixed_own_translated_alpha",
        "actual_raster_calls": 0, "actual_backward_calls": 0, "gradient_capability_verified": False,
        "settings": {"seed": 0, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False,
                     "deterministic_algorithms": True, "CPU_threads": 4, "dtype": "float32"}}
    with (out/"report.json").open("x") as handle:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-start
            handle.seek(0); json.dump(report, handle, allow_nan=False); handle.write("\n")
            handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*_): raise TimeoutError("Soft silhouette capability exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try:
            persist(); run(report, persist)
        except BaseException as error:
            report.update(status="fail", error=type(error).__name__, message=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
            persist(); (out/"report.json").chmod(0o444)
    print(json.dumps({k: report[k] for k in ("stage", "status", "loss_before", "loss_after", "elapsed_seconds")}))


if __name__ == "__main__": main()
