"""D96 bounded native root refit; public evidence only, no quality selection."""
import argparse
import json
import os
from pathlib import Path
import platform
import re
import signal
import sys
import time

import numpy as np
import keypoint_rgb_public as public
from world_reward import metric_alignment, root_refit as policy
from world_reward.data import sha256

baseline = public.baseline
native = public.native
BASE, OUT, STAGE, BUDGET, OPTIMIZER = public.BASE, public.OUT, public.STAGE, public.BUDGET, public.OPTIMIZER
helper_identities = public.helper_identities
public_predictions = public.public_predictions
object_sample = public.object_sample
validate_proxy = public.validate_proxy
validate_candidate = public.validate_candidate
frozen_proxies = public.frozen_proxies
frozen_candidates = public.frozen_candidates

def optimization_schedule(evaluate, update):
    """Sixty observed states,59 updates; the final unevaluated state cannot win."""
    losses = []
    for index in range(policy.EVALUATED_STATES):
        loss = float(evaluate(index))
        if not np.isfinite(loss) or loss < 0: raise ValueError("Every evaluated native objective must be finite/nonnegative")
        losses.append(loss)
        if index+1 < policy.EVALUATED_STATES: update(index)
    return losses, policy.best_evaluated(np.asarray(losses, np.float64))


def strict_forward(torch):
    if (not torch.are_deterministic_algorithms_enabled() or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32 or torch.backends.cudnn.benchmark
            or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or torch.is_inference_mode_enabled() or not torch.is_grad_enabled()):
        raise ValueError("Strict native differentiable settings required without kernel suppression")
    if max(torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()) > 32*1024**3:
        raise MemoryError("Declared32GiB GPU envelope exceeded")


def differentiable_native(torch, head, fixed, latent, limits):
    strict_forward(torch)
    delta = torch.tanh(latent)*torch.tensor(policy.PHYSICAL_BOUNDS, device="cuda", dtype=torch.float32)
    t0 = fixed["pred_cam_t"][0]
    translation = torch.stack((t0[0]+delta[0], t0[1]+delta[1], t0[2]*torch.exp(delta[2])))
    euler = fixed["global_rot"]+delta[3:][None]
    values = head.mhr_forward(global_trans=torch.zeros_like(euler), global_rot=euler, body_pose_params=fixed["body_pose_params"],
        hand_pose_params=fixed["hand_pose_params"], shape_params=fixed["shape_params"], scale_params=fixed["scale_params"],
        expr_params=fixed["expr_params"], return_keypoints=True, return_joint_coords=True, return_model_params=True, return_joint_rotations=True)
    expected = ((1, native.VERTICES, 3), (1, 308, 3), (1, 127, 3), (1, 204), (1, 127, 3, 3))
    if (not isinstance(values, tuple) or len(values) != 5 or any(tuple(v.shape) != s or v.dtype != torch.float32
            or not torch.isfinite(v).all() for v, s in zip(values, expected))): raise ValueError("Actual differentiable full native ABI differs")
    controls = torch.cat((values[3][0], fixed["shape_params"][0]))
    if (torch.any(controls < limits[:, 0]) or torch.any(controls > limits[:, 1])
            or not torch.equal(values[3][0, 3:6], euler[0])):
        raise ValueError("Complete249 native bounds/rootEuler mapping violated; no clipping")
    flip = torch.tensor([1., -1., -1.], device="cuda", dtype=torch.float32)
    v, kp, j = [x[0]*flip+translation for x in values[:3]]
    if any(torch.any(x[:, 2] <= 0) for x in (v, kp, j)): raise ValueError("No negative cameraZ, clipping or dropped native point")
    strict_forward(torch)
    return (v, kp, j, values[3][0], values[4][0]), delta, euler[0], translation


def fit_frame(torch, head, limits, raw, pair, observation, record, report, persist):
    target, indices, validity = policy.training_observations(observation["keypoints"][0], observation["scores"][0])
    fixed = {k: torch.tensor(raw[k].copy(), device="cuda", dtype=torch.float32)[None] for k in native.BLOCKS if k != "mhr_model_params"}
    fixed["shape_params"] = torch.tensor(pair["shared_shape_params"].copy(), device="cuda")[None]
    fixed["scale_params"] = torch.tensor(pair["shared_scale_params"].copy(), device="cuda")[None]
    u = torch.zeros(6, device="cuda", dtype=torch.float32, requires_grad=True)
    optimizer = torch.optim.Adam([u], lr=.01, betas=(.9, .999), eps=1e-8)
    K = torch.tensor(pair["camera_K"].astype(np.float32), device="cuda")
    observed = torch.tensor(target.astype(np.float32), device="cuda"); selected = torch.tensor(indices, device="cuda")
    row = dict(file=record["file"], training_validity=validity.tolist(), training_MHR_indices=indices.tolist(), training_points=len(target),
               evaluated_losses=[], best_evaluated_index=None, native_forward_calls=0, adam_updates=0)
    report["fit_records"].append(row); persist(); state = {}
    cpu = lambda x: x.detach().cpu().numpy().copy()
    def evaluate(index):
        arrays, delta, euler, translation = differentiable_native(torch, head, fixed, u, limits)
        report["counters"]["objective_native_heads"] += 1; row["native_forward_calls"] += 1
        keypoints = arrays[1][selected]; projected = keypoints[:, :2]/keypoints[:, 2:]*K.diag()[:2]+K[:2, 2]
        residual = projected-observed
        if index == 0:
            checks = {name: float(np.linalg.norm(cpu(value).astype(np.float64)-pair[original].astype(np.float64), axis=-1).max())
                for name, value, original in zip(("vertices", "keypoints", "joints"), arrays[:3],
                    ("shared_vertices_camera_m", "shared_keypoints_camera_m", "shared_joints_camera_m"))}
            checks["controls"] = float(np.abs(cpu(arrays[3]).astype(np.float64)-pair["shared_model_controls"].astype(np.float64)).max())
            row["initial_parity"] = checks; persist()
            if any(not np.isfinite(v) or v > 1e-5 for v in checks.values()): raise ValueError("Initial shared native parity exceeds1e-5m")
            jacobian = torch.stack([torch.autograd.grad(r, u, retain_graph=True, create_graph=False)[0] for r in residual.reshape(-1)])
            report["counters"]["initial_jacobian_rows"] += 2*len(target)
            row["initial_observation_jacobian"] = policy.jacobian_evidence(cpu(jacobian), len(target)); persist()
        q = torch.linalg.vector_norm(residual, dim=-1)/policy.HUBER_PIXELS
        data_loss = torch.where(q <= 1, .5*q*q, q-.5).mean()
        loss = data_loss+.5*torch.square(delta/policy.PRIOR_SIGMA).sum()
        total = float(loss.detach().cpu()); row["evaluated_losses"].append(total)
        if not np.isfinite(total) or total < 0: raise ValueError("Nonfinite objective; no clipping")
        if "best_loss" not in state or total < state["best_loss"]:
            state.update(best_loss=total, best_latent=u.detach().clone(), best_index=index, best_delta=cpu(delta))
        state["loss"] = loss; persist(); return total
    def update(_):
        optimizer.zero_grad(set_to_none=True); state["loss"].backward()
        if u.grad is None or not torch.isfinite(u.grad).all(): raise ValueError("Finite actual native gradient required")
        optimizer.step(); report["counters"]["adam_updates"] += 1; row["adam_updates"] += 1; strict_forward(torch)
    losses, best = optimization_schedule(evaluate, update)
    if best != state["best_index"]: raise ValueError("First-tie best evaluated state changed")
    arrays, delta, euler, translation = differentiable_native(torch, head, fixed, state["best_latent"], limits)
    report["counters"]["final_native_heads"] += 1
    if cpu(delta).tobytes() != state["best_delta"].tobytes(): raise ValueError("Selected candidate changed on final native replay")
    candidate = {k: raw[k].copy() for k in native.BLOCKS}
    candidate.update(global_rot=cpu(euler), pred_cam_t=cpu(translation), shape_params=pair["shared_shape_params"].copy(),
        scale_params=pair["shared_scale_params"].copy(), mhr_model_params=cpu(arrays[3]), vertices_camera_m=cpu(arrays[0]),
        keypoints_camera_m=cpu(arrays[1]), joints_camera_m=cpu(arrays[2]), joint_global_rotations=cpu(arrays[4]),
        human_faces=pair["human_faces"].copy(), hand_mask_left=pair["hand_mask_left"].copy(), hand_mask_right=pair["hand_mask_right"].copy(),
        camera_K=pair["camera_K"].copy(), clip_index=raw["clip_index"].copy(), frame_index=raw["frame_index"].copy(),
        latent=cpu(state["best_latent"]), physical_delta=cpu(delta))
    validate_candidate(candidate, record, raw, pair)
    row.update(best_evaluated_index=best, selected_total_objective=losses[best], selected_physical_delta=cpu(delta).tolist(),
        heldout_COCO_indices=list(policy.HELDOUT_COCO), heldout_used_for_fit=False)
    persist(); return candidate


def perform(root, out, report, persist):
    records, raw, pairs, dw, lineage = public_predictions(root); helpers = helper_identities()
    report.update(lineage=lineage, source_helpers=helpers, phase="native_model_load"); persist()
    if "torch" in sys.modules: raise ValueError("CUBLAS must precede Torch import")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    import pytorch3d
    if not torch.cuda.is_available(): raise RuntimeError("Actual offline CUDA required")
    torch.manual_seed(0); torch.cuda.manual_seed_all(0); torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False; torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True; torch.cuda.reset_peak_memory_stats()
    model, estimator, faces, body_source = native.human.load_model(root, torch)
    model.eval(); model.requires_grad_(False)
    head = model.head_pose; head.eval(); head.requires_grad_(False)
    limits = head.mhr.get_parameter_limits().detach().clone()
    if (tuple(limits.shape) != (249, 2) or torch.isnan(limits).any() or torch.any(limits[:, 0] > limits[:, 1])):
        raise ValueError("Actual native249 parameter limits required")
    if (any(p.requires_grad for p in head.parameters()) or not np.array_equal(faces, pairs[0]["human_faces"])
            or head.enable_hand_model or tuple(head.scale_mean.shape) != (68,) or tuple(head.scale_comps.shape) != (28, 68)):
        raise ValueError("Frozen full-body native head/scales/topology required")
    frozen_source = json.loads((root/BASE/"baseline_v1/report.json").read_text())
    if body_source != frozen_source["body_model"]: raise ValueError("Current native load differs from frozen baseline")
    source_path = Path(sys.modules[type(head).__module__].__file__)
    head_source = native.regular(source_path)
    report.update(native_head_source=head_source, body_model=body_source, torch_version=str(torch.__version__),
        CUDA_version=torch.version.cuda, pytorch3d_version=str(pytorch3d.__version__), seed=0, TF32=False,
        CUBLAS_WORKSPACE_CONFIG=":4096:8", deterministic_algorithms=True, warn_only=False, threads=4, phase="common_object_proxy"); persist()
    del model, estimator; torch.cuda.empty_cache()
    rendered = []
    for record, pair in zip(records, pairs):
        mask, depth = native.raster_camera_mesh(pair["shared_vertices_camera_m"], pair["human_faces"], pair["camera_K"], native.WIDTH, native.HEIGHT)
        report["counters"]["proxy_rasters"] += 1; rendered.append((mask.cpu().numpy(), depth.cpu().numpy())); persist()
    for clip in range(3):
        indices = range(clip*5, clip*5+5)
        visibility = [rendered[i][0] & raw[i]["human_mask"] & ~raw[i]["object_mask"] & raw[i]["validity"] for i in indices]
        alignment = metric_alignment.fit_shared_depth_scale([raw[i]["raw_depth"] for i in indices],
            [rendered[i][1] for i in indices], visibility, list(range(5)), min_correspondences_per_frame=32, min_supported_frames=5)
        if alignment.supported_frames != 5: raise ValueError("All5 original shared-human depth supports required")
        report["proxy_alignments"].append(alignment.to_dict())
        for i in indices:
            points, pixels = object_sample(raw[i], alignment.shared_scale)
            proxy = dict(object_points_camera_m=points, pixel_indices=pixels, shared_depth_scale=np.asarray(alignment.shared_scale, np.float64),
                clip_index=raw[i]["clip_index"].copy(), frame_index=raw[i]["frame_index"].copy())
            validate_proxy(proxy, records[i], raw[i]); report["proxy_outputs"].append(native.save(out, "proxies", records[i], proxy))
    frozen_proxies(root, records, report["proxy_outputs"], raw, pairs)
    report.update(object_proxies_frozen_before_fit=True, phase="native_root_optimization"); persist()
    for record, original, pair, observation in zip(records, raw, pairs, dw):
        report.update(active_file=record["file"]); persist()
        candidate = fit_frame(torch, head, limits, original, pair, observation, record, report, persist)
        report["candidate_outputs"].append(native.save(out, "candidates", record, candidate)); persist()
    frozen_candidates(root, records, report["candidate_outputs"], raw, pairs); frozen_proxies(root, records, report["proxy_outputs"], raw, pairs)
    report.update(all_candidates_frozen=True, native_arrays_verified=True, phase="final_public_audit"); persist()
    _, _, _, _, final_lineage = public_predictions(root)
    if final_lineage != lineage or helper_identities() != helpers or native.regular(source_path) != head_source:
        raise ValueError("Original frozen public/source/asset evidence changed")
    expected = dict(proxy_rasters=15, objective_native_heads=900, final_native_heads=15, adam_updates=885)
    if any(report["counters"][k] != v for k, v in expected.items()): raise ValueError("Exact preregistered15-frame call/update budget required")
    if {p.name for p in out.iterdir()} != {"proxies", "candidates", "report.json"}: raise ValueError("Exact public fit artifact inventory required")
    report.update(status="pass", phase="complete", frames=15, inputs_unchanged=True, all_cases_retained=True,
        max_gpu_allocated_bytes=torch.cuda.max_memory_allocated(), max_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    report.pop("active_file", None)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root=Path(os.environ.get("WR_ROOT", "")); out=root/OUT; revision=os.environ.get("WR_CODE_REVISION", "")
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or out.resolve()!=out.absolute()
            or os.geteuid()!=1000 or {p.name for p in Path("/sys/class/net").iterdir()}!={"lo"}
            or os.environ.get("WR_IMAGE_ID")!=native.IMAGE_ID or not re.fullmatch("[0-9a-f]{40}", revision)
            or not out.is_dir() or any(out.iterdir()) or any(p.is_symlink() for p in (out,*out.parents))):
        raise ValueError("Fresh reserved offline native root-fit output required")
    (out/"proxies").mkdir(); (out/"candidates").mkdir(); started=time.perf_counter(); path=out/"report.json"
    report=dict(stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,image_id=native.IMAGE_ID,
        script_sha256=sha256(Path(__file__)),network="none",device="cuda",budget_seconds=BUDGET,optimizer=OPTIMIZER,
        training_COCO_indices=list(policy.TRAIN_COCO),training_MHR_indices=list(policy.TRAIN_MHR),initial_jacobian_includes_priors=False,
        private_truth_read=False,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        quality_verified=False,accuracy_verified=False,adoption_authorized=False,full_HOI_verified=False,
        shared_identity_constant=True,body_hands_fixed=True,camera_fixed=True,object_proxies_frozen_before_fit=False,
        all_candidates_frozen=False,native_arrays_verified=False,inputs_unchanged=False,all_cases_retained=False,
        candidate_outputs=[],proxy_outputs=[],proxy_alignments=[],fit_records=[],
        counters=dict(proxy_rasters=0,objective_native_heads=0,final_native_heads=0,adam_updates=0,initial_jacobian_rows=0))
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"]=time.perf_counter()-started; stream.seek(0); json.dump(report,stream,allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole public native root fit exceeded180s")
        old=signal.signal(signal.SIGALRM,expired); term=signal.signal(signal.SIGTERM,expired); signal.alarm(BUDGET)
        try: persist(); perform(root,out,report,persist)
        except Exception as error: report.update(error_type=type(error).__name__,error=str(error)); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM,old); signal.signal(signal.SIGTERM,term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
