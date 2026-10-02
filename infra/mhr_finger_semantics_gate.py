"""Own neutral MHR finger-control semantics/support gate, never image accuracy.

The SHA-bound MHRDemo artifact (249 internal parameters) is inspected directly;
generic upstream MHR source is not treated as its executable specification.
Corrective vertex motion outside a hand is descriptive, not a forbidden effect.
"""
import argparse
from collections.abc import Mapping
import json
import os
from pathlib import Path
import platform
import re
import random
import signal
import time

import numpy as np
from world_reward.data import sha256
import mhr_determinism_gate as determinism
import mhr_fresh_replay_gate as fresh

MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
BODY_SHA = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
BODY_REVISION = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
BODY_RELATIVE = "weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"
STEPS = (.001, .002)
STATE_ATOL, QUAT_NORM_ATOL, DERIVATIVE_RTOL = 1e-7, 1e-5, .01
PARAMETER = re.compile(r"([rl])_(thumb|index|middle|ring|pinky)([0-3])_r([xyz])")
JOINT = re.compile(r"([rl])_(thumb|index|middle|ring|pinky)(?:[0-3]|_?null)")
METHODS = {"get_joint_names": "List[str]", "get_parameter_names": "List[str]",
           "get_parameter_transform": "Tensor", "get_num_identity_blendshapes": "int",
           "get_num_face_expression_blendshapes": "int"}


def validate_schemas(schemas):
    expected = {name: ([], result) for name, result in METHODS.items()}
    expected["forward"] = ([('identity_coeffs', 'Tensor'), ('model_parameters', 'Tensor'),
                            ('face_expr_coeffs', 'Tensor'), ('apply_correctives', 'bool')], "Tuple[Tensor, Tensor]")
    for name, (arguments, result) in expected.items():
        if schemas.get(name) != (arguments, result):
            raise ValueError(f"Unexpected reference method schema: {name}")


def finger_partition(joints, parameters, parents, transform, left, right):
    """Use actual names, exact linear-map support and a validated parent tree."""
    if (not isinstance(joints, list) or not isinstance(parameters, list)
            or not joints or len(parameters) != 249
            or any(not isinstance(v, str) or not v for v in joints + parameters)
            or len(set(joints)) != len(joints) or len(set(parameters)) != len(parameters)):
        raise ValueError("Require unique exposed reference names and 249 internal parameters")
    p, matrix = np.asarray(parents), np.asarray(transform)
    if (np.ma.isMaskedArray(parents) or p.shape != (len(joints),) or p.dtype.kind not in "iu"
            or np.any(p < -1) or np.any(p >= len(joints)) or np.count_nonzero(p == -1) != 1):
        raise ValueError("Require one-root integer parent tree")
    chains = []
    for index in range(len(joints)):
        chain = set(); current = index
        while current != -1:
            if current in chain: raise ValueError("Cyclic reference parent tree")
            chain.add(current); current = int(p[current])
        chains.append(chain)
    if (np.ma.isMaskedArray(transform) or matrix.shape != (7 * len(joints), 249)
            or matrix.dtype.kind != "f" or not np.isfinite(matrix).all()):
        raise ValueError("Require finite reference joint7-by-parameter transform")
    matches = [PARAMETER.fullmatch(name) for name in parameters[:204]]
    if [i for i, match in enumerate(matches) if match] != list(range(68, 122)):
        raise ValueError("Named finger rotation controls must partition exactly 68:122")
    for side, indices in (("l", left), ("r", right)):
        indices = np.asarray(indices)
        if (indices.shape != (27,) or indices.dtype.kind not in "iu"
                or not np.array_equal(np.sort(indices), [i for i, m in enumerate(matches) if m and m[1] == side])):
            raise ValueError("SHA-verified Body hand indices disagree with reference named sides")
    records = []
    for column in range(68, 122):
        match = matches[column]; side, finger = match.groups()[:2]
        stem = parameters[column].rsplit("_", 1)[0]
        if stem not in joints or f"{side}_wrist" not in joints:
            raise ValueError("Named finger stem/wrist missing from reference skeleton")
        affected = np.flatnonzero(matrix[:, column] != 0)
        if not len(affected) or np.any(~np.isin(affected % 7, [3, 4, 5])):
            raise ValueError("Finger column must affect local rotations only")
        local = sorted(set((affected // 7).tolist()))
        if any(not JOINT.fullmatch(joints[j]) or JOINT.fullmatch(joints[j]).groups()[:2] != (side, finger) for j in local):
            raise ValueError("Finger column affects another side/finger/nonfinger joint")
        wrist = joints.index(f"{side}_wrist")
        if any(wrist not in chains[j] for j in local):
            raise ValueError("Finger support does not descend from its named wrist")
        closure = [j for j, chain in enumerate(chains) if any(a in chain for a in local)]
        if any(not JOINT.fullmatch(joints[j]) or JOINT.fullmatch(joints[j]).groups()[:2] != (side, finger) for j in closure):
            raise ValueError("Finger descendant closure includes another semantic region")
        records.append({"column": column, "parameter": parameters[column], "side": side, "finger": finger,
                        "local_rotation_rows": affected.tolist(), "local_joints": local, "global_closure": closure})
    return records


def check_geometry(vertices, skeleton, joint_count):
    v, s = np.asarray(vertices), np.asarray(skeleton)
    if (v.ndim != 3 or v.shape[1:] != (18439, 3) or s.shape != (len(v), joint_count, 8)
            or v.dtype.kind != "f" or s.dtype.kind != "f" or not np.isfinite(v).all() or not np.isfinite(s).all()
            or np.any(s[..., 7] <= 0)
            or np.max(np.abs(np.linalg.norm(s[..., 3:7].astype(np.float64), axis=-1) - 1)) > QUAT_NORM_ATOL):
        raise ValueError("Reference vertices/global skeleton must be finite with positive scales and unit quaternions")


def support_evidence(neutral_v, neutral_s, vertices, skeleton, records, *, evidence=None, on_progress=None):
    """Quaternion sign alignment is representation-only; no pose/geometry repair."""
    output = [] if evidence is None else evidence
    for i, record in enumerate(records):
        allowed = np.asarray(record["global_closure"]); outside = np.setdiff1d(np.arange(len(neutral_s)), allowed)
        derivatives, measurements = [], []
        diagnostic = {"column": record["column"], "parameter": record["parameter"], "status": "fail", "steps": measurements}
        output.append(diagnostic)
        for step_index, step in enumerate(STEPS):
            indices = [i * 4 + step_index * 2, i * 4 + step_index * 2 + 1]
            states = skeleton[indices].astype(np.float64)
            q = states[..., 3:7]
            q = q * np.where((q * neutral_s[None, :, 3:7]).sum(-1, keepdims=True) < 0, -1., 1.)
            position_delta_m = np.abs(states[..., :3] - neutral_s[None, :, :3]) / 100
            quat_delta = np.abs(q - neutral_s[None, :, 3:7])
            scale_delta = np.abs(states[..., 7] - neutral_s[None, :, 7])
            maximum = lambda array: float(np.max(array)) if array.size else 0.
            derivative = (q[1] - q[0]) / (2 * step)
            effect = float(np.linalg.norm(derivative[allowed], axis=-1).max())
            derivatives.append(derivative)
            displacement = np.linalg.norm(vertices[indices].astype(np.float64) - neutral_v[None], axis=-1) / 100
            vertex_derivative = (vertices[indices[1]].astype(np.float64) - vertices[indices[0]]) / (200 * step)
            vertex_effect = float(np.linalg.norm(vertex_derivative, axis=-1).max())
            measurements.append({"step_rad": step, "excluded_joint_position_max_m": maximum(position_delta_m[:, outside]),
                                 "excluded_joint_quaternion_max": maximum(quat_delta[:, outside]),
                                 "all_joint_scale_max_difference": maximum(scale_delta),
                                 "orientation_derivative_max_per_rad": effect,
                                 "vertex_derivative_max_m_per_rad": vertex_effect,
                                 "vertex_displacement_mean_m": float(displacement.mean()),
                                 "vertex_displacement_max_m": float(displacement.max())})
            if on_progress is not None: on_progress()
            if (maximum(position_delta_m[:, outside]) > STATE_ATOL / 100
                    or maximum(quat_delta[:, outside]) > STATE_ATOL or maximum(scale_delta) > STATE_ATOL):
                raise ValueError(f"Finger {record['parameter']} changed an excluded joint/scale")
            if effect <= 1e-6: raise ValueError(f"Finger {record['parameter']} has no observable orientation effect")
            if not np.isfinite(vertex_effect) or vertex_effect <= 0:
                raise ValueError(f"Finger {record['parameter']} has no finite nonzero central vertex derivative")
        relative = float(np.max(np.linalg.norm(derivatives[0] - derivatives[1], axis=-1))
                         / max(np.max(np.linalg.norm(derivatives[1], axis=-1)), 1e-12))
        diagnostic["orientation_derivative_relative_difference"] = relative
        if relative > DERIVATIVE_RTOL: raise ValueError(f"Finger {record['parameter']} two-step central orientation derivative is inconsistent")
        diagnostic["status"] = "pass"
        if on_progress is not None: on_progress()
    return output


def regular_hash(path, root, expected, size=None):
    if (not path.is_file() or any(p.is_symlink() for p in (path, *path.parents) if p.is_relative_to(root))
            or (size is not None and path.stat().st_size != size) or sha256(path) != expected):
        raise ValueError(f"Frozen input missing or altered: {path}")


def identity_rows(shared_identity, count):
    """MHRDemo concatenates rows: explicit identical identity, never broadcast."""
    value = np.asarray(shared_identity)
    if (np.ma.isMaskedArray(shared_identity) or value.shape != (45,) or value.dtype.kind != "f"
            or not np.isfinite(value).all() or type(count) is not int or count < 1):
        raise ValueError("Require one finite shared identity45 and explicit positive batch size")
    return np.repeat(value[None], count, axis=0)



def require_determinism(report):
    expected = {"stage": "reference_mhr_failed_fixture_determinism_diagnosis", "status": "pass",
                "model_sha256": MODEL_SHA, "script_sha256": sha256(Path(determinism.__file__)),
                "strict_replay_bitexact": True, "strict_repair_route_observed": True,
                "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "TF32": False, "network": "none",
                "challenge_inputs_used": False, "execution_dtype": "float32", "forward_calls": 20}
    if (any(type(report.get(k)) is not type(v) or report.get(k) != v for k,v in expected.items())
            or not determinism.outcome(report.get("groups", []))):
        raise ValueError("Require actual SHA-bound strict CPU/CUDA replay diagnosis before semantics-v3")



def require_fresh_replay(report):
    expected={"stage":fresh.STAGE,"status":"pass","model_sha256":MODEL_SHA,
              "script_sha256":sha256(Path(fresh.__file__)),"diagnostic_helper_sha256":sha256(Path(determinism.__file__)),
              "unoptimized_fresh_replay_bitexact":True,"fresh_process_per_condition":True,"warmup_calls":0,
              "forward_calls":12,"strict_algorithms":True,"network":"none","challenge_inputs_used":False}
    if (any(type(report.get(k)) is not type(v) or report.get(k)!=v for k,v in expected.items())
            or not fresh.outcome(report.get("conditions",[]))):
        raise ValueError("Require actual fresh-process unoptimized reference replay before semantics-v4")

def strict_reference_runtime(torch):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise RuntimeError("Require deterministic CUBLAS workspace configured before torch import")
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux" or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}:
        raise RuntimeError("Require Azure Linux CUDA container with network none")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", ""); image = os.environ.get("WR_IMAGE_ID", "")
    if not re.fullmatch(r"[0-9a-f]{40}", revision) or not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Require immutable source revision/image ID")
    output = root / "results/mhr-finger-semantics-v4.json"
    if output.is_symlink(): raise FileExistsError("Frozen semantic report already exists")
    with output.open("x") as handle:
        started = time.perf_counter()
        report = {"stage": "own_reference_mhr_finger_semantics_support", "status": "fail", "phase": "input_integrity",
                  "code_revision": revision, "source_image_id": image, "script_sha256": sha256(Path(__file__)),
                  "network": "none", "challenge_inputs_used": False, "hand_labeled_test": False,
                  "adoption_performed": False, "hand_accuracy_verified": False, "challenge_performance_verified": False,
                  "scope": "own_neutral_reference_control_semantics_not_RGB_accuracy_or_nativeSAM_geometry_parity",
                  "budgets": {"total_seconds": 300, "CUDA_phase_seconds": 120, "forward_calls": 6, "batch_sizes": [1, 216, 216]},
                  "tolerances": {"state_absolute_model_cm_quaternion_scale": STATE_ATOL,
                                 "quaternion_unit_norm_absolute": QUAT_NORM_ATOL, "orientation_derivative_relative": DERIVATIVE_RTOL},
                  "steps_rad": list(STEPS), "conditions": [], "forward_calls": 0}
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started
            handle.seek(0); handle.write(json.dumps(report, allow_nan=False) + "\n"); handle.truncate(); handle.flush(); os.fsync(handle.fileno())
        def expired(*args): raise TimeoutError("MHR semantic gate exceeded its frozen time budget")
        previous_alarm = signal.signal(signal.SIGALRM, expired); previous_term = signal.signal(signal.SIGTERM, expired)
        signal.alarm(300)
        try:
            persist()
            model_path = root / "weights/mhr/mhr_model.pt"; body = root / BODY_RELATIVE; checkpoint = body / "model.ckpt"
            acquisition_path = root / "results/weights-acquisition.json"; acquisition_sha = sha256(acquisition_path)
            regular_hash(acquisition_path, root, acquisition_sha)
            acquisition = json.loads(acquisition_path.read_text())
            assets = [a for a in acquisition["assets"] if a.get("repo_id") == "facebook/sam-3d-body-dinov3"]
            if len(assets) != 1 or assets[0].get("revision") != BODY_REVISION or assets[0].get("path") != str(body):
                raise ValueError("Require pinned official Body acquisition revision/path")
            regular_hash(model_path, root, MODEL_SHA, 696110248); regular_hash(checkpoint, root, BODY_SHA, 2109129346)
            report.update(model_sha256=MODEL_SHA, body_checkpoint_sha256=BODY_SHA, body_revision=BODY_REVISION,
                          acquisition_report_sha256=acquisition_sha, phase="metadata_and_checkpoint")
            persist()
            diagnosis_path = root / "results/mhr-determinism.json"; diagnosis_sha = sha256(diagnosis_path)
            regular_hash(diagnosis_path, root, diagnosis_sha)
            require_determinism(json.loads(diagnosis_path.read_text()))
            fresh_path=root/"results/mhr-fresh-replay.json";fresh_sha=sha256(fresh_path)
            regular_hash(fresh_path,root,fresh_sha);require_fresh_replay(json.loads(fresh_path.read_text()))
            if "torch" in __import__("sys").modules: raise RuntimeError("CUBLAS setup must precede torch import")
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
            import torch
            strict_reference_runtime(torch)
            report.update(determinism_report_sha256=diagnosis_sha, deterministic_algorithms=True,
                          CUBLAS_WORKSPACE_CONFIG=":4096:8", TF32=False, seed=0,
                          jit_optimized_execution=False, fresh_replay_report_sha256=fresh_sha)
            if not torch.cuda.is_available(): raise RuntimeError("CUDA required; no CPU/local forward fallback")
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            state = payload.get("state_dict", payload) if isinstance(payload, Mapping) else None
            if not isinstance(state, Mapping): raise ValueError("Pinned Body checkpoint state missing")
            hands = []
            for side in ("left", "right"):
                value = state.get(f"head_pose.hand_joint_idxs_{side}")
                if not torch.is_tensor(value): raise ValueError("Pinned Body hand indices missing")
                hands.append(value.detach().cpu().numpy().copy())
            del state, payload
            with torch.jit.optimized_execution(False):
                model = torch.jit.load(str(model_path), map_location="cuda").float().eval()
            schemas = {}
            for name in (*METHODS, "forward"):
                schema = model._c._get_method(name).schema
                if not schema.arguments or schema.arguments[0].name != "self" or "MHRDemo" not in str(schema.arguments[0].type):
                    raise ValueError("Require actual frozen MHRDemo method schemas")
                schemas[name] = ([(a.name, str(a.type)) for a in schema.arguments[1:]], str(schema.returns[0].type))
            validate_schemas(schemas)
            joints, parameters = model.get_joint_names(), model.get_parameter_names()
            matrix = model.get_parameter_transform().detach().cpu().numpy()
            parents = model.character_torch.skeleton.joint_parents.detach().cpu().numpy()
            if len(joints) != 127 or tuple(matrix.shape) != (889, 249): raise ValueError("Frozen reference dimensions differ")
            if (model.get_num_identity_blendshapes(), model.get_num_face_expression_blendshapes()) != (45, 72):
                raise ValueError("Frozen reference external identity/expression ABI differs from kit")
            if (joints != list(model.character_torch.skeleton.joint_names)
                    or parameters != list(model.character_torch.parameter_transform.parameter_names)
                    or not np.array_equal(matrix, model.character_torch.parameter_transform.parameter_transform.detach().cpu().numpy())):
                raise ValueError("Exported metadata differs from actual forward submodule metadata")
            records = finger_partition(joints, parameters, parents, matrix, *hands)
            report.update(phase_A_verified=True, joint_names=joints, named_finger_controls=records,
                          internal_parameter_count=len(parameters), model_class="MHRDemo", schemas=schemas,
                          hand_indices_left=hands[0].tolist(), hand_indices_right=hands[1].tolist(),
                          phase="CUDA_support", torch=torch.__version__, execution_dtype="float32", geometry_units="model_cm")
            persist()
            controls = torch.zeros(216, 204, device="cuda")
            for i in range(54):
                controls[i*4:(i+1)*4, 68+i] = torch.tensor([-.001, .001, -.002, .002], device="cuda")
            fixed = torch.cat((controls[:, :68], controls[:, 122:]), dim=1)
            if torch.count_nonzero(fixed): raise ValueError("Own nuisance/scales must stay exactly zero")
            controls_copy = controls.clone()
            identity = torch.as_tensor(identity_rows(np.zeros(45, np.float32), 216), device="cuda")
            expr = torch.zeros(216, 72, device="cuda")
            deadline = min(started + 300, time.perf_counter() + 120); signal.alarm(max(1, int(deadline-time.perf_counter())))
            previous_skeleton = None
            with torch.inference_mode(), torch.jit.optimized_execution(False):
                for correctives in (False, True):
                    report["active_correctives"] = correctives
                    persist()
                    neutral_v, neutral_s = model(identity[:1], torch.zeros(1, 204, device="cuda"), expr[:1], correctives); report["forward_calls"] += 1
                    persist()
                    v, s = model(identity, controls, expr, correctives); report["forward_calls"] += 1
                    persist()
                    replay_v, replay_s = model(identity, controls, expr, correctives); report["forward_calls"] += 1
                    persist()
                    torch.cuda.synchronize()
                    if time.perf_counter() > deadline: raise TimeoutError("CUDA semantic phase exceeded120s")
                    if not torch.equal(v, replay_v) or not torch.equal(s, replay_s): raise ValueError("Reference replay changed geometry")
                    if not torch.equal(controls, controls_copy) or torch.count_nonzero(identity) or torch.count_nonzero(expr):
                        raise ValueError("Reference forward modified supplied own controls/identity/expressions")
                    arrays = [a.detach().cpu().numpy() for a in (neutral_v, neutral_s, v, s)]
                    check_geometry(arrays[0], arrays[1], 127); check_geometry(arrays[2], arrays[3], 127)
                    if previous_skeleton is not None and (not np.array_equal(previous_skeleton[0], arrays[1]) or not np.array_equal(previous_skeleton[1], arrays[3])):
                        raise ValueError("Correctives changed the skeleton instead of geometry only")
                    condition = {"apply_correctives": correctives, "replay_bit_identical": True,
                                 "finite_geometry_and_unit_skeleton": True, "controls": []}
                    report["conditions"].append(condition)
                    persist()
                    support_evidence(arrays[0][0], arrays[1][0], arrays[2], arrays[3], records,
                                     evidence=condition["controls"], on_progress=persist)
                    previous_skeleton = (arrays[1].copy(), arrays[3].copy())
                    print(json.dumps({"phase": "support_complete", "apply_correctives": correctives, "controls": 54}), flush=True)
            signal.alarm(max(1, int(started+300-time.perf_counter())))
            regular_hash(model_path, root, MODEL_SHA, 696110248); regular_hash(checkpoint, root, BODY_SHA, 2109129346)
            regular_hash(acquisition_path, root, acquisition_sha)
            regular_hash(diagnosis_path, root, diagnosis_sha)
            regular_hash(fresh_path,root,fresh_sha)
            if time.perf_counter() > started+300: raise TimeoutError("Semantic gate exceeded300s")
            report.update(status="pass", phase="complete", phase_B_verified=True,
                          named_finger_joint_partition_verified=True, excluded_joint_invariance_verified=True,
                          correctives_skeleton_bit_identical=True, nonhand_vertex_invariance_claimed=False,
                          neutral_only_support_generalization_verified=False, SAM70_mapping_verified=False)
            report.pop("active_correctives", None)
        except Exception as exc:
            report["error"] = {"type": type(exc).__name__, "message": str(exc)}
            raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, previous_alarm); signal.signal(signal.SIGTERM, previous_term)
            persist()
            print(json.dumps({k: report[k] for k in ("stage", "status", "phase", "forward_calls", "elapsed_seconds")}), flush=True)


if __name__ == "__main__": main()
