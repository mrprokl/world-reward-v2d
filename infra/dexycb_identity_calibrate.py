"""CPU initial-identity micro-gate after immutable automatic full-clip evidence.

Public features average ALL supported hand/interval slots per object. Units are
image-diagonal fractions per original frame index, not seconds. No future mask,
contact, calibrated probability, temporal identity MAP or 3D accuracy is claimed.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

if __name__ == "__main__":
    _code, _revision = Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    if (not re.fullmatch(r"[0-9a-f]{40}", _revision)
            or _code != Path("/srv/scenesmith/world-reward/jobs") / _revision / "run_dexycb_identity_calibrate/code"
            or _code.resolve() != _code or Path(__file__).resolve() != _code / "infra/dexycb_identity_calibrate.py"):
        raise ValueError("Actual immutable CPU producer required before imports")
    sys.path[:0] = [str(_code / "infra"), str(_code / "src")]

import numpy as np
import dexycb_identity_infer as infer
from world_reward import identity_calibration as policy
from world_reward.relational_motion import RelationalMotionEvidence, relational_motion_features
from world_reward.prompt_selection import BoxDetection, non_maximum_suppression, _select_detection

ROOT, BASE = infer.ROOT, infer.BASE
ENTRY, BUDGET = "run_dexycb_identity_calibrate", 120
FOLDERS = {"public_features": "public_features_v1", "private_calibration": "private_calibration_v1"}
FEATURE_KEYS = {"frame_index", "candidate_index", "features", "supported", "support_count"}
RULE = dict(aggregation="mean_all_supported_interval_hand_pairs_per_object", minimum_support_count=1,
    feature_names=["relative_velocity_norm", "hand_velocity_energy", "object_velocity_energy"],
    time_basis="original_frame_index_not_seconds", l2=1., minimum_raw_score=0., minimum_raw_gap=0.,
    decision_clips=3, decision_required_successes=3, evaluation_clips=6, evaluation_minimum_success_gain=1,
    allowed_wrong_id_increase=0, identity_majority="strictly_greater_than_half",
    reciprocal_coverage="strictly_greater_than_half_target_recall", baseline_confidence=.3,
    baseline_nms_iou=.7, baseline_ambiguity_margin=.05)
require, binding = infer.require, infer.binding


def closure(code, revision, entry, helpers):
    """Original closure format, with genuine empty package initializers retained."""
    require(binding.canonical(code) == ROOT / "jobs" / revision / entry / "code"
            and re.fullmatch(r"[0-9a-f]{40}", revision), "Canonical producer namespace required")
    markers = {}
    for name in ("revision", "source-sha256"):
        raw, _ = binding.selected.read(code.parent / name, 100, readonly=False)
        require(raw == (revision + "\n").encode() if name == "revision"
                else bool(re.fullmatch(b"[0-9a-f]{64}\n", raw)), "Actual dispatch markers required")
        markers[name] = raw.decode()
    digest, pins = hashlib.sha256(), {}
    for path in (code, *sorted(code.rglob("*"))):
        binding.canonical(path); s = path.lstat()
        require(not s.st_mode & 0o222 and (stat.S_ISREG(s.st_mode) or stat.S_ISDIR(s.st_mode)), "Frozen code closure required")
        if path.is_file():
            _, pin = binding.selected.read(path, 2_000_000, empty=True)
            name = str(path.relative_to(code)); pins[name] = {k: pin[k] for k in ("bytes", "sha256")}
            digest.update(name.encode() + b"\0" + bytes.fromhex(pin["sha256"]))
    require(set(helpers) <= set(pins), "Complete literal helper closure required")
    return dict(markers=markers, closure_sha256=digest.hexdigest(), helpers={n: pins[n] for n in helpers})


def source_binding(code, revision):
    require(Path(__file__).resolve() == code / "infra/dexycb_identity_calibrate.py"
            and Path(infer.__file__).resolve() == code / "infra/dexycb_identity_infer.py"
            and Path(policy.__file__).resolve() == code / "src/world_reward/identity_calibration.py", "Loaded source origins differ")
    for module, name in ((binding, "bridge_frontend_bindings"), (binding.selected, "frontend_selected_assets"),
                          (infer.detector_policy, "hand_synthetic_masks"), (infer.boots, "robotap_boots_infer"),
                          (infer.boots.source, "robotap_boots_acquire")):
        require(Path(module.__file__).resolve() == code / f"infra/{name}.py", "Loaded public proof helper differs")
    for name in ("relational_motion", "prompt_selection"):
        require(Path(sys.modules["world_reward." + name].__file__).resolve() == code / f"src/world_reward/{name}.py", "Loaded feature/baseline math differs")
    helpers = (*infer.HELPERS, "infra/dexycb_identity_calibrate.py", "src/world_reward/identity_calibration.py")
    return closure(code, revision, ENTRY, helpers)


def code_identity(path):
    _, pin = binding.selected.read(path, 2_000_000, empty=True)
    return {k: pin[k] for k in ("bytes", "sha256")}


def load_arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {k: saved[k].copy() for k in saved.files}


def baseline_selection(bank, row):
    """Unchanged unique-box policy, not a nearest-hand or coverage selector."""
    detections = tuple(BoxDetection(tuple(map(float, b)), float(s))
                       for b, s in zip(bank["raw_object_boxes"], bank["raw_object_scores"]))
    kept = non_maximum_suppression(detections, 640, 480, .3, .7)
    selected, reason = _select_detection(kept, 640, 480, .3, .05)
    if selected is None: return None, reason
    indices = [i for i, c in enumerate(row["candidates"]) if c["kind"] == "object"
               and np.array_equal(bank["boxes"][i], np.asarray(selected.box, dtype=np.float64))]
    require(len(indices) == 1, "Original selected box must identify exactly one frozen proposal")
    return row["candidates"][indices[0]]["stable_id"], None


def validate_tracks(data, bank, row, frames):
    motion = {"motion_" + f.name for f in fields(RelationalMotionEvidence)}
    require(set(data) == motion | {"tracks", "tracks_256", "visible", "occlusion", "expected_dist",
            "query_points", "point_indices", "frame_index"}, "Exact raw tracking/evidence fields required")
    count = len(bank["query_points"])
    for name, shape in (("tracks", (count, frames, 2)), ("tracks_256", (count, frames, 2)),
                        ("occlusion", (count, frames)), ("expected_dist", (count, frames))):
        require(data[name].dtype == np.float32 and data[name].shape == shape and np.isfinite(data[name]).all(), "Finite full native tracks required")
    require(data["visible"].dtype == np.bool_ and data["visible"].shape == (count, frames)
        and data["query_points"].dtype == np.float64 and np.array_equal(data["query_points"], bank["query_points"])
        and data["point_indices"].dtype == data["frame_index"].dtype == np.int64
        and np.array_equal(data["point_indices"], np.arange(count, dtype=np.int64))
        and np.array_equal(data["frame_index"], bank["frame_index"]), "Original support/queries/timeline changed")
    require(np.array_equal(data["tracks"], (data["tracks_256"] * np.array([640, 480], np.float32)) / np.float32(256)), "Native coordinate conversion changed")
    groups = {k: ([], []) for k in ("hand", "object")}
    for i, candidate in enumerate(row["candidates"]):
        a, b = bank["query_offsets"][i:i + 2]
        groups[candidate["kind"]][0].append(data["tracks"][a:b]); groups[candidate["kind"]][1].append(data["visible"][a:b])
    start = bank["query_offsets"][-2]
    evidence = relational_motion_features(data["frame_index"], data["frame_index"].astype(np.float64),
        tuple(groups["hand"][0]), tuple(groups["hand"][1]), tuple(groups["object"][0]), tuple(groups["object"][1]),
        data["tracks"][start:], data["visible"][start:], image_width=640, image_height=480)
    require(all(data["motion_" + f.name].dtype == getattr(evidence, f.name).dtype
        and np.array_equal(data["motion_" + f.name], getattr(evidence, f.name), equal_nan=True)
        for f in fields(evidence)), "Saved movement evidence differs from original raw tracks")


def summarize_clip(bank, row, tracks):
    """No best-hand maximization or supported-interval selection by scores."""
    objects = [i for i, c in enumerate(row["candidates"]) if c["kind"] == "object"]
    values, support = tracks["motion_features"], tracks["motion_pair_supported"]
    require(values.dtype == np.float64 and values.shape == (*support.shape, 3)
        and support.dtype == np.bool_ and support.shape[2] == len(objects)
        and np.isfinite(values[support]).all(), "Typed supported motion evidence required")
    count = support.sum(axis=(0, 1), dtype=np.int64)
    features = np.full((len(objects), 3), np.nan)
    for j in range(len(objects)):
        if count[j]: features[j] = values[:, :, j][support[:, :, j]].mean(axis=0)
    require(np.isfinite(features[count > 0]).all(), "Feature summary overflowed")
    return dict(frame_index=bank["frame_index"].copy(), candidate_index=np.array(objects, np.int64),
                features=features, supported=count > 0, support_count=count)


def public_evidence(code, pins):
    clips, frozen, manifest = infer.public_inputs(ROOT / BASE / "inputs", pins["manifest"])
    infer.acquisition_proof(pins["acquisition"], pins["manifest"], manifest, code)
    frozen[ROOT / BASE / "report.json"] = pins["acquisition"]
    rows, sources = {}, {}
    for stage in ("masks", "tracks"):
        item = pins[stage]; revision = item["producer_revision"]
        old = ROOT / "jobs" / revision / infer.ENTRY / "code"
        sources[stage] = closure(old, revision, infer.ENTRY, infer.HELPERS)
        require(all(code_identity(code / n) == p for n, p in sources[stage]["helpers"].items()), "Native evidence helper bytes changed")
        folder = ROOT / BASE / f"identity_{stage}_{revision}"
        pin = {k: item[k] for k in ("bytes", "sha256")}; receipt = binding.pinned(folder / "report.json", pin, 2 << 20)
        expected = dict(schema="world-reward-dexycb-identity-infer-v1", stage=stage, status="pass", phase="complete",
            producer_revision=revision, script_sha256=item["script_sha256"], source_binding=sources[stage],
            public_manifest=pins["manifest"], image_id=infer.IMAGES[stage], network="none", device="cuda",
            budget_seconds=infer.BUDGETS[stage], original_rehashed_after=True, private_annotations_read=False,
            challenge_inputs_used=False, oracle_modes=[], quality_verified=False, identity_accepted=False)
        require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items())
            and item["script_sha256"] == sources[stage]["helpers"][infer.HELPERS[0]]["sha256"], "Frozen complete public native producer required")
        counts = {"detector_attempts": 24, "detector_calls": 24, "sam2_encoder_attempts": 12,
            "sam2_encoder_calls": 12, "sam2_batch_attempts": 12, "sam2_batch_calls": 12} if stage == "masks" else {
            "native_calls_attempted": 12, "native_calls_returned": 12, "native_calls_completed": 12}
        require(all(type(receipt.get(k)) is int and receipt[k] == v for k, v in counts.items()), "Full native calls required")
        model = infer.frontend_proof(code)[1] if stage == "masks" else infer.boots_proof(code)
        require(receipt.get("model_assets") == model, "Actual source/model proof changed")
        if stage == "tracks": require(receipt.get("masks_report") == {k: pins["masks"][k] for k in ("bytes", "sha256")}, "Tracking used another candidate bank")
        require(type(receipt.get("clips")) is list and len(receipt["clips"]) == 12
            and {p.name for p in folder.iterdir()} == {"report.json", ".container.cid", *[f"clip_{i:03d}.npz" for i in range(12)]}, "Exclusive all-twelve native outputs required")
        frozen[folder / "report.json"] = pin; rows[stage] = receipt["clips"]
        for clip, row in zip(clips, rows[stage]):
            require(type(row.get("clip_index")) is int and row["clip_index"] == clip["clip_index"]
                and type(row.get("frames")) is int and row["frames"] == clip["frames"]
                and row.get("file") == f"clip_{clip['clip_index']:03d}.npz", "Native clip order/frame count differs")
            path = folder / row["file"]; pin = {k: row[k] for k in ("bytes", "sha256")}
            require(binding.identity(path) == pin, "Frozen native arrays changed"); frozen[path] = pin
    arrays = []
    for clip, mr, tr in zip(clips, rows["masks"], rows["tracks"]):
        bank = load_arrays(ROOT / BASE / f"identity_masks_{pins['masks']['producer_revision']}" / mr["file"])
        track = load_arrays(ROOT / BASE / f"identity_tracks_{pins['tracks']['producer_revision']}" / tr["file"])
        infer.validate_bank(bank, mr, clip["frames"]); validate_tracks(track, bank, mr, clip["frames"])
        require(tr["query_count"] == len(bank["query_points"]) and tr["pair_supported"] == int(track["motion_pair_supported"].sum())
            and tr["camera_supported"] == int(track["motion_camera_supported"].sum()), "Full track support diagnostics differ")
        arrays.append((bank, mr, summarize_clip(bank, mr, track)))
    return clips, arrays, frozen, sources


def validate_pins(pins, private=False):
    require(type(pins) is dict and set(pins) == {"manifest", "acquisition", "masks", "tracks"} | ({"features"} if private else set()), "Exact public evidence pins required")
    for name, item in pins.items():
        require(type(item) is dict and set(item) == {"bytes", "sha256"} | ({"producer_revision", "script_sha256"} if name not in ("manifest", "acquisition") else set()), "Typed producer pins required")
        binding.validate_file_pins({name: {k: item[k] for k in ("bytes", "sha256")}}, (name,), 16 << 20)
        if name not in ("manifest", "acquisition"):
            require(type(item["producer_revision"]) is str and re.fullmatch(r"[0-9a-f]{40}", item["producer_revision"])
                and type(item["script_sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", item["script_sha256"]), "Explicit producer revision/source SHA required")


def public_features(out, clips, arrays, report):
    rows = []
    for clip, (bank, row, features) in zip(clips, arrays):
        name = f"clip_{clip['clip_index']:03d}.npz"; pin = infer.save_arrays(out / name, features)
        baseline, reason = baseline_selection(bank, row)
        rows.append(dict(clip_index=clip["clip_index"], file=name, **pin, frames=clip["frames"],
            candidate_ids=[row["candidates"][i]["stable_id"] for i in features["candidate_index"]],
            support_count=features["support_count"].tolist(), baseline_proposal=baseline, baseline_abstention=reason))
    report.update(clips=rows, all_twelve_features_frozen=True)


def frozen_features(pins, clips, arrays):
    item = pins["features"]; folder = ROOT / BASE / FOLDERS["public_features"]
    old = ROOT / "jobs" / item["producer_revision"] / ENTRY / "code"
    helpers = (*infer.HELPERS, "infra/dexycb_identity_calibrate.py", "src/world_reward/identity_calibration.py")
    source = closure(old, item["producer_revision"], ENTRY, helpers)
    require(all(code_identity(Path(__file__).parents[1] / n) == pin for n, pin in source["helpers"].items())
        and item["script_sha256"] == source["helpers"]["infra/dexycb_identity_calibrate.py"]["sha256"], "Feature math source changed")
    pin = {k: item[k] for k in ("bytes", "sha256")}; receipt = binding.pinned(folder / "report.json", pin, 2 << 20)
    expected = dict(stage="public_features", status="pass", phase="complete", producer_revision=item["producer_revision"],
        script_sha256=item["script_sha256"], source_binding=source, recipe=RULE, original_rehashed_after=True,
        private_annotations_read=False, all_twelve_features_frozen=True)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items())
        and receipt.get("pins") == {k: v for k, v in pins.items() if k != "features"}
        and len(receipt.get("clips", [])) == 12, "All public features must be sealed before private reads")
    frozen = {folder / "report.json": pin}
    for clip, (bank, row, expected_arrays), record in zip(clips, arrays, receipt["clips"]):
        name = f"clip_{clip['clip_index']:03d}.npz"; path = folder / name
        filepin = {k: record[k] for k in ("bytes", "sha256")}; require(binding.identity(path) == filepin, "Feature bytes changed")
        data = load_arrays(path)
        require(set(data) == FEATURE_KEYS and all(data[k].dtype == expected_arrays[k].dtype
            and np.array_equal(data[k], expected_arrays[k], equal_nan=True) for k in FEATURE_KEYS)
            and record["clip_index"] == clip["clip_index"] and record["file"] == name
            and record["frames"] == clip["frames"] and record["support_count"] == data["support_count"].tolist()
            and record["candidate_ids"] == [row["candidates"][i]["stable_id"] for i in data["candidate_index"]]
            and (record["baseline_proposal"], record["baseline_abstention"]) == baseline_selection(bank, row), "Frozen feature recipe/baseline differs")
        frozen[path] = filepin
    require({p.name for p in folder.iterdir()} == {"report.json", *[f"clip_{i:03d}.npz" for i in range(12)]}, "Exclusive complete features required")
    return frozen


def private_anchor(clip, acquisition, frozen):
    """Decode ONLY seg and two target meta fields, never other NPZ payloads."""
    import yaml
    prefix = f"{clip['subject']}/{clip['sequence']}"
    names = (prefix + "/meta.yml", prefix + f"/{infer.CAMERA}/labels_000000.npz")
    paths = [ROOT / BASE / "eval_private" / n for n in names]
    for path, name in zip(paths, names):
        pin = acquisition["retained_files"].get(name); require(type(pin) is dict and binding.identity(path, 64 << 20) == pin, "Original anchor bytes required")
        frozen[path] = pin
    loader = yaml.SafeLoader(paths[0].read_bytes())
    try:
        node = loader.get_single_node()
        require(isinstance(node, yaml.MappingNode), "Native YAML metadata must be one flat mapping")
        nodes, seen = [node], set()
        while nodes:
            current = nodes.pop()
            if id(current) in seen: continue
            seen.add(id(current))
            require(current.tag in yaml.SafeLoader.yaml_constructors and current.tag is not None,
                    "Only safe native YAML tags allowed")
            if isinstance(current, yaml.MappingNode): nodes.extend(n for pair in current.value for n in pair)
            elif isinstance(current, yaml.SequenceNode): nodes.extend(current.value)
        fields_by_name = {}
        for keynode, valuenode in node.value:
            require(isinstance(keynode, yaml.ScalarNode) and keynode.tag == "tag:yaml.org,2002:str"
                    and keynode.value not in fields_by_name, "Native YAML keys must be unique text")
            fields_by_name[keynode.value] = valuenode
        require({"ycb_ids", "ycb_grasp_ind"} <= set(fields_by_name), "Native target metadata missing")
        # Unknown values remain syntax nodes, not decoded MANO/camera/pose data.
        meta = {key: loader.construct_object(fields_by_name[key], deep=True) for key in ("ycb_ids", "ycb_grasp_ind")}
    finally: loader.dispose()
    require(type(meta.get("ycb_ids")) is list and type(meta.get("ycb_grasp_ind")) is int,
            "Native target metadata type differs")
    target = policy.target_object_id(meta["ycb_ids"], meta["ycb_grasp_ind"])
    with np.load(paths[1], allow_pickle=False) as archive:
        require("seg" in archive.files, "Native segmentation field missing")
        seg = archive["seg"].copy()
    require(seg.dtype == np.uint8 and seg.shape == (480, 640), "Native initial segmentation grid/dtype differs")
    require(all(binding.identity(p, 64 << 20) == frozen[p] for p in paths), "Original anchor changed while decoding")
    return seg, target


def score_clip(model, clip, arrays):
    bank, row, data = arrays
    scores = policy.score_identity_candidates(model, data["features"], data["supported"])
    proposals = tuple(row["candidates"][i]["stable_id"] for i in data["candidate_index"])
    decision = policy.decide_identity(scores, data["supported"], proposals, minimum_gap=0.)
    baseline, baseline_reason = baseline_selection(bank, row)
    return dict(clip_index=clip["clip_index"], learned_proposal=decision.proposal_id,
        learned_reason=decision.reason, raw_best_score=decision.raw_best_score, raw_gap=decision.raw_score_gap,
        ranked_scores=[list(r) for r in decision.ranked_scores], baseline_proposal=baseline, baseline_abstention=baseline_reason)


def anchor_metrics(clip, arrays, predicted, acquisition, frozen):
    seg, target = private_anchor(clip, acquisition, frozen)
    bank, row, _ = arrays; result = dict(clip_index=clip["clip_index"])
    for method in ("learned", "baseline"):
        selected = predicted[method + "_proposal"]
        indices = [i for i, c in enumerate(row["candidates"]) if c["stable_id"] == selected]
        require(len(indices) == (0 if selected is None else 1), "Selection must reference one frozen automatic proposal")
        mask = None if not indices else bank["initial_masks"][indices[0]]
        timeline = policy.evaluate_identity_timeline(np.array([0], np.int64), (mask,), (seg,), target)
        result[method] = {k: bool(getattr(timeline, k)[0]) for k in ("correct_identity", "reciprocal_coverage", "wrong_id", "unknown_mask", "missing", "target_visible")}
        result[method]["success"] = result[method]["correct_identity"] and result[method]["reciprocal_coverage"]
    return result


def private_calibration(out, clips, arrays, report, frozen, pins):
    acquisition = binding.pinned(ROOT / BASE / "report.json", pins["acquisition"], 16 << 20)
    x, support, labels, clipkeys, identities = [], [], [], [], []
    for index in (6, 8, 10):
        clip, (bank, row, data) = clips[index], arrays[index]
        seg, target = private_anchor(clip, acquisition, frozen)
        for i, feature, ok in zip(data["candidate_index"], data["features"], data["supported"]):
            assigned = policy.majority_identity(bank["initial_masks"][i], seg)
            x.append(feature); support.append(ok); labels.append(-1 if assigned.object_id is None else int(assigned.object_id == target))
            clipkeys.append(str(index)); identities.append("private-class:" + str(assigned.object_id) if assigned.object_id is not None else "unknown:" + row["candidates"][i]["stable_id"])
    model = policy.fit_identity_logistic(np.array(x, np.float64), np.array(support, bool), np.array(labels, np.int64), tuple(clipkeys), tuple(identities))
    saved = dict(mean=model.mean, scale=model.scale, constant_features=model.constant_features,
                 coefficients=model.coefficients, intercept=np.array(model.intercept, np.float64))
    modelpin = infer.save_arrays(out / "model.npz", saved); frozen[out / "model.npz"] = modelpin
    reread = load_arrays(out / "model.npz")
    require(set(reread) == set(saved) and all(np.array_equal(reread[k], v) for k, v in saved.items()), "Saved fitted model differs")
    report.update(model=modelpin, model_frozen_before_decision=True, fit_clips=[6, 8, 10],
        solver=dict(l2=model.l2, iterations=model.iterations, gradient_inf_norm=model.gradient_inf_norm, objective=model.objective))
    # No decision/evaluation annotation is read until these proposals are sealed.
    predicted = [score_clip(model, clip, data) for clip, data in zip(clips, arrays)]
    path = out / "predictions.json"
    with path.open("x") as stream:
        json.dump(predicted, stream, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444); predictionpin = binding.identity(path); frozen[path] = predictionpin
    report.update(public_score_predictions=predictionpin, predictions_frozen_before_decision=True)
    decision = [anchor_metrics(clips[i], arrays[i], predicted[i], acquisition, frozen) for i in (7, 9, 11)]
    gate = all(r["learned"]["success"] for r in decision) and sum(r["learned"]["wrong_id"] for r in decision) <= sum(r["baseline"]["wrong_id"] for r in decision)
    report.update(decision_metrics=decision, decision_gate=bool(gate), evaluation_subject_read=False)
    if gate:
        evaluation = [anchor_metrics(clips[i], arrays[i], predicted[i], acquisition, frozen) for i in range(6)]
        gain = sum(r["learned"]["success"] for r in evaluation) - sum(r["baseline"]["success"] for r in evaluation)
        report.update(evaluation_metrics=evaluation, evaluation_subject_read=True, evaluation_success_gain=gain,
            hypothesis_supported=bool(gain >= 1 and sum(r["learned"]["wrong_id"] for r in evaluation) <= sum(r["baseline"]["wrong_id"] for r in evaluation)))
    else: report["hypothesis_supported"] = False


def reserve_output(out):
    """Only the source-bound wrapper may supply an empty, narrowly RW-mounted leaf."""
    flag = os.environ.get("WR_DEXYCB_CPU_OUTPUT_RESERVED")
    if flag is None:
        out.mkdir(mode=0o700)
        return
    require(flag == "1" and sys.platform == "linux" and os.geteuid() == 0
        and os.environ.get("WR_CPU_IMAGE_ID") == binding.IMAGE
        and re.fullmatch(r"[0-9a-f]{64}", os.environ.get("WR_HOST_PROOF_SHA256", ""))
        and os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
        and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}, "Verified restricted CPU wrapper required")
    s = out.lstat()
    require(stat.S_ISDIR(s.st_mode) and s.st_uid == 0 and s.st_mode & 0o777 == 0o700
        and not tuple(out.iterdir()), "Exclusive empty root-owned output reservation required")


def run(stage, code, revision, pins):
    require(stage in FOLDERS, "Explicit public/private CPU stage required")
    validate_pins(pins, stage == "private_calibration")
    started = time.monotonic(); source = source_binding(code, revision)
    clips, arrays, frozen, historical = public_evidence(code, pins)
    if stage == "private_calibration": frozen.update(frozen_features(pins, clips, arrays))
    else: require(not (ROOT / BASE / "eval_private").exists(), "Public feature container must not mount private annotations")
    out = binding.canonical(ROOT / BASE / FOLDERS[stage]); reserve_output(out)
    report = dict(stage=stage, status="fail", phase="frozen_public_audit", producer_revision=revision,
        script_sha256=source["helpers"]["infra/dexycb_identity_calibrate.py"]["sha256"], source_binding=source,
        pins=pins, recipe=RULE, historical_sources=historical, budget_seconds=BUDGET, network="none", device="cpu",
        private_annotations_read=False, initial_identity_only=True, full_timeline_identity_verified=False,
        contact_verified=False, geometry_verified=False, adoption=False, training_overlap_verified=False,
        low_statistical_power=True, original_rehashed_after=False)
    try:
        if stage == "public_features":
            public_features(out, clips, arrays, report)
            frozen.update({out / row["file"]: {k: row[k] for k in ("bytes", "sha256")} for row in report["clips"]})
        else:
            binding.recheck(frozen)
            report.update(private_annotations_read=True, phase="private_fit_and_initial_identity_gate")
            private_calibration(out, clips, arrays, report, frozen, pins)
        report.update(status="pass", phase="complete")
    except BaseException as error:
        report.update(error_type=type(error).__name__, error="CPU initial-identity stage failed at recorded phase")
        raise
    finally:
        try:
            binding.recheck(frozen)
            require(source_binding(code, revision) == source and public_evidence(code, pins)[3] == historical, "Public/source/assets changed")
            expected = ({"report.json", *[f"clip_{i:03d}.npz" for i in range(12)]} if stage == "public_features"
                        else {"report.json", "model.npz", "predictions.json"})
            if report["status"] == "pass": require({p.name for p in out.iterdir()} | {"report.json"} == expected, "Exclusive complete CPU outputs required")
            report["original_rehashed_after"] = True
            require(time.monotonic() - started <= BUDGET, "Frozen inclusive CPU120s budget exceeded")
        except BaseException:
            report.update(status="fail", original_rehashed_after=False, integrity_error="Bounded postverify failed")
            raise
        finally:
            report["elapsed_seconds"] = time.monotonic() - started
            path = out / "report.json"
            with path.open("x") as stream:
                json.dump(report, stream, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
            path.chmod(0o444)
    return report


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--stage", choices=tuple(FOLDERS), required=True)
    for name in ("manifest", "acquisition", "masks", "tracks", "features"):
        required = name != "features"
        p.add_argument(f"--{name}-sha256", required=required); p.add_argument(f"--{name}-bytes", type=int, required=required)
        if name not in ("manifest", "acquisition"):
            p.add_argument(f"--{name}-producer-revision", required=required); p.add_argument(f"--{name}-script-sha256", required=required)
    return p


def main(argv=None):
    args = parser().parse_args(argv); code, revision = Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    require(sys.platform == "linux" and os.environ.get("WR_ROOT") == str(ROOT)
        and os.geteuid() == 0 and os.environ.get("WR_CPU_IMAGE_ID") == binding.IMAGE
        and os.environ.get("WR_DEXYCB_CPU_OUTPUT_RESERVED") == "1"
        and re.fullmatch(r"[0-9a-f]{64}", os.environ.get("WR_HOST_PROOF_SHA256", ""))
        and os.environ.get("CUDA_VISIBLE_DEVICES") in ("", "-1")
        and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}, "Offline CPU-only Azure container required")
    pins = {}
    for name in ("manifest", "acquisition", "masks", "tracks", "features"):
        item = {key: getattr(args, name + "_" + key) for key in ("bytes", "sha256")}
        if name not in ("manifest", "acquisition"):
            item.update({key: getattr(args, name + "_" + key) for key in ("producer_revision", "script_sha256")})
        if name == "features" and args.stage == "public_features":
            require(all(v is None for v in item.values()), "Public stage cannot consume previous features"); continue
        pins[name] = item
    def expired(*_): raise TimeoutError("Frozen inclusive CPU120s budget exceeded")
    previous = signal.signal(signal.SIGALRM, expired); signal.alarm(BUDGET)
    try:
        result = run(args.stage, code, revision, pins)
        print(json.dumps({k: result[k] for k in ("stage", "status", "elapsed_seconds", "adoption")}))
    finally: signal.alarm(0); signal.signal(signal.SIGALRM, previous)


if __name__ == "__main__": main()
