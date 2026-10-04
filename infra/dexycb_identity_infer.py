"""Azure-only GroundingDINO/SAM2 frame-zero banks and native BootsTAPIR tracks.

No target identity, private labels, contact, 3D or performance is inferred here.
Original timestamps are unavailable: motion uses original frame coordinates,
NOT seconds or an assumed frame rate. All frame-zero proposals remain present.
"""
from __future__ import annotations

import argparse
from dataclasses import fields
import hashlib
import importlib.util
from importlib import metadata
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

# A direct python -I invocation must not resolve helpers from the working directory.
if __name__ == "__main__":
    _code = Path(os.environ["WR_CODE"])
    _revision = os.environ["WR_CODE_REVISION"]
    if (not re.fullmatch(r"[0-9a-f]{40}", _revision)
            or _code != Path("/srv/scenesmith/world-reward/jobs") / _revision / "run_dexycb_identity_infer/code"
            or _code.resolve() != _code or Path(__file__).resolve() != _code / "infra/dexycb_identity_infer.py"
            or any(p.is_symlink() for p in (_code, *_code.parents))):
        raise ValueError("Actual immutable identity producer required before helper imports")
    sys.path[:0] = [str(_code / "infra"), str(_code / "src")]

import bridge_frontend_bindings as binding
import hand_synthetic_masks as detector_policy
import robotap_boots_infer as boots

ROOT, BASE, ENTRY = binding.ROOT, "validation/dexycb_identity_v1", "run_dexycb_identity_infer"
BUDGETS = {"masks": 300, "tracks": 7200}
IMAGES = {"masks": binding.IMAGE, "tracks": boots.IMAGE}
SUBJECTS = ("20200709-subject-01", "20200813-subject-02")
INDICES, CAMERA = (0, 16, 32, 48, 64, 80), "836212060125"
HELPERS = ("infra/dexycb_identity_infer.py", "infra/run_dexycb_identity_infer.sh",
    "infra/bridge_frontend_bindings.py", "infra/frontend_selected_assets.py", "infra/frontend_sam2_kernel_gate.py",
    "infra/hand_synthetic_masks.py", "infra/robotap_boots_infer.py", "infra/robotap_boots_acquire.py",
    binding.CONFIG, "configs/frontend_asset_archive_pins.json", "configs/robotap_boots_protocol.json",
    "configs/dexycb_identity_protocol.json",
    "src/world_reward/__init__.py", "src/world_reward/automatic_candidate_bank.py",
    "src/world_reward/prompt_selection.py", "src/world_reward/relational_motion.py")
RUNTIME_REV = "f3cfde1992c6cfc9a6504b393d8a1f24735dc94d"
RUNTIME_PIN = dict(bytes=3051, sha256="794b6e5313e992c2c86610f855f4a86dc8c9a994ede37c22e33093e6ad9fc5e7")
RUNTIME_SCRIPT_SHA = "5b2974b3e36ced5b8cbf67f6be02d6e9ee6e773403cd9575397f4a940206de9c"
require = binding.require


def source_binding(code, revision):
    require(binding.canonical(code) == ROOT / "jobs" / revision / ENTRY / "code"
        and Path(__file__).resolve() == code / HELPERS[0], "Actual new producer, not an old approved entrypoint")
    require(binding.identity(code / "infra/frontend_sam2_kernel_gate.py") == binding.KERNEL_SOURCE_PIN,
        "Original measured proof helper required before import")
    require(set(HELPERS).issubset({str(p.relative_to(code)) for p in code.rglob("*") if p.is_file()}),
        "Required public/model/native-math source closure required")
    for module, name in ((binding, "bridge_frontend_bindings"), (binding.selected, "frontend_selected_assets"),
                         (detector_policy, "hand_synthetic_masks"), (boots, "robotap_boots_infer"),
                         (boots.source, "robotap_boots_acquire")):
        require(Path(module.__file__).resolve() == code / f"infra/{name}.py", "Mounted helper origin differs")
    for name in ("automatic_candidate_bank", "relational_motion", "prompt_selection"):
        spec = importlib.util.find_spec("world_reward." + name)
        require(spec is not None and Path(spec.origin).resolve() == code / f"src/world_reward/{name}.py",
            "Actual reusable math source required")
    return binding.kernel_helper().closure(code, revision, ENTRY, HELPERS)


def frontend_proof(code, *, live=False):
    """Authenticate actual original receipts through lower primitives, no spoof."""
    kernel = binding.kernel_helper()
    build = binding.pinned(binding.BUILD_REPORT, binding.BUILD_PIN)
    receipt = binding.pinned(binding.KERNEL_REPORT, binding.KERNEL_PIN)
    old = kernel.closure(binding.BUILD_CODE, binding.BUILD_REV, "run_frontend_grounding_build", kernel.BUILD_HELPERS)
    gate = kernel.closure(binding.KERNEL_CODE, binding.KERNEL_REV, "run_frontend_sam2_kernel_gate",
        ("infra/frontend_sam2_kernel_gate.py", "infra/run_frontend_sam2_kernel_gate.sh", binding.CONFIG))
    require(old["helpers"]["infra/frontend_grounding_build.py"]["sha256"] == binding.BUILD_SHA
        and gate["helpers"]["infra/frontend_sam2_kernel_gate.py"] == binding.KERNEL_SOURCE_PIN
        and build.get("source_binding") == old and receipt.get("source_binding") == gate
        and binding.identity(code / binding.CONFIG) == old["helpers"][binding.CONFIG] == gate["helpers"][binding.CONFIG],
        "Measured original build/kernel/source/config ancestry differs")
    expected = dict(schema="world_reward.frontend_grounding_build.v6", stage="frontend_grounding_build", status="pass",
        phase="complete", producer_revision=binding.BUILD_REV, offline_build_exit_code=0, child_probe_exit_code=0,
        parent_unchanged_verified=True, source_rechecked_before_and_after=True, extension_import_verified=True,
        CUDA_execution_verified=False, replica_ready=False, license_eligibility_verified=False, training_overlap_verified=False)
    require(all(type(build.get(k)) is type(v) and build[k] == v for k, v in expected.items()), "Actual CPU build required")
    expected = dict(schema="world_reward.frontend_sam2_kernel_gate.v1", stage="frontend_sam2_kernel_gate", status="pass",
        producer_revision=binding.KERNEL_REV, script_sha256=binding.KERNEL_SOURCE_PIN["sha256"], image_id=binding.IMAGE,
        models_loaded=False, challenge_data_read=False, CUDA_operator_execution_verified=True,
        build_report_identity=binding.BUILD_PIN, replica_ready=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items())
        and receipt.get("operator_source_identities", {}).get("extension") == binding.EXTENSION_PIN,
        "Measured original native CUDA operator required")
    child, parent = build["child_image"], build["parent_image"]
    owner = hashlib.sha256((binding.BUILD_REV + old["closure_sha256"]).encode()).hexdigest()
    require(child["Id"] == binding.IMAGE and parent["Id"] == kernel.BASE and child["Architecture"] == "amd64"
        and child["Os"] == "linux" and len(parent["RootFS"]["Layers"]) == 44
        and child["RootFS"]["Layers"][:44] == parent["RootFS"]["Layers"] and build.get("owner") == owner,
        "Actual source-owned image/rootfs differs")
    probe = build.get("private_child_probe_log", {})
    require(probe.get("relative_path") == "results/frontend-grounding-build-v6/child-CPU-probe.log"
        and binding.identity(ROOT / probe["relative_path"], 128 * 1024) == {k: probe[k] for k in ("bytes", "sha256")},
        "Original CPU import log differs")
    proof = dict(child_image=child, parent_image=parent, owner=owner, source_files=build["source_files"],
        selected_contract=binding.selected.load_contract(ROOT, binding.MANIFEST))
    if live: kernel.validate_live_image(proof)
    assets = binding.selected_assets(proof["selected_contract"], detector_policy.ASSETS)
    acquisition = binding.strict_json((binding.DEST / "results/weights-acquisition.json").read_bytes())
    for repo, revision, folder in (("IDEA-Research/grounding-dino-base", detector_policy.DETECTOR_REVISION, "grounding_dino"),
                                  ("facebook/sam2.1-hiera-large", detector_policy.SAM2_REVISION, "sam2")):
        rows = [r for r in acquisition.get("assets", []) if r.get("repo_id") == repo]
        require(len(rows) == 1 and rows[0].get("revision") == revision
            and rows[0].get("path") == str(ROOT / "weights" / folder), "Original selected model revision differs")
    return proof, assets


def boots_proof(code, *, live=False):
    """Reuse only pinned native sources/checkpoint and real CPU verification."""
    protocol = boots.source.read_protocol(code / "configs/robotap_boots_protocol.json")
    native = {name: binding.identity(ROOT / boots.BASE / "assets/tapnet_source" / name)
              for name in protocol["source"]["files"]}
    require(native == {n: {k: r[k] for k in ("bytes", "sha256")} for n, r in protocol["source"]["files"].items()},
        "Pinned native tracking source differs")
    checkpoint = binding.identity(ROOT / boots.BASE / "assets" / protocol["checkpoint"]["file"])
    require(checkpoint == {k: protocol["checkpoint"][k] for k in ("bytes", "sha256")}, "Pinned tracking checkpoint differs")
    runtime = binding.pinned(ROOT / f"results/bootstapir-runtime-verify-{RUNTIME_REV}/report.json", RUNTIME_PIN)
    expected = dict(stage="bootstapir_runtime_verify", status="pass", phase="complete", producer_revision=RUNTIME_REV,
        child_image_id=boots.IMAGE, native_source_revision=protocol["source"]["revision"], original_build_status="fail",
        original_build_failure_preserved=True, source_rehashed_after=True, gpu_execution=False,
        checkpoint_read=False, rgb_or_labels_read=False)
    require(all(type(runtime.get(k)) is type(v) and runtime[k] == v for k, v in expected.items())
        and runtime.get("native_sources") == native, "Original independent native CPU verification differs")
    original = ROOT / "jobs" / RUNTIME_REV / "run_bootstapir_runtime_verify/code"
    helpers = {n: binding.identity(original / n, 2_000_000) for n in runtime.get("source_helpers", {})}
    require(helpers == runtime.get("source_helpers")
        and helpers.get("infra/run_bootstapir_runtime_verify.sh", {}).get("sha256") == RUNTIME_SCRIPT_SHA
        and (original.parent / "revision").read_bytes() == (RUNTIME_REV + "\n").encode(), "Actual CPU verifier source differs")
    cpu = runtime.get("cpu_import", {})
    require(cpu.get("versions") == {"dm-tree": "0.1.10", "einshape": "1.0", "absl-py": "2.5.0", "attrs": "26.1.0", "wrapt": "1.17.3"}
        and (cpu.get("python"), cpu.get("torch"), cpu.get("numpy")) == ("3.11", "2.5.1+cu124", "1.26.3")
        and all(cpu.get(k) is True for k in ("tree_cpu_verified", "source_native_einshape_cpu_verified",
            "native_bilinear_cpu_verified", "native_tapir_modules_imported"))
        and cpu.get("models_instantiated") is False and cpu.get("cuda_initialized") is False,
        "Actual native CPU dependency/API verification differs")
    if live:
        raw = binding.kernel_helper().command(["docker", "image", "inspect", boots.IMAGE, "--format", "{{json .Id}}"])
        require(binding.strict_json(raw) == boots.IMAGE, "Actual immutable tracking image differs")
    return dict(runtime=RUNTIME_PIN, native_sources=native, checkpoint=checkpoint, image_id=boots.IMAGE)


def public_inputs(directory, manifest_pin):
    """Read exclusively RGB and public acquisition metadata, never private values."""
    directory = binding.canonical(directory)
    manifest = binding.pinned(directory / "manifest.json", manifest_pin, 16 << 20)
    require(set(manifest) == {"schema", "license", "sequences", "images", "timestamps_available",
        "frame_count_evidence", "source_archives", "training_overlap_verified", "challenge_overlap_verified"}
        and manifest["schema"] == "world-reward-dexycb-identity-rgb-v1" and manifest["license"] == "CC-BY-NC-4.0"
        and all(manifest[k] is False for k in ("timestamps_available", "training_overlap_verified", "challenge_overlap_verified"))
        and manifest["frame_count_evidence"] == "all_RGB_and_label_member_headers_no_meta_values",
        "Original public RGB-only manifest required; no fabricated timestamps")
    sequences, images = manifest["sequences"], manifest["images"]
    require(type(sequences) is list and len(sequences) == 12 and type(images) is list, "Frozen full twelve-clip cohort required")
    require(type(manifest["source_archives"]) is dict and set(manifest["source_archives"]) == set(SUBJECTS), "Original archive pins required")
    binding.validate_file_pins(manifest["source_archives"], SUBJECTS, 13_000_000_000)
    clips, frozen, position = [], {directory / "manifest.json": manifest_pin}, 0
    for clip, sequence in enumerate(sequences):
        subject, index = SUBJECTS[clip // 6], INDICES[clip % 6]
        partition = "blind_evaluation" if clip < 6 else ("fit" if index in (0, 32, 64) else "decision")
        require(type(sequence) is dict and set(sequence) == {"subject", "sequence", "sequence_lex_index", "camera", "frames", "partition"}
            and sequence["subject"] == subject and type(sequence["sequence_lex_index"]) is int
            and sequence["sequence_lex_index"] == index and sequence["camera"] == CAMERA and sequence["partition"] == partition
            and type(sequence["frames"]) is int and sequence["frames"] >= 2
            and type(sequence["sequence"]) is str and re.fullmatch(r"[0-9]{8}_[0-9]{6}", sequence["sequence"]),
            "Original preselected sequence/timeline differs")
        rows = images[position:position + sequence["frames"]]; position += sequence["frames"]
        require(len(rows) == sequence["frames"], "Full original clip missing RGB frames")
        checked = []
        for frame, row in enumerate(rows):
            name = f"subject_{clip // 6 + 1:02d}_sequence_{index:03d}_frame_{frame:06d}.jpg"
            require(type(row) is dict and set(row) == {"file", "bytes", "sha256", "width", "height", "subject",
                "sequence", "camera", "frame_position", "source_frame_id"} and row["file"] == name
                and all(type(row[k]) is int and row[k] == value for k, value in
                    (("width", 640), ("height", 480), ("frame_position", frame), ("source_frame_id", frame)))
                and all(row[k] == sequence[k] for k in ("subject", "sequence", "camera")), "Original RGB indices/grid differs")
            pin = {k: row[k] for k in ("bytes", "sha256")}
            binding.validate_file_pins({name: pin}, (name,), 64 << 20)
            require(binding.identity(directory / name, 64 << 20) == pin, "Original public RGB bytes differ")
            frozen[directory / name] = pin; checked.append({**row, "path": directory / name})
        clips.append(dict(clip_index=clip, **sequence, records=checked))
    require(position == len(images) and {p.name for p in directory.iterdir()} == {p.name for p in frozen},
        "Exactly all full original RGB frames, no auxiliary/private files")
    return clips, frozen, manifest


def acquisition_proof(pin, manifest_pin, manifest, code):
    """Host only: bind acquired RGB to original source without opening labels."""
    report = binding.pinned(ROOT / BASE / "report.json", pin, 16 << 20)
    expected = dict(stage="external_dexycb_identity_rgb_private_byte_acquisition", status="pass", phase="complete",
        sequences=12, annotation_values_parsed=False, private_values_interpreted=False, inference_performed=False,
        gpu_used=False, source_rehashed_after=True, public_manifest=manifest_pin, archive_proofs=manifest["source_archives"])
    require(all(type(report.get(k)) is type(v) and report[k] == v for k, v in expected.items())
        and report.get("source_before") == report.get("source_after"), "Successful actual byte acquisition required")
    revision = report.get("producer_revision")
    require(type(revision) is str and re.fullmatch(r"[0-9a-f]{40}", revision), "Original acquisition producer required")
    original = ROOT / "jobs" / revision / "run_dexycb_acquire/code"
    for name, wanted in report["source_before"].items():
        path = binding.canonical(Path(name))
        require(path.is_relative_to(original) or path in (original.parent / "revision", original.parent / "source-sha256",
            ROOT / "vendor/research/dexycb_identity_v1/publisher.html", ROOT / "vendor/research/dexycb_identity_v1/dex_ycb.py"),
            "Only actual acquisition code/primary text may be read, never private labels")
        if path.name in ("revision", "source-sha256"):
            raw, actual = binding.kernel_helper().read(path, 100, readonly=False)
            require(actual == wanted and (raw == (revision + "\n").encode() if path.name == "revision"
                else bool(re.fullmatch(b"[0-9a-f]{64}\n", raw))), "Actual acquisition marker differs")
        else:
            _, actual = binding.selected.read(path, 2_000_000, empty=True)
            require({k: actual[k] for k in ("bytes", "sha256")} == wanted, "Actual acquisition source differs")
    require(report["source_before"].get(str(original / "infra/dexycb_acquire.py"), {}).get("sha256") == report.get("script_sha256")
        and report["source_before"].get(str(original / "configs/dexycb_identity_protocol.json"))
            == binding.identity(code / "configs/dexycb_identity_protocol.json")
        and report.get("frames") == len(manifest["images"])
        and all(report.get("retained_files", {}).get(r["file"]) == {k: r[k] for k in ("bytes", "sha256")}
            for r in manifest["images"]), "Acquisition source/public-file ancestry differs")
    return pin


def read_rgb(record):
    import numpy as np
    from PIL import Image
    pin = {k: record[k] for k in ("bytes", "sha256")}
    require(binding.identity(record["path"], 64 << 20) == pin, "Original RGB changed before decode")
    with Image.open(record["path"]) as image:
        require(image.format == "JPEG" and image.mode == "RGB" and image.size == (record["width"], record["height"]),
            "Original RGB JPEG/grid required")
        rgb = np.asarray(image).copy()
    require(rgb.dtype == np.uint8 and rgb.shape == (record["height"], record["width"], 3)
        and binding.identity(record["path"], 64 << 20) == pin, "RGB decode/source changed")
    return rgb


def bank_arrays(bank, frames):
    """Typed ragged query offsets retain every proposal, including Q=0."""
    import numpy as np
    h, w = bank.image_size
    groups = [c.query_points for c in bank.candidates] + [bank.background_query_points]
    arrays = dict(frame_index=np.arange(frames, dtype=np.int64), image_size=np.array((h, w), np.int64),
        initial_masks=np.stack([c.mask for c in bank.candidates]) if bank.candidates else np.empty((0, h, w), bool),
        boxes=np.array([c.box for c in bank.candidates], np.float64).reshape(-1, 4),
        detector_scores=np.array([c.detector_score for c in bank.candidates], np.float64),
        sam2_scores=np.array([c.sam2_score for c in bank.candidates], np.float64),
        query_points=np.concatenate(groups), query_offsets=np.cumsum([0] + [len(q) for q in groups], dtype=np.int64))
    for raw in bank.detector_records:
        arrays[f"raw_{raw.kind}_boxes"], arrays[f"raw_{raw.kind}_scores"] = raw.boxes, raw.scores
    rows = [dict(kind=c.kind, stable_id=c.stable_id, query_count=len(c.query_points), mask_pixels=int(c.mask.sum()))
            for c in bank.candidates]
    return arrays, rows


def save_arrays(path, arrays):
    import numpy as np
    require(all(isinstance(a, np.ndarray) and a.dtype.kind != "O" for a in arrays.values()), "No object/pickle serialization")
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "wb") as stream:
        np.savez_compressed(stream, **arrays); stream.flush(); os.fsync(stream.fileno())
    path.chmod(0o444)
    return binding.identity(path)


def observe_masks(clips, out, report, persist, *, detect, predictor):
    from world_reward.automatic_candidate_bank import build_automatic_candidate_bank
    for clip in clips:
        report.update(phase="automatic_candidate_bank", current_clip=clip["clip_index"]); persist()
        bank = build_automatic_candidate_bank(read_rgb(clip["records"][0]), detect, detect, predictor)
        arrays, candidates = bank_arrays(bank, clip["frames"])
        name = f"clip_{clip['clip_index']:03d}.npz"; pin = save_arrays(out / name, arrays)
        report["clips"].append(dict(clip_index=clip["clip_index"], frames=clip["frames"], file=name, **pin,
            candidates=candidates, background_query_count=len(bank.background_query_points)))
        persist()
        require({c.kind for c in bank.candidates} == {"hand", "object"}, "Automatic required-class coverage failed; no prompt/nearest fallback")
    require(len(report["clips"]) == 12, "All twelve original candidate banks required")


def load_banks(directory, pin, clips, revision, manifest_pin, *, decode=True):
    receipt = binding.pinned(directory / "report.json", pin, 2 << 20)
    expected = dict(schema="world-reward-dexycb-identity-infer-v1", stage="masks", status="pass", phase="complete",
        producer_revision=revision, image_id=binding.IMAGE, public_manifest=manifest_pin, original_rehashed_after=True,
        private_annotations_read=False, challenge_inputs_used=False, oracle_modes=[], quality_verified=False,
        detector_calls=24, sam2_encoder_calls=12, sam2_batch_calls=12, budget_seconds=300)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k, v in expected.items())
        and receipt.get("script_sha256") == binding.identity(Path(__file__))["sha256"]
        and receipt.get("source_binding") == source_binding(ROOT / "jobs" / revision / ENTRY / "code", revision)
        and len(receipt.get("clips", [])) == len(clips) == 12, "Pinned completed all-candidate masks stage required")
    banks, frozen = [], {directory / "report.json": pin}
    for clip, row in zip(clips, receipt["clips"]):
        name = f"clip_{clip['clip_index']:03d}.npz"; file_pin = {k: row[k] for k in ("bytes", "sha256")}
        require(row["clip_index"] == clip["clip_index"] and row["frames"] == clip["frames"] and row["file"] == name
            and binding.identity(directory / name) == file_pin, "Exact original bank/clip bytes required")
        if decode:
            import numpy as np
            with np.load(directory / name, allow_pickle=False) as saved: arrays = {k: saved[k].copy() for k in saved.files}
            validate_bank(arrays, row, clip["frames"]); banks.append((arrays, row))
        frozen[directory / name] = file_pin
    require({p.name for p in directory.iterdir()} == {"report.json", ".container.cid", *[f"clip_{i:03d}.npz" for i in range(12)]},
        "Only complete sealed automatic banks may feed tracking")
    return banks, frozen


def validate_bank(arrays, row, frames):
    import numpy as np
    names = {"frame_index", "image_size", "initial_masks", "boxes", "detector_scores", "sam2_scores", "query_points",
             "query_offsets", "raw_hand_boxes", "raw_hand_scores", "raw_object_boxes", "raw_object_scores"}
    n = len(row["candidates"])
    require(set(arrays) == names and arrays["frame_index"].dtype == np.int64
        and np.array_equal(arrays["frame_index"], np.arange(frames, dtype=np.int64))
        and arrays["image_size"].dtype == np.int64 and arrays["image_size"].tolist() == [480, 640], "Original bank grid/timeline required")
    require(arrays["initial_masks"].dtype == np.bool_ and arrays["initial_masks"].shape == (n, 480, 640)
        and arrays["query_offsets"].dtype == np.int64 and arrays["query_offsets"].shape == (n + 2,)
        and arrays["query_offsets"][0] == 0 and np.all(np.diff(arrays["query_offsets"]) >= 0), "Typed complete ragged bank required")
    for name, shape in (("boxes", (n, 4)), ("detector_scores", (n,)), ("sam2_scores", (n,)),
                        ("query_points", (int(arrays["query_offsets"][-1]), 3))):
        require(arrays[name].dtype == np.float64 and arrays[name].shape == shape and np.isfinite(arrays[name]).all(), "Finite original bank arrays required")
    for kind in ("hand", "object"):
        b, s = arrays[f"raw_{kind}_boxes"], arrays[f"raw_{kind}_scores"]
        require(b.ndim == 2 and b.shape[1:] == (4,) and s.shape == (len(b),)
            and b.dtype.kind in "fiu" and s.dtype.kind in "fiu" and np.isfinite(b).all() and np.isfinite(s).all(), "All raw detector records required")
    q = arrays["query_points"]
    require(np.all(q[:, 0] == 0) and np.all(q[:, 1:] >= .5) and np.all(q[:, 1:] < [480, 640])
        and np.all(q[:, 1:] - .5 == np.floor(q[:, 1:])), "Exact frame-zero original pixel-center queries required")
    for i, candidate in enumerate(row["candidates"]):
        require(candidate["kind"] in ("hand", "object") and candidate["stable_id"] == f"{candidate['kind']}:{i:06d}"
            and candidate["query_count"] == int(np.diff(arrays["query_offsets"])[i])
            and candidate["mask_pixels"] == int(arrays["initial_masks"][i].sum()), "No omitted/replaced candidate support")
        points = q[arrays["query_offsets"][i]:arrays["query_offsets"][i + 1], 1:].astype(np.int64)
        require(arrays["initial_masks"][i, points[:, 0], points[:, 1]].all(), "Candidate queries must use its actual initial mask")
    require({c["kind"] for c in row["candidates"]} == {"hand", "object"}
        and int(np.diff(arrays["query_offsets"])[-1]) == row["background_query_count"], "Whole bank/class support required")
    bg = q[arrays["query_offsets"][-2]:, 1:].astype(np.int64)
    require(not arrays["initial_masks"].any(axis=0)[bg[:, 0], bg[:, 1]].any(), "Background must exclude union of ALL initial masks")


def track_clip(video, bank, row, predict):
    """ONE native full-T call for every candidate and background query together."""
    import numpy as np
    from world_reward.relational_motion import relational_motion_features
    require(isinstance(video, np.ndarray) and video.dtype == np.uint8
        and video.shape == (len(bank["frame_index"]), 480, 640, 3), "Full original RGB timeline/grid required")
    queries = bank["query_points"]; indices = np.arange(len(queries), dtype=np.int64)
    require(len(queries) > 0, "No measurable queries; no manufactured placeholder")
    data = dict(video=video, query_points=queries, point_indices=indices)
    arrays = predict(data)
    native_row = dict(query_count=len(queries), frames=len(video), width=640, height=480, point_indices=indices.tolist())
    boots.validate_prediction(arrays, native_row, queries, indices)
    groups = {kind: ([], []) for kind in ("hand", "object")}
    for i, candidate in enumerate(row["candidates"]):
        start, end = bank["query_offsets"][i:i + 2]
        groups[candidate["kind"]][0].append(arrays["tracks"][start:end])
        groups[candidate["kind"]][1].append(arrays["visible"][start:end])
    start = bank["query_offsets"][-2]
    evidence = relational_motion_features(arrays["frame_index"], arrays["frame_index"].astype(np.float64),
        tuple(groups["hand"][0]), tuple(groups["hand"][1]), tuple(groups["object"][0]), tuple(groups["object"][1]),
        arrays["tracks"][start:], arrays["visible"][start:], image_width=640, image_height=480)
    # Historical native helper creates static controls. Validate, then OMIT them:
    # they are never predictions or claimed full interaction reconstructions.
    result = {k: v for k, v in arrays.items() if k not in ("static_tracks", "static_visible")}
    result.update({"motion_" + field.name: getattr(evidence, field.name) for field in fields(evidence)})
    return result


def observe_tracks(clips, banks, out, report, persist, *, predict):
    import numpy as np
    require(len(clips) == len(banks) == 12, "All twelve full clip banks required")
    for clip, (bank, row) in zip(clips, banks):
        report.update(phase="native_full_clip_tracking", current_clip=clip["clip_index"]); persist()
        video = np.stack([read_rgb(r) for r in clip["records"]])
        arrays = track_clip(video, bank, row, predict)
        name = f"clip_{clip['clip_index']:03d}.npz"; pin = save_arrays(out / name, arrays)
        report["clips"].append(dict(clip_index=clip["clip_index"], frames=clip["frames"], query_count=len(bank["query_points"]),
            file=name, **pin, pair_supported=int(arrays["motion_pair_supported"].sum()),
            camera_supported=int(arrays["motion_camera_supported"].sum())))
        persist(); del video, arrays


def run_models(stage, clips, banks, out, report, persist, proof):
    import numpy as np
    import torch
    require(torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Actual H100 CUDA required")
    torch.manual_seed(0); np.random.seed(0); torch.cuda.manual_seed_all(0)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = False
    require(metadata.version("numpy") == "1.26.3" and metadata.version("torch") == "2.5.1+cu124", "Measured runtime packages required")
    if stage == "masks":
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        require(metadata.version("transformers") == "4.53.3" and metadata.version("tokenizers") == "0.21.4", "Measured detector runtime required")
        installed = binding.installed_sam2(proof)
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        processor = AutoProcessor.from_pretrained(binding.DEST / "weights/grounding_dino", local_files_only=True)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(binding.DEST / "weights/grounding_dino", local_files_only=True).to("cuda").eval()
        # Explicitly preserve the real existing hand/anchor image initializer:
        # upstream native dynamic-stability postprocessing remains enabled.
        predictor = SAM2ImagePredictor(build_sam2("configs/sam2.1/sam2.1_hiera_l.yaml",
            str(binding.DEST / "weights/sam2/sam2.1_hiera_large.pt"), device="cuda", mode="eval", apply_postprocessing=True))
        def detect(rgb, query):
            require(query in ("hand.", "object."), "Fixed automatic queries only")
            report["detector_attempts"] += 1; persist()
            inputs = processor(images=Image.fromarray(rgb), text=query, return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result = processor.post_process_grounded_object_detection(detector(**inputs), inputs.input_ids,
                    threshold=.3, text_threshold=.25, target_sizes=[rgb.shape[:2]])[0]
            torch.cuda.synchronize(); report["detector_calls"] += 1; persist()
            return result["boxes"].cpu().numpy().copy(), result["scores"].cpu().numpy().copy()
        class NativeImage:
            def set_image(self, rgb):
                report["sam2_encoder_attempts"] += 1; persist()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): predictor.set_image(rgb)
                torch.cuda.synchronize(); report["sam2_encoder_calls"] += 1; persist()
            def predict(self, **kwargs):
                report["sam2_batch_attempts"] += 1; persist()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16): result = predictor.predict(**kwargs)
                torch.cuda.synchronize(); report["sam2_batch_calls"] += 1; persist()
                return result
        observe_masks(clips, out, report, persist, detect=detect, predictor=NativeImage())
        require(report["detector_calls"] == 24 and report["sam2_encoder_calls"] == report["sam2_batch_calls"] == 12,
            "Exact all-candidate native call counts required")
        require(binding.installed_sam2(proof) == installed, "Installed native source changed")
    else:
        tapir, utils = boots.native_modules(ROOT / boots.BASE / "assets/tapnet_source")
        model = boots.load_model(torch, tapir, ROOT / boots.BASE / "assets/bootstapir_checkpoint_v2.pt")
        def predict(data):
            persist(); arrays = boots.native_prediction(torch, utils, model, data, report)
            torch.cuda.synchronize(); report["native_calls_completed"] += 1; persist(); return arrays
        observe_tracks(clips, banks, out, report, persist, predict=predict)
        require(report["native_calls_attempted"] == report["native_calls_returned"] == report["native_calls_completed"] == 12,
            "Exactly one complete full original-T native call per clip required")
    torch.cuda.synchronize()


def output_path(stage, revision):
    return ROOT / BASE / f"identity_{stage}_{revision}"


def host_mounts(stage, code, revision):
    """Readonly public/model/proof leaves; no dataset-private or broad ROOT mount."""
    paths = [code, code.parent / "revision", code.parent / "source-sha256", ROOT / BASE / "inputs"]
    if stage == "masks":
        paths += [binding.BUILD_CODE, binding.KERNEL_CODE, binding.BUILD_REPORT, binding.KERNEL_REPORT,
            ROOT / "results/frontend-grounding-build-v6/child-CPU-probe.log", binding.MANIFEST, binding.EXTRACTION,
            binding.DEST / "results/weights-acquisition.json"]
        for old in (binding.BUILD_CODE, binding.KERNEL_CODE): paths += [old.parent / n for n in ("revision", "source-sha256")]
        paths += [binding.DEST / "weights" / relative for relative in detector_policy.ASSETS]
    else:
        runtime_path = ROOT / f"results/bootstapir-runtime-verify-{RUNTIME_REV}/report.json"
        runtime = binding.pinned(runtime_path, RUNTIME_PIN)
        original = ROOT / "jobs" / RUNTIME_REV / "run_bootstapir_runtime_verify/code"
        protocol = boots.source.read_protocol(code / "configs/robotap_boots_protocol.json")
        paths += [runtime_path, original.parent / "revision", output_path("masks", revision)]
        paths += [original / name for name in runtime["source_helpers"]]
        paths += [ROOT / boots.BASE / "assets/tapnet_source" / name for name in protocol["source"]["files"]]
        paths += [ROOT / boots.BASE / "assets" / protocol["checkpoint"]["file"]]
    require(all(binding.canonical(p) == p and not any(c in str(p) for c in (",", "\t", "\n")) for p in paths),
        "Canonical readonly mount paths required")
    return sorted(set(paths))


def parser():
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument("--stage", choices=tuple(BUDGETS), required=True)
    mode = p.add_mutually_exclusive_group(); mode.add_argument("--host-proof", action="store_true"); mode.add_argument("--mounts", action="store_true")
    for name in ("manifest", "acquisition-report"):
        p.add_argument(f"--{name}-sha256", required=True); p.add_argument(f"--{name}-bytes", type=int, required=True)
    p.add_argument("--masks-report-sha256"); p.add_argument("--masks-report-bytes", type=int)
    return p


def main(argv=None):
    args = parser().parse_args(argv); stage = args.stage
    code, revision = Path(os.environ["WR_CODE"]), os.environ["WR_CODE_REVISION"]
    require(sys.platform == "linux" and os.environ.get("WR_ROOT") == str(ROOT), "Heavy arrays/models stay on Azure Linux")
    source = source_binding(code, revision)
    manifest_pin = dict(bytes=args.manifest_bytes, sha256=args.manifest_sha256)
    acquisition_pin = dict(bytes=args.acquisition_report_bytes, sha256=args.acquisition_report_sha256)
    masks_pin = dict(bytes=args.masks_report_bytes, sha256=args.masks_report_sha256)
    binding.validate_file_pins({"manifest": manifest_pin, "acquisition": acquisition_pin}, ("manifest", "acquisition"), 16 << 20)
    if stage == "tracks": binding.validate_file_pins({"masks": masks_pin}, ("masks",), 2 << 20)
    else: require(args.masks_report_bytes is None and args.masks_report_sha256 is None, "Masks stage cannot consume any previous candidate bank")
    if args.host_proof or args.mounts:
        require(os.uname().nodename == "world-reward-ncc-h100-02", "Actual owned VM02 host required")
        clips, rgb, manifest = public_inputs(ROOT / BASE / "inputs", manifest_pin)
        acquisition_proof(acquisition_pin, manifest_pin, manifest, code)
        model = frontend_proof(code, live=True)[1] if stage == "masks" else boots_proof(code, live=True)
        banks = load_banks(output_path("masks", revision), masks_pin, clips, revision, manifest_pin, decode=False)[1] if stage == "tracks" else {}
        proof = dict(source=source, manifest=manifest_pin, acquisition=acquisition_pin, model=model,
            rgb={str(k): v for k, v in rgb.items()}, banks={str(k): v for k, v in banks.items()})
        if args.mounts:
            for path in host_mounts(stage, code, revision): print(str(path) + "\t" + str(path))
        else: print(hashlib.sha256(json.dumps(proof, sort_keys=True).encode()).hexdigest())
        return
    require(os.geteuid() == 0 and os.environ.get("WR_AZURE_VM02_VERIFIED") == "1"
        and os.environ.get("WR_IMAGE_ID") == IMAGES[stage]
        and re.fullmatch(r"[0-9a-f]{64}", os.environ.get("WR_HOST_PROOF_SHA256", ""))
        and {p.name for p in Path("/sys/class/net").iterdir()} == {"lo"}, "Actual restricted offline native container required")
    require(not any((ROOT / name).exists() for name in (BASE + "/eval_private", BASE + "/report.json", "data", "vendor")),
        "No private acquisition/annotation/challenge mount may be readable")
    out = binding.canonical(output_path(stage, revision)); state = out.lstat()
    require(stat.S_ISDIR(state.st_mode) and state.st_uid == 0 and state.st_mode & 0o777 == 0o700
        and {p.name for p in out.iterdir()} == {".container.cid"}, "Fresh root-owned output reserved by actual wrapper required")
    report = dict(schema="world-reward-dexycb-identity-infer-v1", stage=stage, status="fail", phase="preflight",
        producer_revision=revision, script_sha256=source["helpers"][HELPERS[0]]["sha256"], source_binding=source,
        image_id=IMAGES[stage], public_manifest=manifest_pin, host_proof_sha256=os.environ["WR_HOST_PROOF_SHA256"],
        network="none", device="cuda", budget_seconds=BUDGETS[stage], motion_time_basis="original_frame_index_not_seconds",
        timestamps_available=False, private_annotations_read=False, challenge_inputs_used=False, oracle_modes=[],
        detector="GroundingDINO", hand_query="hand.", object_query="object.", detector_confidence=.3,
        detector_text_threshold=.25, nms_iou=.7, sam2_multimask_output=False, sam2_native_postprocessing=True,
        quality_verified=False, identity_accepted=False, contacts_inferred=False, geometry_inferred=False,
        license_eligibility_verified=False, training_overlap_verified=False, challenge_overlap_verified=False,
        detector_attempts=0, detector_calls=0, sam2_encoder_attempts=0, sam2_encoder_calls=0,
        sam2_batch_attempts=0, sam2_batch_calls=0, native_calls_attempted=0, native_calls_returned=0,
        native_calls_completed=0, clips=[], original_rehashed_after=False)
    started, frozen, model, model_proof = time.monotonic(), {}, None, None
    path = out / "report.json"
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o400), "w") as stream:
        def persist():
            report["elapsed_seconds"] = time.monotonic() - started
            stream.seek(0); json.dump(report, stream, sort_keys=True, allow_nan=False); stream.write("\n")
            stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Frozen all-twelve-clip stage budget exceeded; no partial-time fallback")
        handlers = {s: signal.signal(s, expired) for s in (signal.SIGALRM, signal.SIGTERM, signal.SIGINT)}
        signal.alarm(BUDGETS[stage])
        try:
            persist(); clips, frozen, _ = public_inputs(ROOT / BASE / "inputs", manifest_pin)
            if stage == "masks": model_proof, model = frontend_proof(code)
            else: model = boots_proof(code)
            report["model_assets"] = model
            banks, bank_frozen = load_banks(output_path("masks", revision), masks_pin, clips, revision, manifest_pin) if stage == "tracks" else (None, {})
            frozen.update(bank_frozen)
            if stage == "tracks": report["masks_report"] = masks_pin
            report["phase"] = "load_native_models"; persist()
            run_models(stage, clips, banks, out, report, persist, model_proof)
            require({p.name for p in out.iterdir()} == {"report.json", ".container.cid", *[r["file"] for r in report["clips"]]}
                and all(binding.identity(out / r["file"]) == {k: r[k] for k in ("bytes", "sha256")} for r in report["clips"]),
                "Exclusive complete original outputs required")
            report.update(status="pass", phase="complete")
        except BaseException as error:
            report.update(status="fail", error_type=type(error).__name__, error="Native automatic stage failed at recorded phase")
            raise
        finally:
            signal.alarm(0)
            for s, handler in handlers.items(): signal.signal(s, handler)
            try:
                require(source_binding(code, revision) == source, "Original source changed")
                binding.recheck(frozen)
                if model is not None:
                    require((frontend_proof(code)[1] if stage == "masks" else boots_proof(code)) == model, "Original model/runtime assets changed")
                require(time.monotonic() - started <= BUDGETS[stage],
                    "Inclusive native stage budget exceeded during post-verification")
                report["original_rehashed_after"] = True
            except BaseException:
                report.update(status="fail", original_rehashed_after=False, integrity_error="Post-run original proof changed")
                raise
            finally: persist(); path.chmod(0o444)
    print(json.dumps({k: report[k] for k in ("stage", "status", "elapsed_seconds", "quality_verified")}))


if __name__ == "__main__": main()
