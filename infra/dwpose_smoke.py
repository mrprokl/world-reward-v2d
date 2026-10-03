"""Native, unmodified RGB DWPose-133 CPU ABI and two-session replay only.

The already-scored synthetic cohort is NOT independent accuracy validation.
Native float64-list feeds are delegated unchanged: failure is not repaired by
casting. Raw scores are not probabilities; mapped invalid coordinates survive.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
from importlib import metadata, util
import json
import os
from pathlib import Path
import platform
import re
import signal
import subprocess
import sys
import tempfile
import time

import numpy as np
import dwpose_acquire as acquisition
import dwpose_wheel_audit as audit

OUT = "validation/dwpose_smoke_v2"
STAGE = "native_dwpose_rgb133_cpu_abi_and_replay_v2"
PREVIOUS_SMOKE = "validation/dwpose_smoke_v1/report.json"
PREVIOUS_SMOKE_SHA = "f95a4dbdb03bd02e9bd65a216bec233de2cb469bf78e71dd4162568bae41ca41"
PREVIOUS_SMOKE_REVISION = "ee91a530398f2f8bc0aced813f94717c114f5ee7"
PREVIOUS_SMOKE_SOURCE_SHA = "277ec36e23a19ba13e9637f6a0e03f56c0c23d3997336bf00d31125ee19b5447"
PUBLIC = "validation/identity_rgb_v2"
WIDTH, HEIGHT, BUDGET = 1024, 768, 180
AUDIT_SHA = "e7fa6fce0397654ec5d1d2c07c49bd6f2a50655cda185d4297bfb8f0f4aae3e2"
AUDIT_SOURCE_SHA = "4f43263edcb366c06a310a51cb2300057fe3ee1357fc02c8c36013dc29464e97"
AUDIT_REVISION = "b04cfe24e11b5469ca7d12a2747ed6c51643044e"
MANIFEST_SHA = "2c584ea633a958c737520d53c68c12b1429b8358f182c07bf46e53627b8f8267"
MASK_SHA = "aa1c8346cfd7609a58d055f71762060aca238c216100bd8aa990ca1de79ea909"
MASK_REVISION = "0236b0bd99d4c15054e7bdd88e8839cd0591e02a"
SELECTED = (
    (0, "clip_00_frame_000.png", "ecb5ad1da3d2274599d7e2bd352ccd0c38c53902399220f4b09161ee5c09b6a8",
     "fde6c6b1f047dc7d600b8466cef827302aaee11ffe1331aedd2f90bb18414d8b", 2328, 13735),
    (5, "clip_01_frame_000.png", "ad7aa675f57150040355f650938936fe1ef574ab87f81e4bc7671e7990a2e3d1",
     "e093e721dc64a800bd52aa660f578439a4f5263a5cfaf4b0a999411dc19a1bcb", 2828, 23824),
)


def identity(path, *, sha=None, size=None):
    path = audit.regular(path); actual = {"sha256": acquisition.digest(path), "bytes": path.stat().st_size}
    if (sha is not None and actual["sha256"] != sha) or (size is not None and actual["bytes"] != size):
        raise ValueError("Pinned regular input bytes/SHA differ")
    return actual


def source_identity():
    paths = [Path(__file__), Path(__file__).with_name("run_dwpose_smoke.sh"),
             Path(acquisition.__file__), Path(audit.__file__)]
    rows = {p.name: identity(p) for p in paths}
    if rows["dwpose_acquire.py"]["sha256"] != audit.SOURCE_SHA or rows["dwpose_wheel_audit.py"]["sha256"] != AUDIT_SOURCE_SHA:
        raise ValueError("Frozen read-only acquisition/audit source differs")
    return rows


def validate_previous_smoke(root):
    path = root / PREVIOUS_SMOKE; receipt = identity(path, sha=PREVIOUS_SMOKE_SHA)
    data = json.loads(path.read_text())
    expected = {"stage": "native_dwpose_rgb133_cpu_abi_and_replay", "status": "fail", "phase": "native_source_load",
                "producer_revision": PREVIOUS_SMOKE_REVISION, "script_sha256": PREVIOUS_SMOKE_SOURCE_SHA, "image_id": audit.IMAGE,
                "error_type": "ValueError", "error": "Require native two SimCC133 graph outputs", "sessions": [],
                "private_prefix_packages_installed": True, "private_prefix_removed": True, "device": "cpu", "network": "none",
                "native_cpu_abi_verified": False, "two_session_byte_replay_verified": False, "gpu_used": False,
                "own_feed_cast": False, "native_source_modified": False, "private_truth_read": False, "ground_truth_used": False,
                "challenge_inputs_used": False, "oracle_modes": [], "adoption_authorized": False}
    if any(data.get(k) != v or type(data.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Preserved v1 metadata-only failure provenance differs")
    return receipt


def validate_assets(root):
    audit.validate_previous(root, acquisition); original = audit.validate_failed(root, acquisition)
    path = root / audit.OUT / "report.json"; receipt = identity(path, sha=AUDIT_SHA)
    data = json.loads(path.read_text())
    expected = {"stage": "pinned_dwpose_wheels_notice_audit_v3", "status": "pass", "phase": "complete",
                "producer_revision": AUDIT_REVISION, "script_sha256": AUDIT_SOURCE_SHA, "image_id": audit.IMAGE,
                "original_receipt_sha256": audit.FAILED_SHA, "original_producer_revision": audit.FAILED_REVISION,
                "previous_failed_audit_sha256": audit.PREVIOUS_SHA, "original_acquisition_script_sha256": audit.SOURCE_SHA,
                "legacy_wheel_audits_omitted": True, "final_assets_receipt_source_rehashed": True, "network": "none", "device": "cpu"}
    if any(data.get(k) != v or type(data.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Pinned completed v3 notice audit provenance differs")
    for key in ("gpu_used", "assets_redownloaded", "original_assets_modified", "original_failure_rewritten", "packages_installed",
                "upstream_source_executed", "inference_performed", "runtime_verified", "accuracy_verified", "license_clearance_verified",
                "training_data_rights_verified", "training_overlap_excluded", "challenge_eligibility_verified", "adoption_authorized",
                "private_truth_read", "challenge_inputs_used", "hand_labeled_test"):
        if data.get(key) is not False: raise ValueError("Pinned audit safety flag differs")
    if data.get("oracle_modes") != []: raise ValueError("Oracle audit input forbidden")
    rows = data.get("wheel_audits")
    if not isinstance(rows, list) or len(rows) != 2: raise ValueError("Both exact wheel audits required")
    texts = []
    for row, package, count in zip(rows, ("onnxruntime", "flatbuffers"), (353, 14)):
        spec = acquisition.WHEELS[package]
        if (row.get("package") != package or row.get("version") != spec["version"] or row.get("member_count") != count
                or sorted(row.get("tags", [])) != sorted(spec["tags"]) or sorted(row.get("requires_dist", [])) != sorted(spec["requires"])
                or row.get("safe_member_paths_verified") is not True or row.get("selected_text_crc_verified") is not True
                or row.get("symlinks_present") is not False or row.get("archive_expanded") is not False
                or row.get("embedded_license_present") is not (package == "onnxruntime")
                or row.get("external_primary_license_verified") is not (package == "flatbuffers")):
            raise ValueError("Exact wheel metadata/notice inventory differs")
        records = row.get("retained_texts")
        if not isinstance(records, list) or not records or sum(r.get("bytes", 0) for r in records) > 10000000:
            raise ValueError("Bounded retained notices required")
        seen = set()
        for item in records:
            name = str(acquisition.safe_relative(item["file"]))
            if name in seen or type(item.get("bytes")) is not int or item["bytes"] <= 0:
                raise ValueError("Unique regular notice byte inventory required")
            seen.add(name); actual = identity(root / audit.OUT / package / name, sha=item["sha256"], size=item["bytes"])
            texts.append({"file": f"{package}/{name}", **actual})
        if package == "flatbuffers":
            license_record = next(r for r in acquisition.ASSETS if r[0] == "licenses/flatbuffers-LICENSE")
            external = [r for r in records if r["file"] == "external-primary/flatbuffers-LICENSE"]
            if (row.get("external_primary_license_source_revision") != acquisition.FLATBUFFERS_REV
                    or row.get("external_primary_license_url") != license_record[3] or len(external) != 1
                    or external[0].get("member") is not None or (external[0]["bytes"], external[0]["sha256"]) != license_record[1:3]):
                raise ValueError("Exact external primary Flatbuffers license binding differs")
    return {"audit_receipt": receipt, "original_receipt_sha256": audit.FAILED_SHA,
            "previous_failed_receipt_sha256": audit.PREVIOUS_SHA, "files": original["files"], "retained_notices": texts}


def validate_public(root):
    base = root / PUBLIC; manifest_path = base / "inputs/manifest.json"; mask_path = base / "automatic_masks/report.json"
    manifest_id = identity(manifest_path, sha=MANIFEST_SHA, size=2199); mask_id = identity(mask_path, sha=MASK_SHA, size=19818)
    manifest = json.loads(manifest_path.read_text()); masks = json.loads(mask_path.read_text())
    if set(manifest) != {"schema", "images"} or manifest["schema"] != "world-reward-identity-rgb-v1" or not isinstance(manifest["images"], list) or len(manifest["images"]) != 15:
        raise ValueError("Exact complete public RGB-only manifest required")
    expected = {"stage": "public_identity_rgb_automatic_masks", "status": "pass", "phase": "complete", "frames": 15,
                "producer_revision": MASK_REVISION, "network": "none", "private_truth_read": False, "ground_truth_used": False,
                "challenge_inputs_used": False, "hand_labeled_test": False, "oracle_modes": [], "human_query": "person.", "object_query": "bottle.",
                "actual_detector_calls": 30, "actual_sam2_calls": 30, "actual_sam2_image_encoder_calls": 15,
                "actual_automatic_inference_verified": True, "all_cases_retained": True,
                "input_manifest_sha256": MANIFEST_SHA, "input_manifest_bytes": 2199}
    if any(masks.get(k) != v or type(masks.get(k)) is not type(v) for k, v in expected.items()):
        raise ValueError("Exact completed automatic RGB-only mask receipt required")
    if not isinstance(masks.get("records"), list) or len(masks["records"]) != 15: raise ValueError("All15 original mask records required")
    for i, (image, row) in enumerate(zip(manifest["images"], masks["records"])):
        clip, frame = divmod(i, 5); name = f"clip_{clip:02d}_frame_{frame:03d}.png"
        if (set(image) != {"file", "sha256", "width", "height"} or image["file"] != name
                or type(image["width"]) is not int or type(image["height"]) is not int or (image["width"], image["height"]) != (WIDTH, HEIGHT)
                or not re.fullmatch(r"[0-9a-f]{64}", str(image["sha256"])) or row.get("file") != name
                or row.get("rgb_sha256") != image["sha256"] or type(row.get("clip_index")) is not int or row["clip_index"] != clip
                or type(row.get("frame_index")) is not int or row["frame_index"] != frame):
            raise ValueError("Original ordered manifest/mask RGB bindings differ")
        for label, query in (("human", "person."), ("object", "bottle.")):
            if (row.get(label + "_query") != query or row.get(label + "_mask_file") != Path(name).stem + "_" + label + ".png"
                    or not re.fullmatch(r"[0-9a-f]{64}", str(row.get(label + "_mask_sha256")))
                    or type(row.get(label + "_mask_bytes")) is not int or row[label + "_mask_bytes"] <= 0
                    or type(row.get(label + "_mask_pixels")) is not int or not 0 < row[label + "_mask_pixels"] <= WIDTH * HEIGHT):
                raise ValueError("Complete automatic mask metadata required")
    selected = []
    for index, name, rgb_sha, human_sha, size, pixels in SELECTED:
        image, row = manifest["images"][index], masks["records"][index]
        if image["file"] != name or image["sha256"] != rgb_sha or (row["human_mask_sha256"], row["human_mask_bytes"], row["human_mask_pixels"]) != (human_sha, size, pixels):
            raise ValueError("Two preregistered RGB/mask identities differ")
        rgb = identity(base / "inputs" / name, sha=rgb_sha)
        human = identity(base / "automatic_masks" / row["human_mask_file"], sha=human_sha, size=size)
        selected.append({"file": name, "rgb": rgb, "human_mask_file": row["human_mask_file"], "human_mask": human, "human_mask_pixels": pixels})
    return {"manifest": manifest_id, "automatic_masks_receipt": mask_id, "selected": selected}


def actor_bbox(mask):
    if type(mask) is not np.ndarray or mask.dtype != np.uint8 or mask.shape != (HEIGHT, WIDTH) or not np.isin(mask, (0, 255)).all():
        raise ValueError("Full-grid binary automatic uint8 mask required")
    y, x = np.nonzero(mask)
    if not len(x): raise ValueError("Empty bbox would trigger forbidden native full-image fallback")
    return np.array([[x.min(), y.min(), x.max() + 1, y.max() + 1]], dtype=np.float32)


def pip_argv(prefix, root):
    wheels = [str(root / acquisition.BASE / r[0]) for r in acquisition.ASSETS if r[0].startswith("wheels/")]
    return [sys.executable, "-I", "-m", "pip", "--isolated", "--disable-pip-version-check", "install", "--no-index", "--no-deps",
            "--no-compile", "--no-warn-script-location", "--target", str(prefix), *wheels]


@contextmanager
def private_prefix(report, persist):
    temporary = tempfile.TemporaryDirectory(prefix="world-reward-dwpose-", dir="/tmp")
    prefix = Path(temporary.name); report.update(private_prefix=str(prefix), private_prefix_removed=False); persist()
    try: yield prefix
    finally:
        temporary.cleanup(); report["private_prefix_removed"] = not prefix.exists(); persist()


def dependency_identity():
    expected = {"numpy": "1.26.3", "packaging": "24.1", "protobuf": "7.36.2", "pip": "24.2",
                "opencv-python": "4.11.0.86", "opencv-contrib-python": "4.11.0.86"}
    actual = {name: metadata.version(name) for name in expected}
    if actual != expected or platform.python_version() != "3.11.10" or platform.libc_ver() != ("glibc", "2.35"):
        raise ValueError("Pinned image Python/glibc/dependency ABI differs")
    import cv2
    if cv2.__version__ != "4.11.0": raise ValueError("Pinned cv2 runtime version differs")
    return {"python": platform.python_version(), "libc": list(platform.libc_ver()), "distributions": actual,
            "numpy_origin": str(audit.regular(np.__file__)), "cv2_origin": str(audit.regular(cv2.__file__)), "cv2_version": cv2.__version__}


def import_runtime(prefix):
    sys.path.insert(0, str(prefix)); import onnxruntime; import flatbuffers
    rows = {}
    for name, module, version in (("onnxruntime", onnxruntime, "1.30.0"), ("flatbuffers", flatbuffers, "25.12.19")):
        path = audit.regular(module.__file__); distribution = metadata.distribution(name)
        if (module.__version__ != version or distribution.version != version or not path.is_relative_to(prefix)
                or not Path(distribution.locate_file("")).resolve().is_relative_to(prefix)):
            raise ValueError("Runtime must import exact newly installed private-prefix distributions")
        rows[name] = {"version": module.__version__, "origin": str(path), "origin_sha256": acquisition.digest(path)}
    return onnxruntime, rows


def array_identity(value):
    if type(value) is not np.ndarray or value.dtype.hasobject or not np.isfinite(value).all():
        raise ValueError("Finite native numeric array required")
    return {"dtype": str(value.dtype), "shape": list(value.shape), "sha256": hashlib.sha256(value.tobytes(order="C")).hexdigest()}


def session_metadata(session):
    def rows(values): return [{"name": x.name, "shape": x.shape, "type": x.type} for x in values]
    return {"inputs": rows(session.get_inputs()), "outputs": rows(session.get_outputs()),
            "providers": session.get_providers(), "custom_metadata": session.get_modelmeta().custom_metadata_map}


def validate_session(session, *, graph=None):
    graph = session_metadata(session) if graph is None else graph
    expected = {"inputs": [{"name": "input", "type": "tensor(float)", "shape": ["batch", 3, 384, 288]}],
                "outputs": [{"name": f"simcc_{axis}", "type": "tensor(float)",
                             "shape": ["batch", f"MatMulsimcc_{axis}_dim_1", f"MatMulsimcc_{axis}_dim_2"]} for axis in ("x", "y")],
                "providers": ["CPUExecutionProvider"], "custom_metadata": {}}
    if graph != expected:
        raise ValueError("Require exact pinned symbolic DWPose exported graph signature")
    return graph  # Actual numeric SimCC shapes remain independently checked in SessionProxy.run.


def validate_options(session, ort):
    options = session.get_session_options()
    if (options.intra_op_num_threads != 4 or options.inter_op_num_threads != 1
            or options.execution_mode != ort.ExecutionMode.ORT_SEQUENTIAL):
        raise ValueError("Actual session options must be4/1/sequential")


class SessionProxy:
    """Observe supplied native values, never transform feeds or model outputs."""
    def __init__(self, session, records, persist): self.session, self.records, self.persist = session, records, persist
    def get_inputs(self): return self.session.get_inputs()
    def get_outputs(self): return self.session.get_outputs()
    def run(self, names, feed):
        if names != [x.name for x in self.get_outputs()] or set(feed) != {self.get_inputs()[0].name}:
            raise ValueError("Native actual-name output/input binding differs")
        value = feed[self.get_inputs()[0].name]
        if not isinstance(value, list) or len(value) != 1 or type(value[0]) is not np.ndarray or value[0].shape != (3, 384, 288) or value[0].dtype != np.float64:
            raise ValueError("Unmodified pinned native list-of-float64-CHW feed required")
        row = {"supplied_container": "list", "supplied_array": array_identity(value[0]), "effective_feed_shape": [1, 3, 384, 288],
               "effective_runtime_conversion_observed": False, "delegated_unmodified": True, "run_completed": False}
        self.records.append(row); self.persist(); start = time.perf_counter()
        outputs = self.session.run(names, feed)
        row["run_seconds"] = time.perf_counter() - start
        if not isinstance(outputs, list) or len(outputs) != 2: raise ValueError("Exactly two native SimCC arrays required")
        for value, shape in zip(outputs, ((1, 133, 576), (1, 133, 768))):
            if type(value) is not np.ndarray or value.dtype != np.float32 or value.shape != shape or not np.isfinite(value).all():
                raise ValueError("Native finite float32 SimCC133 ABI differs")
        row.update(raw_simcc=[array_identity(v) for v in outputs], run_completed=True); self.persist()
        return outputs


def validate_prediction(points, scores):
    if (type(points) is not np.ndarray or points.dtype not in (np.float32, np.float64) or points.shape != (1, 133, 2)
            or type(scores) is not np.ndarray or scores.dtype != np.float32 or scores.shape != (1, 133)
            or not np.isfinite(points).all() or not np.isfinite(scores).all()):
        raise ValueError("Native finite133 decoded coordinate/raw-score ABI differs")
    return scores > 0  # Native sentinel precedes affine mapping; final positions are not a validity indicator.


def validate_replay(sessions):
    if len(sessions) != 2 or any(len(s["predictions"]) != 2 or len(s["calls"]) != 2 for s in sessions):
        raise ValueError("Exactly two fresh sessions/four calls required")
    for left, right in zip(sessions[0]["predictions"], sessions[1]["predictions"]):
        if any(left[k] != right[k] for k in ("file", "keypoints", "scores", "validity", "raw_simcc")):
            raise ValueError("Two-session decoded/raw native byte replay differs")
    if any(not row["run_completed"] for s in sessions for row in s["calls"]): raise ValueError("Every delegated call must complete")


def validate_artifacts(out, sessions):
    names = {"report.json"}
    for session in sessions:
        for row in session["predictions"]:
            name = str(acquisition.safe_relative(row["prediction_file"]))
            if Path(name).name != name or name in names: raise ValueError("Unique flat prediction inventory required")
            names.add(name)
            if identity(out / name) != row["prediction"]: raise ValueError("Frozen native prediction SHA/bytes changed")
    if len(names) != 5 or {p.name for p in out.iterdir()} != names: raise ValueError("Exactly four frozen predictions and receipt required")


def perform(root, out, report, persist, started):
    previous = validate_previous_smoke(root); sources = source_identity(); assets = validate_assets(root); public = validate_public(root)
    report.update(previous_failed_smoke_receipt=previous, sources=sources, assets=assets, public_inputs=public,
                  phase="private_prefix_install", dependencies=dependency_identity()); persist()
    if any(name in sys.modules or util.find_spec(name) is not None for name in ("onnxruntime", "flatbuffers")):
        raise ValueError("Runtime packages must not preexist private-prefix installation")
    from PIL import Image
    images = []
    for row in public["selected"]:
        with Image.open(root / PUBLIC / "inputs" / row["file"]) as rgb, Image.open(root / PUBLIC / "automatic_masks" / row["human_mask_file"]) as human:
            if rgb.format != "PNG" or rgb.mode != "RGB" or rgb.size != (WIDTH, HEIGHT) or human.format != "PNG" or human.mode != "L" or human.size != (WIDTH, HEIGHT):
                raise ValueError("Original RGB/binary automatic mask PNG decoding differs")
            image = np.ascontiguousarray(np.asarray(rgb)); mask = np.asarray(human)
            if image.dtype != np.uint8 or image.shape != (HEIGHT, WIDTH, 3) or np.count_nonzero(mask) != row["human_mask_pixels"]:
                raise ValueError("Original uint8 RGB/nonempty automatic mask area differs")
            images.append((image, actor_bbox(mask)))
    with private_prefix(report, persist) as prefix:
        command = pip_argv(prefix, root); report["pip_argv"] = command; persist()
        subprocess.run(command, check=True, timeout=max(.001, BUDGET - (time.perf_counter() - started)), stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        report["private_prefix_packages_installed"] = True; ort, origins = import_runtime(prefix)
        report.update(runtime_imports=origins, phase="native_source_load"); persist()
        source = root / acquisition.BASE / "source/onnxpose.py"; spec = util.spec_from_file_location("world_reward_native_onnxpose", source)
        native = util.module_from_spec(spec); spec.loader.exec_module(native)
        for session_index in range(2):
            options = ort.SessionOptions(); options.intra_op_num_threads = 4; options.inter_op_num_threads = 1
            options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            session = ort.InferenceSession(str(root / acquisition.BASE / acquisition.ASSETS[0][0]), sess_options=options, providers=["CPUExecutionProvider"])
            session.disable_fallback(); graph = session_metadata(session)
            row = {"index": session_index, "fresh_constructor": True, "graph": graph, "intra_threads": 4, "inter_threads": 1,
                   "execution": "sequential", "checkpoint_sha256": acquisition.ASSETS[0][2], "calls": [], "predictions": []}
            report["sessions"].append(row); report["phase"] = "graph_metadata_validation"; persist()
            validate_session(session, graph=graph); validate_options(session, ort)
            report["phase"] = "native_rgb_inference"; persist()
            proxy = SessionProxy(session, row["calls"], persist)
            for selected, (image, bbox) in zip(public["selected"], images):
                points, scores = native.inference_pose(proxy, bbox.copy(), image)
                valid = validate_prediction(points, scores)
                filename = f"session_{session_index}_{Path(selected['file']).stem}.npz"; target = out / filename
                with target.open("xb") as stream: np.savez(stream, keypoints=points, scores=scores, validity=valid, bbox=bbox)
                target.chmod(0o444)
                with np.load(target, allow_pickle=False) as saved:
                    if set(saved.files) != {"keypoints", "scores", "validity", "bbox"} or any(saved[k].tobytes() != value.tobytes() for k, value in (("keypoints", points), ("scores", scores), ("validity", valid), ("bbox", bbox))):
                        raise ValueError("Frozen native prediction bytes differ")
                row["predictions"].append({"file": selected["file"], "prediction_file": filename, "prediction": identity(target),
                    "keypoints": array_identity(points), "scores": array_identity(scores), "validity": array_identity(valid),
                    "raw_simcc": row["calls"][-1]["raw_simcc"], "positive_score_count": int(valid.sum()), "automatic_bbox": bbox[0].tolist()}); persist()
            del proxy, session; gc.collect()
        validate_replay(report["sessions"])
        validate_artifacts(out, report["sessions"])
        if previous != validate_previous_smoke(root) or assets != validate_assets(root) or public != validate_public(root) or sources != source_identity():
            raise ValueError("Frozen source/assets/notices/public inputs changed")
        report.update(final_inputs_source_assets_rehashed=True, native_cpu_abi_verified=True, two_session_byte_replay_verified=True)
    report.update(private_prefix_removed=not prefix.exists(), status="pass", phase="complete")
    if not report["private_prefix_removed"]: raise ValueError("Disposable private-prefix install was not removed")


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root = Path(os.environ.get("WR_ROOT", "")); out = root / OUT; revision = os.environ.get("WR_CODE_REVISION", "")
    if (platform.system() != "Linux" or root != Path("/srv/scenesmith/world-reward") or root.resolve() != root.absolute()
            or os.geteuid() != 1000 or {p.name for p in Path("/sys/class/net").iterdir()} != {"lo"}
            or os.environ.get("WR_IMAGE_ID") != audit.IMAGE or os.environ.get("CUDA_VISIBLE_DEVICES") != ""
            or not re.fullmatch(r"[0-9a-f]{40}", revision) or os.environ.get("WR_DWPOSE_SMOKE_RESERVED") != "1"
            or not out.is_dir() or any(out.iterdir()) or any(p.is_symlink() for p in (out, *out.parents))):
        raise ValueError("Require fresh reserved remote CPU/network-none pinned-image output")
    report = {"stage": STAGE, "status": "fail", "phase": "public_assets_integrity",
              "producer_revision": revision, "script_sha256": acquisition.digest(Path(__file__)), "image_id": audit.IMAGE,
              "previous_failed_smoke_sha256": PREVIOUS_SMOKE_SHA, "previous_failure_rewritten": False,
              "budget_seconds": BUDGET, "device": "cpu", "network": "none", "sessions": [], "selection_rule": "two_fixed_protocol_first_frames",
              "native_source_modified": False, "own_feed_cast": False, "channel_swap": False, "full_image_fallback": False,
              "raw_scores_clamped": False, "confidence_threshold_applied": False, "wrapper_neck134_used": False,
              "private_prefix_packages_installed": False, "private_prefix_removed": False, "global_image_modified": False,
              "gpu_used": False, "private_truth_read": False, "ground_truth_used": False, "challenge_inputs_used": False,
              "hand_labeled_test": False, "oracle_modes": [], "native_cpu_abi_verified": False, "two_session_byte_replay_verified": False,
              "independent_quality_cohort": False, "accuracy_verified": False, "semantic_detection_verified": False,
              "license_clearance_verified": False, "training_overlap_excluded": False, "adoption_authorized": False}
    started = time.perf_counter(); path = out / "report.json"
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter() - started; stream.seek(0)
            json.dump(report, stream, indent=2, allow_nan=False); stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Whole native CPU smoke exceeded180s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
        try: persist(); perform(root, out, report, persist, started)
        except Exception as error:
            report.update(status="fail", error_type=type(error).__name__, error=str(error)[:3000]); raise
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
