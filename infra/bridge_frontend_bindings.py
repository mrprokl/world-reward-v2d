"""Small shared bindings for NEW RGB anchors; no models or media acquisition."""
from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
import re
import subprocess

import frontend_selected_assets as selected

ROOT = selected.ROOT
DEST = selected.DISK / "frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b"
MANIFEST = DEST / selected.MANIFEST
EXTRACTION = ROOT / "results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json"
BASE = "validation/bridge_rgb_anchor_v1"
IMAGE = "sha256:fd26863fd69d8fa1bb0bcc137bc7ddbee18fd5955484dcba672404a73326e252"
BUILD_REV = "d1fcb8ad158cf4ad4b1f7fe2bba7ea533ae565cc"
KERNEL_REV = "168a809369a294c1ab65154a585b83014c98d976"
BUILD_PIN = dict(bytes=24023, sha256="fffedb80e00b7c5e0f29bd39c491043227330a42a32237aa388a07c417bd5093")
KERNEL_PIN = dict(bytes=2069, sha256="dfad765d2d8e9021191b6fc6b6111b944e222a268b2c37ea59e58a1e432f916b")
BUILD_SHA = "3e72495079be545ea022aaa27648b64c00a1fae1e0c14c455ac27091c19727b8"
KERNEL_SOURCE_PIN = dict(bytes=15441, sha256="156bf96f6f875ccd5c58e13a25bc10c52a8cceffeeaceff1ace58e26b1737ed6")
EXTENSION_PIN = dict(bytes=1353848, sha256="acd3d17cc98745b8da081bed64f0015362bafc5ebc897276d0522c1a790d27bd")
BUILD_CODE = ROOT / "jobs" / BUILD_REV / "run_frontend_grounding_build/code"
KERNEL_CODE = ROOT / "jobs" / KERNEL_REV / "run_frontend_sam2_kernel_gate/code"
BUILD_REPORT = ROOT / "results/frontend-grounding-build-v6/report.json"
KERNEL_REPORT = ROOT / "results/frontend-sam2-kernel-gate-v1/report.json"
CONFIG = "configs/frontend_grounding_source_pins.json"
INPUT_PINS = "configs/bridge_rgb_anchor_input_pins.json"
ENTRIES = {"run_bridge_rgb_anchor_masks", "run_bridge_rgb_anchor_infer"}
FILENAMES = tuple(f"clip_{clip:06d}_frame_000000.png" for clip in range(4))
require = selected.require
canonical = selected.canonical
strict_json = selected.strict_json


def identity(path, maximum=7_000_000_000):
    """Stream bounded source/model bytes; don't allocate checkpoints in RAM."""
    path = canonical(path); before = path.lstat()
    import stat
    require(stat.S_ISREG(before.st_mode) and before.st_nlink == 1 and not before.st_mode & 0o222
            and 0 < before.st_size <= maximum, "Bounded readonly regular artifact required")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b""): digest.update(block)
    after = path.lstat()
    require(all(getattr(before, k) == getattr(after, k) for k in
        ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")), "Artifact changed during hashing")
    return dict(bytes=before.st_size, sha256=digest.hexdigest())


def pinned(path, wanted, maximum=200000):
    require(type(wanted) is dict and set(wanted) == {"bytes", "sha256"}
        and type(wanted["bytes"]) is int and 0 < wanted["bytes"] <= maximum
        and re.fullmatch("[0-9a-f]{64}", str(wanted["sha256"])), "Independent bounded byte pins required")
    require(identity(path, maximum) == wanted, "Independent artifact SHA/bytes differ")
    return strict_json(Path(path).read_bytes())


def validate_file_pins(files, names, maximum):
    require(type(files) is dict and set(files) == set(names), "Exact independently pinned file set required")
    for pin in files.values():
        require(type(pin) is dict and set(pin) == {"bytes","sha256"} and type(pin["bytes"]) is int
            and 0 < pin["bytes"] <= maximum and re.fullmatch("[0-9a-f]{64}",str(pin["sha256"])), "Strict file byte identities required")


def kernel_helper():
    path = Path(__file__).with_name("frontend_sam2_kernel_gate.py")
    require(identity(path, 200000) == KERNEL_SOURCE_PIN, "Original kernel helper bytes required before import")
    spec = importlib.util.spec_from_file_location("bridge_original_kernel_bindings", path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    return module


def authenticate(root, code, entry, live=False):
    """Measured old proofs first; original --verify runs only in its real job."""
    require(canonical(root) == ROOT and entry in ENTRIES, "Explicit new VM02 frontend entry required")
    code = canonical(code); revision = code.parent.parent.name; kernel = kernel_helper()
    own = kernel.closure(code, revision, entry, (f"infra/{entry[4:]}.py", f"infra/{entry}.sh",
        "infra/bridge_frontend_bindings.py", "infra/frontend_selected_assets.py", "infra/frontend_sam2_kernel_gate.py", CONFIG))
    build = pinned(BUILD_REPORT, BUILD_PIN); receipt = pinned(KERNEL_REPORT, KERNEL_PIN)
    old_build = kernel.closure(BUILD_CODE, BUILD_REV, "run_frontend_grounding_build", kernel.BUILD_HELPERS)
    old_kernel = kernel.closure(KERNEL_CODE, KERNEL_REV, "run_frontend_sam2_kernel_gate",
        ("infra/frontend_sam2_kernel_gate.py", "infra/run_frontend_sam2_kernel_gate.sh", CONFIG))
    require(old_build["helpers"]["infra/frontend_grounding_build.py"]["sha256"] == BUILD_SHA
        and old_kernel["helpers"]["infra/frontend_sam2_kernel_gate.py"] == KERNEL_SOURCE_PIN,
        "Original measured source scripts differ")
    require(build.get("source_binding") == old_build and receipt.get("source_binding") == old_kernel
        and own["helpers"][CONFIG] == old_build["helpers"][CONFIG] == old_kernel["helpers"][CONFIG], "Original proof/source/config ancestry differs")
    expected = dict(schema="world_reward.frontend_grounding_build.v6", stage="frontend_grounding_build", status="pass",
        phase="complete", producer_revision=BUILD_REV, offline_build_exit_code=0, child_probe_exit_code=0,
        parent_unchanged_verified=True, source_rechecked_before_and_after=True, extension_import_verified=True,
        CUDA_execution_verified=False, replica_ready=False, license_eligibility_verified=False, training_overlap_verified=False)
    require(all(type(build.get(k)) is type(v) and build[k] == v for k,v in expected.items()), "Genuine original CPU build PASS required")
    expected = dict(schema="world_reward.frontend_sam2_kernel_gate.v1", stage="frontend_sam2_kernel_gate", status="pass",
        producer_revision=KERNEL_REV, script_sha256=KERNEL_SOURCE_PIN["sha256"], image_id=IMAGE,
        models_loaded=False, challenge_data_read=False, CUDA_operator_execution_verified=True,
        build_report_identity=BUILD_PIN, replica_ready=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in expected.items()), "Genuine measured CUDA operator PASS required")
    require(receipt.get("operator_source_identities", {}).get("extension") == EXTENSION_PIN, "Measured original CUDA extension identity required")
    child, parent = build["child_image"], build["parent_image"]
    require(child["Id"] == IMAGE and parent["Id"] == kernel.BASE and len(parent["RootFS"]["Layers"]) == 44
        and child["RootFS"]["Layers"][:44] == parent["RootFS"]["Layers"], "Exact preserved original Body parent/new child required")
    owner = hashlib.sha256((BUILD_REV + old_build["closure_sha256"]).encode()).hexdigest()
    require(build.get("owner") == owner, "Original image ownership differs")
    probe = build.get("private_child_probe_log", {})
    require(probe.get("relative_path") == "results/frontend-grounding-build-v6/child-CPU-probe.log", "Exact original CPU probe required")
    require(identity(ROOT/probe["relative_path"], 128*1024) == {k:probe[k] for k in ("bytes","sha256")}, "Original private CPU probe changed")
    if live:
        require(os.uname().nodename == "world-reward-ncc-h100-02", "VM02 host preflight required")
        # This is the actual authenticated old entrypoint, never a spoofed job.
        args = ["/usr/bin/python3", "-I", "-B", str(KERNEL_CODE/"infra/frontend_sam2_kernel_gate.py"), "--verify",
            "--build-revision", BUILD_REV, "--build-report-sha256", BUILD_PIN["sha256"], "--build-report-bytes", str(BUILD_PIN["bytes"]), "--build-script-sha256", BUILD_SHA]
        env = dict(PATH="/usr/bin:/bin", HOME="/nonexistent", WR_ROOT=str(ROOT), WR_CODE=str(KERNEL_CODE),
            WR_CODE_REVISION=KERNEL_REV, PYTHONDONTWRITEBYTECODE="1")
        result = subprocess.run(args, env=env, capture_output=True, timeout=45, check=False)
        require(result.returncode == 0 and result.stdout.strip() == b"kernel_gate_bindings_verified", "Original source-bound live image verification failed")
    contract = selected.load_contract(ROOT, MANIFEST)
    return dict(selected_contract=contract, source_binding=own, build_report_identity=BUILD_PIN,
        kernel_report_identity=KERNEL_PIN, build_source=old_build, kernel_source=old_kernel,
        child_image=child, parent_image=parent, extension_identity=EXTENSION_PIN, source_files=build["source_files"])


def selected_assets(contract, constants):
    receipt_name = "results/weights-acquisition.json"; row = contract["entries"][receipt_name]
    require(row.get("type") == "file" and row.get("role") == "source_receipt", "Original safe acquisition receipt required")
    require(identity(DEST/receipt_name, 2_000_000) == {k:row[k] for k in ("bytes","sha256")}, "Original acquisition receipt changed")
    result = {}
    for relative, (digest, size) in constants.items():
        name = "weights/" + relative; row = contract["entries"].get(name, {})
        require(row.get("type") == "file" and {k:row.get(k) for k in ("bytes","sha256")} == dict(bytes=size,sha256=digest), "Independent selected model asset differs")
        actual = identity(DEST/name)
        require(actual == dict(bytes=size,sha256=digest), "Selected model bytes changed")
        result[relative] = actual
    return result


def installed_sam2(proof):
    """Installed Python source/extension match authenticated build, before models."""
    import sam2
    from sam2 import _C
    folder = canonical(Path(sam2.__file__).parent)
    expected = {n[len("sam2/sam2/"):]:row for n,row in proof["source_files"].items()
        if n.startswith("sam2/sam2/") and n.endswith(".py")}
    actual = selected.python_inventory(folder)
    require(set(actual) == set(expected), "Installed SAM2 Python source set differs")
    for name,path in actual.items():
        _,pin = selected.read(path, empty=True, readonly=False)
        require(all(pin[k] == expected[name].get(k) for k in ("bytes","sha256")), "Installed SAM2 Python bytes changed")
    name = "configs/sam2.1/sam2.1_hiera_l.yaml"
    _,pin = selected.read(folder/name, readonly=False)
    require(all(pin[k] == proof["source_files"].get("sam2/sam2/"+name,{}).get(k) for k in ("bytes","sha256")), "Installed original SAM2 configuration changed")
    _,pin = selected.read(canonical(Path(_C.__file__)), 20_000_000, readonly=False)
    require(Path(_C.__file__).parent == folder and {k:pin[k] for k in ("bytes","sha256")} == EXTENSION_PIN, "Installed original CUDA extension changed")
    return dict(python_files=len(actual), extension_identity=EXTENSION_PIN)


def public_inputs(root, pins_path):
    directory = canonical(Path(root)/BASE/"inputs"); pin_identity = identity(pins_path,200000)
    pins = strict_json(Path(pins_path).read_bytes())
    require(pins.get("schema") == "world_reward.bridge_rgb_anchor_input_pins.v1", "Independently frozen NEW RGB pins required")
    files = pins.get("public_files")
    validate_file_pins(files,{"manifest.json",*FILENAMES},20_000_000)
    require({p.name for p in directory.iterdir()} == set(files), "No recipe/camera/private file in RGB input directory")
    frozen = {Path(pins_path):pin_identity}
    for name,pin in files.items():
        require(identity(directory/name,20_000_000) == pin, "Frozen NEW public RGB bytes differ")
        frozen[directory/name] = pin
    manifest = strict_json((directory/"manifest.json").read_bytes())
    require(type(manifest) is dict and set(manifest) == {"schema","images"}
        and manifest["schema"] == "world_reward.bridge_rgb_anchor_public.v1" and len(manifest["images"]) == 4,
        "Exact RGB-only four-anchor public schema required")
    records = []
    for clip,row in enumerate(manifest["images"]):
        require(type(row) is dict and set(row) == {"clip_id","frame_id","file","sha256","bytes","width","height"}
            and type(row["clip_id"]) is int and row["clip_id"] == clip and type(row["frame_id"]) is int and row["frame_id"] == 0
            and row["file"] == FILENAMES[clip] and type(row["width"]) is int and row["width"] == 640
            and type(row["height"]) is int and row["height"] == 480 and type(row["bytes"]) is int
            and {k:row[k] for k in ("bytes","sha256")} == files[row["file"]],
            "Original ordered RGB indices/grid/identity differ")
        records.append({**row,"path":directory/row["file"]})
    recheck(frozen)
    return records,frozen


def recheck(frozen):
    require(all(identity(path) == pin for path,pin in frozen.items()), "Frozen source/public artifacts changed")


def load_masks(root, pins_path, records):
    directory = canonical(Path(root)/BASE/"automatic_masks"); ownpin = identity(pins_path,200000)
    pins = strict_json(Path(pins_path).read_bytes()); files = pins.get("files",{})
    names = {"report.json",*[label+"_"+name for label in ("person","object") for name in FILENAMES]}
    require(pins.get("schema") == "world_reward.bridge_rgb_anchor_mask_pins.v1"
        and re.fullmatch("[0-9a-f]{40}",str(pins.get("producer_revision")))
        and re.fullmatch("[0-9a-f]{64}",str(pins.get("script_sha256"))),
        "Exact nine independently frozen automatic mask files required")
    validate_file_pins(files,names,2_000_000)
    require({p.name for p in directory.iterdir()} == names, "Complete no-extra automatic mask inventory required")
    frozen = {Path(pins_path):ownpin}
    for name,pin in files.items():
        require(identity(directory/name,2_000_000) == pin, "Frozen automatic mask bytes changed")
        frozen[directory/name] = pin
    receipt = strict_json((directory/"report.json").read_bytes())
    wanted = dict(schema="world_reward.bridge_rgb_anchor_masks.v1",stage="bridge_rgb_anchor_automatic_masks",status="pass",phase="complete",
        image_id=IMAGE,frames=4,ground_truth_used=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],network="none",
        confidence=.3,text_threshold=.25,nms_iou=.7,ambiguity_margin=.05,detector_calls=8,sam2_calls=8,
        build_report_identity=BUILD_PIN,kernel_report_identity=KERNEL_PIN)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in wanted.items())
        and receipt.get("producer_revision") == pins.get("producer_revision")
        and receipt.get("script_sha256") == pins.get("script_sha256"), "Actual full automatic model/source receipt required")
    rows = receipt.get("images",[]); require(type(rows) is list and len(rows) == len(records) == 4, "All four masks required")
    for original,row in zip(records,rows):
        require(type(row) is dict and all(type(row.get(k)) is type(v) and row[k] == v for k,v in original.items() if k != "path"), "Mask/RGB original indices or grid changed")
        for label,query in (("person","person."),("object","bottle.")):
            part = row.get(label,{}); name = label+"_"+original["file"]
            require(part.get("query") == query and part.get("mask_file") == name and type(part.get("mask_pixels")) is int
                and part["mask_pixels"] > 0 and dict(bytes=part.get("mask_bytes"),sha256=part.get("mask_sha256")) == files[name], "Frozen automatic mask provenance differs")
    recheck(frozen)
    return rows,frozen


def control_paths():
    return (BUILD_CODE.parent,KERNEL_CODE.parent,BUILD_REPORT,KERNEL_REPORT,
        ROOT/"results/frontend-grounding-build-v6/child-CPU-probe.log",MANIFEST,EXTRACTION,DEST/"results/weights-acquisition.json")
