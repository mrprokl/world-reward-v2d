"""Public YCBV RGB → fixed object. detection and unmodified native SAM2 video masks."""
from __future__ import annotations

import argparse
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import re
import signal
import stat
import sys
import time

import bridge_frontend_bindings as binding
import frontend_sam2_kernel_gate as kernel
import hand_synthetic_masks as detector_policy
import ycbv_point_depth as public
from world_reward.prompt_selection import BoxDetection

ROOT, BASE, IMAGE = binding.ROOT, public.BASE, binding.IMAGE
ENTRY, OUTPUT = "run_ycbv_point_masks", "automatic_masks_v1"
STAGE, BUDGET = "public_ycbv_point_native_object_masks", 600
PIN_FILE = "configs/ycbv_point_input_pins.json"
HELPERS = ("infra/ycbv_point_masks.py", "infra/run_ycbv_point_masks.sh", "infra/ycbv_point_depth.py",
    "infra/bridge_frontend_bindings.py", "infra/frontend_selected_assets.py", "infra/frontend_sam2_kernel_gate.py",
    "infra/hand_synthetic_masks.py", "infra/tudl_holdout_inputs.py", binding.CONFIG,
    "configs/frontend_asset_archive_pins.json", PIN_FILE,
    "src/world_reward/__init__.py", "src/world_reward/prompt_selection.py")
NATIVE_MASK_SHA = "5193404292cfc7e66053e261049e58b76d3484f92ac1124c481f9951cc4907ec"
require = binding.require


def source_binding(root, code, revision, *, container=False):
    require(root == ROOT and code == root/"jobs"/revision/ENTRY/"code"
        and Path(__file__).resolve() == code/HELPERS[0], "Actual immutable masks entry required")
    require(binding.identity(code/"infra/frontend_sam2_kernel_gate.py", 200000) == binding.KERNEL_SOURCE_PIN,
        "Original pinned kernel proof helper required")
    if not container:
        return dict(scope="complete_host_dispatch", **kernel.closure(code, revision, ENTRY, HELPERS))
    modules = ((binding,"bridge_frontend_bindings"),(kernel,"frontend_sam2_kernel_gate"),
        (detector_policy,"hand_synthetic_masks"),(public,"ycbv_point_depth"),
        (binding.selected,"frontend_selected_assets"),(public.files,"tudl_holdout_inputs"))
    require(all(Path(module.__file__).resolve() == code/f"infra/{name}.py" for module,name in modules),
        "Actual mounted blind helper modules required")
    helpers = {name:binding.identity(code/name,2_000_000) for name in HELPERS}
    markers = {name:public.marker_identity(code.parent/name) for name in ("revision","source-sha256")}
    require((code.parent/"revision").read_bytes() == (revision+"\n").encode()
        and re.fullmatch(b"[0-9a-f]{64}\n",(code.parent/"source-sha256").read_bytes()), "Actual narrow dispatch markers differ")
    require({str(p.relative_to(code)) for p in code.rglob("*") if p.is_file()} == set(HELPERS),
        "Only the exact blind source/public/model-proof configuration closure may be mounted")
    return dict(scope="narrow_container_source",helpers=helpers,markers=markers)


def frontend_proof(code, *, live=False):
    """Reuse measured image/kernel/extraction contracts, never spoof an old entry."""
    build = binding.pinned(binding.BUILD_REPORT, binding.BUILD_PIN)
    receipt = binding.pinned(binding.KERNEL_REPORT, binding.KERNEL_PIN)
    old = kernel.closure(binding.BUILD_CODE, binding.BUILD_REV, "run_frontend_grounding_build", kernel.BUILD_HELPERS)
    gate = kernel.closure(binding.KERNEL_CODE, binding.KERNEL_REV, "run_frontend_sam2_kernel_gate",
        ("infra/frontend_sam2_kernel_gate.py", "infra/run_frontend_sam2_kernel_gate.sh", binding.CONFIG))
    require(old["helpers"]["infra/frontend_grounding_build.py"]["sha256"] == binding.BUILD_SHA
        and gate["helpers"]["infra/frontend_sam2_kernel_gate.py"] == binding.KERNEL_SOURCE_PIN
        and build.get("source_binding") == old and receipt.get("source_binding") == gate
        and binding.identity(code/binding.CONFIG) == old["helpers"][binding.CONFIG] == gate["helpers"][binding.CONFIG],
        "Measured original source/build/kernel ancestry differs")
    expected = dict(schema="world_reward.frontend_grounding_build.v6", stage="frontend_grounding_build", status="pass",
        phase="complete", producer_revision=binding.BUILD_REV, offline_build_exit_code=0, child_probe_exit_code=0,
        parent_unchanged_verified=True, source_rechecked_before_and_after=True, extension_import_verified=True,
        original_grounding_image_parity_claimed=False, CUDA_execution_verified=False, replica_ready=False,
        license_eligibility_verified=False, training_overlap_verified=False)
    require(all(type(build.get(k)) is type(v) and build[k] == v for k,v in expected.items()), "Actual complete CPU build required")
    expected = dict(schema="world_reward.frontend_sam2_kernel_gate.v1", stage="frontend_sam2_kernel_gate", status="pass",
        producer_revision=binding.KERNEL_REV, script_sha256=binding.KERNEL_SOURCE_PIN["sha256"], image_id=IMAGE,
        models_loaded=False, challenge_data_read=False, CUDA_operator_execution_verified=True,
        build_report_identity=binding.BUILD_PIN, replica_ready=False)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in expected.items())
        and receipt.get("operator_source_identities", {}).get("extension") == binding.EXTENSION_PIN,
        "Actual measured native CUDA operator required")
    owner = hashlib.sha256((binding.BUILD_REV+old["closure_sha256"]).encode()).hexdigest()
    child, parent = build["child_image"], build["parent_image"]
    require(child["Id"] == IMAGE and parent["Id"] == kernel.BASE and child["Architecture"] == "amd64"
        and child["Os"] == "linux" and len(parent["RootFS"]["Layers"]) == 44
        and child["RootFS"]["Layers"][:44] == parent["RootFS"]["Layers"] and build.get("owner") == owner,
        "Exact preserved image/rootfs ownership required")
    probe = build.get("private_child_probe_log", {})
    require(probe.get("relative_path") == "results/frontend-grounding-build-v6/child-CPU-probe.log"
        and binding.identity(ROOT/probe["relative_path"], 128*1024) == {k:probe[k] for k in ("bytes","sha256")},
        "Original measured CPU import log changed")
    proof = dict(child_image=child, parent_image=parent, owner=owner, source_files=build["source_files"],
        build_report_identity=binding.BUILD_PIN, kernel_report_identity=binding.KERNEL_PIN,
        extension_identity=binding.EXTENSION_PIN, selected_contract=binding.selected.load_contract(ROOT, binding.MANIFEST))
    if live: kernel.validate_live_image(proof)
    return proof


def host_proof(root, code, revision):
    source = source_binding(root, code, revision)
    pins = binding.strict_json((code/PIN_FILE).read_bytes()); public.validate_pins(pins)
    pin = pins["acquisition_report"]; path = root/BASE/"report.json"
    receipt = binding.pinned(path, {k:pin[k] for k in ("bytes","sha256")})
    expected = dict(stage="external_ycbv_contiguous_rgb_only_acquisition", status="pass", phase="complete",
        producer_revision=pin["producer_revision"], script_sha256=pin["script_sha256"], dataset_revision=public.REVISION,
        license="MIT", image_id=public.IMAGE, device="cpu", gpu_used=False, inference_performed=False,
        challenge_inputs_used=False, models_downloaded=False, train_downloaded=False, sparse_test_downloaded=False,
        private_annotations_exported_as_inference_inputs=False, selection_before_private_annotation_values=True,
        selected_frames=288, all_instances_retained=True, disposable_archives_removed=True, source_rehashed_after=True)
    require(all(type(receipt.get(k)) is type(v) and receipt[k] == v for k,v in expected.items())
        and receipt.get("public_manifest") == pins["manifest"], "Exact successful original acquisition required")
    original = root/"jobs"/pin["producer_revision"]/"run_ycbv_point_acquire/code"
    actual = dict(files={n:binding.identity(original/n) for n in public.ACQUISITION_FILES},
        markers={n:public.marker_identity(original.parent/n) for n in ("revision","source-sha256")})
    require(actual == receipt.get("source_helpers") and actual["files"][public.ACQUISITION_FILES[0]]["sha256"] == pin["script_sha256"]
        and (original.parent/"revision").read_bytes() == (pin["producer_revision"]+"\n").encode()
        and re.fullmatch(b"[0-9a-f]{64}\n", (original.parent/"source-sha256").read_bytes()), "Actual acquisition source/markers differ")
    _, rgb = public.public_inputs(root/BASE/"inputs", pins)
    proof = frontend_proof(code, live=True)
    assets = selected_model_assets(proof)
    require(binding.identity(path) == {k:pin[k] for k in ("bytes","sha256")}, "Acquisition receipt changed")
    return hashlib.sha256(json.dumps(dict(source=source,pins=pins,public=rgb,assets=assets,
        build=binding.BUILD_PIN,kernel=binding.KERNEL_PIN),sort_keys=True).encode()).hexdigest()


def selected_model_assets(proof):
    assets = binding.selected_assets(proof["selected_contract"], detector_policy.ASSETS)
    receipt = binding.strict_json((binding.DEST/"results/weights-acquisition.json").read_bytes())
    for repo, revision, folder in (("IDEA-Research/grounding-dino-base", detector_policy.DETECTOR_REVISION, "grounding_dino"),
        ("facebook/sam2.1-hiera-large", detector_policy.SAM2_REVISION, "sam2")):
        rows = [r for r in receipt.get("assets",[]) if r.get("repo_id") == repo]
        require(len(rows) == 1 and rows[0].get("revision") == revision
            and rows[0].get("path") == str(ROOT/"weights"/folder), "Original published model revision/path differs")
    return assets


def installed_sources(proof):
    import importlib.util
    result = {"sam2": binding.installed_sam2(proof), "v2d": {}}
    for module,prefix in (("v2d.common", "nvidia/reconstruction/modules/v2d_common/"),
        ("v2d.sam2.lib", "nvidia/reconstruction/modules/v2d_sam2/lib/")):
        spec = importlib.util.find_spec(module)
        require(spec is not None and spec.submodule_search_locations, "Original installed v2d package required")
        directory = Path(next(iter(spec.submodule_search_locations)))
        expected = {n[len(prefix):]:r for n,r in proof["source_files"].items() if n.startswith(prefix) and n.endswith(".py")}
        actual = binding.selected.python_inventory(directory)
        require(expected and set(actual) == set(expected), "Complete original installed v2d Python set differs")
        rows = {}
        for name,path in actual.items():
            _,pin = binding.selected.read(path, empty=True, readonly=False)
            require(all(pin[k] == expected[name].get(k) for k in ("bytes","sha256","git_blob_sha1")), "Installed v2d source bytes differ")
            rows[name] = {k:pin[k] for k in ("bytes","sha256")}
        result["v2d"][module] = rows
    require(result["v2d"]["v2d.sam2.lib"]["video_to_masks.py"]["sha256"] == NATIVE_MASK_SHA,
        "Exact native video_to_masks implementation required")
    versions = {n:metadata.version(n) for n in ("transformers","tokenizers","torch","numpy")}
    require(versions == dict(transformers="4.53.3",tokenizers="0.21.4",torch="2.5.1+cu124",numpy="1.26.3"), "Measured image package versions differ")
    return {**result,"versions":versions}


def read_rgb(record):
    import numpy as np
    from PIL import Image
    before = binding.identity(record["path"])
    require(before["sha256"] == record["sha256"], "Original RGB changed before decode")
    with Image.open(record["path"]) as image:
        require(image.format == "PNG" and image.mode == "RGB" and image.size == (640,480), "Original RGB PNG required")
        value = np.asarray(image).copy()
    require(value.dtype == np.uint8 and value.shape == (480,640,3) and binding.identity(record["path"]) == before, "RGB decode bytes/grid changed")
    return value


def stage_frames(records, directory):
    """Exclusive byte copies only: retain source hashes and original 0..95 order."""
    require(len(records) == 96 and [r["frame_id"] for r in records] == list(range(96)), "Full contiguous scene required")
    directory.mkdir(mode=0o700); owner = directory.lstat(); frozen = {}
    try:
        for record in records:
            before = binding.identity(record["path"]);raw = record["path"].read_bytes()
            require(before["sha256"] == record["sha256"] and hashlib.sha256(raw).hexdigest() == record["sha256"]
                and binding.identity(record["path"]) == before, "Public RGB changed before staging")
            path = directory/f'{record["frame_id"]:06d}.png'
            with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o444),"wb") as stream:
                stream.write(raw);stream.flush();os.fsync(stream.fileno())
            path.chmod(0o444); state = path.lstat()
            frozen[path.name] = dict(identity=binding.identity(path), state=(state.st_dev,state.st_ino,state.st_uid,state.st_mode))
    except BaseException:
        remove_stage(directory,owner,frozen);raise
    return owner, frozen


def remove_stage(directory, owner, frozen):
    state = directory.lstat()
    require(stat.S_ISDIR(state.st_mode) and (state.st_dev,state.st_ino,state.st_uid,state.st_mode) ==
        (owner.st_dev,owner.st_ino,owner.st_uid,owner.st_mode) and {p.name for p in directory.iterdir()} == set(frozen), "Owned RGB stage changed; do not delete unknown files")
    for name,row in frozen.items():
        path = directory/name;state = path.lstat()
        require((state.st_dev,state.st_ino,state.st_uid,state.st_mode) == row["state"]
            and binding.identity(path) == row["identity"], "Owned staged file changed; do not delete")
    for name in frozen: (directory/name).unlink()
    directory.rmdir()


def mask_inventory(folder, records):
    import numpy as np
    from PIL import Image
    require({p.name for p in folder.iterdir()} == {"1"} and {p.name for p in (folder/"1").iterdir()} == {f"{i:06d}.png" for i in range(96)}, "Exactly object1/full96 native masks required")
    rows = []
    for record in records:
        path = binding.canonical(folder/"1"/f'{record["frame_id"]:06d}.png'); s = path.lstat()
        require(stat.S_ISREG(s.st_mode) and s.st_nlink == 1 and s.st_uid == os.geteuid(), "Native mask file must be unaliased regular")
        with Image.open(path) as image:
            require(image.format == "PNG" and image.mode == "L" and image.size == (640,480), "Native mask PNG/grid differs")
            value = np.asarray(image)
        require(value.dtype == np.uint8 and np.all((value == 0)|(value == 255)) and np.count_nonzero(value) > 0,
            "All original frames require nonempty native binary masks")
        path.chmod(0o444); pin = binding.identity(path, 2_000_000)
        rows.append(dict(scene_id=record["scene_id"],frame_id=record["frame_id"],rgb_file=record["file"],rgb_sha256=record["sha256"],
            file=f'scene_{record["scene_id"]:06d}/masks/1/{path.name}',mask_pixels=int(np.count_nonzero(value)),**pin))
    return rows


def observe(records, out, report, persist, *, detect, propagate):
    require(len(records) == 288, "All three full public scenes required")
    seeds = []
    for scene_index, scene in enumerate(public.SCENES):
        record = records[scene_index*96];rgb = read_rgb(record)
        report.update(phase="automatic_detection", current_scene=scene);report["detector_attempts"] += 1;persist()
        boxes = detect(rgb,"object.");report["detector_calls"] += 1
        chosen = detector_policy.select_person(boxes,640,480)
        seeds.append(dict(scene_id=scene,frame_id=0,rgb_file=record["file"],rgb_sha256=record["sha256"],query="object.",
            box=list(chosen.box),detector_score=chosen.score,candidate_count=len(boxes),decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest()))
        report["seeds"] = seeds;persist()
    report["all_three_detections_passed_before_SAM2"] = True
    for i, seed in enumerate(seeds):
        rows = records[i*96:(i+1)*96];scene_dir = out/f'scene_{seed["scene_id"]:06d}';scene_dir.mkdir(mode=0o700)
        stage = out/f'.rgb-stage-{seed["scene_id"]:06d}';owner,frozen = stage_frames(rows,stage)
        prompt = dict(prompts=[dict(frame_index=0,object_id=1,points=None,point_labels=None,
            box=dict(zip(("x0","y0","x1","y1"),seed["box"])),mask_path=None)])
        path = scene_dir/"prompts.json"
        with os.fdopen(os.open(path,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400),"w") as stream:
            json.dump(prompt,stream,allow_nan=False);stream.flush();os.fsync(stream.fileno())
        started=time.monotonic()
        try:
            report.update(phase="native_SAM2_propagation", current_scene=seed["scene_id"]);report["sam2_attempts"] += 1;persist()
            propagate(str(stage),str(path),str(scene_dir/"masks"),str(binding.DEST/"weights/sam2"));report["sam2_calls"] += 1
            masks = mask_inventory(scene_dir/"masks",rows)
            report["masks"].extend(masks);report["frames_completed"] += len(masks)
            seed.update(propagation_elapsed_seconds=time.monotonic()-started,prompt_identity=binding.identity(path));persist()
        finally: remove_stage(stage,owner,frozen)
    require(report["detector_attempts"] == report["detector_calls"] == report["sam2_attempts"] == report["sam2_calls"] == 3
        and report["frames_completed"] == 288, "Exact full-scene model/coverage counts required")


def run(root, code, revision, out, report, persist):
    source = source_binding(root,code,revision,container=True);proof = frontend_proof(code)
    pins = binding.strict_json((code/PIN_FILE).read_bytes());records, rgb_proof = public.public_inputs(root/BASE/"inputs",pins)
    assets = selected_model_assets(proof)
    installed = None
    try:
        report.update(source_binding=source,public_manifest=pins["manifest"],acquisition_report_identity=pins["acquisition_report"],
            public_RGB_identities=rgb_proof["RGB_identities"],model_assets=assets,build_report_identity=binding.BUILD_PIN,
            kernel_report_identity=binding.KERNEL_PIN,phase="native_model_load");persist()
        import numpy as np
        import torch
        from PIL import Image
        from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
        require(torch.cuda.is_available() and "H100" in torch.cuda.get_device_name(), "Actual H100 CUDA required")
        require(not any((root/name).exists() for name in (BASE+"/eval_private",BASE+"/report.json","data","vendor")), "No private acquisition/annotation/challenge mount allowed")
        torch.manual_seed(0);np.random.seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(False,warn_only=False)
        installed = installed_sources(proof);report["installed_sources"] = installed;persist()
        from v2d.sam2.lib.video_to_masks import video_to_masks
        processor = AutoProcessor.from_pretrained(binding.DEST/"weights/grounding_dino",local_files_only=True)
        detector = AutoModelForZeroShotObjectDetection.from_pretrained(binding.DEST/"weights/grounding_dino",local_files_only=True).to("cuda").eval()
        released = False
        def detect(rgb,query):
            inputs=processor(images=Image.fromarray(rgb),text=query,return_tensors="pt").to("cuda")
            with torch.inference_mode():
                result=processor.post_process_grounded_object_detection(detector(**inputs),inputs.input_ids,
                    threshold=.3,text_threshold=.25,target_sizes=[(480,640)])[0]
            return tuple(BoxDetection(tuple(box),float(score)) for box,score in zip(result["boxes"].cpu().tolist(),result["scores"].cpu().tolist()))
        def propagate(*args):
            nonlocal released,detector,processor
            if not released:
                del detector,processor;torch.cuda.empty_cache();released=True
            return video_to_masks(*args)
        observe(records,out,report,persist,detect=detect,propagate=propagate)
        torch.cuda.synchronize()
    finally:
        require(source_binding(root,code,revision,container=True) == source and frontend_proof(code) == proof
            and public.public_inputs(root/BASE/"inputs",pins)[1] == rgb_proof
            and selected_model_assets(proof) == assets
            and (installed is None or installed_sources(proof) == installed), "Original public/source/assets/runtime changed")
        report["original_inputs_sources_assets_rehashed_after"] = True;persist()
    for scene_index,scene in enumerate(public.SCENES):
        require({p.name for p in (out/f"scene_{scene:06d}").iterdir()} == {"prompts.json","masks"}
            and mask_inventory(out/f"scene_{scene:06d}/masks",records[scene_index*96:(scene_index+1)*96]) == report["masks"][scene_index*96:(scene_index+1)*96], "Complete frozen native scene inventory changed")
    for row in report["masks"]:
        require(binding.identity(out/row["file"]) == {k:row[k] for k in ("bytes","sha256")}, "Native output mask changed")
    require({p.name for p in out.iterdir()} == {"report.json", ".container.cid",*[f"scene_{s:06d}" for s in public.SCENES]}, "Exclusive full native mask output inventory required")
    report.update(status="pass",phase="complete",all_inputs_sources_assets_outputs_rehashed=True,disposable_RGB_stages_removed=True)


def main(argv=None):
    parser=argparse.ArgumentParser(allow_abbrev=False);parser.add_argument("--host-proof",action="store_true");args=parser.parse_args(argv)
    root=Path(os.environ["WR_ROOT"]);code=Path(os.environ["WR_CODE"]);revision=os.environ["WR_CODE_REVISION"]
    require(sys.platform == "linux" and os.geteuid() == 0, "Restricted UID0 Linux required for original root400 assets")
    if args.host_proof:
        require(os.uname().nodename == "world-reward-ncc-h100-02", "Owned VM02 host required")
        print(host_proof(root,code,revision));return
    require({p.name for p in Path("/sys/class/net").iterdir()} == {"lo"} and os.environ.get("WR_IMAGE_ID") == IMAGE
        and re.fullmatch("[0-9a-f]{64}",os.environ.get("WR_YCBV_HOST_PROOF_SHA256","")), "Source-bound offline preflight required")
    out=binding.canonical(root/BASE/OUTPUT);s=out.lstat()
    require(stat.S_ISDIR(s.st_mode) and s.st_uid == 0 and s.st_mode & 0o777 == 0o700
        and {p.name for p in out.iterdir()} == {".container.cid"}, "Exclusive newly reserved root output required")
    report=dict(schema="world-reward-ycbv-point-masks-v1",stage=STAGE,status="fail",phase="public_integrity",producer_revision=revision,
        script_sha256=binding.identity(Path(__file__))["sha256"],image_id=IMAGE,execution_uid=0,network="none",device="cuda",
        budget_seconds=BUDGET,host_preflight_sha256=os.environ["WR_YCBV_HOST_PROOF_SHA256"],query="object.",confidence=.3,text_threshold=.25,nms_iou=.7,ambiguity_margin=.05,
        ground_truth_used=False,private_annotations_read=False,challenge_inputs_used=False,hand_labeled_test=False,oracle_modes=[],
        license_eligibility_verified=False,training_overlap_verified=False,accuracy_verified=False,adoption_performed=False,
        detector_attempts=0,detector_calls=0,sam2_attempts=0,sam2_calls=0,frames_completed=0,seeds=[],masks=[])
    started=time.monotonic()
    with os.fdopen(os.open(out/"report.json",os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o400),"w") as stream:
        def persist():
            report["elapsed_seconds"]=time.monotonic()-started;stream.seek(0);json.dump(report,stream,allow_nan=False)
            stream.write("\n");stream.truncate();stream.flush();os.fsync(stream.fileno())
        def expired(*_):raise TimeoutError("Frozen all-three-scenes 600s mask budget exceeded")
        handlers={s:signal.signal(s,expired) for s in (signal.SIGALRM,signal.SIGTERM,signal.SIGINT)};signal.alarm(BUDGET)
        try:persist();run(root,code,revision,out,report,persist)
        except BaseException as error:
            report.update(status="fail",error_type=type(error).__name__,error="Native automatic masks failed at recorded phase");raise
        finally:
            signal.alarm(0)
            for s,handler in handlers.items():signal.signal(s,handler)
            persist()
    print(json.dumps({k:report[k] for k in ("stage","status","frames_completed","detector_calls","sam2_calls","elapsed_seconds")}))


if __name__ == "__main__":main()
