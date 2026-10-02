"""Bounded Azure-only transfer of the new own camera cohort, not predictions.

Only nine RGBs, their manifest, two private camera-evaluation files and the
original MHR semantic receipt are transported. No models, challenge data,
credentials, Docker state or user directories enter the archive.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path
import platform
import re
import tarfile

from research_transfer import reject_secrets

BASE = "validation/perspective_rgb_v1"
SEMANTIC = "results/mhr-finger-semantics-v4.json"
MANIFEST = "perspective-transfer.json"
MAX_BYTES = 100_000_000
MODEL_SHA = "352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc"
BODY_SHA = "b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf"
BODY_REV = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
# Explicit source-only audit closure; none of these model/raster modules import
# into this standard-library transfer runtime.
SOURCE_FILES = {"script_sha256":"/infra/perspective_rgb_render.py",
    "render_helper_sha256":"/infra/hand_synthetic_render.py",
    "joint_helper_sha256":"/infra/joint_rgb_render.py", "camera_helper_sha256":"/infra/camera_render.py"}
SEMANTIC_SOURCE = "/infra/mhr_finger_semantics_gate.py"
NAMES = tuple([BASE+"/inputs/manifest.json"]+[BASE+f"/inputs/case_{i:02d}.png" for i in range(9)]
              +[BASE+"/eval_private/render-report.json", BASE+"/eval_private/calibration_truth.npz", SEMANTIC])


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for b in iter(lambda: stream.read(1024*1024), b""): h.update(b)
    return h.hexdigest()


def regular(path):
    path = Path(path)
    if path.resolve() != path.absolute() or not path.is_file(): raise ValueError("Canonical regular file required")
    return path


def metadata(read):
    public_bytes = read(NAMES[0]); public = json.loads(public_bytes)
    if set(public) != {"schema", "images"} or public["schema"] != "world-reward-perspective-rgb-v1" or len(public["images"]) != 9:
        raise ValueError("Exact new RGB-only cohort required")
    expected = {NAMES[0]: hashlib.sha256(public_bytes).hexdigest()}
    for i, row in enumerate(public["images"]):
        if (set(row) != {"file", "sha256", "width", "height"} or row["file"] != f"case_{i:02d}.png"
                or type(row["width"]) is not int or type(row["height"]) is not int
                or (row["width"],row["height"]) != (1024,768) or not re.fullmatch("[0-9a-f]{64}", row["sha256"])):
            raise ValueError("Public ordered image inventory differs")
        expected[BASE+"/inputs/"+row["file"]] = row["sha256"]
    report_bytes = read(BASE+"/eval_private/render-report.json"); report = json.loads(report_bytes)
    required = {"stage":"own_procedural_perspective_rgb_render", "status":"pass",
                "challenge_inputs_used":False, "inference_performed":False, "actual_MHR_reference_used":True,
                "actual_reference_forward_calls":2, "synthetic_truth_used_for_rendering_only":True,
                "public_manifest_sha256":expected[NAMES[0]]}
    if any(type(report.get(k)) is not type(v) or report.get(k) != v for k,v in required.items()):
        raise ValueError("Actual completed independent render required")
    source_checks={"model_sha256":MODEL_SHA,
        **{key:digest(Path(__file__).with_name(Path(name).name)) for key,name in SOURCE_FILES.items()},
        "image_id":"sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"}
    if (not re.fullmatch("[0-9a-f]{40}", report.get("code_revision", ""))
            or any(report.get(k)!=v for k,v in source_checks.items())):
        raise ValueError("Immutable current renderer sources/model/image required")
    expected[BASE+"/eval_private/render-report.json"] = hashlib.sha256(report_bytes).hexdigest()
    expected[BASE+"/eval_private/calibration_truth.npz"] = report["truth_sha256"]
    expected[SEMANTIC] = report["semantic_report_sha256"]
    semantic = json.loads(read(SEMANTIC))
    semantic_checks={"stage":"own_reference_mhr_finger_semantics_support","status":"pass", "phase_A_verified":True,
        "phase_B_verified":True,"named_finger_joint_partition_verified":True,"excluded_joint_invariance_verified":True,
        "correctives_skeleton_bit_identical":True,"model_sha256":MODEL_SHA,"body_checkpoint_sha256":BODY_SHA,
        "body_revision":BODY_REV,"network":"none","challenge_inputs_used":False,
        "script_sha256":digest(Path(__file__).with_name(Path(SEMANTIC_SOURCE).name)),
        "deterministic_algorithms":True,"CUBLAS_WORKSPACE_CONFIG":":4096:8","TF32":False,"jit_optimized_execution":False}
    if any(type(semantic.get(k)) is not type(v) or semantic.get(k)!=v for k,v in semantic_checks.items()):
        raise ValueError("Actual pinned passing semantic receipt required")
    if semantic.get("source_image_id") != report.get("image_id"):
        raise ValueError("Semantic and renderer images differ")
    reject_secrets(public); reject_secrets(report); reject_secrets(semantic)
    return expected


def inventory(root):
    root = Path(root); expected = metadata(lambda name: regular(root/name).read_bytes())
    if {p.name for p in (root/BASE/"inputs").iterdir()} != {"manifest.json", *[f"case_{i:02d}.png" for i in range(9)]}:
        raise ValueError("Only new public RGB inputs may be transferred")
    if {p.name for p in (root/BASE/"eval_private").iterdir()} != {"render-report.json", "calibration_truth.npz"}:
        raise ValueError("Only new camera-evaluation truth may be transferred")
    entries = []
    for name in NAMES:
        path = regular(root/name); size = path.stat().st_size; sha = digest(path)
        if not 0 < size <= MAX_BYTES or sha != expected[name]: raise ValueError("Frozen source bytes differ")
        entries.append({"path":name, "bytes":size, "sha256":sha})
    if sum(e["bytes"] for e in entries) > MAX_BYTES: raise ValueError("Camera-cohort size bound exceeded")
    return {"schema":"world-reward-perspective-transfer-v1", "entries":entries,
            "challenge_inputs_included":False, "models_included":False, "predictions_included":False,
            "credentials_included":False, "private_camera_evaluation_included":True}


def export(root, archive):
    manifest = inventory(root); data = (json.dumps(manifest,sort_keys=True)+"\n").encode()
    fd=os.open(archive,os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o600)
    with os.fdopen(fd,"wb") as stream, tarfile.open(fileobj=stream,mode="w",format=tarfile.USTAR_FORMAT) as tar:
        m = tarfile.TarInfo(MANIFEST); m.size=len(data); m.mode=0o600; tar.addfile(m,io.BytesIO(data))
        for e in manifest["entries"]:
            m=tarfile.TarInfo(e["path"]); m.size=e["bytes"]; m.mode=0o400
            with regular(Path(root)/e["path"]).open("rb") as file: tar.addfile(m,file)
    if inventory(root) != manifest: raise ValueError("Source changed during transfer export")
    return digest(archive)


def verify(archive, expected_sha):
    archive=regular(archive)
    if (not re.fullmatch("[0-9a-f]{64}",expected_sha) or archive.stat().st_size>MAX_BYTES+100_000
            or digest(archive)!=expected_sha): raise ValueError("Independent archive SHA/size required")
    with tarfile.open(archive,mode="r:") as tar:
        members=tar.getmembers()
        if (len(members)!=len(NAMES)+1 or [m.name for m in members] != [MANIFEST,*NAMES]
                or any(not m.isfile() or m.linkname or not 0<m.size<=MAX_BYTES for m in members)
                or members[0].size>100_000): raise ValueError("Exact regular transfer inventory/order required")
        manifest=json.load(tar.extractfile(members[0])); reject_secrets(manifest)
        expected_flags={"schema":"world-reward-perspective-transfer-v1", "challenge_inputs_included":False,
            "models_included":False, "predictions_included":False, "credentials_included":False,
            "private_camera_evaluation_included":True}
        if set(manifest)!={*expected_flags,"entries"} or any(type(manifest.get(k)) is not type(v) or manifest[k]!=v for k,v in expected_flags.items()):
            raise ValueError("Transfer purpose/flags differ")
        entries=manifest["entries"]
        if not isinstance(entries,list) or len(entries)!=len(NAMES): raise ValueError("Exact transfer entry count required")
        if sum(m.size for m in members)>MAX_BYTES+100_000: raise ValueError("Expanded archive exceeds bound")
        expected=metadata(lambda name: tar.extractfile(tar.getmember(name)).read())
        for m,e in zip(members[1:],entries):
            if (set(e)!={"path","bytes","sha256"} or e["path"]!=m.name or type(e["bytes"]) is not int
                    or e["bytes"]!=m.size or m.mode!=0o400 or e["sha256"]!=expected[m.name]
                    or hashlib.sha256(tar.extractfile(m).read()).hexdigest()!=e["sha256"]):
                raise ValueError("Verified transfer member bytes differ")
    return manifest


def extract(root, archive, expected_sha):
    manifest=verify(archive,expected_sha); root=Path(root)
    if (root/BASE).exists() or (root/BASE).is_symlink(): raise FileExistsError("New destination cohort must not exist")
    for name in NAMES:
        path=root/name
        if any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in path.parents): raise ValueError("Canonical destination parents required")
        if path.exists() or path.is_symlink(): raise FileExistsError("Never overwrite destination artifacts")
    with tarfile.open(archive,mode="r:") as tar:
        for e in manifest["entries"]:
            path=root/e["path"]; path.parent.mkdir(parents=True,exist_ok=True,mode=0o700)
            with path.open("xb") as file, tar.extractfile(e["path"]) as source:
                for b in iter(lambda:source.read(1024*1024),b""): file.write(b)
            path.chmod(0o400 if "/eval_private/" in e["path"] else 0o444)
    if inventory(root)!=manifest: raise ValueError("Extracted cohort identity differs")
    return manifest


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__,allow_abbrev=False)
    parser.add_argument("mode",choices=("export","extract")); parser.add_argument("--archive",type=Path,required=True)
    parser.add_argument("--sha256"); args=parser.parse_args(argv)
    root=Path(os.environ["WR_ROOT"])
    if platform.system()!="Linux" or root!=Path("/srv/scenesmith/world-reward") or root.resolve()!=root or os.geteuid()!=1000:
        raise RuntimeError("Transfer stays on canonical Azure UID1000 task root")
    if args.archive.resolve()!=args.archive.absolute(): raise ValueError("Canonical transfer archive required")
    if args.mode=="export":
        if args.sha256 is not None: raise ValueError("Export does not accept a claimed SHA")
        print(json.dumps({"archive_sha256":export(root,args.archive),"bytes":args.archive.stat().st_size}))
    else:
        if args.sha256 is None: raise ValueError("Independent export SHA required before extraction")
        manifest=extract(root,args.archive,args.sha256)
        print(json.dumps({"status":"pass","archive_sha256":args.sha256,"files":len(manifest["entries"]),
                          "models_included":False,"challenge_inputs_included":False}))


if __name__=="__main__":main()
