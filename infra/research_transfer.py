"""Remote-only exact MoGe/TUD-L transfer; never Docker, credentials or predictions.

Archive hashes must be conveyed independently. Verification finishes before
extracting anything; extraction is exclusive and creates private700/file600.
This is engineering provenance, not a licence/leakage/security attestation.
"""
import argparse
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import signal
import stat
import tarfile

MODEL_REV = "b135031bae30b5ac2ae141a0e68717795ce38340"
MODEL_SHA = "280741fd09bc3f403ccff9967784c2a391b52d2c0742ae3efdb21d9f90cc1a01"
MODEL_BYTES = 1323815904
XET = "9f4c4857a8203605fd29a80f0e81e9ed52fc1654c1e657d437ab29b73d8db37c"
TUDL_REV = "6527f7d4b25d3e2e8dec84529284d9797b15f7b5"
PUBLIC_SHA = "171e89b563520bb9311220b0ddbce068a63d43f843815edf139cb74b61352ecc"
ACQUISITION_SHA = "d096f1eddbca90c8d037ba85e323aa826974c97ba2fd658b0c9cdd76de2dd6ea"
BASE = "validation/tudl_rgb_v1"
HF = "weights/cari4d/hf_home/hub"
REPO = HF+"/models--Ruicheng--moge-2-vitl-normal"
SNAPSHOT, REPO_BLOB, BLOB = REPO+f"/snapshots/{MODEL_REV}/model.pt", REPO+"/blobs/"+MODEL_SHA, HF+"/blobs/9f/"+XET
MANIFEST = "research-transfer-manifest.json"
FRAMES = ((0, 4074, 8227), (3, 4013, 7710), (4, 4028, 7969))


def digest(stream):
    result = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b""): result.update(block)
    return result.hexdigest()


def file_hash(path):
    with path.open("rb") as stream: return digest(stream)


def safe_path(name):
    if not isinstance(name, str) or not name or PurePosixPath(name).is_absolute() or "\\" in name or "\x00" in name or any(p in ("", ".", "..") for p in name.split("/")):
        raise ValueError("Unsafe transfer path")
    return name


def link_target(name, link):
    if not isinstance(link, str) or not link or PurePosixPath(link).is_absolute() or "\\" in link or "\x00" in link: raise ValueError("Unsafe HF relative link")
    target = posixpath.normpath(posixpath.join(posixpath.dirname(name), link))
    safe_path(target)
    if target not in (REPO_BLOB, BLOB): raise ValueError("HF link escapes exact frozen blob whitelist")
    return target


def public_names():
    return [BASE+f"/inputs/scene_{scene:06d}_frame_{frame:06d}.png" for scene, frames in enumerate(FRAMES, 1) for frame in frames]


def private_names():
    source = BASE+"/eval_private/source/"
    names = [source+"licenses/"+n for n in ("huggingface-README.md", "BOP-TUD-L-section.html", "attribution.json")]
    names += [source+"base/tudl/"+n for n in ("camera.json", "dataset_info.md", "test_targets_bop19.json")]
    names += [source+folder+"/"+n for folder in ("models", "models_eval") for n in ("models_info.json", "obj_000001.ply", "obj_000002.ply", "obj_000003.ply")]
    for scene, frames in enumerate(FRAMES, 1):
        prefix = source+f"test/{scene:06d}/"
        names += [prefix+f"scene_{n}.json" for n in ("camera", "gt", "gt_info")]
        names += [prefix+f"depth/{frame:06d}.png" for frame in frames]
        names += [prefix+f"mask_visib/{frame:06d}_000000.png" for frame in frames]
    return names


def whitelist(include_private):
    return {SNAPSHOT, REPO_BLOB, BLOB, "results/weights-acquisition.json", BASE+"/inputs/manifest.json", *public_names(),
            *(private_names()+[BASE+"/eval_private/acquisition-report.json"] if include_private else [])}


def reject_secrets(value):
    if isinstance(value, dict):
        if any(re.search(r"(?i)(?:^|_)(token|password|secret|credential|authorization|api_key)(?:$|_)", k) for k in value): raise ValueError("Secret-like metadata fields forbidden")
        for v in value.values(): reject_secrets(v)
    elif isinstance(value, list):
        for v in value: reject_secrets(v)
    elif isinstance(value, str) and re.search(r"(?:hf_[A-Za-z0-9]{20,}|Bearer\s+\S+|AccountKey=|[?&](?:sig|token)=)", value): raise ValueError("Credential-like metadata values forbidden")


def metadata(root, include_private, read=None):
    read = read or (lambda name: (root/name).read_bytes())
    public_bytes = read(BASE+"/inputs/manifest.json")
    if hashlib.sha256(public_bytes).hexdigest() != PUBLIC_SHA: raise ValueError("Frozen public cohort manifest differs")
    public = json.loads(public_bytes)
    if (public.get("schema") != "world-reward-tudl-rgb-v1" or public.get("revision") != TUDL_REV or public.get("license") != "CC-BY-SA-4.0"
            or len(public.get("images", [])) != 9): raise ValueError("RGB-only cohort metadata mismatch")
    expected = {}
    for name, item in zip(public_names(), public["images"]):
        if set(item) != {"scene_id", "frame_id", "file", "sha256", "width", "height"} or item["file"] != PurePosixPath(name).name or (item["width"], item["height"]) != (640, 480): raise ValueError("Public RGB route differs")
        expected[name] = item["sha256"]
    acquisition = json.loads(read("results/weights-acquisition.json")); reject_secrets(acquisition)
    moge = [a for a in acquisition["assets"] if a.get("repo_id") == "Ruicheng/moge-2-vitl-normal"]
    if len(moge) != 1 or moge[0].get("revision") != MODEL_REV or moge[0].get("cache_dir") != str(root/HF): raise ValueError("Pinned local MoGe acquisition receipt mismatch")
    if include_private:
        raw = read(BASE+"/eval_private/acquisition-report.json")
        if hashlib.sha256(raw).hexdigest() != ACQUISITION_SHA: raise ValueError("Frozen private acquisition receipt differs")
        receipt = json.loads(raw); reject_secrets(receipt)
        if (receipt.get("stage") != "external_tudl_rgb_only_validation_acquisition" or receipt.get("status") != "pass"
                or receipt.get("dataset_revision") != TUDL_REV or receipt.get("public_manifest_sha256") != PUBLIC_SHA
                or receipt.get("challenge_inputs_used") is not False or receipt.get("inference_performed") is not False): raise ValueError("Private acquisition provenance mismatch")
        records = receipt.get("retained_files", [])
        prefix = BASE+"/eval_private/"
        if len(records) != 41 or {prefix+r["file"] for r in records} != set(private_names()): raise ValueError("Require exact41 private source records")
        expected.update({prefix+r["file"]: r["sha256"] for r in records})
    return expected


def inventory(root, include_private):
    expected = metadata(root, include_private); entries = []
    for name in sorted(whitelist(include_private)):
        path = root/name
        if any(p.is_symlink() for p in path.parents if p != root and p.is_relative_to(root)): raise ValueError("Transfer parent may not be a symlink")
        mode = path.lstat().st_mode
        if name in (SNAPSHOT, REPO_BLOB):
            if not stat.S_ISLNK(mode): raise ValueError("Require audited two-link HF snapshot chain")
            link = os.readlink(path); target = link_target(name, link)
            if target != (REPO_BLOB if name == SNAPSHOT else BLOB): raise ValueError("HF source link chain changed")
            entry = {"path": name, "type": "symlink", "link": link, "bytes": 0, "mode": 0o777}
        else:
            if not stat.S_ISREG(mode): raise ValueError("Only exact regular assets may transfer")
            size, sha = path.stat().st_size, file_hash(path)
            if size > (MODEL_BYTES if name == BLOB else 16_000_000) or sha != expected.get(name, sha): raise ValueError("Frozen transfer asset size/SHA mismatch")
            if name == BLOB and (size, sha) != (MODEL_BYTES, MODEL_SHA): raise ValueError("Exact MoGe model SHA/size required")
            entry = {"path": name, "type": "file", "bytes": size, "sha256": sha, "mode": 0o600 if "/eval_private/" in name else 0o444}
        entries.append(entry)
    return {"schema": "world-reward-research-transfer-v1", "include_private": include_private, "entries": entries,
            "docker_image_included": False, "challenge_inputs_included": False, "predictions_included": False, "secrets_included": False}


def export(root, path, include_private):
    manifest = inventory(root, include_private); data = (json.dumps(manifest, sort_keys=True)+"\n").encode()
    if path.exists() or path.is_symlink(): raise FileExistsError("Transfer archive is immutable")
    with path.open("xb") as output, tarfile.open(fileobj=output, mode="w", format=tarfile.USTAR_FORMAT) as archive:
        member = tarfile.TarInfo(MANIFEST); member.size = len(data); member.mode = 0o600; archive.addfile(member, io.BytesIO(data))
        for entry in manifest["entries"]:
            member = tarfile.TarInfo(entry["path"]); member.mode = entry["mode"]; member.size = entry["bytes"]
            if entry["type"] == "symlink": member.type = tarfile.SYMTYPE; member.linkname = entry["link"]; archive.addfile(member)
            else:
                with (root/entry["path"]).open("rb") as stream: archive.addfile(member, stream)
    if inventory(root, include_private) != manifest: raise ValueError("Transfer source changed while archiving")
    return file_hash(path)


def verify(path, expected_sha, root=None):
    if (path.is_symlink() or not path.is_file() or path.resolve() != path.absolute() or path.stat().st_size > 2_200_000_000
            or not isinstance(expected_sha, str) or not re.fullmatch("[0-9a-f]{64}", expected_sha)
            or file_hash(path) != expected_sha): raise ValueError("Transfer archive SHA mismatch")
    with tarfile.open(path, "r:") as archive:
        members = []
        for member in archive:
            members.append(member)
            if len(members) > 57: raise ValueError("Transfer archive member bound exceeded")
        if not members or members[0].name != MANIFEST or not members[0].isfile() or members[0].size > 100_000: raise ValueError("Transfer manifest must be first bounded regular file")
        manifest = json.load(archive.extractfile(members[0])); private = manifest.get("include_private")
        if (set(manifest) != {"schema", "include_private", "entries", "docker_image_included", "challenge_inputs_included", "predictions_included", "secrets_included"}
                or manifest.get("schema") != "world-reward-research-transfer-v1" or type(private) is not bool
                or any(manifest[k] is not False for k in ("docker_image_included", "challenge_inputs_included", "predictions_included", "secrets_included"))): raise ValueError("Transfer manifest schema mismatch")
        entries = manifest.get("entries", [])
        if len(entries) != len(whitelist(private)) or {e["path"] for e in entries} != whitelist(private) or [m.name for m in members[1:]] != [e["path"] for e in entries]: raise ValueError("Transfer inventory/order/duplicate mismatch")
        for member, entry in zip(members[1:], entries):
            safe_path(member.name)
            link = member.name in (SNAPSHOT, REPO_BLOB); keys = {"path", "type", "bytes", "mode", "link" if link else "sha256"}
            mode = 0o777 if link else (0o600 if "/eval_private/" in member.name else 0o444)
            if (set(entry) != keys or type(entry["bytes"]) is not int or entry["bytes"] < 0
                    or entry["type"] != ("symlink" if link else "file") or member.size != entry["bytes"]
                    or member.mode != entry["mode"] or member.mode != mode): raise ValueError("Transfer member metadata mismatch")
            if entry["type"] == "symlink":
                if not member.issym() or member.size or member.linkname != entry["link"] or link_target(member.name, member.linkname) != (REPO_BLOB if member.name == SNAPSHOT else BLOB): raise ValueError("Unsafe transfer HF link")
            elif entry["type"] == "file":
                if (not member.isfile() or member.size > (MODEL_BYTES if member.name == BLOB else 16_000_000)
                        or not isinstance(entry["sha256"], str) or not re.fullmatch("[0-9a-f]{64}", entry["sha256"])
                        or digest(archive.extractfile(member)) != entry["sha256"]): raise ValueError("Transfer regular member SHA/type/size mismatch")
                if member.name == BLOB and (member.size, entry["sha256"]) != (MODEL_BYTES, MODEL_SHA): raise ValueError("Model transfer identity differs")
            else: raise ValueError("Unsupported transfer member")
        if root is not None:
            expected = metadata(root, private, read=lambda name: archive.extractfile(name).read(16_000_001))
            if any(e["type"] == "file" and e["sha256"] != expected.get(e["path"], e["sha256"]) for e in entries): raise ValueError("Transfer provenance file SHA differs from original receipt")
    return manifest


def extract(root, path, expected_sha):
    manifest = verify(path, expected_sha, root)
    if manifest["include_private"] and (root/(BASE+"/eval_private")).exists(): raise FileExistsError("Private transfer root must be absent")
    for entry in manifest["entries"]:
        target = root/entry["path"]
        if target.exists() or target.is_symlink() or any(p.is_symlink() or (p.exists() and not p.is_dir()) for p in target.parents if p.is_relative_to(root)): raise FileExistsError("Transfer target paths must be fresh and nonsymlink")
    with tarfile.open(path, "r:") as archive:
        for entry in manifest["entries"]:
            target = root/entry["path"]
            missing = []; parent = target.parent
            while not parent.exists(): missing.append(parent); parent = parent.parent
            for folder in reversed(missing): folder.mkdir(mode=0o700 if "eval_private" in folder.parts else 0o755)
            if entry["type"] == "file":
                with archive.extractfile(entry["path"]) as source, target.open("xb") as output:
                    for block in iter(lambda: source.read(1024*1024), b""): output.write(block)
                target.chmod(entry["mode"])
            else: target.symlink_to(entry["link"])
    if inventory(root, manifest["include_private"]) != manifest: raise ValueError("Extracted exact asset/provenance verification failed")
    if file_hash(path) != expected_sha: raise ValueError("Transfer archive changed during extraction")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("mode", choices=("export", "verify", "extract")); parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--sha256"); parser.add_argument("--include-private", action="store_true"); args = parser.parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Heavy research transfer stays remote Linux only")
    root = Path(os.environ["WR_ROOT"])
    if not root.is_dir() or root.resolve() != root.absolute() or args.archive.resolve() != args.archive.absolute(): raise ValueError("Require canonical remote root/archive")
    if args.mode != "export" and (not args.sha256 or args.include_private): raise ValueError("Verification/extraction requires SHA; scope is read from archive")
    if args.mode == "extract" and os.geteuid() != 1000: raise ValueError("Extract as scenesmith UID1000, never recursively chown parent trees")
    def expired(*_): raise TimeoutError("Research transfer exceeded600s")
    signal.signal(signal.SIGALRM, expired); signal.alarm(600)
    try:
        if args.mode == "export": print(json.dumps({"archive_sha256": export(root, args.archive, args.include_private)}))
        elif args.mode == "verify": verify(args.archive, args.sha256, root)
        else: extract(root, args.archive, args.sha256)
    finally: signal.alarm(0)


if __name__ == "__main__": main()
