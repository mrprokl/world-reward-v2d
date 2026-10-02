"""Bind OCI-index exports to Docker's classic config-ID without weakening hashes.

Docker's containerd image store reports an OCI index digest as image ID. Its
classic overlay2 store reports the selected platform's config digest instead.
The same sealed export must prove that graph and every ordered rootfs diff-ID;
an arbitrary changed Docker ID is never accepted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile

EXPORT_SHA = "203af62c8c03931919acd2fab28b1fa53a802be73d0f04d99febf60c26321b4e"
INDEX_ID = "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7"
CONFIG_ID = "sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3"
TAG = "world-reward/cari4d-source:0.1"


def image_identity(archive, inspected):
    path = Path(archive)
    if path.is_symlink() or not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Canonical sealed image TAR required")
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""): h.update(block)
    if h.hexdigest() != EXPORT_SHA:
        raise ValueError("Exact immutable source export SHA required")
    with tarfile.open(path, "r:") as tar:
        names = tar.getnames()
        if len(set(names)) != len(names): raise ValueError("Duplicate image members")
        def small(name, descriptor=None):
            member = tar.getmember(name)
            if not member.isfile() or not 0 < member.size <= 1_000_000:
                raise ValueError("Small regular image graph member required")
            data = tar.extractfile(member).read()
            if descriptor is not None and (member.size != descriptor["size"] or
                    "sha256:" + hashlib.sha256(data).hexdigest() != descriptor["digest"]):
                raise ValueError("OCI graph digest/size mismatch")
            return json.loads(data)
        def blob(d):
            if not re.fullmatch("sha256:[0-9a-f]{64}", d.get("digest", "")):
                raise ValueError("SHA256 OCI graph required")
            return small("blobs/sha256/" + d["digest"][7:], d)
        if small("oci-layout") != {"imageLayoutVersion": "1.0.0"}:
            raise ValueError("OCI1 image layout required")
        descriptors = small("index.json")["manifests"]
        if len(descriptors) != 1 or descriptors[0]["digest"] != INDEX_ID:
            raise ValueError("Original source OCI index differs")
        source_index = blob(descriptors[0])
        manifests = source_index["manifests"]
        selected = [d for d in manifests if d.get("platform") == {"architecture": "amd64", "os": "linux"}]
        if len(selected) != 1: raise ValueError("Exactly one Linux AMD64 platform required")
        platform = blob(selected[0]); config_d = platform["config"]
        if config_d["digest"] != CONFIG_ID: raise ValueError("Pinned source platform config differs")
        config = blob(config_d)
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            raise ValueError("Runtime platform differs")
        legacy = small("manifest.json")
        if (len(legacy) != 1 or legacy[0]["Config"] != "blobs/sha256/" + CONFIG_ID[7:]
                or legacy[0]["RepoTags"] != [TAG]
                or legacy[0]["Layers"] != ["blobs/sha256/" + d["digest"][7:] for d in platform["layers"]]):
            raise ValueError("Classic export does not select the sealed OCI platform")
        diff_ids = config.get("rootfs", {}).get("diff_ids")
        if not isinstance(diff_ids, list) or len(diff_ids) != len(platform["layers"]) or not diff_ids:
            raise ValueError("Full ordered source rootfs required")
        if any(not re.fullmatch("sha256:[0-9a-f]{64}", d) for d in diff_ids):
            raise ValueError("Rootfs SHA256 required")
    if (not isinstance(inspected, list) or len(inspected) != 1 or inspected[0].get("Id") != CONFIG_ID
            or inspected[0].get("Architecture") != "amd64" or inspected[0].get("Os") != "linux"
            or inspected[0].get("RootFS") != {"Type": "layers", "Layers": diff_ids}
            or TAG not in inspected[0].get("RepoTags", [])):
        raise ValueError("Imported classic Docker config/platform/ordered rootfs differs")
    return {"source_OCI_index_id": INDEX_ID, "platform_manifest_id": selected[0]["digest"],
            "image_id": CONFIG_ID, "rootfs_layers": len(diff_ids),
            "rootfs_diff_ids": diff_ids, "sealed_image_tar_sha256": EXPORT_SHA,
            "image_content_changed": False, "image_rebuilt": False}


def main():
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(image_identity(args.archive, json.load(sys.stdin))), flush=True)


if __name__ == "__main__": main()
