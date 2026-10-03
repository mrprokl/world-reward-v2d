"""Azure-only minimal frontend prerequisites; report, never export or run models.

File identities/link graphs are retained ON AZURE; stdout is a bounded summary.
Constants mirror audited existing binders (tested independently), without
importing GPU/renderer drivers. Inventory PASS is not replica/license/CUDA PASS.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import posixpath
import re
import signal
import stat
import subprocess
import time

ROOT = Path("/srv/scenesmith/world-reward")
BUDGET = 600
UPSTREAM = "7c0d3b94ce97b28deb571b4e7fdfeb5b2158df80"
BODY_REV = "11aaa346c7204874a1cbafe3d39a979080b2c55a"
OBJECT_REV = "2e73555018d2741ccd486e56c24fac41155a1dc6"
DINO_REVS = {"dinov3": "6876159a11b4df116f30f667f8c9888617df0751", "dinov2": "7764ea0f912e53c92e82eb78a2a1631e92725fc8"}
MOGE_REV = "ad326bfb61facd6c52b5a825bc1e34d7c97d9672"
MOGE_EXPECTED = ("da96b09a0485a3c45a5aa455e67743c8b4efc4dd8437c1f2aa93c2b4303d957f", 1256823446)
BODY = "weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3"
OBJECT = "weights/sam3d/hf-download"
HF = "weights/sam3d/hf_home/hub"
MOGE_REPO = HF + "/models--Ruicheng--moge-vitl"
MOGE_SNAPSHOT = MOGE_REPO + "/snapshots/" + MOGE_REV
BODY_PACKAGE = "reconstruction/modules/v2d_sam3d_body/lib/sam_3d_body"
CARI = "reconstruction/modules/v2d_cari4d/lib/cari4d"
LIB_MHR = tuple(n + ".py" for n in ("__init__", "body_pose", "camera_conventions", "collision_proxy", "contact", "delta", "geometry_provider", "human_texture", "io", "metrics", "mhr_layer", "object_pose_frame", "object_symmetry", "refit", "rotations", "schema", "sdf"))
PREP = tuple(n + ".py" for n in ("__init__", "prepare_mhr_wild_export", "mhr_depth_h5", "mhr_depth_backend", "mhr_export_utils", "mhr_effective_masks", "mhr_rgb_h5", "mhr_wild_depth", "mhr_ffv1_sidecar", "mhr_sensor_depth_h5"))
EXTRA_SOURCE = ("behave_data/__init__.py", "behave_data/video_reader.py", "tools/__init__.py", "tools/pipeline_timing.py")
BODY_DATA = {"data/__init__.py", "data/transforms/__init__.py", "data/transforms/bbox_utils.py", "data/transforms/common.py", "data/utils/io.py", "data/utils/prepare_batch.py"}
GROUNDING = {
    "grounding_dino/model.safetensors": ("5548f844c928c4b6f411fa8cbcc2bfa8dbbba437cb1d513975519f93c2a9ed21", 933400872),
    "grounding_dino/config.json": ("eda416dae6f49419ff831b1c190ec430a060b19aae688dbaf2425a075b650608", 1737),
    "grounding_dino/preprocessor_config.json": ("8454179ba95e2ad22947835aad7b45862a601fc0055ab88bf1ee70892d3aea60", 457),
    "grounding_dino/special_tokens_map.json": ("b6d346be366a7d1d48332dbc9fdf3bf8960b5d879522b7799ddba59e76237ee3", 125),
    "grounding_dino/tokenizer.json": ("d241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66", 711396),
    "grounding_dino/tokenizer_config.json": ("d40ab645b68211910b9170d22433d43186a6ec8ee6fd10ba170524b25bf4fb56", 1237),
    "grounding_dino/vocab.txt": ("07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3", 231508),
    "sam2/sam2.1_hiera_large.pt": ("2647878d5dfa5098f2f8649825738a9345572bae2d4350a2468587ece47dd318", 898083611),
    "sam2/sam2.1_hiera_l.yaml": ("545e4325aa5c19a1615d43c946b07276ed4c57214eacf1437e38fa3d9374f636", 3798),
}
CHECKPOINTS = {
    "ss_generator": ("225f40479e4cff4f39d6fa14c55be3abad1475bf55b61af3bec1e19ed2f6c146", 6690136964),
    "slat_generator": ("91529bde8e7daa12d09618a66c319e3a5a6398db6b23b958cedcb1c3f28faabb", 4906537684),
    "ss_decoder": ("6dac1cd7b7fda5a38e0614fadae441f1794f80e39ea2981f1ac8aff0a7e99340", 147609242),
    "slat_decoder_gs": ("f8077c36a06eaf890dd93cda1937411f793dea1eb80b3dd9329f2038ba84a111", 171476155),
    "slat_decoder_gs_4": ("731a0eceaa47945b52aa27f650d695b2aea9cc70945751e5609e5cb5b49f0186", 170269801),
    "slat_decoder_mesh": ("85907b37b67d8ce5b099a96629bdcfbd873eb407dee6b3aa9a75deb15038db33", 363726862),
}
YAMLS = {
    "pipeline": ("53c3d226b21df85c0bb3d16e6e4fa63abde0d6167525765eb929d02bfa9d358c", 3548),
    "ss_generator": ("3c265448bca7c057f94e3ef56adea3a895a10bcd9f15f992a41dd03fa35412cd", 5076),
    "slat_generator": ("53029fadff6fe34a0344381a16d64d65d0567d603484952712e8969319559c4e", 1986),
    "ss_decoder": ("baacff269b664f84f7aa1896ebd66128065f6568cdb88d419d5c3d2ccb4193ae", 244),
    "slat_decoder_gs": ("53f054e02a0c185f0a6885d30bb6ca0ec92efe0a98148688e7f05f7c26afe670", 576),
    "slat_decoder_gs_4": ("3d1dfd4c56cdac56f30e0cf5eb310d4df973fe25c60ac0a462c86ca4e3bf8b48", 575),
    "slat_decoder_mesh": ("8f46952764aa985c50109a56f0b4f07625cdb06457aaded74957d26f3520c69e", 300),
}
FIXED = {BODY + "/model.ckpt": ("b5a2f9d305dd02626b967aa2e86021fba07065df66ce7a7e00ffb9664f150abf", 2109129346),
    BODY + "/assets/mhr_model.pt": ("352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc", 696110248),
    **{"weights/" + n: v for n, v in GROUNDING.items()},
    **{OBJECT + "/checkpoints/" + n + ".ckpt": v for n, v in CHECKPOINTS.items()},
    **{OBJECT + "/checkpoints/" + n + ".yaml": v for n, v in YAMLS.items()},
    "vendor/v2d_submission_kit/v2dlb/mesh_budget.py": ("42ab8ab35f37b806fb1465eadd96abe43eaac04575da47a4855d08eefe6167b0", 2031)}
CARDS = (BODY + "/model_config.yaml", BODY + "/LICENSE", BODY + "/README.md", OBJECT + "/LICENSE",
    "weights/grounding_dino/README.md", "weights/sam2/README.md", "vendor/v2d_submission_kit/v2dlb/__init__.py")
RECEIPTS = ("results/weights-acquisition.json", "results/auxiliary-assets.json", "results/image-cari4d-source.json", "results/image-grounding.json", "results/image-sam3d-runtime.json")
IMAGES = {"body": ("world-reward/cari4d-source:0.1", "sha256:b47e4450b24219c2a746f4795e27bde8c436f5cc310b7f8c527316f55c9380a7", RECEIPTS[2]),
    "grounding": ("world-reward/grounding:0.1", "sha256:53b33bc4b60e0e3e8f83b401775b4701b18eef54408fd585fbe3a5d376c042e1", RECEIPTS[3]),
    "objects": ("world-reward/sam3d-runtime:0.1", "sha256:eb389b26358c49778a14303b5875c66d887824011388ce9f8666ed7cc1841ce5", RECEIPTS[4])}
# Evidence only: not imported or shipped as GPU-driver dependencies.
CONSTANT_EVIDENCE = {
    "body_smoke": (32390, "e4d659de33bacff9d5c85c1cb2fedce34f4aa34bdbae93b3b9b7e08950613ce3"),
    "object_smoke": (10994, "1e249274153859150130be415203ae7cf4cb77862cba92d056493dbd51337407"),
    "hand_synthetic_infer": (20006, "10db2e23b4e76b0272ab038778de4ce87b25a66d82175ec477aab6be13fd156f"),
    "hand_synthetic_masks": (13491, "4694c3f61556dc60bb6b76b2b5ca328c7dd00308662bac651757c57ea92d3083"),
    "multiview_full_gate": (18091, "1747218fa8151dd91d0381104b5cfb932ecf6379b5ee5b2265bc4ad1219add8d"),
    "multiview_ss_gate": (15496, "f1540033c4a09fe97c1a8fee54f48c83764c9a6693cf1da0de56dad19cd79545"),
    "acquire_auxiliary": (4256, "27b0887ffeec489117acdd44b3fdb711b47a436bebc5b5925c11d07d1e1bf8c3"),
    "acquire_weights": (4652, "f1a8cdc68963e6a65916df897a1acef21236ad840f5d537deff8f688aab63d20"),
    "official_track1_pack_gate": (38158, "1aa758e7a7de116172a479e3dd24b2aff845dc3f2330bc0adbbcbd6f00cb37b4")}


def safe_name(name):
    if (type(name) is not str or not name or PurePosixPath(name).is_absolute() or "\\" in name or "\x00" in name
            or str(PurePosixPath(name)) != name or any(p in ("", ".", "..") for p in name.split("/"))):
        raise ValueError("Noncanonical whitelist name")
    return name


def canonical(path):
    path = Path(path)
    if not path.is_absolute() or any(p.is_symlink() for p in (path, *path.parents)) or path.resolve() != path:
        raise ValueError("Canonical nonsymlink prerequisite required")
    return path


def state(path):
    s = Path(path).lstat()
    return (s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns, s.st_ctime_ns, s.st_nlink)


def identity(path, maximum=7_000_000_000, allow_empty=False):
    path = canonical(path); before = state(path)
    if not stat.S_ISREG(before[2]) or not (0 if allow_empty else 1) <= before[3] <= maximum:
        raise ValueError("Bounded regular prerequisite required")
    sha256 = hashlib.sha256(); sha1 = hashlib.sha1(b"blob " + str(before[3]).encode() + b"\0")
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): sha256.update(block); sha1.update(block)
    if state(path) != before: raise ValueError("Prerequisite changed while hashing")
    return {"bytes": before[3], "sha256": sha256.hexdigest(), "git_blob_sha1": sha1.hexdigest()}


def strict_json(raw):
    def pairs(rows):
        result = {}
        for key, value in rows:
            if key in result: raise ValueError("Duplicate receipt key")
            result[key] = value
        return result
    def invalid(_): raise ValueError("Nonfinite receipt JSON")
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=invalid)


def no_secrets(value):
    if isinstance(value, dict):
        if any(re.search(r"(?i)(?:^|_)(token|password|secret|credential|authorization|api_key)(?:$|_)", k) for k in value):
            raise ValueError("Secret-like prerequisite metadata forbidden")
        for item in value.values(): no_secrets(item)
    elif isinstance(value, list):
        for item in value: no_secrets(item)
    elif isinstance(value, str) and re.search(r"hf_[A-Za-z0-9]{20,}|Bearer\s+\S+|AccountKey=|[?&](?:sig|token)=", value):
        raise ValueError("Credential-like prerequisite metadata forbidden")


def error_text(error):
    """Precondition/path only; never return subprocess stderr or credential text."""
    message = str(error)
    if re.search(r"hf_[A-Za-z0-9]{20,}|Bearer\s+\S+|AccountKey=|[?&](?:sig|token)=|(?i:password|api_key|authorization)\s*[=:]", message):
        return "Prerequisite metadata rejected; sensitive diagnostic omitted"
    return message[:250]


def command(arguments):
    environment = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "HOME": "/nonexistent", "LANG": "C.UTF-8",
        "GIT_OPTIONAL_LOCKS": "0", "GIT_NO_LAZY_FETCH": "1", "GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "GIT_CONFIG_GLOBAL": "/dev/null",
        "DOCKER_HOST": "unix://" + str(ROOT / "docker.sock")}
    result = subprocess.run(arguments, env=environment, capture_output=True, timeout=20, check=False)
    if result.returncode or len(result.stdout) > 2_000_000: raise ValueError("Bounded readonly prerequisite query failed")
    return result.stdout


def git(repository, *args):
    canonical(repository)
    return command(["git", "-c", "safe.directory=" + str(repository), "-C", str(repository), *args])


def selected_source(name, *, dino=None):
    safe_name(name)
    if dino:
        return (name.startswith(dino + "/") and name.endswith(".py")
            or name in ("hubconf.py", "LICENSE.md" if dino == "dinov3" else "LICENSE", "MODEL_CARD.md"))
    return (name == "LICENSE" or name.startswith(BODY_PACKAGE + "/") and (name.endswith(".py") or name == BODY_PACKAGE + "/LICENSE")
        or name in {CARI + "/lib_mhr/" + n for n in LIB_MHR} | {CARI + "/prep/" + n for n in PREP} | {CARI + "/" + n for n in EXTRA_SOURCE})


def source_inventory(root, repository_name, revision, dino=None):
    repository = canonical(root / safe_name(repository_name))
    if git(repository, "rev-parse", "HEAD").decode().strip() != revision: raise ValueError("Original source revision differs")
    prefixes = [dino, "hubconf.py", "LICENSE.md" if dino == "dinov3" else "LICENSE", "MODEL_CARD.md"] if dino else [BODY_PACKAGE, CARI + "/lib_mhr", CARI + "/prep", *[CARI + "/" + n for n in EXTRA_SOURCE], "LICENSE"]
    records = {}
    for raw in git(repository, "ls-tree", "-r", "-z", revision, "--", *prefixes).split(b"\0"):
        if not raw: continue
        metadata, path_raw = raw.split(b"\t", 1); mode, kind, blob = metadata.decode().split(); name = path_raw.decode()
        if not selected_source(name, dino=dino): continue
        if kind != "blob" or mode not in ("100644", "100755") or not re.fullmatch(r"[0-9a-f]{40}", blob): raise ValueError("Original regular source blob required")
        value = identity(repository / name, 1_000_000, allow_empty=True)
        if value["git_blob_sha1"] != blob: raise ValueError("Selected local source differs from committed Git blob")
        records[repository_name + "/" + name] = {"type": "file", **value, "role": "public_source"}
    names = {n[len(repository_name) + 1:] for n in records}
    required = {"hubconf.py", "LICENSE.md" if dino == "dinov3" else "LICENSE", "MODEL_CARD.md"} if dino else {"LICENSE", BODY_PACKAGE + "/LICENSE", BODY_PACKAGE + "/__init__.py", *[CARI + "/lib_mhr/" + n for n in LIB_MHR], *[CARI + "/prep/" + n for n in PREP], *[CARI + "/" + n for n in EXTRA_SOURCE]}
    if not required <= names or not any(n.startswith((dino + "/") if dino else BODY_PACKAGE + "/") and n.endswith(".py") for n in names):
        raise ValueError("Complete selected public source/license closure required")
    if not dino and {n[len(BODY_PACKAGE) + 1:] for n in names if n.startswith(BODY_PACKAGE + "/data/")} != BODY_DATA:
        raise ValueError("All six Body data-Python sources required; no dataset assets")
    if git(repository, "status", "--porcelain", "--untracked-files=all", "--", *prefixes): raise ValueError("Selected source worktree is modified")
    return records, {"path": repository_name, "revision": revision, "selected_worktree_clean_verified": True,
        "complete_checkout_clean_verified": False, "source_files": len(records), "source_sha256": digest_json(records)}


def moge_chain(root, name):
    """Discover only this exact HF model/README graph, never scan caches or guess XET."""
    if name not in ("model.pt", "README.md"): raise ValueError("Only internal MoGe1 model/card allowed")
    canonical(root / MOGE_REPO); snapshots = canonical(root / (MOGE_REPO + "/snapshots"))
    if {p.name for p in snapshots.iterdir()} != {MOGE_REV}: raise ValueError("MoGe1 first-snapshot constructor must see only the pinned revision")
    relative = MOGE_SNAPSHOT + "/" + name; seen = set(); records = {}
    for _ in range(5):
        if relative in seen: raise ValueError("MoGe1 link cycle")
        seen.add(relative); path = root / safe_name(relative); canonical(path.parent); before = state(path)
        if stat.S_ISLNK(before[2]):
            target = os.readlink(path)
            if (not target or PurePosixPath(target).is_absolute() or "\\" in target or "\x00" in target): raise ValueError("Relative MoGe1 blob link required")
            destination = safe_name(posixpath.normpath(posixpath.join(posixpath.dirname(relative), target)))
            allowed = (re.fullmatch(re.escape(MOGE_REPO) + r"/blobs/[0-9a-f]{40,64}", destination)
                or re.fullmatch(re.escape(HF) + r"/blobs/[0-9a-f]{2}/[0-9a-f]{40,64}", destination))
            if not allowed or state(path) != before: raise ValueError("MoGe1 link escapes exact model blob roots or changed")
            records[relative] = {"type": "symlink", "link": target, "target": destination, "role": "internal_moge1"}; relative = destination
        else:
            value = identity(path, MOGE_EXPECTED[1] if name == "model.pt" else 4096)
            if name == "model.pt" and (value["sha256"], value["bytes"]) != MOGE_EXPECTED: raise ValueError("Actual MoGe1 differs from independent primary model SHA/bytes")
            records[relative] = {"type": "file", **value, "role": "internal_moge1"}
            return records, {"snapshot": MOGE_SNAPSHOT + "/" + name, "resolved_file": relative, "links": len(records) - 1, **value,
                "independent_primary_model_identity_verified": name == "model.pt", "XET_path_discovered_not_fabricated": True}
    raise ValueError("MoGe1 blob chain exceeds four links")


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def image_record(tag, expected_id, receipt):
    # Do not inspect Config.Env: images/receipts may contain inherited credentials.
    fmt = '{"Id":{{json .Id}},"Architecture":{{json .Architecture}},"Os":{{json .Os}},"Size":{{json .Size}},"RootFS":{{json .RootFS}}}'
    actual = strict_json(command(["docker", "image", "inspect", tag, "--format", fmt])); no_secrets(actual)
    keys = {"Id", "Architecture", "Os", "Size", "RootFS"}
    if type(actual) is not dict or set(actual) != keys or actual["Id"] != expected_id or actual["Architecture"] != "amd64" or actual["Os"] != "linux" or type(actual["Size"]) is not int or actual["Size"] <= 0:
        raise ValueError("Actual frontend image ID/platform differs")
    rootfs = actual["RootFS"]
    if (type(rootfs) is not dict or set(rootfs) != {"Type", "Layers"} or rootfs["Type"] != "layers" or type(rootfs["Layers"]) is not list or not rootfs["Layers"]
            or any(type(n) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", n) is None for n in rootfs["Layers"])):
        raise ValueError("Full ordered image rootfs diff-IDs required")
    if any(receipt.get(k) != actual[k] for k in keys): raise ValueError("Actual image contradicts original build receipt")
    return {"tag": tag, **actual, "platform_config_id": None, "sealed_export_graph_verified": False,
        "rootfs_sha256": digest_json(rootfs["Layers"]), "inspected_ID_is_not_assumed_classic_config_ID": True}


def inventory(root):
    root = canonical(root); entries = {}; states = {}; receipts = {}; sources = []; evidence = {}
    def add(name, expected=None, role="asset", allow_empty=False, maximum=7_000_000_000):
        safe_name(name); path = root / name; value = identity(path, maximum, allow_empty); states[name] = state(path)
        if expected and (value["sha256"], value["bytes"]) != expected: raise ValueError("Independent asset SHA/bytes differ: " + name)
        entries[name] = {"type": "file", **value, "role": role}
    for name in RECEIPTS:
        add(name, role="source_receipt", maximum=2_000_000); raw = (root / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != entries[name]["sha256"]: raise ValueError("Receipt changed before JSON")
        receipts[name] = strict_json(raw)
        if name in RECEIPTS[:2]: no_secrets(receipts[name])
        else:
            # Full original build JSON may contain Config.Env. Its opaque byte
            # identity is evidence ONLY, never a transfer-eligible entry; retain
            # only the bounded image projection below, not inherited settings.
            evidence[name] = {**entries.pop(name), "transfer_eligible": False, "raw_build_receipt_must_not_be_exported": True}
    acquisition = receipts[RECEIPTS[0]]
    bindings = (("facebook/sam-3d-body-dinov3", BODY_REV, "path", BODY), ("facebook/sam-3d-objects", OBJECT_REV, "path", OBJECT),
        ("IDEA-Research/grounding-dino-base", "12bdfa3120f3e7ec7b434d90674b3396eccf88eb", "path", "weights/grounding_dino"),
        ("facebook/sam2.1-hiera-large", "665f8e2ad61cf5f53d65644ff27c8ee525124610", "path", "weights/sam2"),
        ("Ruicheng/moge-vitl", MOGE_REV, "cache_dir", HF))
    if type(acquisition) is not dict or type(acquisition.get("assets")) is not list: raise ValueError("Original principal acquisition receipt required")
    for repo, rev, field, directory in bindings:
        matches = [r for r in acquisition["assets"] if isinstance(r, dict) and r.get("repo_id") == repo]
        if len(matches) != 1 or matches[0].get("revision") != rev or matches[0].get(field) != str(root / directory): raise ValueError("Original frontend acquisition path/revision binding differs")
    aux = receipts[RECEIPTS[1]]
    if type(aux) is not dict or aux.get("source_revisions") != DINO_REVS or type(aux.get("checkpoints")) is not list: raise ValueError("Original pinned auxiliary source receipt required")
    reg4 = [r for r in aux["checkpoints"] if isinstance(r, dict) and "_reg4_" in r.get("filename", "")]
    if len(reg4) != 2 or {r.get("filename") for r in reg4} != {"dinov2_vitl14_reg4_pretrain.pth", "dinov2_vitb14_reg4_pretrain.pth"}: raise ValueError("Exactly two original Objects reg4 checkpoint receipts required")
    for row in reg4:
        if (type(row.get("sha256")) is not str or re.fullmatch(r"[0-9a-f]{64}", row["sha256"]) is None or type(row.get("bytes")) is not int or row["bytes"] <= 0
                or row.get("hash_source") != "first_observed_https_download"
                or row.get("url") != "https://dl.fbaipublicfiles.com/dinov2/dinov2_" + ("vitl14" if "vitl14" in row["filename"] else "vitb14") + "/" + row["filename"]): raise ValueError("Original reg4 source/identity receipt required; no fabricated independent release pin")
        add("weights/sam3d/torch_home/hub/checkpoints/" + row["filename"], (row["sha256"], row["bytes"]), "reg4_first_observed_receipt_bound")
    for name, expected in FIXED.items(): add(name, expected)
    for name in CARDS: add(name, role="license_card_or_config", allow_empty=name.endswith("__init__.py"), maximum=1_000_000)
    for repository, revision, dino in (("vendor/video_to_data", UPSTREAM, None),
            ("weights/cari4d/sam3d_body/torch_home/hub/facebookresearch_dinov3_main", DINO_REVS["dinov3"], "dinov3"),
            ("weights/sam3d/torch_home/hub/facebookresearch_dinov2_main", DINO_REVS["dinov2"], "dinov2")):
        records, summary = source_inventory(root, repository, revision, dino); entries.update(records); sources.append(summary)
        states.update({n: state(root / n) for n in records})
    moge = {}
    for name in ("model.pt", "README.md"):
        records, binding = moge_chain(root, name); entries.update(records); moge[name] = binding
        states.update({n: state(root / n) for n in records})
    images = {name: image_record(tag, image_id, receipts[receipt]) for name, (tag, image_id, receipt) in IMAGES.items()}
    if {name: image_record(tag, image_id, receipts[receipt]) for name, (tag, image_id, receipt) in IMAGES.items()} != images:
        raise ValueError("Actual image metadata changed during inventory")
    if any(state(root / name) != before for name, before in states.items()): raise ValueError("Original prerequisite changed during complete inventory")
    for name, row in entries.items():
        if row["type"] == "symlink" and os.readlink(root / name) != row["link"]: raise ValueError("MoGe1 link changed after inventory")
    for row in sources:
        if git(root / row["path"], "rev-parse", "HEAD").decode().strip() != row["revision"]: raise ValueError("Public source HEAD changed")
    if {p.name for p in (root / (MOGE_REPO + "/snapshots")).iterdir()} != {MOGE_REV}: raise ValueError("MoGe1 snapshot selection changed")
    return dict(entries=entries, evidence_only_files=evidence, sources=sources, internal_moge1=moge, images=images,
        actual_prerequisites_unchanged=True, entries_sha256=digest_json(entries),
        constant_evidence={name: {"bytes": size, "sha256": sha, "repository_path": "infra/" + name + ".py"} for name, (size, sha) in CONSTANT_EVIDENCE.items()},
        source_URLs={"upstream": "https://github.com/nvidia-isaac/video_to_data/tree/" + UPSTREAM,
            "body": "https://huggingface.co/facebook/sam-3d-body-dinov3/tree/" + BODY_REV, "objects": "https://huggingface.co/facebook/sam-3d-objects/tree/" + OBJECT_REV,
            "moge1": "https://huggingface.co/Ruicheng/moge-vitl/tree/" + MOGE_REV},
        replica_ready=False, blockers=["image_export_OCI_index_platform_config_graph_not_sealed", "complete_selected_dependency_license_closure_unverified", "VM02_independent_CUDA_camera_gate_not_run", "new_scoped_frontend_launcher_not_validated"],
        upstream_installed_body_source_parity_verified=False, installed_objects_dependency_source_verified=False,
        challenge_data_read=False, private_validation_read=False, predictions_read=False, GPU_used=False, models_loaded=False,
        Docker_mutated=False, image_exported=False, transferred=False, credential_files_opened=False,
        original_build_receipt_JSON_parsed=True, inherited_environment_not_inspected_or_recorded=True,
        raw_build_receipts_transfer_eligible=False, training_overlap_verified=False, license_eligibility_verified=False)


def small_summary(report, path=None):
    result = {"stage": report["stage"], "status": report["status"], "replica_ready": False, "elapsed_seconds": report.get("elapsed_seconds", 0.)}
    if path: result["report"] = str(path)
    if "entries" in report:
        result.update(files=sum(r["type"] == "file" for r in report["entries"].values()), links=sum(r["type"] == "symlink" for r in report["entries"].values()),
            bytes=sum(r.get("bytes", 0) for r in report["entries"].values()), entries_sha256=report["entries_sha256"], blockers=report["blockers"],
            moge1={k: report["internal_moge1"]["model.pt"][k] for k in ("bytes", "sha256", "links", "resolved_file")},
            images={k: {"Id": r["Id"], "Size": r["Size"], "rootfs_layers": len(r["RootFS"]["Layers"]), "rootfs_sha256": r["rootfs_sha256"], "platform_config_id": None} for k, r in report["images"].items()})
    else: result.update(error_type=report.get("error_type", "unknown"), error=report.get("error", "unspecified prerequisite"))
    raw = json.dumps(result, sort_keys=True, allow_nan=False)
    if len(raw.encode()) > 4000: raise ValueError("Control-plane summary exceeds4KB")
    return raw


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    root, code = Path(os.environ["WR_ROOT"]), Path(os.environ["WR_CODE"]); revision = os.environ["WR_CODE_REVISION"]
    if (platform.system() != "Linux" or root != ROOT or re.fullmatch(r"[0-9a-f]{40}", revision) is None
            or code != root / "jobs" / revision / "run_frontend_replica_inventory/code" or canonical(code) != code
            or Path(__file__) != code / "infra/frontend_replica_inventory.py"):
        raise ValueError("Actual immutable Azure host inventory source required")
    owned = root / "results" / ("frontend-replica-inventory-" + revision)
    canonical(owned)
    if not owned.is_dir() or any(owned.iterdir()) or os.environ.get("WR_FRONTEND_INVENTORY_RESERVED") != "1": raise ValueError("Exclusive empty new report directory required")
    helpers = {n: identity(code / n, 1_000_000) for n in ("infra/frontend_replica_inventory.py", "infra/run_frontend_replica_inventory.sh")}
    if any((code / n).stat().st_mode & 0o222 for n in helpers): raise ValueError("Frozen inventory source must be readonly")
    report = dict(stage="frontend_replica_prerequisite_inventory", status="fail", producer_revision=revision, source_helpers=helpers, budget_seconds=BUDGET)
    started = time.perf_counter(); error = None
    def expired(*_): raise TimeoutError("Frontend prerequisite CPU inventory exceeded600s")
    alarm, term = signal.signal(signal.SIGALRM, expired), signal.signal(signal.SIGTERM, expired); signal.alarm(BUDGET)
    try: report.update(inventory(root)); report["status"] = "pass"
    except Exception as caught: error = caught; report.update(error_type=type(caught).__name__, error=error_text(caught))
    finally:
        try:
            if {n: identity(code / n, 1_000_000) for n in helpers} != helpers: raise ValueError("Frozen inventory source changed")
        except Exception as caught: error = caught; report.update(status="fail", error_type=type(caught).__name__, error=error_text(caught))
        signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term)
        report["elapsed_seconds"] = time.perf_counter() - started; path = owned / "report.json"
        with path.open("x") as stream:
            json.dump(report, stream, sort_keys=True, allow_nan=False); stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
        path.chmod(0o400); print(small_summary(report, path), flush=True)
    if error: raise RuntimeError("Frontend prerequisite inventory failed closed; inspect owned Azure receipt") from None


if __name__ == "__main__": main()
